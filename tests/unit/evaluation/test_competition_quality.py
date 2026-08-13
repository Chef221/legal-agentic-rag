"""Tests for policy-gated competition batches and paired scorer reports."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.batch_inference import (
    CompetitionBatchRunner,
)
from legal_agentic_rag.evaluation.competition_quality import (
    BATCH_READINESS_FILENAME,
    WARMUP_SCORE_COMPARISON_FILENAME,
    CompetitionBatchReadinessService,
    CompetitionWarmupScoreComparisonService,
    persist_batch_readiness_report,
    persist_warmup_score_comparison_report,
    require_ready_competition_batch,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import (
    AnswerResponse,
    CompetitionBatchReadinessPolicy,
    CompetitionMetricMode,
    CompetitionWarmupCaseScore,
    CompetitionWarmupScoreReport,
    RetrievalStrategy,
)


class _Answerer:
    """Small deterministic answerer that only creates internal test records."""

    def __init__(self, *, failing: bool) -> None:
        self._failing = failing
        self._calls = 0

    def answer(self, request):  # type: ignore[no-untyped-def]
        self._calls += 1
        warnings = ["generator:model_error"] if self._failing else []
        return AnswerResponse(
            question=request.question,
            answer=f"Internal test answer {self._calls}.",
            insufficient_evidence=self._failing,
            warnings=warnings,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            trace_id=f"trace-{self._calls}",
            metadata={
                "agent": {
                    "stop_reason": (
                        "insufficient_evidence" if self._failing else "answer_verified"
                    ),
                    "total_latency_ms": 10.0,
                },
                "context": {
                    "selected_count": 1,
                    "selection_trace": [
                        {
                            "chunk_id": f"chunk-{self._calls}",
                            "source_rank": 1,
                            "selection_rank": 1,
                            "applicability": "compatible",
                            "document_reference_match": None,
                            "article_reference_match": None,
                            "lexical_overlap_score": 0.5,
                            "selection_score": 0.5,
                            "selected": True,
                            "reason": "selected",
                        }
                    ],
                },
            },
        )


def _questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "q1": {"question": "Cau hoi mot?", "answer": "Gold mot"},
                "q2": {"question": "Cau hoi hai?", "answer": "Gold hai"},
            }
        ),
        encoding="utf-8",
    )


def _run_batch(questions: Path, output: Path, *, failing: bool) -> None:
    CompetitionBatchRunner(
        _Answerer(failing=failing),
        application_config_hash="a" * 64,
    ).run(questions, output)


def _policy(**overrides: object) -> CompetitionBatchReadinessPolicy:
    payload: dict[str, object] = {
        "max_retrieval_model_error_count": 0,
        "max_generator_model_error_count": 0,
        "max_citation_verification_failure_count": 0,
        "max_insufficient_evidence_rate": 0.0,
        "require_context_selection_trace": True,
    }
    payload.update(overrides)
    return CompetitionBatchReadinessPolicy.model_validate(payload)


def test_readiness_gate_accepts_complete_batch_that_meets_explicit_policy(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "official.json"
    batch = tmp_path / "batch"
    _questions(questions)
    _run_batch(questions, batch, failing=False)

    report = CompetitionBatchReadinessService(
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC)
    ).check(
        questions_source=questions,
        batch_directory=batch,
        policy=_policy(),
    )

    assert report.is_ready is True
    assert report.violations == []
    output = tmp_path / "readiness"
    persist_batch_readiness_report(report, output)
    payload = (output / BATCH_READINESS_FILENAME).read_text(encoding="utf-8")
    assert "Internal test answer" not in payload


def test_readiness_gate_rejects_model_errors_and_insufficient_responses(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "official.json"
    batch = tmp_path / "batch"
    _questions(questions)
    _run_batch(questions, batch, failing=True)

    report = CompetitionBatchReadinessService().check(
        questions_source=questions,
        batch_directory=batch,
        policy=_policy(),
    )

    assert report.is_ready is False
    assert report.violations == [
        "generator_model_error_limit_exceeded",
        "insufficient_evidence_rate_limit_exceeded",
    ]


def test_submission_boundary_requires_matching_passed_readiness_report(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "official.json"
    batch = tmp_path / "batch"
    _questions(questions)
    _run_batch(questions, batch, failing=False)
    report = CompetitionBatchReadinessService().check(
        questions_source=questions,
        batch_directory=batch,
        policy=_policy(),
    )
    readiness_directory = tmp_path / "readiness"
    persist_batch_readiness_report(report, readiness_directory)

    loaded = require_ready_competition_batch(
        questions_source=questions,
        batch_directory=batch,
        readiness_report_directory=readiness_directory,
    )
    assert loaded.is_ready is True

    (batch / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="SHA-256"):
        require_ready_competition_batch(
            questions_source=questions,
            batch_directory=batch,
            readiness_report_directory=readiness_directory,
        )


def test_readiness_gate_rejects_batch_for_different_question_source(
    tmp_path: Path,
) -> None:
    original = tmp_path / "official.json"
    changed = tmp_path / "different.json"
    batch = tmp_path / "batch"
    _questions(original)
    _run_batch(original, batch, failing=False)
    changed.write_text(
        json.dumps({"q1": {"question": "Cau hoi khac?"}}), encoding="utf-8"
    )

    report = CompetitionBatchReadinessService().check(
        questions_source=changed,
        batch_directory=batch,
        policy=_policy(max_insufficient_evidence_rate=1.0),
    )

    assert report.is_ready is False
    assert "question_source_sha256_mismatch" in report.violations
    assert "question_ids_or_order_mismatch" in report.violations


def _score_report(
    *,
    meteor: float,
    rouge_l: float,
    case_metrics: list[tuple[float, float]],
    reference_hash: str = "b" * 64,
) -> CompetitionWarmupScoreReport:
    return CompetitionWarmupScoreReport(
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        code_version="0.47.0",
        metric_mode=CompetitionMetricMode.DIAGNOSTIC,
        reference_source_sha256=reference_hash,
        submission_archive_sha256="c" * 64,
        submission_json_sha256="d" * 64,
        question_count=2,
        exact_match=0.0,
        meteor=meteor,
        rouge_l=rouge_l,
        cases=[
            CompetitionWarmupCaseScore(
                question_id=f"q{index}",
                exact_match=0.0,
                meteor=case_meteor,
                rouge_l=case_rouge_l,
            )
            for index, (case_meteor, case_rouge_l) in enumerate(
                case_metrics, start=1
            )
        ],
    )


def _write_score_report(directory: Path, report: CompetitionWarmupScoreReport) -> None:
    directory.mkdir()
    (directory / "warmup_score.json").write_text(
        json.dumps(report.model_dump(mode="json")), encoding="utf-8"
    )


def test_paired_score_comparison_reports_aggregate_and_per_id_deltas(
    tmp_path: Path,
) -> None:
    baseline_directory = tmp_path / "baseline"
    candidate_directory = tmp_path / "candidate"
    _write_score_report(
        baseline_directory,
        _score_report(
            meteor=0.2,
            rouge_l=0.3,
            case_metrics=[(0.1, 0.2), (0.3, 0.4)],
        ),
    )
    _write_score_report(
        candidate_directory,
        _score_report(
            meteor=0.4,
            rouge_l=0.25,
            case_metrics=[(0.3, 0.1), (0.5, 0.4)],
        ),
    )

    report = CompetitionWarmupScoreComparisonService().compare(
        baseline_directory,
        candidate_directory,
    )

    assert report.meteor.mean_delta == pytest.approx(0.2)
    assert report.meteor.improved_case_count == 2
    assert report.rouge_l.mean_delta == pytest.approx(-0.05)
    assert report.rouge_l.improved_case_count == 0
    assert report.rouge_l.regressed_case_count == 1
    assert report.rouge_l.tied_case_count == 1
    assert report.cases[0].meteor_delta == pytest.approx(0.2)

    output = tmp_path / "comparison"
    persist_warmup_score_comparison_report(report, output)
    assert (output / WARMUP_SCORE_COMPARISON_FILENAME).exists()


def test_paired_score_comparison_rejects_different_reference_source(
    tmp_path: Path,
) -> None:
    baseline_directory = tmp_path / "baseline"
    candidate_directory = tmp_path / "candidate"
    _write_score_report(
        baseline_directory,
        _score_report(meteor=0.2, rouge_l=0.2, case_metrics=[(0.2, 0.2), (0.2, 0.2)]),
    )
    _write_score_report(
        candidate_directory,
        _score_report(
            meteor=0.2,
            rouge_l=0.2,
            case_metrics=[(0.2, 0.2), (0.2, 0.2)],
            reference_hash="e" * 64,
        ),
    )

    with pytest.raises(ArtifactCompatibilityError, match="incompatible"):
        CompetitionWarmupScoreComparisonService().compare(
            baseline_directory,
            candidate_directory,
        )


def test_paired_score_comparison_rejects_tampered_aggregate(
    tmp_path: Path,
) -> None:
    baseline_directory = tmp_path / "baseline"
    candidate_directory = tmp_path / "candidate"
    baseline = _score_report(
        meteor=0.2,
        rouge_l=0.2,
        case_metrics=[(0.2, 0.2), (0.2, 0.2)],
    )
    _write_score_report(baseline_directory, baseline)
    _write_score_report(
        candidate_directory,
        baseline.model_copy(update={"meteor": 0.9}),
    )

    with pytest.raises(ArtifactCompatibilityError, match="aggregates"):
        CompetitionWarmupScoreComparisonService().compare(
            baseline_directory,
            candidate_directory,
        )
