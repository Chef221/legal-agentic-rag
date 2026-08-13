"""Tests for content-free analysis of completed competition batches."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.batch_inference import (
    CompetitionBatchRunner,
)
from legal_agentic_rag.evaluation.batch_analysis import (
    BATCH_ANALYSIS_FILENAME,
    BATCH_COMPARISON_FILENAME,
    CompetitionBatchAnalysisService,
    CompetitionBatchComparisonService,
    persist_batch_analysis_report,
    persist_batch_comparison_report,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import AnswerResponse, RetrievalStrategy


class _Answerer:
    def __init__(self, *, candidate: bool) -> None:
        self._candidate = candidate
        self._calls = 0

    def answer(self, request):  # type: ignore[no-untyped-def]
        self._calls += 1
        is_first = self._calls == 1
        if self._candidate:
            answer = f"Cau tra loi cai tien {self._calls}."
            warnings: list[str] = []
            insufficient = False
            stop_reason = "answer_verified"
            citation_metadata = {
                "is_valid": True,
                "valid_citations": [],
                "invalid_citations": [],
                "claim_verifications": [],
                "claim_coverage_score": None,
                "claim_level_verification_performed": False,
                "semantic_verification": None,
                "errors": [],
                "warnings": [],
            }
        elif is_first:
            answer = "Cau tra loi control mot."
            warnings = ["citation_verification_failed"]
            insufficient = False
            stop_reason = "citation_verification_failed"
            citation_metadata = {
                "is_valid": False,
                "valid_citations": [],
                "invalid_citations": [],
                "claim_verifications": [
                    {
                        "claim_id": "C1",
                        "claim_text": "Nhan dinh kiem thu.",
                        "evidence_ids": [],
                        "status": "unsupported",
                        "lexical_support_score": 0.0,
                        "numeric_match": True,
                        "negation_match": True,
                        "errors": ["missing_inline_evidence:C1"],
                    }
                ],
                "claim_coverage_score": 0.0,
                "claim_level_verification_performed": True,
                "semantic_verification": None,
                "errors": ["missing_inline_evidence:C1"],
                "warnings": [],
            }
        else:
            answer = "Cau tra loi control hai."
            warnings = ["generator:model_error"]
            insufficient = True
            stop_reason = "insufficient_evidence"
            citation_metadata = None
        metadata = {
            "agent": {"stop_reason": stop_reason, "total_latency_ms": 10.0 * self._calls},
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
        }
        if citation_metadata is not None:
            metadata["citation_verification"] = citation_metadata
        return AnswerResponse(
            question=request.question,
            answer=answer,
            insufficient_evidence=insufficient,
            warnings=warnings,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            trace_id=f"trace-{self._calls}",
            metadata=metadata,
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


def _run_batch(
    questions: Path,
    output: Path,
    *,
    config_hash: str,
    candidate: bool,
) -> None:
    CompetitionBatchRunner(
        _Answerer(candidate=candidate),
        application_config_hash=config_hash,
    ).run(questions, output)


def test_batch_analysis_summarizes_failures_without_answer_content(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "development.json"
    batch = tmp_path / "control"
    _questions(questions)
    _run_batch(questions, batch, config_hash="a" * 64, candidate=False)

    report = CompetitionBatchAnalysisService(
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC)
    ).analyze(batch)

    assert report.record_count == 2
    assert report.insufficient_evidence_count == 1
    assert report.generator_model_error_count == 1
    assert report.citation.verification_failed_count == 1
    assert report.citation.claim_error_counts == {"missing_inline_evidence:C1": 1}
    assert report.context_trace.trace_present_count == 2
    assert report.context_trace.selection_reason_counts == {"selected": 2}
    assert report.agent_latency.mean_ms == 15.0

    output = tmp_path / "analysis"
    persist_batch_analysis_report(report, output)
    payload = (output / BATCH_ANALYSIS_FILENAME).read_text(encoding="utf-8")
    assert "Cau tra loi control" not in payload
    assert "Cau hoi mot" not in payload


def test_batch_comparison_detects_changed_outcomes_and_accepts_new_config(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "development.json"
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _questions(questions)
    _run_batch(questions, control, config_hash="a" * 64, candidate=False)
    _run_batch(questions, candidate, config_hash="b" * 64, candidate=True)

    report = CompetitionBatchComparisonService().compare(control, candidate)

    assert report.record_count == 2
    assert report.answer_changed_count == 2
    assert report.citation_failure_transition_counts == {"false->false": 1, "true->false": 1}
    assert report.generator_model_error_transition_counts == {"false->false": 1, "true->false": 1}
    assert {case.question_id for case in report.changed_cases} == {"q1", "q2"}

    output = tmp_path / "comparison"
    persist_batch_comparison_report(report, output)
    assert (output / BATCH_COMPARISON_FILENAME).exists()


def test_batch_analysis_rejects_records_that_do_not_match_manifest(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "development.json"
    batch = tmp_path / "control"
    _questions(questions)
    _run_batch(questions, batch, config_hash="a" * 64, candidate=False)
    (batch / "results.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="SHA-256"):
        CompetitionBatchAnalysisService().analyze(batch)
