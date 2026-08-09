"""Tests for answer-only official warm-up scoring."""

from datetime import UTC, datetime
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from legal_agentic_rag.competition.uit_dsc_2026 import CompetitionWarmupScorer
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas import CompetitionMetricMode


def _references(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "q_000001": {
                    "question": "Người lao động được nghỉ bao nhiêu ngày?",
                    "answer": "Người lao động được nghỉ 12 ngày.",
                },
                "q_000002": {
                    "question": "Doanh nghiệp có nghĩa vụ gì?",
                    "answer": "Doanh nghiệp phải nộp thuế.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _submission(
    path: Path,
    *,
    reverse: bool = False,
    extra: bool = False,
) -> None:
    records = [
        {
            "id": "q_000001",
            "answer": "Người lao động được nghỉ 12 ngày.",
        },
        {
            "id": "q_000002",
            "answer": "Doanh nghiệp nộp thuế.",
        },
    ]
    if reverse:
        records.reverse()
    payload = {
        record["id"]: {"answer": record["answer"]} for record in records
    }
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "submission.json",
            json.dumps(payload, ensure_ascii=False),
        )
        if extra:
            archive.writestr("manifest.json", "{}")


def test_warmup_scorer_persists_content_free_aggregate_report(
    tmp_path: Path,
) -> None:
    references = tmp_path / "warmup.json"
    submission = tmp_path / "submission.zip"
    output = tmp_path / "score"
    _references(references)
    _submission(submission)

    report = CompetitionWarmupScorer(
        clock=lambda: datetime(2026, 8, 1, tzinfo=UTC),
    ).score(references, submission, output)

    assert report.question_count == 2
    assert report.exact_match == 0.5
    assert 0 < report.meteor < 1
    assert 0 < report.rouge_l < 1
    assert len(report.reference_source_sha256) == 64
    persisted = (output / "warmup_score.json").read_text(encoding="utf-8")
    assert "Người lao động được nghỉ" not in persisted
    assert "Doanh nghiệp nộp thuế" not in persisted
    assert "competition_text_metrics_are_diagnostic" in persisted


def test_warmup_scorer_rejects_reordered_ids_and_existing_output(
    tmp_path: Path,
) -> None:
    references = tmp_path / "warmup.json"
    submission = tmp_path / "submission.zip"
    output = tmp_path / "score"
    _references(references)
    _submission(submission, reverse=True)

    with pytest.raises(DataValidationError, match="exactly match"):
        CompetitionWarmupScorer().score(references, submission, output)

    submission.unlink()
    _submission(submission)
    CompetitionWarmupScorer().score(references, submission, output)
    with pytest.raises(ArtifactCompatibilityError, match="already exists"):
        CompetitionWarmupScorer().score(references, submission, output)


def test_warmup_scorer_rejects_submission_with_extra_member(
    tmp_path: Path,
) -> None:
    references = tmp_path / "warmup.json"
    submission = tmp_path / "submission.zip"
    _references(references)
    _submission(submission, extra=True)

    with pytest.raises(DataValidationError, match="only submission.json"):
        CompetitionWarmupScorer().score(
            references,
            submission,
            tmp_path / "score",
        )


def test_warmup_scorer_records_official_compatible_identity(tmp_path: Path) -> None:
    references = tmp_path / "warmup.json"
    submission = tmp_path / "submission.zip"
    _references(references)
    _submission(submission)

    report = CompetitionWarmupScorer(
        official_meteor_scorer=lambda _references, _prediction: 0.5,
    ).score(
        references,
        submission,
        tmp_path / "official-score",
        metric_mode=CompetitionMetricMode.OFFICIAL_COMPATIBLE,
    )

    assert report.metric_mode == CompetitionMetricMode.OFFICIAL_COMPATIBLE
    assert report.nltk_version == "3.7"
    assert report.official_scorer_sha256 is not None
    assert "diagnostic" not in " ".join(report.warnings)
