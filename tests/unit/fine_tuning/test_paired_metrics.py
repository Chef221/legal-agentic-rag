"""Tests for paired direct QA scoring and bootstrap statistics."""

from __future__ import annotations

from datetime import UTC, datetime
import pytest

from legal_agentic_rag.fine_tuning.paired_metrics import (
    DirectQAPairedScorer,
    compute_paired_bootstrap_ci,
)
from legal_agentic_rag.schemas import (
    CompetitionQuestion,
    DirectQACaseResult,
)


def test_paired_bootstrap_ci_computation() -> None:
    # Deliberate positive deltas
    deltas = [0.05, 0.04, 0.06, 0.03, 0.05, 0.07, 0.04, 0.05]
    ci = compute_paired_bootstrap_ci(deltas, "meteor", resamples=500, seed=2026)

    assert ci.metric_name == "meteor"
    assert ci.mean_delta > 0.04
    assert ci.ci_lower_95 > 0.02
    assert ci.ci_upper_95 >= ci.ci_lower_95


def test_direct_qa_paired_scorer() -> None:
    now = datetime.now(UTC)
    references = [
        CompetitionQuestion(question_id="1", question="Q1?", reference_answer="Hợp đồng lao động có hiệu lực ngay."),
        CompetitionQuestion(question_id="2", question="Q2?", reference_answer="Thời hiệu xử phạt là một năm."),
    ]

    base_results = [
        DirectQACaseResult(
            question_id="1",
            question="Q1?",
            generated_answer="Hợp đồng có hiệu lực.",
            generated_token_count=10,
            model_id="base",
            model_revision="rev1",
            created_at=now,
        ),
        DirectQACaseResult(
            question_id="2",
            question="Q2?",
            generated_answer="Xử phạt trong một năm.",
            generated_token_count=10,
            model_id="base",
            model_revision="rev1",
            created_at=now,
        ),
    ]

    treatment_results = [
        DirectQACaseResult(
            question_id="1",
            question="Q1?",
            generated_answer="Hợp đồng lao động có hiệu lực ngay khi ký kết.",
            generated_token_count=15,
            model_id="treatment",
            model_revision="rev2",
            created_at=now,
        ),
        DirectQACaseResult(
            question_id="2",
            question="Q2?",
            generated_answer="Thời hiệu xử phạt vi phạm hành chính là một năm.",
            generated_token_count=16,
            model_id="treatment",
            model_revision="rev2",
            created_at=now,
        ),
    ]

    scorer = DirectQAPairedScorer(use_diagnostic_fallback=True)
    report = scorer.compare(
        base_results=base_results,
        treatment_results=treatment_results,
        references=references,
        bootstrap_resamples=200,
        bootstrap_seed=2026,
    )

    assert report.question_count == 2
    assert len(report.cases) == 2
    assert report.meteor.mean_delta is not None
    assert report.rouge_l.mean_delta is not None
    assert report.meteor.bootstrap_ci_95.resamples == 200


def test_direct_qa_paired_scorer_rejects_failed_generations() -> None:
    from legal_agentic_rag.exceptions import DataValidationError

    now = datetime.now(UTC)
    references = [
        CompetitionQuestion(question_id="1", question="Q1?", reference_answer="Answer 1."),
    ]

    base_results = [
        DirectQACaseResult(
            question_id="1",
            question="Q1?",
            generated_answer="GENERATION_ERROR: CUDA OOM",
            generated_token_count=10,
            status="error",
            model_id="base",
            model_revision="rev1",
            created_at=now,
        )
    ]
    treatment_results = [
        DirectQACaseResult(
            question_id="1",
            question="Q1?",
            generated_answer="Valid answer.",
            generated_token_count=10,
            status="success",
            model_id="treatment",
            model_revision="rev2",
            created_at=now,
        )
    ]

    scorer = DirectQAPairedScorer(use_diagnostic_fallback=True)
    with pytest.raises(DataValidationError, match="Generation error detected"):
        scorer.compare(
            base_results=base_results,
            treatment_results=treatment_results,
            references=references,
        )


def test_direct_qa_paired_scorer_censorship_warnings() -> None:
    now = datetime.now(UTC)
    references = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Answer {i}.")
        for i in range(100)
    ]

    base_results = [
        DirectQACaseResult(
            question_id=str(i),
            question=f"Q{i}?",
            generated_answer=f"Answer {i}.",
            generated_token_count=10,
            status="success",
            hit_max_tokens=(i < 5),  # 5% hit max tokens -> triggers censorship warning
            model_id="base",
            model_revision="rev1",
            created_at=now,
        )
        for i in range(100)
    ]
    treatment_results = [
        DirectQACaseResult(
            question_id=str(i),
            question=f"Q{i}?",
            generated_answer=f"Answer {i}.",
            generated_token_count=10,
            status="success",
            hit_max_tokens=False,
            model_id="treatment",
            model_revision="rev2",
            created_at=now,
        )
        for i in range(100)
    ]

    scorer = DirectQAPairedScorer(use_diagnostic_fallback=True)
    report = scorer.compare(
        base_results=base_results,
        treatment_results=treatment_results,
        references=references,
    )

    assert report.base_hit_max_tokens_count == 5
    assert report.treatment_hit_max_tokens_count == 0
    assert "screening_potentially_censored" in report.warnings
    assert "base_generations_hit_max_tokens_count_5" in report.warnings
