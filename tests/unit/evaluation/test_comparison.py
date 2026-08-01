"""Comparable-run validation, Pareto, and explicit selection tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    EvaluationCandidateConfig,
    EvaluationComparisonConfig,
    EvaluationObjectiveConfig,
)
from legal_agentic_rag.evaluation import EvaluationComparisonService
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas import (
    EvaluationBenchmarkLabelStatus,
    EvaluationMetricDirection,
    EvaluationResourceUsage,
    EvaluationSelectionMode,
    EvaluationSummary,
    LatencySummary,
    RetrievalStrategy,
)


def test_pareto_comparison_does_not_invent_one_winner() -> None:
    """Quality/latency trade-offs remain Pareto candidates by default."""
    config = _config(selection_mode=EvaluationSelectionMode.PARETO_ONLY)
    summaries = {
        "accurate": _summary("accurate", quality=0.9, latency=20),
        "fast": _summary("fast", quality=0.8, latency=10),
    }

    report = EvaluationComparisonService().compare(config, summaries)

    assert report.selected_candidate_id is None
    assert report.pareto_candidate_ids == ["accurate", "fast"]
    assert all(candidate.eligible for candidate in report.candidates)
    assert "competition_metric_not_assumed" in report.warnings


def test_lexicographic_policy_selects_by_declared_objective_order() -> None:
    """A reviewed benchmark plus explicit ordered policy may select one."""
    config = _config(selection_mode=EvaluationSelectionMode.LEXICOGRAPHIC)
    summaries = {
        "accurate": _summary("accurate", quality=0.9, latency=20),
        "fast": _summary("fast", quality=0.8, latency=10),
    }

    report = EvaluationComparisonService().compare(config, summaries)

    assert report.selected_candidate_id == "accurate"
    assert "selection_uses_user_declared_lexicographic_policy" in report.warnings


def test_diagnostic_benchmark_cannot_select_a_winner() -> None:
    """Diagnostic fixture labels can compare but cannot crown a candidate."""
    config = _config(selection_mode=EvaluationSelectionMode.LEXICOGRAPHIC)
    summaries = {
        "accurate": _summary(
            "accurate",
            quality=0.9,
            latency=20,
            label_status=EvaluationBenchmarkLabelStatus.DIAGNOSTIC,
        ),
        "fast": _summary(
            "fast",
            quality=0.8,
            latency=10,
            label_status=EvaluationBenchmarkLabelStatus.DIAGNOSTIC,
        ),
    }

    report = EvaluationComparisonService().compare(config, summaries)

    assert report.selected_candidate_id is None
    assert "selection_blocked_untrusted_benchmark" in report.warnings


def test_dominated_and_ineligible_candidates_are_explained() -> None:
    """Threshold and dominance outcomes retain machine-readable reasons."""
    config = _config(
        selection_mode=EvaluationSelectionMode.PARETO_ONLY,
        quality_threshold=0.75,
    )
    summaries = {
        "accurate": _summary("accurate", quality=0.9, latency=10),
        "fast": _summary("fast", quality=0.7, latency=20),
    }

    report = EvaluationComparisonService().compare(config, summaries)

    weak = next(
        candidate
        for candidate in report.candidates
        if candidate.candidate_id == "fast"
    )
    assert weak.eligible is False
    assert weak.exclusion_reasons == [
        "objective_threshold_failed:retrieval.ndcg@5"
    ]
    assert report.pareto_candidate_ids == ["accurate"]
    assert "fewer_than_two_eligible_candidates" in report.warnings


def test_regression_gate_excludes_candidate_below_baseline_tolerance() -> None:
    """A candidate cannot trade away more quality than explicitly allowed."""
    config = _config(
        selection_mode=EvaluationSelectionMode.PARETO_ONLY,
        baseline_candidate_id="accurate",
        maximum_quality_regression=0.05,
    )
    summaries = {
        "accurate": _summary("accurate", quality=0.9, latency=20),
        "fast": _summary("fast", quality=0.8, latency=10),
    }

    report = EvaluationComparisonService().compare(config, summaries)

    fast = next(
        candidate
        for candidate in report.candidates
        if candidate.candidate_id == "fast"
    )
    assert fast.eligible is False
    assert fast.exclusion_reasons == [
        "objective_regression_failed:retrieval.ndcg@5"
    ]
    assert report.baseline_candidate_id == "accurate"


def test_regression_gate_honors_minimize_direction() -> None:
    """Latency may rise only by the declared absolute tolerance."""
    config = _config(
        selection_mode=EvaluationSelectionMode.PARETO_ONLY,
        baseline_candidate_id="accurate",
        maximum_latency_regression=5,
    )
    summaries = {
        "accurate": _summary("accurate", quality=0.9, latency=20),
        "fast": _summary("fast", quality=0.95, latency=30),
    }

    report = EvaluationComparisonService().compare(config, summaries)

    changed = next(
        candidate
        for candidate in report.candidates
        if candidate.candidate_id == "fast"
    )
    assert changed.exclusion_reasons == [
        "objective_regression_failed:latency.generation.p95_ms"
    ]


def test_comparison_rejects_different_exact_benchmark_bytes() -> None:
    """Reports from different benchmark payloads are never compared."""
    config = _config(selection_mode=EvaluationSelectionMode.PARETO_ONLY)
    summaries = {
        "accurate": _summary("accurate", quality=0.9, latency=20),
        "fast": _summary(
            "fast",
            quality=0.8,
            latency=10,
            benchmark_sha256="b" * 64,
        ),
    }

    with pytest.raises(DataValidationError, match="different benchmark bytes"):
        EvaluationComparisonService().compare(config, summaries)


def test_regression_threshold_requires_known_baseline() -> None:
    """Regression policies cannot float without a configured reference run."""
    with pytest.raises(ValidationError, match="baseline_candidate_id"):
        _config(
            selection_mode=EvaluationSelectionMode.PARETO_ONLY,
            maximum_quality_regression=0.01,
        )


def _config(
    *,
    selection_mode: EvaluationSelectionMode,
    quality_threshold: float | None = None,
    baseline_candidate_id: str | None = None,
    maximum_quality_regression: float | None = None,
    maximum_latency_regression: float | None = None,
) -> EvaluationComparisonConfig:
    return EvaluationComparisonConfig(
        comparison_name="fixture comparison",
        candidates=[
            EvaluationCandidateConfig(
                candidate_id="accurate",
                report_directory="accurate",
            ),
            EvaluationCandidateConfig(
                candidate_id="fast",
                report_directory="fast",
            ),
        ],
        objectives=[
            EvaluationObjectiveConfig(
                metric="retrieval.ndcg@5",
                direction=EvaluationMetricDirection.MAXIMIZE,
                eligibility_threshold=quality_threshold,
                maximum_regression=maximum_quality_regression,
            ),
            EvaluationObjectiveConfig(
                metric="latency.generation.p95_ms",
                direction=EvaluationMetricDirection.MINIMIZE,
                maximum_regression=maximum_latency_regression,
            ),
        ],
        selection_mode=selection_mode,
        baseline_candidate_id=baseline_candidate_id,
    )


def _summary(
    run_id: str,
    *,
    quality: float,
    latency: float,
    benchmark_sha256: str = "a" * 64,
    label_status: EvaluationBenchmarkLabelStatus = (
        EvaluationBenchmarkLabelStatus.HUMAN_REVIEWED
    ),
) -> EvaluationSummary:
    return EvaluationSummary(
        run_id=run_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        code_version="0.25.0",
        benchmark_name="fixture.jsonl",
        benchmark_sha256=benchmark_sha256,
        benchmark_manifest_sha256="e" * 64,
        benchmark_version="fixture-v1",
        benchmark_label_status=label_status,
        benchmark_verified_at=(
            datetime(2026, 1, 1, tzinfo=UTC)
            if label_status
            != EvaluationBenchmarkLabelStatus.DIAGNOSTIC
            else None
        ),
        benchmark_label_provenance_reference=(
            "review-protocol:v1"
            if label_status
            != EvaluationBenchmarkLabelStatus.DIAGNOSTIC
            else None
        ),
        dataset_name="fixture/legal",
        dataset_revision="revision-1",
        strategy=RetrievalStrategy.HYBRID_RERANK,
        cutoffs=[1, 5],
        case_count=2,
        successful_case_count=2,
        failed_case_count=0,
        retrieval_metrics={"ndcg@5": quality},
        metric_case_counts={"ndcg@5": 2},
        retrieval_latency=LatencySummary(
            count=2,
            mean_ms=5,
            p50_ms=5,
            p95_ms=5,
            max_ms=5,
        ),
        generation_latency=LatencySummary(
            count=2,
            mean_ms=latency,
            p50_ms=latency,
            p95_ms=latency,
            max_ms=latency,
        ),
        resources=EvaluationResourceUsage(
            wall_time_ms=latency * 2,
            process_cpu_time_ms=1,
            python_peak_traced_memory_bytes=1,
        ),
        runtime_config_sha256=("c" if run_id == "accurate" else "d") * 64,
        component_provenance={"generator": {"model_name": run_id}},
    )
