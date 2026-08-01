"""Conservative comparison of reproducible evaluation summaries."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from uuid import uuid4

from legal_agentic_rag.configuration import EvaluationComparisonConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas import (
    EvaluationCandidateResult,
    EvaluationBenchmarkLabelStatus,
    EvaluationComparisonReport,
    EvaluationMetricDirection,
    EvaluationObjective,
    EvaluationSelectionMode,
    EvaluationSummary,
)


class EvaluationComparisonService:
    """Compare like-for-like reports without assuming an official metric."""

    def compare(
        self,
        config: EvaluationComparisonConfig,
        summaries: dict[str, EvaluationSummary],
    ) -> EvaluationComparisonReport:
        """Validate comparability, project metrics, and compute Pareto results."""
        expected_ids = [candidate.candidate_id for candidate in config.candidates]
        if set(summaries) != set(expected_ids):
            raise DataValidationError(
                "Comparison summaries must match configured candidates"
            )
        ordered = [summaries[candidate_id] for candidate_id in expected_ids]
        reference = ordered[0]
        self._validate_comparability(reference, ordered[1:])

        objectives = [
            EvaluationObjective(
                metric=objective.metric,
                direction=objective.direction,
                eligibility_threshold=objective.eligibility_threshold,
                maximum_regression=objective.maximum_regression,
            )
            for objective in config.objectives
        ]
        objective_counts = self._validate_objective_counts(objectives, ordered)
        candidates = [
            self._candidate_result(
                candidate_id,
                summary,
                objectives,
                objective_counts,
                require_zero_failures=config.require_zero_failures,
            )
            for candidate_id, summary in zip(expected_ids, ordered, strict=True)
        ]
        candidates = self._apply_regression_gates(
            candidates,
            objectives,
            config.baseline_candidate_id,
        )
        candidates = self._attach_dominance(candidates, objectives)
        pareto_ids = [
            candidate.candidate_id
            for candidate in candidates
            if candidate.eligible and not candidate.dominated_by
        ]
        eligible = [candidate for candidate in candidates if candidate.eligible]
        selected_id = None
        trusted_benchmark = (
            reference.benchmark_label_status
            != EvaluationBenchmarkLabelStatus.DIAGNOSTIC
        )
        warnings = ["competition_metric_not_assumed"]
        warnings.append(
            "benchmark_trust_is_manifest_declared"
            if trusted_benchmark
            else "benchmark_labels_are_diagnostic_only"
        )
        if len(eligible) < 2:
            warnings.append("fewer_than_two_eligible_candidates")
        elif config.selection_mode == EvaluationSelectionMode.LEXICOGRAPHIC:
            if not trusted_benchmark:
                warnings.append("selection_blocked_untrusted_benchmark")
            else:
                selected_id = min(
                    eligible,
                    key=lambda candidate: self._selection_key(
                        candidate,
                        objectives,
                    ),
                ).candidate_id
                warnings.append(
                    "selection_uses_user_declared_lexicographic_policy"
                )
        else:
            warnings.append("no_candidate_selected_pareto_only")

        return EvaluationComparisonReport(
            comparison_id=str(uuid4()),
            comparison_name=config.comparison_name,
            created_at=datetime.now(UTC),
            benchmark_name=reference.benchmark_name,
            benchmark_sha256=reference.benchmark_sha256,
            benchmark_manifest_sha256=reference.benchmark_manifest_sha256,
            benchmark_version=reference.benchmark_version,
            benchmark_label_status=reference.benchmark_label_status,
            benchmark_verified_at=reference.benchmark_verified_at,
            benchmark_label_provenance_reference=(
                reference.benchmark_label_provenance_reference
            ),
            case_count=reference.case_count,
            objectives=objectives,
            selection_mode=config.selection_mode,
            baseline_candidate_id=config.baseline_candidate_id,
            candidates=candidates,
            pareto_candidate_ids=pareto_ids,
            selected_candidate_id=selected_id,
            warnings=warnings,
        )

    @staticmethod
    def _validate_comparability(
        reference: EvaluationSummary,
        others: list[EvaluationSummary],
    ) -> None:
        for summary in others:
            if (
                summary.benchmark_manifest_sha256
                != reference.benchmark_manifest_sha256
            ):
                raise DataValidationError(
                    "Comparison candidates use different benchmark manifests"
                )
            if summary.benchmark_sha256 != reference.benchmark_sha256:
                raise DataValidationError(
                    "Comparison candidates use different benchmark bytes"
                )
            if summary.case_count != reference.case_count:
                raise DataValidationError(
                    "Comparison candidates use different case counts"
                )
            if summary.cutoffs != reference.cutoffs:
                raise DataValidationError(
                    "Comparison candidates use different retrieval cutoffs"
                )
            if (
                summary.benchmark_version != reference.benchmark_version
                or summary.benchmark_label_status
                != reference.benchmark_label_status
                or summary.benchmark_verified_at
                != reference.benchmark_verified_at
                or summary.benchmark_label_provenance_reference
                != reference.benchmark_label_provenance_reference
            ):
                raise DataValidationError(
                    "Comparison candidates use different benchmark provenance"
                )
            if (
                reference.dataset_name is None
                or reference.dataset_revision is None
                or summary.dataset_name != reference.dataset_name
                or summary.dataset_revision != reference.dataset_revision
            ):
                raise DataValidationError(
                    "Comparison candidates use different or unpinned dataset lineage"
                )

    @staticmethod
    def _validate_objective_counts(
        objectives: list[EvaluationObjective],
        summaries: list[EvaluationSummary],
    ) -> dict[str, int | None]:
        counts: dict[str, int | None] = {}
        for objective in objectives:
            raw_name = _case_count_metric_name(objective.metric)
            if raw_name is None:
                counts[objective.metric] = None
                continue
            values = [
                summary.metric_case_counts.get(raw_name) for summary in summaries
            ]
            if any(value is None for value in values):
                raise DataValidationError(
                    "Comparison objective lacks labeled case counts"
                )
            present = [value for value in values if value is not None]
            if len(set(present)) != 1:
                raise DataValidationError(
                    "Comparison objective uses different labeled case counts"
                )
            counts[objective.metric] = present[0] if present else None
        return counts

    @staticmethod
    def _candidate_result(
        candidate_id: str,
        summary: EvaluationSummary,
        objectives: list[EvaluationObjective],
        objective_counts: dict[str, int | None],
        *,
        require_zero_failures: bool,
    ) -> EvaluationCandidateResult:
        available = _metric_values(summary)
        values: dict[str, float] = {}
        reasons: list[str] = []
        if require_zero_failures and summary.failed_case_count:
            reasons.append("evaluation_failures_present")
        if summary.runtime_config_sha256 is None:
            reasons.append("runtime_config_identity_missing")
        if not summary.component_provenance:
            reasons.append("component_provenance_missing")
        for objective in objectives:
            value = available.get(objective.metric)
            if value is None or not isfinite(value):
                reasons.append(f"objective_metric_missing:{objective.metric}")
                continue
            if objective_counts[objective.metric] == 0:
                reasons.append(f"objective_has_no_labeled_cases:{objective.metric}")
                continue
            values[objective.metric] = value
            threshold = objective.eligibility_threshold
            if threshold is None:
                continue
            if (
                objective.direction == EvaluationMetricDirection.MAXIMIZE
                and value < threshold
            ) or (
                objective.direction == EvaluationMetricDirection.MINIMIZE
                and value > threshold
            ):
                reasons.append(f"objective_threshold_failed:{objective.metric}")
        return EvaluationCandidateResult(
            candidate_id=candidate_id,
            run_id=summary.run_id,
            code_version=summary.code_version,
            strategy=summary.strategy,
            runtime_config_sha256=summary.runtime_config_sha256,
            component_provenance=summary.component_provenance,
            artifact_versions=summary.artifact_versions,
            metric_values=values,
            eligible=not reasons,
            exclusion_reasons=reasons,
        )

    @staticmethod
    def _apply_regression_gates(
        candidates: list[EvaluationCandidateResult],
        objectives: list[EvaluationObjective],
        baseline_candidate_id: str | None,
    ) -> list[EvaluationCandidateResult]:
        gated = [
            objective
            for objective in objectives
            if objective.maximum_regression is not None
        ]
        if not gated:
            return candidates
        baseline = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == baseline_candidate_id
        )
        if not baseline.eligible:
            return [
                _exclude(candidate, "regression_baseline_ineligible")
                if candidate.eligible
                else candidate
                for candidate in candidates
            ]
        output: list[EvaluationCandidateResult] = []
        for candidate in candidates:
            if (
                not candidate.eligible
                or candidate.candidate_id == baseline.candidate_id
            ):
                output.append(candidate)
                continue
            reasons: list[str] = []
            for objective in gated:
                maximum = objective.maximum_regression
                if maximum is None:
                    continue
                baseline_value = baseline.metric_values[objective.metric]
                candidate_value = candidate.metric_values[objective.metric]
                failed = (
                    objective.direction
                    == EvaluationMetricDirection.MAXIMIZE
                    and candidate_value < baseline_value - maximum
                ) or (
                    objective.direction
                    == EvaluationMetricDirection.MINIMIZE
                    and candidate_value > baseline_value + maximum
                )
                if failed:
                    reasons.append(
                        f"objective_regression_failed:{objective.metric}"
                    )
            updated = candidate
            for reason in reasons:
                updated = _exclude(updated, reason)
            output.append(updated)
        return output

    @staticmethod
    def _attach_dominance(
        candidates: list[EvaluationCandidateResult],
        objectives: list[EvaluationObjective],
    ) -> list[EvaluationCandidateResult]:
        eligible = [candidate for candidate in candidates if candidate.eligible]
        output: list[EvaluationCandidateResult] = []
        for candidate in candidates:
            dominators = (
                sorted(
                    other.candidate_id
                    for other in eligible
                    if other.candidate_id != candidate.candidate_id
                    and _dominates(other, candidate, objectives)
                )
                if candidate.eligible
                else []
            )
            output.append(candidate.model_copy(update={"dominated_by": dominators}))
        return output

    @staticmethod
    def _selection_key(
        candidate: EvaluationCandidateResult,
        objectives: list[EvaluationObjective],
    ) -> tuple[float | str, ...]:
        values: list[float | str] = []
        for objective in objectives:
            value = candidate.metric_values[objective.metric]
            values.append(
                -value
                if objective.direction == EvaluationMetricDirection.MAXIMIZE
                else value
            )
        values.append(candidate.candidate_id)
        return tuple(values)


def _metric_values(summary: EvaluationSummary) -> dict[str, float]:
    values = {
        f"retrieval.{name}": value
        for name, value in summary.retrieval_metrics.items()
    }
    values.update(
        {
            f"generation.{name}": value
            for name, value in summary.generation_metrics.items()
        }
    )
    for prefix, latency in (
        ("latency.retrieval", summary.retrieval_latency),
        ("latency.generation", summary.generation_latency),
    ):
        for field_name in ("mean_ms", "p50_ms", "p95_ms", "max_ms"):
            value = getattr(latency, field_name)
            if value is not None:
                values[f"{prefix}.{field_name}"] = value
    values.update(
        {
            "resources.wall_time_ms": summary.resources.wall_time_ms,
            "resources.process_cpu_time_ms": (
                summary.resources.process_cpu_time_ms
            ),
            "resources.python_peak_traced_memory_bytes": float(
                summary.resources.python_peak_traced_memory_bytes
            ),
            "failures.count": float(summary.failed_case_count),
            "failures.rate": (
                summary.failed_case_count / summary.case_count
                if summary.case_count
                else 0.0
            ),
        }
    )
    if summary.resources.accelerator_peak_memory_bytes is not None:
        values["resources.accelerator_peak_memory_bytes"] = float(
            summary.resources.accelerator_peak_memory_bytes
        )
    return values


def _case_count_metric_name(metric: str) -> str | None:
    for prefix in ("retrieval.", "generation."):
        if metric.startswith(prefix):
            return metric.removeprefix(prefix)
    return None


def _dominates(
    left: EvaluationCandidateResult,
    right: EvaluationCandidateResult,
    objectives: list[EvaluationObjective],
) -> bool:
    no_worse = True
    strictly_better = False
    for objective in objectives:
        left_value = left.metric_values[objective.metric]
        right_value = right.metric_values[objective.metric]
        if objective.direction == EvaluationMetricDirection.MAXIMIZE:
            no_worse = no_worse and left_value >= right_value
            strictly_better = strictly_better or left_value > right_value
        else:
            no_worse = no_worse and left_value <= right_value
            strictly_better = strictly_better or left_value < right_value
    return no_worse and strictly_better


def _exclude(
    candidate: EvaluationCandidateResult,
    reason: str,
) -> EvaluationCandidateResult:
    reasons = [*candidate.exclusion_reasons]
    if reason not in reasons:
        reasons.append(reason)
    return candidate.model_copy(
        update={"eligible": False, "exclusion_reasons": reasons}
    )
