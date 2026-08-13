"""CPU-only readiness gates and paired official score comparisons."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from statistics import fmean
from typing import Callable

from pydantic import ValidationError

from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.competition.uit_dsc_2026.warmup_scoring import (
    WARMUP_SCORE_REPORT_FILENAME,
)
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    ConfigurationError,
    DataValidationError,
)
from legal_agentic_rag.evaluation.batch_analysis import (
    CompetitionBatchAnalysisService,
    load_completed_competition_batch,
)
from legal_agentic_rag.schemas import (
    CompetitionBatchReadinessPolicy,
    CompetitionBatchReadinessReport,
    CompetitionMetricMode,
    CompetitionWarmupMetricComparison,
    CompetitionWarmupScoreComparisonCase,
    CompetitionWarmupScoreComparisonReport,
    CompetitionWarmupScoreReport,
)

BATCH_READINESS_FILENAME = "batch_readiness.json"
WARMUP_SCORE_COMPARISON_FILENAME = "warmup_score_comparison.json"


class CompetitionBatchReadinessService:
    """Check a finished batch against official question bytes and operator policy."""

    def __init__(
        self,
        loader: UitDsc2026DataLoader | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._loader = loader or UitDsc2026DataLoader()
        self._clock = clock or (lambda: datetime.now(UTC))

    def check(
        self,
        *,
        questions_source: Path,
        batch_directory: Path,
        policy: CompetitionBatchReadinessPolicy,
    ) -> CompetitionBatchReadinessReport:
        """Return a durable decision without generating, scoring, or changing data."""
        source_bytes = _read_bytes(questions_source, "Official question source")
        questions = self._loader.load_questions(
            questions_source,
            require_reference_answers=False,
        )
        if _read_bytes(questions_source, "Official question source") != source_bytes:
            raise DataValidationError("Official question source changed while loading")
        batch = load_completed_competition_batch(batch_directory)
        source_sha256 = _sha256(source_bytes)
        expected_ids = [question.question_id for question in questions]
        actual_ids = [record.question_id for record in batch.records]
        violations: list[str] = []
        if batch.manifest.question_source_sha256 != source_sha256:
            violations.append("question_source_sha256_mismatch")
        if actual_ids != expected_ids:
            violations.append("question_ids_or_order_mismatch")

        analysis = CompetitionBatchAnalysisService(clock=self._clock).analyze(
            batch_directory
        )
        if (
            analysis.retrieval_model_error_count
            > policy.max_retrieval_model_error_count
        ):
            violations.append("retrieval_model_error_limit_exceeded")
        if (
            analysis.generator_model_error_count
            > policy.max_generator_model_error_count
        ):
            violations.append("generator_model_error_limit_exceeded")
        if (
            analysis.citation.verification_failed_count
            > policy.max_citation_verification_failure_count
        ):
            violations.append("citation_verification_failure_limit_exceeded")
        insufficient_rate = analysis.insufficient_evidence_count / analysis.record_count
        if insufficient_rate > policy.max_insufficient_evidence_rate:
            violations.append("insufficient_evidence_rate_limit_exceeded")
        if (
            policy.require_context_selection_trace
            and analysis.context_trace.trace_present_count != analysis.record_count
        ):
            violations.append("context_selection_trace_required")

        return CompetitionBatchReadinessReport(
            checked_at=self._clock(),
            batch_directory=str(batch.directory),
            question_source_sha256=source_sha256,
            records_sha256=batch.manifest.records_sha256,
            policy_sha256=canonical_sha256(policy),
            record_count=analysis.record_count,
            is_ready=not violations,
            violations=violations,
            analysis=analysis,
        )


class CompetitionWarmupScoreComparisonService:
    """Compare two local score reports only when their scorer contracts match."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def compare(
        self,
        baseline_directory: Path,
        candidate_directory: Path,
    ) -> CompetitionWarmupScoreComparisonReport:
        """Return paired case deltas for two scoring reports over one reference set."""
        baseline = load_warmup_score_report(baseline_directory)
        candidate = load_warmup_score_report(candidate_directory)
        _validate_comparable_score_reports(baseline, candidate)
        cases = [
            CompetitionWarmupScoreComparisonCase(
                question_id=baseline_case.question_id,
                exact_match_delta=(
                    candidate_case.exact_match - baseline_case.exact_match
                ),
                meteor_delta=candidate_case.meteor - baseline_case.meteor,
                rouge_l_delta=candidate_case.rouge_l - baseline_case.rouge_l,
            )
            for baseline_case, candidate_case in zip(
                baseline.cases, candidate.cases, strict=True
            )
        ]
        return CompetitionWarmupScoreComparisonReport(
            compared_at=self._clock(),
            baseline_report_directory=str(baseline_directory.resolve()),
            candidate_report_directory=str(candidate_directory.resolve()),
            metric_mode=baseline.metric_mode,
            reference_source_sha256=baseline.reference_source_sha256,
            official_scorer_sha256=baseline.official_scorer_sha256,
            nltk_version=baseline.nltk_version,
            numpy_version=baseline.numpy_version,
            question_count=len(cases),
            exact_match=_metric_comparison(
                baseline.exact_match,
                candidate.exact_match,
                [case.exact_match_delta for case in cases],
            ),
            meteor=_metric_comparison(
                baseline.meteor,
                candidate.meteor,
                [case.meteor_delta for case in cases],
            ),
            rouge_l=_metric_comparison(
                baseline.rouge_l,
                candidate.rouge_l,
                [case.rouge_l_delta for case in cases],
            ),
            cases=cases,
            warnings=(
                [
                    "official_metric_parity_depends_on_exact_wordnet_resource_bytes",
                ]
                if baseline.metric_mode == CompetitionMetricMode.OFFICIAL_COMPATIBLE
                else ["comparison_uses_diagnostic_metrics_not_official_equivalent"]
            ),
        )


def load_batch_readiness_policy(path: Path) -> CompetitionBatchReadinessPolicy:
    """Load one explicit operator policy without implicit quality thresholds."""
    try:
        return CompetitionBatchReadinessPolicy.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ConfigurationError("Competition batch readiness policy is invalid") from error


def load_warmup_score_report(directory: Path) -> CompetitionWarmupScoreReport:
    """Load a typed score report from an immutable scorer output directory."""
    try:
        return CompetitionWarmupScoreReport.model_validate_json(
            (directory / WARMUP_SCORE_REPORT_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError("Warm-up score report cannot be loaded") from error


def require_ready_competition_batch(
    *,
    questions_source: Path,
    batch_directory: Path,
    readiness_report_directory: Path,
) -> CompetitionBatchReadinessReport:
    """Require one ready report that still matches exact submission inputs."""
    try:
        report = CompetitionBatchReadinessReport.model_validate_json(
            (readiness_report_directory / BATCH_READINESS_FILENAME).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError("Batch readiness report cannot be loaded") from error
    if not report.is_ready:
        raise ArtifactCompatibilityError("Batch readiness report rejected this batch")

    source_sha256 = _sha256(_read_bytes(questions_source, "Official question source"))
    batch = load_completed_competition_batch(batch_directory)
    if report.question_source_sha256 != source_sha256:
        raise ArtifactCompatibilityError(
            "Batch readiness report does not match official question bytes"
        )
    if report.records_sha256 != batch.manifest.records_sha256:
        raise ArtifactCompatibilityError(
            "Batch readiness report does not match completed batch records"
        )
    if report.analysis.question_source_sha256 != batch.manifest.question_source_sha256:
        raise ArtifactCompatibilityError(
            "Batch readiness analysis does not match completed batch source"
        )
    return report


def persist_batch_readiness_report(
    report: CompetitionBatchReadinessReport,
    destination: Path,
) -> Path:
    """Persist one readiness decision without overwriting prior evidence."""
    return _persist_json_report(report, destination, BATCH_READINESS_FILENAME)


def persist_warmup_score_comparison_report(
    report: CompetitionWarmupScoreComparisonReport,
    destination: Path,
) -> Path:
    """Persist one paired score comparison without overwriting prior evidence."""
    return _persist_json_report(
        report, destination, WARMUP_SCORE_COMPARISON_FILENAME
    )


def _validate_comparable_score_reports(
    baseline: CompetitionWarmupScoreReport,
    candidate: CompetitionWarmupScoreReport,
) -> None:
    _validate_score_report_aggregates(baseline)
    _validate_score_report_aggregates(candidate)
    if (
        baseline.reference_source_sha256 != candidate.reference_source_sha256
        or baseline.metric_mode != candidate.metric_mode
        or baseline.question_count != candidate.question_count
        or baseline.official_scorer_sha256 != candidate.official_scorer_sha256
        or baseline.nltk_version != candidate.nltk_version
        or baseline.numpy_version != candidate.numpy_version
    ):
        raise ArtifactCompatibilityError("Warm-up score reports are incompatible")
    if [case.question_id for case in baseline.cases] != [
        case.question_id for case in candidate.cases
    ]:
        raise ArtifactCompatibilityError("Warm-up score report question IDs differ")


def _validate_score_report_aggregates(
    report: CompetitionWarmupScoreReport,
) -> None:
    expected = {
        "exact_match": fmean(case.exact_match for case in report.cases),
        "meteor": fmean(case.meteor for case in report.cases),
        "rouge_l": fmean(case.rouge_l for case in report.cases),
    }
    actual = {
        "exact_match": report.exact_match,
        "meteor": report.meteor,
        "rouge_l": report.rouge_l,
    }
    if any(abs(actual[name] - value) > 1e-12 for name, value in expected.items()):
        raise ArtifactCompatibilityError(
            "Warm-up score report aggregates do not match per-question scores"
        )


def _metric_comparison(
    baseline_mean: float,
    candidate_mean: float,
    deltas: list[float],
) -> CompetitionWarmupMetricComparison:
    tolerance = 1e-12
    return CompetitionWarmupMetricComparison(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        mean_delta=candidate_mean - baseline_mean,
        improved_case_count=sum(delta > tolerance for delta in deltas),
        regressed_case_count=sum(delta < -tolerance for delta in deltas),
        tied_case_count=sum(abs(delta) <= tolerance for delta in deltas),
    )


def _persist_json_report(report: object, destination: Path, filename: str) -> Path:
    if destination.exists():
        raise ArtifactCompatibilityError("Competition report destination already exists")
    try:
        destination.mkdir(parents=True)
        (destination / filename).write_text(
            json.dumps(
                report.model_dump(mode="json"),  # type: ignore[attr-defined]
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise ArtifactCompatibilityError("Competition report could not be persisted") from error
    return destination


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise DataValidationError(f"{label} cannot be read") from error


def _sha256(value: bytes) -> str:
    from hashlib import sha256

    return sha256(value).hexdigest()
