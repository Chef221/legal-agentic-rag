"""Content-free analysis and comparison of completed competition batches."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from statistics import fmean
from typing import Any, Callable

from pydantic import ValidationError

from legal_agentic_rag.competition.uit_dsc_2026.batch_inference import (
    BATCH_MANIFEST_FILENAME,
    BATCH_RECORDS_FILENAME,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import (
    CompetitionBatchAnalysisReport,
    CompetitionBatchCaseComparison,
    CompetitionBatchCitationSummary,
    CompetitionBatchComparisonReport,
    CompetitionBatchContextTraceSummary,
    CompetitionBatchLatencySummary,
    CompetitionBatchManifest,
    CompetitionBatchRecord,
    CitationVerificationResult,
)

BATCH_ANALYSIS_FILENAME = "batch_analysis.json"
BATCH_COMPARISON_FILENAME = "batch_comparison.json"
_CITATION_FAILURE_WARNING = "citation_verification_failed"
_GENERATOR_MODEL_ERROR_WARNING = "generator:model_error"
_RETRIEVAL_MODEL_ERROR_WARNING = "retrieval:model_error"


class CompetitionBatchAnalysisService:
    """Validate and summarize a complete internal competition batch."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def analyze(self, batch_directory: Path) -> CompetitionBatchAnalysisReport:
        """Return one content-free quality and failure report for a batch."""
        batch = load_completed_competition_batch(batch_directory)
        warnings = Counter[str]()
        stop_reasons = Counter[str]()
        claim_errors = Counter[str]()
        selection_reasons = Counter[str]()
        insufficient_count = 0
        generator_error_count = 0
        generation_failure_codes = Counter[str]()
        generator_error_unclassified_count = 0
        schema_issue_codes = Counter[str]()
        schema_repair_codes = Counter[str]()
        schema_recovery_attempted_count = 0
        schema_recovery_succeeded_count = 0
        schema_recovery_failed_count = 0
        schema_recovery_outcomes = Counter[str]()
        retrieval_error_count = 0
        verification_present_count = 0
        verification_failed_count = 0
        citation_warning_failure_count = 0
        numeric_repair_attempted_count = 0
        numeric_repair_succeeded_count = 0
        numeric_repair_failed_count = 0
        numeric_repair_outcomes = Counter[str]()
        supported_claim_salvage_attempted_count = 0
        supported_claim_salvage_succeeded_count = 0
        supported_claim_salvage_failed_count = 0
        supported_claim_salvage_outcomes = Counter[str]()
        context_trace_present_count = 0
        selected_evidence_count = 0
        latencies: list[float] = []

        for record in batch.records:
            response = record.response
            metadata = _object_metadata(response.metadata)
            warnings.update(response.warnings)
            insufficient_count += int(response.insufficient_evidence)
            generator_error_count += int(
                _GENERATOR_MODEL_ERROR_WARNING in response.warnings
            )
            if _GENERATOR_MODEL_ERROR_WARNING in response.warnings:
                failure = _object_value(metadata, "generation_failure")
                failure_code = _string_value(failure, "failure_code")
                if failure_code is None:
                    generator_error_unclassified_count += 1
                else:
                    generation_failure_codes[failure_code] += 1
            failure = _object_value(metadata, "generation_failure")
            recovery = _object_value(metadata, "schema_recovery")
            if recovery is None:
                recovery = _object_value(failure, "schema_recovery")
            issue_values = _list_value(recovery, "issue_codes")
            if issue_values is None:
                issue_values = _list_value(failure, "schema_issue_codes")
            if issue_values is not None:
                schema_issue_codes.update(
                    value for value in issue_values if isinstance(value, str) and value.strip()
                )
            if recovery is not None and recovery.get("attempted") is True:
                schema_recovery_attempted_count += 1
                outcome = _string_value(recovery, "outcome") or "missing"
                schema_recovery_outcomes[outcome] += 1
                repair_values = _list_value(recovery, "repair_codes") or []
                schema_repair_codes.update(
                    value for value in repair_values if isinstance(value, str) and value.strip()
                )
                if outcome == "succeeded":
                    schema_recovery_succeeded_count += 1
                else:
                    schema_recovery_failed_count += 1
            retrieval_error_count += int(
                _RETRIEVAL_MODEL_ERROR_WARNING in response.warnings
            )
            citation_warning_failure_count += int(
                _CITATION_FAILURE_WARNING in response.warnings
            )
            stop_reason = _string_value(_object_value(metadata, "agent"), "stop_reason")
            if stop_reason is None:
                stop_reasons["missing"] += 1
            else:
                stop_reasons[stop_reason] += 1

            numeric_repair = _object_value(metadata, "numeric_repair")
            numeric_repair_attempted = (
                numeric_repair is not None and numeric_repair.get("attempted") is True
            )
            citation_result = _citation_result(metadata)
            if citation_result is not None:
                verification_present_count += 1
                if not numeric_repair_attempted:
                    for claim in citation_result.claim_verifications:
                        claim_errors.update(claim.errors)

            if numeric_repair_attempted:
                assert numeric_repair is not None
                numeric_repair_attempted_count += 1
                outcome = _string_value(numeric_repair, "outcome") or "missing"
                numeric_repair_outcomes[outcome] += 1
                if outcome in {
                    "salvage_succeeded",
                    "model_regeneration_succeeded",
                    "succeeded",
                }:
                    numeric_repair_succeeded_count += 1
                else:
                    numeric_repair_failed_count += 1
                _collect_initial_numeric_repair_errors(numeric_repair, claim_errors)

            claim_salvage = _object_value(metadata, "claim_salvage")
            if claim_salvage is not None and claim_salvage.get("attempted") is True:
                supported_claim_salvage_attempted_count += 1
                outcome = _string_value(claim_salvage, "outcome") or "missing"
                supported_claim_salvage_outcomes[outcome] += 1
                if outcome == "succeeded":
                    supported_claim_salvage_succeeded_count += 1
                else:
                    supported_claim_salvage_failed_count += 1
                _collect_initial_claim_salvage_errors(claim_salvage, claim_errors)

            context = _object_value(metadata, "context")
            selection_trace = _list_value(context, "selection_trace")
            if selection_trace is not None:
                context_trace_present_count += 1
                selected_evidence_count += _integer_value(context, "selected_count")
                for trace in selection_trace:
                    if isinstance(trace, dict):
                        reason = trace.get("reason")
                        if isinstance(reason, str) and reason.strip():
                            selection_reasons[reason] += 1
            latency = _number_value(_object_value(metadata, "agent"), "total_latency_ms")
            if latency is not None:
                latencies.append(latency)

        report_warnings: list[str] = []
        if not context_trace_present_count:
            report_warnings.append("context_selection_trace_not_present")
        elif context_trace_present_count != len(batch.records):
            report_warnings.append("context_selection_trace_partially_present")
        if verification_present_count != len(batch.records):
            report_warnings.append("citation_verification_metadata_partially_present")
        verification_failed_count = citation_warning_failure_count

        return CompetitionBatchAnalysisReport(
            analyzed_at=self._clock(),
            batch_directory=str(batch.directory),
            question_source_sha256=batch.manifest.question_source_sha256,
            application_config_hash=batch.manifest.application_config_hash,
            code_version=batch.manifest.code_version,
            records_sha256=batch.manifest.records_sha256,
            record_count=len(batch.records),
            unique_question_id_count=len({record.question_id for record in batch.records}),
            insufficient_evidence_count=insufficient_count,
            generator_model_error_count=generator_error_count,
            generation_failure_code_counts=dict(sorted(generation_failure_codes.items())),
            generator_model_error_unclassified_count=generator_error_unclassified_count,
            generation_schema_issue_counts=dict(sorted(schema_issue_codes.items())),
            generation_schema_repair_code_counts=dict(sorted(schema_repair_codes.items())),
            schema_recovery_attempted_count=schema_recovery_attempted_count,
            schema_recovery_succeeded_count=schema_recovery_succeeded_count,
            schema_recovery_failed_count=schema_recovery_failed_count,
            schema_recovery_outcome_counts=dict(sorted(schema_recovery_outcomes.items())),
            retrieval_model_error_count=retrieval_error_count,
            stop_reason_counts=dict(sorted(stop_reasons.items())),
            warning_counts=dict(sorted(warnings.items())),
            citation=CompetitionBatchCitationSummary(
                verification_present_count=verification_present_count,
                verification_failed_count=verification_failed_count,
                claim_error_counts=dict(sorted(claim_errors.items())),
                numeric_repair_attempted_count=numeric_repair_attempted_count,
                numeric_repair_succeeded_count=numeric_repair_succeeded_count,
                numeric_repair_failed_count=numeric_repair_failed_count,
                numeric_repair_outcome_counts=dict(sorted(numeric_repair_outcomes.items())),
                supported_claim_salvage_attempted_count=(
                    supported_claim_salvage_attempted_count
                ),
                supported_claim_salvage_succeeded_count=(
                    supported_claim_salvage_succeeded_count
                ),
                supported_claim_salvage_failed_count=supported_claim_salvage_failed_count,
                supported_claim_salvage_outcome_counts=dict(
                    sorted(supported_claim_salvage_outcomes.items())
                ),
            ),
            context_trace=CompetitionBatchContextTraceSummary(
                trace_present_count=context_trace_present_count,
                selected_evidence_count=selected_evidence_count,
                selection_reason_counts=dict(sorted(selection_reasons.items())),
            ),
            agent_latency=_latency_summary(latencies),
            warnings=report_warnings,
        )


class CompetitionBatchComparisonService:
    """Compare two complete batches over exactly the same official questions."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def compare(
        self,
        baseline_directory: Path,
        candidate_directory: Path,
    ) -> CompetitionBatchComparisonReport:
        """Return content-free per-question outcome deltas for compatible batches."""
        baseline = load_completed_competition_batch(baseline_directory)
        candidate = load_completed_competition_batch(candidate_directory)
        _validate_comparable_batches(baseline, candidate)

        answer_changed_count = 0
        stop_transitions = Counter[str]()
        insufficient_transitions = Counter[str]()
        citation_transitions = Counter[str]()
        generator_transitions = Counter[str]()
        changed_cases: list[CompetitionBatchCaseComparison] = []

        for baseline_record, candidate_record in zip(
            baseline.records, candidate.records, strict=True
        ):
            baseline_snapshot = _outcome_snapshot(baseline_record)
            candidate_snapshot = _outcome_snapshot(candidate_record)
            answer_changed = _answer_sha256(baseline_record) != _answer_sha256(
                candidate_record
            )
            answer_changed_count += int(answer_changed)
            stop_transitions[
                _transition(
                    baseline_snapshot.stop_reason, candidate_snapshot.stop_reason
                )
            ] += 1
            insufficient_transitions[
                _transition_bool(
                    baseline_snapshot.insufficient_evidence,
                    candidate_snapshot.insufficient_evidence,
                )
            ] += 1
            citation_transitions[
                _transition_bool(
                    baseline_snapshot.citation_verification_failed,
                    candidate_snapshot.citation_verification_failed,
                )
            ] += 1
            generator_transitions[
                _transition_bool(
                    baseline_snapshot.generator_model_error,
                    candidate_snapshot.generator_model_error,
                )
            ] += 1
            case = CompetitionBatchCaseComparison(
                question_id=baseline_record.question_id,
                answer_changed=answer_changed,
                baseline_stop_reason=baseline_snapshot.stop_reason,
                candidate_stop_reason=candidate_snapshot.stop_reason,
                baseline_insufficient_evidence=baseline_snapshot.insufficient_evidence,
                candidate_insufficient_evidence=candidate_snapshot.insufficient_evidence,
                baseline_citation_verification_failed=(
                    baseline_snapshot.citation_verification_failed
                ),
                candidate_citation_verification_failed=(
                    candidate_snapshot.citation_verification_failed
                ),
                baseline_generator_model_error=(
                    baseline_snapshot.generator_model_error
                ),
                candidate_generator_model_error=(
                    candidate_snapshot.generator_model_error
                ),
                baseline_citation_count=baseline_snapshot.citation_count,
                candidate_citation_count=candidate_snapshot.citation_count,
                agent_latency_delta_ms=_latency_delta(
                    baseline_snapshot.agent_latency_ms,
                    candidate_snapshot.agent_latency_ms,
                ),
            )
            if _case_changed(case):
                changed_cases.append(case)

        return CompetitionBatchComparisonReport(
            compared_at=self._clock(),
            baseline_directory=str(baseline.directory),
            candidate_directory=str(candidate.directory),
            question_source_sha256=baseline.manifest.question_source_sha256,
            baseline_application_config_hash=(
                baseline.manifest.application_config_hash
            ),
            candidate_application_config_hash=(
                candidate.manifest.application_config_hash
            ),
            baseline_code_version=baseline.manifest.code_version,
            candidate_code_version=candidate.manifest.code_version,
            record_count=len(baseline.records),
            answer_changed_count=answer_changed_count,
            stop_reason_transition_counts=dict(sorted(stop_transitions.items())),
            insufficient_evidence_transition_counts=dict(
                sorted(insufficient_transitions.items())
            ),
            citation_failure_transition_counts=dict(sorted(citation_transitions.items())),
            generator_model_error_transition_counts=dict(
                sorted(generator_transitions.items())
            ),
            changed_cases=changed_cases,
        )


def persist_batch_analysis_report(
    report: CompetitionBatchAnalysisReport,
    destination: Path,
) -> Path:
    """Persist an analysis report in a new immutable output directory."""
    return _persist_report(report, destination, BATCH_ANALYSIS_FILENAME)


def persist_batch_comparison_report(
    report: CompetitionBatchComparisonReport,
    destination: Path,
) -> Path:
    """Persist a comparison report in a new immutable output directory."""
    return _persist_report(report, destination, BATCH_COMPARISON_FILENAME)


class _LoadedCompetitionBatch:
    """Validated internal representation of one complete batch directory."""

    def __init__(
        self,
        directory: Path,
        manifest: CompetitionBatchManifest,
        records: list[CompetitionBatchRecord],
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.records = records


class _OutcomeSnapshot:
    """Content-free fields needed for a single batch comparison row."""

    def __init__(
        self,
        *,
        stop_reason: str | None,
        insufficient_evidence: bool,
        citation_verification_failed: bool,
        generator_model_error: bool,
        citation_count: int,
        agent_latency_ms: float | None,
    ) -> None:
        self.stop_reason = stop_reason
        self.insufficient_evidence = insufficient_evidence
        self.citation_verification_failed = citation_verification_failed
        self.generator_model_error = generator_model_error
        self.citation_count = citation_count
        self.agent_latency_ms = agent_latency_ms


def load_completed_competition_batch(batch_directory: Path) -> _LoadedCompetitionBatch:
    """Load a complete batch only after checking its manifest and record bytes."""
    directory = batch_directory.resolve()
    manifest_path = directory / BATCH_MANIFEST_FILENAME
    records_path = directory / BATCH_RECORDS_FILENAME
    try:
        manifest = CompetitionBatchManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        records_payload = records_path.read_bytes()
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Completed competition batch cannot be loaded"
        ) from error
    if manifest.output_format != "internal_answer_response_jsonl_v1":
        raise ArtifactCompatibilityError("Completed competition batch format is unsupported")
    if _sha256_bytes(records_payload) != manifest.records_sha256:
        raise ArtifactCompatibilityError(
            "Completed competition batch records do not match manifest SHA-256"
        )
    try:
        lines = records_payload.decode("utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError("records must contain no blank lines")
        records = [CompetitionBatchRecord.model_validate_json(line) for line in lines]
    except (UnicodeError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Completed competition batch records are invalid"
        ) from error
    record_ids = [record.question_id for record in records]
    if len(records) != manifest.record_count or len(record_ids) != len(set(record_ids)):
        raise ArtifactCompatibilityError(
            "Completed competition batch record count or IDs are invalid"
        )
    return _LoadedCompetitionBatch(directory, manifest, records)


def _persist_report(report: Any, destination: Path, filename: str) -> Path:
    if destination.exists():
        raise ArtifactCompatibilityError("Batch report destination already exists")
    try:
        destination.mkdir(parents=True)
        (destination / filename).write_text(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise ArtifactCompatibilityError("Batch report could not be persisted") from error
    return destination


def _validate_comparable_batches(
    baseline: _LoadedCompetitionBatch,
    candidate: _LoadedCompetitionBatch,
) -> None:
    if baseline.manifest.question_source_sha256 != candidate.manifest.question_source_sha256:
        raise ArtifactCompatibilityError("Batch question source SHA-256 differs")
    baseline_ids = [record.question_id for record in baseline.records]
    candidate_ids = [record.question_id for record in candidate.records]
    if baseline_ids != candidate_ids:
        raise ArtifactCompatibilityError("Batch question IDs or order differ")


def _outcome_snapshot(record: CompetitionBatchRecord) -> _OutcomeSnapshot:
    response = record.response
    metadata = _object_metadata(response.metadata)
    agent = _object_value(metadata, "agent")
    return _OutcomeSnapshot(
        stop_reason=_string_value(agent, "stop_reason"),
        insufficient_evidence=response.insufficient_evidence,
        citation_verification_failed=(
            _CITATION_FAILURE_WARNING in response.warnings
        ),
        generator_model_error=(_GENERATOR_MODEL_ERROR_WARNING in response.warnings),
        citation_count=len(response.citations),
        agent_latency_ms=_number_value(agent, "total_latency_ms"),
    )


def _citation_result(metadata: dict[str, Any]) -> CitationVerificationResult | None:
    payload = metadata.get("citation_verification")
    if payload is None:
        return None
    try:
        return CitationVerificationResult.model_validate(payload)
    except ValidationError:
        return None


def _collect_initial_numeric_repair_errors(
    numeric_repair: dict[str, Any],
    claim_errors: Counter[str],
) -> None:
    """Count content-free initial verifier errors once for numeric-repair records."""
    initial = _object_value(numeric_repair, "initial_verification")
    claims = _list_value(initial, "claim_verifications")
    if claims is None:
        return
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        errors = claim.get("errors")
        if isinstance(errors, list):
            claim_errors.update(
                value for value in errors if isinstance(value, str) and value.strip()
            )


def _collect_initial_claim_salvage_errors(
    claim_salvage: dict[str, Any],
    claim_errors: Counter[str],
) -> None:
    """Count initial verifier codes for a general salvage exactly once."""
    initial = _object_value(claim_salvage, "initial_verification")
    claims = _list_value(initial, "claim_verifications")
    if claims is None:
        return
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        errors = claim.get("errors")
        if isinstance(errors, list):
            claim_errors.update(
                value for value in errors if isinstance(value, str) and value.strip()
            )


def _object_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items()}


def _object_value(
    parent: dict[str, Any] | None,
    key: str,
) -> dict[str, Any] | None:
    if parent is None:
        return None
    value = parent.get(key)
    return value if isinstance(value, dict) else None


def _list_value(
    parent: dict[str, Any] | None,
    key: str,
) -> list[Any] | None:
    if parent is None:
        return None
    value = parent.get(key)
    return value if isinstance(value, list) else None


def _string_value(parent: dict[str, Any] | None, key: str) -> str | None:
    if parent is None:
        return None
    value = parent.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _integer_value(parent: dict[str, Any] | None, key: str) -> int:
    if parent is None:
        return 0
    value = parent.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _number_value(parent: dict[str, Any] | None, key: str) -> float | None:
    if parent is None:
        return None
    value = parent.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _latency_summary(values: list[float]) -> CompetitionBatchLatencySummary:
    if not values:
        return CompetitionBatchLatencySummary(count=0)
    ordered = sorted(values)
    return CompetitionBatchLatencySummary(
        count=len(ordered),
        mean_ms=fmean(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        max_ms=ordered[-1],
    )


def _percentile(values: list[float], fraction: float) -> float:
    index = max(0, min(len(values) - 1, round((len(values) - 1) * fraction)))
    return values[index]


def _answer_sha256(record: CompetitionBatchRecord) -> str:
    return _sha256_bytes(record.response.answer.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _transition(before: str | None, after: str | None) -> str:
    return f"{before or 'missing'}->{after or 'missing'}"


def _transition_bool(before: bool, after: bool) -> str:
    return f"{str(before).lower()}->{str(after).lower()}"


def _latency_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _case_changed(case: CompetitionBatchCaseComparison) -> bool:
    return any(
        (
            case.answer_changed,
            case.baseline_stop_reason != case.candidate_stop_reason,
            case.baseline_insufficient_evidence != case.candidate_insufficient_evidence,
            case.baseline_citation_verification_failed
            != case.candidate_citation_verification_failed,
            case.baseline_generator_model_error != case.candidate_generator_model_error,
            case.baseline_citation_count != case.candidate_citation_count,
        )
    )
