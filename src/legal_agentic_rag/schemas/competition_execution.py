"""Recovery and result schemas for official competition execution."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from legal_agentic_rag.schemas.answering import AnswerResponse
from legal_agentic_rag.schemas.build_validation import BuildValidationReport


def _validate_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")
    return value


def _validate_source_revision(value: str) -> str:
    prefix = "sha256:"
    if not value.startswith(prefix):
        raise ValueError("source revision must use the sha256 prefix")
    _validate_sha256(value[len(prefix) :])
    return value


class CompetitionBuildStage(StrEnum):
    """Ordered durable stages of one official corpus build."""

    CORPUS = "corpus"
    DOCUMENT_PROCESSING = "document_processing"
    BM25 = "bm25"
    VECTOR = "vector"
    VALIDATION = "validation"


class CompetitionBuildState(BaseModel):
    """Atomic recovery identity and completed stages for one official build."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source_revision: str
    application_config_hash: str
    code_version: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    completed_stages: list[CompetitionBuildStage] = Field(default_factory=list)

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        """Require the canonical prefixed official source revision."""
        return _validate_source_revision(value)

    @field_validator("application_config_hash")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        """Require an exact application configuration digest."""
        return _validate_sha256(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require unambiguous recovery timestamps."""
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_stage_order(self) -> "CompetitionBuildState":
        """Completed stages must be a unique prefix of the fixed build order."""
        order = list(CompetitionBuildStage)
        if self.completed_stages != order[: len(self.completed_stages)]:
            raise ValueError("completed build stages must be an ordered prefix")
        if self.updated_at < self.created_at:
            raise ValueError("build update time cannot precede creation time")
        return self


class CompetitionOfflineBuildResult(BaseModel):
    """Summary returned after one complete or stage-limited official build."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: str = Field(min_length=1)
    source_revision: str
    resumed: bool
    completed_stages: list[CompetitionBuildStage]
    validation_report: BuildValidationReport | None = None

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        """Pin the exact official corpus bytes used by the build."""
        return _validate_source_revision(value)

    @model_validator(mode="after")
    def validate_report_stage(self) -> "CompetitionOfflineBuildResult":
        """Expose final validation only when the validation stage is complete."""
        validation_completed = (
            CompetitionBuildStage.VALIDATION in self.completed_stages
        )
        if validation_completed != (self.validation_report is not None):
            raise ValueError(
                "validation report must match validation-stage completion"
            )
        return self


class CompetitionBatchRecord(BaseModel):
    """One internal prediction checkpoint keyed by official question ID."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    response: AnswerResponse


class CompetitionBatchState(BaseModel):
    """Mutable-by-atomic-replacement progress state for batch inference."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    question_source_sha256: str
    application_config_hash: str
    code_version: str = Field(min_length=1)
    question_count: int = Field(gt=0)
    completed_question_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @field_validator("question_source_sha256", "application_config_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require exact source and runtime identities."""
        return _validate_sha256(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require unambiguous progress timestamps."""
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_progress(self) -> "CompetitionBatchState":
        """Reject duplicate or impossible completed-question counts."""
        if len(self.completed_question_ids) != len(set(self.completed_question_ids)):
            raise ValueError("completed question IDs must be unique")
        if len(self.completed_question_ids) > self.question_count:
            raise ValueError("completed question count exceeds source count")
        if self.updated_at < self.created_at:
            raise ValueError("batch update time cannot precede creation time")
        return self


class CompetitionBatchManifest(BaseModel):
    """Immutable proof that one internal prediction batch is complete."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    question_source_sha256: str
    application_config_hash: str
    code_version: str = Field(min_length=1)
    created_at: datetime
    record_count: int = Field(gt=0)
    records_sha256: str
    output_format: str = "internal_answer_response_jsonl_v1"

    @field_validator(
        "question_source_sha256", "application_config_hash", "records_sha256"
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require reproducible source, runtime, and output identities."""
        return _validate_sha256(value)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require an unambiguous completion timestamp."""
        return _validate_timestamp(value)


class CompetitionBatchLatencySummary(BaseModel):
    """Content-free latency distribution for one completed competition batch."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    count: int = Field(ge=0)
    mean_ms: float | None = Field(default=None, ge=0)
    p50_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    max_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_population(self) -> "CompetitionBatchLatencySummary":
        """Require values exactly when at least one latency was observed."""
        values = (self.mean_ms, self.p50_ms, self.p95_ms, self.max_ms)
        if (self.count == 0) != all(value is None for value in values):
            raise ValueError("latency values must match the observed count")
        if self.count and any(value is None for value in values):
            raise ValueError("observed latencies require a complete summary")
        return self


class CompetitionBatchCitationSummary(BaseModel):
    """Aggregate verifier and bounded numeric-repair outcomes without legal text."""

    model_config = ConfigDict(extra="forbid")

    verification_present_count: int = Field(ge=0)
    verification_failed_count: int = Field(ge=0)
    claim_error_counts: dict[str, int] = Field(default_factory=dict)
    numeric_repair_attempted_count: int = Field(default=0, ge=0)
    numeric_repair_succeeded_count: int = Field(default=0, ge=0)
    numeric_repair_failed_count: int = Field(default=0, ge=0)
    numeric_repair_outcome_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("claim_error_counts")
    @classmethod
    def validate_error_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require non-empty error codes and positive aggregate counts."""
        if any(not key.strip() or count <= 0 for key, count in value.items()):
            raise ValueError("claim error counts must have non-empty positive entries")
        return value

    @field_validator("numeric_repair_outcome_counts")
    @classmethod
    def validate_repair_outcome_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require non-empty bounded-repair outcomes with positive counts."""
        if any(not key.strip() or count <= 0 for key, count in value.items()):
            raise ValueError("numeric repair outcomes must have non-empty positive entries")
        return value

    @model_validator(mode="after")
    def validate_numeric_repair_counts(self) -> "CompetitionBatchCitationSummary":
        """Keep one terminal outcome for every recorded numeric repair attempt."""
        if self.numeric_repair_succeeded_count + self.numeric_repair_failed_count != (
            self.numeric_repair_attempted_count
        ):
            raise ValueError("numeric repair outcomes must equal attempted repairs")
        if self.numeric_repair_outcome_counts and sum(
            self.numeric_repair_outcome_counts.values()
        ) != self.numeric_repair_attempted_count:
            raise ValueError("numeric repair outcome counts must equal attempted repairs")
        return self


class CompetitionBatchContextTraceSummary(BaseModel):
    """Aggregate persisted evidence-selection telemetry when a batch has it."""

    model_config = ConfigDict(extra="forbid")

    trace_present_count: int = Field(ge=0)
    selected_evidence_count: int = Field(ge=0)
    selection_reason_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("selection_reason_counts")
    @classmethod
    def validate_reason_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require non-empty selection reasons with positive aggregate counts."""
        if any(not key.strip() or count <= 0 for key, count in value.items()):
            raise ValueError("selection reason counts must have non-empty positive entries")
        return value


class CompetitionBatchAnalysisReport(BaseModel):
    """Immutable, content-free analysis of one completed internal batch."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    analyzed_at: datetime
    batch_directory: str = Field(min_length=1)
    question_source_sha256: str
    application_config_hash: str
    code_version: str = Field(min_length=1)
    records_sha256: str
    record_count: int = Field(gt=0)
    unique_question_id_count: int = Field(gt=0)
    insufficient_evidence_count: int = Field(ge=0)
    generator_model_error_count: int = Field(ge=0)
    retrieval_model_error_count: int = Field(ge=0)
    stop_reason_counts: dict[str, int] = Field(default_factory=dict)
    warning_counts: dict[str, int] = Field(default_factory=dict)
    citation: CompetitionBatchCitationSummary
    context_trace: CompetitionBatchContextTraceSummary
    agent_latency: CompetitionBatchLatencySummary
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "question_source_sha256",
        "application_config_hash",
        "records_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require exact input and batch-output identities."""
        return _validate_sha256(value)

    @field_validator("analyzed_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require an unambiguous report timestamp."""
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_counts(self) -> "CompetitionBatchAnalysisReport":
        """Keep batch outcome counts within the completed batch population."""
        if self.unique_question_id_count != self.record_count:
            raise ValueError("batch question IDs must be unique and complete")
        bounded_counts = (
            self.insufficient_evidence_count,
            self.generator_model_error_count,
            self.retrieval_model_error_count,
            self.citation.verification_present_count,
            self.citation.verification_failed_count,
            self.citation.numeric_repair_attempted_count,
            self.citation.numeric_repair_succeeded_count,
            self.citation.numeric_repair_failed_count,
            self.context_trace.trace_present_count,
        )
        if any(count > self.record_count for count in bounded_counts):
            raise ValueError("aggregate outcome count exceeds batch record count")
        return self


class CompetitionBatchCaseComparison(BaseModel):
    """Content-free outcome delta for one common official question ID."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    question_id: str = Field(min_length=1)
    answer_changed: bool
    baseline_stop_reason: str | None = None
    candidate_stop_reason: str | None = None
    baseline_insufficient_evidence: bool
    candidate_insufficient_evidence: bool
    baseline_citation_verification_failed: bool
    candidate_citation_verification_failed: bool
    baseline_generator_model_error: bool
    candidate_generator_model_error: bool
    baseline_citation_count: int = Field(ge=0)
    candidate_citation_count: int = Field(ge=0)
    agent_latency_delta_ms: float | None = None

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        """Reject blank official question identities."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question ID must not be blank")
        return normalized


class CompetitionBatchComparisonReport(BaseModel):
    """Immutable, per-question comparison of two compatible completed batches."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    compared_at: datetime
    baseline_directory: str = Field(min_length=1)
    candidate_directory: str = Field(min_length=1)
    question_source_sha256: str
    baseline_application_config_hash: str
    candidate_application_config_hash: str
    baseline_code_version: str = Field(min_length=1)
    candidate_code_version: str = Field(min_length=1)
    record_count: int = Field(gt=0)
    answer_changed_count: int = Field(ge=0)
    stop_reason_transition_counts: dict[str, int] = Field(default_factory=dict)
    insufficient_evidence_transition_counts: dict[str, int] = Field(
        default_factory=dict
    )
    citation_failure_transition_counts: dict[str, int] = Field(default_factory=dict)
    generator_model_error_transition_counts: dict[str, int] = Field(
        default_factory=dict
    )
    changed_cases: list[CompetitionBatchCaseComparison] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "question_source_sha256",
        "baseline_application_config_hash",
        "candidate_application_config_hash",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require exact source and runtime identities for both runs."""
        return _validate_sha256(value)

    @field_validator("compared_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require an unambiguous report timestamp."""
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_changed_cases(self) -> "CompetitionBatchComparisonReport":
        """Require unique, genuinely changed case entries within the batch size."""
        ids = [case.question_id for case in self.changed_cases]
        if len(ids) != len(set(ids)) or len(ids) > self.record_count:
            raise ValueError("changed comparison case IDs must be unique and bounded")
        if self.answer_changed_count > self.record_count:
            raise ValueError("answer changed count exceeds batch record count")
        return self


class CompetitionBatchReadinessPolicy(BaseModel):
    """Explicit operator limits required before a completed batch is submitted."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    max_retrieval_model_error_count: int = Field(ge=0)
    max_generator_model_error_count: int = Field(ge=0)
    max_citation_verification_failure_count: int = Field(ge=0)
    max_insufficient_evidence_rate: float = Field(ge=0, le=1)
    require_context_selection_trace: bool


class CompetitionBatchReadinessReport(BaseModel):
    """Content-free submission-readiness result under one explicit policy."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    checked_at: datetime
    batch_directory: str = Field(min_length=1)
    question_source_sha256: str
    records_sha256: str
    policy_sha256: str
    record_count: int = Field(gt=0)
    is_ready: bool
    violations: list[str] = Field(default_factory=list)
    analysis: CompetitionBatchAnalysisReport

    @field_validator(
        "question_source_sha256", "records_sha256", "policy_sha256"
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require exact source, completed-record, and policy identities."""
        return _validate_sha256(value)

    @field_validator("checked_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require an unambiguous readiness timestamp."""
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_readiness(self) -> "CompetitionBatchReadinessReport":
        """Require violations exactly when the gate rejects the batch."""
        if self.is_ready == bool(self.violations):
            raise ValueError("readiness violations must match readiness state")
        if self.analysis.record_count != self.record_count:
            raise ValueError("readiness analysis must match batch record count")
        return self


class CompetitionWarmupMetricComparison(BaseModel):
    """Paired outcome counts and mean delta for one score-facing metric."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    baseline_mean: float = Field(ge=0, le=1)
    candidate_mean: float = Field(ge=0, le=1)
    mean_delta: float = Field(ge=-1, le=1)
    improved_case_count: int = Field(ge=0)
    regressed_case_count: int = Field(ge=0)
    tied_case_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_delta(self) -> "CompetitionWarmupMetricComparison":
        """Keep paired counts and aggregate delta internally consistent."""
        if (
            self.improved_case_count
            + self.regressed_case_count
            + self.tied_case_count
            <= 0
        ):
            raise ValueError("metric comparison requires at least one paired case")
        if abs((self.candidate_mean - self.baseline_mean) - self.mean_delta) > 1e-12:
            raise ValueError("mean delta must match candidate minus baseline")
        return self


class CompetitionWarmupScoreComparisonCase(BaseModel):
    """Content-free per-ID score delta from two comparable scoring reports."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    question_id: str = Field(min_length=1)
    exact_match_delta: float = Field(ge=-1, le=1)
    meteor_delta: float = Field(ge=-1, le=1)
    rouge_l_delta: float = Field(ge=-1, le=1)


class CompetitionWarmupScoreComparisonReport(BaseModel):
    """Immutable paired comparison of compatible official score reports."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    compared_at: datetime
    baseline_report_directory: str = Field(min_length=1)
    candidate_report_directory: str = Field(min_length=1)
    metric_mode: CompetitionMetricMode
    reference_source_sha256: str
    official_scorer_sha256: str | None = None
    nltk_version: str | None = None
    numpy_version: str | None = None
    question_count: int = Field(gt=0)
    exact_match: CompetitionWarmupMetricComparison
    meteor: CompetitionWarmupMetricComparison
    rouge_l: CompetitionWarmupMetricComparison
    cases: list[CompetitionWarmupScoreComparisonCase] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("reference_source_sha256")
    @classmethod
    def validate_reference_hash(cls, value: str) -> str:
        """Require the exact common reference source identity."""
        return _validate_sha256(value)

    @field_validator("compared_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require an unambiguous comparison timestamp."""
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_score_contract(self) -> "CompetitionWarmupScoreComparisonReport":
        """Align case population and official scorer provenance with its mode."""
        if self.question_count != len(self.cases):
            raise ValueError("comparison question count must match cases")
        ids = [case.question_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("comparison question IDs must be unique")
        if self.metric_mode == CompetitionMetricMode.OFFICIAL_COMPATIBLE:
            if not all(
                (self.official_scorer_sha256, self.nltk_version, self.numpy_version)
            ):
                raise ValueError("official comparison must retain scorer provenance")
            _validate_sha256(self.official_scorer_sha256 or "")
        elif any(
            value is not None
            for value in (
                self.official_scorer_sha256,
                self.nltk_version,
                self.numpy_version,
            )
        ):
            raise ValueError("diagnostic comparison cannot claim official provenance")
        return self


class CompetitionSubmissionItem(BaseModel):
    """One exact Codabench prediction item published in submission.json."""

    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(min_length=1)
    answer: StrictStr

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Reject an empty official question identity without rewriting it."""
        if not value.strip():
            raise ValueError("submission ID must contain non-whitespace text")
        return value


class CompetitionSubmissionResult(BaseModel):
    """Local proof returned after validating an official submission archive."""

    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(min_length=1)
    question_count: int = Field(gt=0)
    submission_json_sha256: str
    archive_sha256: str

    @field_validator("submission_json_sha256", "archive_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Require exact checksums for the payload and final ZIP bytes."""
        return _validate_sha256(value)


class CompetitionWarmupCaseScore(BaseModel):
    """Content-free diagnostic scores for one official warm-up question."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    question_id: StrictStr = Field(min_length=1)
    exact_match: float = Field(ge=0, le=1)
    meteor: float = Field(ge=0, le=1)
    rouge_l: float = Field(ge=0, le=1)


class CompetitionMetricMode(StrEnum):
    """Supported local answer-scoring contracts."""

    DIAGNOSTIC = "diagnostic"
    OFFICIAL_COMPATIBLE = "official_compatible"


class CompetitionSplitSource(BaseModel):
    """Immutable identity of one official split input."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    sha256: str
    question_count: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value)


class CompetitionSplitPartition(BaseModel):
    """One deterministic local development partition."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    question_count: int = Field(ge=0)
    sha256: str
    question_ids: list[StrictStr]

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value)

    @model_validator(mode="after")
    def validate_identities(self) -> "CompetitionSplitPartition":
        if self.question_count != len(self.question_ids):
            raise ValueError("split partition count must match question IDs")
        if len(self.question_ids) != len(set(self.question_ids)):
            raise ValueError("split partition question IDs must be unique")
        return self


class CompetitionDevelopmentSplitManifest(BaseModel):
    """Reproducible proof of a leakage-aware official development split."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    training_source: CompetitionSplitSource
    holdout_sources: list[CompetitionSplitSource]
    seed: int
    dev_fraction: float = Field(gt=0, lt=1)
    near_duplicate_threshold: float = Field(ge=0.5, le=1)
    exact_duplicate_pair_count: int = Field(ge=0)
    near_duplicate_pair_count: int = Field(ge=0)
    partitions: list[CompetitionSplitPartition] = Field(min_length=3, max_length=3)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_partitions(self) -> "CompetitionDevelopmentSplitManifest":
        names = [partition.filename for partition in self.partitions]
        if len(names) != len(set(names)):
            raise ValueError("split partition filenames must be unique")
        identities = [
            identity
            for partition in self.partitions
            for identity in partition.question_ids
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("question IDs cannot cross split partitions")
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("split warnings must be unique")
        return self


class CompetitionWarmupScoreReport(BaseModel):
    """Reproducible answer-only diagnostic report for one submission archive."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.1"
    created_at: datetime
    code_version: str = Field(min_length=1)
    metric_mode: CompetitionMetricMode = CompetitionMetricMode.DIAGNOSTIC
    official_scorer_sha256: str | None = None
    nltk_version: str | None = None
    numpy_version: str | None = None
    reference_source_sha256: str
    submission_archive_sha256: str
    submission_json_sha256: str
    question_count: int = Field(gt=0)
    exact_match: float = Field(ge=0, le=1)
    meteor: float = Field(ge=0, le=1)
    rouge_l: float = Field(ge=0, le=1)
    cases: list[CompetitionWarmupCaseScore] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "reference_source_sha256",
        "submission_archive_sha256",
        "submission_json_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        """Pin exact reference, archive, and submitted JSON bytes."""
        return _validate_sha256(value)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        """Require an unambiguous scoring timestamp."""
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_cases(self) -> "CompetitionWarmupScoreReport":
        """Keep aggregate count and case identities structurally consistent."""
        if self.question_count != len(self.cases):
            raise ValueError("warm-up score count must match case scores")
        identities = [case.question_id for case in self.cases]
        if len(identities) != len(set(identities)):
            raise ValueError("warm-up score question IDs must be unique")
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("warm-up score warnings must be unique")
        if self.metric_mode == CompetitionMetricMode.OFFICIAL_COMPATIBLE:
            if (
                self.official_scorer_sha256 is None
                or self.nltk_version is None
                or self.numpy_version is None
            ):
                raise ValueError("official-compatible scoring identity is required")
            _validate_sha256(self.official_scorer_sha256)
        elif any(
            value is not None
            for value in (
                self.official_scorer_sha256,
                self.nltk_version,
                self.numpy_version,
            )
        ):
            raise ValueError("diagnostic scoring cannot claim official identity")
        return self
