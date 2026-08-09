"""Recovery and result schemas for official competition execution."""

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
