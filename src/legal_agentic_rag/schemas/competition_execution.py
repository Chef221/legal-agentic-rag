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


class CompetitionWarmupScoreReport(BaseModel):
    """Reproducible answer-only diagnostic report for one submission archive."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
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
        return self
