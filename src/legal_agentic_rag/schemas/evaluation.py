"""Versioned benchmark, per-case metric, and evaluation report schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legal_agentic_rag.schemas.answering import AnswerResponse
from legal_agentic_rag.schemas.retrieval import RetrievalStrategy


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


class EvaluationTargetGranularity(StrEnum):
    """Identity level used by retrieval relevance labels."""

    CHUNK = "chunk"
    DOCUMENT = "document"


class EvaluationCase(BaseModel):
    """One labeled question without competition-specific field names."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    target_granularity: EvaluationTargetGranularity
    relevance_grades: dict[str, int]
    reference_answer: str | None = None
    expected_citation_chunk_ids: list[str] = Field(default_factory=list)
    should_abstain: bool | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("case_id", "question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("reference_answer", mode="before")
    @classmethod
    def normalize_reference(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("relevance_grades")
    @classmethod
    def validate_relevance(cls, values: dict[str, int]) -> dict[str, int]:
        if not values:
            raise ValueError("relevance_grades must not be empty")
        normalized = {_non_empty(key): value for key, value in values.items()}
        if any(value <= 0 for value in normalized.values()):
            raise ValueError("relevance grades must be positive")
        return normalized

    @field_validator("expected_citation_chunk_ids", "tags")
    @classmethod
    def validate_unique_text(cls, values: list[str]) -> list[str]:
        normalized = [_non_empty(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("values must be unique")
        return normalized


class RetrievalCaseMetrics(BaseModel):
    """Transparent metrics for one ranked retrieval response."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    reciprocal_rank: float = Field(ge=0, le=1)
    first_relevant_rank: int | None = Field(default=None, ge=1)


class GenerationCaseMetrics(BaseModel):
    """Automatic generation metrics computed only when labels exist."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    exact_match: float | None = Field(default=None, ge=0, le=1)
    abstention_accuracy: float | None = Field(default=None, ge=0, le=1)
    citation_precision: float | None = Field(default=None, ge=0, le=1)
    citation_recall: float | None = Field(default=None, ge=0, le=1)


class EvaluationCaseResult(BaseModel):
    """Compact inspectable outcome for one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    success: bool
    retrieved_ids: list[str] = Field(default_factory=list)
    missing_relevant_ids: list[str] = Field(default_factory=list)
    retrieval_metrics: RetrievalCaseMetrics | None = None
    generation_metrics: GenerationCaseMetrics | None = None
    answer_response: AnswerResponse | None = None
    retrieval_latency_ms: float | None = Field(default=None, ge=0)
    generation_latency_ms: float | None = Field(default=None, ge=0)
    error_stage: str | None = None
    error_type: str | None = None

    @model_validator(mode="after")
    def validate_success(self) -> "EvaluationCaseResult":
        if self.success and (
            self.retrieval_metrics is None
            or self.error_stage is not None
            or self.error_type is not None
        ):
            raise ValueError("successful result requires retrieval metrics")
        if not self.success and (
            self.error_stage is None or self.error_type is None
        ):
            raise ValueError("failed result requires a sanitized error")
        return self


class LatencySummary(BaseModel):
    """Distribution summary for successful measured operations."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    count: int = Field(ge=0)
    mean_ms: float | None = Field(default=None, ge=0)
    p50_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    max_ms: float | None = Field(default=None, ge=0)


class EvaluationResourceUsage(BaseModel):
    """Portable process-level resource observations."""

    model_config = ConfigDict(extra="forbid")

    wall_time_ms: float = Field(ge=0)
    process_cpu_time_ms: float = Field(ge=0)
    python_peak_traced_memory_bytes: int = Field(ge=0)


class EvaluationSummary(BaseModel):
    """Versioned aggregate report without inventing unavailable metrics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    created_at: datetime
    code_version: str
    benchmark_name: str
    benchmark_sha256: str
    strategy: RetrievalStrategy
    cutoffs: list[int]
    case_count: int = Field(ge=0)
    successful_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    generation_metrics: dict[str, float] = Field(default_factory=dict)
    metric_case_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_latency: LatencySummary
    generation_latency: LatencySummary
    resources: EvaluationResourceUsage
    artifact_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class EvaluationRunResult(BaseModel):
    """Summary plus per-case results consumed by report persistence."""

    model_config = ConfigDict(extra="forbid")

    summary: EvaluationSummary
    cases: list[EvaluationCaseResult]

    @model_validator(mode="after")
    def validate_counts(self) -> "EvaluationRunResult":
        if self.summary.case_count != len(self.cases):
            raise ValueError("summary case count must match case results")
        return self
