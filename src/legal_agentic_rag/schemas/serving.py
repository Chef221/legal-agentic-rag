"""Public request, health, and sanitized API error schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from legal_agentic_rag.schemas.manifests import ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    RetrievalFilters,
    RetrievalStrategy,
)


class LegalQuestionRequest(BaseModel):
    """Validated user question and optional backend-neutral retrieval controls."""

    model_config = ConfigDict(extra="forbid")

    question: str
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    top_k: int | None = Field(default=None, gt=0)
    candidate_k: int | None = Field(default=None, gt=0)
    requested_strategy: RetrievalStrategy | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject blank input while preserving Vietnamese legal text."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> "LegalQuestionRequest":
        """Keep explicit request limits internally consistent."""
        if (
            self.top_k is not None
            and self.candidate_k is not None
            and self.candidate_k < self.top_k
        ):
            raise ValueError("candidate_k must be at least top_k")
        if self.requested_strategy == RetrievalStrategy.RERANK:
            raise ValueError("rerank is an internal stage, not a public strategy")
        return self


class ServiceStatus(StrEnum):
    """Public readiness states for the serving process."""

    READY = "ready"


class ArtifactHealth(BaseModel):
    """Non-sensitive identity of one startup-validated artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: ArtifactType
    artifact_version: str
    record_count: int = Field(ge=0)
    backend: str | None = None
    model_name: str | None = None


class HealthResponse(BaseModel):
    """Readiness response for the loaded immutable online runtime."""

    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus
    service_version: str
    dataset_name: str
    dataset_revision: str | None = None
    artifacts: list[ArtifactHealth]
    tool_count: int = Field(ge=0)


class ApiErrorDetail(BaseModel):
    """Sanitized machine-readable serving failure."""

    model_config = ConfigDict(extra="forbid")

    error_type: str
    message: str
    trace_id: str | None = None


class ApiErrorResponse(BaseModel):
    """Uniform HTTP error envelope."""

    model_config = ConfigDict(extra="forbid")

    error: ApiErrorDetail
