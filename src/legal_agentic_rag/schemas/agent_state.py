"""Typed state contracts reserved for the later bounded Agent workflow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from legal_agentic_rag.schemas.answering import Citation, ContextGrade, Evidence
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


class RetrievalHistoryItem(BaseModel):
    """One bounded retrieval attempt retained in Agent state."""

    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(ge=1, le=3)
    query: RetrievalQuery
    strategy: RetrievalStrategy
    response: RetrievalResponse | None = None
    context_grade: ContextGrade | None = None
    error_type: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("error_type", mode="before")
    @classmethod
    def normalize_error_type(cls, value: object) -> object:
        """Normalize an empty optional error category to null."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value


class AgentState(BaseModel):
    """Serializable state contract for a future bounded Agent workflow."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    original_question: str
    normalized_question: str
    current_query: str
    selected_strategy: RetrievalStrategy | None = None
    retrieval_history: list[RetrievalHistoryItem] = Field(default_factory=list)
    candidate_hits: list[RetrievalHit] = Field(default_factory=list)
    selected_evidence: list[Evidence] = Field(default_factory=list)
    context_grade: ContextGrade | None = None
    retry_count: int = Field(default=0, ge=0, le=2)
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "trace_id", "original_question", "normalized_question", "current_query"
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject Agent state without trace or query identity."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_answer(cls, value: object) -> object:
        """Normalize an absent generated answer consistently."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("retrieval_history")
    @classmethod
    def validate_retrieval_history(
        cls, values: list[RetrievalHistoryItem]
    ) -> list[RetrievalHistoryItem]:
        """Require ordered, non-duplicated attempt numbers."""
        attempts = [value.attempt_number for value in values]
        if attempts != sorted(attempts) or len(attempts) != len(set(attempts)):
            raise ValueError("retrieval history attempts must be unique and ordered")
        return values
