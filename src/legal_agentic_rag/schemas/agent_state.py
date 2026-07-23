"""Typed state and result contracts for the bounded Agent workflow."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    ContextGrade,
    Evidence,
)
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
    """Serializable state of one bounded Agent workflow run."""

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


class AgentStopReason(StrEnum):
    """Explicit terminal reasons for a bounded Agent run."""

    ANSWER_VERIFIED = "answer_verified"
    MAX_RETRY_REACHED = "max_retry_reached"
    NO_NEW_STRATEGY = "no_new_strategy"
    NON_RETRYABLE_TOOL_ERROR = "non_retryable_tool_error"
    TIMEOUT = "timeout"
    GENERATION_FAILED = "generation_failed"
    CITATION_VERIFICATION_FAILED = "citation_verification_failed"


class AgentRunResult(BaseModel):
    """Final answer and inspectable state returned by an Agent workflow."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    state: AgentState
    response: AnswerResponse
    stop_reason: AgentStopReason
    total_latency_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_state_response_alignment(self) -> "AgentRunResult":
        """Keep the public answer aligned with the serialized terminal state."""
        if self.response.trace_id != self.state.trace_id:
            raise ValueError("response trace_id must match Agent state")
        if self.response.answer != self.state.answer:
            raise ValueError("response answer must match Agent state")
        if self.response.citations != self.state.citations:
            raise ValueError("response citations must match Agent state")
        if (
            self.state.selected_strategy is not None
            and self.response.retrieval_strategy != self.state.selected_strategy
        ):
            raise ValueError("response strategy must match terminal Agent state")
        if self.state.retry_count != max(0, len(self.state.retrieval_history) - 1):
            raise ValueError("retry_count must equal completed attempts minus one")
        if (
            self.stop_reason == AgentStopReason.ANSWER_VERIFIED
        ) == self.response.insufficient_evidence:
            raise ValueError(
                "only answer_verified may return a non-abstaining response"
            )
        return self
