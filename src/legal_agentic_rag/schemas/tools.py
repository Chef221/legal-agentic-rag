"""Typed tool inputs, descriptors, invocation results, and safe errors."""

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
    Evidence,
)
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy


class ToolName(StrEnum):
    """Closed set of capabilities exposed to the later Agent workflow."""

    BM25_SEARCH = "bm25_search"
    DENSE_SEARCH = "dense_search"
    HYBRID_SEARCH = "hybrid_search"
    RERANK_SEARCH = "rerank_search"
    GRAPH_SEARCH = "graph_search"
    CONTEXT_GRADING = "context_grading"
    ANSWER_GENERATION = "answer_generation"
    CITATION_VERIFICATION = "citation_verification"


class ToolErrorType(StrEnum):
    """Safe error categories returned across the tool boundary."""

    INVALID_INPUT = "invalid_input"
    TOOL_NOT_REGISTERED = "tool_not_registered"
    TOOL_CONTRACT_ERROR = "tool_contract_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATASET_SCHEMA_ERROR = "dataset_schema_error"
    DATA_VALIDATION_ERROR = "data_validation_error"
    ARTIFACT_COMPATIBILITY_ERROR = "artifact_compatibility_error"
    BACKEND_INITIALIZATION_ERROR = "backend_initialization_error"
    RETRIEVAL_ERROR = "retrieval_error"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    EXTERNAL_SERVICE_ERROR = "external_service_error"


class ContextGradingInput(BaseModel):
    """Typed input for the context-grading tool."""

    model_config = ConfigDict(extra="forbid")

    query: RetrievalQuery
    evidence: list[Evidence] = Field(default_factory=list)


class AnswerGenerationInput(BaseModel):
    """Typed input for the grounded answer-generation tool."""

    model_config = ConfigDict(extra="forbid")

    query: RetrievalQuery
    evidence: list[Evidence] = Field(default_factory=list)
    retrieval_strategy: RetrievalStrategy
    trace_id: str

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        """Require a non-empty trace identity at the tool boundary."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("trace_id must not be empty")
        return normalized


class CitationVerificationInput(BaseModel):
    """Typed input for referential citation verification."""

    model_config = ConfigDict(extra="forbid")

    response: AnswerResponse
    evidence: list[Evidence] = Field(default_factory=list)


class ToolDescriptor(BaseModel):
    """Human- and machine-readable contract for one registered tool."""

    model_config = ConfigDict(extra="forbid")

    name: ToolName
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    timeout_seconds: float = Field(gt=0)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        """Require an actionable non-empty capability description."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool description must not be empty")
        return normalized


class ToolInvocationRequest(BaseModel):
    """Validated request for one explicitly named registered tool."""

    model_config = ConfigDict(extra="forbid")

    invocation_id: str
    tool_name: ToolName
    payload: dict[str, JsonValue]

    @field_validator("invocation_id")
    @classmethod
    def validate_invocation_id(cls, value: str) -> str:
        """Require traceable invocation identity without inspecting payload."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("invocation_id must not be empty")
        return normalized


class ToolError(BaseModel):
    """Sanitized failure safe to expose beyond the internal service boundary."""

    model_config = ConfigDict(extra="forbid")

    error_type: ToolErrorType
    message: str
    retryable: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Reject empty error messages."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool error message must not be empty")
        return normalized


class ToolInvocationResult(BaseModel):
    """Uniform success or failure envelope for registry execution."""

    model_config = ConfigDict(extra="forbid")

    invocation_id: str
    tool_name: ToolName
    success: bool
    output: dict[str, JsonValue] | None = None
    error: ToolError | None = None
    latency_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> "ToolInvocationResult":
        """Require exactly one output on success or one error on failure."""
        if self.success:
            if self.output is None or self.error is not None:
                raise ValueError("successful tool result requires output only")
        elif self.output is not None or self.error is None:
            raise ValueError("failed tool result requires error only")
        return self
