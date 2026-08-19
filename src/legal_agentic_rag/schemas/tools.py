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


class StructuredGenerationFailureCode(StrEnum):
    """Closed, content-free classifications for rejected model output."""

    JSON_DECODE_ERROR = "json_decode_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    MARKER_IN_CLAIM_TEXT = "marker_in_claim_text"
    CLAIM_BOUNDARY_MISMATCH = "claim_boundary_mismatch"
    NON_VIETNAMESE_CLAIM = "non_vietnamese_claim"
    MODEL_OUTPUT_VALIDATION = "model_output_validation"


class StructuredGenerationSchemaIssueCode(StrEnum):
    """Closed, content-free reasons a model JSON draft missed its schema."""

    TOP_LEVEL_EXTRA_FIELDS = "top_level_extra_fields"
    CLAIM_EXTRA_FIELDS = "claim_extra_fields"
    CLAIMS_OBJECT_INSTEAD_OF_LIST = "claims_object_instead_of_list"
    CLAIM_EVIDENCE_ID_SCALAR = "claim_evidence_id_scalar"
    DUPLICATE_CLAIM_EVIDENCE_IDS = "duplicate_claim_evidence_ids"
    DUPLICATE_WARNINGS = "duplicate_warnings"
    CLAIM_LIMIT_EXCEEDED = "claim_limit_exceeded"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_TOP_LEVEL_FIELD = "missing_top_level_field"
    MISSING_CLAIM_FIELD = "missing_claim_field"
    INVALID_TOP_LEVEL_TYPE = "invalid_top_level_type"
    INVALID_CLAIM_TYPE = "invalid_claim_type"
    INVALID_CLAIM_TEXT = "invalid_claim_text"
    INVALID_CLAIM_EVIDENCE_IDS = "invalid_claim_evidence_ids"
    INVALID_WARNINGS = "invalid_warnings"
    GROUNDING_STATE_MISMATCH = "grounding_state_mismatch"
    OTHER_SCHEMA_VALIDATION_ERROR = "other_schema_validation_error"


class StructuredGenerationSchemaRepairCode(StrEnum):
    """Closed deterministic edits that preserve accepted claim text exactly."""

    REMOVED_TOP_LEVEL_EXTRA_FIELDS = "removed_top_level_extra_fields"
    REMOVED_CLAIM_EXTRA_FIELDS = "removed_claim_extra_fields"
    WRAPPED_SINGLE_CLAIM = "wrapped_single_claim"
    WRAPPED_SCALAR_EVIDENCE_ID = "wrapped_scalar_evidence_id"
    DEDUPLICATED_EVIDENCE_IDS = "deduplicated_evidence_ids"
    DEDUPLICATED_WARNINGS = "deduplicated_warnings"
    DROPPED_EXCESS_CLAIMS = "dropped_excess_claims"


class StructuredGenerationSchemaRecoveryOutcome(StrEnum):
    """Terminal result of one bounded local schema-recovery evaluation."""

    SUCCEEDED = "succeeded"
    NOT_RECOVERABLE = "not_recoverable"
    REVALIDATION_FAILED = "revalidation_failed"


class StructuredGenerationMissingFieldCorrectionOutcome(StrEnum):
    """Terminal outcome of one bounded missing-required-field correction attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AnswerGenerationCorrectionSignal(StrEnum):
    """Closed, content-free request for one generation correction mode."""

    NUMERIC_MISMATCH = "numeric_mismatch"


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
    correction_signal: AnswerGenerationCorrectionSignal | None = None

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
    generation_failure_code: StructuredGenerationFailureCode | None = None
    generation_schema_issue_codes: list[StructuredGenerationSchemaIssueCode] = (
        Field(default_factory=list)
    )
    generation_schema_repair_codes: list[StructuredGenerationSchemaRepairCode] = (
        Field(default_factory=list)
    )
    generation_schema_recovery_outcome: (
        StructuredGenerationSchemaRecoveryOutcome | None
    ) = None
    generation_missing_field_correction_attempted: bool = False
    generation_missing_field_correction_outcome: (
        StructuredGenerationMissingFieldCorrectionOutcome | None
    ) = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Reject empty error messages."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool error message must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_generation_failure_code(self) -> "ToolError":
        """Expose only content-free schema detail on the matching model failure."""
        if (
            self.generation_failure_code is not None
            and self.error_type is not ToolErrorType.MODEL_ERROR
        ):
            raise ValueError(
                "generation failure code is valid only for model errors"
            )
        schema_details_present = bool(
            self.generation_schema_issue_codes
            or self.generation_schema_repair_codes
            or self.generation_schema_recovery_outcome is not None
        )
        correction_details_present = bool(
            self.generation_missing_field_correction_attempted
            or self.generation_missing_field_correction_outcome is not None
        )
        details_present = bool(
            schema_details_present
            or correction_details_present
        )
        if (
            details_present
            and self.error_type is not ToolErrorType.MODEL_ERROR
        ):
            raise ValueError(
                "generation schema detail is valid only for model errors"
            )
        if (
            schema_details_present
            and self.generation_failure_code
            is not StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR
            and not self.generation_missing_field_correction_attempted
        ):
            raise ValueError(
                "generation schema detail requires a schema model error "
                "or an attempted missing-field correction chain"
            )
        if (
            self.generation_missing_field_correction_outcome is not None
            and not self.generation_missing_field_correction_attempted
        ):
            raise ValueError(
                "missing-field correction outcome requires an attempted correction"
            )
        for values in (
            self.generation_schema_issue_codes,
            self.generation_schema_repair_codes,
        ):
            if len(values) != len(set(values)):
                raise ValueError("generation schema detail must be unique")
        return self


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
