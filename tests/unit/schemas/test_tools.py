"""Schema invariants for typed tool invocation boundaries."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas import (
    AnswerGenerationCorrectionSignal,
    AnswerGenerationInput,
    ToolError,
    ToolErrorType,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolName,
    StructuredGenerationFailureCode,
    StructuredGenerationSchemaIssueCode,
    StructuredGenerationSchemaRecoveryOutcome,
    StructuredGenerationSchemaRepairCode,
)
from legal_agentic_rag.schemas import Evidence, RetrievalQuery, RetrievalStrategy


def test_tool_invocation_result_requires_exactly_output_or_error() -> None:
    """Success and failure envelopes cannot be ambiguous."""
    success = ToolInvocationResult(
        invocation_id="invoke-1",
        tool_name=ToolName.BM25_SEARCH,
        success=True,
        output={},
    )
    failure = ToolInvocationResult(
        invocation_id="invoke-2",
        tool_name=ToolName.BM25_SEARCH,
        success=False,
        error=ToolError(
            error_type=ToolErrorType.INVALID_INPUT,
            message="Invalid input.",
        ),
    )

    assert success.error is None
    assert failure.output is None
    with pytest.raises(ValidationError):
        ToolInvocationResult(
            invocation_id="invoke-3",
            tool_name=ToolName.BM25_SEARCH,
            success=True,
        )


def test_tool_request_rejects_unknown_fields_and_empty_identity() -> None:
    """The registry boundary stays closed and traceable."""
    with pytest.raises(ValidationError):
        ToolInvocationRequest(
            invocation_id=" ",
            tool_name=ToolName.BM25_SEARCH,
            payload={},
        )


def test_answer_generation_input_allows_only_the_closed_numeric_repair_signal() -> None:
    """The repair request is typed and cannot carry an arbitrary instruction."""
    query = RetrievalQuery(
        query_id="typed-repair",
        original_question="Quy định thế nào?",
        normalized_question="quy định",
        top_k=1,
        candidate_k=1,
    )
    payload = AnswerGenerationInput(
        query=query,
        evidence=[],
        retrieval_strategy=RetrievalStrategy.HYBRID,
        trace_id=query.query_id,
        correction_signal=AnswerGenerationCorrectionSignal.NUMERIC_MISMATCH,
    )

    assert payload.correction_signal == AnswerGenerationCorrectionSignal.NUMERIC_MISMATCH
    with pytest.raises(ValidationError):
        AnswerGenerationInput.model_validate(
            {**payload.model_dump(mode="json"), "correction_signal": "rewrite_anything"}
        )


def test_structured_generation_error_code_is_limited_to_model_failures() -> None:
    """Telemetry cannot attach model-output taxonomy to a non-model error."""
    error = ToolError(
        error_type=ToolErrorType.MODEL_ERROR,
        message="Model output rejected.",
        generation_failure_code=StructuredGenerationFailureCode.JSON_DECODE_ERROR,
    )

    assert error.generation_failure_code == StructuredGenerationFailureCode.JSON_DECODE_ERROR
    with pytest.raises(ValidationError):
        ToolError(
            error_type=ToolErrorType.DATA_VALIDATION_ERROR,
            message="Invalid data.",
            generation_failure_code=StructuredGenerationFailureCode.JSON_DECODE_ERROR,
        )
    with pytest.raises(ValidationError):
        ToolInvocationRequest.model_validate(
            {
                "invocation_id": "invoke",
                "tool_name": "bm25_search",
                "payload": {},
                "raw_database_client": "forbidden",
            }
        )


def test_schema_recovery_detail_requires_a_schema_model_error() -> None:
    """Only terminal schema failures may carry content-free repair telemetry."""
    error = ToolError(
        error_type=ToolErrorType.MODEL_ERROR,
        message="Model output rejected.",
        generation_failure_code=StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR,
        generation_schema_issue_codes=[
            StructuredGenerationSchemaIssueCode.TOP_LEVEL_EXTRA_FIELDS
        ],
        generation_schema_repair_codes=[
            StructuredGenerationSchemaRepairCode.REMOVED_TOP_LEVEL_EXTRA_FIELDS
        ],
        generation_schema_recovery_outcome=(
            StructuredGenerationSchemaRecoveryOutcome.SUCCEEDED
        ),
    )

    assert error.generation_schema_issue_codes == [
        StructuredGenerationSchemaIssueCode.TOP_LEVEL_EXTRA_FIELDS
    ]
    with pytest.raises(ValidationError):
        ToolError(
            error_type=ToolErrorType.MODEL_ERROR,
            message="Model output rejected.",
            generation_failure_code=StructuredGenerationFailureCode.JSON_DECODE_ERROR,
            generation_schema_issue_codes=[
                StructuredGenerationSchemaIssueCode.TOP_LEVEL_EXTRA_FIELDS
            ],
        )


def test_missing_field_correction_chain_allows_terminal_non_schema_error() -> None:
    """TEST 1: Earlier schema diagnostics preserved during an attempted correction chain are accepted."""
    from legal_agentic_rag.schemas import (
        StructuredGenerationMissingFieldCorrectionOutcome,
    )

    error = ToolError(
        error_type=ToolErrorType.MODEL_ERROR,
        message="Model output rejected.",
        generation_failure_code=StructuredGenerationFailureCode.JSON_DECODE_ERROR,
        generation_schema_issue_codes=[
            StructuredGenerationSchemaIssueCode.MISSING_REQUIRED_FIELD
        ],
        generation_missing_field_correction_attempted=True,
        generation_missing_field_correction_outcome=(
            StructuredGenerationMissingFieldCorrectionOutcome.FAILED
        ),
    )
    assert error.generation_failure_code == StructuredGenerationFailureCode.JSON_DECODE_ERROR
    assert error.generation_schema_issue_codes == [
        StructuredGenerationSchemaIssueCode.MISSING_REQUIRED_FIELD
    ]
    assert error.generation_missing_field_correction_attempted is True
    assert (
        error.generation_missing_field_correction_outcome
        == StructuredGenerationMissingFieldCorrectionOutcome.FAILED
    )


def test_free_floating_schema_diagnostics_without_attempted_correction_are_rejected() -> None:
    """TEST 2: Free-floating schema diagnostics on non-schema terminal failures remain rejected."""
    with pytest.raises(ValidationError, match="generation schema detail requires a schema model error"):
        ToolError(
            error_type=ToolErrorType.MODEL_ERROR,
            message="Model output rejected.",
            generation_failure_code=StructuredGenerationFailureCode.JSON_DECODE_ERROR,
            generation_schema_issue_codes=[
                StructuredGenerationSchemaIssueCode.MISSING_REQUIRED_FIELD
            ],
            generation_missing_field_correction_attempted=False,
        )


def test_generation_diagnostics_on_non_model_error_are_rejected() -> None:
    """TEST 3: Any generation schema or correction diagnostics on non-model error are rejected."""
    from legal_agentic_rag.schemas import (
        StructuredGenerationMissingFieldCorrectionOutcome,
    )

    with pytest.raises(ValidationError, match="generation schema detail is valid only for model errors"):
        ToolError(
            error_type=ToolErrorType.DATA_VALIDATION_ERROR,
            message="Invalid data.",
            generation_schema_issue_codes=[
                StructuredGenerationSchemaIssueCode.MISSING_REQUIRED_FIELD
            ],
        )

    with pytest.raises(ValidationError, match="generation schema detail is valid only for model errors"):
        ToolError(
            error_type=ToolErrorType.TIMEOUT,
            message="Operation timed out.",
            generation_missing_field_correction_attempted=True,
            generation_missing_field_correction_outcome=(
                StructuredGenerationMissingFieldCorrectionOutcome.FAILED
            ),
        )


def test_correction_outcome_without_attempted_correction_is_rejected() -> None:
    """TEST 4: A missing-field correction outcome requires generation_missing_field_correction_attempted=True."""
    from legal_agentic_rag.schemas import (
        StructuredGenerationMissingFieldCorrectionOutcome,
    )

    with pytest.raises(ValidationError, match="missing-field correction outcome requires an attempted correction"):
        ToolError(
            error_type=ToolErrorType.MODEL_ERROR,
            message="Model output rejected.",
            generation_failure_code=StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR,
            generation_missing_field_correction_attempted=False,
            generation_missing_field_correction_outcome=(
                StructuredGenerationMissingFieldCorrectionOutcome.FAILED
            ),
        )
