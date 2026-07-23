"""Tests for closed registration, schema discovery, and safe tool errors."""

from dataclasses import dataclass
from time import sleep

import pytest
from pydantic import BaseModel

from legal_agentic_rag.exceptions import ConfigurationError, RetrievalError
from legal_agentic_rag.schemas import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    ToolInvocationRequest,
    ToolName,
)
from legal_agentic_rag.schemas.tools import ToolErrorType
from legal_agentic_rag.tools import RetrievalTool, ToolRegistry


def _query_payload() -> dict[str, object]:
    return {
        "query_id": "registry-query",
        "original_question": "Câu hỏi",
        "normalized_question": "câu hỏi",
        "top_k": 1,
        "candidate_k": 2,
    }


def _request(
    tool_name: ToolName = ToolName.BM25_SEARCH,
    payload: dict[str, object] | None = None,
) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        invocation_id="invocation-1",
        tool_name=tool_name,
        payload=payload or _query_payload(),
    )


@dataclass
class _Retriever:
    delay_seconds: float = 0.0
    domain_error: bool = False
    unexpected_error: bool = False

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        if self.delay_seconds:
            sleep(self.delay_seconds)
        if self.domain_error:
            raise RetrievalError("sensitive local path must not escape")
        if self.unexpected_error:
            raise ValueError("programming bug")
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.BM25,
        )


class _WrongOutputTool:
    name = ToolName.BM25_SEARCH
    description = "Return the wrong output for contract testing."
    input_model = RetrievalQuery
    output_model = RetrievalResponse
    timeout_seconds = 30.0

    def invoke(self, payload: BaseModel) -> BaseModel:
        return payload


def test_registry_describes_executes_and_rejects_duplicate_registration() -> None:
    """Only explicitly registered tools are discoverable and callable."""
    tool = RetrievalTool(ToolName.BM25_SEARCH, _Retriever())
    registry = ToolRegistry([tool])

    descriptor = registry.descriptors()[0]
    result = registry.execute(_request())

    assert descriptor.name == ToolName.BM25_SEARCH
    assert descriptor.input_schema["title"] == "RetrievalQuery"
    assert descriptor.output_schema["title"] == "RetrievalResponse"
    assert result.success is True
    assert result.output["strategy"] == "bm25"
    assert result.error is None
    with pytest.raises(ConfigurationError, match="already"):
        registry.register(tool)


def test_registry_returns_sanitized_input_missing_and_domain_errors() -> None:
    """Known failures are normalized without exposing internal exception text."""
    registry = ToolRegistry(
        [
            RetrievalTool(
                ToolName.BM25_SEARCH,
                _Retriever(domain_error=True),
            )
        ]
    )

    invalid = registry.execute(_request(payload={"query_id": "only-id"}))
    missing = ToolRegistry().execute(_request())
    domain = registry.execute(_request())

    assert invalid.error.error_type == ToolErrorType.INVALID_INPUT
    assert missing.error.error_type == ToolErrorType.TOOL_NOT_REGISTERED
    assert domain.error.error_type == ToolErrorType.RETRIEVAL_ERROR
    assert domain.error.retryable is True
    assert "sensitive" not in domain.error.message
    assert all(result.output is None for result in (invalid, missing, domain))


def test_registry_rejects_wrong_output_and_classifies_elapsed_timeout() -> None:
    """Contract violations and exceeded time budgets never return tool output."""
    wrong = ToolRegistry([_WrongOutputTool()]).execute(_request())
    slow = ToolRegistry(
        [
            RetrievalTool(
                ToolName.BM25_SEARCH,
                _Retriever(delay_seconds=0.003),
                timeout_seconds=0.0001,
            )
        ]
    ).execute(_request())

    assert wrong.error.error_type == ToolErrorType.TOOL_CONTRACT_ERROR
    assert slow.error.error_type == ToolErrorType.TIMEOUT
    assert slow.error.retryable is True


def test_registry_does_not_hide_unexpected_programming_errors() -> None:
    """Unknown exceptions remain visible to operators instead of becoming fake data."""
    registry = ToolRegistry(
        [
            RetrievalTool(
                ToolName.BM25_SEARCH,
                _Retriever(unexpected_error=True),
            )
        ]
    )

    with pytest.raises(ValueError, match="programming bug"):
        registry.execute(_request())
