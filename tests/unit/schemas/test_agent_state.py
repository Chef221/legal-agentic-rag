"""Tests for bounded Agent state and terminal result contracts."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.agent_state import (
    AgentRunResult,
    AgentState,
    AgentStopReason,
    RetrievalHistoryItem,
)
from legal_agentic_rag.schemas.answering import AnswerResponse
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy


def _query(identifier: str) -> RetrievalQuery:
    return RetrievalQuery(
        query_id=identifier,
        original_question="Câu hỏi",
        normalized_question="Câu hỏi",
    )


def test_agent_state_enforces_retry_limit() -> None:
    """The accepted baseline permits no more than two retries."""
    with pytest.raises(ValidationError):
        AgentState(
            trace_id="trace-1",
            original_question="Câu hỏi",
            normalized_question="Câu hỏi",
            current_query="Câu hỏi",
            retry_count=3,
        )


def test_retrieval_history_is_ordered_and_bounded() -> None:
    """History records ordered attempts within the bounded Agent contract."""
    attempt_one = RetrievalHistoryItem(
        attempt_number=1,
        query=_query("query-1"),
        strategy=RetrievalStrategy.BM25,
    )
    attempt_two = RetrievalHistoryItem(
        attempt_number=2,
        query=_query("query-2"),
        strategy=RetrievalStrategy.DENSE,
    )
    state = AgentState(
        trace_id="trace-1",
        original_question="Câu hỏi",
        normalized_question="Câu hỏi",
        current_query="Câu hỏi viết lại",
        retrieval_history=[attempt_one, attempt_two],
        retry_count=1,
    )
    assert len(state.retrieval_history) == 2

    with pytest.raises(ValidationError):
        AgentState(
            trace_id="trace-1",
            original_question="Câu hỏi",
            normalized_question="Câu hỏi",
            current_query="Câu hỏi",
            retrieval_history=[attempt_two, attempt_one],
        )


def test_agent_run_result_requires_response_and_state_alignment() -> None:
    """Terminal answer identity cannot drift from the serialized Agent state."""
    state = AgentState(
        trace_id="trace-agent",
        original_question="Câu hỏi",
        normalized_question="câu hỏi",
        current_query="câu hỏi",
        answer="Chưa đủ căn cứ.",
    )
    response = AnswerResponse(
        question="Câu hỏi",
        answer="Chưa đủ căn cứ.",
        insufficient_evidence=True,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        trace_id="trace-agent",
    )

    result = AgentRunResult(
        state=state,
        response=response,
        stop_reason=AgentStopReason.NO_NEW_STRATEGY,
    )

    assert result.response.answer == result.state.answer


def test_agent_state_normalizes_whitespace_to_baseline() -> None:
    """AgentState strips surrounding whitespace and treats blank answers as None."""
    state_padded = AgentState(
        trace_id="trace-ws",
        original_question="Câu hỏi",
        normalized_question="câu hỏi",
        current_query="câu hỏi",
        answer="  legacy answer  ",
    )
    assert state_padded.answer == "legacy answer"

    state_blank = AgentState(
        trace_id="trace-ws-blank",
        original_question="Câu hỏi",
        normalized_question="câu hỏi",
        current_query="câu hỏi",
        answer="   \n\t  ",
    )
    assert state_blank.answer is None
