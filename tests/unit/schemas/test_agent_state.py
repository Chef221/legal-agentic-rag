"""Tests for bounded Agent state contracts without implementing an Agent."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.agent_state import AgentState, RetrievalHistoryItem
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
    """History records ordered attempts without adding Agent behavior."""
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
