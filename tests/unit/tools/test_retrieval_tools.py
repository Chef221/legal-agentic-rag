"""Tests for fixed-strategy typed retrieval wrappers."""

from dataclasses import dataclass, field

import pytest

from legal_agentic_rag.exceptions import InvalidUserInputError, RetrievalError
from legal_agentic_rag.schemas import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    ToolName,
)
from legal_agentic_rag.tools import RetrievalTool, TypedTool, fixed_retrieval_tools


@dataclass
class _Retriever:
    calls: list[RetrievalQuery] = field(default_factory=list)
    wrong_strategy: bool = False

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.calls.append(query)
        strategy = (
            RetrievalStrategy.DENSE
            if self.wrong_strategy
            else query.requested_strategy
        )
        assert strategy is not None
        return RetrievalResponse(
            query=query,
            strategy=strategy,
        )


def _query(
    strategy: RetrievalStrategy | None = None,
) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="tool-query",
        original_question="Câu hỏi",
        normalized_question="câu hỏi",
        top_k=1,
        candidate_k=2,
        requested_strategy=strategy,
    )


def test_fixed_retrieval_tools_route_exactly_five_public_strategies() -> None:
    """Each wrapper overrides no strategy except its own fixed capability."""
    retriever = _Retriever()
    tools = fixed_retrieval_tools(retriever)

    responses = [tool.invoke(_query()) for tool in tools]

    assert [tool.name for tool in tools] == [
        ToolName.BM25_SEARCH,
        ToolName.DENSE_SEARCH,
        ToolName.HYBRID_SEARCH,
        ToolName.RERANK_SEARCH,
        ToolName.RELATIONSHIP_RERANK_SEARCH,
    ]
    assert [response.strategy for response in responses] == [
        RetrievalStrategy.BM25,
        RetrievalStrategy.DENSE,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.HYBRID_RERANK,
        RetrievalStrategy.HYBRID_RERANK,
    ]
    assert all(isinstance(tool, TypedTool) for tool in tools)
    assert all(tool.description for tool in tools)


def test_retrieval_tool_rejects_strategy_escape_and_bad_response() -> None:
    """A caller cannot route one tool into another strategy or accept bad output."""
    tool = RetrievalTool(ToolName.BM25_SEARCH, _Retriever())
    with pytest.raises(InvalidUserInputError, match="outside"):
        tool.invoke(_query(RetrievalStrategy.DENSE))

    bad = RetrievalTool(
        ToolName.BM25_SEARCH,
        _Retriever(wrong_strategy=True),
    )
    with pytest.raises(RetrievalError, match="incompatible"):
        bad.invoke(_query())
