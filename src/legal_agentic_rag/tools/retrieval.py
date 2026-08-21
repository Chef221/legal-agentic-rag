"""Typed wrappers over fixed, backend-neutral retrieval strategies."""

from __future__ import annotations

from typing import Protocol

from legal_agentic_rag.exceptions import (
    ConfigurationError,
    InvalidUserInputError,
    RetrievalError,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.schemas.tools import ToolName


class _Retriever(Protocol):
    def search(self, query: RetrievalQuery) -> RetrievalResponse: ...


_STRATEGIES: dict[ToolName, RetrievalStrategy] = {
    ToolName.BM25_SEARCH: RetrievalStrategy.BM25,
    ToolName.DENSE_SEARCH: RetrievalStrategy.DENSE,
    ToolName.HYBRID_SEARCH: RetrievalStrategy.HYBRID,
    ToolName.RERANK_SEARCH: RetrievalStrategy.HYBRID_RERANK,
}

_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.BM25_SEARCH: (
        "Search legal chunks using lexical BM25 retrieval only."
    ),
    ToolName.DENSE_SEARCH: (
        "Search legal chunks using dense semantic retrieval only."
    ),
    ToolName.HYBRID_SEARCH: (
        "Fuse BM25 and dense legal retrieval with reciprocal rank fusion."
    ),
    ToolName.RERANK_SEARCH: (
        "Retrieve hybrid candidates and apply the configured cross-encoder."
    ),
}


class RetrievalTool:
    """Bind exactly one public retrieval strategy to the fixed retriever."""

    def __init__(
        self,
        name: ToolName,
        retriever: _Retriever,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if name not in _STRATEGIES:
            raise ConfigurationError("RetrievalTool requires a retrieval tool name")
        if timeout_seconds <= 0:
            raise ConfigurationError("Tool timeout must be positive")
        self._name = name
        self._strategy = _STRATEGIES[name]
        self._retriever = retriever
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> ToolName:
        """Return the registered retrieval capability."""
        return self._name

    @property
    def description(self) -> str:
        """Describe the fixed retrieval strategy and its scope."""
        return _DESCRIPTIONS[self._name]

    @property
    def input_model(self) -> type[RetrievalQuery]:
        """Return the unified retrieval query schema."""
        return RetrievalQuery

    @property
    def output_model(self) -> type[RetrievalResponse]:
        """Return the unified retrieval response schema."""
        return RetrievalResponse

    @property
    def timeout_seconds(self) -> float:
        """Return the configured retrieval invocation budget."""
        return self._timeout_seconds

    def invoke(self, payload: RetrievalQuery) -> RetrievalResponse:
        """Run only this tool's fixed strategy and validate the response."""
        if payload.requested_strategy not in (None, self._strategy):
            raise InvalidUserInputError(
                "Retrieval query requests a strategy outside this tool"
            )
        routed = payload.model_copy(
            update={"requested_strategy": self._strategy}
        )
        response = self._retriever.search(routed)
        if (
            response.strategy != self._strategy
            or response.query.query_id != payload.query_id
            or response.query.requested_strategy != self._strategy
        ):
            raise RetrievalError(
                "Retriever returned an incompatible tool response"
            )
        return response


def fixed_retrieval_tools(
    retriever: _Retriever,
    *,
    timeout_seconds: float = 30.0,
) -> list[RetrievalTool]:
    """Create approved fixed retrieval tools."""
    return [
        RetrievalTool(name, retriever, timeout_seconds=timeout_seconds)
        for name in _STRATEGIES
    ]
