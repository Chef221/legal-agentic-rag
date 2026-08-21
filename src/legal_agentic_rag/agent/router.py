"""Deterministic routing over explicitly registered retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass

from legal_agentic_rag.configuration.online import (
    AgentConfig,
    QueryUnderstandingConfig,
)
from legal_agentic_rag.exceptions import InvalidUserInputError
from legal_agentic_rag.schemas.retrieval import (
    QueryIntent,
    RetrievalQuery,
    RetrievalStrategy,
)
from legal_agentic_rag.schemas.tools import ToolName

_STRATEGY_TO_TOOL: dict[RetrievalStrategy, ToolName] = {
    RetrievalStrategy.BM25: ToolName.BM25_SEARCH,
    RetrievalStrategy.DENSE: ToolName.DENSE_SEARCH,
    RetrievalStrategy.HYBRID: ToolName.HYBRID_SEARCH,
    RetrievalStrategy.HYBRID_RERANK: ToolName.RERANK_SEARCH,
    RetrievalStrategy.GRAPH: ToolName.GRAPH_SEARCH,
}


@dataclass(frozen=True, slots=True)
class RetrievalRoute:
    """One approved retrieval strategy and its registered tool name."""

    strategy: RetrievalStrategy
    tool_name: ToolName


class DeterministicStrategyRouter:
    """Build a bounded quality-first route plan without model inference."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        query_config: QueryUnderstandingConfig | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._query_config = query_config or QueryUnderstandingConfig()

    def plan(
        self,
        query: RetrievalQuery,
        registered_tools: set[ToolName],
    ) -> list[RetrievalRoute]:
        """Return unique registered routes, respecting an explicit first strategy."""
        requested = query.requested_strategy
        if requested is not None and requested not in _STRATEGY_TO_TOOL:
            raise InvalidUserInputError(
                "The requested strategy is not available to the Agent"
            )
        strategies = [
            *([requested] if requested is not None else []),
            *self._strategy_order(query),
        ]
        routes: list[RetrievalRoute] = []
        seen: set[RetrievalStrategy] = set()
        for strategy in strategies:
            if strategy in seen:
                continue
            seen.add(strategy)
            tool_name = _STRATEGY_TO_TOOL[strategy]
            if tool_name in registered_tools:
                routes.append(RetrievalRoute(strategy, tool_name))
            if len(routes) >= self._config.max_retry + 1:
                break
        return routes

    def _strategy_order(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalStrategy]:
        analysis = query.query_analysis
        if (
            not self._query_config.adaptive_routing_enabled
            or analysis is None
        ):
            return self._config.strategy_order
        if analysis.intent == QueryIntent.RELATIONSHIP:
            adaptive = [
                RetrievalStrategy.HYBRID_RERANK,
                RetrievalStrategy.HYBRID,
            ]
        elif (
            analysis.has_explicit_legal_reference
            or analysis.intent == QueryIntent.QUANTITATIVE
        ):
            adaptive = [
                RetrievalStrategy.HYBRID_RERANK,
                RetrievalStrategy.BM25,
                RetrievalStrategy.HYBRID,
            ]
        else:
            adaptive = []
        return [*adaptive, *self._config.strategy_order]
