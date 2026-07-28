"""Unit tests for bounded strategy routing and conservative query rewriting."""

from legal_agentic_rag.agent import (
    ConservativeQueryRewriter,
    DeterministicStrategyRouter,
)
from legal_agentic_rag.configuration import AgentConfig
from legal_agentic_rag.schemas import (
    QueryAnalysis,
    QueryIntent,
    QueryVariant,
    QueryVariantKind,
    RetrievalQuery,
    RetrievalStrategy,
    ToolName,
)


def _query(
    requested_strategy: RetrievalStrategy | None = None,
) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="agent-route",
        original_question="Doanh nghiệp phải nộp thuế khi nào?",
        normalized_question="doanh nghiệp nộp thuế",
        requested_strategy=requested_strategy,
    )


def test_router_uses_requested_strategy_then_registered_fallbacks() -> None:
    """An explicit route is first and unregistered tools are never selected."""
    routes = DeterministicStrategyRouter().plan(
        _query(RetrievalStrategy.BM25),
        {
            ToolName.BM25_SEARCH,
            ToolName.RERANK_SEARCH,
            ToolName.HYBRID_SEARCH,
        },
    )

    assert [route.strategy for route in routes] == [
        RetrievalStrategy.BM25,
        RetrievalStrategy.HYBRID_RERANK,
        RetrievalStrategy.HYBRID,
    ]
    assert all(
        route.tool_name
        in {
            ToolName.BM25_SEARCH,
            ToolName.RERANK_SEARCH,
            ToolName.HYBRID_SEARCH,
        }
        for route in routes
    )


def test_router_respects_attempt_limit_and_configured_order() -> None:
    """Zero retries yields exactly one deterministic registered strategy."""
    config = AgentConfig(
        max_retry=0,
        strategy_order=[RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID],
    )
    routes = DeterministicStrategyRouter(config).plan(
        _query(),
        {ToolName.GRAPH_SEARCH, ToolName.HYBRID_SEARCH},
    )

    assert len(routes) == 1
    assert routes[0].strategy == RetrievalStrategy.GRAPH


def test_router_prioritizes_graph_for_relationship_queries() -> None:
    """An explicit amendment/effect query starts with bounded graph retrieval."""
    query = _query().model_copy(
        update={
            "query_analysis": QueryAnalysis(
                intent=QueryIntent.RELATIONSHIP,
                relationship_cues=["sửa đổi"],
            )
        }
    )

    routes = DeterministicStrategyRouter().plan(
        query,
        {
            ToolName.GRAPH_SEARCH,
            ToolName.RERANK_SEARCH,
            ToolName.HYBRID_SEARCH,
        },
    )

    assert [route.strategy for route in routes] == [
        RetrievalStrategy.GRAPH,
        RetrievalStrategy.HYBRID_RERANK,
        RetrievalStrategy.HYBRID,
    ]


def test_rewriter_uses_only_an_unused_user_supplied_query_form() -> None:
    """The baseline rewriter adds no inferred legal terms."""
    query = _query()
    rewriter = ConservativeQueryRewriter()

    rewritten = rewriter.rewrite(
        query,
        current_query=query.normalized_question,
        previously_used={query.normalized_question},
    )
    unchanged = rewriter.rewrite(
        query,
        current_query=query.original_question,
        previously_used={
            query.normalized_question,
            query.original_question,
        },
    )

    assert rewritten == query.original_question
    assert unchanged is None


def test_rewriter_uses_planned_user_derived_variant_before_original_form() -> None:
    """Retry uses a bounded analyzer variant without inventing legal language."""
    query = _query().model_copy(
        update={
            "query_variants": [
                QueryVariant(
                    variant_id="qv1",
                    text="doanh nghiệp nộp thuế",
                    kind=QueryVariantKind.NORMALIZED,
                ),
                QueryVariant(
                    variant_id="qv2",
                    text="nộp thuế",
                    kind=QueryVariantKind.FRAMING_STRIPPED,
                ),
            ]
        }
    )

    rewritten = ConservativeQueryRewriter().rewrite(
        query,
        current_query=query.normalized_question,
        previously_used={query.normalized_question},
    )

    assert rewritten == "nộp thuế"
