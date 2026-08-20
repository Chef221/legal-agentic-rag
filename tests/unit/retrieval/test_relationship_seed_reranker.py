"""Unit tests for RelationshipSeedRerankingRetriever and B1B graphless wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from legal_agentic_rag.configuration.online import (
    AgentConfig,
    QueryUnderstandingConfig,
    RerankerConfig,
    RetrievalConfig,
)
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import (
    ConfigurationError,
    InvalidUserInputError,
    RetrievalError,
)
from legal_agentic_rag.retrieval.fixed import FixedRetriever
from legal_agentic_rag.retrieval.rerank import (
    RelationshipSeedRerankingRetriever,
    RerankingRetriever,
)
from legal_agentic_rag.agent.router import (
    DeterministicStrategyRouter,
    RetrievalRoute,
)
from legal_agentic_rag.runtime.build_validation import (
    COMPETITION_REQUIRED_ARTIFACT_TYPES,
    ArtifactSetValidator,
)
from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    ArtifactType,
    DatasetManifest,
)
from legal_agentic_rag.schemas.retrieval import (
    QueryAnalysis,
    QueryIntent,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.schemas.tools import ToolName
from legal_agentic_rag.tools.retrieval import (
    RetrievalTool,
    fixed_retrieval_tools,
)


def _make_query(
    query_id: str = "q1",
    question: str = "test question",
    top_k: int = 8,
    candidate_k: int = 40,
    requested_strategy: RetrievalStrategy | None = None,
    query_analysis: QueryAnalysis | None = None,
) -> RetrievalQuery:
    return RetrievalQuery(
        query_id=query_id,
        original_question=question,
        normalized_question=question,
        top_k=top_k,
        candidate_k=candidate_k,
        requested_strategy=requested_strategy,
        query_analysis=query_analysis,
    )


def _make_hit(chunk_id: str, doc_id: str = "doc1", score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=doc_id,
        text="Content text",
        score=score,
        strategy=RetrievalStrategy.HYBRID,
        rank=1,
    )


def _make_reranked_response(
    query: RetrievalQuery,
    candidates: list[RetrievalHit],
    warnings: list[str] | None = None,
) -> RetrievalResponse:
    rerank_query = query.model_copy(
        update={"requested_strategy": RetrievalStrategy.RERANK}
    )
    hits = [
        hit.model_copy(
            update={
                "rank": idx + 1,
                "strategy": RetrievalStrategy.RERANK,
                "retrieval_trace": hit.retrieval_trace.model_copy(
                    update={"reranker_score": hit.score}
                ),
            }
        )
        for idx, hit in enumerate(candidates[:query.top_k])
    ]
    return RetrievalResponse(
        query=rerank_query,
        strategy=RetrievalStrategy.RERANK,
        hits=hits,
        latency_ms=1.0,
        warnings=warnings or [],
    )


def _make_manifest(artifact_type: ArtifactType, code_version: str = "0.50.7") -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=artifact_type,
        artifact_version="1.0",
        dataset_name="uit_dsc_2026",
        dataset_revision="sha256:test",
        created_at=datetime.now(UTC),
        record_count=10,
        processing_config_hash="hash123",
        code_version=code_version,
        metadata={},
    )


class TestRelationshipSeedRerankingRetriever:
    """Test suite for RelationshipSeedRerankingRetriever and B1B graphless wiring."""

    def test_01_init_default_and_custom_configs(self) -> None:
        candidate_retriever = MagicMock()
        candidate_retriever.source_artifact_identity = ("bm25", "vector", "hybrid")
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"

        retriever_default = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        assert retriever_default._retrieval_config.relationship_rerank_fusion_k == 20
        assert retriever_default._reranker_config.max_candidates == 100

        custom_retrieval_cfg = RetrievalConfig(relationship_rerank_fusion_k=15)
        custom_reranker_cfg = RerankerConfig(max_candidates=30)
        retriever_custom = RelationshipSeedRerankingRetriever(
            candidate_retriever, reranker, custom_retrieval_cfg, custom_reranker_cfg
        )
        assert retriever_custom._retrieval_config.relationship_rerank_fusion_k == 15
        assert retriever_custom._reranker_config.max_candidates == 30

    def test_02_source_artifact_identity(self) -> None:
        candidate_retriever = MagicMock()
        candidate_retriever.source_artifact_identity = ("art1", "v1", "hash1")
        reranker = MagicMock(spec=Reranker)
        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        assert retriever.source_artifact_identity == ("art1", "v1", "hash1")

    def test_03_search_rejects_incompatible_strategy(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        query = _make_query(requested_strategy=RetrievalStrategy.BM25)
        with pytest.raises(RetrievalError, match="incompatible request"):
            retriever.search(query)

    def test_04_search_rejects_candidate_k_exceeding_max(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        retriever = RelationshipSeedRerankingRetriever(
            candidate_retriever, reranker, reranker_config=RerankerConfig(max_candidates=40)
        )
        query = _make_query(candidate_k=50)
        with pytest.raises(RetrievalError, match="exceeds the reranker limit"):
            retriever.search(query)

    def test_05_search_delegates_with_fusion_limit(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"
        hit = _make_hit("c1")
        cand_resp = RetrievalResponse(
            query=_make_query(top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.HYBRID),
            strategy=RetrievalStrategy.HYBRID,
            hits=[hit],
            latency_ms=10.0,
            artifact_versions={"chunk": "v1"},
        )
        candidate_retriever.search.return_value = cand_resp

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates))

        reranker.rerank.side_effect = _mock_rerank

        retriever = RelationshipSeedRerankingRetriever(
            candidate_retriever, reranker, RetrievalConfig(relationship_rerank_fusion_k=20)
        )
        query = _make_query(top_k=8, candidate_k=40)
        resp = retriever.search(query)

        assert candidate_retriever.search.call_count == 1
        called_cand_query = candidate_retriever.search.call_args[0][0]
        assert called_cand_query.top_k == 20
        assert called_cand_query.candidate_k == 40
        assert called_cand_query.requested_strategy == RetrievalStrategy.HYBRID
        assert resp.strategy == RetrievalStrategy.HYBRID_RERANK

    def test_06_maximum_seed_slots_fallback_when_candidate_k_is_1(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"
        hit = _make_hit("c1")
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(top_k=1, candidate_k=1, requested_strategy=RetrievalStrategy.HYBRID),
            strategy=RetrievalStrategy.HYBRID,
            hits=[hit],
            latency_ms=5.0,
            artifact_versions={"chunk": "v1"},
        )

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates))

        reranker.rerank.side_effect = _mock_rerank

        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        query = _make_query(top_k=1, candidate_k=1)
        resp = retriever.search(query)
        called_cand_query = candidate_retriever.search.call_args[0][0]
        assert called_cand_query.top_k == 1

    def test_07_candidate_k_40_fusion_20_yields_top_k_20(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"
        hit = _make_hit("c1")
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.HYBRID),
            strategy=RetrievalStrategy.HYBRID,
            hits=[hit],
            latency_ms=5.0,
            artifact_versions={"chunk": "v1"},
        )

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates))

        reranker.rerank.side_effect = _mock_rerank

        retriever = RelationshipSeedRerankingRetriever(
            candidate_retriever, reranker, RetrievalConfig(relationship_rerank_fusion_k=20)
        )
        retriever.search(_make_query(top_k=8, candidate_k=40))
        called_cand_query = candidate_retriever.search.call_args[0][0]
        assert called_cand_query.top_k == 20

    def test_08_candidate_query_preserves_candidate_k_and_requests_hybrid(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"
        hit = _make_hit("c1")
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.HYBRID),
            strategy=RetrievalStrategy.HYBRID,
            hits=[hit],
            latency_ms=5.0,
            artifact_versions={"chunk": "v1"},
        )

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates))

        reranker.rerank.side_effect = _mock_rerank

        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        retriever.search(_make_query(top_k=8, candidate_k=40))
        called_cand_query = candidate_retriever.search.call_args[0][0]
        assert called_cand_query.candidate_k == 40
        assert called_cand_query.requested_strategy == RetrievalStrategy.HYBRID

    def test_09_search_fails_if_candidate_response_strategy_not_hybrid(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        hit = _make_hit("c1")
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.BM25),
            strategy=RetrievalStrategy.BM25,
            hits=[hit],
            latency_ms=5.0,
            artifact_versions={"chunk": "v1"},
        )
        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        with pytest.raises(RetrievalError, match="incompatible response"):
            retriever.search(_make_query(top_k=8, candidate_k=40))

    def test_10_search_fails_if_candidate_response_query_id_mismatch(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        hit = _make_hit("c1")
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(
                query_id="different-id", top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.HYBRID
            ),
            strategy=RetrievalStrategy.HYBRID,
            hits=[hit],
            latency_ms=5.0,
            artifact_versions={"chunk": "v1"},
        )
        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        with pytest.raises(RetrievalError, match="incompatible response"):
            retriever.search(_make_query(query_id="my-id", top_k=8, candidate_k=40))

    def test_11_search_fails_if_candidate_hits_exceed_fusion_limit(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        hits = [_make_hit(f"c{i}") for i in range(25)]
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.HYBRID),
            strategy=RetrievalStrategy.HYBRID,
            hits=hits,
            latency_ms=5.0,
            artifact_versions={"chunk": "v1"},
        )
        retriever = RelationshipSeedRerankingRetriever(
            candidate_retriever, reranker, RetrievalConfig(relationship_rerank_fusion_k=20)
        )
        with pytest.raises(RetrievalError, match="incompatible response"):
            retriever.search(_make_query(top_k=8, candidate_k=40))

    def test_12_search_passes_candidate_hits_to_inner_reranker(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"
        hits = [_make_hit("c1"), _make_hit("c2")]
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.HYBRID),
            strategy=RetrievalStrategy.HYBRID,
            hits=hits,
            latency_ms=5.0,
            artifact_versions={"chunk": "v1"},
        )

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates), warnings=["warn1"])

        reranker.rerank.side_effect = _mock_rerank

        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        resp = retriever.search(_make_query(top_k=8, candidate_k=40))

        assert reranker.rerank.call_count == 1
        rerank_hits_arg = reranker.rerank.call_args[0][1]
        assert len(rerank_hits_arg) == 2
        assert rerank_hits_arg[0].chunk_id == "c1"
        assert resp.warnings == ["reranker:warn1"]

    def test_13_search_returns_hybrid_rerank_strategy_and_provenance(self) -> None:
        candidate_retriever = MagicMock()
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"
        hits = [_make_hit("c1", score=0.95)]
        candidate_retriever.search.return_value = RetrievalResponse(
            query=_make_query(top_k=20, candidate_k=40, requested_strategy=RetrievalStrategy.HYBRID),
            strategy=RetrievalStrategy.HYBRID,
            hits=hits,
            latency_ms=5.0,
            artifact_versions={"chunk": "v1", "vector": "v2"},
        )

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates))

        reranker.rerank.side_effect = _mock_rerank

        retriever = RelationshipSeedRerankingRetriever(candidate_retriever, reranker)
        resp = retriever.search(_make_query(top_k=8, candidate_k=40))

        assert resp.strategy == RetrievalStrategy.HYBRID_RERANK
        assert resp.query.requested_strategy == RetrievalStrategy.HYBRID_RERANK
        assert resp.hits[0].strategy == RetrievalStrategy.HYBRID_RERANK
        assert resp.artifact_versions == {"chunk": "v1", "vector": "v2"}

    def test_14_fixed_retriever_instantiates_relationship_reranker(self) -> None:
        bm25 = MagicMock()
        dense = MagicMock()
        reranker = MagicMock(spec=Reranker)
        retriever = FixedRetriever(bm25, dense, reranker=reranker)
        assert isinstance(retriever.relationship_reranker, RelationshipSeedRerankingRetriever)

    def test_15_fixed_retriever_relationship_reranker_property_none_when_no_reranker(self) -> None:
        bm25 = MagicMock()
        dense = MagicMock()
        retriever = FixedRetriever(bm25, dense, reranker=None)
        assert retriever.relationship_reranker is None

    def test_16_fixed_retriever_search_relationship_rerank(self) -> None:
        bm25 = MagicMock()
        dense = MagicMock()
        bm25.source_artifact_identity = ("chunk", "1.0", "hash1")
        dense.source_artifact_identity = ("chunk", "1.0", "hash1")
        hit_bm25 = _make_hit("c1").model_copy(update={"strategy": RetrievalStrategy.BM25})
        hit_dense = _make_hit("c1").model_copy(update={"strategy": RetrievalStrategy.DENSE})
        bm25.search.return_value = RetrievalResponse(
            query=_make_query(requested_strategy=RetrievalStrategy.BM25),
            strategy=RetrievalStrategy.BM25,
            hits=[hit_bm25],
            latency_ms=1.0,
            artifact_versions={"chunk": "1.0"},
        )
        dense.search.return_value = RetrievalResponse(
            query=_make_query(requested_strategy=RetrievalStrategy.DENSE),
            strategy=RetrievalStrategy.DENSE,
            hits=[hit_dense],
            latency_ms=1.0,
            artifact_versions={"chunk": "1.0"},
        )
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates))

        reranker.rerank.side_effect = _mock_rerank

        retriever = FixedRetriever(bm25, dense, reranker=reranker)
        resp = retriever.search_relationship_rerank(_make_query(top_k=8, candidate_k=40))
        assert resp.strategy == RetrievalStrategy.HYBRID_RERANK

    def test_17_fixed_retriever_search_relationship_rerank_raises_without_reranker(self) -> None:
        bm25 = MagicMock()
        dense = MagicMock()
        retriever = FixedRetriever(bm25, dense, reranker=None)
        with pytest.raises(RetrievalError, match="has no reranker"):
            retriever.search_relationship_rerank(_make_query())

    def test_18_fixed_retriever_search_rejects_graph_strategy(self) -> None:
        bm25 = MagicMock()
        dense = MagicMock()
        retriever = FixedRetriever(bm25, dense)
        with pytest.raises(RetrievalError, match="not implemented"):
            retriever.search(_make_query(requested_strategy=RetrievalStrategy.GRAPH))

    def test_19_fixed_retriever_search_dispatches_bm25_dense_hybrid_hybrid_rerank(self) -> None:
        bm25 = MagicMock()
        dense = MagicMock()
        bm25.source_artifact_identity = ("chunk", "1.0", "hash1")
        dense.source_artifact_identity = ("chunk", "1.0", "hash1")
        hit_bm25 = _make_hit("c1").model_copy(update={"strategy": RetrievalStrategy.BM25})
        hit_dense = _make_hit("c1").model_copy(update={"strategy": RetrievalStrategy.DENSE})
        bm25.search.return_value = RetrievalResponse(
            query=_make_query(requested_strategy=RetrievalStrategy.BM25),
            strategy=RetrievalStrategy.BM25,
            hits=[hit_bm25],
            latency_ms=1.0,
            artifact_versions={"chunk": "1.0"},
        )
        dense.search.return_value = RetrievalResponse(
            query=_make_query(requested_strategy=RetrievalStrategy.DENSE),
            strategy=RetrievalStrategy.DENSE,
            hits=[hit_dense],
            latency_ms=1.0,
            artifact_versions={"chunk": "1.0"},
        )
        reranker = MagicMock(spec=Reranker)
        reranker.model_name = "test-reranker"

        def _mock_rerank(query: RetrievalQuery, candidates: list[RetrievalHit]) -> RetrievalResponse:
            return _make_reranked_response(query, list(candidates))

        reranker.rerank.side_effect = _mock_rerank

        retriever = FixedRetriever(bm25, dense, reranker=reranker)

        # BM25
        assert retriever.search(_make_query(requested_strategy=RetrievalStrategy.BM25)).strategy == RetrievalStrategy.BM25
        # DENSE
        assert retriever.search(_make_query(requested_strategy=RetrievalStrategy.DENSE)).strategy == RetrievalStrategy.DENSE
        # HYBRID
        assert retriever.search(_make_query(requested_strategy=RetrievalStrategy.HYBRID)).strategy == RetrievalStrategy.HYBRID
        # HYBRID_RERANK
        assert retriever.search(_make_query(requested_strategy=RetrievalStrategy.HYBRID_RERANK)).strategy == RetrievalStrategy.HYBRID_RERANK

    def test_20_fixed_retrieval_tools_creates_five_tools(self) -> None:
        retriever = MagicMock()
        tools = fixed_retrieval_tools(retriever)
        tool_names = [t.name for t in tools]
        assert len(tools) == 5
        assert ToolName.RELATIONSHIP_RERANK_SEARCH in tool_names
        assert ToolName.RERANK_SEARCH in tool_names
        assert ToolName.HYBRID_SEARCH in tool_names
        assert ToolName.DENSE_SEARCH in tool_names
        assert ToolName.BM25_SEARCH in tool_names

    def test_21_graph_search_not_in_fixed_retrieval_tools(self) -> None:
        retriever = MagicMock()
        tools = fixed_retrieval_tools(retriever)
        tool_names = [t.name.value for t in tools]
        assert "graph_search" not in tool_names

    def test_22_relationship_rerank_search_maps_to_hybrid_rerank(self) -> None:
        retriever = MagicMock()
        tool = RetrievalTool(ToolName.RELATIONSHIP_RERANK_SEARCH, retriever)
        assert tool._strategy == RetrievalStrategy.HYBRID_RERANK

    def test_23_relationship_rerank_search_invokes_relationship_retriever(self) -> None:
        rel_retriever = MagicMock()
        hit = _make_hit("c1")
        rel_retriever.search.return_value = RetrievalResponse(
            query=_make_query(requested_strategy=RetrievalStrategy.HYBRID_RERANK),
            strategy=RetrievalStrategy.HYBRID_RERANK,
            hits=[hit],
            latency_ms=2.0,
            artifact_versions={},
        )
        tool = RetrievalTool(ToolName.RELATIONSHIP_RERANK_SEARCH, rel_retriever)
        query = _make_query(requested_strategy=RetrievalStrategy.HYBRID_RERANK)
        resp = tool.invoke(query)
        assert resp.strategy == RetrievalStrategy.HYBRID_RERANK
        assert rel_retriever.search.call_count == 1

    def test_24_deterministic_strategy_router_rejects_graph(self) -> None:
        router = DeterministicStrategyRouter()
        query = _make_query(requested_strategy=RetrievalStrategy.GRAPH)
        with pytest.raises(InvalidUserInputError, match="not available"):
            router.plan(query, {ToolName.BM25_SEARCH, ToolName.RELATIONSHIP_RERANK_SEARCH})

    def test_25_router_plans_3_attempts_for_relationship_intent(self) -> None:
        router = DeterministicStrategyRouter(
            AgentConfig(max_retry=2),
            QueryUnderstandingConfig(adaptive_routing_enabled=True),
        )
        analysis = QueryAnalysis(intent=QueryIntent.RELATIONSHIP)
        query = _make_query(question="quan hệ giữa văn bản A và B", query_analysis=analysis)
        registered = {
            ToolName.RELATIONSHIP_RERANK_SEARCH,
            ToolName.RERANK_SEARCH,
            ToolName.HYBRID_SEARCH,
            ToolName.BM25_SEARCH,
            ToolName.DENSE_SEARCH,
        }
        routes = router.plan(query, registered)
        assert len(routes) == 3
        assert routes[0] == RetrievalRoute(RetrievalStrategy.HYBRID_RERANK, ToolName.RELATIONSHIP_RERANK_SEARCH)
        assert routes[1] == RetrievalRoute(RetrievalStrategy.HYBRID_RERANK, ToolName.RERANK_SEARCH)
        assert routes[2] == RetrievalRoute(RetrievalStrategy.HYBRID, ToolName.HYBRID_SEARCH)

    def test_26_router_preserves_both_hybrid_rerank_routes(self) -> None:
        router = DeterministicStrategyRouter(
            AgentConfig(max_retry=2),
            QueryUnderstandingConfig(adaptive_routing_enabled=True),
        )
        analysis = QueryAnalysis(intent=QueryIntent.RELATIONSHIP)
        query = _make_query(question="quan hệ giữa văn bản A và B", query_analysis=analysis)
        registered = {
            ToolName.RELATIONSHIP_RERANK_SEARCH,
            ToolName.RERANK_SEARCH,
            ToolName.HYBRID_SEARCH,
        }
        routes = router.plan(query, registered)
        tool_names = [r.tool_name for r in routes]
        assert tool_names == [
            ToolName.RELATIONSHIP_RERANK_SEARCH,
            ToolName.RERANK_SEARCH,
            ToolName.HYBRID_SEARCH,
        ]

    def test_27_router_respects_max_retry_limit(self) -> None:
        router = DeterministicStrategyRouter(
            AgentConfig(max_retry=1),
            QueryUnderstandingConfig(adaptive_routing_enabled=True),
        )
        analysis = QueryAnalysis(intent=QueryIntent.RELATIONSHIP)
        query = _make_query(question="quan hệ", query_analysis=analysis)
        registered = {
            ToolName.RELATIONSHIP_RERANK_SEARCH,
            ToolName.RERANK_SEARCH,
            ToolName.HYBRID_SEARCH,
        }
        routes = router.plan(query, registered)
        assert len(routes) == 2  # max_retry (1) + 1 = 2

    def test_28_artifact_set_validator_supports_required_artifact_types(self) -> None:
        artifacts = MagicMock()
        policy = MagicMock()
        policy.require_full_corpus = False
        validator = ArtifactSetValidator(
            artifacts, policy, required_artifact_types=COMPETITION_REQUIRED_ARTIFACT_TYPES
        )
        assert validator._required_artifact_types == COMPETITION_REQUIRED_ARTIFACT_TYPES

    def test_29_competition_required_artifact_types_contains_exact_six_types(self) -> None:
        assert len(COMPETITION_REQUIRED_ARTIFACT_TYPES) == 6
        assert ArtifactType.NORMALIZED_DOCUMENTS in COMPETITION_REQUIRED_ARTIFACT_TYPES
        assert ArtifactType.CLEANED_DOCUMENTS in COMPETITION_REQUIRED_ARTIFACT_TYPES
        assert ArtifactType.LEGAL_BLOCKS in COMPETITION_REQUIRED_ARTIFACT_TYPES
        assert ArtifactType.LEGAL_CHUNKS in COMPETITION_REQUIRED_ARTIFACT_TYPES
        assert ArtifactType.BM25_INDEX in COMPETITION_REQUIRED_ARTIFACT_TYPES
        assert ArtifactType.VECTOR_INDEX in COMPETITION_REQUIRED_ARTIFACT_TYPES
        assert ArtifactType.RELATIONSHIP_MAPPING not in COMPETITION_REQUIRED_ARTIFACT_TYPES
        assert ArtifactType.GRAPH_INDEX not in COMPETITION_REQUIRED_ARTIFACT_TYPES

    def test_30_validate_lineage_passes_for_competition_profile(self) -> None:
        artifacts = MagicMock()
        policy = MagicMock()
        policy.require_full_corpus = False
        validator = ArtifactSetValidator(
            artifacts, policy, required_artifact_types=COMPETITION_REQUIRED_ARTIFACT_TYPES
        )

        norm = _make_manifest(ArtifactType.NORMALIZED_DOCUMENTS)
        norm.processing_config_hash = "norm_hash"
        norm.record_count = 10

        clean = _make_manifest(ArtifactType.CLEANED_DOCUMENTS)
        clean.metadata["source_processing_config_hash"] = "norm_hash"
        clean.processing_config_hash = "clean_hash"
        clean.record_count = 10

        blocks = _make_manifest(ArtifactType.LEGAL_BLOCKS)
        blocks.metadata["source_processing_config_hash"] = "clean_hash"
        blocks.processing_config_hash = "block_hash"
        blocks.record_count = 20

        chunks = _make_manifest(ArtifactType.LEGAL_CHUNKS)
        chunks.metadata["source_processing_config_hash"] = "block_hash"
        chunks.metadata["runtime_normalized_processing_config_hash"] = "norm_hash"
        chunks.processing_config_hash = "chunk_hash"
        chunks.record_count = 30

        bm25 = _make_manifest(ArtifactType.BM25_INDEX)
        bm25.metadata["source_processing_config_hash"] = "chunk_hash"
        bm25.record_count = 30

        vec = _make_manifest(ArtifactType.VECTOR_INDEX)
        vec.metadata["source_processing_config_hash"] = "chunk_hash"
        vec.record_count = 30

        manifests = {
            ArtifactType.NORMALIZED_DOCUMENTS: norm,
            ArtifactType.CLEANED_DOCUMENTS: clean,
            ArtifactType.LEGAL_BLOCKS: blocks,
            ArtifactType.LEGAL_CHUNKS: chunks,
            ArtifactType.BM25_INDEX: bm25,
            ArtifactType.VECTOR_INDEX: vec,
        }

        errors = validator._validate_lineage(manifests)
        assert errors == []

    def test_31_online_runtime_manifests_has_exact_3_types(self) -> None:
        chunk_manifest = _make_manifest(ArtifactType.LEGAL_CHUNKS)
        bm25_manifest = _make_manifest(ArtifactType.BM25_INDEX)
        vector_manifest = _make_manifest(ArtifactType.VECTOR_INDEX)
        manifests = {
            m.artifact_type.value: m
            for m in (chunk_manifest, bm25_manifest, vector_manifest)
        }
        assert len(manifests) == 3
        assert ArtifactType.LEGAL_CHUNKS.value in manifests
        assert ArtifactType.BM25_INDEX.value in manifests
        assert ArtifactType.VECTOR_INDEX.value in manifests
        assert ArtifactType.GRAPH_INDEX.value not in manifests

    def test_32_tool_registry_does_not_contain_graph_search(self) -> None:
        from legal_agentic_rag.tools.factory import build_fixed_tool_registry
        retriever = MagicMock()
        registry = build_fixed_tool_registry(
            retriever=retriever,
            context_grader=MagicMock(),
            answer_generator=MagicMock(),
            citation_verifier=MagicMock(),
        )
        descriptors = registry.descriptors()
        descriptor_names = [d.name.value for d in descriptors]
        assert "graph_search" not in descriptor_names
        assert ToolName.RELATIONSHIP_RERANK_SEARCH.value in descriptor_names
