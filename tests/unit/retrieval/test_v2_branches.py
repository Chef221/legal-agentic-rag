"""Unit tests for V2 BM25, Dense retrieval branch adapters, hybrid fusion, and reranker integration."""

from collections.abc import Sequence
from typing import Any
import pytest

from legal_agentic_rag.configuration.online import RerankerConfig
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.retrieval.fixed import FixedRetriever, HybridRetriever
from legal_agentic_rag.retrieval.v2_branches import (
    V2BM25RetrievalBranch,
    V2DenseRetrievalBranch,
    build_v2_fixed_retriever,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)


class MockBM25Backend:
    def __init__(self, record_count: int = 100, source_sha: str = "sha_v2_common"):
        self.record_count = record_count
        self.manifest = {
            "schema": "m54_v2_bm25_index_v1",
            "source_retrieval_units_sha256": source_sha,
        }
        self.last_query: RetrievalQuery | None = None
        self.returned_hits: list[RetrievalHit] = []

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.last_query = query
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.BM25,
            hits=self.returned_hits,
            latency_ms=1.5,
            artifact_versions={"bm25_index": "m54_v2_bm25_index_v1"},
        )


class MockDenseBackend:
    def __init__(
        self,
        record_count: int = 100,
        source_sha: str = "sha_v2_common",
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        model_revision: str = "dea33aa1ab339f38d66ae0a40e6c40e0a9249568",
        dimension: int = 1024,
    ):
        self.record_count = record_count
        self.model_name = model_name
        self.model_revision = model_revision
        self.dimension = dimension
        self.manifest = {
            "schema": "m54_v2_dense_matrix_index_v1",
            "source_retrieval_units_sha256": source_sha,
            "model_name": model_name,
            "model_revision": model_revision,
            "dimension": dimension,
        }
        self.last_query: RetrievalQuery | None = None
        self.last_vector: Any = None
        self.returned_hits: list[RetrievalHit] = []

    def retrieve(self, query: RetrievalQuery, query_vector: Any) -> RetrievalResponse:
        self.last_query = query
        self.last_vector = query_vector
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.DENSE,
            hits=self.returned_hits,
            latency_ms=2.0,
            artifact_versions={"vector_index": "m54_v2_dense_matrix_index_v1"},
        )


class MockEmbeddingProvider:
    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        model_revision: str = "dea33aa1ab339f38d66ae0a40e6c40e0a9249568",
        dimension: int = 1024,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.dimension = dimension
        self.last_embedded_text: str | None = None

    def embed_query(self, text: str) -> list[float]:
        self.last_embedded_text = text
        return [0.05] * self.dimension


class MockReranker:
    """Mock cross-encoder reranker conforming to Reranker protocol."""

    def __init__(self, model_name: str = "mock-reranker"):
        self.provider_name = "mock_provider"
        self.provider_version = "1.0"
        self.model_name = model_name
        self.model_revision = "mock_rev_1"
        self.last_query: RetrievalQuery | None = None
        self.last_candidates: Sequence[RetrievalHit] = []

    def rerank(
        self, query: RetrievalQuery, candidates: Sequence[RetrievalHit]
    ) -> RetrievalResponse:
        self.last_query = query
        self.last_candidates = list(candidates)
        top_candidates = list(candidates)[: query.top_k]
        hits: list[RetrievalHit] = []
        for rank_idx, cand in enumerate(top_candidates, start=1):
            score = 10.0 - float(rank_idx)
            updated_trace = cand.retrieval_trace.model_copy(
                update={"reranker_score": score}
            )
            hit = cand.model_copy(
                update={
                    "rank": rank_idx,
                    "score": score,
                    "strategy": RetrievalStrategy.RERANK,
                    "retrieval_trace": updated_trace,
                }
            )
            hits.append(hit)
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.RERANK,
            hits=hits,
            latency_ms=3.5,
            artifact_versions={"reranker": self.model_name},
        )


def _create_hit(
    unit_id: str,
    doc_id: str,
    rank: int,
    score: float,
    strategy: RetrievalStrategy,
    text: str = "Nội dung điều luật",
    metadata: dict[str, Any] | None = None,
) -> RetrievalHit:
    meta = metadata or {
        "provision_id": f"{doc_id}::art:1",
        "retrieval_text": "Text tim kiem",
        "document_identity": {
            "title": "Luật Mẫu",
            "document_number": "01/2024/QH15",
        },
        "hierarchy": {
            "article_label": "1",
            "clause_label": "1",
            "point_label": None,
            "heading_path": [{"type": "CHAPTER", "label": "I", "title": "QUY ĐỊNH CHUNG"}],
        },
        "strategy": "WHOLE_PROVISION",
    }
    trace = RetrievalTrace()
    if strategy == RetrievalStrategy.BM25:
        trace.bm25_rank = rank
        trace.bm25_score = score
    elif strategy == RetrievalStrategy.DENSE:
        trace.dense_rank = rank
        trace.dense_score = score

    return RetrievalHit(
        chunk_id=unit_id,
        document_id=doc_id,
        rank=rank,
        score=score,
        strategy=strategy,
        text=text,
        metadata=meta,
        retrieval_trace=trace,
    )


# 1. BM25 adapter exposes truthful V2 source identity
def test_v2_bm25_source_identity():
    bm25_backend = MockBM25Backend(record_count=1190081, source_sha="sha_test_123")
    branch = V2BM25RetrievalBranch(bm25_backend)
    assert branch.source_artifact_identity == ("retrieval_units_v2", "1190081", "sha_test_123")


# 2. Dense adapter exposes same truthful V2 source identity
def test_v2_dense_source_identity():
    dense_backend = MockDenseBackend(record_count=1190081, source_sha="sha_test_123")
    provider = MockEmbeddingProvider()
    branch = V2DenseRetrievalBranch(dense_backend, provider)
    assert branch.source_artifact_identity == ("retrieval_units_v2", "1190081", "sha_test_123")


# 3. Mismatched sparse/dense source SHA causes existing HybridRetriever to reject
def test_hybrid_rejects_source_sha_mismatch():
    bm25_backend = MockBM25Backend(record_count=100, source_sha="sha_A")
    dense_backend = MockDenseBackend(record_count=100, source_sha="sha_B")
    provider = MockEmbeddingProvider()

    bm25_branch = V2BM25RetrievalBranch(bm25_backend)
    dense_branch = V2DenseRetrievalBranch(dense_backend, provider)

    hybrid = HybridRetriever(bm25_branch, dense_branch)
    with pytest.raises(ArtifactCompatibilityError, match="different chunk artifacts"):
        _ = hybrid.source_artifact_identity


# 4. Mismatched record count causes source identity mismatch
def test_hybrid_rejects_record_count_mismatch():
    bm25_backend = MockBM25Backend(record_count=100, source_sha="sha_common")
    dense_backend = MockDenseBackend(record_count=200, source_sha="sha_common")
    provider = MockEmbeddingProvider()

    bm25_branch = V2BM25RetrievalBranch(bm25_backend)
    dense_branch = V2DenseRetrievalBranch(dense_backend, provider)

    hybrid = HybridRetriever(bm25_branch, dense_branch)
    with pytest.raises(ArtifactCompatibilityError, match="different chunk artifacts"):
        _ = hybrid.source_artifact_identity


# 5, 6, 7. Dense adapter rejects provider model / revision / dimension mismatch
def test_dense_adapter_compatibility_gate():
    dense_backend = MockDenseBackend(
        model_name="AITeamVN/Vietnamese_Embedding",
        model_revision="dea33aa1ab339f38d66ae0a40e6c40e0a9249568",
        dimension=1024,
    )

    # Model name mismatch
    with pytest.raises(ArtifactCompatibilityError, match="model 'Qwen/Qwen2.5-Coder' does not match"):
        V2DenseRetrievalBranch(
            dense_backend,
            MockEmbeddingProvider(model_name="Qwen/Qwen2.5-Coder"),
        )

    # Model revision mismatch
    with pytest.raises(ArtifactCompatibilityError, match="revision 'wrong_rev' does not match"):
        V2DenseRetrievalBranch(
            dense_backend,
            MockEmbeddingProvider(model_revision="wrong_rev"),
        )

    # Dimension mismatch
    with pytest.raises(ArtifactCompatibilityError, match="dimension 384 does not match"):
        V2DenseRetrievalBranch(
            dense_backend,
            MockEmbeddingProvider(dimension=384),
        )


# 8, 9, 10. Dense adapter query embedding and vector passing
def test_dense_adapter_query_embedding_routing():
    dense_backend = MockDenseBackend()
    provider = MockEmbeddingProvider()
    branch = V2DenseRetrievalBranch(dense_backend, provider)

    # Test rewritten_question prioritized
    q1 = RetrievalQuery(
        query_id="q1",
        original_question="cau hoi goc",
        normalized_question="cau hoi chuan hoa",
        rewritten_question="cau hoi viet lai",
    )
    branch.search(q1)
    assert provider.last_embedded_text == "cau hoi viet lai"
    assert dense_backend.last_query.query_id == "q1"
    assert len(dense_backend.last_vector) == 1024

    # Test normalized_question when rewritten is None
    q2 = RetrievalQuery(
        query_id="q2",
        original_question="cau hoi goc 2",
        normalized_question="cau hoi chuan hoa 2",
        rewritten_question=None,
    )
    branch.search(q2)
    assert provider.last_embedded_text == "cau hoi chuan hoa 2"


# 11. BM25 adapter delegates search
def test_bm25_adapter_delegation():
    bm25_backend = MockBM25Backend()
    branch = V2BM25RetrievalBranch(bm25_backend)

    q = RetrievalQuery(
        query_id="q_bm25",
        original_question="hoi luat",
        normalized_question="hoi luat",
    )
    branch.search(q)
    assert bm25_backend.last_query.query_id == "q_bm25"


# 12, 13, 14, 15, 16. HybridRetriever RRF fusion with shared and branch-only hits
def test_hybrid_rrf_fusion_details():
    shared_meta = {"provision_id": "doc:1::art:1", "strategy": "WHOLE_PROVISION"}

    bm25_hit_1 = _create_hit("doc:1::art:1", "doc:1", rank=1, score=-1.5, strategy=RetrievalStrategy.BM25, metadata=shared_meta)
    bm25_hit_2 = _create_hit("doc:1::art:2", "doc:1", rank=2, score=-3.2, strategy=RetrievalStrategy.BM25)

    dense_hit_1 = _create_hit("doc:1::art:1", "doc:1", rank=1, score=0.88, strategy=RetrievalStrategy.DENSE, metadata=shared_meta)
    dense_hit_3 = _create_hit("doc:2::art:1", "doc:2", rank=2, score=0.75, strategy=RetrievalStrategy.DENSE)

    bm25_backend = MockBM25Backend(source_sha="sha_same")
    bm25_backend.returned_hits = [bm25_hit_1, bm25_hit_2]

    dense_backend = MockDenseBackend(source_sha="sha_same")
    dense_backend.returned_hits = [dense_hit_1, dense_hit_3]

    provider = MockEmbeddingProvider()

    retriever = build_v2_fixed_retriever(bm25_backend, dense_backend, provider)

    query = RetrievalQuery(
        query_id="q_hybrid",
        original_question="tim kiem hon hop",
        normalized_question="tim kiem hon hop",
        requested_strategy=RetrievalStrategy.HYBRID,
        top_k=5,
    )

    response = retriever.search(query)

    assert response.strategy == RetrievalStrategy.HYBRID
    assert len(response.hits) == 3

    # Rank 1 must be shared unit (doc:1::art:1)
    h0 = response.hits[0]
    assert h0.chunk_id == "doc:1::art:1"
    assert h0.strategy == RetrievalStrategy.HYBRID
    assert h0.retrieval_trace.bm25_rank == 1
    assert h0.retrieval_trace.dense_rank == 1
    assert h0.retrieval_trace.bm25_score == -1.5
    assert h0.retrieval_trace.dense_score == 0.88
    assert h0.retrieval_trace.bm25_rrf_contribution > 0
    assert h0.retrieval_trace.dense_rrf_contribution > 0
    assert h0.retrieval_trace.rrf_score > 0

    chunk_ids = {h.chunk_id for h in response.hits}
    assert "doc:1::art:2" in chunk_ids
    assert "doc:2::art:1" in chunk_ids


# 17, 18. FixedRetriever strategy routing (BM25, DENSE, HYBRID) without reranker
def test_fixed_retriever_routing():
    bm25_backend = MockBM25Backend(source_sha="sha_same")
    bm25_backend.returned_hits = [_create_hit("doc:1::art:1", "doc:1", 1, -1.0, RetrievalStrategy.BM25)]

    dense_backend = MockDenseBackend(source_sha="sha_same")
    dense_backend.returned_hits = [_create_hit("doc:2::art:1", "doc:2", 1, 0.9, RetrievalStrategy.DENSE)]

    provider = MockEmbeddingProvider()

    retriever = build_v2_fixed_retriever(bm25_backend, dense_backend, provider)

    # Route BM25
    q_bm25 = RetrievalQuery(query_id="q1", original_question="test", normalized_question="test", requested_strategy=RetrievalStrategy.BM25)
    r_bm25 = retriever.search(q_bm25)
    assert r_bm25.strategy == RetrievalStrategy.BM25
    assert r_bm25.hits[0].chunk_id == "doc:1::art:1"

    # Route DENSE
    q_dense = RetrievalQuery(query_id="q2", original_question="test", normalized_question="test", requested_strategy=RetrievalStrategy.DENSE)
    r_dense = retriever.search(q_dense)
    assert r_dense.strategy == RetrievalStrategy.DENSE
    assert r_dense.hits[0].chunk_id == "doc:2::art:1"

    # Route HYBRID
    q_hybrid = RetrievalQuery(query_id="q3", original_question="test", normalized_question="test", requested_strategy=RetrievalStrategy.HYBRID)
    r_hybrid = retriever.search(q_hybrid)
    assert r_hybrid.strategy == RetrievalStrategy.HYBRID
    assert len(r_hybrid.hits) == 2


# ==============================================================================
# STEP 8: HYBRID-RERANK INTEGRATION TESTS
# ==============================================================================

def test_v2_fixed_retriever_accepts_reranker_and_routes_hybrid_rerank():
    shared_meta = {
        "provision_id": "doc:1::art:1",
        "document_identity": {"title": "Luật Đất đai", "document_number": "31/2024/QH15"},
        "hierarchy": {"article_label": "1", "clause_label": "1", "point_label": None, "heading_path": []},
        "strategy": "WHOLE_PROVISION",
    }

    bm25_hit = _create_hit("doc:1::art:1", "doc:1", 1, -1.0, RetrievalStrategy.BM25, metadata=shared_meta)
    dense_hit = _create_hit("doc:1::art:1", "doc:1", 1, 0.95, RetrievalStrategy.DENSE, metadata=shared_meta)
    dense_hit_2 = _create_hit("doc:2::art:5", "doc:2", 2, 0.70, RetrievalStrategy.DENSE)

    bm25_backend = MockBM25Backend(source_sha="sha_common_v2")
    bm25_backend.returned_hits = [bm25_hit]

    dense_backend = MockDenseBackend(source_sha="sha_common_v2")
    dense_backend.returned_hits = [dense_hit, dense_hit_2]

    provider = MockEmbeddingProvider()
    mock_reranker = MockReranker()
    reranker_cfg = RerankerConfig(max_candidates=40)

    # 1. build_v2_fixed_retriever accepts optional reranker and config
    retriever = build_v2_fixed_retriever(
        bm25_backend=bm25_backend,
        dense_backend=dense_backend,
        embedding_provider=provider,
        reranker=mock_reranker,
        reranker_config=reranker_cfg,
    )

    # 11. source artifact identity remains the V2 common lineage
    assert retriever._hybrid.source_artifact_identity == ("retrieval_units_v2", "100", "sha_common_v2")

    query = RetrievalQuery(
        query_id="q_hybrid_rerank",
        original_question="hỏi đất đai",
        normalized_question="hoi dat dai",
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        top_k=2,
        candidate_k=10,
    )

    # 2. HYBRID_RERANK routing succeeds
    response = retriever.search(query)

    # 4. Candidate stage was HYBRID (checked via mock reranker inputs)
    assert mock_reranker.last_query is not None
    assert mock_reranker.last_query.requested_strategy == RetrievalStrategy.RERANK

    # 5. Reranker receives candidates up to candidate_k
    assert len(mock_reranker.last_candidates) <= 10

    # 6. Final response strategy is HYBRID_RERANK
    assert response.strategy == RetrievalStrategy.HYBRID_RERANK

    # 7. Final hit count <= top_k
    assert len(response.hits) <= 2
    assert len(response.hits) >= 1

    top_hit = response.hits[0]

    # 8. Retrieval provenance survives reranking (bm25_rank, dense_rank, rrf fields)
    assert top_hit.retrieval_trace.bm25_rank == 1
    assert top_hit.retrieval_trace.dense_rank == 1
    assert top_hit.retrieval_trace.bm25_rrf_contribution > 0
    assert top_hit.retrieval_trace.dense_rrf_contribution > 0
    assert top_hit.retrieval_trace.rrf_score > 0

    # 9. reranker_score survives in trace
    assert top_hit.retrieval_trace.reranker_score is not None
    assert top_hit.score == top_hit.retrieval_trace.reranker_score

    # 10. V2 metadata survives reranking
    assert top_hit.metadata["document_identity"]["title"] == "Luật Đất đai"
    assert top_hit.metadata["document_identity"]["document_number"] == "31/2024/QH15"
    assert top_hit.metadata["provision_id"] == "doc:1::art:1"


# 3. No reranker => fail-closed behavior remains for HYBRID_RERANK
def test_v2_fixed_retriever_fails_closed_without_reranker():
    bm25_backend = MockBM25Backend()
    dense_backend = MockDenseBackend()
    provider = MockEmbeddingProvider()

    retriever = build_v2_fixed_retriever(
        bm25_backend=bm25_backend,
        dense_backend=dense_backend,
        embedding_provider=provider,
        reranker=None,
    )

    query = RetrievalQuery(
        query_id="q_fail",
        original_question="test",
        normalized_question="test",
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )

    with pytest.raises(RetrievalError, match="has no reranker"):
        retriever.search(query)
