"""Unit tests for V2 BM25 and Dense retrieval branch adapters and hybrid fusion."""

from typing import Any
import pytest

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

    # Unit 1 is in BOTH branches (shared)
    bm25_hit_1 = _create_hit("doc:1::art:1", "doc:1", rank=1, score=-1.5, strategy=RetrievalStrategy.BM25, metadata=shared_meta)
    # Unit 2 is BM25-only
    bm25_hit_2 = _create_hit("doc:1::art:2", "doc:1", rank=2, score=-3.2, strategy=RetrievalStrategy.BM25)

    dense_hit_1 = _create_hit("doc:1::art:1", "doc:1", rank=1, score=0.88, strategy=RetrievalStrategy.DENSE, metadata=shared_meta)
    # Unit 3 is Dense-only
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

    # Branch-only hits remain eligible
    chunk_ids = {h.chunk_id for h in response.hits}
    assert "doc:1::art:2" in chunk_ids  # BM25-only hit
    assert "doc:2::art:1" in chunk_ids  # Dense-only hit


# 17, 18. FixedRetriever strategy routing (BM25, DENSE, HYBRID) without reranker or graph
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
