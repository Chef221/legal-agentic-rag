"""Tests for online query embedding and dense search orchestration."""

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.indexing.vector import NumpyVectorBackend
from legal_agentic_rag.retrieval import DenseRetriever
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    LegalStructure,
    RetrievalQuery,
    RetrievalStrategy,
)


def _chunks() -> list[LegalChunk]:
    values: list[LegalChunk] = []
    for index, text in enumerate(("quá tốc độ", "nộp thuế"), start=1):
        values.append(
            LegalChunk(
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                chunk_index=0,
                text=text,
                search_text=text,
                token_count=2,
                structure=LegalStructure(article_number=str(index)),
                source_dataset="aio",
                metadata={"source_block_ids": [f"block-{index}"]},
            )
        )
    return values


def _source_manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        record_count=2,
        processing_config_hash="chunk-hash",
    )


class _QueryProvider:
    provider_name = "fixture-provider"
    provider_version = "1.0"
    model_name = "fixture/e5"
    model_revision = "fixture-revision"
    dimension = 2

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


def _backend(
    chunks: list[LegalChunk],
    vectors: list[list[float]],
    source_manifest: ArtifactManifest,
) -> NumpyVectorBackend:
    backend = NumpyVectorBackend()
    backend.build(
        chunks,
        vectors,
        source_manifest,
        model_name="fixture/e5",
        model_revision="fixture-revision",
        embedding_provider_name="fixture-provider",
        embedding_provider_version="1.0",
        dimension=2,
        embedding_batch_size=2,
    )
    return backend


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-dense",
        original_question="Câu hỏi gốc",
        normalized_question="câu hỏi chuẩn hóa",
        requested_strategy=RetrievalStrategy.DENSE,
    )


def test_dense_retriever_embeds_normalized_question_and_returns_total_latency(
) -> None:
    """Online orchestration embeds only normalized text and includes its latency."""
    provider = _QueryProvider()
    backend = _backend(_chunks(), [[1.0, 0.0], [0.0, 1.0]], _source_manifest())

    response = DenseRetriever(provider, backend).search(_query())

    assert provider.queries == ["câu hỏi chuẩn hóa"]
    assert response.strategy == RetrievalStrategy.DENSE
    assert response.hits
    assert response.latency_ms >= 0
    assert provider.queries
    assert DenseRetriever(provider, backend).source_artifact_identity == (
        "legal_chunks",
        "1.0",
        "chunk-hash",
    )


def test_dense_retriever_prefers_agent_rewritten_question() -> None:
    """Agent retries embed the rewritten query rather than stale normalized text."""
    provider = _QueryProvider()
    backend = _backend(_chunks(), [[1.0, 0.0], [0.0, 1.0]], _source_manifest())
    query = _query().model_copy(
        update={"rewritten_question": "truy vấn đã viết lại"}
    )

    DenseRetriever(provider, backend).search(query)

    assert provider.queries == ["truy vấn đã viết lại"]


def test_dense_retriever_rejects_provider_model_revision_and_dimension_mismatch(
) -> None:
    """Online provider must exactly match the persisted vector artifact."""
    backend = _backend(_chunks(), [[1.0, 0.0], [0.0, 1.0]], _source_manifest())

    class WrongProvider(_QueryProvider):
        provider_name = "other-provider"

    with pytest.raises(ArtifactCompatibilityError, match="provider"):
        DenseRetriever(WrongProvider(), backend).search(_query())

    class WrongProviderVersion(_QueryProvider):
        provider_version = "2.0"

    with pytest.raises(ArtifactCompatibilityError, match="provider version"):
        DenseRetriever(WrongProviderVersion(), backend).search(_query())

    class WrongModel(_QueryProvider):
        model_name = "fixture/other"

    with pytest.raises(ArtifactCompatibilityError, match="model"):
        DenseRetriever(WrongModel(), backend).search(_query())

    class WrongRevision(_QueryProvider):
        model_revision = "other-revision"

    with pytest.raises(ArtifactCompatibilityError, match="revision"):
        DenseRetriever(WrongRevision(), backend).search(_query())

    class WrongDimension(_QueryProvider):
        dimension = 3

    with pytest.raises(ArtifactCompatibilityError, match="dimension"):
        DenseRetriever(WrongDimension(), backend).search(_query())


def test_dense_retriever_rejects_wrong_strategy_before_embedding() -> None:
    """A routed BM25 request never triggers an expensive query model call."""
    provider = _QueryProvider()
    query = _query().model_copy(
        update={"requested_strategy": RetrievalStrategy.BM25}
    )

    with pytest.raises(RetrievalError, match="non-dense"):
        DenseRetriever(provider, object()).search(query)  # type: ignore[arg-type]
    assert provider.queries == []
