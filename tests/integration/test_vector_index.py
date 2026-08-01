"""Integration from legal chunks to persisted dense retrieval."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from legal_agentic_rag.indexing.vector import NumpyVectorBackend, VectorIndexBuilder
from legal_agentic_rag.retrieval import DenseRetriever
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    LegalStructure,
    RetrievalQuery,
    RetrievalStrategy,
)


class _KeywordEmbeddingProvider:
    provider_name = "fixture-provider"
    provider_version = "1.0"
    model_name = "fixture/semantic-vietnamese"
    model_revision = "fixture-revision"
    dimension = 3

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        return [
            float(any(term in normalized for term in ("xe", "tốc độ", "giao thông"))),
            float(any(term in normalized for term in ("thuế", "doanh nghiệp"))),
            0.1,
        ]


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    legal_field: str,
) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        search_text=f"Lĩnh vực: {legal_field}\nNội dung: {text}",
        token_count=len(text.split()),
        structure=LegalStructure(article_number="1", structure_path=["Điều 1"]),
        document_title=f"Văn bản {document_id}",
        document_number=f"{document_id}/2026/QH",
        document_type="Luật",
        effect_status="Còn hiệu lực",
        legal_field=legal_field,
        source_dataset="fixture-corpus",
        metadata={"source_block_ids": [f"block-{chunk_id}"]},
    )


def test_chunks_embed_persist_reload_and_dense_retrieve(tmp_path: Path) -> None:
    """Offline and online vector phases compose through approved contracts."""
    chunks = [
        _chunk(
            "chunk-traffic",
            "doc-traffic",
            "Người lái xe không được chạy quá tốc độ.",
            "Giao thông",
        ),
        _chunk(
            "chunk-tax",
            "doc-tax",
            "Doanh nghiệp phải nộp thuế đúng thời hạn.",
            "Thuế",
        ),
    ]
    source_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        record_count=2,
        processing_config_hash="chunk-hash",
    )
    provider = _KeywordEmbeddingProvider()
    backend = NumpyVectorBackend()
    VectorIndexBuilder(provider, backend).build(chunks, source_manifest)
    destination = tmp_path / "vector-v1"
    manifest = backend.persist(destination)
    loaded = NumpyVectorBackend()
    loaded.load(destination, manifest)

    response = DenseRetriever(provider, loaded).search(
        RetrievalQuery(
            query_id="query-1",
            original_question="Mức phạt khi chạy xe nhanh?",
            normalized_question="mức phạt chạy quá tốc độ xe",
            requested_strategy=RetrievalStrategy.DENSE,
        )
    )

    assert response.hits[0].chunk_id == "chunk-traffic"
    assert response.hits[0].metadata["legal_field"] == "Giao thông"
    assert response.hits[0].retrieval_trace.dense_rank == 1
    assert response.artifact_versions == {"vector_index": "1.0"}
