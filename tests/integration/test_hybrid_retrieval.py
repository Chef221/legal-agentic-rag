"""Integration test for persisted BM25 and vector artifacts with RRF."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.vector import NumpyVectorBackend, VectorIndexBuilder
from legal_agentic_rag.retrieval import DenseRetriever, FixedRetriever
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    RetrievalQuery,
    RetrievalStrategy,
)


class _KeywordEmbeddingProvider:
    provider_name = "fixture-provider"
    provider_version = "1.0"
    model_name = "fixture/semantic"
    model_revision = "fixture-revision"
    dimension = 4

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        value = text.casefold()
        return [
            float(any(term in value for term in ("tốc độ", "chạy quá"))),
            float("giấy phép" in value),
            float(any(term in value for term in ("thuế", "doanh nghiệp"))),
            0.1,
        ]


def _chunk(chunk_id: str, document_id: str, text: str, legal_field: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        search_text=f"Lĩnh vực: {legal_field}\nNội dung: {text}",
        token_count=len(text.split()),
        document_title=f"Văn bản {document_id}",
        document_number=f"{document_id}/2026/QH",
        document_type="Luật",
        effect_status="Còn hiệu lực",
        legal_field=legal_field,
        source_dataset="fixture",
        metadata={"source_block_ids": [f"block-{chunk_id}"]},
    )


def test_persisted_indexes_run_fixed_hybrid_rrf(tmp_path: Path) -> None:
    """Both real reference backends compose into traceable hybrid retrieval."""
    chunks = [
        _chunk(
            "chunk-traffic",
            "doc-traffic",
            "Người lái xe không được chạy quá tốc độ quy định.",
            "Giao thông",
        ),
        _chunk(
            "chunk-license",
            "doc-license",
            "Người điều khiển xe phải có giấy phép lái xe.",
            "Giao thông",
        ),
        _chunk(
            "chunk-tax",
            "doc-tax",
            "Doanh nghiệp phải nộp thuế đúng thời hạn.",
            "Thuế",
        ),
    ]
    source = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        record_count=len(chunks),
        processing_config_hash="shared-chunk-hash",
    )
    bm25 = SQLiteFTS5BM25Backend()
    bm25.build(chunks, source)
    bm25_manifest = bm25.persist(tmp_path / "bm25")
    loaded_bm25 = SQLiteFTS5BM25Backend()
    loaded_bm25.load(tmp_path / "bm25", bm25_manifest)

    provider = _KeywordEmbeddingProvider()
    vector = NumpyVectorBackend()
    VectorIndexBuilder(provider, vector).build(chunks, source)
    vector_manifest = vector.persist(tmp_path / "vector")
    loaded_vector = NumpyVectorBackend()
    loaded_vector.load(tmp_path / "vector", vector_manifest)

    response = FixedRetriever(
        loaded_bm25,
        DenseRetriever(provider, loaded_vector),
    ).search(
        RetrievalQuery(
            query_id="query-hybrid",
            original_question="Mức phạt chạy xe nhanh?",
            normalized_question="mức phạt chạy quá tốc độ xe",
            top_k=2,
            candidate_k=3,
            requested_strategy=RetrievalStrategy.HYBRID,
        )
    )

    assert response.strategy == RetrievalStrategy.HYBRID
    assert response.hits[0].chunk_id == "chunk-traffic"
    assert response.hits[0].retrieval_trace.bm25_rank is not None
    assert response.hits[0].retrieval_trace.dense_rank is not None
    assert response.hits[0].retrieval_trace.rrf_score == response.hits[0].score
    assert response.artifact_versions == {
        "bm25_index": "1.0",
        "vector_index": "1.0",
    }
