"""Integration test from legal chunks through persisted BM25 retrieval."""

from datetime import UTC, datetime
from pathlib import Path

from legal_agentic_rag.configuration import ChunkingConfig
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.offline.chunking import LegalChunker
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalBlock,
    LegalBlockType,
    LegalDocument,
    LegalStructure,
    RetrievalQuery,
    RetrievalStrategy,
)


def test_legal_chunk_artifact_builds_persists_and_queries_bm25(
    tmp_path: Path,
) -> None:
    """Milestone 6 output is directly consumable by the BM25 backend."""
    document = LegalDocument(
        document_id="doc-1",
        title="Luật giao thông đường bộ",
        document_number="01/2026/QH",
        document_type="Luật",
        effect_status="Còn hiệu lực",
        legal_field="Giao thông",
        clean_text="fixture",
        has_content=True,
        source_dataset="aio",
    )
    structure = LegalStructure(
        article_number="5",
        article_title="Tốc độ xe",
        structure_path=["Điều 5"],
    )
    block = LegalBlock(
        block_id="block-1",
        document_id="doc-1",
        block_type=LegalBlockType.ARTICLE,
        block_number="5",
        title="Tốc độ xe",
        text="Điều 5. Người lái xe không được chạy quá tốc độ quy định.",
        order_index=0,
        structure=structure,
    )
    block_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_BLOCKS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        record_count=1,
        processing_config_hash="parser-hash",
    )
    chunked = LegalChunker(
        ChunkingConfig(max_tokens=100, min_tokens=1, overlap_tokens=10)
    ).chunk(
        documents=[document], blocks=[block], source_manifest=block_manifest
    )
    backend = SQLiteFTS5BM25Backend()
    backend.build(chunked.chunks, chunked.manifest)
    destination = tmp_path / "bm25-v1"
    manifest = backend.persist(destination)
    reloaded = SQLiteFTS5BM25Backend()
    reloaded.load(destination, manifest)

    response = reloaded.search(
        RetrievalQuery(
            query_id="query-1",
            original_question="chạy quá tốc độ",
            normalized_question="chạy quá tốc độ",
            requested_strategy=RetrievalStrategy.BM25,
        )
    )

    assert len(response.hits) == 1
    assert response.hits[0].chunk_id == chunked.chunks[0].chunk_id
    assert response.hits[0].metadata["document_number"] == "01/2026/QH"
    assert response.hits[0].metadata["structure"]["article_number"] == "5"
    assert response.artifact_versions == {"bm25_index": "1.0"}
