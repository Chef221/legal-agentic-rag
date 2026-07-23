"""Local integration from cleaning through parsing into legal chunks."""

from datetime import UTC, datetime

from legal_agentic_rag.configuration import ChunkingConfig
from legal_agentic_rag.offline.chunking import LegalChunker
from legal_agentic_rag.offline.cleaning import LegalHtmlCleaner
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalDocument,
)


def test_html_flows_through_parser_into_article_chunk() -> None:
    """Milestones 4–6 compose locally using only unified contracts."""
    normalized_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=1,
        processing_config_hash="normalization-hash",
    )
    document = LegalDocument(
        document_id="doc-1",
        title="Luật mẫu",
        document_number="01/2026/QH",
        content_html=(
            "<h2>Chương I</h2><h3>QUY ĐỊNH CHUNG</h3>"
            "<p>Điều 1. Phạm vi áp dụng</p>"
            "<p>1. Không áp dụng mức 100 đồng.</p>"
        ),
        has_content=True,
        source_dataset="fixture",
    )
    cleaned = LegalHtmlCleaner().clean(
        documents=[document], source_manifest=normalized_manifest
    )
    parsed = LegalStructureParser().parse(
        documents=cleaned.documents, source_manifest=cleaned.manifest
    )

    chunked = LegalChunker(
        ChunkingConfig(max_tokens=100, min_tokens=1, overlap_tokens=10)
    ).chunk(
        documents=parsed.documents,
        blocks=parsed.blocks,
        source_manifest=parsed.manifest,
    )

    article_chunks = [
        chunk
        for chunk in chunked.chunks
        if chunk.structure.article_number == "1"
    ]
    assert len(article_chunks) == 1
    assert article_chunks[0].metadata["chunk_strategy"] == "article"
    assert "Không áp dụng mức 100 đồng" in article_chunks[0].text
    assert article_chunks[0].document_title == "Luật mẫu"
    assert article_chunks[0].document_number == "01/2026/QH"
    assert chunked.diagnostics[0].block_coverage == 1.0
    assert chunked.manifest.artifact_type == ArtifactType.LEGAL_CHUNKS
