"""Local integration from normalized HTML through cleaning and parsing."""

from datetime import UTC, datetime

from legal_agentic_rag.offline.cleaning import LegalHtmlCleaner
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalBlockType,
    LegalDocument,
)


def test_cleaned_document_flows_into_legal_blocks() -> None:
    """The parser consumes unified clean text and retains legal hierarchy."""
    normalized_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=1,
        processing_config_hash="normalization-hash",
    )
    document = LegalDocument(
        document_id="doc-1",
        content_html=(
            "<h2>Chương I</h2><h3>QUY ĐỊNH CHUNG</h3>"
            "<p>Điều 1. Phạm vi áp dụng</p>"
            "<p>1. Không áp dụng mức 100 đồng.</p>"
        ),
        has_content=True,
        source_dataset="fixture",
    )
    cleaning_result = LegalHtmlCleaner(
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC)
    ).clean(documents=[document], source_manifest=normalized_manifest)

    parsing_result = LegalStructureParser(
        clock=lambda: datetime(2026, 7, 18, 2, tzinfo=UTC)
    ).parse(
        documents=cleaning_result.documents,
        source_manifest=cleaning_result.manifest,
    )

    assert [block.block_type for block in parsing_result.blocks] == [
        LegalBlockType.CHAPTER,
        LegalBlockType.ARTICLE,
        LegalBlockType.CLAUSE,
    ]
    assert parsing_result.blocks[1].structure.chapter == "Chương I"
    assert parsing_result.blocks[2].structure.article_number == "1"
    assert "Không áp dụng mức 100 đồng" in parsing_result.blocks[2].text
    assert parsing_result.diagnostics[0].text_coverage == 1.0
    assert parsing_result.manifest.artifact_type == ArtifactType.LEGAL_BLOCKS
