"""Local integration from unified legal documents through HTML cleaning."""

from datetime import UTC, datetime

from legal_agentic_rag.offline.cleaning import LegalHtmlCleaner
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalDocument,
)


def test_normalized_document_flows_into_cleaned_artifact() -> None:
    """The cleaner consumes only unified documents, never raw source fields."""
    source_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture-corpus",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=1,
        processing_config_hash="dataset-hash",
    )
    documents = [
        LegalDocument(
            document_id="doc-1",
            title="Luật mẫu",
            content_html=(
                "<style>bad</style><p>Điều 1. Không áp dụng mức 100 đồng.</p>"
            ),
            has_content=True,
            source_dataset="fixture-corpus",
        )
    ]

    cleaning_result = LegalHtmlCleaner(
        clock=lambda: datetime(2026, 7, 18, 2, tzinfo=UTC)
    ).clean(
        documents=documents,
        source_manifest=source_manifest,
    )

    assert cleaning_result.documents[0].clean_text == (
        "Điều 1. Không áp dụng mức 100 đồng."
    )
    assert cleaning_result.documents[0].content_html is not None
    assert cleaning_result.manifest.artifact_type == ArtifactType.CLEANED_DOCUMENTS
    assert (
        cleaning_result.manifest.metadata["source_processing_config_hash"]
        == source_manifest.processing_config_hash
    )
