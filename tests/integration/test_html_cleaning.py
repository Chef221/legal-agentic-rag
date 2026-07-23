"""Local integration from AIO normalization through legal HTML cleaning."""

from datetime import UTC, datetime

from legal_agentic_rag.offline.cleaning import LegalHtmlCleaner
from legal_agentic_rag.offline.datasets.aio import AioDocumentNormalizer
from legal_agentic_rag.schemas import ArtifactType, DatasetManifest


def test_normalized_document_flows_into_cleaned_artifact() -> None:
    """The cleaner consumes only unified documents, never raw AIO field names."""
    dataset_manifest = DatasetManifest(
        schema_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture-revision",
        loaded_at=datetime(2026, 7, 18, tzinfo=UTC),
        configs=["metadata", "content", "relationships"],
        record_counts={"metadata": 1, "content": 1, "relationships": 0},
        processing_config_hash="dataset-hash",
    )
    normalization_result = AioDocumentNormalizer(
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC)
    ).normalize(
        metadata_records=[{"id": "doc-1", "title": "Luật mẫu"}],
        content_records=[
            {
                "id": "doc-1",
                "content_html": (
                    "<style>bad</style><p>Điều 1. Không áp dụng mức 100 đồng.</p>"
                ),
            }
        ],
        dataset_manifest=dataset_manifest,
    )

    cleaning_result = LegalHtmlCleaner(
        clock=lambda: datetime(2026, 7, 18, 2, tzinfo=UTC)
    ).clean(
        documents=normalization_result.documents,
        source_manifest=normalization_result.manifest,
    )

    assert cleaning_result.documents[0].clean_text == (
        "Điều 1. Không áp dụng mức 100 đồng."
    )
    assert cleaning_result.documents[0].content_html is not None
    assert cleaning_result.manifest.artifact_type == ArtifactType.CLEANED_DOCUMENTS
    assert (
        cleaning_result.manifest.metadata["source_processing_config_hash"]
        == normalization_result.manifest.processing_config_hash
    )
