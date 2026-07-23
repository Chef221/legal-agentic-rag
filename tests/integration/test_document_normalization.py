"""Local integration from the dataset source to unified legal documents."""

from datetime import UTC, datetime

from legal_agentic_rag.configuration import DatasetSourceConfig
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.offline.datasets.aio import (
    AioDatasetSource,
    AioDocumentNormalizer,
)


def test_fixture_source_normalizes_without_network(load_raw_aio_fixture) -> None:
    """Loader output crosses the raw boundary into LegalDocument contracts."""
    components = {
        "metadata": load_raw_aio_fixture("metadata"),
        "content": load_raw_aio_fixture("content"),
        "relationships": load_raw_aio_fixture("relationships"),
    }

    def load_dataset(**kwargs: object) -> list[dict[str, object]]:
        return components[str(kwargs["name"])]

    source = AioDatasetSource(
        DatasetSourceConfig(
            dataset_name="th1nhng0/vietnamese-legal-documents",
            dataset_revision="fixture-revision",
        ),
        load_dataset=load_dataset,
        clock=lambda: datetime(2026, 7, 18, tzinfo=UTC),
    )
    metadata = list(source.iter_records(DatasetComponent.METADATA))
    content = list(source.iter_records(DatasetComponent.CONTENT))
    list(source.iter_records(DatasetComponent.RELATIONSHIPS))

    result = AioDocumentNormalizer(
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC)
    ).normalize(
        metadata_records=metadata,
        content_records=content,
        dataset_manifest=source.dataset_manifest(),
    )

    assert [document.document_id for document in result.documents] == ["doc-2"]
    assert result.documents[0].has_content is False
    assert result.documents[0].raw_metadata["title"] == "Văn bản thiếu nội dung"
    assert result.manifest.dataset_revision == "fixture-revision"
    assert result.manifest.metadata["input_metadata_count"] == 3
