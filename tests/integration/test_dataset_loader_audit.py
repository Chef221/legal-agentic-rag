"""Local end-to-end test from AIO source through persisted audit reports."""

from datetime import UTC, datetime

from legal_agentic_rag.configuration import DatasetSourceConfig
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.offline.audit import DatasetAuditReportWriter, DatasetAuditService
from legal_agentic_rag.offline.datasets.aio import AioDatasetSource


def test_local_fixture_load_audit_and_persist(tmp_path, load_raw_aio_fixture) -> None:
    """Milestone 2 works end-to-end without network or a full dataset download."""
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
            streaming=True,
        ),
        load_dataset=load_dataset,
        clock=lambda: datetime(2026, 7, 18, tzinfo=UTC),
    )
    metadata = list(source.iter_records(DatasetComponent.METADATA))
    content = list(source.iter_records(DatasetComponent.CONTENT))
    relationships = list(source.iter_records(DatasetComponent.RELATIONSHIPS))

    report = DatasetAuditService(
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC)
    ).audit(
        metadata_records=metadata,
        content_records=content,
        relationship_records=relationships,
        manifest=source.dataset_manifest(),
    )
    paths = DatasetAuditReportWriter().write(report, tmp_path / "audit")

    assert report.dataset_manifest.record_counts == {
        "metadata": 3,
        "content": 3,
        "relationships": 6,
    }
    assert all(path.is_file() for path in paths.values())
