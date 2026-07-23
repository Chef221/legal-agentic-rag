"""Unit tests for raw dataset audit findings."""

from datetime import UTC, datetime

from legal_agentic_rag.configuration import DatasetAuditConfig
from legal_agentic_rag.offline.audit import DatasetAuditService
from legal_agentic_rag.schemas import DatasetManifest


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        schema_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture",
        loaded_at=datetime(2026, 7, 18, tzinfo=UTC),
        configs=["metadata", "content", "relationships"],
        record_counts={"metadata": 3, "content": 3, "relationships": 6},
        processing_config_hash="fixture-hash",
    )


def test_audit_measures_schema_identity_join_and_relationships(
    load_raw_aio_fixture,
) -> None:
    """A small raw fixture exercises every required audit boundary."""
    service = DatasetAuditService(
        DatasetAuditConfig(
            known_relationship_labels={
                "Sửa đổi",
                "Được sửa đổi bởi",
                "Liên quan",
                "Hướng dẫn",
                "Được hướng dẫn bởi",
            }
        ),
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC),
    )

    report = service.audit(
        metadata_records=load_raw_aio_fixture("metadata"),
        content_records=load_raw_aio_fixture("content"),
        relationship_records=load_raw_aio_fixture("relationships"),
        manifest=_manifest(),
    )

    assert report.components["metadata"].total_records == 3
    assert len(report.audit_config_hash) == 64
    assert report.components["metadata"].duplicate_ids == 1
    assert report.components["content"].duplicate_ids == 1
    assert report.joins.metadata_with_content == 1
    assert report.joins.metadata_without_content == 1
    assert report.joins.orphan_content_ids == 1
    assert report.joins.invalid_relationship_sources == 1
    assert report.joins.invalid_relationship_targets == 1
    assert report.effect_status_distribution["Còn hiệu lực"] == 2
    assert report.relationship_distribution["Sửa đổi"] == 2
    issue_types = {issue.issue_type for issue in report.issues}
    assert {
        "duplicate_id",
        "duplicate_content",
        "missing_content",
        "orphan_content",
        "invalid_relationship_source",
        "invalid_relationship_target",
        "duplicate_edge",
        "reciprocal_edge",
        "self_loop",
        "invalid_date",
        "expiry_before_effective",
        "navigation_only_content",
    } <= issue_types
    metadata_fields = {
        profile.field_name for profile in report.components["metadata"].field_profiles
    }
    assert {"id", "title", "so_ky_hieu"} <= metadata_fields


def test_audit_detects_missing_fields_empty_and_malformed_ids() -> None:
    """Incompatible raw records are reported instead of silently skipped."""
    service = DatasetAuditService(
        DatasetAuditConfig(minimum_content_characters=0),
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC),
    )

    report = service.audit(
        metadata_records=[{"id": ""}, {"id": {"nested": "bad"}}],
        content_records=[{"id": "doc-1"}],
        relationship_records=[
            {"doc_id": None, "other_doc_id": "doc-1", "relationship": ""}
        ],
        manifest=_manifest(),
    )

    issue_types = [issue.issue_type for issue in report.issues]
    assert "missing_required_field" in issue_types
    assert "empty_id" in issue_types
    assert "malformed_id" in issue_types
    assert "empty_content" in issue_types
    assert "empty_relationship_endpoint" in issue_types
    assert "empty_relationship_label" in issue_types
