"""Tests for persisted JSON and focused CSV audit outputs."""

import csv
from datetime import UTC, datetime
import json

import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.offline.audit import DatasetAuditReportWriter, DatasetAuditService
from legal_agentic_rag.schemas import DatasetManifest


def test_writer_creates_all_outputs_and_refuses_silent_overwrite(
    tmp_path, load_raw_aio_fixture
) -> None:
    """The required audit artifacts are UTF-8 and protected from replacement."""
    manifest = DatasetManifest(
        schema_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture",
        loaded_at=datetime(2026, 7, 18, tzinfo=UTC),
        configs=["metadata", "content", "relationships"],
        record_counts={"metadata": 3, "content": 3, "relationships": 6},
        processing_config_hash="fixture-hash",
    )
    report = DatasetAuditService(
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC)
    ).audit(
        metadata_records=load_raw_aio_fixture("metadata"),
        content_records=load_raw_aio_fixture("content"),
        relationship_records=load_raw_aio_fixture("relationships"),
        manifest=manifest,
    )
    writer = DatasetAuditReportWriter()

    paths = writer.write(report, tmp_path / "audit")

    assert set(paths) == {
        "data_audit.json",
        "missing_content.csv",
        "orphan_content.csv",
        "invalid_relationships.csv",
        "duplicate_records.csv",
    }
    payload = json.loads(paths["data_audit.json"].read_text(encoding="utf-8"))
    assert payload["dataset_manifest"]["dataset_revision"] == "fixture"
    with paths["invalid_relationships.csv"].open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["reason"] for row in rows} >= {
        "invalid_relationship_source",
        "invalid_relationship_target",
    }
    with pytest.raises(ArtifactCompatibilityError):
        writer.write(report, tmp_path / "audit")
