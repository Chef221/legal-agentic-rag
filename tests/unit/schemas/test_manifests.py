"""Tests for dataset, artifact, and validation manifests."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    ArtifactValidationResult,
    DatasetManifest,
)
from legal_agentic_rag.schemas import BuildValidationReport, OfflineBuildState


def test_dataset_manifest_parses_timezone_and_counts(load_schema_sample: object) -> None:
    """Dataset ingestion provenance must be unambiguous and reproducible."""
    data = load_schema_sample("valid_dataset_manifest.json")  # type: ignore[operator]
    manifest = DatasetManifest.model_validate(data)

    assert manifest.loaded_at.utcoffset() is not None
    assert manifest.record_counts["metadata"] == 1


def test_dataset_manifest_rejects_naive_datetime(load_schema_sample: object) -> None:
    """Persisted datetimes must include timezone information."""
    data = load_schema_sample("valid_dataset_manifest.json")  # type: ignore[operator]
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate({**data, "loaded_at": "2026-07-15T08:00:00"})


def test_artifact_manifest_uses_expanded_contract(load_schema_sample: object) -> None:
    """Processed artifacts include dataset and processing-config provenance."""
    data = load_schema_sample("valid_artifact_manifest.json")  # type: ignore[operator]
    manifest = ArtifactManifest.model_validate(data)

    assert manifest.artifact_type.value == "legal_chunks"
    assert manifest.dataset_name == "th1nhng0/vietnamese-legal-documents"


def test_artifact_revision_requires_model_name(load_schema_sample: object) -> None:
    """A model revision without model identity is not reproducible."""
    data = load_schema_sample("valid_artifact_manifest.json")  # type: ignore[operator]
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate({**data, "model_revision": "revision-1"})


def test_artifact_validation_flag_matches_errors(load_schema_sample: object) -> None:
    """An artifact cannot be valid while validation errors are present."""
    manifest = ArtifactManifest.model_validate(
        load_schema_sample("valid_artifact_manifest.json")  # type: ignore[operator]
    )
    valid = ArtifactValidationResult(
        manifest=manifest,
        is_valid=True,
        checked_at=datetime.now(timezone.utc),
    )
    assert valid.errors == []

    with pytest.raises(ValidationError):
        ArtifactValidationResult(
            manifest=manifest,
            is_valid=True,
            checked_at=datetime.now(timezone.utc),
            errors=["count mismatch"],
        )


def test_build_validation_report_requires_an_identified_failure() -> None:
    """Top-level build validity cannot contradict artifact validation results."""
    with pytest.raises(ValidationError):
        BuildValidationReport(
            checked_at=datetime.now(timezone.utc),
            is_full_corpus=False,
            is_valid=False,
        )
    report = BuildValidationReport(
        checked_at=datetime.now(timezone.utc),
        is_full_corpus=False,
        is_valid=False,
        errors=["artifact set is incomplete"],
    )
    assert report.is_valid is False


def test_offline_build_state_requires_sha256_and_timezone() -> None:
    """Resume identity rejects ambiguous timestamps and non-SHA config values."""
    state = OfflineBuildState(
        application_config_hash="a" * 64,
        code_version="0.19.1",
        created_at=datetime.now(timezone.utc),
    )
    assert state.schema_version == "1.1"
    with pytest.raises(ValidationError):
        OfflineBuildState(
            application_config_hash="not-a-sha",
            code_version="0.19.1",
            created_at=datetime.now(timezone.utc),
        )
