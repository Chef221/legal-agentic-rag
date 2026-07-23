"""Tests for runtime-owned JSONL artifact persistence and validation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.runtime import (
    load_artifact_manifest,
    persist_model_artifact,
)
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalDocument,
)


def test_model_artifact_roundtrip_verifies_manifest_and_payload_hash(
    tmp_path: Path,
) -> None:
    """Runtime persistence records a checksum and rejects payload mutation."""
    destination = tmp_path / "normalized"
    manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
        record_count=1,
        processing_config_hash="runtime-hash",
    )

    stored = persist_model_artifact(
        records=[
            LegalDocument(
                document_id="doc-1",
                has_content=False,
                source_dataset="fixture",
            )
        ],
        destination=destination,
        manifest=manifest,
    )
    loaded = load_artifact_manifest(
        destination,
        expected_type=ArtifactType.NORMALIZED_DOCUMENTS,
        verify_payload=True,
    )

    assert loaded == stored
    assert loaded.metadata["record_model"] == "LegalDocument"
    with (destination / "records.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ArtifactCompatibilityError, match="checksum"):
        load_artifact_manifest(
            destination,
            expected_type=ArtifactType.NORMALIZED_DOCUMENTS,
            verify_payload=True,
        )


def test_model_artifact_never_overwrites_existing_destination(
    tmp_path: Path,
) -> None:
    """An immutable processed artifact cannot be silently replaced."""
    destination = tmp_path / "chunks"
    destination.mkdir()
    manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
        record_count=0,
        processing_config_hash="runtime-hash",
    )

    with pytest.raises(ArtifactCompatibilityError, match="already"):
        persist_model_artifact(
            records=[],
            destination=destination,
            manifest=manifest,
        )
