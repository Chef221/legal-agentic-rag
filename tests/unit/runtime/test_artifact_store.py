"""Tests for runtime-owned JSONL artifact persistence and validation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.runtime import (
    ModelArtifactWriter,
    load_artifact_manifest,
    load_model_artifact,
    persist_model_artifact,
    stream_model_artifact,
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
    records, loaded_with_records = load_model_artifact(
        destination,
        expected_type=ArtifactType.NORMALIZED_DOCUMENTS,
        record_type=LegalDocument,
    )
    assert loaded_with_records == stored
    assert records[0].document_id == "doc-1"
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


def test_streaming_artifact_consumes_one_pass_and_publishes_at_finalize(
    tmp_path: Path,
) -> None:
    """Streaming persistence never materializes or publishes partial records."""
    destination = tmp_path / "streamed"
    manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
        record_count=3,
        processing_config_hash="runtime-hash",
    )
    iteration_count = 0

    def records():
        nonlocal iteration_count
        iteration_count += 1
        for index in range(3):
            yield LegalDocument(
                document_id=f"doc-{index}",
                has_content=False,
                source_dataset="fixture",
            )

    stored = persist_model_artifact(
        records=records(),
        destination=destination,
        manifest=manifest,
    )
    streamed, loaded = stream_model_artifact(
        destination,
        expected_type=ArtifactType.NORMALIZED_DOCUMENTS,
        record_type=LegalDocument,
    )

    assert iteration_count == 1
    assert loaded == stored
    assert [record.document_id for record in streamed] == [
        "doc-0",
        "doc-1",
        "doc-2",
    ]


def test_streaming_writer_discards_invalid_partial_artifact(
    tmp_path: Path,
) -> None:
    """A failed finalize never exposes an incomplete destination."""
    destination = tmp_path / "partial"
    manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
        record_count=2,
        processing_config_hash="runtime-hash",
    )

    with pytest.raises(DataValidationError, match="record count"):
        with ModelArtifactWriter(destination) as writer:
            writer.write(
                LegalDocument(
                    document_id="doc-1",
                    has_content=False,
                    source_dataset="fixture",
                )
            )
            writer.finalize(manifest)

    assert not destination.exists()
