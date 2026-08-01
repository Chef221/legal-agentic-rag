"""Tests for persisted normalized relationship artifacts."""

from datetime import UTC, datetime

import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.offline.relationships import (
    load_relationship_artifact,
    persist_relationship_artifact,
)
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalRelationship,
)


def _relationships() -> list[LegalRelationship]:
    return [
        LegalRelationship(
            source_document_id="doc-1",
            target_document_id="doc-2",
            relationship_type="amends",
            raw_relationship="Sửa đổi",
            source_dataset="fixture-corpus",
        )
    ]


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.RELATIONSHIP_MAPPING,
        artifact_version="1.0",
        dataset_name="fixture",
        dataset_revision="revision",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        record_count=1,
        processing_config_hash="relationship-hash",
    )


def test_relationship_artifact_round_trip_and_checksum(tmp_path: object) -> None:
    """Persisted mappings reload exactly and reject payload tampering."""
    destination = tmp_path / "relationships"
    persisted = persist_relationship_artifact(
        relationships=_relationships(),
        destination=destination,
        manifest=_manifest(),
    )

    loaded, loaded_manifest = load_relationship_artifact(
        source=destination,
        supplied_manifest=persisted,
    )

    assert loaded == _relationships()
    assert loaded_manifest == persisted
    relationships_path = destination / "relationships.jsonl"
    relationships_path.write_text(
        relationships_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactCompatibilityError, match="checksum"):
        load_relationship_artifact(
            source=destination,
            supplied_manifest=persisted,
        )
