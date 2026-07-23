"""Versioned normalized-relationship artifact persistence."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from pydantic import ValidationError

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.schemas.legal_relationships import LegalRelationship
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

RELATIONSHIPS_FILENAME = "relationships.jsonl"
MANIFEST_FILENAME = "manifest.json"


def persist_relationship_artifact(
    *,
    relationships: list[LegalRelationship],
    destination: Path,
    manifest: ArtifactManifest,
) -> ArtifactManifest:
    """Persist normalized relationships without replacing an existing artifact."""
    destination = destination.resolve()
    if destination.exists():
        raise BackendInitializationError(
            "Relationship artifact destination already exists"
        )
    if not destination.parent.exists():
        raise BackendInitializationError(
            "Relationship artifact parent directory does not exist"
        )
    if (
        manifest.artifact_type != ArtifactType.RELATIONSHIP_MAPPING
        or manifest.record_count != len(relationships)
    ):
        raise ArtifactCompatibilityError(
            "Relationship manifest is incompatible with the payload"
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    destination_created = False
    try:
        relationships_path = temporary / RELATIONSHIPS_FILENAME
        with relationships_path.open("w", encoding="utf-8", newline="\n") as stream:
            for relationship in relationships:
                stream.write(relationship.model_dump_json())
                stream.write("\n")
        metadata = dict(manifest.metadata)
        metadata.update(
            {
                "relationships_filename": RELATIONSHIPS_FILENAME,
                "relationships_sha256": _sha256_file(relationships_path),
                "manifest_filename": MANIFEST_FILENAME,
            }
        )
        final_manifest = manifest.model_copy(update={"metadata": metadata})
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(
                final_manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        destination.mkdir(exist_ok=False)
        destination_created = True
        for staged_file in temporary.iterdir():
            staged_file.replace(destination / staged_file.name)
        temporary.rmdir()
        return final_manifest
    except (OSError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise BackendInitializationError(
            "Relationship artifact could not be persisted"
        ) from error
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def load_relationship_artifact(
    *,
    source: Path,
    supplied_manifest: ArtifactManifest,
) -> tuple[list[LegalRelationship], ArtifactManifest]:
    """Load a checksum-validated normalized relationship artifact."""
    source = source.resolve()
    relationships_path = source / RELATIONSHIPS_FILENAME
    manifest_path = source / MANIFEST_FILENAME
    if not source.is_dir() or not all(
        path.is_file() for path in (relationships_path, manifest_path)
    ):
        raise ArtifactCompatibilityError(
            "Relationship artifact files are incomplete"
        )
    try:
        stored_manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Relationship artifact manifest is invalid"
        ) from error
    if stored_manifest != supplied_manifest:
        raise ArtifactCompatibilityError(
            "Supplied relationship manifest does not match persisted manifest"
        )
    if stored_manifest.artifact_type != ArtifactType.RELATIONSHIP_MAPPING:
        raise ArtifactCompatibilityError(
            "Artifact is not a normalized relationship mapping"
        )
    expected_checksum = stored_manifest.metadata.get("relationships_sha256")
    if (
        not isinstance(expected_checksum, str)
        or _sha256_file(relationships_path) != expected_checksum
    ):
        raise ArtifactCompatibilityError(
            "Relationship artifact checksum does not match"
        )
    try:
        relationships = [
            LegalRelationship.model_validate_json(line)
            for line in relationships_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Relationship artifact payload is invalid"
        ) from error
    identities = [_relationship_identity(item) for item in relationships]
    if (
        len(relationships) != stored_manifest.record_count
        or len(identities) != len(set(identities))
        or identities != sorted(identities)
    ):
        raise ArtifactCompatibilityError(
            "Relationship artifact ordering or record count is incompatible"
        )
    return relationships, stored_manifest


def _relationship_identity(
    relationship: LegalRelationship,
) -> tuple[str, str, str, str]:
    return (
        relationship.source_document_id,
        relationship.target_document_id,
        relationship.raw_relationship,
        relationship.relationship_type or "",
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
