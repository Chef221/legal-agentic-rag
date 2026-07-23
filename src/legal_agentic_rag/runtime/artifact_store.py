"""Persistence and validation for runtime-owned processed artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    ArtifactType,
    DatasetManifest,
)

_MANIFEST_FILENAME = "manifest.json"
_RECORDS_FILENAME = "records.jsonl"
_DATASET_MANIFEST_FILENAME = "dataset_manifest.json"


def persist_dataset_manifest(
    manifest: DatasetManifest,
    root: Path,
) -> Path:
    """Persist dataset provenance once without silently overwriting it."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / _DATASET_MANIFEST_FILENAME
    if path.exists():
        raise ArtifactCompatibilityError("Dataset manifest already exists")
    path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def persist_model_artifact(
    *,
    records: Sequence[BaseModel],
    destination: Path,
    manifest: ArtifactManifest,
) -> ArtifactManifest:
    """Persist typed records as deterministic JSONL with manifest checksum."""
    values = list(records)
    if manifest.record_count != len(values):
        raise DataValidationError(
            "Artifact manifest record count does not match payload"
        )
    record_models = {type(record) for record in values}
    if len(record_models) > 1:
        raise DataValidationError(
            "Artifact payload must contain one record model"
        )
    if destination.exists():
        raise ArtifactCompatibilityError("Artifact destination already exists")
    destination.mkdir(parents=True)
    payload_path = destination / _RECORDS_FILENAME
    with payload_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in values:
            stream.write(
                json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    metadata = dict(manifest.metadata)
    metadata.update(
        {
            "payload_file": _RECORDS_FILENAME,
            "payload_sha256": _sha256_file(payload_path),
            "record_model": (
                next(iter(record_models)).__name__ if values else "empty"
            ),
        }
    )
    stored_manifest = manifest.model_copy(update={"metadata": metadata})
    (destination / _MANIFEST_FILENAME).write_text(
        json.dumps(
            stored_manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return stored_manifest


def load_artifact_manifest(
    source: Path,
    *,
    expected_type: ArtifactType,
    verify_payload: bool = False,
) -> ArtifactManifest:
    """Load a manifest and optionally verify its runtime-owned JSONL payload."""
    manifest_path = source / _MANIFEST_FILENAME
    try:
        manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise ArtifactCompatibilityError(
            "Artifact manifest is missing or invalid"
        ) from error
    if manifest.artifact_type != expected_type:
        raise ArtifactCompatibilityError("Artifact type is incompatible")
    if verify_payload:
        payload_file = manifest.metadata.get("payload_file")
        payload_hash = manifest.metadata.get("payload_sha256")
        if (
            not isinstance(payload_file, str)
            or not isinstance(payload_hash, str)
            or _sha256_file(source / payload_file) != payload_hash
        ):
            raise ArtifactCompatibilityError(
                "Artifact payload checksum is incompatible"
            )
    return manifest


def _sha256_file(path: Path) -> str:
    try:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as error:
        raise ArtifactCompatibilityError(
            "Artifact payload is missing or unreadable"
        ) from error
