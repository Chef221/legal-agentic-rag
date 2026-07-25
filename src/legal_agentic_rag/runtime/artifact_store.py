"""Persistence and validation for runtime-owned processed artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import TypeVar

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
ModelT = TypeVar("ModelT", bound=BaseModel)


class ModelArtifactWriter:
    """Incrementally stage one typed JSONL artifact and publish it atomically."""

    def __init__(self, destination: Path) -> None:
        destination = destination.resolve()
        if destination.exists():
            raise ArtifactCompatibilityError(
                "Artifact destination already exists"
            )
        if not destination.parent.is_dir():
            raise ArtifactCompatibilityError(
                "Artifact parent directory does not exist"
            )
        self._destination = destination
        self._temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}-",
                dir=destination.parent,
            )
        )
        self._payload_path = self._temporary / _RECORDS_FILENAME
        self._stream = self._payload_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        )
        self._digest = sha256()
        self._record_count = 0
        self._record_model: type[BaseModel] | None = None
        self._finalized = False

    @property
    def record_count(self) -> int:
        """Return the number of records staged so far."""
        return self._record_count

    def write(self, record: BaseModel) -> None:
        """Append one typed record without retaining it in memory."""
        if self._finalized or self._stream.closed:
            raise RuntimeError("Artifact writer is not open")
        record_model = type(record)
        if self._record_model is None:
            self._record_model = record_model
        elif self._record_model is not record_model:
            raise DataValidationError(
                "Artifact payload must contain one record model"
            )
        serialized = (
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        encoded = serialized.encode("utf-8")
        self._stream.write(serialized)
        self._digest.update(encoded)
        self._record_count += 1

    def write_many(self, records: Iterable[BaseModel]) -> None:
        """Append records from a one-pass iterable."""
        for record in records:
            self.write(record)

    def finalize(self, manifest: ArtifactManifest) -> ArtifactManifest:
        """Validate counts, write the manifest, and atomically publish files."""
        if self._finalized:
            raise RuntimeError("Artifact writer is already finalized")
        if manifest.record_count != self._record_count:
            raise DataValidationError(
                "Artifact manifest record count does not match payload"
            )
        self._stream.flush()
        self._stream.close()
        metadata = dict(manifest.metadata)
        metadata.update(
            {
                "payload_file": _RECORDS_FILENAME,
                "payload_sha256": self._digest.hexdigest(),
                "record_model": (
                    self._record_model.__name__
                    if self._record_model is not None
                    else "empty"
                ),
            }
        )
        stored_manifest = manifest.model_copy(update={"metadata": metadata})
        (self._temporary / _MANIFEST_FILENAME).write_text(
            json.dumps(
                stored_manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        destination_created = False
        try:
            self._destination.mkdir(exist_ok=False)
            destination_created = True
            for staged_file in self._temporary.iterdir():
                staged_file.replace(self._destination / staged_file.name)
            self._temporary.rmdir()
            self._finalized = True
            return stored_manifest
        except Exception:
            if destination_created:
                shutil.rmtree(self._destination, ignore_errors=True)
            raise

    def close(self) -> None:
        """Discard an unpublished staging directory."""
        if not self._stream.closed:
            self._stream.close()
        if not self._finalized:
            shutil.rmtree(self._temporary, ignore_errors=True)

    def __enter__(self) -> "ModelArtifactWriter":
        """Return this open writer."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Discard staged files unless finalize completed."""
        _ = (exception_type, exception, traceback)
        self.close()


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


def load_dataset_manifest(root: Path) -> DatasetManifest:
    """Load a persisted dataset manifest for a validated partial-build resume."""
    try:
        return DatasetManifest.model_validate_json(
            (root / _DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Dataset manifest is missing or invalid"
        ) from error


def persist_model_artifact(
    *,
    records: Iterable[BaseModel],
    destination: Path,
    manifest: ArtifactManifest,
) -> ArtifactManifest:
    """Persist typed records as deterministic JSONL with manifest checksum."""
    with ModelArtifactWriter(destination) as writer:
        writer.write_many(records)
        return writer.finalize(manifest)


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


def load_model_artifact(
    source: Path,
    *,
    expected_type: ArtifactType,
    record_type: type[ModelT],
) -> tuple[list[ModelT], ArtifactManifest]:
    """Stream-parse a checksum-validated runtime JSONL artifact."""
    records, manifest = stream_model_artifact(
        source,
        expected_type=expected_type,
        record_type=record_type,
    )
    return list(records), manifest


def stream_model_artifact(
    source: Path,
    *,
    expected_type: ArtifactType,
    record_type: type[ModelT],
) -> tuple[Iterator[ModelT], ArtifactManifest]:
    """Return a one-pass typed iterator over a checksum-validated artifact."""
    manifest = load_artifact_manifest(
        source,
        expected_type=expected_type,
        verify_payload=True,
    )
    payload_file = manifest.metadata.get("payload_file")
    record_model = manifest.metadata.get("record_model")
    if not isinstance(payload_file, str) or record_model not in {
        record_type.__name__,
        "empty",
    }:
        raise ArtifactCompatibilityError(
            "Artifact record model metadata is incompatible"
        )
    payload_path = source / payload_file

    def iterate() -> Iterator[ModelT]:
        count = 0
        try:
            with payload_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        raise ArtifactCompatibilityError(
                            "Artifact JSONL contains a blank record"
                        )
                    yield record_type.model_validate_json(line)
                    count += 1
        except ArtifactCompatibilityError:
            raise
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Artifact JSONL payload is invalid"
            ) from error
        if count != manifest.record_count:
            raise ArtifactCompatibilityError(
                "Artifact payload count differs from manifest"
            )

    return iterate(), manifest


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
