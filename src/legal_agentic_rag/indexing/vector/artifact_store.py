"""Versioned NumPy vector artifact persistence and compatibility validation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from pydantic import ValidationError

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

VECTORS_FILENAME = "vectors.npy"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"


def persist_vector_artifact(
    *,
    vectors: np.ndarray,
    chunks: list[LegalChunk],
    destination: Path,
    manifest: ArtifactManifest,
) -> ArtifactManifest:
    """Persist aligned vectors and chunks without replacing existing data."""
    destination = destination.resolve()
    if destination.exists():
        raise BackendInitializationError("Vector artifact destination already exists")
    if not destination.parent.exists():
        raise BackendInitializationError(
            "Vector artifact parent directory does not exist"
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    destination_created = False
    try:
        vectors_path = temporary / VECTORS_FILENAME
        chunks_path = temporary / CHUNKS_FILENAME
        np.save(vectors_path, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
        with chunks_path.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(chunk.model_dump_json())
                stream.write("\n")
        metadata = dict(manifest.metadata)
        metadata.update(
            {
                "vectors_filename": VECTORS_FILENAME,
                "vectors_sha256": _sha256_file(vectors_path),
                "chunks_filename": CHUNKS_FILENAME,
                "chunks_sha256": _sha256_file(chunks_path),
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
            "Vector artifact could not be persisted"
        ) from error
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def load_vector_artifact(
    *,
    source: Path,
    supplied_manifest: ArtifactManifest,
    expected_backend: str,
    expected_artifact_version: str,
    expected_distance_metric: str,
    expected_dtype: str,
) -> tuple[np.ndarray, list[LegalChunk], ArtifactManifest]:
    """Validate checksums and load an immutable, memory-mapped vector artifact."""
    source = source.resolve()
    if not source.is_dir():
        raise ArtifactCompatibilityError("Vector artifact source must be a directory")
    vectors_path = source / VECTORS_FILENAME
    chunks_path = source / CHUNKS_FILENAME
    manifest_path = source / MANIFEST_FILENAME
    if not all(path.is_file() for path in (vectors_path, chunks_path, manifest_path)):
        raise ArtifactCompatibilityError("Vector artifact files are incomplete")
    try:
        stored_manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError("Vector manifest is invalid") from error
    if stored_manifest != supplied_manifest:
        raise ArtifactCompatibilityError(
            "Supplied vector manifest does not match persisted manifest"
        )
    _validate_manifest(
        stored_manifest,
        expected_backend=expected_backend,
        expected_artifact_version=expected_artifact_version,
        expected_distance_metric=expected_distance_metric,
        expected_dtype=expected_dtype,
    )
    _validate_checksum(
        vectors_path,
        stored_manifest.metadata.get("vectors_sha256"),
        "vector matrix",
    )
    _validate_checksum(
        chunks_path,
        stored_manifest.metadata.get("chunks_sha256"),
        "chunk metadata",
    )
    try:
        vectors = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
        chunks = _load_chunks(chunks_path)
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactCompatibilityError("Vector artifact payload is invalid") from error
    dimension = stored_manifest.metadata.get("dimension")
    embedding_batch_size = stored_manifest.metadata.get("embedding_batch_size")
    if (
        not isinstance(dimension, int)
        or dimension <= 0
        or not isinstance(embedding_batch_size, int)
        or embedding_batch_size <= 0
        or vectors.dtype != np.float32
        or vectors.shape != (stored_manifest.record_count, dimension)
        or len(chunks) != stored_manifest.record_count
    ):
        raise ArtifactCompatibilityError(
            "Vector artifact shape or record count is incompatible"
        )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ArtifactCompatibilityError("Vector artifact contains duplicate chunk IDs")
    if chunk_ids != sorted(chunk_ids):
        raise ArtifactCompatibilityError("Vector artifact chunk order is incompatible")
    if vectors.size:
        if not np.isfinite(vectors).all():
            raise ArtifactCompatibilityError("Vector artifact contains non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise ArtifactCompatibilityError("Vector artifact is not normalized")
    return vectors, chunks, stored_manifest


def _validate_manifest(
    manifest: ArtifactManifest,
    *,
    expected_backend: str,
    expected_artifact_version: str,
    expected_distance_metric: str,
    expected_dtype: str,
) -> None:
    if manifest.artifact_type != ArtifactType.VECTOR_INDEX:
        raise ArtifactCompatibilityError("Manifest does not describe a vector index")
    if manifest.backend != expected_backend:
        raise ArtifactCompatibilityError("Vector backend is incompatible")
    if manifest.artifact_version != expected_artifact_version:
        raise ArtifactCompatibilityError("Vector artifact version is incompatible")
    if manifest.model_name is None or manifest.model_revision is None:
        raise ArtifactCompatibilityError("Vector artifact model identity is incomplete")
    metadata = manifest.metadata
    expected_values = {
        "distance_metric": expected_distance_metric,
        "dtype": expected_dtype,
        "vectors_filename": VECTORS_FILENAME,
        "chunks_filename": CHUNKS_FILENAME,
        "source_artifact_type": ArtifactType.LEGAL_CHUNKS.value,
        "normalized_vectors": True,
    }
    if any(metadata.get(key) != value for key, value in expected_values.items()):
        raise ArtifactCompatibilityError("Vector artifact metadata is incompatible")
    for field_name in ("embedding_provider_name", "embedding_provider_version"):
        value = metadata.get(field_name)
        if not isinstance(value, str) or not value:
            raise ArtifactCompatibilityError(
                "Vector embedding provider metadata is incompatible"
            )


def _load_chunks(path: Path) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("blank chunk record")
            chunks.append(LegalChunk.model_validate_json(line))
    return chunks


def _validate_checksum(path: Path, expected: object, label: str) -> None:
    try:
        actual = _sha256_file(path)
    except OSError as error:
        raise ArtifactCompatibilityError(f"Vector {label} cannot be read") from error
    if not isinstance(expected, str) or actual != expected:
        raise ArtifactCompatibilityError(f"Vector {label} checksum mismatch")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
