"""Versioned NumPy vector artifact persistence and compatibility validation."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from pydantic import ValidationError

from legal_agentic_rag.contracts.vector_backend import VectorBuildBatchFactory
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
)
from legal_agentic_rag.schemas.build_validation import VectorBuildCheckpoint
from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

VECTORS_FILENAME = "vectors.npy"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "checkpoint.json"
_CHECKPOINT_TEMP_FILENAME = ".checkpoint.json.tmp"
_LOGGER = logging.getLogger(__name__)


def persist_vector_batches(
    *,
    batch_factory: VectorBuildBatchFactory,
    destination: Path,
    manifest: ArtifactManifest,
    dimension: int,
    checkpoint_interval_batches: int,
) -> ArtifactManifest:
    """Resume bounded vector batches and atomically publish the complete artifact."""
    destination = destination.resolve()
    if destination.exists():
        raise BackendInitializationError(
            "Vector artifact destination already exists"
        )
    if not destination.parent.exists():
        raise BackendInitializationError(
            "Vector artifact parent directory does not exist"
        )
    if checkpoint_interval_batches <= 0:
        raise DataValidationError(
            "Vector checkpoint interval must be positive"
        )
    partial = destination.parent / f".{destination.name}.partial"
    try:
        checkpoint = _prepare_vector_workspace(
            partial=partial,
            manifest=manifest,
            dimension=dimension,
        )
        committed_manifest = checkpoint.artifact_manifest
        vectors_path = partial / VECTORS_FILENAME
        chunks_path = partial / CHUNKS_FILENAME
        vectors = np.load(vectors_path, mmap_mode="r+")
        if vectors.shape != (committed_manifest.record_count, dimension):
            raise ArtifactCompatibilityError(
                "Vector checkpoint matrix shape is incompatible"
            )
        if vectors.dtype != np.dtype(np.float32):
            raise ArtifactCompatibilityError(
                "Vector checkpoint matrix dtype is incompatible"
            )
        seen_chunk_ids = _load_committed_chunk_ids(
            chunks_path,
            checkpoint,
        )
        offset = checkpoint.next_offset
        batch_count = 0
        with chunks_path.open("r+b") as stream:
            stream.truncate(checkpoint.chunks_byte_count)
            stream.seek(checkpoint.chunks_byte_count)
            _LOGGER.info(
                "vector_build_resumed",
                extra={
                    "chunk_count": offset,
                    "total_chunk_count": committed_manifest.record_count,
                },
            )
            for batch in batch_factory(offset):
                chunk_values = list(batch.chunks)
                matrix = np.asarray(batch.vectors, dtype=np.float32)
                if matrix.shape != (len(chunk_values), dimension):
                    raise DataValidationError(
                        "Vector batch shape does not match chunks"
                    )
                if not np.isfinite(matrix).all():
                    raise DataValidationError(
                        "Vector batch contains non-finite values"
                    )
                if chunk_values:
                    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                    if np.any(norms <= 0):
                        raise DataValidationError(
                            "Vector batch contains a zero vector"
                        )
                    matrix = matrix / norms
                end = offset + len(chunk_values)
                if end > manifest.record_count:
                    raise DataValidationError(
                        "Vector batches exceed source manifest count"
                    )
                vectors[offset:end] = matrix
                for chunk in chunk_values:
                    if chunk.chunk_id in seen_chunk_ids:
                        raise DataValidationError(
                            "Vector build requires unique chunk IDs"
                        )
                    seen_chunk_ids.add(chunk.chunk_id)
                    stream.write(chunk.model_dump_json().encode("utf-8"))
                    stream.write(b"\n")
                offset = end
                batch_count += 1
                if batch_count % checkpoint_interval_batches == 0:
                    checkpoint = _commit_vector_checkpoint(
                        partial=partial,
                        stream=stream,
                        vectors=vectors,
                        artifact_manifest=committed_manifest,
                        next_offset=offset,
                    )
            checkpoint = _commit_vector_checkpoint(
                partial=partial,
                stream=stream,
                vectors=vectors,
                artifact_manifest=committed_manifest,
                next_offset=offset,
            )
        del vectors
        if offset != committed_manifest.record_count:
            raise DataValidationError(
                "Vector batch count differs from source manifest"
            )
        metadata = dict(committed_manifest.metadata)
        metadata.update(
            {
                "vectors_filename": VECTORS_FILENAME,
                "vectors_sha256": _sha256_file(vectors_path),
                "chunks_filename": CHUNKS_FILENAME,
                "chunks_sha256": _sha256_file(chunks_path),
                "manifest_filename": MANIFEST_FILENAME,
                "chunk_order": "source_artifact_order",
            }
        )
        final_manifest = committed_manifest.model_copy(
            update={"metadata": metadata}
        )
        _write_json_atomic(
            partial / MANIFEST_FILENAME,
            final_manifest.model_dump(mode="json"),
        )
        partial.replace(destination)
        (destination / CHECKPOINT_FILENAME).unlink(missing_ok=True)
        (destination / _CHECKPOINT_TEMP_FILENAME).unlink(missing_ok=True)
        return final_manifest
    except (OSError, ValueError) as error:
        raise BackendInitializationError(
            "Vector artifact could not be persisted"
        ) from error


def _prepare_vector_workspace(
    *,
    partial: Path,
    manifest: ArtifactManifest,
    dimension: int,
) -> VectorBuildCheckpoint:
    checkpoint_path = partial / CHECKPOINT_FILENAME
    if partial.exists():
        checkpoint = _load_vector_checkpoint(checkpoint_path)
        if _manifest_identity(checkpoint.artifact_manifest) != _manifest_identity(
            manifest
        ):
            raise ArtifactCompatibilityError(
                "Vector checkpoint identity is incompatible"
            )
        if not (partial / VECTORS_FILENAME).is_file():
            raise ArtifactCompatibilityError(
                "Vector checkpoint matrix is missing"
            )
        if not (partial / CHUNKS_FILENAME).is_file():
            raise ArtifactCompatibilityError(
                "Vector checkpoint chunks are missing"
            )
        return checkpoint

    partial.mkdir(exist_ok=False)
    try:
        vectors = np.lib.format.open_memmap(
            partial / VECTORS_FILENAME,
            mode="w+",
            dtype=np.float32,
            shape=(manifest.record_count, dimension),
        )
        vectors.flush()
        del vectors
        (partial / CHUNKS_FILENAME).touch(exist_ok=False)
        checkpoint = VectorBuildCheckpoint(
            artifact_manifest=manifest,
            next_offset=0,
            chunks_byte_count=0,
            updated_at=datetime.now(UTC),
        )
        _write_json_atomic(
            checkpoint_path,
            checkpoint.model_dump(mode="json"),
        )
        return checkpoint
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _load_vector_checkpoint(path: Path) -> VectorBuildCheckpoint:
    try:
        checkpoint = VectorBuildCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Vector checkpoint is missing or invalid"
        ) from error
    if checkpoint.schema_version != "1.0":
        raise ArtifactCompatibilityError(
            "Vector checkpoint schema is incompatible"
        )
    return checkpoint


def _load_committed_chunk_ids(
    chunks_path: Path,
    checkpoint: VectorBuildCheckpoint,
) -> set[str]:
    if chunks_path.stat().st_size < checkpoint.chunks_byte_count:
        raise ArtifactCompatibilityError(
            "Vector checkpoint chunks are shorter than the committed offset"
        )
    seen_chunk_ids: set[str] = set()
    committed_count = 0
    consumed_bytes = 0
    try:
        with chunks_path.open("rb") as stream:
            while consumed_bytes < checkpoint.chunks_byte_count:
                line = stream.readline()
                if not line:
                    raise ArtifactCompatibilityError(
                        "Vector checkpoint chunks ended before the committed offset"
                    )
                consumed_bytes += len(line)
                if consumed_bytes > checkpoint.chunks_byte_count:
                    raise ArtifactCompatibilityError(
                        "Vector checkpoint byte offset splits a chunk record"
                    )
                chunk = LegalChunk.model_validate_json(line)
                if chunk.chunk_id in seen_chunk_ids:
                    raise ArtifactCompatibilityError(
                        "Vector checkpoint contains duplicate chunk IDs"
                    )
                seen_chunk_ids.add(chunk.chunk_id)
                committed_count += 1
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Vector checkpoint chunk records are invalid"
        ) from error
    if committed_count != checkpoint.next_offset:
        raise ArtifactCompatibilityError(
            "Vector checkpoint chunk count differs from its committed offset"
        )
    return seen_chunk_ids


def _commit_vector_checkpoint(
    *,
    partial: Path,
    stream,
    vectors: np.memmap,
    artifact_manifest: ArtifactManifest,
    next_offset: int,
) -> VectorBuildCheckpoint:
    vectors.flush()
    with (partial / VECTORS_FILENAME).open("r+b") as vector_stream:
        os.fsync(vector_stream.fileno())
    stream.flush()
    os.fsync(stream.fileno())
    checkpoint = VectorBuildCheckpoint(
        artifact_manifest=artifact_manifest,
        next_offset=next_offset,
        chunks_byte_count=stream.tell(),
        updated_at=datetime.now(UTC),
    )
    _write_json_atomic(
        partial / CHECKPOINT_FILENAME,
        checkpoint.model_dump(mode="json"),
    )
    _LOGGER.info(
        "vector_build_checkpoint_persisted",
        extra={
            "chunk_count": next_offset,
            "total_chunk_count": artifact_manifest.record_count,
        },
    )
    return checkpoint


def _manifest_identity(manifest: ArtifactManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("created_at", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.parent / _CHECKPOINT_TEMP_FILENAME
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


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
    chunk_order = stored_manifest.metadata.get("chunk_order")
    if chunk_order not in (None, "source_artifact_order"):
        raise ArtifactCompatibilityError("Vector artifact chunk order is incompatible")
    if chunk_order is None and chunk_ids != sorted(chunk_ids):
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
