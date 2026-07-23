"""Versioned SQLite BM25 artifact persistence and compatibility checks."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from pydantic import ValidationError

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

INDEX_FILENAME = "index.sqlite3"
MANIFEST_FILENAME = "manifest.json"


def persist_sqlite_artifact(
    *,
    connection: sqlite3.Connection,
    destination: Path,
    manifest: ArtifactManifest,
) -> ArtifactManifest:
    """Persist an index without overwriting an existing artifact."""
    destination = destination.resolve()
    if destination.exists():
        raise BackendInitializationError("BM25 artifact destination already exists")
    if not destination.parent.exists():
        raise BackendInitializationError("BM25 artifact parent directory does not exist")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    destination_created = False
    try:
        index_path = temporary / INDEX_FILENAME
        persisted = sqlite3.connect(index_path)
        try:
            connection.backup(persisted)
        finally:
            persisted.close()
        checksum = _sha256_file(index_path)
        metadata = dict(manifest.metadata)
        metadata.update(
            {
                "index_filename": INDEX_FILENAME,
                "index_sha256": checksum,
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
    except (OSError, sqlite3.Error) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise BackendInitializationError(
            "BM25 artifact could not be persisted"
        ) from error
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def load_sqlite_artifact(
    *,
    source: Path,
    supplied_manifest: ArtifactManifest,
    expected_backend: str,
    expected_artifact_version: str,
    expected_analyzer: str,
    expected_match_mode: str,
) -> tuple[sqlite3.Connection, ArtifactManifest]:
    """Validate and open a persisted BM25 artifact in read-only mode."""
    source = source.resolve()
    if not source.is_dir():
        raise ArtifactCompatibilityError("BM25 artifact source must be a directory")
    manifest_path = source / MANIFEST_FILENAME
    index_path = source / INDEX_FILENAME
    if not manifest_path.is_file() or not index_path.is_file():
        raise ArtifactCompatibilityError("BM25 artifact files are incomplete")

    try:
        stored_manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError("BM25 manifest is invalid") from error
    if stored_manifest != supplied_manifest:
        raise ArtifactCompatibilityError(
            "Supplied BM25 manifest does not match persisted manifest"
        )
    _validate_manifest(
        stored_manifest,
        expected_backend=expected_backend,
        expected_artifact_version=expected_artifact_version,
        expected_analyzer=expected_analyzer,
        expected_match_mode=expected_match_mode,
    )
    expected_checksum = stored_manifest.metadata.get("index_sha256")
    try:
        actual_checksum = _sha256_file(index_path)
    except OSError as error:
        raise ArtifactCompatibilityError("BM25 index cannot be read") from error
    if not isinstance(expected_checksum, str) or actual_checksum != expected_checksum:
        raise ArtifactCompatibilityError("BM25 index checksum does not match manifest")

    try:
        connection = sqlite3.connect(
            f"{index_path.as_uri()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        integrity = connection.execute("PRAGMA quick_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ArtifactCompatibilityError("BM25 SQLite integrity check failed")
        row = connection.execute(
            "SELECT COUNT(*) FROM bm25_documents"
        ).fetchone()
        if row is None or row[0] != stored_manifest.record_count:
            raise ArtifactCompatibilityError(
                "BM25 index record count does not match manifest"
            )
    except ArtifactCompatibilityError:
        connection.close()
        raise
    except sqlite3.Error as error:
        if "connection" in locals():
            connection.close()
        raise ArtifactCompatibilityError("BM25 SQLite index cannot be loaded") from error
    return connection, stored_manifest


def _validate_manifest(
    manifest: ArtifactManifest,
    *,
    expected_backend: str,
    expected_artifact_version: str,
    expected_analyzer: str,
    expected_match_mode: str,
) -> None:
    if manifest.artifact_type != ArtifactType.BM25_INDEX:
        raise ArtifactCompatibilityError("Manifest does not describe a BM25 index")
    if manifest.backend != expected_backend:
        raise ArtifactCompatibilityError("BM25 backend is incompatible")
    if manifest.artifact_version != expected_artifact_version:
        raise ArtifactCompatibilityError("BM25 artifact version is incompatible")
    if manifest.metadata.get("analyzer_name") != expected_analyzer:
        raise ArtifactCompatibilityError("BM25 analyzer is incompatible")
    if manifest.metadata.get("match_mode") != expected_match_mode:
        raise ArtifactCompatibilityError("BM25 match mode is incompatible")
    if manifest.metadata.get("index_filename") != INDEX_FILENAME:
        raise ArtifactCompatibilityError("BM25 index filename is incompatible")
    if (
        manifest.metadata.get("source_artifact_type")
        != ArtifactType.LEGAL_CHUNKS.value
    ):
        raise ArtifactCompatibilityError("BM25 source artifact type is incompatible")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
