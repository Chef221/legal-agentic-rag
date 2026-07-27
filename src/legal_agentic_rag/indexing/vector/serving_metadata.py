"""Persisted SQLite random-access metadata for fast vector serving startup."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import tempfile
from threading import RLock

import numpy as np
from pydantic import ValidationError

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import RetrievalFilters

DATABASE_FILENAME = "metadata.sqlite3"
MANIFEST_FILENAME = "manifest.json"
_TABLE_NAME = "chunk_metadata"
_LOGGER = logging.getLogger(__name__)
_FILTER_COLUMNS = (
    ("document_ids", "document_id"),
    ("document_types", "document_type"),
    ("legal_fields", "legal_field"),
    ("effect_statuses", "effect_status"),
)


class SQLiteVectorChunkStore(Sequence[LegalChunk]):
    """Read final vector-hit metadata through persisted byte offsets."""

    def __init__(
        self,
        *,
        chunks_path: Path,
        connection: sqlite3.Connection,
        record_count: int,
    ) -> None:
        self._chunks_path = chunks_path
        self._connection = connection
        self._record_count = record_count
        self._lock = RLock()

    @classmethod
    def load(
        cls,
        source: Path,
        *,
        chunks_path: Path,
        vector_manifest: ArtifactManifest,
        verify_integrity: bool,
    ) -> "SQLiteVectorChunkStore":
        """Load a compatible immutable sidecar without scanning chunk JSONL."""
        source = source.resolve()
        manifest_path = source / MANIFEST_FILENAME
        database_path = source / DATABASE_FILENAME
        if not source.is_dir() or not all(
            path.is_file() for path in (manifest_path, database_path, chunks_path)
        ):
            raise ArtifactCompatibilityError(
                "Vector serving metadata files are incomplete"
            )
        try:
            manifest = ArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Vector serving metadata manifest is invalid"
            ) from error
        _validate_manifest(manifest, vector_manifest)
        if verify_integrity:
            expected_checksum = manifest.metadata.get("database_sha256")
            if (
                not isinstance(expected_checksum, str)
                or _sha256_file(database_path) != expected_checksum
            ):
                raise ArtifactCompatibilityError(
                    "Vector serving metadata checksum does not match"
                )
        try:
            connection = sqlite3.connect(
                f"{database_path.as_uri()}?mode=ro&immutable=1",
                uri=True,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            if verify_integrity:
                result = connection.execute("PRAGMA quick_check").fetchone()
                if result is None or result[0] != "ok":
                    raise ArtifactCompatibilityError(
                        "Vector serving metadata integrity check failed"
                    )
            count = connection.execute(
                f"SELECT COUNT(*) FROM {_TABLE_NAME}"
            ).fetchone()
            if count is None or count[0] != manifest.record_count:
                raise ArtifactCompatibilityError(
                    "Vector serving metadata record count differs"
                )
        except ArtifactCompatibilityError:
            if "connection" in locals():
                connection.close()
            raise
        except sqlite3.Error as error:
            if "connection" in locals():
                connection.close()
            raise ArtifactCompatibilityError(
                "Vector serving metadata database cannot be loaded"
            ) from error
        return cls(
            chunks_path=chunks_path,
            connection=connection,
            record_count=manifest.record_count,
        )

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int | slice) -> LegalChunk | list[LegalChunk]:
        if isinstance(index, slice):
            return self.get_many(range(*index.indices(len(self))))
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("chunk index out of range")
        return self.get_many([normalized])[0]

    def __iter__(self) -> Iterator[LegalChunk]:
        try:
            with self._chunks_path.open("rb") as stream:
                for line in stream:
                    yield LegalChunk.model_validate_json(line)
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Vector chunk metadata payload is invalid"
            ) from error

    def chunk_id(self, index: int) -> str:
        """Return one stable chunk ID without reading the JSONL payload."""
        row = self._row(index, columns="chunk_id")
        return str(row["chunk_id"])

    def get_many(self, indexes: Sequence[int]) -> list[LegalChunk]:
        """Read only selected final records from the aligned chunk JSONL."""
        offsets = [
            int(self._row(int(index), columns="byte_offset")["byte_offset"])
            for index in indexes
        ]
        values: list[LegalChunk] = []
        try:
            with self._chunks_path.open("rb") as stream:
                for offset in offsets:
                    stream.seek(offset)
                    line = stream.readline()
                    if not line:
                        raise ValueError("missing chunk record")
                    values.append(LegalChunk.model_validate_json(line))
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Vector chunk metadata payload is invalid"
            ) from error
        return values

    def filtered_indexes(self, filters: RetrievalFilters) -> np.ndarray | None:
        """Resolve exact unified filters through indexed SQLite columns."""
        conditions: list[str] = []
        parameters: list[str] = []
        for filter_field, column in _FILTER_COLUMNS:
            values = getattr(filters, filter_field)
            if values:
                placeholders = ", ".join("?" for _ in values)
                conditions.append(f"{column} IN ({placeholders})")
                parameters.extend(values)
        if not conditions:
            return None
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT row_index
                FROM {_TABLE_NAME}
                WHERE {' AND '.join(conditions)}
                ORDER BY row_index
                """,
                parameters,
            )
            return np.fromiter(
                (int(row["row_index"]) for row in rows),
                dtype=np.int64,
            )

    def close(self) -> None:
        """Close the immutable SQLite metadata connection."""
        with self._lock:
            self._connection.close()

    def _row(self, index: int, *, columns: str) -> sqlite3.Row:
        if index < 0 or index >= len(self):
            raise IndexError("chunk index out of range")
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT {columns}
                FROM {_TABLE_NAME}
                WHERE row_index = ?
                """,
                (index,),
            ).fetchone()
        if row is None:
            raise ArtifactCompatibilityError(
                "Vector serving metadata row is missing"
            )
        return row


def prepare_vector_serving_metadata(
    *,
    vector_directory: Path,
    destination: Path,
    vector_manifest: ArtifactManifest,
    batch_size: int = 10_000,
    progress_interval_records: int = 100_000,
) -> ArtifactManifest:
    """Build one immutable SQLite sidecar from validated vector chunk metadata."""
    if batch_size <= 0 or progress_interval_records <= 0:
        raise BackendInitializationError(
            "Vector serving metadata execution bounds must be positive"
        )
    destination = destination.resolve()
    if destination.exists():
        raise BackendInitializationError(
            "Vector serving metadata destination already exists"
        )
    chunks_filename = vector_manifest.metadata.get("chunks_filename")
    chunks_checksum = vector_manifest.metadata.get("chunks_sha256")
    if not isinstance(chunks_filename, str) or not isinstance(
        chunks_checksum, str
    ):
        raise ArtifactCompatibilityError(
            "Vector manifest chunk metadata is incomplete"
        )
    chunks_path = vector_directory.resolve() / chunks_filename
    if not chunks_path.is_file():
        raise ArtifactCompatibilityError("Vector chunk metadata is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    connection: sqlite3.Connection | None = None
    try:
        _LOGGER.info(
            "vector_serving_metadata_preparation_started",
            extra={"total_chunk_count": vector_manifest.record_count},
        )
        database_path = temporary / DATABASE_FILENAME
        connection = sqlite3.connect(database_path)
        _create_schema(connection)
        rows: list[tuple[object, ...]] = []
        count = 0
        chunks_digest = sha256()
        with chunks_path.open("rb") as stream:
            while True:
                byte_offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                chunks_digest.update(line)
                try:
                    chunk = LegalChunk.model_validate_json(line)
                except (ValidationError, ValueError) as error:
                    raise ArtifactCompatibilityError(
                        "Vector chunk metadata payload is invalid"
                    ) from error
                rows.append(
                    (
                        count,
                        byte_offset,
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.document_type,
                        chunk.legal_field,
                        chunk.effect_status,
                    )
                )
                count += 1
                if len(rows) >= batch_size:
                    _insert_rows(connection, rows)
                    rows.clear()
                if count % progress_interval_records == 0:
                    _LOGGER.info(
                        "vector_serving_metadata_preparation_progress",
                        extra={
                            "chunk_count": count,
                            "total_chunk_count": vector_manifest.record_count,
                        },
                    )
        if rows:
            _insert_rows(connection, rows)
        if count != vector_manifest.record_count:
            raise ArtifactCompatibilityError(
                "Vector serving metadata count differs from vector manifest"
            )
        if chunks_digest.hexdigest() != chunks_checksum:
            raise ArtifactCompatibilityError(
                "Vector chunk metadata checksum does not match manifest"
            )
        _create_filter_indexes(connection)
        connection.commit()
        connection.close()
        connection = None
        metadata = {
            "database_filename": DATABASE_FILENAME,
            "database_sha256": _sha256_file(database_path),
            "source_artifact_type": vector_manifest.artifact_type.value,
            "source_artifact_version": vector_manifest.artifact_version,
            "source_processing_config_hash": (
                vector_manifest.processing_config_hash
            ),
            "source_chunks_sha256": chunks_checksum,
            "backend_schema_version": "1.0",
        }
        manifest = ArtifactManifest(
            schema_version=vector_manifest.schema_version,
            artifact_type=ArtifactType.VECTOR_SERVING_METADATA,
            artifact_version="1.0",
            dataset_name=vector_manifest.dataset_name,
            dataset_revision=vector_manifest.dataset_revision,
            created_at=datetime.now(UTC),
            record_count=count,
            processing_config_hash=canonical_sha256(metadata),
            code_version=__version__,
            backend="sqlite_chunk_metadata",
            warnings=[],
            metadata=metadata,
        )
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        _LOGGER.info(
            "vector_serving_metadata_preparation_completed",
            extra={
                "chunk_count": count,
                "total_chunk_count": vector_manifest.record_count,
            },
        )
        return manifest
    except (OSError, sqlite3.Error) as error:
        raise BackendInitializationError(
            "Vector serving metadata could not be prepared"
        ) from error
    finally:
        if connection is not None:
            connection.close()
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _validate_manifest(
    manifest: ArtifactManifest,
    vector_manifest: ArtifactManifest,
) -> None:
    metadata = manifest.metadata
    expected = (
        vector_manifest.artifact_type.value,
        vector_manifest.artifact_version,
        vector_manifest.processing_config_hash,
        vector_manifest.metadata.get("chunks_sha256"),
    )
    actual = (
        metadata.get("source_artifact_type"),
        metadata.get("source_artifact_version"),
        metadata.get("source_processing_config_hash"),
        metadata.get("source_chunks_sha256"),
    )
    if (
        manifest.artifact_type != ArtifactType.VECTOR_SERVING_METADATA
        or manifest.backend != "sqlite_chunk_metadata"
        or manifest.record_count != vector_manifest.record_count
        or manifest.dataset_name != vector_manifest.dataset_name
        or manifest.dataset_revision != vector_manifest.dataset_revision
        or actual != expected
    ):
        raise ArtifactCompatibilityError(
            "Vector serving metadata is incompatible with vector artifact"
        )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE {_TABLE_NAME} (
            row_index INTEGER PRIMARY KEY,
            byte_offset INTEGER NOT NULL,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            document_type TEXT,
            legal_field TEXT,
            effect_status TEXT
        )
        """
    )


def _insert_rows(
    connection: sqlite3.Connection,
    rows: list[tuple[object, ...]],
) -> None:
    connection.executemany(
        f"""
        INSERT INTO {_TABLE_NAME} (
            row_index, byte_offset, chunk_id, document_id,
            document_type, legal_field, effect_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _create_filter_indexes(connection: sqlite3.Connection) -> None:
    for _, column in _FILTER_COLUMNS:
        connection.execute(
            f"CREATE INDEX idx_{_TABLE_NAME}_{column} ON {_TABLE_NAME} ({column})"
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
