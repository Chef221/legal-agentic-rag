"""Persistent SQLite FTS5 BM25 indexing and retrieval backend."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
import logging
from pathlib import Path
import sqlite3
from threading import RLock
from time import perf_counter
from typing import Callable

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import BM25IndexConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.bm25.analyzer import UnicodeBM25Analyzer
from legal_agentic_rag.indexing.bm25.artifact_store import (
    load_sqlite_artifact,
    persist_sqlite_artifact,
)
from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)

Clock = Callable[[], datetime]
_LOGGER = logging.getLogger(__name__)
_TABLE_NAME = "bm25_documents"


class SQLiteFTS5BM25Backend:
    """Reference BM25 backend using Python's bundled SQLite FTS5 engine."""

    backend_name = "sqlite_fts5"

    def __init__(
        self,
        config: BM25IndexConfig | None = None,
        *,
        analyzer: UnicodeBM25Analyzer | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or BM25IndexConfig()
        self._analyzer = analyzer or UnicodeBM25Analyzer()
        if self._analyzer.name != self._config.analyzer_name:
            raise BackendInitializationError(
                "Configured BM25 analyzer does not match backend analyzer"
            )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        self._manifest: ArtifactManifest | None = None

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return the legal-chunks identity used to build the active index."""
        with self._lock:
            if self._manifest is None:
                raise BackendInitializationError(
                    "BM25 index has not been built or loaded"
                )
            metadata = self._manifest.metadata
            values = (
                metadata.get("source_artifact_type"),
                metadata.get("source_artifact_version"),
                metadata.get("source_processing_config_hash"),
            )
            if any(not isinstance(value, str) or not value for value in values):
                raise ArtifactCompatibilityError(
                    "BM25 source artifact identity is invalid"
                )
            return str(values[0]), str(values[1]), str(values[2])

    def build(
        self,
        chunks: Iterable[LegalChunk],
        source_manifest: ArtifactManifest,
    ) -> ArtifactManifest:
        """Build an in-memory BM25 index from validated legal chunks."""
        with self._lock:
            chunk_list = list(chunks)
            self._validate_build_input(chunk_list, source_manifest)
            connection = self._new_connection()
            try:
                self._create_schema(connection)
                rows = [
                    self._chunk_row(chunk)
                    for chunk in sorted(chunk_list, key=lambda item: item.chunk_id)
                ]
                connection.executemany(
                    f"""
                    INSERT INTO {_TABLE_NAME} (
                        chunk_id, document_id, document_type, legal_field,
                        effect_status, search_terms, chunk_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.commit()
            except sqlite3.Error as error:
                connection.close()
                raise BackendInitializationError(
                    "Failed to build SQLite FTS5 index"
                ) from error

            self.close()
            self._connection = connection
            self._manifest = self._build_manifest(chunk_list, source_manifest)
            _LOGGER.info(
                "bm25_index_built",
                extra={
                    "backend": self.backend_name,
                    "chunk_count": len(chunk_list),
                    "dataset_name": source_manifest.dataset_name,
                },
            )
            return self._manifest

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Return deterministic, filtered BM25 hits for a validated query."""
        started = perf_counter()
        with self._lock:
            connection, manifest = self._require_ready()
            if query.requested_strategy not in (None, RetrievalStrategy.BM25):
                raise RetrievalError(
                    "BM25 backend received a non-BM25 retrieval request"
                )
            effective_query = query.rewritten_question or query.normalized_question
            terms = self._analyzer.analyze(effective_query)
            warnings: list[str] = []
            hits: list[RetrievalHit] = []
            if not terms:
                warnings.append("query_has_no_indexable_terms")
            else:
                expression = self._match_expression(terms)
                sql, parameters = self._search_statement(query, expression)
                try:
                    rows = connection.execute(sql, parameters).fetchall()
                except sqlite3.Error as error:
                    raise RetrievalError(
                        "SQLite FTS5 BM25 search failed"
                    ) from error
                hits = [
                    self._retrieval_hit(row, rank)
                    for rank, row in enumerate(rows, start=1)
                ]
                if not hits:
                    warnings.append("no_bm25_matches")
            artifact_version = manifest.artifact_version
        latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "bm25_search_completed",
            extra={
                "query_id": query.query_id,
                "strategy": RetrievalStrategy.BM25.value,
                "candidate_count": len(hits),
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.BM25,
            hits=hits,
            latency_ms=latency_ms,
            warnings=warnings,
            artifact_versions={"bm25_index": artifact_version},
        )

    def persist(self, destination: Path) -> ArtifactManifest:
        """Persist the current SQLite index and versioned manifest."""
        with self._lock:
            connection, manifest = self._require_ready()
            final_manifest = persist_sqlite_artifact(
                connection=connection,
                destination=destination,
                manifest=manifest,
            )
            self._manifest = final_manifest
            _LOGGER.info(
                "bm25_index_persisted",
                extra={
                    "backend": self.backend_name,
                    "chunk_count": final_manifest.record_count,
                },
            )
            return final_manifest

    def load(self, source: Path, manifest: ArtifactManifest) -> None:
        """Load a compatible, checksum-validated SQLite BM25 artifact."""
        with self._lock:
            connection, stored_manifest = load_sqlite_artifact(
                source=source,
                supplied_manifest=manifest,
                expected_backend=self.backend_name,
                expected_artifact_version=self._config.artifact_version,
                expected_analyzer=self._config.analyzer_name,
                expected_match_mode=self._config.match_mode,
            )
            self.close()
            self._connection = connection
            self._manifest = stored_manifest
            _LOGGER.info(
                "bm25_index_loaded",
                extra={
                    "backend": self.backend_name,
                    "chunk_count": stored_manifest.record_count,
                },
            )

    def close(self) -> None:
        """Release the current SQLite connection if one is open."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._manifest = None

    def _new_connection(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(":memory:", check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise BackendInitializationError("SQLite backend cannot initialize") from error

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE {_TABLE_NAME} USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                document_type UNINDEXED,
                legal_field UNINDEXED,
                effect_status UNINDEXED,
                search_terms,
                chunk_json UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 0'
            )
            """
        )

    def _chunk_row(self, chunk: LegalChunk) -> tuple[str | None, ...]:
        terms = self._analyzer.analyze(chunk.search_text)
        return (
            chunk.chunk_id,
            chunk.document_id,
            chunk.document_type,
            chunk.legal_field,
            chunk.effect_status,
            " ".join(terms),
            chunk.model_dump_json(),
        )

    def _build_manifest(
        self,
        chunks: list[LegalChunk],
        source_manifest: ArtifactManifest,
    ) -> ArtifactManifest:
        config_hash_payload = {
            "config": self._config,
            "source_artifact_type": source_manifest.artifact_type.value,
            "source_artifact_version": source_manifest.artifact_version,
            "source_processing_config_hash": source_manifest.processing_config_hash,
            "chunk_ids": sorted(chunk.chunk_id for chunk in chunks),
        }
        processing_config_hash = canonical_sha256(config_hash_payload)
        return ArtifactManifest(
            schema_version=source_manifest.schema_version,
            artifact_type=ArtifactType.BM25_INDEX,
            artifact_version=self._config.artifact_version,
            dataset_name=source_manifest.dataset_name,
            dataset_revision=source_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=len(chunks),
            processing_config_hash=processing_config_hash,
            code_version=__version__,
            backend=self.backend_name,
            warnings=[],
            metadata={
                "analyzer_name": self._config.analyzer_name,
                "match_mode": self._config.match_mode,
                "source_artifact_type": source_manifest.artifact_type.value,
                "source_artifact_version": source_manifest.artifact_version,
                "source_processing_config_hash": source_manifest.processing_config_hash,
                "sqlite_version": sqlite3.sqlite_version,
            },
        )

    def _match_expression(self, terms: list[str]) -> str:
        operator = " OR " if self._config.match_mode == "any" else " AND "
        unique_terms = list(dict.fromkeys(terms))
        return operator.join(f'"{term}"' for term in unique_terms)

    @staticmethod
    def _search_statement(
        query: RetrievalQuery,
        expression: str,
    ) -> tuple[str, list[object]]:
        conditions = [f"{_TABLE_NAME} MATCH ?"]
        parameters: list[object] = [expression]
        filters = (
            ("document_id", query.filters.document_ids),
            ("document_type", query.filters.document_types),
            ("legal_field", query.filters.legal_fields),
            ("effect_status", query.filters.effect_statuses),
        )
        for column, values in filters:
            if values:
                placeholders = ", ".join("?" for _ in values)
                conditions.append(f"{column} IN ({placeholders})")
                parameters.extend(values)
        parameters.append(query.top_k)
        statement = f"""
            SELECT chunk_id, chunk_json, bm25({_TABLE_NAME}) AS bm25_rank
            FROM {_TABLE_NAME}
            WHERE {' AND '.join(conditions)}
            ORDER BY bm25_rank ASC, chunk_id ASC
            LIMIT ?
        """
        return statement, parameters

    @staticmethod
    def _retrieval_hit(row: sqlite3.Row, rank: int) -> RetrievalHit:
        try:
            chunk = LegalChunk.model_validate_json(row["chunk_json"])
        except (ValueError, TypeError) as error:
            raise ArtifactCompatibilityError(
                "BM25 index contains an invalid legal chunk payload"
            ) from error
        score = -float(row["bm25_rank"])
        payload = chunk.model_dump(mode="json")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"chunk_id", "document_id", "text"}
        }
        return RetrievalHit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            rank=rank,
            score=score,
            strategy=RetrievalStrategy.BM25,
            text=chunk.text,
            metadata=metadata,
            retrieval_trace=RetrievalTrace(
                bm25_rank=rank,
                bm25_score=score,
            ),
        )

    def _require_ready(self) -> tuple[sqlite3.Connection, ArtifactManifest]:
        if self._connection is None or self._manifest is None:
            raise BackendInitializationError("BM25 index has not been built or loaded")
        return self._connection, self._manifest

    @staticmethod
    def _validate_build_input(
        chunks: list[LegalChunk],
        source_manifest: ArtifactManifest,
    ) -> None:
        if source_manifest.artifact_type != ArtifactType.LEGAL_CHUNKS:
            raise ArtifactCompatibilityError(
                "BM25 build requires a legal-chunks source artifact"
            )
        if source_manifest.record_count != len(chunks):
            raise DataValidationError(
                "Legal-chunks manifest count does not match BM25 build input"
            )
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise DataValidationError("BM25 build requires unique chunk IDs")
