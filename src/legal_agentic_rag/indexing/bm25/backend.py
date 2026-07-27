"""Persistent SQLite FTS5 BM25 indexing and retrieval backend."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
import logging
from pathlib import Path
import sqlite3
import tempfile
from threading import RLock
from time import perf_counter
from typing import Callable

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import BM25IndexConfig
from legal_agentic_rag.configuration.online import BM25RuntimeConfig
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
from legal_agentic_rag.indexing.bm25.query_planner import BM25QueryPlanner
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
_VOCAB_TABLE_NAME = "bm25_query_vocabulary"


class SQLiteFTS5BM25Backend:
    """Reference BM25 backend using Python's bundled SQLite FTS5 engine."""

    backend_name = "sqlite_fts5"

    def __init__(
        self,
        config: BM25IndexConfig | None = None,
        *,
        analyzer: UnicodeBM25Analyzer | None = None,
        runtime_config: BM25RuntimeConfig | None = None,
        verify_integrity_on_load: bool = True,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or BM25IndexConfig()
        self._analyzer = analyzer or UnicodeBM25Analyzer()
        self._query_planner = BM25QueryPlanner(runtime_config)
        self._verify_integrity_on_load = verify_integrity_on_load
        if self._analyzer.name != self._config.analyzer_name:
            raise BackendInitializationError(
                "Configured BM25 analyzer does not match backend analyzer"
            )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        self._manifest: ArtifactManifest | None = None
        self._temporary_index_path: Path | None = None

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
        """Build a disk-backed BM25 index from a one-pass chunk stream."""
        with self._lock:
            self._validate_source_manifest(source_manifest)
            temporary_file = tempfile.NamedTemporaryFile(
                prefix="legal-rag-bm25-",
                suffix=".sqlite3",
                delete=False,
            )
            temporary_file.close()
            temporary_path = Path(temporary_file.name)
            try:
                connection = self._new_connection(temporary_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
            count = 0
            seen_chunk_ids: set[str] = set()
            try:
                self._create_schema(connection)
                rows: list[tuple[str | None, ...]] = []
                for chunk in chunks:
                    if chunk.chunk_id in seen_chunk_ids:
                        raise DataValidationError(
                            "BM25 build requires unique chunk IDs"
                        )
                    seen_chunk_ids.add(chunk.chunk_id)
                    rows.append(self._chunk_row(chunk))
                    count += 1
                    if len(rows) >= self._config.write_batch_size:
                        self._insert_rows(connection, rows)
                        rows.clear()
                if rows:
                    self._insert_rows(connection, rows)
                connection.commit()
                if count != source_manifest.record_count:
                    raise DataValidationError(
                        "Legal-chunks manifest count does not match BM25 build input"
                    )
            except DataValidationError:
                connection.close()
                temporary_path.unlink(missing_ok=True)
                raise
            except sqlite3.Error as error:
                connection.close()
                temporary_path.unlink(missing_ok=True)
                raise BackendInitializationError(
                    "Failed to build SQLite FTS5 index"
                ) from error
            except Exception:
                connection.close()
                temporary_path.unlink(missing_ok=True)
                raise

            self.close()
            self._connection = connection
            self._temporary_index_path = temporary_path
            self._manifest = self._build_manifest(count, source_manifest)
            _LOGGER.info(
                "bm25_index_built",
                extra={
                    "backend": self.backend_name,
                    "chunk_count": count,
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
            unique_term_count = len(dict.fromkeys(terms))
            warnings: list[str] = []
            hits: list[RetrievalHit] = []
            if not terms:
                warnings.append("query_has_no_indexable_terms")
            else:
                frequencies = self._document_frequencies(connection, terms)
                plan = self._query_planner.plan(
                    terms,
                    document_frequencies=frequencies,
                    document_count=manifest.record_count,
                )
                if plan.was_limited:
                    warnings.append("bm25_query_terms_limited")
                expression = self._match_expression(list(plan.terms))
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
                "query_term_count": len(terms),
                "query_unique_term_count": unique_term_count,
                "query_selected_term_count": (
                    len(plan.terms) if terms else 0
                ),
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
                verify_integrity=self._verify_integrity_on_load,
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
            if self._temporary_index_path is not None:
                self._temporary_index_path.unlink(missing_ok=True)
            self._connection = None
            self._manifest = None
            self._temporary_index_path = None

    def _new_connection(self, path: Path | None = None) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                str(path) if path is not None else ":memory:",
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise BackendInitializationError("SQLite backend cannot initialize") from error

    @staticmethod
    def _insert_rows(
        connection: sqlite3.Connection,
        rows: list[tuple[str | None, ...]],
    ) -> None:
        connection.executemany(
            f"""
            INSERT INTO {_TABLE_NAME} (
                chunk_id, document_id, document_type, legal_field,
                effect_status, search_terms, chunk_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

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
        SQLiteFTS5BM25Backend._create_query_vocabulary(connection)

    @staticmethod
    def _create_query_vocabulary(connection: sqlite3.Connection) -> None:
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE temp.{_VOCAB_TABLE_NAME}
            USING fts5vocab(main, {_TABLE_NAME}, 'row')
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
        chunk_count: int,
        source_manifest: ArtifactManifest,
    ) -> ArtifactManifest:
        config_hash_payload = {
            "config": self._config,
            "source_artifact_type": source_manifest.artifact_type.value,
            "source_artifact_version": source_manifest.artifact_version,
            "source_processing_config_hash": source_manifest.processing_config_hash,
            "source_payload_sha256": source_manifest.metadata.get(
                "payload_sha256"
            ),
        }
        processing_config_hash = canonical_sha256(config_hash_payload)
        return ArtifactManifest(
            schema_version=source_manifest.schema_version,
            artifact_type=ArtifactType.BM25_INDEX,
            artifact_version=self._config.artifact_version,
            dataset_name=source_manifest.dataset_name,
            dataset_revision=source_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=chunk_count,
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
    def _document_frequencies(
        connection: sqlite3.Connection,
        terms: list[str],
    ) -> dict[str, int]:
        unique_terms = list(dict.fromkeys(terms))
        placeholders = ", ".join("?" for _ in unique_terms)
        try:
            rows = connection.execute(
                f"""
                SELECT term, doc
                FROM temp.{_VOCAB_TABLE_NAME}
                WHERE term IN ({placeholders})
                """,
                unique_terms,
            ).fetchall()
        except sqlite3.Error as error:
            raise RetrievalError(
                "SQLite FTS5 vocabulary lookup failed"
            ) from error
        return {str(row["term"]): int(row["doc"]) for row in rows}

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
            SELECT chunk_id, chunk_json, rank AS bm25_rank
            FROM {_TABLE_NAME}
            WHERE {' AND '.join(conditions)}
            ORDER BY rank ASC
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
    def _validate_source_manifest(source_manifest: ArtifactManifest) -> None:
        if source_manifest.artifact_type != ArtifactType.LEGAL_CHUNKS:
            raise ArtifactCompatibilityError(
                "BM25 build requires a legal-chunks source artifact"
            )
