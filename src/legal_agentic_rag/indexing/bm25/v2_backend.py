"""Persistent SQLite FTS5 BM25 backend for V2 RetrievalUnit records."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
import json
import logging
from pathlib import Path
import sqlite3
from threading import RLock
from time import perf_counter
from typing import Any

from legal_agentic_rag.configuration.online import BM25RuntimeConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.bm25.analyzer import UnicodeBM25Analyzer
from legal_agentic_rag.indexing.bm25.query_planner import BM25QueryPlanner
from legal_agentic_rag.indexing.vector.v2_retrieval_unit_store import V2RetrievalUnitStore
from legal_agentic_rag.schemas.preprocessing_v2 import RetrievalUnitV2
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)

_LOGGER = logging.getLogger(__name__)

EXPECTED_SCHEMA = "m54_v2_bm25_index_v1"
EXPECTED_BACKEND = "sqlite_fts5_v2"
EXPECTED_ANALYZER = "unicode_word_casefold_v1"
_TABLE_NAME = "bm25_v2_units"
_VOCAB_TABLE_NAME = "bm25_v2_vocab"


def _compute_sha256(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class V2SQLiteFTS5BM25Backend:
    """Exact BM25 search over V2 retrieval units using SQLite FTS5."""

    backend_name = EXPECTED_BACKEND

    def __init__(
        self,
        *,
        artifact_dir: Path,
        manifest: dict[str, Any],
        connection: sqlite3.Connection,
        store: V2RetrievalUnitStore,
        analyzer: UnicodeBM25Analyzer | None = None,
        runtime_config: BM25RuntimeConfig | None = None,
    ) -> None:
        self._artifact_dir = Path(artifact_dir).resolve()
        self._manifest = manifest
        self._connection = connection
        self._store = store
        self._analyzer = analyzer or UnicodeBM25Analyzer()
        self._query_planner = BM25QueryPlanner(runtime_config)
        self._lock = RLock()
        self._record_count = int(manifest.get("record_count", len(store)))

    @property
    def artifact_dir(self) -> Path:
        return self._artifact_dir

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def store(self) -> V2RetrievalUnitStore:
        return self._store

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def analyzer_name(self) -> str:
        return self._analyzer.name

    @classmethod
    def build(
        cls,
        source_units_path: Path,
        destination_dir: Path,
        *,
        source_sha256: str | None = None,
        analyzer: UnicodeBM25Analyzer | None = None,
        match_mode: str = "any",
        progress_callback: Callable[[int, float, int], None] | None = None,
        progress_interval: int = 50000,
        write_batch_size: int = 5000,
    ) -> dict[str, Any]:
        """Build compact SQLite FTS5 index from streaming V2 retrieval unit JSONL."""
        source_units_path = Path(source_units_path).resolve()
        destination_dir = Path(destination_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)

        if not source_units_path.is_file():
            raise ArtifactCompatibilityError(f"Source units file not found: {source_units_path}")

        analyzer = analyzer or UnicodeBM25Analyzer()
        db_path = destination_dir / "bm25_v2.sqlite3"
        manifest_path = destination_dir / "bm25_v2_manifest_v1.json"
        success_path = destination_dir / "SUCCESS.json"

        # Remove existing DB file if rebuilding
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        try:
            # Fast offline build pragmas
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA cache_size = -64000")

            # Create FTS5 table with unindexed metadata columns
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE {_TABLE_NAME} USING fts5(
                    row_index UNINDEXED,
                    retrieval_unit_id UNINDEXED,
                    document_id UNINDEXED,
                    provision_id UNINDEXED,
                    document_number UNINDEXED,
                    article_label UNINDEXED,
                    clause_label UNINDEXED,
                    point_label UNINDEXED,
                    search_terms,
                    tokenize = 'unicode61 remove_diacritics 0'
                )
                """
            )

            insert_sql = f"""
                INSERT INTO {_TABLE_NAME} (
                    row_index, retrieval_unit_id, document_id, provision_id,
                    document_number, article_label, clause_label, point_label,
                    search_terms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            seen_ids: set[str] = set()
            rows_batch: list[tuple[Any, ...]] = []
            count = 0
            t_start = perf_counter()

            with open(source_units_path, "rb") as f_in:
                for line in f_in:
                    line_s = line.strip()
                    if not line_s:
                        continue

                    try:
                        unit = RetrievalUnitV2.model_validate_json(line_s)
                    except Exception as e:
                        raise DataValidationError(f"Malformed RetrievalUnitV2 at row {count}") from e

                    if unit.retrieval_unit_id in seen_ids:
                        raise DataValidationError(f"Duplicate retrieval_unit_id: {unit.retrieval_unit_id}")
                    seen_ids.add(unit.retrieval_unit_id)

                    terms = analyzer.analyze(unit.retrieval_text)
                    search_terms_str = " ".join(terms)

                    doc_num = unit.document_identity.document_number or ""
                    art_lbl = unit.hierarchy.article_label or ""
                    cl_lbl = unit.hierarchy.clause_label or ""
                    pt_lbl = unit.hierarchy.point_label or ""

                    rows_batch.append((
                        count,
                        unit.retrieval_unit_id,
                        unit.document_id,
                        unit.provision_id,
                        doc_num,
                        art_lbl,
                        cl_lbl,
                        pt_lbl,
                        search_terms_str,
                    ))

                    count += 1

                    if len(rows_batch) >= write_batch_size:
                        conn.executemany(insert_sql, rows_batch)
                        rows_batch.clear()

                    if progress_callback is not None and count % progress_interval == 0:
                        elapsed = perf_counter() - t_start
                        db_size = db_path.stat().st_size if db_path.exists() else 0
                        progress_callback(count, elapsed, db_size)

            if rows_batch:
                conn.executemany(insert_sql, rows_batch)
                rows_batch.clear()

            conn.commit()

            # Optimize FTS5 structure
            conn.execute(f"INSERT INTO {_TABLE_NAME}({_TABLE_NAME}) VALUES('optimize')")
            conn.commit()

            # Integrity check
            check_res = conn.execute("PRAGMA integrity_check").fetchall()
            if not check_res or check_res[0][0] != "ok":
                raise BackendInitializationError(f"SQLite integrity check failed: {check_res}")

            # Verify row count
            db_count = conn.execute(f"SELECT count(*) FROM {_TABLE_NAME}").fetchone()[0]
            if db_count != count:
                raise DataValidationError(f"DB count {db_count} != expected {count}")

        finally:
            conn.close()

        # Compute file size and hash
        db_size = db_path.stat().st_size
        db_sha = _compute_sha256(db_path)

        manifest = {
            "schema": EXPECTED_SCHEMA,
            "backend": EXPECTED_BACKEND,
            "record_count": count,
            "source_schema_version": "m54-preprocessing-v2.1",
            "source_retrieval_units_sha256": source_sha256 or "",
            "analyzer_name": analyzer.name,
            "match_mode": match_mode,
            "sqlite_version": sqlite3.sqlite_version,
            "database_filename": "bm25_v2.sqlite3",
            "database_size_bytes": db_size,
            "database_sha256": db_sha,
            "protected_split_used": False,
        }

        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        success_info = {
            "status": "SUCCESS",
            "schema": EXPECTED_SCHEMA,
            "record_count": count,
            "database_sha256": db_sha,
        }
        success_path.write_text(json.dumps(success_info, indent=2) + "\n", encoding="utf-8")

        return manifest

    @classmethod
    def load(
        cls,
        artifact_dir: Path,
        units_path: Path,
        *,
        verify_db_sha: bool = False,
        analyzer: UnicodeBM25Analyzer | None = None,
        runtime_config: BM25RuntimeConfig | None = None,
        strict_manifest: bool = True,
    ) -> "V2SQLiteFTS5BM25Backend":
        """Load and validate persisted SQLite FTS5 artifact and aligned V2 retrieval-unit store."""
        artifact_dir = Path(artifact_dir).resolve()
        units_path = Path(units_path).resolve()

        manifest_path = artifact_dir / "bm25_v2_manifest_v1.json"
        if not manifest_path.is_file():
            raise ArtifactCompatibilityError(f"Missing BM25 manifest at {manifest_path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ArtifactCompatibilityError(f"Malformed manifest JSON at {manifest_path}") from e

        if strict_manifest:
            schema = manifest.get("schema")
            if schema != EXPECTED_SCHEMA:
                raise ArtifactCompatibilityError(f"Unsupported BM25 schema {schema}, expected {EXPECTED_SCHEMA}")

            backend = manifest.get("backend")
            if backend != EXPECTED_BACKEND:
                raise ArtifactCompatibilityError(f"Unsupported backend {backend}, expected {EXPECTED_BACKEND}")

            an_name = manifest.get("analyzer_name")
            if an_name != EXPECTED_ANALYZER:
                raise ArtifactCompatibilityError(f"Unsupported analyzer {an_name}, expected {EXPECTED_ANALYZER}")

        db_filename = manifest.get("database_filename", "bm25_v2.sqlite3")
        db_path = artifact_dir / db_filename
        if not db_path.is_file():
            raise ArtifactCompatibilityError(f"Missing BM25 SQLite database at {db_path}")

        if verify_db_sha:
            expected_sha = manifest.get("database_sha256")
            actual_sha = _compute_sha256(db_path)
            if expected_sha and actual_sha != expected_sha:
                raise ArtifactCompatibilityError(f"Database SHA256 mismatch: {actual_sha} != {expected_sha}")

        # Connect to SQLite
        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Create temp query vocabulary table for DF statistics
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.{_VOCAB_TABLE_NAME} USING fts5vocab(main, {_TABLE_NAME}, 'row')"
            )
        except Exception as e:
            raise BackendInitializationError(f"Failed to initialize SQLite connection for {db_path}") from e

        # Load store with expected count
        rec_count = manifest.get("record_count")
        store = V2RetrievalUnitStore.load(
            units_path,
            expected_count=rec_count if strict_manifest else None,
            verify_alignment=False,
        )

        return cls(
            artifact_dir=artifact_dir,
            manifest=manifest,
            connection=conn,
            store=store,
            analyzer=analyzer,
            runtime_config=runtime_config,
        )

    def close(self) -> None:
        """Close SQLite connection."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()

    def _match_expression(self, terms: list[str]) -> str:
        mode = self._manifest.get("match_mode", "any")
        operator = " OR " if mode == "any" else " AND "
        unique_terms = list(dict.fromkeys(terms))
        return operator.join(f'"{term}"' for term in unique_terms)

    def _document_frequencies(self, terms: list[str]) -> dict[str, int]:
        unique_terms = list(dict.fromkeys(terms))
        if not unique_terms:
            return {}
        placeholders = ", ".join("?" for _ in unique_terms)
        try:
            rows = self._connection.execute(
                f"""
                SELECT term, doc
                FROM temp.{_VOCAB_TABLE_NAME}
                WHERE term IN ({placeholders})
                """,
                unique_terms,
            ).fetchall()
        except sqlite3.Error as error:
            raise RetrievalError("SQLite FTS5 vocabulary lookup failed") from error
        return {str(row["term"]): int(row["doc"]) for row in rows}

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Execute BM25 search over SQLite FTS5 and resolve hits through V2RetrievalUnitStore."""
        t_start = perf_counter()

        # Reject unsupported legacy filters explicitly
        if (
            query.filters.document_types
            or query.filters.legal_fields
            or query.filters.effect_statuses
        ):
            raise RetrievalError(
                "V2 BM25 backend only supports document_ids filtering; "
                "legacy filters [document_types, legal_fields, effect_statuses] are not supported in V2"
            )

        if query.requested_strategy not in (None, RetrievalStrategy.BM25):
            raise RetrievalError("BM25 backend received a non-BM25 retrieval request")

        effective_query = query.rewritten_question or query.normalized_question
        terms = self._analyzer.analyze(effective_query)
        warnings: list[str] = []
        hits: list[RetrievalHit] = []

        if not terms:
            warnings.append("query_has_no_indexable_terms")
            elapsed_ms = (perf_counter() - t_start) * 1000.0
            return RetrievalResponse(
                query=query,
                strategy=RetrievalStrategy.BM25,
                hits=[],
                latency_ms=elapsed_ms,
                warnings=warnings,
                artifact_versions={"bm25_index": EXPECTED_SCHEMA},
            )

        with self._lock:
            frequencies = self._document_frequencies(terms)
            plan = self._query_planner.plan(
                terms,
                document_frequencies=frequencies,
                document_count=self._record_count,
            )

            if plan.was_limited:
                warnings.append("bm25_query_terms_limited")

            if not plan.terms:
                warnings.append("no_query_plan_terms")
                expression = self._match_expression(terms[:5])
            else:
                expression = self._match_expression(list(plan.terms))

            conditions = [f"{_TABLE_NAME} MATCH ?"]
            parameters: list[Any] = [expression]

            if query.filters.document_ids:
                placeholders = ", ".join("?" for _ in query.filters.document_ids)
                conditions.append(f"document_id IN ({placeholders})")
                parameters.extend(query.filters.document_ids)

            parameters.append(query.top_k)

            sql = f"""
                SELECT row_index, retrieval_unit_id, document_id, rank AS bm25_rank
                FROM {_TABLE_NAME}
                WHERE {' AND '.join(conditions)}
                ORDER BY rank ASC
                LIMIT ?
            """

            try:
                db_rows = self._connection.execute(sql, parameters).fetchall()
            except sqlite3.Error as error:
                raise RetrievalError("SQLite FTS5 BM25 search failed") from error

        if not db_rows:
            warnings.append("no_bm25_matches")
            elapsed_ms = (perf_counter() - t_start) * 1000.0
            return RetrievalResponse(
                query=query,
                strategy=RetrievalStrategy.BM25,
                hits=[],
                latency_ms=elapsed_ms,
                warnings=warnings,
                artifact_versions={"bm25_index": EXPECTED_SCHEMA},
            )

        row_indices = [int(r["row_index"]) for r in db_rows]
        units = self._store.get_many(row_indices)

        for rank_idx, (db_row, unit) in enumerate(zip(db_rows, units, strict=True), start=1):
            expected_uid = db_row["retrieval_unit_id"]
            if unit.retrieval_unit_id != expected_uid:
                raise ArtifactCompatibilityError(
                    f"Row resolution mismatch at row_index {db_row['row_index']}: "
                    f"SQLite unit_id '{expected_uid}' != Store unit_id '{unit.retrieval_unit_id}'"
                )

            bm25_score = -float(db_row["bm25_rank"])

            metadata: dict[str, Any] = {
                "provision_id": unit.provision_id,
                "retrieval_text": unit.retrieval_text,
                "document_identity": unit.document_identity.model_dump(mode="json"),
                "hierarchy": unit.hierarchy.model_dump(mode="json"),
                "strategy": unit.strategy,
                "quality_flags": list(unit.quality_flags),
                "segment_index": unit.segment_index,
                "segment_count": unit.segment_count,
            }

            hit = RetrievalHit(
                chunk_id=unit.retrieval_unit_id,
                document_id=unit.document_id,
                rank=rank_idx,
                score=bm25_score,
                strategy=RetrievalStrategy.BM25,
                text=unit.authority_text,
                metadata=metadata,
                retrieval_trace=RetrievalTrace(
                    bm25_rank=rank_idx,
                    bm25_score=bm25_score,
                ),
            )
            hits.append(hit)

        elapsed_ms = (perf_counter() - t_start) * 1000.0
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.BM25,
            hits=hits,
            latency_ms=elapsed_ms,
            warnings=warnings,
            artifact_versions={"bm25_index": EXPECTED_SCHEMA},
        )
