"""Unit tests for deterministic SQLite FTS5 BM25 indexing and search."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from legal_agentic_rag.configuration import BM25IndexConfig, BM25RuntimeConfig
from legal_agentic_rag.contracts import BM25Backend
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    RetrievalFilters,
    RetrievalQuery,
    RetrievalStrategy,
)


def _query(
    text: str,
    *,
    top_k: int = 10,
    filters: RetrievalFilters | None = None,
    strategy: RetrievalStrategy | None = RetrievalStrategy.BM25,
) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-1",
        original_question=text,
        normalized_question=text,
        filters=filters or RetrievalFilters(),
        top_k=top_k,
        candidate_k=max(100, top_k),
        requested_strategy=strategy,
    )


def _backend() -> SQLiteFTS5BM25Backend:
    return SQLiteFTS5BM25Backend(
        clock=lambda: datetime(2026, 7, 22, tzinfo=UTC)
    )


def test_backend_builds_manifest_and_satisfies_protocol(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Build records backend identity, provenance, and corpus count."""
    backend = _backend()

    manifest = backend.build(legal_chunks, chunk_manifest)

    assert isinstance(backend, BM25Backend)
    assert manifest.artifact_type == ArtifactType.BM25_INDEX
    assert manifest.backend == "sqlite_fts5"
    assert manifest.code_version == "0.20.4"
    assert manifest.record_count == 3
    assert manifest.dataset_revision == "fixture-revision"
    assert manifest.metadata["analyzer_name"] == "unicode_word_casefold_v1"
    assert manifest.metadata["source_processing_config_hash"] == (
        "chunk-config-hash"
    )
    assert backend.source_artifact_identity == (
        "legal_chunks",
        "1.0",
        "chunk-config-hash",
    )


def test_backend_consumes_chunk_stream_in_configured_batches(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """BM25 build accepts a one-pass corpus without a corpus-sized row list."""
    iteration_count = 0

    def chunks():
        nonlocal iteration_count
        iteration_count += 1
        yield from legal_chunks

    backend = SQLiteFTS5BM25Backend(
        BM25IndexConfig(write_batch_size=1),
    )
    manifest = backend.build(chunks(), chunk_manifest)

    assert iteration_count == 1
    assert manifest.record_count == len(legal_chunks)


def test_search_ranks_lexical_match_and_returns_chunk_metadata(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """BM25 hits retain source chunk identity, legal metadata, and trace."""
    backend = _backend()
    backend.build(legal_chunks, chunk_manifest)

    response = backend.search(_query("phạt tiền vì chạy quá tốc độ"))

    assert response.strategy == RetrievalStrategy.BM25
    assert response.hits[0].chunk_id == "chunk-speed"
    assert response.hits[0].document_id == "doc-traffic"
    assert response.hits[0].metadata["document_number"] == (
        "doc-traffic/2026/QH"
    )
    assert response.hits[0].metadata["structure"]["article_number"] == "5"
    assert response.hits[0].retrieval_trace.bm25_rank == 1
    assert response.hits[0].retrieval_trace.bm25_score == response.hits[0].score
    assert response.artifact_versions == {"bm25_index": "1.0"}
    assert response.latency_ms >= 0


def test_search_prefers_agent_rewritten_question(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """BM25 retries search the rewritten query instead of the stale first form."""
    backend = _backend()
    backend.build(legal_chunks, chunk_manifest)
    query = _query("không khớp").model_copy(
        update={"rewritten_question": "nộp thuế"}
    )

    response = backend.search(query)

    assert [hit.chunk_id for hit in response.hits] == ["chunk-tax"]


def test_search_enforces_top_k_and_exact_filters(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Top-k and unified retrieval filters are applied by the backend."""
    backend = _backend()
    backend.build(legal_chunks, chunk_manifest)

    limited = backend.search(_query("phạt", top_k=1))
    filtered = backend.search(
        _query(
            "thời hạn",
            filters=RetrievalFilters(
                legal_fields=["Thuế"], effect_statuses=["Hết hiệu lực"]
            ),
        )
    )

    assert len(limited.hits) == 1
    assert [hit.chunk_id for hit in filtered.hits] == ["chunk-tax"]


def test_search_statement_uses_fts5_rank_limit_optimization() -> None:
    """FTS5 handles the bounded rank scan without a global secondary sort."""
    sql, parameters = SQLiteFTS5BM25Backend._search_statement(
        _query("nghỉ hằng năm", top_k=5),
        '"nghỉ" OR "hằng" OR "năm"',
    )

    normalized_sql = " ".join(sql.split())
    assert "rank AS bm25_rank" in normalized_sql
    assert "ORDER BY rank ASC LIMIT ?" in normalized_sql
    assert "bm25(bm25_documents)" not in normalized_sql
    assert "chunk_id ASC" not in normalized_sql
    assert parameters == ['"nghỉ" OR "hằng" OR "năm"', 5]


def test_search_limits_high_frequency_query_terms_without_losing_target(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """A bounded corpus-aware plan still returns the discriminative legal hit."""
    backend = SQLiteFTS5BM25Backend(
        runtime_config=BM25RuntimeConfig(
            max_query_terms=2,
            max_document_frequency_ratio=0.5,
        )
    )
    backend.build(legal_chunks, chunk_manifest)

    response = backend.search(
        _query("thời hạn nộp thuế người xử phạt giấy phép")
    )

    assert response.hits[0].chunk_id == "chunk-tax"
    assert "bm25_query_terms_limited" in response.warnings


def test_punctuation_only_query_returns_empty_response_with_warning(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """A query without indexable terms does not execute invalid FTS syntax."""
    backend = _backend()
    backend.build(legal_chunks, chunk_manifest)

    response = backend.search(_query("...?!"))

    assert response.hits == []
    assert response.warnings == ["query_has_no_indexable_terms"]


def test_empty_index_returns_no_matches_without_failure(
    chunk_manifest: ArtifactManifest,
) -> None:
    """A valid empty legal-chunks artifact remains queryable and traceable."""
    backend = _backend()
    empty_manifest = chunk_manifest.model_copy(update={"record_count": 0})
    backend.build([], empty_manifest)

    response = backend.search(_query("phạt"))

    assert response.hits == []
    assert response.warnings == ["no_bm25_matches"]


def test_build_and_search_are_deterministic(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Input ordering does not change manifest hash or ranked identities."""
    first = _backend()
    second = _backend()

    first_manifest = first.build(legal_chunks, chunk_manifest)
    second_manifest = second.build(reversed(legal_chunks), chunk_manifest)
    first_hits = first.search(_query("phạt")).hits
    second_hits = second.search(_query("phạt")).hits

    assert first_manifest.processing_config_hash == (
        second_manifest.processing_config_hash
    )
    assert [(hit.chunk_id, hit.score) for hit in first_hits] == [
        (hit.chunk_id, hit.score) for hit in second_hits
    ]


def test_loaded_backend_supports_serialized_cross_thread_search(
    tmp_path: Path,
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Gradio workers can safely query an index loaded during app startup."""
    built = _backend()
    built.build(legal_chunks, chunk_manifest)
    persisted_manifest = built.persist(tmp_path / "bm25")
    loaded = _backend()
    loaded.load(tmp_path / "bm25", persisted_manifest)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                loaded.search,
                _query("phạt tiền vì chạy quá tốc độ"),
            )
            for _ in range(6)
        ]

    assert [
        response.hits[0].chunk_id
        for response in (future.result() for future in futures)
    ] == ["chunk-speed"] * 6


def test_all_match_mode_requires_every_query_term(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Configured all-term matching narrows lexical retrieval explicitly."""
    backend = SQLiteFTS5BM25Backend(BM25IndexConfig(match_mode="all"))
    backend.build(legal_chunks, chunk_manifest)

    response = backend.search(_query("giấy phép"))

    assert [hit.chunk_id for hit in response.hits] == ["chunk-license"]


def test_backend_rejects_invalid_state_strategy_and_source(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Backend fails clearly for unavailable state and incompatible inputs."""
    backend = _backend()
    with pytest.raises(BackendInitializationError):
        backend.search(_query("phạt"))

    wrong_manifest = chunk_manifest.model_copy(
        update={"artifact_type": ArtifactType.LEGAL_BLOCKS}
    )
    with pytest.raises(ArtifactCompatibilityError):
        backend.build(legal_chunks, wrong_manifest)

    wrong_count = chunk_manifest.model_copy(update={"record_count": 1})
    with pytest.raises(DataValidationError):
        backend.build(legal_chunks, wrong_count)

    backend.build(legal_chunks, chunk_manifest)
    with pytest.raises(RetrievalError):
        backend.search(_query("phạt", strategy=RetrievalStrategy.DENSE))


def test_backend_rejects_duplicate_chunk_ids(
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Duplicate retrieval identities cannot enter the index."""
    duplicates = [legal_chunks[0], legal_chunks[0]]
    manifest = chunk_manifest.model_copy(update={"record_count": 2})

    with pytest.raises(DataValidationError):
        _backend().build(duplicates, manifest)
