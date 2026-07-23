"""Persistence and compatibility tests for SQLite BM25 artifacts."""

from pathlib import Path

import pytest

from legal_agentic_rag.configuration import BM25IndexConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    RetrievalQuery,
    RetrievalStrategy,
)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-persist",
        original_question="giấy phép lái xe",
        normalized_question="giấy phép lái xe",
        requested_strategy=RetrievalStrategy.BM25,
    )


def test_persist_reload_returns_identical_ranked_results(
    tmp_path: Path,
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Reloaded artifact produces the same IDs, scores, and metadata."""
    built = SQLiteFTS5BM25Backend()
    built.build(legal_chunks, chunk_manifest)
    before = built.search(_query())
    destination = tmp_path / "bm25-v1"

    manifest = built.persist(destination)
    loaded = SQLiteFTS5BM25Backend()
    loaded.load(destination, manifest)
    after = loaded.search(_query())

    assert (destination / "index.sqlite3").is_file()
    assert (destination / "manifest.json").is_file()
    assert manifest.metadata["index_sha256"]
    assert [(hit.chunk_id, hit.score, hit.metadata) for hit in before.hits] == [
        (hit.chunk_id, hit.score, hit.metadata) for hit in after.hits
    ]


def test_persist_refuses_to_overwrite_existing_destination(
    tmp_path: Path,
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Artifact versions are never silently replaced."""
    backend = SQLiteFTS5BM25Backend()
    backend.build(legal_chunks, chunk_manifest)
    destination = tmp_path / "bm25-v1"
    backend.persist(destination)

    with pytest.raises(BackendInitializationError):
        backend.persist(destination)


def test_load_rejects_manifest_mismatch_and_tampered_index(
    tmp_path: Path,
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Load verifies both typed manifest compatibility and index checksum."""
    backend = SQLiteFTS5BM25Backend()
    backend.build(legal_chunks, chunk_manifest)
    destination = tmp_path / "bm25-v1"
    manifest = backend.persist(destination)

    wrong_manifest = manifest.model_copy(
        update={"artifact_type": ArtifactType.VECTOR_INDEX}
    )
    with pytest.raises(ArtifactCompatibilityError):
        SQLiteFTS5BM25Backend().load(destination, wrong_manifest)

    with (destination / "index.sqlite3").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ArtifactCompatibilityError, match="checksum"):
        SQLiteFTS5BM25Backend().load(destination, manifest)


def test_load_rejects_incompatible_runtime_configuration(
    tmp_path: Path,
    legal_chunks: list[LegalChunk],
    chunk_manifest: ArtifactManifest,
) -> None:
    """Artifact version is an explicit runtime compatibility gate."""
    backend = SQLiteFTS5BM25Backend()
    backend.build(legal_chunks, chunk_manifest)
    destination = tmp_path / "bm25-v1"
    manifest = backend.persist(destination)

    incompatible = SQLiteFTS5BM25Backend(
        BM25IndexConfig(artifact_version="2.0")
    )
    with pytest.raises(ArtifactCompatibilityError, match="version"):
        incompatible.load(destination, manifest)
