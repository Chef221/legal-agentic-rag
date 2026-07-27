"""Persistence tests for checksum-validated NumPy vector artifacts."""

from pathlib import Path

import pytest

from legal_agentic_rag.configuration import VectorIndexConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.indexing.vector import NumpyVectorBackend
from legal_agentic_rag.indexing.vector.chunk_store import JsonlChunkStore
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    RetrievalQuery,
    RetrievalFilters,
    RetrievalStrategy,
)


def _build(
    backend: NumpyVectorBackend,
    chunks: list[LegalChunk],
    vectors: list[list[float]],
    source_manifest: ArtifactManifest,
) -> None:
    backend.build(
        chunks,
        vectors,
        source_manifest,
        model_name="fixture/e5",
        model_revision="fixture-revision",
        embedding_provider_name="fixture-provider",
        embedding_provider_version="1.0",
        dimension=2,
        embedding_batch_size=2,
    )


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-persist",
        original_question="tốc độ",
        normalized_question="tốc độ",
        requested_strategy=RetrievalStrategy.DENSE,
    )


def test_persist_reload_returns_identical_dense_results(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Memory-mapped reload retains vector scores and unified chunk metadata."""
    built = NumpyVectorBackend()
    _build(built, vector_chunks, vectors, vector_source_manifest)
    before = built.search(_query(), [1.0, 0.0])
    destination = tmp_path / "vector-v1"

    manifest = built.persist(destination)
    loaded = NumpyVectorBackend()
    loaded.load(destination, manifest)
    after = loaded.search(_query(), [1.0, 0.0])

    assert (destination / "vectors.npy").is_file()
    assert (destination / "chunks.jsonl").is_file()
    assert (destination / "manifest.json").is_file()
    assert manifest.metadata["vectors_sha256"]
    assert manifest.metadata["chunks_sha256"]
    assert [(hit.chunk_id, hit.score, hit.metadata) for hit in before.hits] == [
        (hit.chunk_id, hit.score, hit.metadata) for hit in after.hits
    ]
    assert isinstance(loaded._chunks, JsonlChunkStore)


def test_reload_uses_disk_backed_metadata_for_unified_filters(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """A loaded index filters through compact postings and parses only final hits."""
    built = NumpyVectorBackend()
    _build(built, vector_chunks, vectors, vector_source_manifest)
    destination = tmp_path / "vector-filtered"
    manifest = built.persist(destination)
    loaded = NumpyVectorBackend()
    loaded.load(destination, manifest)

    query = _query().model_copy(
        update={
            "filters": RetrievalFilters(
                legal_fields=["Thuáº¿"],
                effect_statuses=["Háº¿t hiá»‡u lá»±c"],
            )
        }
    )
    query.filters = RetrievalFilters(document_ids=["doc-tax"])
    response = loaded.search(query, [0.0, 1.0])

    assert [hit.chunk_id for hit in response.hits] == ["chunk-tax"]


def test_persist_refuses_overwrite_and_load_rejects_incompatibility(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Existing paths and incompatible manifests are explicit failures."""
    backend = NumpyVectorBackend()
    _build(backend, vector_chunks, vectors, vector_source_manifest)
    destination = tmp_path / "vector-v1"
    manifest = backend.persist(destination)

    with pytest.raises(BackendInitializationError):
        backend.persist(destination)
    wrong = manifest.model_copy(update={"artifact_type": ArtifactType.BM25_INDEX})
    with pytest.raises(ArtifactCompatibilityError):
        NumpyVectorBackend().load(destination, wrong)
    incompatible = NumpyVectorBackend(VectorIndexConfig(artifact_version="2.0"))
    with pytest.raises(ArtifactCompatibilityError, match="version"):
        incompatible.load(destination, manifest)


def test_load_rejects_tampered_vector_and_chunk_payloads(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Both numeric matrix and metadata payload are protected by checksums."""
    first = NumpyVectorBackend()
    _build(first, vector_chunks, vectors, vector_source_manifest)
    vector_destination = tmp_path / "vector-tampered"
    vector_manifest = first.persist(vector_destination)
    with (vector_destination / "vectors.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ArtifactCompatibilityError, match="matrix checksum"):
        NumpyVectorBackend().load(vector_destination, vector_manifest)

    second = NumpyVectorBackend()
    _build(second, vector_chunks, vectors, vector_source_manifest)
    chunk_destination = tmp_path / "chunks-tampered"
    chunk_manifest = second.persist(chunk_destination)
    with (chunk_destination / "chunks.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ArtifactCompatibilityError, match="metadata checksum"):
        NumpyVectorBackend().load(chunk_destination, chunk_manifest)
