"""Tests for persisted vector serving metadata."""

from pathlib import Path

import pytest

from legal_agentic_rag.configuration import VectorRuntimeConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.indexing.vector import (
    NumpyVectorBackend,
    SQLiteVectorChunkStore,
    prepare_vector_serving_metadata,
)
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    RetrievalFilters,
    RetrievalQuery,
    RetrievalStrategy,
)


def _build_vector(
    destination: Path,
    chunks: list[LegalChunk],
    vectors: list[list[float]],
    source_manifest: ArtifactManifest,
) -> ArtifactManifest:
    backend = NumpyVectorBackend()
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
    return backend.persist(destination)


def test_prepare_reload_and_filter_use_sqlite_sidecar(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Prepared offsets and filters replace the JSONL startup scan."""
    vector_directory = tmp_path / "vector"
    vector_manifest = _build_vector(
        vector_directory,
        vector_chunks,
        vectors,
        vector_source_manifest,
    )
    sidecar = tmp_path / "vector-serving"

    serving_manifest = prepare_vector_serving_metadata(
        vector_directory=vector_directory,
        destination=sidecar,
        vector_manifest=vector_manifest,
        batch_size=1,
    )
    loaded = NumpyVectorBackend(
        runtime_config=VectorRuntimeConfig(require_serving_metadata=True),
        serving_metadata_source=sidecar,
    )
    loaded.load(vector_directory, vector_manifest)
    response = loaded.search(
        RetrievalQuery(
            query_id="query-sidecar",
            original_question="thuế",
            normalized_question="thuế",
            requested_strategy=RetrievalStrategy.DENSE,
            filters=RetrievalFilters(document_ids=["doc-tax"]),
        ),
        [0.0, 1.0],
    )

    assert serving_manifest.artifact_type == (
        ArtifactType.VECTOR_SERVING_METADATA
    )
    assert serving_manifest.record_count == len(vector_chunks)
    assert (sidecar / "metadata.sqlite3").is_file()
    assert isinstance(loaded._chunks, SQLiteVectorChunkStore)
    assert [hit.chunk_id for hit in response.hits] == ["chunk-tax"]


def test_prepare_refuses_overwrite_and_source_mismatch(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Sidecar creation and load fail closed for mutable or unrelated inputs."""
    vector_directory = tmp_path / "vector"
    vector_manifest = _build_vector(
        vector_directory,
        vector_chunks,
        vectors,
        vector_source_manifest,
    )
    sidecar = tmp_path / "vector-serving"
    prepare_vector_serving_metadata(
        vector_directory=vector_directory,
        destination=sidecar,
        vector_manifest=vector_manifest,
    )

    with pytest.raises(BackendInitializationError, match="already"):
        prepare_vector_serving_metadata(
            vector_directory=vector_directory,
            destination=sidecar,
            vector_manifest=vector_manifest,
        )

    wrong_manifest = vector_manifest.model_copy(
        update={"processing_config_hash": "different-vector"}
    )
    with pytest.raises(ArtifactCompatibilityError, match="incompatible"):
        SQLiteVectorChunkStore.load(
            sidecar,
            chunks_path=vector_directory / "chunks.jsonl",
            vector_manifest=wrong_manifest,
            verify_integrity=False,
        )


def test_backend_can_require_prepared_serving_metadata(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Production config can reject accidental slow JSONL fallback."""
    vector_directory = tmp_path / "vector"
    vector_manifest = _build_vector(
        vector_directory,
        vector_chunks,
        vectors,
        vector_source_manifest,
    )
    backend = NumpyVectorBackend(
        runtime_config=VectorRuntimeConfig(require_serving_metadata=True),
        serving_metadata_source=tmp_path / "missing-sidecar",
    )

    with pytest.raises(ArtifactCompatibilityError, match="Required"):
        backend.load(vector_directory, vector_manifest)
