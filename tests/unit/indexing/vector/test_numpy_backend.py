"""Unit tests for exact normalized NumPy vector retrieval."""

from datetime import UTC, datetime

import pytest

from legal_agentic_rag.contracts import VectorBackend
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.vector import NumpyVectorBackend
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    RetrievalFilters,
    RetrievalQuery,
    RetrievalStrategy,
)

MODEL = "fixture/e5"
REVISION = "fixture-revision"
PROVIDER = "fixture-provider"
PROVIDER_VERSION = "1.0"


def _backend() -> NumpyVectorBackend:
    return NumpyVectorBackend(clock=lambda: datetime(2026, 7, 22, tzinfo=UTC))


def _build(
    backend: NumpyVectorBackend,
    chunks: list[LegalChunk],
    vectors: list[list[float]],
    source_manifest: ArtifactManifest,
) -> ArtifactManifest:
    return backend.build(
        chunks,
        vectors,
        source_manifest,
        model_name=MODEL,
        model_revision=REVISION,
        embedding_provider_name=PROVIDER,
        embedding_provider_version=PROVIDER_VERSION,
        dimension=2,
        embedding_batch_size=2,
    )


def _query(
    *,
    top_k: int = 10,
    filters: RetrievalFilters | None = None,
    strategy: RetrievalStrategy | None = RetrievalStrategy.DENSE,
) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-1",
        original_question="quy định giao thông",
        normalized_question="quy định giao thông",
        filters=filters or RetrievalFilters(),
        top_k=top_k,
        candidate_k=max(top_k, 100),
        requested_strategy=strategy,
    )


def test_build_records_vector_and_model_provenance(
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Manifest stores model, dimension, metric, source hash, and backend."""
    backend = _backend()

    manifest = _build(backend, vector_chunks, vectors, vector_source_manifest)

    assert isinstance(backend, VectorBackend)
    assert manifest.artifact_type == ArtifactType.VECTOR_INDEX
    assert manifest.backend == "numpy_flat"
    assert manifest.model_name == MODEL
    assert manifest.model_revision == REVISION
    assert manifest.metadata["embedding_provider_name"] == PROVIDER
    assert manifest.metadata["embedding_provider_version"] == PROVIDER_VERSION
    assert manifest.record_count == 3
    assert manifest.metadata["dimension"] == 2
    assert manifest.metadata["distance_metric"] == "cosine"
    assert manifest.metadata["normalized_vectors"] is True
    assert manifest.metadata["embedding_batch_size"] == 2
    assert backend.dimension == 2
    assert backend.embedding_provider_name == PROVIDER
    assert backend.embedding_provider_version == PROVIDER_VERSION
    assert backend.source_artifact_identity == (
        "legal_chunks",
        "1.0",
        "chunk-hash",
    )


def test_search_ranks_cosine_similarity_and_keeps_chunk_metadata(
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Exact cosine ranking emits unified dense hits and trace fields."""
    backend = _backend()
    _build(backend, vector_chunks, vectors, vector_source_manifest)

    response = backend.search(_query(top_k=2), [1.0, 0.0])

    assert [hit.chunk_id for hit in response.hits] == [
        "chunk-speed",
        "chunk-license",
    ]
    assert response.hits[0].metadata["document_number"] == "doc-speed/2026/QH"
    assert response.hits[0].retrieval_trace.dense_rank == 1
    assert response.hits[0].retrieval_trace.dense_score == response.hits[0].score
    assert response.strategy == RetrievalStrategy.DENSE
    assert response.artifact_versions == {"vector_index": "1.0"}
    assert response.latency_ms >= 0


def test_search_applies_unified_filters_and_top_k(
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Dense retrieval uses the same exact metadata filter contract as BM25."""
    backend = _backend()
    _build(backend, vector_chunks, vectors, vector_source_manifest)

    response = backend.search(
        _query(
            top_k=1,
            filters=RetrievalFilters(
                legal_fields=["Thuế"], effect_statuses=["Hết hiệu lực"]
            ),
        ),
        [0.0, 1.0],
    )

    assert [hit.chunk_id for hit in response.hits] == ["chunk-tax"]


def test_build_and_tie_order_are_deterministic(
    vector_chunks: list[LegalChunk],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Input order does not affect artifact hash or equal-score ordering."""
    equal_vectors = [[1.0, 0.0]] * 3
    first = _backend()
    second = _backend()

    first_manifest = _build(
        first, vector_chunks, equal_vectors, vector_source_manifest
    )
    second_manifest = _build(
        second,
        list(reversed(vector_chunks)),
        list(reversed(equal_vectors)),
        vector_source_manifest,
    )

    assert first_manifest.processing_config_hash == (
        second_manifest.processing_config_hash
    )
    assert [hit.chunk_id for hit in first.search(_query(), [1.0, 0.0]).hits] == [
        "chunk-license",
        "chunk-speed",
        "chunk-tax",
    ]


def test_empty_index_and_empty_filter_result_are_valid(
    vector_source_manifest: ArtifactManifest,
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
) -> None:
    """Empty candidate sets return warnings instead of backend failures."""
    empty = _backend()
    empty_manifest = vector_source_manifest.model_copy(update={"record_count": 0})
    _build(empty, [], [], empty_manifest)
    assert empty.search(_query(), [1.0, 0.0]).warnings == ["no_dense_matches"]

    filtered = _backend()
    _build(filtered, vector_chunks, vectors, vector_source_manifest)
    response = filtered.search(
        _query(filters=RetrievalFilters(document_ids=["missing"])), [1.0, 0.0]
    )
    assert response.hits == []
    assert response.warnings == ["no_dense_matches"]


def test_backend_rejects_invalid_inputs_and_state(
    vector_chunks: list[LegalChunk],
    vectors: list[list[float]],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Source, vector shape/value, query dimension, and strategy are validated."""
    backend = _backend()
    with pytest.raises(BackendInitializationError):
        backend.search(_query(), [1.0, 0.0])

    wrong_type = vector_source_manifest.model_copy(
        update={"artifact_type": ArtifactType.LEGAL_BLOCKS}
    )
    with pytest.raises(ArtifactCompatibilityError):
        _build(backend, vector_chunks, vectors, wrong_type)

    with pytest.raises(DataValidationError, match="shape"):
        _build(backend, vector_chunks, [[1.0, 0.0]], vector_source_manifest)
    invalid_values = [[1.0, 0.0], [float("nan"), 1.0], [0.0, 1.0]]
    with pytest.raises(DataValidationError, match="non-finite"):
        _build(backend, vector_chunks, invalid_values, vector_source_manifest)
    zero_values = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    with pytest.raises(DataValidationError, match="zero"):
        _build(backend, vector_chunks, zero_values, vector_source_manifest)

    _build(backend, vector_chunks, vectors, vector_source_manifest)
    with pytest.raises(RetrievalError, match="shape"):
        backend.search(_query(), [1.0])
    with pytest.raises(RetrievalError, match="zero"):
        backend.search(_query(), [0.0, 0.0])
    with pytest.raises(RetrievalError, match="non-dense"):
        backend.search(_query(strategy=RetrievalStrategy.BM25), [1.0, 0.0])
