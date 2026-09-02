"""V2 retrieval branch adapters for fixed BM25, dense, and hybrid orchestration."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any, Protocol

import numpy as np

from legal_agentic_rag.configuration.online import (
    QueryUnderstandingConfig,
    RerankerConfig,
    RetrievalConfig,
)
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.indexing.bm25.v2_backend import V2SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.vector.v2_precomputed_backend import (
    V2PrecomputedDenseBackend,
)
from legal_agentic_rag.retrieval.fixed import FixedRetriever, HybridRetriever
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)


class _QueryEmbeddingProvider(Protocol):
    """Protocol for compatible query embedding providers."""

    @property
    def model_name(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_query(self, text: str) -> Sequence[float] | np.ndarray: ...


class V2BM25RetrievalBranch:
    """Adapts V2 SQLite FTS5 backend to the common retrieval branch interface."""

    def __init__(self, backend: V2SQLiteFTS5BM25Backend) -> None:
        self._backend = backend

    @property
    def backend(self) -> V2SQLiteFTS5BM25Backend:
        return self._backend

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return truthful V2 retrieval units lineage (dataset, row count, source SHA256)."""
        manifest = self._backend.manifest
        record_count = self._backend.record_count
        source_sha = manifest.get("source_retrieval_units_sha256")

        if record_count <= 0 or not isinstance(source_sha, str) or not source_sha.strip():
            raise ArtifactCompatibilityError(
                "V2 BM25 backend manifest is missing valid record count or source retrieval units SHA256"
            )

        return (
            "retrieval_units_v2",
            str(record_count),
            source_sha.strip(),
        )

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Execute BM25 search through the V2 backend."""
        if query.requested_strategy not in (None, RetrievalStrategy.BM25):
            raise RetrievalError("V2 BM25 retrieval branch received a non-BM25 request")
        return self._backend.search(query)


class V2DenseRetrievalBranch:
    """Adapts V2 precomputed dense backend and query embedding provider to retrieval branch interface."""

    def __init__(
        self,
        backend: V2PrecomputedDenseBackend,
        embedding_provider: _QueryEmbeddingProvider,
    ) -> None:
        # Compatibility gate: verify provider matches dense matrix specification
        prov_model = getattr(embedding_provider, "model_name", None)
        prov_rev = getattr(embedding_provider, "model_revision", None)
        prov_dim = getattr(embedding_provider, "dimension", None)

        if prov_model != backend.model_name:
            raise ArtifactCompatibilityError(
                f"Embedding provider model '{prov_model}' does not match dense backend model '{backend.model_name}'"
            )
        if prov_rev != backend.model_revision:
            raise ArtifactCompatibilityError(
                f"Embedding provider revision '{prov_rev}' does not match dense backend revision '{backend.model_revision}'"
            )
        if prov_dim != backend.dimension:
            raise ArtifactCompatibilityError(
                f"Embedding provider dimension {prov_dim} does not match dense backend dimension {backend.dimension}"
            )

        self._backend = backend
        self._provider = embedding_provider

    @property
    def backend(self) -> V2PrecomputedDenseBackend:
        return self._backend

    @property
    def embedding_provider(self) -> _QueryEmbeddingProvider:
        return self._provider

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return truthful V2 retrieval units lineage (dataset, row count, source SHA256)."""
        manifest = self._backend.manifest
        record_count = self._backend.record_count
        source_sha = manifest.get("source_retrieval_units_sha256")

        if record_count <= 0 or not isinstance(source_sha, str) or not source_sha.strip():
            raise ArtifactCompatibilityError(
                "V2 Dense backend manifest is missing valid record count or source retrieval units SHA256"
            )

        return (
            "retrieval_units_v2",
            str(record_count),
            source_sha.strip(),
        )

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Encode query text and retrieve exact cosine hits from precomputed matrix."""
        if query.requested_strategy not in (None, RetrievalStrategy.DENSE):
            raise RetrievalError("V2 Dense retrieval branch received a non-dense request")

        effective_query = query.rewritten_question or query.normalized_question
        query_vector = self._provider.embed_query(effective_query)
        return self._backend.retrieve(query, query_vector)


def build_v2_fixed_retriever(
    bm25_backend: V2SQLiteFTS5BM25Backend,
    dense_backend: V2PrecomputedDenseBackend,
    embedding_provider: _QueryEmbeddingProvider,
    *,
    retrieval_config: RetrievalConfig | None = None,
    query_understanding_config: QueryUnderstandingConfig | None = None,
    reranker: Reranker | None = None,
    reranker_config: RerankerConfig | None = None,
) -> FixedRetriever:
    """Build a standard FixedRetriever backed by V2 BM25 and Dense branches with optional reranker."""
    bm25_branch = V2BM25RetrievalBranch(bm25_backend)
    dense_branch = V2DenseRetrievalBranch(dense_backend, embedding_provider)

    # Lineage check
    if bm25_branch.source_artifact_identity != dense_branch.source_artifact_identity:
        raise ArtifactCompatibilityError(
            f"V2 BM25 and Dense branches originate from different source artifacts: "
            f"{bm25_branch.source_artifact_identity} != {dense_branch.source_artifact_identity}"
        )

    return FixedRetriever(
        bm25_retriever=bm25_branch,
        dense_retriever=dense_branch,
        config=retrieval_config,
        query_understanding_config=query_understanding_config,
        reranker=reranker,
        reranker_config=reranker_config,
    )
