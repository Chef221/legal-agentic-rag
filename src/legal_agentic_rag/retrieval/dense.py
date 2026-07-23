"""Online dense retrieval orchestration for query embedding and vector search."""

import logging
from time import perf_counter

from legal_agentic_rag.contracts.embedding_provider import EmbeddingProvider
from legal_agentic_rag.contracts.vector_backend import VectorBackend
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)


class DenseRetriever:
    """Embed one normalized query and search a compatible vector backend."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        backend: VectorBackend,
    ) -> None:
        self._provider = provider
        self._backend = backend

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Expose the vector index source identity for hybrid compatibility."""
        return self._backend.source_artifact_identity

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Return dense hits while including query embedding in total latency."""
        if query.requested_strategy not in (None, RetrievalStrategy.DENSE):
            raise RetrievalError("Dense retriever received a non-dense request")
        self._validate_compatibility()
        started = perf_counter()
        effective_query = query.rewritten_question or query.normalized_question
        query_vector = self._provider.embed_query(effective_query)
        response = self._backend.search(query, query_vector)
        total_latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "dense_retrieval_completed",
            extra={
                "query_id": query.query_id,
                "strategy": response.strategy.value,
                "candidate_count": len(response.hits),
                "latency_ms": total_latency_ms,
            },
        )
        return response.model_copy(update={"latency_ms": total_latency_ms})

    def _validate_compatibility(self) -> None:
        if self._provider.provider_name != self._backend.embedding_provider_name:
            raise ArtifactCompatibilityError(
                "Dense embedding provider does not match vector artifact"
            )
        if self._provider.provider_version != self._backend.embedding_provider_version:
            raise ArtifactCompatibilityError(
                "Dense embedding provider version does not match vector artifact"
            )
        if self._provider.model_name != self._backend.model_name:
            raise ArtifactCompatibilityError(
                "Dense provider model does not match vector artifact"
            )
        if self._provider.model_revision != self._backend.model_revision:
            raise ArtifactCompatibilityError(
                "Dense provider revision does not match vector artifact"
            )
        if self._provider.dimension != self._backend.dimension:
            raise ArtifactCompatibilityError(
                "Dense provider dimension does not match vector artifact"
            )
