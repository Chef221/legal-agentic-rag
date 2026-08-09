"""Orchestration from hybrid candidates to cross-encoder final ranking."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from legal_agentic_rag.configuration.online import RerankerConfig
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.retrieval.rerank_validation import (
    validate_reranked_response,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)


class _CandidateRetriever(Protocol):
    @property
    def source_artifact_identity(self) -> tuple[str, str, str]: ...

    def search(self, query: RetrievalQuery) -> RetrievalResponse: ...


class RerankingRetriever:
    """Retrieve bounded hybrid candidates and rerank them to final top-k."""

    def __init__(
        self,
        candidate_retriever: _CandidateRetriever,
        reranker: Reranker,
        config: RerankerConfig | None = None,
    ) -> None:
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._config = config or RerankerConfig()

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return the source identity of the underlying candidate retriever."""
        return self._candidate_retriever.source_artifact_identity

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Return hybrid-rerank hits while preserving retrieval provenance."""
        if query.requested_strategy not in (
            None,
            RetrievalStrategy.HYBRID_RERANK,
        ):
            raise RetrievalError("Reranking retriever received an incompatible request")
        if query.candidate_k > self._config.max_candidates:
            raise RetrievalError("Query candidate-k exceeds the reranker limit")
        candidate_query = query.model_copy(
            update={
                "top_k": query.candidate_k,
                "requested_strategy": RetrievalStrategy.HYBRID,
            }
        )
        candidate_response = self._candidate_retriever.search(candidate_query)
        if (
            candidate_response.strategy != RetrievalStrategy.HYBRID
            or candidate_response.query.query_id != query.query_id
            or len(candidate_response.hits) > query.candidate_k
        ):
            raise RetrievalError("Candidate retriever returned an incompatible response")
        return self.rerank_candidates(query, candidate_response)

    def rerank_candidates(
        self,
        query: RetrievalQuery,
        candidate_response: RetrievalResponse,
    ) -> RetrievalResponse:
        """Rerank an already retrieved compatible hybrid candidate response."""
        if query.requested_strategy not in (
            None,
            RetrievalStrategy.HYBRID_RERANK,
        ):
            raise RetrievalError("Reranking retriever received an incompatible request")
        if query.candidate_k > self._config.max_candidates:
            raise RetrievalError("Query candidate-k exceeds the reranker limit")
        if (
            candidate_response.strategy != RetrievalStrategy.HYBRID
            or candidate_response.query.query_id != query.query_id
            or len(candidate_response.hits) > query.candidate_k
        ):
            raise RetrievalError("Candidate retriever returned an incompatible response")
        started = perf_counter()
        rerank_query = query.model_copy(
            update={"requested_strategy": RetrievalStrategy.RERANK}
        )
        reranked = self._reranker.rerank(rerank_query, candidate_response.hits)
        validate_reranked_response(
            reranked,
            candidate_response.hits,
            rerank_query,
        )
        hits = [
            hit.model_copy(update={"strategy": RetrievalStrategy.HYBRID_RERANK})
            for hit in reranked.hits
        ]
        warnings = list(candidate_response.warnings)
        warnings.extend(f"reranker:{warning}" for warning in reranked.warnings)
        reranker_latency_ms = (perf_counter() - started) * 1000
        latency_ms = candidate_response.latency_ms + reranker_latency_ms
        _LOGGER.info(
            "hybrid_rerank_completed",
            extra={
                "query_id": query.query_id,
                "strategy": RetrievalStrategy.HYBRID_RERANK.value,
                "candidate_count": len(candidate_response.hits),
                "selected_count": len(hits),
                "model_name": self._reranker.model_name,
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResponse(
            query=query.model_copy(
                update={"requested_strategy": RetrievalStrategy.HYBRID_RERANK}
            ),
            strategy=RetrievalStrategy.HYBRID_RERANK,
            hits=hits,
            latency_ms=latency_ms,
            warnings=list(dict.fromkeys(warnings)),
            artifact_versions=candidate_response.artifact_versions,
        )
