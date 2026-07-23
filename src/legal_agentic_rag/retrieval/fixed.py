"""Fixed sparse, dense, and hybrid retrieval orchestration."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from legal_agentic_rag.configuration.online import RerankerConfig, RetrievalConfig
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.contracts.graph_backend import GraphBackend
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.retrieval.graph import GraphExpandedRetriever
from legal_agentic_rag.retrieval.rerank import RerankingRetriever
from legal_agentic_rag.retrieval.rrf import reciprocal_rank_fusion
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest

_LOGGER = logging.getLogger(__name__)


class _RetrievalBranch(Protocol):
    @property
    def source_artifact_identity(self) -> tuple[str, str, str]: ...

    def search(self, query: RetrievalQuery) -> RetrievalResponse: ...


class HybridRetriever:
    """Run compatible BM25 and dense branches, then fuse their ranks."""

    def __init__(
        self,
        bm25_retriever: _RetrievalBranch,
        dense_retriever: _RetrievalBranch,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._config = config or RetrievalConfig()

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return the common legal-chunks identity behind both branches."""
        sparse_identity = self._bm25.source_artifact_identity
        dense_identity = self._dense.source_artifact_identity
        if sparse_identity != dense_identity:
            raise ArtifactCompatibilityError(
                "BM25 and vector indexes originate from different chunk artifacts"
            )
        return sparse_identity

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Return deterministic hybrid hits with per-branch RRF contributions."""
        if query.requested_strategy not in (None, RetrievalStrategy.HYBRID):
            raise RetrievalError("Hybrid retriever received a non-hybrid request")
        _ = self.source_artifact_identity
        started = perf_counter()
        branch_base = query.model_copy(
            update={"top_k": query.candidate_k, "candidate_k": query.candidate_k}
        )
        bm25_response = self._bm25.search(
            branch_base.model_copy(
                update={"requested_strategy": RetrievalStrategy.BM25}
            )
        )
        dense_response = self._dense.search(
            branch_base.model_copy(
                update={"requested_strategy": RetrievalStrategy.DENSE}
            )
        )
        self._validate_branch_response(
            bm25_response, query.query_id, RetrievalStrategy.BM25
        )
        self._validate_branch_response(
            dense_response, query.query_id, RetrievalStrategy.DENSE
        )
        hits = reciprocal_rank_fusion(
            bm25_response.hits,
            dense_response.hits,
            rrf_constant=self._config.rrf_constant,
            top_k=query.top_k,
        )
        warnings = self._warnings(bm25_response, dense_response, bool(hits))
        artifact_versions = self._artifact_versions(
            bm25_response, dense_response
        )
        latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "hybrid_retrieval_completed",
            extra={
                "query_id": query.query_id,
                "strategy": RetrievalStrategy.HYBRID.value,
                "candidate_count": len(hits),
                "bm25_candidate_count": len(bm25_response.hits),
                "dense_candidate_count": len(dense_response.hits),
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResponse(
            query=query.model_copy(
                update={"requested_strategy": RetrievalStrategy.HYBRID}
            ),
            strategy=RetrievalStrategy.HYBRID,
            hits=hits,
            latency_ms=latency_ms,
            warnings=warnings,
            artifact_versions=artifact_versions,
        )

    @staticmethod
    def _validate_branch_response(
        response: RetrievalResponse,
        query_id: str,
        strategy: RetrievalStrategy,
    ) -> None:
        if response.strategy != strategy or response.query.query_id != query_id:
            raise RetrievalError("Retrieval branch returned an incompatible response")

    @staticmethod
    def _warnings(
        bm25: RetrievalResponse,
        dense: RetrievalResponse,
        has_hits: bool,
    ) -> list[str]:
        warnings = [f"bm25:{warning}" for warning in bm25.warnings]
        warnings.extend(f"dense:{warning}" for warning in dense.warnings)
        if not has_hits:
            warnings.append("no_hybrid_matches")
        return list(dict.fromkeys(warnings))

    @staticmethod
    def _artifact_versions(
        bm25: RetrievalResponse,
        dense: RetrievalResponse,
    ) -> dict[str, str]:
        versions = dict(bm25.artifact_versions)
        for name, version in dense.artifact_versions.items():
            if name in versions and versions[name] != version:
                raise ArtifactCompatibilityError(
                    "Retrieval branches report conflicting artifact versions"
                )
            versions[name] = version
        return versions


class FixedRetriever:
    """Route explicitly among the completed fixed retrieval strategies."""

    def __init__(
        self,
        bm25_retriever: _RetrievalBranch,
        dense_retriever: _RetrievalBranch,
        config: RetrievalConfig | None = None,
        *,
        reranker: Reranker | None = None,
        reranker_config: RerankerConfig | None = None,
        graph_backend: GraphBackend | None = None,
        chunk_manifest: ArtifactManifest | None = None,
    ) -> None:
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._config = config or RetrievalConfig()
        self._hybrid = HybridRetriever(bm25_retriever, dense_retriever, self._config)
        self._hybrid_rerank = (
            RerankingRetriever(self._hybrid, reranker, reranker_config)
            if reranker is not None
            else None
        )
        self._graph = (
            GraphExpandedRetriever(
                self._hybrid,
                graph_backend,
                reranker,
                chunk_manifest,
                self._config,
                reranker_config,
            )
            if (
                graph_backend is not None
                and reranker is not None
                and chunk_manifest is not None
            )
            else None
        )

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Run BM25, dense, or hybrid retrieval without an Agent."""
        strategy = query.requested_strategy or self._config.default_strategy
        routed_query = query.model_copy(update={"requested_strategy": strategy})
        if strategy == RetrievalStrategy.BM25:
            return self._bm25.search(routed_query)
        if strategy == RetrievalStrategy.DENSE:
            return self._dense.search(routed_query)
        if strategy == RetrievalStrategy.HYBRID:
            return self._hybrid.search(routed_query)
        if strategy == RetrievalStrategy.HYBRID_RERANK:
            if self._hybrid_rerank is None:
                raise RetrievalError("Fixed hybrid-rerank strategy has no reranker")
            return self._hybrid_rerank.search(routed_query)
        if strategy == RetrievalStrategy.GRAPH:
            if self._graph is None:
                raise RetrievalError(
                    "Fixed graph strategy requires graph, chunks, and reranker"
                )
            return self._graph.search(routed_query)
        raise RetrievalError(f"Fixed retrieval strategy is not implemented: {strategy}")
