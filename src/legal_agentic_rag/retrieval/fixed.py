"""Fixed sparse, dense, and hybrid retrieval orchestration."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from legal_agentic_rag.configuration.online import (
    QueryUnderstandingConfig,
    RerankerConfig,
    RetrievalConfig,
)
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.contracts.graph_backend import GraphBackend
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.retrieval.graph import GraphExpandedRetriever
from legal_agentic_rag.retrieval.multi_query import (
    QueryBranchResult,
    fuse_query_branches,
)
from legal_agentic_rag.retrieval.rerank import RerankingRetriever
from legal_agentic_rag.retrieval.rrf import reciprocal_rank_fusion
from legal_agentic_rag.schemas.retrieval import (
    QueryVariant,
    QueryVariantKind,
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
        query_understanding_config: QueryUnderstandingConfig | None = None,
    ) -> None:
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._config = config or RetrievalConfig()
        self._query_config = (
            query_understanding_config or QueryUnderstandingConfig()
        )

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
        return self.search_with_primary_branches(query)[2]

    def search_with_primary_branches(
        self,
        query: RetrievalQuery,
    ) -> tuple[RetrievalResponse, RetrievalResponse, RetrievalResponse]:
        """Return primary sparse/dense responses with their fused response.

        The comparison path uses this method to observe the branches and fusion
        without repeating the same backend searches. Additional bounded query
        variants are still executed exactly once when multi-query is enabled.
        """
        if query.requested_strategy not in (None, RetrievalStrategy.HYBRID):
            raise RetrievalError("Hybrid retriever received a non-hybrid request")
        _ = self.source_artifact_identity
        started = perf_counter()
        variants = self._active_variants(query)
        branch_results: list[QueryBranchResult] = []
        response_pairs: list[
            tuple[str, RetrievalResponse, RetrievalResponse]
        ] = []
        for variant in variants:
            branch_base = query.model_copy(
                update={
                    "top_k": query.candidate_k,
                    "candidate_k": query.candidate_k,
                    "rewritten_question": (
                        variant.text
                        if variant.text != query.normalized_question
                        else None
                    ),
                }
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
            response_pairs.append(
                (variant.variant_id, bm25_response, dense_response)
            )
            branch_results.extend(
                (
                    QueryBranchResult(
                        variant.variant_id,
                        RetrievalStrategy.BM25,
                        bm25_response.hits,
                    ),
                    QueryBranchResult(
                        variant.variant_id,
                        RetrievalStrategy.DENSE,
                        dense_response.hits,
                    ),
                )
            )

        if len(variants) == 1:
            _, bm25_response, dense_response = response_pairs[0]
            hits = reciprocal_rank_fusion(
                bm25_response.hits,
                dense_response.hits,
                rrf_constant=self._config.rrf_constant,
                top_k=query.top_k,
            )
            warnings = self._warnings(
                bm25_response,
                dense_response,
                bool(hits),
            )
        else:
            hits = fuse_query_branches(
                branch_results,
                rrf_constant=self._config.rrf_constant,
                top_k=query.top_k,
            )
            warnings = self._multi_query_warnings(response_pairs, bool(hits))
        artifact_versions = self._multi_query_artifact_versions(response_pairs)
        latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "hybrid_retrieval_completed",
            extra={
                "query_id": query.query_id,
                "strategy": RetrievalStrategy.HYBRID.value,
                "candidate_count": len(hits),
                "query_variant_count": len(variants),
                "bm25_candidate_count": sum(
                    len(pair[1].hits) for pair in response_pairs
                ),
                "dense_candidate_count": sum(
                    len(pair[2].hits) for pair in response_pairs
                ),
                "latency_ms": latency_ms,
            },
        )
        hybrid_response = RetrievalResponse(
            query=query.model_copy(
                update={"requested_strategy": RetrievalStrategy.HYBRID}
            ),
            strategy=RetrievalStrategy.HYBRID,
            hits=hits,
            latency_ms=latency_ms,
            warnings=warnings,
            artifact_versions=artifact_versions,
        )
        _, primary_bm25, primary_dense = response_pairs[0]
        return primary_bm25, primary_dense, hybrid_response

    def _active_variants(self, query: RetrievalQuery) -> list[QueryVariant]:
        effective_query = query.rewritten_question or query.normalized_question
        if (
            self._query_config.multi_query_enabled
            and query.rewritten_question is None
            and query.query_variants
        ):
            return query.query_variants[: self._query_config.max_variants]
        return [
            QueryVariant(
                variant_id="qv-active",
                text=effective_query,
                kind=QueryVariantKind.NORMALIZED,
            )
        ]

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

    @classmethod
    def _multi_query_artifact_versions(
        cls,
        responses: list[tuple[str, RetrievalResponse, RetrievalResponse]],
    ) -> dict[str, str]:
        versions: dict[str, str] = {}
        for _, bm25, dense in responses:
            pair_versions = cls._artifact_versions(bm25, dense)
            for name, version in pair_versions.items():
                if name in versions and versions[name] != version:
                    raise ArtifactCompatibilityError(
                        "Query variants report conflicting artifact versions"
                    )
                versions[name] = version
        return versions

    @staticmethod
    def _multi_query_warnings(
        responses: list[tuple[str, RetrievalResponse, RetrievalResponse]],
        has_hits: bool,
    ) -> list[str]:
        warnings: list[str] = []
        for variant_id, bm25, dense in responses:
            warnings.extend(
                f"{variant_id}:bm25:{warning}" for warning in bm25.warnings
            )
            warnings.extend(
                f"{variant_id}:dense:{warning}" for warning in dense.warnings
            )
        if not has_hits:
            warnings.append("no_hybrid_matches")
        return list(dict.fromkeys(warnings))


class FixedRetriever:
    """Route explicitly among the completed fixed retrieval strategies."""

    def __init__(
        self,
        bm25_retriever: _RetrievalBranch,
        dense_retriever: _RetrievalBranch,
        config: RetrievalConfig | None = None,
        *,
        query_understanding_config: QueryUnderstandingConfig | None = None,
        reranker: Reranker | None = None,
        reranker_config: RerankerConfig | None = None,
        graph_backend: GraphBackend | None = None,
        chunk_manifest: ArtifactManifest | None = None,
    ) -> None:
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._config = config or RetrievalConfig()
        self._hybrid = HybridRetriever(
            bm25_retriever,
            dense_retriever,
            self._config,
            query_understanding_config,
        )
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

    def search_comparison(
        self,
        query: RetrievalQuery,
        *,
        include_reranker: bool = False,
    ) -> list[RetrievalResponse]:
        """Compare fixed branches while executing each backend search once.

        Sparse and dense branches are requested at ``candidate_k`` so the same
        results can feed RRF and reranking. Their diagnostic projections are
        then truncated to ``top_k`` without changing rank order or provenance.
        """
        candidate_query = query.model_copy(
            update={
                "top_k": query.candidate_k,
                "requested_strategy": RetrievalStrategy.HYBRID,
            }
        )
        bm25_candidates, dense_candidates, hybrid_candidates = (
            self._hybrid.search_with_primary_branches(candidate_query)
        )
        responses = [
            _truncate_response(bm25_candidates, query, RetrievalStrategy.BM25),
            _truncate_response(dense_candidates, query, RetrievalStrategy.DENSE),
            _truncate_response(hybrid_candidates, query, RetrievalStrategy.HYBRID),
        ]
        if include_reranker:
            if self._hybrid_rerank is None:
                raise RetrievalError("Fixed comparison has no reranker")
            responses.append(
                self._hybrid_rerank.rerank_candidates(query, hybrid_candidates)
            )
        return responses


def _truncate_response(
    response: RetrievalResponse,
    query: RetrievalQuery,
    strategy: RetrievalStrategy,
) -> RetrievalResponse:
    """Project a candidate response to the caller's final comparison limit."""
    return response.model_copy(
        update={
            "query": query.model_copy(update={"requested_strategy": strategy}),
            "strategy": strategy,
            "hits": response.hits[: query.top_k],
        }
    )
