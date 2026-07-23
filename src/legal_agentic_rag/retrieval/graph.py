"""Bounded graph expansion over text-retrieval seed documents."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from legal_agentic_rag.configuration.online import RerankerConfig, RetrievalConfig
from legal_agentic_rag.contracts.graph_backend import GraphBackend
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.retrieval.rerank_validation import (
    validate_reranked_response,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    GraphPathStep,
    RetrievalFilters,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)


class _HybridCandidateRetriever(Protocol):
    @property
    def source_artifact_identity(self) -> tuple[str, str, str]: ...

    def search(self, query: RetrievalQuery) -> RetrievalResponse: ...


class GraphExpandedRetriever:
    """Expand hybrid seeds through legal relationships, then rerank once."""

    def __init__(
        self,
        candidate_retriever: _HybridCandidateRetriever,
        graph_backend: GraphBackend,
        reranker: Reranker,
        chunk_manifest: ArtifactManifest,
        retrieval_config: RetrievalConfig | None = None,
        reranker_config: RerankerConfig | None = None,
    ) -> None:
        self._candidate_retriever = candidate_retriever
        self._graph = graph_backend
        self._reranker = reranker
        self._chunk_manifest = chunk_manifest
        self._retrieval_config = retrieval_config or RetrievalConfig()
        self._reranker_config = reranker_config or RerankerConfig()

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Return graph-expanded and cross-encoder-reranked legal chunks."""
        if query.requested_strategy not in (None, RetrievalStrategy.GRAPH):
            raise RetrievalError("Graph retriever received an incompatible request")
        if query.candidate_k > self._reranker_config.max_candidates:
            raise RetrievalError("Query candidate-k exceeds the reranker limit")
        self._validate_artifacts()
        started = perf_counter()
        maximum_seed_slots = (
            query.candidate_k - 1 if query.candidate_k > 1 else 1
        )
        seed_limit = min(
            self._retrieval_config.graph_seed_chunk_k,
            maximum_seed_slots,
        )
        seed_query = query.model_copy(
            update={
                "top_k": seed_limit,
                "requested_strategy": RetrievalStrategy.HYBRID,
            }
        )
        seed_response = self._candidate_retriever.search(seed_query)
        self._validate_hybrid_response(seed_response, seed_query)
        seed_hits = seed_response.hits[:seed_limit]
        seed_document_ids = self._seed_documents(seed_hits)
        steps = list(
            self._graph.traverse(
                seed_document_ids,
                self._retrieval_config.graph_hop_limit,
                self._retrieval_config.graph_relationship_types or None,
            )
        )
        paths = self._paths_by_target(seed_document_ids, steps)
        target_document_ids = self._target_documents(
            query.filters,
            seed_document_ids,
            steps,
        )
        remaining = max(0, query.candidate_k - len(seed_hits))
        related_response = self._related_candidates(
            query,
            target_document_ids,
            remaining,
        )
        related_hits = self._with_graph_trace(
            related_response.hits if related_response is not None else [],
            paths,
        )
        candidates = self._deduplicate(seed_hits + related_hits)[: query.candidate_k]
        reranked_hits, reranker_warnings = self._rerank(query, candidates)
        warnings = [f"seed:{warning}" for warning in seed_response.warnings]
        if related_response is not None:
            warnings.extend(
                f"graph_related:{warning}" for warning in related_response.warnings
            )
        warnings.extend(f"reranker:{warning}" for warning in reranker_warnings)
        if seed_hits and not steps:
            warnings.append("no_graph_expansion")
        if not seed_hits:
            warnings.append("no_graph_seed_matches")
        artifact_versions = dict(seed_response.artifact_versions)
        if related_response is not None:
            artifact_versions.update(related_response.artifact_versions)
        artifact_versions["graph_index"] = self._graph.manifest.artifact_version
        latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "graph_retrieval_completed",
            extra={
                "query_id": query.query_id,
                "strategy": RetrievalStrategy.GRAPH.value,
                "seed_document_count": len(seed_document_ids),
                "graph_path_count": len(steps),
                "related_document_count": len(target_document_ids),
                "candidate_count": len(candidates),
                "selected_count": len(reranked_hits),
                "graph_hop": self._retrieval_config.graph_hop_limit,
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResponse(
            query=query.model_copy(
                update={"requested_strategy": RetrievalStrategy.GRAPH}
            ),
            strategy=RetrievalStrategy.GRAPH,
            hits=[
                hit.model_copy(update={"strategy": RetrievalStrategy.GRAPH})
                for hit in reranked_hits
            ],
            latency_ms=latency_ms,
            warnings=list(dict.fromkeys(warnings)),
            artifact_versions=artifact_versions,
        )

    def _validate_artifacts(self) -> None:
        identity = self._candidate_retriever.source_artifact_identity
        expected_identity = (
            self._chunk_manifest.artifact_type.value,
            self._chunk_manifest.artifact_version,
            self._chunk_manifest.processing_config_hash,
        )
        graph_manifest = self._graph.manifest
        if (
            self._chunk_manifest.artifact_type != ArtifactType.LEGAL_CHUNKS
            or identity != expected_identity
        ):
            raise ArtifactCompatibilityError(
                "Graph retrieval chunk artifact is incompatible with text indexes"
            )
        if (
            graph_manifest.artifact_type != ArtifactType.GRAPH_INDEX
            or graph_manifest.dataset_name != self._chunk_manifest.dataset_name
            or graph_manifest.dataset_revision
            != self._chunk_manifest.dataset_revision
        ):
            raise ArtifactCompatibilityError(
                "Graph and chunk artifacts originate from different datasets"
            )

    @staticmethod
    def _validate_hybrid_response(
        response: RetrievalResponse,
        query: RetrievalQuery,
    ) -> None:
        if (
            response.strategy != RetrievalStrategy.HYBRID
            or response.query != query
            or len(response.hits) > query.top_k
        ):
            raise RetrievalError(
                "Graph candidate retriever returned an incompatible response"
            )

    def _seed_documents(self, hits: list[RetrievalHit]) -> list[str]:
        documents: list[str] = []
        for hit in hits:
            if hit.document_id not in documents:
                documents.append(hit.document_id)
            if len(documents) >= self._retrieval_config.graph_seed_document_k:
                break
        return documents

    @staticmethod
    def _paths_by_target(
        seed_document_ids: list[str],
        steps: list[GraphPathStep],
    ) -> dict[str, list[GraphPathStep]]:
        paths: dict[str, list[GraphPathStep]] = {
            document_id: [] for document_id in seed_document_ids
        }
        for step in steps:
            source_path = paths.get(step.source_document_id)
            if source_path is None or step.target_document_id in paths:
                raise RetrievalError("Graph backend returned invalid BFS discovery edges")
            paths[step.target_document_id] = [*source_path, step]
        return paths

    def _target_documents(
        self,
        filters: RetrievalFilters,
        seed_document_ids: list[str],
        steps: list[GraphPathStep],
    ) -> list[str]:
        allowed_ids = set(filters.document_ids) if filters.document_ids else None
        seeds = set(seed_document_ids)
        targets: list[str] = []
        for step in steps:
            target = step.target_document_id
            if (
                target not in seeds
                and target not in targets
                and (allowed_ids is None or target in allowed_ids)
            ):
                targets.append(target)
            if len(targets) >= self._retrieval_config.graph_related_document_k:
                break
        return targets

    def _related_candidates(
        self,
        query: RetrievalQuery,
        target_document_ids: list[str],
        remaining: int,
    ) -> RetrievalResponse | None:
        if not target_document_ids or remaining <= 0:
            return None
        related_filters = query.filters.model_copy(
            update={"document_ids": target_document_ids}
        )
        related_query = query.model_copy(
            update={
                "filters": related_filters,
                "top_k": remaining,
                "requested_strategy": RetrievalStrategy.HYBRID,
            }
        )
        response = self._candidate_retriever.search(related_query)
        self._validate_hybrid_response(response, related_query)
        return response

    @staticmethod
    def _with_graph_trace(
        hits: list[RetrievalHit],
        paths: dict[str, list[GraphPathStep]],
    ) -> list[RetrievalHit]:
        enriched: list[RetrievalHit] = []
        for hit in hits:
            path = paths.get(hit.document_id)
            if not path:
                raise RetrievalError(
                    "Graph-related retrieval returned a document without a path"
                )
            trace = hit.retrieval_trace.model_copy(
                update={"graph_hop": path[-1].hop, "graph_path": path}
            )
            enriched.append(hit.model_copy(update={"retrieval_trace": trace}))
        return enriched

    @staticmethod
    def _deduplicate(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        by_id: dict[str, RetrievalHit] = {}
        for hit in hits:
            by_id.setdefault(hit.chunk_id, hit)
        return list(by_id.values())

    def _rerank(
        self,
        query: RetrievalQuery,
        candidates: list[RetrievalHit],
    ) -> tuple[list[RetrievalHit], list[str]]:
        if not candidates:
            return [], []
        rerank_query = query.model_copy(
            update={"requested_strategy": RetrievalStrategy.RERANK}
        )
        response = self._reranker.rerank(rerank_query, candidates)
        validate_reranked_response(response, candidates, rerank_query)
        return response.hits, response.warnings
