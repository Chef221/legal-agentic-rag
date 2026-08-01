"""Load compatible artifacts and compose the complete online Agent runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Protocol

from legal_agentic_rag.agent import (
    DeterministicAgentWorkflow,
    DeterministicStrategyRouter,
)
from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.contracts import (
    AgentWorkflow,
    AnswerGenerator,
    CitationVerifier,
    ContextGrader,
    EmbeddingProvider,
    Reranker,
)
from legal_agentic_rag.embeddings import SentenceTransformerEmbeddingProvider
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.generation import (
    RuleBasedContextGrader,
    build_answer_generator,
    build_citation_verifier,
    build_generation_components,
)
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.indexing.vector import NumpyVectorBackend
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import (
    DenseRetriever,
    FixedRetriever,
    QueryUnderstandingService,
)
from legal_agentic_rag.schemas import (
    AgentRunResult,
    ArtifactManifest,
    ArtifactType,
    RetrievalQuery,
    RetrievalResponse,
    ToolDescriptor,
)
from legal_agentic_rag.tools import ToolRegistry, build_fixed_tool_registry
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.runtime.startup_validation import (
    validate_competition_artifact_lineage,
    validate_startup_report,
)

_LOGGER = logging.getLogger(__name__)


class _Retriever(Protocol):
    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Return one unified retrieval response."""
        ...


class OnlineRuntime:
    """Ready online application boundary backed only by loaded artifacts."""

    def __init__(
        self,
        *,
        workflow: AgentWorkflow,
        retriever: _Retriever,
        registry: ToolRegistry,
        manifests: dict[str, ArtifactManifest],
        query_understanding: QueryUnderstandingService,
    ) -> None:
        self._workflow = workflow
        self._retriever = retriever
        self._registry = registry
        self._manifests = dict(manifests)
        self._query_understanding = query_understanding

    @property
    def workflow(self) -> AgentWorkflow:
        """Return the assembled Agent workflow."""
        return self._workflow

    def tool_descriptors(self) -> list[ToolDescriptor]:
        """Expose tool schemas without granting registry mutation access."""
        return self._registry.descriptors()

    @property
    def manifests(self) -> dict[str, ArtifactManifest]:
        """Return a copy of startup-validated artifact manifests."""
        return {
            key: manifest.model_copy(deep=True)
            for key, manifest in self._manifests.items()
        }

    def answer(self, query: RetrievalQuery) -> AgentRunResult:
        """Run one typed legal question through the assembled Agent."""
        return self._workflow.run(self._understand(query))

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """Run one fixed retrieval strategy without exposing backend clients."""
        return self._retriever.search(self._understand(query))

    def _understand(self, query: RetrievalQuery) -> RetrievalQuery:
        enriched = self._query_understanding.enrich(query)
        analysis = enriched.query_analysis
        _LOGGER.info(
            "query_understanding_completed",
            extra={
                "query_id": query.query_id,
                "query_intent": (
                    analysis.intent.value if analysis is not None else None
                ),
                "query_variant_count": len(enriched.query_variants),
                "document_reference_count": (
                    len(analysis.document_numbers)
                    if analysis is not None
                    else 0
                ),
                "structure_reference_count": (
                    len(analysis.article_numbers)
                    + len(analysis.clause_numbers)
                    + len(analysis.point_numbers)
                    if analysis is not None
                    else 0
                ),
            },
        )
        return enriched


class OnlineRuntimeFactory:
    """Fail-fast composition root for persisted reference backends and Agent."""

    def __init__(
        self,
        config: ApplicationConfig,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        context_grader: ContextGrader | None = None,
        answer_generator: AnswerGenerator | None = None,
        citation_verifier: CitationVerifier | None = None,
    ) -> None:
        self._config = config
        self._embedding_provider = (
            embedding_provider
            or SentenceTransformerEmbeddingProvider(config.offline.embedding)
        )
        self._reranker = reranker or CrossEncoderReranker(
            config.online.reranker
        )
        self._context_grader = context_grader or RuleBasedContextGrader(
            config.online.context_grading
        )
        if answer_generator is None and citation_verifier is None:
            (
                self._answer_generator,
                self._citation_verifier,
            ) = build_generation_components(
                config.online.generation,
                config.online.claim_verification,
                config.online.semantic_verification,
            )
        else:
            self._answer_generator = (
                answer_generator
                or build_answer_generator(config.online.generation)
            )
            self._citation_verifier = (
                citation_verifier
                or build_citation_verifier(
                    config.online.claim_verification,
                    config.online.semantic_verification,
                )
            )

    def build(self) -> OnlineRuntime:
        """Load, validate, and compose all online capabilities without mutation."""
        startup_started = perf_counter()
        deep_validation = (
            self._config.online.startup_validation.mode == "full"
        )
        _LOGGER.info(
            "online_artifact_manifest_validation_started",
            extra={
                "startup_validation_mode": (
                    self._config.online.startup_validation.mode
                )
            },
        )
        chunk_manifest = load_artifact_manifest(
            self._directory("legal_chunks_directory"),
            expected_type=ArtifactType.LEGAL_CHUNKS,
            verify_payload=deep_validation,
        )
        bm25_manifest = load_artifact_manifest(
            self._directory("bm25_directory"),
            expected_type=ArtifactType.BM25_INDEX,
        )
        vector_manifest = load_artifact_manifest(
            self._directory("vector_directory"),
            expected_type=ArtifactType.VECTOR_INDEX,
        )
        graph_manifest = load_artifact_manifest(
            self._directory("graph_directory"),
            expected_type=ArtifactType.GRAPH_INDEX,
        )
        self._validate_manifests(
            chunk_manifest,
            bm25_manifest,
            vector_manifest,
            graph_manifest,
        )
        if not deep_validation:
            validate_startup_report(
                self._config.artifacts.root_path
                / self._config.build_validation.report_filename,
                (
                    chunk_manifest,
                    bm25_manifest,
                    vector_manifest,
                    graph_manifest,
                ),
            )
        _LOGGER.info(
            "online_artifact_manifest_validation_completed",
            extra={
                "startup_validation_mode": (
                    self._config.online.startup_validation.mode
                )
            },
        )
        self._validate_embedding_provider(vector_manifest)

        bm25_started = perf_counter()
        _LOGGER.info("online_bm25_load_started")
        bm25 = SQLiteFTS5BM25Backend(
            self._config.offline.bm25,
            runtime_config=self._config.online.bm25_runtime,
            verify_integrity_on_load=deep_validation,
        )
        bm25.load(self._directory("bm25_directory"), bm25_manifest)
        _LOGGER.info(
            "online_bm25_load_completed",
            extra={"latency_ms": (perf_counter() - bm25_started) * 1000},
        )
        vector_started = perf_counter()
        _LOGGER.info("online_vector_load_started")
        vector = NumpyVectorBackend(
            self._config.offline.vector_index,
            runtime_config=self._config.online.vector_runtime,
            verify_integrity_on_load=deep_validation,
            serving_metadata_source=self._config.artifacts.directory(
                "vector_serving_directory"
            ),
        )
        vector.load(self._directory("vector_directory"), vector_manifest)
        _LOGGER.info(
            "online_vector_load_completed",
            extra={"latency_ms": (perf_counter() - vector_started) * 1000},
        )
        graph_started = perf_counter()
        _LOGGER.info("online_graph_load_started")
        graph = AdjacencyGraphBackend(
            self._config.offline.graph_index,
            verify_integrity_on_load=deep_validation,
        )
        graph.load(self._directory("graph_directory"), graph_manifest)
        _LOGGER.info(
            "online_graph_load_completed",
            extra={"latency_ms": (perf_counter() - graph_started) * 1000},
        )

        dense = DenseRetriever(self._embedding_provider, vector)
        query_understanding = QueryUnderstandingService(
            self._config.online.query_understanding
        )
        retriever = FixedRetriever(
            bm25,
            dense,
            self._config.online.retrieval,
            query_understanding_config=(
                self._config.online.query_understanding
            ),
            reranker=self._reranker,
            reranker_config=self._config.online.reranker,
            graph_backend=graph,
            chunk_manifest=chunk_manifest,
        )
        registry = build_fixed_tool_registry(
            retriever=retriever,
            context_grader=self._context_grader,
            answer_generator=self._answer_generator,
            citation_verifier=self._citation_verifier,
            retrieval_timeout_seconds=(
                self._config.online.retrieval.timeout_seconds
            ),
            generation_timeout_seconds=(
                self._config.online.generation.timeout_seconds
            ),
            verification_timeout_seconds=(
                self._config.online.semantic_verification.timeout_seconds
                if (
                    self._config.online.semantic_verification.backend
                    != "disabled"
                )
                else self._config.online.generation.timeout_seconds
            ),
        )
        workflow = DeterministicAgentWorkflow(
            registry,
            agent_config=self._config.online.agent,
            generation_config=self._config.online.generation,
            evidence_selection_config=(
                self._config.online.evidence_selection
            ),
            router=DeterministicStrategyRouter(
                self._config.online.agent,
                self._config.online.query_understanding,
            ),
        )
        manifests = {
            manifest.artifact_type.value: manifest
            for manifest in (
                chunk_manifest,
                bm25_manifest,
                vector_manifest,
                graph_manifest,
            )
        }
        _LOGGER.info(
            "online_runtime_initialized",
            extra={
                "dataset_name": chunk_manifest.dataset_name,
                "artifact_count": len(manifests),
                "tool_count": len(registry.descriptors()),
                "embedding_model": vector_manifest.model_name,
                "reranker_model": self._reranker.model_name,
                "latency_ms": (perf_counter() - startup_started) * 1000,
            },
        )
        return OnlineRuntime(
            workflow=workflow,
            retriever=retriever,
            registry=registry,
            manifests=manifests,
            query_understanding=query_understanding,
        )

    def _validate_embedding_provider(
        self,
        vector_manifest: ArtifactManifest,
    ) -> None:
        metadata = vector_manifest.metadata
        expected_dimension = metadata.get("dimension")
        if (
            metadata.get("embedding_provider_name")
            != self._embedding_provider.provider_name
            or metadata.get("embedding_provider_version")
            != self._embedding_provider.provider_version
            or vector_manifest.model_name != self._embedding_provider.model_name
            or vector_manifest.model_revision
            != self._embedding_provider.model_revision
            or expected_dimension != self._embedding_provider.dimension
        ):
            raise ArtifactCompatibilityError(
                "Configured embedding provider is incompatible with vector artifact"
            )

    def _validate_manifests(
        self,
        chunks: ArtifactManifest,
        bm25: ArtifactManifest,
        vector: ArtifactManifest,
        graph: ArtifactManifest,
    ) -> None:
        validate_competition_artifact_lineage(
            (chunks, bm25, vector, graph),
            self._config.competition,
        )
        expected_source = (
            chunks.artifact_type.value,
            chunks.artifact_version,
            chunks.processing_config_hash,
        )
        for manifest in (bm25, vector):
            actual_source = (
                manifest.metadata.get("source_artifact_type"),
                manifest.metadata.get("source_artifact_version"),
                manifest.metadata.get("source_processing_config_hash"),
            )
            if actual_source != expected_source:
                raise ArtifactCompatibilityError(
                    "Text index does not originate from runtime legal chunks"
                )
        normalized_hash = chunks.metadata.get(
            "runtime_normalized_processing_config_hash"
        )
        if (
            not isinstance(normalized_hash, str)
            or graph.metadata.get(
                "source_document_processing_config_hash"
            )
            != normalized_hash
        ):
            raise ArtifactCompatibilityError(
                "Graph and legal chunks do not share normalized-document lineage"
            )

    def _directory(self, field_name: str) -> Path:
        return self._config.artifacts.directory(field_name)
