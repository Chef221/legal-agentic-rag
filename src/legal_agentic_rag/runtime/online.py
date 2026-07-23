"""Load compatible artifacts and compose the complete online Agent runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from legal_agentic_rag.agent import DeterministicAgentWorkflow
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
    ExtractiveAnswerGenerator,
    RuleBasedCitationVerifier,
    RuleBasedContextGrader,
)
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.indexing.vector import NumpyVectorBackend
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import DenseRetriever, FixedRetriever
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
    ) -> None:
        self._workflow = workflow
        self._retriever = retriever
        self._registry = registry
        self._manifests = dict(manifests)

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
        return self._workflow.run(query)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """Run one fixed retrieval strategy without exposing backend clients."""
        return self._retriever.search(query)


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
        self._answer_generator = (
            answer_generator or ExtractiveAnswerGenerator()
        )
        self._citation_verifier = (
            citation_verifier or RuleBasedCitationVerifier()
        )

    def build(self) -> OnlineRuntime:
        """Load, validate, and compose all online capabilities without mutation."""
        chunk_manifest = load_artifact_manifest(
            self._directory("legal_chunks_directory"),
            expected_type=ArtifactType.LEGAL_CHUNKS,
            verify_payload=True,
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
        self._validate_embedding_provider(vector_manifest)

        bm25 = SQLiteFTS5BM25Backend(self._config.offline.bm25)
        bm25.load(self._directory("bm25_directory"), bm25_manifest)
        vector = NumpyVectorBackend(self._config.offline.vector_index)
        vector.load(self._directory("vector_directory"), vector_manifest)
        graph = AdjacencyGraphBackend(self._config.offline.graph_index)
        graph.load(self._directory("graph_directory"), graph_manifest)

        dense = DenseRetriever(self._embedding_provider, vector)
        retriever = FixedRetriever(
            bm25,
            dense,
            self._config.online.retrieval,
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
        )
        workflow = DeterministicAgentWorkflow(
            registry,
            agent_config=self._config.online.agent,
            generation_config=self._config.online.generation,
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
            },
        )
        return OnlineRuntime(
            workflow=workflow,
            retriever=retriever,
            registry=registry,
            manifests=manifests,
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

    @staticmethod
    def _validate_manifests(
        chunks: ArtifactManifest,
        bm25: ArtifactManifest,
        vector: ArtifactManifest,
        graph: ArtifactManifest,
    ) -> None:
        dataset_identities = {
            (manifest.dataset_name, manifest.dataset_revision)
            for manifest in (chunks, bm25, vector, graph)
        }
        if len(dataset_identities) != 1:
            raise ArtifactCompatibilityError(
                "Runtime artifacts originate from different datasets"
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
