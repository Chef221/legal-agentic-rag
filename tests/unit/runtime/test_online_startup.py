"""Tests for online runtime startup with optional graph artifacts."""

from collections.abc import Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from legal_agentic_rag.configuration import (
    ApplicationConfig,
    ArtifactConfig,
    EmbeddingConfig,
    OfflineConfig,
    OnlineConfig,
    RetrievalConfig,
    StartupValidationConfig,
    VectorIndexConfig,
)
from legal_agentic_rag.contracts.answer_generator import AnswerGenerator
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.contracts.context_grader import ContextGrader
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.runtime import OnlineRuntimeFactory
from legal_agentic_rag.runtime.competition_offline import (
    CompetitionOfflineBuildRuntime,
)
from legal_agentic_rag.schemas import (
    AnswerResponse,
    ArtifactType,
    CitationVerificationResult,
    ContextGrade,
    Evidence,
    RetrievalHit,
    RetrievalQuery,
    RetrievalStrategy,
    ToolName,
)


class _FixtureEmbeddingProvider:
    provider_name = "fixture-embedding"
    provider_version = "1.0"
    model_name = "fixture/model"
    model_revision = "fixture-revision"
    dimension = 2

    def embed_documents(self, texts, *, batch_size):  # type: ignore[no-untyped-def]
        del batch_size
        return [np.array([1.0, float(index + 1)]) for index, _ in enumerate(texts)]

    def embed_query(self, text):  # type: ignore[no-untyped-def]
        del text
        return np.array([1.0, 1.0])


class _FixtureReranker(Reranker):
    provider_name = "fixture-reranker"
    provider_version = "1.0"
    model_name = "fixture/reranker"
    model_revision = "fixture-revision"

    def rerank(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> tuple[list[RetrievalHit], list[str]]:
        return hits[:query.top_k], []


class _FixtureContextGrader:
    def grade(
        self, query: RetrievalQuery, evidence: Sequence[Evidence]
    ) -> ContextGrade:
        return ContextGrade(
            is_sufficient=True,
            coverage_score=1.0,
            relevance_score=1.0,
            factual_consistency_score=1.0,
        )


class _FixtureAnswerGenerator:
    def generate(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
    ) -> AnswerResponse:
        return AnswerResponse(
            answer="Câu trả lời.",
            insufficient_evidence=False,
            trace_id=trace_id,
        )


class _FixtureCitationVerifier:
    def verify(
        self, response: AnswerResponse, evidence: Sequence[Evidence]
    ) -> CitationVerificationResult:
        return CitationVerificationResult(
            is_valid=True,
            valid_citations=[],
            invalid_citations=[],
        )


def _build_test_corpus_artifacts(root: Path) -> ApplicationConfig:
    source = root / "contexts"
    artifacts_dir = root / "artifacts"
    source.mkdir(parents=True, exist_ok=True)
    records = [
        ("1", "Luật thử nghiệm", "Điều 1. Phạm vi\nQuy định thử nghiệm."),
        ("2", "Nghị định thử nghiệm", "Điều 2. Trách nhiệm\nPhải thực hiện."),
    ]
    for context_id, title, passage in records:
        (source / f"context_{context_id}.json").write_text(
            json.dumps(
                {
                    "id": context_id,
                    "name": title,
                    "link": f"https://example.test/{context_id}",
                    "passage": passage,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    config = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=artifacts_dir),
        offline=OfflineConfig(
            embedding=EmbeddingConfig(expected_dimension=2),
            vector_index=VectorIndexConfig(),
        ),
        online=OnlineConfig(
            retrieval=RetrievalConfig(
                top_k=2,
                candidate_k=5,
                graph_runtime_enabled=True,
            ),
            startup_validation=StartupValidationConfig(mode="validated_report"),
        ),
    )

    CompetitionOfflineBuildRuntime(
        config,
        source,
        embedding_provider=_FixtureEmbeddingProvider(),
    ).build()

    return config


def _factory(config: ApplicationConfig) -> OnlineRuntimeFactory:
    return OnlineRuntimeFactory(
        config,
        embedding_provider=_FixtureEmbeddingProvider(),
        reranker=_FixtureReranker(),
        context_grader=_FixtureContextGrader(),
        answer_generator=_FixtureAnswerGenerator(),
        citation_verifier=_FixtureCitationVerifier(),
    )


def test_startup_succeeds_when_graph_disabled_and_graph_directory_physically_absent(
    tmp_path: Path,
) -> None:
    """When graph_runtime_enabled is False, startup succeeds without a graph directory."""
    base_config = _build_test_corpus_artifacts(tmp_path)
    graph_dir = base_config.artifacts.root_path / "graph"
    assert graph_dir.is_dir()
    shutil.rmtree(graph_dir)
    assert not graph_dir.exists()

    config = base_config.model_copy(
        update={
            "online": base_config.online.model_copy(
                update={
                    "retrieval": base_config.online.retrieval.model_copy(
                        update={"graph_runtime_enabled": False}
                    )
                }
            )
        }
    )

    runtime = _factory(config).build()

    assert ArtifactType.GRAPH_INDEX.value not in runtime.manifests
    assert ArtifactType.LEGAL_CHUNKS.value in runtime.manifests
    assert ArtifactType.BM25_INDEX.value in runtime.manifests
    assert ArtifactType.VECTOR_INDEX.value in runtime.manifests
    assert len(runtime.manifests) == 3
    assert runtime._retriever._graph is None

    tool_names = [d.name for d in runtime.tool_descriptors()]
    assert ToolName.GRAPH_SEARCH not in tool_names
    assert len(tool_names) == 7


def test_startup_ignores_existing_graph_directory_when_graph_disabled(
    tmp_path: Path,
) -> None:
    """When graph_runtime_enabled is False, existing graph directory is not loaded."""
    base_config = _build_test_corpus_artifacts(tmp_path)
    graph_dir = base_config.artifacts.root_path / "graph"
    assert graph_dir.is_dir()

    config = base_config.model_copy(
        update={
            "online": base_config.online.model_copy(
                update={
                    "retrieval": base_config.online.retrieval.model_copy(
                        update={"graph_runtime_enabled": False}
                    )
                }
            )
        }
    )

    with patch.object(AdjacencyGraphBackend, "load", wraps=AdjacencyGraphBackend.load) as mock_load:
        runtime = _factory(config).build()
        mock_load.assert_not_called()

    assert ArtifactType.GRAPH_INDEX.value not in runtime.manifests
    assert len(runtime.manifests) == 3
    assert runtime._retriever._graph is None


def test_startup_fails_when_graph_enabled_and_graph_directory_absent(
    tmp_path: Path,
) -> None:
    """When graph_runtime_enabled is True, missing graph directory fails startup."""
    base_config = _build_test_corpus_artifacts(tmp_path)
    graph_dir = base_config.artifacts.root_path / "graph"
    shutil.rmtree(graph_dir)

    config = base_config.model_copy(
        update={
            "online": base_config.online.model_copy(
                update={
                    "retrieval": base_config.online.retrieval.model_copy(
                        update={"graph_runtime_enabled": True}
                    )
                }
            )
        }
    )

    with pytest.raises(ArtifactCompatibilityError, match="missing or invalid"):
        _factory(config).build()


def test_startup_succeeds_and_includes_graph_when_graph_enabled(
    tmp_path: Path,
) -> None:
    """When graph_runtime_enabled is True and graph exists, runtime loads all 4 manifests."""
    base_config = _build_test_corpus_artifacts(tmp_path)

    config = base_config.model_copy(
        update={
            "online": base_config.online.model_copy(
                update={
                    "retrieval": base_config.online.retrieval.model_copy(
                        update={"graph_runtime_enabled": True}
                    )
                }
            )
        }
    )

    runtime = _factory(config).build()

    assert ArtifactType.GRAPH_INDEX.value in runtime.manifests
    assert len(runtime.manifests) == 4
    assert runtime._retriever._graph is not None
    tool_names = [d.name for d in runtime.tool_descriptors()]
    assert ToolName.GRAPH_SEARCH in tool_names
    assert len(tool_names) == 8


def test_startup_in_deep_validation_mode_succeeds_without_graph(
    tmp_path: Path,
) -> None:
    """Deep validation mode (mode='full') succeeds without graph when graph_runtime_enabled is False."""
    base_config = _build_test_corpus_artifacts(tmp_path)
    graph_dir = base_config.artifacts.root_path / "graph"
    shutil.rmtree(graph_dir)

    config = base_config.model_copy(
        update={
            "online": base_config.online.model_copy(
                update={
                    "retrieval": base_config.online.retrieval.model_copy(
                        update={"graph_runtime_enabled": False}
                    ),
                    "startup_validation": StartupValidationConfig(mode="full"),
                }
            )
        }
    )

    runtime = _factory(config).build()
    assert ArtifactType.GRAPH_INDEX.value not in runtime.manifests
    assert len(runtime.manifests) == 3
