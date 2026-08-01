"""Local end-to-end official corpus build using deterministic fake embeddings."""

import json
from pathlib import Path

import numpy as np
import pytest

from legal_agentic_rag.configuration import (
    ApplicationConfig,
    ArtifactConfig,
    EmbeddingConfig,
    GenerationConfig,
    OfflineConfig,
    OnlineConfig,
    VectorIndexConfig,
)
from legal_agentic_rag.runtime.competition_offline import (
    CompetitionOfflineBuildRuntime,
)
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.schemas import CompetitionBuildStage


class _EmbeddingProvider:
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


class _InterruptedGraph(AdjacencyGraphBackend):
    def persist(self, destination):  # type: ignore[no-untyped-def]
        del destination
        raise RuntimeError("graph interrupted")


def _config(root: Path) -> ApplicationConfig:
    return ApplicationConfig(
        artifacts=ArtifactConfig(root_path=root),
        offline=OfflineConfig(
            embedding=EmbeddingConfig(expected_dimension=2),
            vector_index=VectorIndexConfig(
                embedding_batch_size=2,
                checkpoint_interval_batches=1,
            ),
        ),
        online=OnlineConfig(generation=GenerationConfig(max_context_tokens=4096)),
    )


def _corpus(path: Path) -> None:
    path.mkdir()
    records = [
        ("1", "Luật thử nghiệm", "Điều 1. Phạm vi\nQuy định thử nghiệm."),
        ("2", "Nghị định thử nghiệm", "Điều 2. Trách nhiệm\nPhải thực hiện."),
    ]
    for context_id, title, passage in records:
        (path / f"context_{context_id}.json").write_text(
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


def test_official_build_persists_valid_artifacts_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "contexts"
    root = tmp_path / "artifacts"
    _corpus(source)
    config = _config(root)

    first = CompetitionOfflineBuildRuntime(
        config, source, embedding_provider=_EmbeddingProvider()
    ).build()
    resumed = CompetitionOfflineBuildRuntime(
        config, source, embedding_provider=_EmbeddingProvider()
    ).build()

    assert first.validation_report.is_valid is True
    assert first.completed_stages == list(CompetitionBuildStage)
    assert first.resumed is False
    assert resumed.resumed is True
    assert resumed.validation_report.is_valid is True
    assert (root / "relationships" / "relationships.jsonl").read_text(
        encoding="utf-8"
    ) == ""
    graph = json.loads((root / "graph" / "graph.json").read_text(encoding="utf-8"))
    assert graph["relationships"] == []
    assert graph["document_ids"] == ["1", "2"]


def test_official_build_recovers_a_partially_persisted_corpus_stage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contexts"
    root = tmp_path / "artifacts"
    _corpus(source)
    config = _config(root)
    with pytest.raises(RuntimeError, match="graph interrupted"):
        CompetitionOfflineBuildRuntime(
            config,
            source,
            embedding_provider=_EmbeddingProvider(),
            graph_backend=_InterruptedGraph(),
        ).build()

    result = CompetitionOfflineBuildRuntime(
        config,
        source,
        embedding_provider=_EmbeddingProvider(),
    ).build()

    assert result.resumed is True
    assert result.validation_report.is_valid is True
    assert result.completed_stages == list(CompetitionBuildStage)
