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
from legal_agentic_rag.configuration.hashing import canonical_sha256
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


class _UnexpectedEmbeddingProvider:
    def embed_documents(self, texts, *, batch_size):  # type: ignore[no-untyped-def]
        del texts, batch_size
        raise AssertionError("document-processing build must not embed chunks")

    def embed_query(self, text):  # type: ignore[no-untyped-def]
        del text
        raise AssertionError("document-processing build must not embed queries")


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
    assert (root / "normalized_documents" / "manifest.json").is_file()
    assert (root / "cleaned_documents" / "manifest.json").is_file()
    assert (root / "audit" / "corpus_audit.json").is_file()


def test_official_build_can_stop_after_parser_and_chunker_then_resume(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contexts"
    root = tmp_path / "artifacts"
    _corpus(source)
    config = _config(root)

    partial = CompetitionOfflineBuildRuntime(
        config,
        source,
        embedding_provider=_UnexpectedEmbeddingProvider(),
    ).build(through=CompetitionBuildStage.DOCUMENT_PROCESSING)

    assert partial.completed_stages == [
        CompetitionBuildStage.CORPUS,
        CompetitionBuildStage.DOCUMENT_PROCESSING,
    ]
    assert partial.validation_report is None
    assert (root / "legal_blocks" / "manifest.json").is_file()
    assert (root / "legal_chunks" / "manifest.json").is_file()
    assert not (root / "bm25").exists()
    assert not (root / "vector").exists()
    assert not (root / "build_validation.json").exists()

    completed = CompetitionOfflineBuildRuntime(
        config,
        source,
        embedding_provider=_EmbeddingProvider(),
    ).build()

    assert completed.resumed is True
    assert completed.completed_stages == list(CompetitionBuildStage)
    assert completed.validation_report is not None
    assert completed.validation_report.is_valid is True


def test_official_build_uses_canonical_typed_application_config_hash(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "artifacts")
    runtime = CompetitionOfflineBuildRuntime(config, tmp_path / "contexts")

    assert runtime._config_hash() == canonical_sha256(config)


def test_official_build_recovers_a_partially_persisted_corpus_stage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contexts"
    root = tmp_path / "artifacts"
    _corpus(source)
    config = _config(root)

    class _InterruptedRuntime(CompetitionOfflineBuildRuntime):
        def _persist_or_validate_audit(self, audit):  # type: ignore[no-untyped-def]
            del audit
            raise RuntimeError("audit interrupted")

    with pytest.raises(RuntimeError, match="audit interrupted"):
        _InterruptedRuntime(
            config,
            source,
            embedding_provider=_EmbeddingProvider(),
        ).build()

    result = CompetitionOfflineBuildRuntime(
        config,
        source,
        embedding_provider=_EmbeddingProvider(),
    ).build()

    assert result.resumed is True
    assert result.validation_report.is_valid is True
    assert result.completed_stages == list(CompetitionBuildStage)
