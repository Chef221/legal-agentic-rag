"""Tests for batched offline vector index orchestration."""

from collections.abc import Sequence
import json
from pathlib import Path

import pytest

from legal_agentic_rag.configuration import VectorIndexConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.indexing.vector import NumpyVectorBackend, VectorIndexBuilder
from legal_agentic_rag.schemas import ArtifactManifest, LegalChunk


class _FixtureProvider:
    provider_name = "fixture-provider"
    provider_version = "1.0"
    model_name = "fixture/e5"
    model_revision = "fixture-revision"
    dimension = 2

    def __init__(self) -> None:
        self.batches: list[tuple[list[str], int]] = []

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        values = list(texts)
        self.batches.append((values, batch_size))
        return [[float(len(text)), 1.0] for text in values]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_builder_batches_chunks_and_records_actual_batch_size(
    vector_chunks: list[LegalChunk],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Offline builder embeds search_text exactly once in configured batches."""
    provider = _FixtureProvider()
    backend = NumpyVectorBackend()
    config = VectorIndexConfig(embedding_batch_size=2)

    manifest = VectorIndexBuilder(provider, backend, config).build(
        vector_chunks, vector_source_manifest
    )

    assert [len(batch) for batch, _ in provider.batches] == [2, 1]
    assert all(size == 2 for _, size in provider.batches)
    assert [text for batch, _ in provider.batches for text in batch] == [
        chunk.search_text for chunk in vector_chunks
    ]
    assert manifest.model_name == "fixture/e5"
    assert manifest.metadata["embedding_provider_name"] == "fixture-provider"
    assert manifest.metadata["embedding_provider_version"] == "1.0"
    assert manifest.metadata["embedding_batch_size"] == 2


def test_builder_rejects_provider_count_and_dimension_mismatch(
    vector_chunks: list[LegalChunk],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """A provider cannot silently drop embeddings or change dimension."""
    class MissingProvider(_FixtureProvider):
        def embed_documents(
            self, texts: Sequence[str], *, batch_size: int
        ) -> list[list[float]]:
            return []

    with pytest.raises(DataValidationError, match="batch size"):
        VectorIndexBuilder(MissingProvider(), NumpyVectorBackend()).build(
            vector_chunks, vector_source_manifest
        )

    class WrongDimensionProvider(_FixtureProvider):
        def embed_documents(
            self, texts: Sequence[str], *, batch_size: int
        ) -> list[list[float]]:
            return [[1.0] for _ in texts]

    with pytest.raises(DataValidationError, match="dimension"):
        VectorIndexBuilder(WrongDimensionProvider(), NumpyVectorBackend()).build(
            vector_chunks, vector_source_manifest
        )


def test_streaming_builder_persists_one_pass_batches(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """Production vector build writes bounded batches directly to disk."""
    provider = _FixtureProvider()
    destination = tmp_path / "vector"
    backend = NumpyVectorBackend()

    stored = VectorIndexBuilder(
        provider,
        backend,
        VectorIndexConfig(embedding_batch_size=2),
    ).build_persisted(
        iter(vector_chunks),
        vector_source_manifest,
        destination,
    )
    loaded = NumpyVectorBackend()
    loaded.load(destination, stored)

    assert stored.record_count == len(vector_chunks)
    assert stored.metadata["chunk_order"] == "source_artifact_order"
    assert [len(batch) for batch, _ in provider.batches] == [2, 1]
    assert loaded.dimension == 2


def test_streaming_builder_does_not_publish_failed_artifact(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """A provider mismatch leaves no loadable vector destination."""
    class MissingProvider(_FixtureProvider):
        def embed_documents(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> list[list[float]]:
            _ = (texts, batch_size)
            return []

    destination = tmp_path / "failed-vector"
    with pytest.raises(DataValidationError, match="mismatched batch"):
        VectorIndexBuilder(
            MissingProvider(),
            NumpyVectorBackend(),
        ).build_persisted(
            iter(vector_chunks),
            vector_source_manifest,
            destination,
        )

    assert not destination.exists()
    assert (tmp_path / ".failed-vector.partial" / "checkpoint.json").is_file()


def test_streaming_builder_resumes_after_committed_batch(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """A later run skips committed chunks instead of embedding them again."""

    class InterruptingProvider(_FixtureProvider):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0

        def embed_documents(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> list[list[float]]:
            self.call_count += 1
            if self.call_count == 2:
                raise RuntimeError("simulated runtime interruption")
            return super().embed_documents(texts, batch_size=batch_size)

    destination = tmp_path / "vector"
    config = VectorIndexConfig(
        embedding_batch_size=1,
        checkpoint_interval_batches=1,
    )
    interrupted = InterruptingProvider()
    with pytest.raises(RuntimeError, match="interruption"):
        VectorIndexBuilder(
            interrupted,
            NumpyVectorBackend(config),
            config,
        ).build_persisted(
            iter(vector_chunks),
            vector_source_manifest,
            destination,
        )

    checkpoint_path = tmp_path / ".vector.partial" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["next_offset"] == 1
    assert not destination.exists()
    with (tmp_path / ".vector.partial" / "chunks.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted":')

    resumed_provider = _FixtureProvider()
    stored = VectorIndexBuilder(
        resumed_provider,
        NumpyVectorBackend(config),
        config,
    ).build_persisted(
        iter(vector_chunks),
        vector_source_manifest,
        destination,
    )

    assert [
        text
        for batch, _ in resumed_provider.batches
        for text in batch
    ] == [chunk.search_text for chunk in vector_chunks[1:]]
    assert stored.record_count == len(vector_chunks)
    assert destination.is_dir()
    assert not (destination / "checkpoint.json").exists()
    assert not (tmp_path / ".vector.partial").exists()


def test_streaming_builder_rejects_incompatible_checkpoint(
    tmp_path: Path,
    vector_chunks: list[LegalChunk],
    vector_source_manifest: ArtifactManifest,
) -> None:
    """A partial vector build cannot be resumed with a different model identity."""

    class InterruptingProvider(_FixtureProvider):
        def embed_documents(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> list[list[float]]:
            raise RuntimeError("simulated runtime interruption")

    config = VectorIndexConfig(
        embedding_batch_size=1,
        checkpoint_interval_batches=1,
    )
    destination = tmp_path / "vector"
    with pytest.raises(RuntimeError, match="interruption"):
        VectorIndexBuilder(
            InterruptingProvider(),
            NumpyVectorBackend(config),
            config,
        ).build_persisted(
            iter(vector_chunks),
            vector_source_manifest,
            destination,
        )

    class DifferentModelProvider(_FixtureProvider):
        model_name = "fixture/different-model"

    provider = DifferentModelProvider()
    with pytest.raises(ArtifactCompatibilityError, match="identity"):
        VectorIndexBuilder(
            provider,
            NumpyVectorBackend(config),
            config,
        ).build_persisted(
            iter(vector_chunks),
            vector_source_manifest,
            destination,
        )
    assert provider.batches == []
