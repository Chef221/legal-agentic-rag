"""Tests for batched offline vector index orchestration."""

from collections.abc import Sequence

import pytest

from legal_agentic_rag.configuration import VectorIndexConfig
from legal_agentic_rag.exceptions import DataValidationError
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
