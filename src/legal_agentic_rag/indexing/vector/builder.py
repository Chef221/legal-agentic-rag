"""Offline orchestration from legal chunk text to a vector index artifact."""

from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
import logging
from pathlib import Path

import numpy as np

from legal_agentic_rag.configuration.offline import VectorIndexConfig
from legal_agentic_rag.contracts.embedding_provider import EmbeddingProvider
from legal_agentic_rag.contracts.vector_backend import (
    VectorBackend,
    VectorBuildBatch,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

_LOGGER = logging.getLogger(__name__)


class VectorIndexBuilder:
    """Batch document embeddings and pass aligned vectors to a backend."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        backend: VectorBackend,
        config: VectorIndexConfig | None = None,
    ) -> None:
        self._provider = provider
        self._backend = backend
        self._config = config or VectorIndexConfig()

    def build(
        self,
        chunks: Sequence[LegalChunk],
        source_manifest: ArtifactManifest,
    ) -> ArtifactManifest:
        """Embed every chunk exactly once and build the vector artifact."""
        chunk_list = list(chunks)
        if source_manifest.artifact_type != ArtifactType.LEGAL_CHUNKS:
            raise ArtifactCompatibilityError(
                "Vector builder requires a legal-chunks source artifact"
            )
        if source_manifest.record_count != len(chunk_list):
            raise DataValidationError(
                "Legal-chunks manifest count does not match vector build input"
            )
        batch_size = self._config.embedding_batch_size
        dimension = self._provider.dimension
        vectors = np.empty((len(chunk_list), dimension), dtype=np.float32)
        for start in range(0, len(chunk_list), batch_size):
            batch = chunk_list[start : start + batch_size]
            batch_vectors = np.asarray(
                list(
                    self._provider.embed_documents(
                        [chunk.search_text for chunk in batch],
                        batch_size=batch_size,
                    )
                ),
                dtype=np.float32,
            )
            if len(batch_vectors) != len(batch):
                raise DataValidationError(
                    "Embedding provider returned a mismatched batch size"
                )
            if batch_vectors.ndim != 2 or batch_vectors.shape[1] != dimension:
                raise DataValidationError(
                    "Embedding provider returned a mismatched dimension"
                )
            vectors[start : start + len(batch)] = batch_vectors
        manifest = self._backend.build(
            chunk_list,
            vectors,
            source_manifest,
            model_name=self._provider.model_name,
            model_revision=self._provider.model_revision,
            embedding_provider_name=self._provider.provider_name,
            embedding_provider_version=self._provider.provider_version,
            dimension=dimension,
            embedding_batch_size=batch_size,
        )
        _LOGGER.info(
            "vector_index_build_completed",
            extra={
                "backend": manifest.backend,
                "chunk_count": len(chunk_list),
                "model_name": self._provider.model_name,
                "dimension": dimension,
            },
        )
        return manifest

    def build_persisted(
        self,
        chunks: Iterable[LegalChunk],
        source_manifest: ArtifactManifest,
        destination: Path,
    ) -> ArtifactManifest:
        """Embed and persist a one-pass chunk stream in bounded batches."""
        if source_manifest.artifact_type != ArtifactType.LEGAL_CHUNKS:
            raise ArtifactCompatibilityError(
                "Vector builder requires a legal-chunks source artifact"
            )
        batch_size = self._config.embedding_batch_size
        dimension = self._provider.dimension
        batches = self._embedded_batches(
            chunks,
            batch_size=batch_size,
            dimension=dimension,
        )
        manifest = self._backend.build_persisted(
            batches,
            source_manifest,
            destination,
            model_name=self._provider.model_name,
            model_revision=self._provider.model_revision,
            embedding_provider_name=self._provider.provider_name,
            embedding_provider_version=self._provider.provider_version,
            dimension=dimension,
            embedding_batch_size=batch_size,
        )
        _LOGGER.info(
            "vector_index_streaming_build_completed",
            extra={
                "backend": manifest.backend,
                "chunk_count": manifest.record_count,
                "model_name": self._provider.model_name,
                "dimension": dimension,
            },
        )
        return manifest

    def _embedded_batches(
        self,
        chunks: Iterable[LegalChunk],
        *,
        batch_size: int,
        dimension: int,
    ) -> Iterator[VectorBuildBatch]:
        iterator = iter(chunks)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                return
            batch_vectors = np.asarray(
                list(
                    self._provider.embed_documents(
                        [chunk.search_text for chunk in batch],
                        batch_size=batch_size,
                    )
                ),
                dtype=np.float32,
            )
            if batch_vectors.shape != (len(batch), dimension):
                raise DataValidationError(
                    "Embedding provider returned a mismatched batch"
                )
            yield VectorBuildBatch(
                chunks=batch,
                vectors=batch_vectors,
            )
