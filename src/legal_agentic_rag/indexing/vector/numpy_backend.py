"""Exact cosine similarity vector backend using normalized NumPy matrices."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from heapq import nsmallest
import logging
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import VectorIndexConfig
from legal_agentic_rag.configuration.online import VectorRuntimeConfig
from legal_agentic_rag.contracts.vector_backend import VectorBuildBatchFactory
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.vector.artifact_store import (
    load_vector_artifact,
    persist_vector_batches,
    persist_vector_artifact,
)
from legal_agentic_rag.indexing.vector.chunk_store import JsonlChunkStore
from legal_agentic_rag.indexing.vector.serving_metadata import (
    SQLiteVectorChunkStore,
)
from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)

Clock = Callable[[], datetime]
_LOGGER = logging.getLogger(__name__)


class NumpyVectorBackend:
    """Reference exact-search backend for normalized float32 embeddings."""

    backend_name = "numpy_flat"

    def __init__(
        self,
        config: VectorIndexConfig | None = None,
        *,
        runtime_config: VectorRuntimeConfig | None = None,
        verify_integrity_on_load: bool = True,
        serving_metadata_source: Path | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or VectorIndexConfig()
        self._runtime_config = runtime_config or VectorRuntimeConfig()
        self._verify_integrity_on_load = verify_integrity_on_load
        self._serving_metadata_source = serving_metadata_source
        self._clock = clock or (lambda: datetime.now(UTC))
        self._vectors: np.ndarray | None = None
        self._chunks: Sequence[LegalChunk] = []
        self._manifest: ArtifactManifest | None = None

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return the legal-chunks identity used to build the active index."""
        metadata = self._require_manifest().metadata
        values = (
            metadata.get("source_artifact_type"),
            metadata.get("source_artifact_version"),
            metadata.get("source_processing_config_hash"),
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise ArtifactCompatibilityError("Vector source artifact identity is invalid")
        return str(values[0]), str(values[1]), str(values[2])

    @property
    def embedding_provider_name(self) -> str:
        """Return the provider identity stored in the active artifact."""
        value = self._require_manifest().metadata.get("embedding_provider_name")
        if not isinstance(value, str) or not value:
            raise ArtifactCompatibilityError("Embedding provider metadata is invalid")
        return value

    @property
    def embedding_provider_version(self) -> str:
        """Return the provider version stored in the active artifact."""
        value = self._require_manifest().metadata.get("embedding_provider_version")
        if not isinstance(value, str) or not value:
            raise ArtifactCompatibilityError("Embedding provider metadata is invalid")
        return value

    @property
    def model_name(self) -> str:
        """Return the model stored in the active vector artifact."""
        return self._require_manifest().model_name or ""

    @property
    def model_revision(self) -> str | None:
        """Return the model revision stored in the active vector artifact."""
        return self._require_manifest().model_revision

    @property
    def dimension(self) -> int:
        """Return the vector dimension stored in the active vector artifact."""
        dimension = self._require_manifest().metadata.get("dimension")
        if not isinstance(dimension, int):
            raise ArtifactCompatibilityError("Vector dimension metadata is invalid")
        return dimension

    def build(
        self,
        chunks: Sequence[LegalChunk],
        vectors: Sequence[Sequence[float]],
        source_manifest: ArtifactManifest,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_provider_name: str,
        embedding_provider_version: str,
        dimension: int,
        embedding_batch_size: int,
    ) -> ArtifactManifest:
        """Build an exact normalized cosine index from aligned inputs."""
        chunk_list = list(chunks)
        self._validate_build_input(
            chunk_list,
            source_manifest,
            model_name=model_name,
            model_revision=model_revision,
            embedding_provider_name=embedding_provider_name,
            embedding_provider_version=embedding_provider_version,
            dimension=dimension,
            embedding_batch_size=embedding_batch_size,
        )
        matrix = self._vector_matrix(vectors, len(chunk_list), dimension)
        order = sorted(range(len(chunk_list)), key=lambda index: chunk_list[index].chunk_id)
        sorted_chunks = [chunk_list[index] for index in order]
        sorted_vectors = matrix[order] if order else matrix
        self._chunks = sorted_chunks
        self._vectors = np.ascontiguousarray(sorted_vectors, dtype=np.float32)
        self._manifest = self._build_manifest(
            sorted_chunks,
            source_manifest,
            model_name=model_name,
            model_revision=model_revision,
            embedding_provider_name=embedding_provider_name,
            embedding_provider_version=embedding_provider_version,
            dimension=dimension,
            embedding_batch_size=embedding_batch_size,
        )
        _LOGGER.info(
            "vector_index_built",
            extra={
                "backend": self.backend_name,
                "chunk_count": len(sorted_chunks),
                "model_name": model_name,
                "dimension": dimension,
            },
        )
        return self._manifest

    def search(
        self,
        query: RetrievalQuery,
        query_vector: Sequence[float],
    ) -> RetrievalResponse:
        """Return exact cosine-ranked dense hits with unified metadata."""
        vectors, manifest = self._require_ready()
        if query.requested_strategy not in (None, RetrievalStrategy.DENSE):
            raise RetrievalError("Vector backend received a non-dense request")
        started = perf_counter()
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise RetrievalError("Dense query vector has an incompatible shape")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise RetrievalError("Dense query vector must not be zero")
        vector = vector / norm
        candidate_indexes = self._filtered_indexes(query)
        candidate_count = (
            len(self._chunks)
            if candidate_indexes is None
            else int(candidate_indexes.size)
        )
        hits: list[RetrievalHit] = []
        warnings: list[str] = []
        if candidate_count:
            scores = self._score_candidates(
                vectors,
                vector,
                candidate_indexes,
            )
            ranked_offsets = self._ranked_offsets(
                scores,
                candidate_indexes,
                query.top_k,
            )
            ranked_indexes = [
                self._row_index(candidate_indexes, offset)
                for offset in ranked_offsets
            ]
            ranked_chunks = self._chunks_at(ranked_indexes)
            hits = [
                self._retrieval_hit(
                    chunk,
                    rank,
                    float(scores[offset]),
                )
                for rank, (offset, chunk) in enumerate(
                    zip(ranked_offsets, ranked_chunks, strict=True),
                    start=1,
                )
            ]
        else:
            warnings.append("no_dense_matches")
        latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "dense_vector_search_completed",
            extra={
                "query_id": query.query_id,
                "strategy": RetrievalStrategy.DENSE.value,
                "candidate_count": candidate_count,
                "hit_count": len(hits),
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.DENSE,
            hits=hits,
            latency_ms=latency_ms,
            warnings=warnings,
            artifact_versions={"vector_index": manifest.artifact_version},
        )

    def persist(self, destination: Path) -> ArtifactManifest:
        """Persist normalized vectors, chunks, checksums, and manifest."""
        vectors, manifest = self._require_ready()
        final_manifest = persist_vector_artifact(
            vectors=vectors,
            chunks=self._chunks,
            destination=destination,
            manifest=manifest,
        )
        self._manifest = final_manifest
        return final_manifest

    def build_persisted(
        self,
        batch_factory: VectorBuildBatchFactory,
        source_manifest: ArtifactManifest,
        destination: Path,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_provider_name: str,
        embedding_provider_version: str,
        dimension: int,
        embedding_batch_size: int,
    ) -> ArtifactManifest:
        """Persist a bounded batch stream without retaining corpus chunks."""
        self._validate_source_identity(
            source_manifest,
            model_name=model_name,
            embedding_provider_name=embedding_provider_name,
            embedding_provider_version=embedding_provider_version,
            dimension=dimension,
            embedding_batch_size=embedding_batch_size,
        )
        if not model_revision:
            raise DataValidationError(
                "Vector build requires pinned model identity"
            )
        manifest = self._build_manifest(
            source_manifest.record_count,
            source_manifest,
            model_name=model_name,
            model_revision=model_revision,
            embedding_provider_name=embedding_provider_name,
            embedding_provider_version=embedding_provider_version,
            dimension=dimension,
            embedding_batch_size=embedding_batch_size,
        )
        stored = persist_vector_batches(
            batch_factory=batch_factory,
            destination=destination,
            manifest=manifest,
            dimension=dimension,
            checkpoint_interval_batches=(
                self._config.checkpoint_interval_batches
            ),
        )
        _LOGGER.info(
            "vector_index_persisted_from_batches",
            extra={
                "backend": self.backend_name,
                "chunk_count": stored.record_count,
                "model_name": model_name,
                "dimension": dimension,
            },
        )
        return stored

    def load(self, source: Path, manifest: ArtifactManifest) -> None:
        """Load a compatible memory-mapped exact cosine vector artifact."""
        vectors, chunks, stored_manifest = load_vector_artifact(
            source=source,
            supplied_manifest=manifest,
            expected_backend=self.backend_name,
            expected_artifact_version=self._config.artifact_version,
            expected_distance_metric=self._config.distance_metric,
            expected_dtype=self._config.dtype,
            validation_batch_size=self._runtime_config.validation_batch_size,
            load_progress_interval_records=(
                self._runtime_config.load_progress_interval_records
            ),
            checksum_progress_interval_bytes=(
                self._runtime_config.checksum_progress_interval_bytes
            ),
            verify_integrity=self._verify_integrity_on_load,
            serving_metadata_source=(
                self._serving_metadata_source
                if self._runtime_config.prefer_serving_metadata
                else None
            ),
            require_serving_metadata=(
                self._runtime_config.require_serving_metadata
            ),
        )
        self._vectors = vectors
        self._chunks = chunks
        self._manifest = stored_manifest
        _LOGGER.info(
            "vector_index_loaded",
            extra={
                "backend": self.backend_name,
                "chunk_count": len(chunks),
                "model_name": stored_manifest.model_name,
                "dimension": self.dimension,
            },
        )

    def _filtered_indexes(
        self,
        query: RetrievalQuery,
    ) -> np.ndarray | None:
        filters = query.filters
        if isinstance(
            self._chunks,
            (JsonlChunkStore, SQLiteVectorChunkStore),
        ):
            return self._chunks.filtered_indexes(filters)
        if not any(
            (
                filters.document_ids,
                filters.document_types,
                filters.legal_fields,
                filters.effect_statuses,
            )
        ):
            return None
        indexes = [
            index
            for index, chunk in enumerate(self._chunks)
            if (not filters.document_ids or chunk.document_id in filters.document_ids)
            and (
                not filters.document_types
                or chunk.document_type in filters.document_types
            )
            and (not filters.legal_fields or chunk.legal_field in filters.legal_fields)
            and (
                not filters.effect_statuses
                or chunk.effect_status in filters.effect_statuses
            )
        ]
        return np.asarray(indexes, dtype=np.int64)

    def _score_candidates(
        self,
        vectors: np.ndarray,
        query_vector: np.ndarray,
        candidate_indexes: np.ndarray | None,
    ) -> np.ndarray:
        """Score exact cosine candidates without a corpus-sized matrix copy."""
        candidate_count = (
            len(self._chunks)
            if candidate_indexes is None
            else int(candidate_indexes.size)
        )
        scores = np.empty(candidate_count, dtype=np.float32)
        batch_size = self._runtime_config.search_batch_size
        for start in range(0, candidate_count, batch_size):
            end = min(start + batch_size, candidate_count)
            if candidate_indexes is None:
                matrix = vectors[start:end]
            else:
                matrix = vectors[candidate_indexes[start:end]]
            scores[start:end] = np.asarray(
                matrix @ query_vector,
                dtype=np.float32,
            )
        return scores

    def _ranked_offsets(
        self,
        scores: np.ndarray,
        candidate_indexes: np.ndarray | None,
        top_k: int,
    ) -> list[int]:
        """Select exact top-k and resolve score ties by stable chunk ID."""
        limit = min(top_k, int(scores.size))
        if limit == scores.size:
            selected = list(range(int(scores.size)))
        else:
            threshold_index = int(scores.size) - limit
            threshold = float(
                np.partition(scores, threshold_index)[threshold_index]
            )
            selected = [
                int(offset)
                for offset in np.flatnonzero(scores > threshold)
            ]
            remaining = limit - len(selected)
            tied_offsets = (
                int(offset)
                for offset in np.flatnonzero(scores == threshold)
            )
            selected.extend(
                nsmallest(
                    remaining,
                    tied_offsets,
                    key=lambda offset: self._chunk_id(
                        self._row_index(candidate_indexes, offset)
                    ),
                )
            )
        selected.sort(
            key=lambda offset: (
                -float(scores[offset]),
                self._chunk_id(self._row_index(candidate_indexes, offset)),
            )
        )
        return selected

    @staticmethod
    def _row_index(
        candidate_indexes: np.ndarray | None,
        offset: int,
    ) -> int:
        if candidate_indexes is None:
            return int(offset)
        return int(candidate_indexes[offset])

    def _chunk_id(self, index: int) -> str:
        if isinstance(
            self._chunks,
            (JsonlChunkStore, SQLiteVectorChunkStore),
        ):
            return self._chunks.chunk_id(index)
        return self._chunks[index].chunk_id

    def _chunks_at(self, indexes: Sequence[int]) -> list[LegalChunk]:
        if isinstance(
            self._chunks,
            (JsonlChunkStore, SQLiteVectorChunkStore),
        ):
            return self._chunks.get_many(indexes)
        return [self._chunks[index] for index in indexes]

    @staticmethod
    def _retrieval_hit(
        chunk: LegalChunk,
        rank: int,
        score: float,
    ) -> RetrievalHit:
        payload = chunk.model_dump(mode="json")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"chunk_id", "document_id", "text"}
        }
        return RetrievalHit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            rank=rank,
            score=score,
            strategy=RetrievalStrategy.DENSE,
            text=chunk.text,
            metadata=metadata,
            retrieval_trace=RetrievalTrace(
                dense_rank=rank,
                dense_score=score,
            ),
        )

    def _build_manifest(
        self,
        chunks: list[LegalChunk] | int,
        source_manifest: ArtifactManifest,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_provider_name: str,
        embedding_provider_version: str,
        dimension: int,
        embedding_batch_size: int,
    ) -> ArtifactManifest:
        chunk_count = chunks if isinstance(chunks, int) else len(chunks)
        hash_payload = {
            "config": self._config,
            "source_artifact_version": source_manifest.artifact_version,
            "source_processing_config_hash": source_manifest.processing_config_hash,
            "model_name": model_name,
            "model_revision": model_revision,
            "embedding_provider_name": embedding_provider_name,
            "embedding_provider_version": embedding_provider_version,
            "dimension": dimension,
            "source_payload_sha256": source_manifest.metadata.get(
                "payload_sha256"
            ),
        }
        config_hash = canonical_sha256(hash_payload)
        return ArtifactManifest(
            schema_version=source_manifest.schema_version,
            artifact_type=ArtifactType.VECTOR_INDEX,
            artifact_version=self._config.artifact_version,
            dataset_name=source_manifest.dataset_name,
            dataset_revision=source_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=chunk_count,
            processing_config_hash=config_hash,
            code_version=__version__,
            backend=self.backend_name,
            model_name=model_name,
            model_revision=model_revision,
            metadata={
                "dimension": dimension,
                "embedding_provider_name": embedding_provider_name,
                "embedding_provider_version": embedding_provider_version,
                "distance_metric": self._config.distance_metric,
                "dtype": self._config.dtype,
                "normalized_vectors": True,
                "embedding_batch_size": embedding_batch_size,
                "source_artifact_type": source_manifest.artifact_type.value,
                "source_artifact_version": source_manifest.artifact_version,
                "source_processing_config_hash": source_manifest.processing_config_hash,
                "numpy_version": np.__version__,
            },
        )

    @staticmethod
    def _vector_matrix(
        vectors: Sequence[Sequence[float]],
        expected_count: int,
        dimension: int,
    ) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if expected_count == 0 and matrix.size == 0:
            matrix = np.empty((0, dimension), dtype=np.float32)
        if matrix.shape != (expected_count, dimension):
            raise DataValidationError("Vector matrix shape does not match chunks")
        if not np.isfinite(matrix).all():
            raise DataValidationError("Vector matrix contains non-finite values")
        if expected_count:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise DataValidationError("Vector matrix contains a zero vector")
            matrix /= norms
        return matrix

    @staticmethod
    def _validate_build_input(
        chunks: list[LegalChunk],
        source_manifest: ArtifactManifest,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_provider_name: str,
        embedding_provider_version: str,
        dimension: int,
        embedding_batch_size: int,
    ) -> None:
        NumpyVectorBackend._validate_source_identity(
            source_manifest,
            model_name=model_name,
            embedding_provider_name=embedding_provider_name,
            embedding_provider_version=embedding_provider_version,
            dimension=dimension,
            embedding_batch_size=embedding_batch_size,
        )
        if source_manifest.record_count != len(chunks):
            raise DataValidationError(
                "Legal-chunks manifest count does not match vector build input"
            )
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise DataValidationError("Vector build requires unique chunk IDs")
        if not model_revision:
            raise DataValidationError("Vector build requires pinned model identity")

    @staticmethod
    def _validate_source_identity(
        source_manifest: ArtifactManifest,
        *,
        model_name: str,
        embedding_provider_name: str,
        embedding_provider_version: str,
        dimension: int,
        embedding_batch_size: int,
    ) -> None:
        if source_manifest.artifact_type != ArtifactType.LEGAL_CHUNKS:
            raise ArtifactCompatibilityError(
                "Vector build requires a legal-chunks source artifact"
            )
        if not model_name.strip():
            raise DataValidationError("Vector build requires model identity")
        if not embedding_provider_name.strip() or not embedding_provider_version.strip():
            raise DataValidationError("Vector build requires embedding provider identity")
        if dimension <= 0 or embedding_batch_size <= 0:
            raise DataValidationError("Vector build dimensions and batch must be positive")

    def _require_manifest(self) -> ArtifactManifest:
        if self._manifest is None:
            raise BackendInitializationError("Vector index has not been built or loaded")
        return self._manifest

    def _require_ready(self) -> tuple[np.ndarray, ArtifactManifest]:
        manifest = self._require_manifest()
        if self._vectors is None:
            raise BackendInitializationError("Vector index has not been built or loaded")
        return self._vectors, manifest
