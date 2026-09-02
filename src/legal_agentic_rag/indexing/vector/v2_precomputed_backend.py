"""Precomputed V2 dense matrix backend for memory-mapped exact cosine serving."""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.vector.v2_retrieval_unit_store import V2RetrievalUnitStore
from legal_agentic_rag.schemas.preprocessing_v2 import RetrievalUnitV2
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)

_LOGGER = logging.getLogger(__name__)

EXPECTED_SCHEMA = "m54_v2_dense_matrix_index_v1"
EXPECTED_MODEL = "AITeamVN/Vietnamese_Embedding"
EXPECTED_REVISION = "dea33aa1ab339f38d66ae0a40e6c40e0a9249568"
EXPECTED_DIMENSION = 1024
EXPECTED_RECORD_COUNT = 1190081
EXPECTED_SOURCE_SHA256 = "e3be650fb8a797811c46b9b9ac2ba892c374304a72828006e37813a5d25a8a59"


class V2PrecomputedDenseBackend:
    """Exact cosine vector backend querying precomputed normalized NumPy embeddings."""

    backend_name = "v2_precomputed_dense"

    def __init__(
        self,
        *,
        matrix_dir: Path,
        manifest: dict[str, Any],
        vectors: np.ndarray,
        store: V2RetrievalUnitStore,
        batch_size: int = 65536,
    ) -> None:
        self._matrix_dir = Path(matrix_dir).resolve()
        self._manifest = manifest
        self._vectors = vectors
        self._store = store
        self._batch_size = max(1, batch_size)
        self._record_count = vectors.shape[0]
        self._dimension = vectors.shape[1]

    @property
    def matrix_dir(self) -> Path:
        return self._matrix_dir

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def store(self) -> V2RetrievalUnitStore:
        return self._store

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return str(self._manifest.get("model_name", EXPECTED_MODEL))

    @property
    def model_revision(self) -> str:
        return str(self._manifest.get("model_revision", EXPECTED_REVISION))

    @classmethod
    def load(
        cls,
        matrix_dir: Path,
        units_path: Path,
        *,
        verify_integrity: bool = True,
        batch_size: int = 65536,
        strict_manifest: bool = True,
    ) -> "V2PrecomputedDenseBackend":
        """Load and validate precomputed matrix and aligned retrieval-unit store."""
        matrix_dir = Path(matrix_dir).resolve()
        units_path = Path(units_path).resolve()

        if not matrix_dir.is_dir():
            raise ArtifactCompatibilityError(f"Matrix directory not found: {matrix_dir}")

        manifest_path = matrix_dir / "index_manifest_v1.json"
        if not manifest_path.is_file():
            raise ArtifactCompatibilityError(f"Missing index manifest at {manifest_path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ArtifactCompatibilityError(f"Malformed manifest JSON at {manifest_path}") from e

        # Validate manifest invariants
        schema = manifest.get("schema")
        if strict_manifest and schema != EXPECTED_SCHEMA:
            raise ArtifactCompatibilityError(f"Unsupported matrix schema: {schema}, expected {EXPECTED_SCHEMA}")

        dim = manifest.get("dimension")
        if strict_manifest and dim != EXPECTED_DIMENSION:
            raise ArtifactCompatibilityError(f"Expected dimension {EXPECTED_DIMENSION}, got {dim}")

        rec_count = manifest.get("record_count")
        if strict_manifest and rec_count != EXPECTED_RECORD_COUNT:
            raise ArtifactCompatibilityError(f"Expected record count {EXPECTED_RECORD_COUNT}, got {rec_count}")

        dtype_str = manifest.get("dtype")
        if dtype_str != "float32":
            raise ArtifactCompatibilityError(f"Expected float32 dtype, got {dtype_str}")

        metric = manifest.get("distance_metric")
        if metric != "cosine":
            raise ArtifactCompatibilityError(f"Expected cosine distance metric, got {metric}")

        if not manifest.get("normalized", False):
            raise ArtifactCompatibilityError("Matrix manifest indicates unnormalized vectors")

        if strict_manifest:
            m_name = manifest.get("model_name")
            if m_name != EXPECTED_MODEL:
                raise ArtifactCompatibilityError(f"Model mismatch: {m_name} != {EXPECTED_MODEL}")

            m_rev = manifest.get("model_revision")
            if m_rev != EXPECTED_REVISION:
                raise ArtifactCompatibilityError(f"Revision mismatch: {m_rev} != {EXPECTED_REVISION}")

            src_sha = manifest.get("source_retrieval_units_sha256")
            if src_sha != EXPECTED_SOURCE_SHA256:
                raise ArtifactCompatibilityError(f"Source units SHA mismatch: {src_sha} != {EXPECTED_SOURCE_SHA256}")

        vectors_filename = manifest.get("vectors_filename", "vectors.npy")
        vectors_path = matrix_dir / vectors_filename
        if not vectors_path.is_file():
            raise ArtifactCompatibilityError(f"Missing vectors file at {vectors_path}")

        # Memory-mapped load without copying 4.9 GB matrix into RAM
        try:
            vectors = np.load(vectors_path, mmap_mode="r")
        except Exception as e:
            raise BackendInitializationError(f"Failed to memory-map vectors at {vectors_path}") from e

        expected_shape = (rec_count, dim) if rec_count is not None and dim is not None else None
        if expected_shape is not None and vectors.shape != expected_shape:
            raise ArtifactCompatibilityError(
                f"Matrix shape {vectors.shape} does not match expected {expected_shape}"
            )
        if vectors.dtype != np.float32:
            raise ArtifactCompatibilityError(f"Matrix dtype {vectors.dtype} != float32")

        ids_filename = manifest.get("ids_filename", "retrieval_unit_ids.jsonl")
        ids_path = matrix_dir / ids_filename

        store = V2RetrievalUnitStore.load(
            units_path,
            ids_path=ids_path if ids_path.is_file() else None,
            expected_count=rec_count,
            verify_alignment=verify_integrity and ids_path.is_file(),
        )

        if len(store) != vectors.shape[0]:
            raise ArtifactCompatibilityError(
                f"Store count ({len(store)}) does not match matrix rows ({vectors.shape[0]})"
            )

        return cls(
            matrix_dir=matrix_dir,
            manifest=manifest,
            vectors=vectors,
            store=store,
            batch_size=batch_size,
        )

    def search_vector(
        self,
        query_vector: Sequence[float] | np.ndarray,
        *,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Perform exact cosine search in memory-bounded batches over the mmap matrix."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Validate query vector
        try:
            q = np.asarray(query_vector, dtype=np.float32)
        except Exception as e:
            raise DataValidationError("query_vector must be convertible to float32 numpy array") from e

        if q.ndim != 1 or q.shape[0] != self._dimension:
            raise DataValidationError(
                f"query_vector shape must be ({self._dimension},), got {q.shape}"
            )

        if not np.all(np.isfinite(q)):
            raise DataValidationError("query_vector contains NaN or infinite values")

        norm_q = float(np.linalg.norm(q))
        if norm_q <= 0.0 or not np.isfinite(norm_q):
            raise DataValidationError("query_vector norm is zero or invalid")

        q_norm = q / norm_q

        n_rows = self._record_count
        actual_k = min(top_k, n_rows)
        if actual_k == 0:
            return []

        # Batched dot product with local top-k accumulation
        all_candidate_indices: list[int] = []
        all_candidate_scores: list[float] = []

        for start_idx in range(0, n_rows, self._batch_size):
            end_idx = min(start_idx + self._batch_size, n_rows)
            # Slice mmap and compute dot product
            batch_slice = self._vectors[start_idx:end_idx]
            batch_scores = np.dot(batch_slice, q_norm)

            if len(batch_scores) > actual_k:
                local_indices = np.argpartition(batch_scores, -actual_k)[-actual_k:]
                local_indices = local_indices[np.argsort(-batch_scores[local_indices])]
            else:
                local_indices = np.argsort(-batch_scores)

            for l_idx in local_indices:
                all_candidate_indices.append(start_idx + int(l_idx))
                all_candidate_scores.append(float(batch_scores[l_idx]))

        # Global top-k selection with deterministic tie-breaking (score desc, index asc)
        cand_indices = np.array(all_candidate_indices, dtype=np.int64)
        cand_scores = np.array(all_candidate_scores, dtype=np.float32)

        # Sort: primary key -score, secondary key index
        sort_order = np.lexsort((cand_indices, -cand_scores))[:actual_k]

        return [(int(cand_indices[i]), float(cand_scores[i])) for i in sort_order]

    def retrieve(
        self,
        query: RetrievalQuery,
        query_vector: Sequence[float] | np.ndarray,
    ) -> RetrievalResponse:
        """Execute vector search and return standard RetrievalResponse with V2 hits."""
        t0 = perf_counter()
        top_results = self.search_vector(query_vector, top_k=query.top_k)

        if not top_results:
            elapsed_ms = (perf_counter() - t0) * 1000.0
            return RetrievalResponse(
                query=query,
                strategy=RetrievalStrategy.DENSE,
                hits=[],
                latency_ms=elapsed_ms,
                artifact_versions={"vector_index": str(self._manifest.get("schema", EXPECTED_SCHEMA))},
            )

        indices = [idx for idx, _ in top_results]
        scores = [score for _, score in top_results]
        units = self._store.get_many(indices)

        hits: list[RetrievalHit] = []
        for rank_idx, (unit, score) in enumerate(zip(units, scores, strict=True), start=1):
            metadata: dict[str, Any] = {
                "provision_id": unit.provision_id,
                "retrieval_text": unit.retrieval_text,
                "document_identity": unit.document_identity.model_dump(mode="json"),
                "hierarchy": unit.hierarchy.model_dump(mode="json"),
                "strategy": unit.strategy,
                "quality_flags": list(unit.quality_flags),
                "segment_index": unit.segment_index,
                "segment_count": unit.segment_count,
            }

            hit = RetrievalHit(
                chunk_id=unit.retrieval_unit_id,
                document_id=unit.document_id,
                rank=rank_idx,
                score=float(score),
                strategy=RetrievalStrategy.DENSE,
                text=unit.authority_text,
                metadata=metadata,
                retrieval_trace=RetrievalTrace(
                    dense_rank=rank_idx,
                    dense_score=float(score),
                ),
            )
            hits.append(hit)

        elapsed_ms = (perf_counter() - t0) * 1000.0
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.DENSE,
            hits=hits,
            latency_ms=elapsed_ms,
            artifact_versions={
                "vector_index": str(self._manifest.get("schema", EXPECTED_SCHEMA)),
                "model_name": self.model_name,
                "model_revision": self.model_revision,
            },
        )
