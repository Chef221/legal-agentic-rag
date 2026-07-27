"""Optional exact vector scoring on a resident PyTorch device matrix."""

from __future__ import annotations

from importlib import import_module
import logging
from types import ModuleType
from typing import Any

import numpy as np

from legal_agentic_rag.exceptions import BackendInitializationError, RetrievalError

_LOGGER = logging.getLogger(__name__)


class TorchExactVectorScorer:
    """Keep normalized vectors on one torch device for exact inner products."""

    def __init__(
        self,
        vectors: np.ndarray,
        *,
        device: str,
        transfer_batch_size: int,
        search_batch_size: int,
        progress_interval_records: int,
        torch_module: ModuleType | None = None,
    ) -> None:
        if (
            vectors.ndim != 2
            or transfer_batch_size <= 0
            or search_batch_size <= 0
            or progress_interval_records <= 0
        ):
            raise BackendInitializationError(
                "Torch vector scorer configuration is invalid"
            )
        self._torch = torch_module or _load_torch()
        self._device = device
        self._search_batch_size = search_batch_size
        self._matrix = self._load_matrix(
            vectors,
            transfer_batch_size=transfer_batch_size,
            progress_interval_records=progress_interval_records,
        )

    @property
    def device(self) -> str:
        """Return the configured torch device."""
        return self._device

    def score(
        self,
        query_vector: np.ndarray,
        candidate_indexes: np.ndarray | None,
    ) -> np.ndarray:
        """Return exact float32 inner products in candidate order."""
        if query_vector.shape != (int(self._matrix.shape[1]),):
            raise RetrievalError("Accelerated query vector shape is incompatible")
        candidate_count = (
            int(self._matrix.shape[0])
            if candidate_indexes is None
            else int(candidate_indexes.size)
        )
        try:
            with self._torch.inference_mode():
                query = self._torch.as_tensor(
                    np.array(query_vector, dtype=np.float32, copy=True),
                    dtype=self._torch.float32,
                    device=self._device,
                )
                if candidate_indexes is None:
                    result = self._torch.mv(self._matrix, query)
                    return result.to("cpu").numpy().astype(
                        np.float32,
                        copy=False,
                    )
                scores = np.empty(candidate_count, dtype=np.float32)
                for start in range(
                    0,
                    candidate_count,
                    self._search_batch_size,
                ):
                    end = min(
                        start + self._search_batch_size,
                        candidate_count,
                    )
                    indexes = self._torch.as_tensor(
                        np.array(
                            candidate_indexes[start:end],
                            dtype=np.int64,
                            copy=True,
                        ),
                        dtype=self._torch.int64,
                        device=self._device,
                    )
                    matrix = self._torch.index_select(
                        self._matrix,
                        0,
                        indexes,
                    )
                    scores[start:end] = (
                        self._torch.mv(matrix, query)
                        .to("cpu")
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                return scores
        except Exception as error:
            if isinstance(error, RetrievalError):
                raise
            raise RetrievalError(
                "Accelerated vector scoring failed"
            ) from error

    def _load_matrix(
        self,
        vectors: np.ndarray,
        *,
        transfer_batch_size: int,
        progress_interval_records: int,
    ) -> Any:
        if self._device.startswith("cuda"):
            try:
                if not self._torch.cuda.is_available():
                    raise BackendInitializationError(
                        "CUDA vector search was requested but is unavailable"
                    )
                self._torch.empty(1, device=self._device)
            except BackendInitializationError:
                raise
            except Exception as error:
                raise BackendInitializationError(
                    "CUDA vector search device cannot be initialized"
                ) from error
        _LOGGER.info(
            "accelerated_vector_matrix_load_started",
            extra={
                "device": self._device,
                "record_count": int(vectors.shape[0]),
                "dimension": int(vectors.shape[1]),
            },
        )
        try:
            matrix = self._torch.empty(
                tuple(int(value) for value in vectors.shape),
                dtype=self._torch.float32,
                device=self._device,
            )
            next_progress = progress_interval_records
            with self._torch.inference_mode():
                for start in range(0, len(vectors), transfer_batch_size):
                    end = min(start + transfer_batch_size, len(vectors))
                    batch = np.array(
                        vectors[start:end],
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )
                    source = self._torch.from_numpy(batch)
                    matrix[start:end].copy_(source)
                    if end >= next_progress:
                        _LOGGER.info(
                            "accelerated_vector_matrix_load_progress",
                            extra={
                                "device": self._device,
                                "record_count": end,
                                "total_record_count": len(vectors),
                            },
                        )
                        next_progress = (
                            (end // progress_interval_records) + 1
                        ) * progress_interval_records
        except Exception as error:
            raise BackendInitializationError(
                "Accelerated vector matrix could not be loaded"
            ) from error
        _LOGGER.info(
            "accelerated_vector_matrix_load_completed",
            extra={
                "device": self._device,
                "record_count": int(vectors.shape[0]),
                "dimension": int(vectors.shape[1]),
            },
        )
        return matrix


def _load_torch() -> ModuleType:
    try:
        return import_module("torch")
    except ImportError as error:
        raise BackendInitializationError(
            "PyTorch is required for accelerated vector search"
        ) from error
