"""Exactness and device validation tests for optional torch vector scoring."""

from __future__ import annotations

import numpy as np
import pytest

from legal_agentic_rag.exceptions import BackendInitializationError
from legal_agentic_rag.indexing.vector.torch_scorer import (
    TorchExactVectorScorer,
)


def _normalized_vectors() -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    query = np.asarray([0.8, 0.6, 0.0], dtype=np.float32)
    return vectors, query


def test_torch_scorer_matches_numpy_for_full_and_filtered_search() -> None:
    """Torch scoring preserves exact candidate order and float32 similarity."""
    pytest.importorskip("torch")
    vectors, query = _normalized_vectors()
    scorer = TorchExactVectorScorer(
        vectors,
        device="cpu",
        transfer_batch_size=2,
        search_batch_size=1,
        progress_interval_records=2,
    )

    np.testing.assert_allclose(
        scorer.score(query, None),
        vectors @ query,
        rtol=1e-6,
        atol=1e-6,
    )
    indexes = np.asarray([3, 0, 2], dtype=np.int64)
    np.testing.assert_allclose(
        scorer.score(query, indexes),
        vectors[indexes] @ query,
        rtol=1e-6,
        atol=1e-6,
    )


def test_cuda_request_fails_closed_when_cuda_is_unavailable() -> None:
    """Explicit CUDA configuration never silently falls back to CPU."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")
    vectors, _ = _normalized_vectors()

    with pytest.raises(BackendInitializationError, match="unavailable"):
        TorchExactVectorScorer(
            vectors,
            device="cuda",
            transfer_batch_size=2,
            search_batch_size=2,
            progress_interval_records=2,
        )


def test_cuda_scorer_live_smoke_when_available() -> None:
    """A real CUDA environment produces the same exact small-matrix scores."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    vectors, query = _normalized_vectors()
    scorer = TorchExactVectorScorer(
        vectors,
        device="cuda",
        transfer_batch_size=2,
        search_batch_size=2,
        progress_interval_records=2,
    )

    np.testing.assert_allclose(
        scorer.score(query, None),
        vectors @ query,
        rtol=1e-5,
        atol=1e-5,
    )
