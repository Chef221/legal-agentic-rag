"""Unit tests for deterministic Reciprocal Rank Fusion."""

import pytest

from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.retrieval import reciprocal_rank_fusion
from legal_agentic_rag.schemas import RetrievalHit, RetrievalStrategy, RetrievalTrace


def _hit(
    chunk_id: str,
    *,
    rank: int,
    score: float,
    strategy: RetrievalStrategy,
    text: str | None = None,
) -> RetrievalHit:
    trace = (
        RetrievalTrace(bm25_rank=rank, bm25_score=score)
        if strategy == RetrievalStrategy.BM25
        else RetrievalTrace(dense_rank=rank, dense_score=score)
    )
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=score,
        strategy=strategy,
        text=text or f"Text {chunk_id}",
        metadata={"article": "1"},
        retrieval_trace=trace,
    )


def test_rrf_uses_ranks_deduplicates_and_records_contributions() -> None:
    """Raw scores never enter fusion and shared chunks receive both contributions."""
    shared_bm25 = _hit(
        "shared", rank=1, score=0.01, strategy=RetrievalStrategy.BM25
    )
    shared_dense = _hit(
        "shared", rank=2, score=0.99, strategy=RetrievalStrategy.DENSE
    )

    hits = reciprocal_rank_fusion(
        [
            shared_bm25,
            _hit("sparse", rank=2, score=1_000_000, strategy=RetrievalStrategy.BM25),
        ],
        [
            _hit("dense", rank=1, score=-999, strategy=RetrievalStrategy.DENSE),
            shared_dense,
        ],
        rrf_constant=60,
        top_k=3,
    )

    assert [hit.chunk_id for hit in hits] == ["shared", "dense", "sparse"]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    shared = hits[0]
    assert shared.strategy == RetrievalStrategy.HYBRID
    assert shared.retrieval_trace.bm25_rank == 1
    assert shared.retrieval_trace.bm25_score == 0.01
    assert shared.retrieval_trace.dense_rank == 2
    assert shared.retrieval_trace.dense_score == 0.99
    assert shared.retrieval_trace.bm25_rrf_contribution == pytest.approx(1 / 61)
    assert shared.retrieval_trace.dense_rrf_contribution == pytest.approx(1 / 62)
    assert shared.score == pytest.approx(1 / 61 + 1 / 62)
    assert shared.retrieval_trace.rrf_score == shared.score
    assert hits[1].retrieval_trace.bm25_rrf_contribution == 0.0
    assert hits[2].retrieval_trace.dense_rrf_contribution == 0.0


def test_rrf_top_k_and_ties_are_deterministic() -> None:
    """Equal fused scores use stable chunk IDs and obey the final result limit."""
    hits = reciprocal_rank_fusion(
        [_hit("z-chunk", rank=1, score=500, strategy=RetrievalStrategy.BM25)],
        [_hit("a-chunk", rank=1, score=-10, strategy=RetrievalStrategy.DENSE)],
        rrf_constant=60,
        top_k=1,
    )

    assert [hit.chunk_id for hit in hits] == ["a-chunk"]


def test_rrf_rejects_duplicate_or_inconsistent_branch_data() -> None:
    """Malformed backend responses cannot silently corrupt fusion provenance."""
    duplicate = _hit("duplicate", rank=1, score=1, strategy=RetrievalStrategy.BM25)
    with pytest.raises(RetrievalError, match="duplicate bm25"):
        reciprocal_rank_fusion(
            [duplicate, duplicate], [], rrf_constant=60, top_k=10
        )

    sparse = _hit("shared", rank=1, score=1, strategy=RetrievalStrategy.BM25)
    dense = _hit(
        "shared",
        rank=1,
        score=1,
        strategy=RetrievalStrategy.DENSE,
        text="Different text",
    )
    with pytest.raises(RetrievalError, match="disagree"):
        reciprocal_rank_fusion([sparse], [dense], rrf_constant=60, top_k=10)

    wrong_branch = _hit("wrong", rank=1, score=1, strategy=RetrievalStrategy.DENSE)
    with pytest.raises(RetrievalError, match="non-bm25"):
        reciprocal_rank_fusion([wrong_branch], [], rrf_constant=60, top_k=10)


def test_rrf_rejects_invalid_runtime_limits() -> None:
    """Direct callers cannot bypass positive fusion limits."""
    with pytest.raises(RetrievalError):
        reciprocal_rank_fusion([], [], rrf_constant=0, top_k=1)
    with pytest.raises(RetrievalError):
        reciprocal_rank_fusion([], [], rrf_constant=60, top_k=0)
