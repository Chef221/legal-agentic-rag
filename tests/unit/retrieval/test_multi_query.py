"""Unit tests for bounded multi-query sparse/dense rank fusion."""

import pytest

from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.retrieval import QueryBranchResult, fuse_query_branches
from legal_agentic_rag.schemas import RetrievalHit, RetrievalStrategy


def _hit(
    chunk_id: str,
    strategy: RetrievalStrategy,
    rank: int,
    *,
    text: str | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=strategy,
        text=text or f"Text {chunk_id}",
        metadata={"source": "fixture"},
    )


def test_multi_query_fusion_sums_each_variant_branch_contribution() -> None:
    """A repeated relevant chunk gains traceable RRF support across variants."""
    branches = [
        QueryBranchResult(
            "qv1",
            RetrievalStrategy.BM25,
            [_hit("shared", RetrievalStrategy.BM25, 1)],
        ),
        QueryBranchResult(
            "qv1",
            RetrievalStrategy.DENSE,
            [_hit("dense", RetrievalStrategy.DENSE, 1)],
        ),
        QueryBranchResult(
            "qv2",
            RetrievalStrategy.BM25,
            [_hit("other", RetrievalStrategy.BM25, 1)],
        ),
        QueryBranchResult(
            "qv2",
            RetrievalStrategy.DENSE,
            [
                _hit("shared", RetrievalStrategy.DENSE, 1),
                _hit("dense", RetrievalStrategy.DENSE, 2),
            ],
        ),
    ]

    hits = fuse_query_branches(branches, rrf_constant=60, top_k=3)

    assert [item.chunk_id for item in hits] == ["shared", "dense", "other"]
    shared = hits[0]
    assert shared.score == pytest.approx(2 / 61)
    assert shared.retrieval_trace.bm25_rrf_contribution == pytest.approx(1 / 61)
    assert shared.retrieval_trace.dense_rrf_contribution == pytest.approx(1 / 61)
    assert [
        (item.variant_id, item.strategy)
        for item in shared.retrieval_trace.query_variant_contributions
    ] == [
        ("qv1", RetrievalStrategy.BM25),
        ("qv2", RetrievalStrategy.DENSE),
    ]


def test_multi_query_fusion_rejects_conflicting_duplicate_payload() -> None:
    """The same chunk ID cannot hide different legal content across variants."""
    branches = [
        QueryBranchResult(
            "qv1",
            RetrievalStrategy.BM25,
            [_hit("same", RetrievalStrategy.BM25, 1)],
        ),
        QueryBranchResult(
            "qv2",
            RetrievalStrategy.DENSE,
            [_hit("same", RetrievalStrategy.DENSE, 1, text="Different text")],
        ),
    ]

    with pytest.raises(RetrievalError, match="disagree"):
        fuse_query_branches(branches, rrf_constant=60, top_k=2)


def test_multi_query_fusion_rejects_duplicate_branch_identity() -> None:
    """Each variant/backend pair must appear exactly once."""
    branches = [
        QueryBranchResult("qv1", RetrievalStrategy.BM25, []),
        QueryBranchResult("qv1", RetrievalStrategy.BM25, []),
    ]

    with pytest.raises(RetrievalError, match="duplicate branch"):
        fuse_query_branches(branches, rrf_constant=60, top_k=1)
