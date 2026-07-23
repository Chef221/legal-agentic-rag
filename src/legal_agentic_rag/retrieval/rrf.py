"""Deterministic Reciprocal Rank Fusion over unified retrieval hits."""

from __future__ import annotations

from dataclasses import dataclass

from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalStrategy,
    RetrievalTrace,
)


@dataclass
class _Candidate:
    """Internal aligned sparse and dense views of one chunk."""

    hit: RetrievalHit
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None


def reciprocal_rank_fusion(
    bm25_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    *,
    rrf_constant: int,
    top_k: int,
) -> list[RetrievalHit]:
    """Fuse sparse and dense ranks without combining backend raw scores."""
    if rrf_constant <= 0 or top_k <= 0:
        raise RetrievalError("RRF constant and top-k must be positive")
    candidates: dict[str, _Candidate] = {}
    _add_branch(candidates, bm25_hits, RetrievalStrategy.BM25)
    _add_branch(candidates, dense_hits, RetrievalStrategy.DENSE)

    scored: list[tuple[float, str, _Candidate, float, float]] = []
    for chunk_id, candidate in candidates.items():
        bm25_contribution = _contribution(candidate.bm25_rank, rrf_constant)
        dense_contribution = _contribution(candidate.dense_rank, rrf_constant)
        score = bm25_contribution + dense_contribution
        scored.append(
            (
                score,
                chunk_id,
                candidate,
                bm25_contribution,
                dense_contribution,
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]))

    fused: list[RetrievalHit] = []
    for rank, item in enumerate(scored[:top_k], start=1):
        score, _, candidate, bm25_contribution, dense_contribution = item
        fused.append(
            candidate.hit.model_copy(
                update={
                    "rank": rank,
                    "score": score,
                    "strategy": RetrievalStrategy.HYBRID,
                    "retrieval_trace": RetrievalTrace(
                        bm25_rank=candidate.bm25_rank,
                        bm25_score=candidate.bm25_score,
                        dense_rank=candidate.dense_rank,
                        dense_score=candidate.dense_score,
                        bm25_rrf_contribution=bm25_contribution,
                        dense_rrf_contribution=dense_contribution,
                        rrf_score=score,
                    ),
                }
            )
        )
    return fused


def _add_branch(
    candidates: dict[str, _Candidate],
    hits: list[RetrievalHit],
    strategy: RetrievalStrategy,
) -> None:
    seen_chunks: set[str] = set()
    seen_ranks: set[int] = set()
    for hit in hits:
        if hit.strategy != strategy:
            raise RetrievalError(f"RRF received a non-{strategy.value} branch hit")
        if hit.chunk_id in seen_chunks or hit.rank in seen_ranks:
            raise RetrievalError(f"RRF received duplicate {strategy.value} rank data")
        seen_chunks.add(hit.chunk_id)
        seen_ranks.add(hit.rank)
        candidate = candidates.get(hit.chunk_id)
        if candidate is None:
            candidate = _Candidate(hit=hit)
            candidates[hit.chunk_id] = candidate
        else:
            _validate_same_chunk(candidate.hit, hit)
        if strategy == RetrievalStrategy.BM25:
            candidate.bm25_rank = hit.rank
            candidate.bm25_score = hit.score
        else:
            candidate.dense_rank = hit.rank
            candidate.dense_score = hit.score


def _validate_same_chunk(first: RetrievalHit, second: RetrievalHit) -> None:
    if (
        first.document_id != second.document_id
        or first.text != second.text
        or first.metadata != second.metadata
    ):
        raise RetrievalError("Retrieval branches disagree on a duplicated chunk")


def _contribution(rank: int | None, rrf_constant: int) -> float:
    return 0.0 if rank is None else 1.0 / (rrf_constant + rank)
