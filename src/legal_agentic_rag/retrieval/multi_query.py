"""Rank fusion across bounded sparse and dense query variants."""

from __future__ import annotations

from dataclasses import dataclass, field

from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.schemas.retrieval import (
    QueryVariantContribution,
    RetrievalHit,
    RetrievalStrategy,
    RetrievalTrace,
)


@dataclass(frozen=True, slots=True)
class QueryBranchResult:
    """One ranked backend result list for a named query variant."""

    variant_id: str
    strategy: RetrievalStrategy
    hits: list[RetrievalHit]


@dataclass(slots=True)
class _Candidate:
    hit: RetrievalHit
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_contribution: float = 0.0
    dense_contribution: float = 0.0
    contributions: list[QueryVariantContribution] = field(default_factory=list)


def fuse_query_branches(
    branches: list[QueryBranchResult],
    *,
    rrf_constant: int,
    top_k: int,
) -> list[RetrievalHit]:
    """Fuse all variant/branch ranks while preserving raw-score provenance."""
    if rrf_constant <= 0 or top_k <= 0:
        raise RetrievalError("Query-fusion RRF constant and top-k must be positive")
    branch_identities = [
        (branch.variant_id, branch.strategy) for branch in branches
    ]
    if len(branch_identities) != len(set(branch_identities)):
        raise RetrievalError("Query fusion received a duplicate branch identity")

    candidates: dict[str, _Candidate] = {}
    for branch in branches:
        _add_branch(candidates, branch, rrf_constant)

    scored = [
        (
            candidate.bm25_contribution + candidate.dense_contribution,
            chunk_id,
            candidate,
        )
        for chunk_id, candidate in candidates.items()
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))

    fused: list[RetrievalHit] = []
    for rank, (score, _, candidate) in enumerate(scored[:top_k], start=1):
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
                        bm25_rrf_contribution=candidate.bm25_contribution,
                        dense_rrf_contribution=candidate.dense_contribution,
                        rrf_score=score,
                        query_variant_contributions=candidate.contributions,
                    ),
                }
            )
        )
    return fused


def _add_branch(
    candidates: dict[str, _Candidate],
    branch: QueryBranchResult,
    rrf_constant: int,
) -> None:
    if branch.strategy not in {
        RetrievalStrategy.BM25,
        RetrievalStrategy.DENSE,
    }:
        raise RetrievalError("Query fusion only accepts bm25 and dense branches")
    if not branch.variant_id.strip():
        raise RetrievalError("Query fusion requires a non-empty variant ID")

    seen_chunks: set[str] = set()
    seen_ranks: set[int] = set()
    for hit in branch.hits:
        if hit.strategy != branch.strategy:
            raise RetrievalError(
                "Query fusion received a hit from an incompatible branch"
            )
        if hit.chunk_id in seen_chunks or hit.rank in seen_ranks:
            raise RetrievalError(
                "Query fusion received duplicate rank data within one branch"
            )
        seen_chunks.add(hit.chunk_id)
        seen_ranks.add(hit.rank)
        candidate = candidates.get(hit.chunk_id)
        if candidate is None:
            candidate = _Candidate(hit=hit)
            candidates[hit.chunk_id] = candidate
        else:
            _validate_same_chunk(candidate.hit, hit)

        contribution = 1.0 / (rrf_constant + hit.rank)
        candidate.contributions.append(
            QueryVariantContribution(
                variant_id=branch.variant_id,
                strategy=branch.strategy,
                rank=hit.rank,
                raw_score=hit.score,
                rrf_contribution=contribution,
            )
        )
        if branch.strategy == RetrievalStrategy.BM25:
            candidate.bm25_contribution += contribution
            if candidate.bm25_rank is None or hit.rank < candidate.bm25_rank:
                candidate.bm25_rank = hit.rank
                candidate.bm25_score = hit.score
        else:
            candidate.dense_contribution += contribution
            if candidate.dense_rank is None or hit.rank < candidate.dense_rank:
                candidate.dense_rank = hit.rank
                candidate.dense_score = hit.score


def _validate_same_chunk(first: RetrievalHit, second: RetrievalHit) -> None:
    if (
        first.document_id != second.document_id
        or first.text != second.text
        or first.metadata != second.metadata
    ):
        raise RetrievalError(
            "Query-fusion branches disagree on a duplicated chunk"
        )
