"""Tests for fixed sparse, dense, and hybrid retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from legal_agentic_rag.configuration import RetrievalConfig
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.retrieval import FixedRetriever, HybridRetriever
from legal_agentic_rag.schemas import (
    QueryVariant,
    QueryVariantKind,
    RetrievalFilters,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

SOURCE_IDENTITY = ("legal_chunks", "1.0", "chunk-hash")


def _query(
    strategy: RetrievalStrategy | None = RetrievalStrategy.HYBRID,
) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-1",
        original_question="Question",
        normalized_question="normalized question",
        filters=RetrievalFilters(legal_fields=["Traffic"]),
        top_k=2,
        candidate_k=5,
        requested_strategy=strategy,
    )


def _hit(chunk_id: str, strategy: RetrievalStrategy, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=strategy,
        text=f"Text {chunk_id}",
        metadata={"field": "Traffic"},
    )


@dataclass
class _Branch:
    strategy: RetrievalStrategy
    hits: list[RetrievalHit]
    source_artifact_identity: tuple[str, str, str] = SOURCE_IDENTITY
    warnings: list[str] = field(default_factory=list)
    calls: list[RetrievalQuery] = field(default_factory=list)

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.calls.append(query)
        return RetrievalResponse(
            query=query,
            strategy=self.strategy,
            hits=self.hits,
            warnings=self.warnings,
            artifact_versions={f"{self.strategy.value}_index": "1.0"},
        )


def _branches() -> tuple[_Branch, _Branch]:
    return (
        _Branch(
            RetrievalStrategy.BM25,
            [
                _hit("shared", RetrievalStrategy.BM25, 1),
                _hit("sparse", RetrievalStrategy.BM25, 2),
            ],
        ),
        _Branch(
            RetrievalStrategy.DENSE,
            [
                _hit("dense", RetrievalStrategy.DENSE, 1),
                _hit("shared", RetrievalStrategy.DENSE, 2),
            ],
        ),
    )


def test_hybrid_retriever_requests_candidate_pool_and_preserves_query() -> None:
    """Each branch receives candidate-k while final response retains caller limits."""
    bm25, dense = _branches()

    response = HybridRetriever(bm25, dense).search(_query())

    assert bm25.calls[0].requested_strategy == RetrievalStrategy.BM25
    assert dense.calls[0].requested_strategy == RetrievalStrategy.DENSE
    assert bm25.calls[0].top_k == dense.calls[0].top_k == 5
    assert bm25.calls[0].filters.legal_fields == ["Traffic"]
    assert response.query.top_k == 2
    assert response.query.candidate_k == 5
    assert response.strategy == RetrievalStrategy.HYBRID
    assert [hit.chunk_id for hit in response.hits] == ["shared", "dense"]
    assert response.artifact_versions == {
        "bm25_index": "1.0",
        "dense_index": "1.0",
    }


def test_hybrid_retriever_keeps_namespaced_warnings_and_empty_status() -> None:
    """Branch warning provenance stays visible and empty fusion is explicit."""
    bm25 = _Branch(RetrievalStrategy.BM25, [], warnings=["no_bm25_matches"])
    dense = _Branch(RetrievalStrategy.DENSE, [], warnings=["no_dense_matches"])

    response = HybridRetriever(bm25, dense).search(_query())

    assert response.hits == []
    assert response.warnings == [
        "bm25:no_bm25_matches",
        "dense:no_dense_matches",
        "no_hybrid_matches",
    ]


def test_hybrid_retriever_fuses_bounded_query_variants() -> None:
    """Hybrid retrieval calls both branches per planned variant and traces them."""
    bm25, dense = _branches()
    query = _query().model_copy(
        update={
            "query_variants": [
                QueryVariant(
                    variant_id="qv1",
                    text="normalized question",
                    kind=QueryVariantKind.NORMALIZED,
                ),
                QueryVariant(
                    variant_id="qv2",
                    text="question",
                    kind=QueryVariantKind.FRAMING_STRIPPED,
                ),
            ]
        }
    )

    response = HybridRetriever(bm25, dense).search(query)

    assert len(bm25.calls) == len(dense.calls) == 2
    assert bm25.calls[0].rewritten_question is None
    assert bm25.calls[1].rewritten_question == "question"
    shared = response.hits[0]
    assert shared.chunk_id == "shared"
    assert len(shared.retrieval_trace.query_variant_contributions) == 4
    assert shared.score == pytest.approx(2 / 61 + 2 / 62)


def test_hybrid_retriever_fails_closed_for_source_or_branch_mismatch() -> None:
    """Different corpora and malformed branch responses cannot be fused."""
    bm25, dense = _branches()
    dense.source_artifact_identity = ("legal_chunks", "1.0", "other-hash")
    with pytest.raises(ArtifactCompatibilityError, match="different chunk"):
        HybridRetriever(bm25, dense).search(_query())
    assert bm25.calls == dense.calls == []

    bad_dense = _Branch(RetrievalStrategy.BM25, [])
    with pytest.raises(RetrievalError, match="incompatible response"):
        HybridRetriever(bm25, bad_dense).search(_query())

    class _FailingBranch(_Branch):
        def search(self, query: RetrievalQuery) -> RetrievalResponse:
            raise RetrievalError("branch failed")

    with pytest.raises(RetrievalError, match="branch failed"):
        HybridRetriever(
            bm25,
            _FailingBranch(RetrievalStrategy.DENSE, []),
        ).search(_query())


def test_fixed_retriever_routes_supported_strategies_and_defaults_to_hybrid() -> None:
    """Fixed routing works without an Agent and rejects future strategies."""
    bm25, dense = _branches()
    retriever = FixedRetriever(bm25, dense)

    assert (
        retriever.search(_query(RetrievalStrategy.BM25)).strategy
        == RetrievalStrategy.BM25
    )
    assert (
        retriever.search(_query(RetrievalStrategy.DENSE)).strategy
        == RetrievalStrategy.DENSE
    )
    assert retriever.search(_query(None)).strategy == RetrievalStrategy.HYBRID

    with pytest.raises(RetrievalError, match="no reranker"):
        retriever.search(_query(RetrievalStrategy.HYBRID_RERANK))


def test_fixed_retriever_honors_configured_default_strategy() -> None:
    """Default strategy is typed configuration rather than routing magic."""
    bm25, dense = _branches()
    retriever = FixedRetriever(
        bm25,
        dense,
        RetrievalConfig(default_strategy=RetrievalStrategy.BM25),
    )

    assert retriever.search(_query(None)).strategy == RetrievalStrategy.BM25

def test_fixed_retriever_rejects_unimplemented_graph_strategy() -> None:
    """FixedRetriever deterministically rejects GRAPH requests as unsupported."""
    bm25, dense = _branches()
    retriever = FixedRetriever(bm25, dense)

    with pytest.raises(
        RetrievalError,
        match="Fixed retrieval strategy is not implemented: graph",
    ):
        retriever.search(_query(RetrievalStrategy.GRAPH))

def test_fixed_retriever_rejects_historical_graph_default_strategy_at_execution() -> None:
    """Historical default_strategy=GRAPH parses but fails at execution in FixedRetriever."""
    bm25, dense = _branches()
    retriever = FixedRetriever(
        bm25,
        dense,
        RetrievalConfig(default_strategy=RetrievalStrategy.GRAPH),
    )

    with pytest.raises(
        RetrievalError,
        match="Fixed retrieval strategy is not implemented: graph",
    ):
        retriever.search(_query(None))
