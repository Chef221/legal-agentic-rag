"""Tests for bounded hybrid retrieval followed by reranking."""

from dataclasses import dataclass, field
from collections.abc import Sequence

import pytest

from legal_agentic_rag.configuration import RerankerConfig
from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.retrieval import RerankingRetriever
from legal_agentic_rag.schemas import (
    QueryAnalysis,
    QueryIntent,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)


def _query(*, candidate_k: int = 3) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-service",
        original_question="Question",
        normalized_question="question",
        top_k=2,
        candidate_k=candidate_k,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )


def _hit(chunk_id: str, rank: int, strategy: RetrievalStrategy) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=strategy,
        text=f"Text {chunk_id}",
        retrieval_trace=RetrievalTrace(rrf_score=1 / (60 + rank)),
    )


@dataclass
class _CandidateRetriever:
    hits: list[RetrievalHit]
    source_artifact_identity: tuple[str, str, str] = (
        "legal_chunks",
        "1.0",
        "chunk-hash",
    )
    calls: list[RetrievalQuery] = field(default_factory=list)

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.calls.append(query)
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            hits=self.hits,
            warnings=["branch-warning"],
            artifact_versions={"bm25_index": "1.0", "vector_index": "1.0"},
        )


class _FixtureReranker:
    provider_name = "fixture-provider"
    provider_version = "1.0"
    model_name = "fixture-reranker"
    model_revision = "fixture-revision"

    def __init__(self) -> None:
        self.calls: list[tuple[RetrievalQuery, list[RetrievalHit]]] = []

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> RetrievalResponse:
        values = list(candidates)
        self.calls.append((query, values))
        selected = [
            hit.model_copy(
                update={
                    "rank": rank,
                    "score": float(100 - hit.rank),
                    "strategy": RetrievalStrategy.RERANK,
                    "retrieval_trace": hit.retrieval_trace.model_copy(
                        update={"reranker_score": float(100 - hit.rank)}
                    ),
                }
            )
            for rank, hit in enumerate(reversed(values[: query.top_k]), start=1)
        ]
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.RERANK,
            hits=selected,
            warnings=["model-warning"],
        )


def test_reranking_service_requests_candidates_and_preserves_provenance() -> None:
    """Hybrid candidate-k becomes reranked final top-k with total provenance."""
    candidates = [
        _hit("one", 1, RetrievalStrategy.HYBRID),
        _hit("two", 2, RetrievalStrategy.HYBRID),
        _hit("three", 3, RetrievalStrategy.HYBRID),
    ]
    base = _CandidateRetriever(candidates)
    reranker = _FixtureReranker()

    response = RerankingRetriever(base, reranker).search(_query())

    assert base.calls[0].requested_strategy == RetrievalStrategy.HYBRID
    assert base.calls[0].top_k == 3
    assert reranker.calls[0][0].requested_strategy == RetrievalStrategy.RERANK
    assert reranker.calls[0][0].top_k == 2
    assert response.strategy == RetrievalStrategy.HYBRID_RERANK
    assert [hit.chunk_id for hit in response.hits] == ["two", "one"]
    assert all(hit.strategy == RetrievalStrategy.HYBRID_RERANK for hit in response.hits)
    assert response.hits[0].retrieval_trace.rrf_score is not None
    assert response.hits[0].retrieval_trace.reranker_score is not None
    assert response.warnings == ["branch-warning", "reranker:model-warning"]
    assert response.artifact_versions == {
        "bm25_index": "1.0",
        "vector_index": "1.0",
    }
    assert response.latency_ms >= 0


def test_reranking_service_rejects_unbounded_query_before_retrieval() -> None:
    """Candidate limits are enforced before expensive retrieval or model calls."""
    base = _CandidateRetriever([])
    with pytest.raises(RetrievalError, match="candidate-k"):
        RerankingRetriever(
            base,
            _FixtureReranker(),
            RerankerConfig(max_candidates=2),
        ).search(_query(candidate_k=3))
    assert base.calls == []


def test_reranking_service_rejects_invented_or_malformed_results() -> None:
    """A reranker cannot create candidates or violate final rank contracts."""
    candidates = [_hit("one", 1, RetrievalStrategy.HYBRID)]

    class _BadReranker(_FixtureReranker):
        def rerank(
            self,
            query: RetrievalQuery,
            values: Sequence[RetrievalHit],
        ) -> RetrievalResponse:
            invented = _hit("invented", 2, RetrievalStrategy.RERANK)
            return RetrievalResponse(
                query=query,
                strategy=RetrievalStrategy.RERANK,
                hits=[invented],
            )

    with pytest.raises(RetrievalError, match="incompatible"):
        RerankingRetriever(
            _CandidateRetriever(candidates), _BadReranker()
        ).search(_query())


def test_reranking_service_rejects_changed_candidate_payload() -> None:
    """Reranking may change score/rank but not legal text or metadata."""
    candidates = [_hit("one", 1, RetrievalStrategy.HYBRID)]

    class _BadReranker(_FixtureReranker):
        def rerank(
            self,
            query: RetrievalQuery,
            values: Sequence[RetrievalHit],
        ) -> RetrievalResponse:
            changed = values[0].model_copy(
                update={"text": "tampered legal text"}
            )
            return RetrievalResponse(
                query=query,
                strategy=RetrievalStrategy.RERANK,
                hits=[changed],
            )

    with pytest.raises(RetrievalError, match="incompatible"):
        RerankingRetriever(
            _CandidateRetriever(candidates), _BadReranker()
        ).search(_query())


def test_reranking_service_rejects_changed_retrieval_provenance() -> None:
    """A backend cannot erase or replace the candidate's retrieval trace."""
    candidates = [_hit("one", 1, RetrievalStrategy.HYBRID)]

    class _BadReranker(_FixtureReranker):
        def rerank(
            self,
            query: RetrievalQuery,
            values: Sequence[RetrievalHit],
        ) -> RetrievalResponse:
            changed = values[0].model_copy(
                update={
                    "score": 1.0,
                    "strategy": RetrievalStrategy.RERANK,
                    "retrieval_trace": RetrievalTrace(reranker_score=1.0),
                }
            )
            return RetrievalResponse(
                query=query,
                strategy=RetrievalStrategy.RERANK,
                hits=[changed],
            )

    with pytest.raises(RetrievalError, match="incompatible"):
        RerankingRetriever(
            _CandidateRetriever(candidates), _BadReranker()
        ).search(_query())

def test_reranking_service_narrows_candidate_pool_for_relationship_query() -> None:
    """Relationship queries with relationship_candidate_k narrow candidate retrieval to 20."""
    candidates = [_hit(f"chunk-{i}", i, RetrievalStrategy.HYBRID) for i in range(1, 21)]
    base = _CandidateRetriever(candidates)
    reranker = _FixtureReranker()
    config = RerankerConfig(max_candidates=40, relationship_candidate_k=20)

    query = RetrievalQuery(
        query_id="rel-query",
        original_question="Nghị định này sửa đổi văn bản nào?",
        normalized_question="nghị định sửa đổi văn bản",
        top_k=10,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        query_analysis=QueryAnalysis(
            intent=QueryIntent.RELATIONSHIP,
            relationship_cues=["sửa đổi"],
        ),
    )

    response = RerankingRetriever(base, reranker, config).search(query)

    assert base.calls[0].requested_strategy == RetrievalStrategy.HYBRID
    assert base.calls[0].top_k == 20
    assert base.calls[0].candidate_k == 40
    assert query.candidate_k == 40
    assert reranker.calls[0][0].top_k == 10
    assert response.strategy == RetrievalStrategy.HYBRID_RERANK
    assert len(response.hits) == 10


def test_reranking_service_preserves_candidate_pool_for_non_relationship_query() -> None:
    """Non-relationship queries retain configured candidate_k (40)."""
    candidates = [_hit(f"chunk-{i}", i, RetrievalStrategy.HYBRID) for i in range(1, 41)]
    base = _CandidateRetriever(candidates)
    reranker = _FixtureReranker()
    config = RerankerConfig(max_candidates=40, relationship_candidate_k=20)

    query = RetrievalQuery(
        query_id="general-query",
        original_question="Quy định về thuế là gì?",
        normalized_question="quy định thuế",
        top_k=10,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        query_analysis=QueryAnalysis(
            intent=QueryIntent.GENERAL,
        ),
    )

    RerankingRetriever(base, reranker, config).search(query)

    assert base.calls[0].requested_strategy == RetrievalStrategy.HYBRID
    assert base.calls[0].top_k == 40


def test_reranking_service_preserves_candidate_pool_when_relationship_candidate_k_is_none() -> None:
    """When relationship_candidate_k is None, relationship queries retain candidate_k (40)."""
    candidates = [_hit(f"chunk-{i}", i, RetrievalStrategy.HYBRID) for i in range(1, 41)]
    base = _CandidateRetriever(candidates)
    reranker = _FixtureReranker()
    config = RerankerConfig(max_candidates=40, relationship_candidate_k=None)

    query = RetrievalQuery(
        query_id="rel-query-default",
        original_question="Nghị định sửa đổi văn bản nào?",
        normalized_question="nghị định sửa đổi văn bản",
        top_k=10,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        query_analysis=QueryAnalysis(
            intent=QueryIntent.RELATIONSHIP,
            relationship_cues=["sửa đổi"],
        ),
    )

    RerankingRetriever(base, reranker, config).search(query)

    assert base.calls[0].requested_strategy == RetrievalStrategy.HYBRID
    assert base.calls[0].top_k == 40


def test_reranking_service_respects_lower_candidate_k_than_relationship_candidate_k() -> None:
    """When query.candidate_k is less than relationship_candidate_k, use min (candidate_k)."""
    candidates = [_hit(f"chunk-{i}", i, RetrievalStrategy.HYBRID) for i in range(1, 16)]
    base = _CandidateRetriever(candidates)
    reranker = _FixtureReranker()
    config = RerankerConfig(max_candidates=40, relationship_candidate_k=20)

    query = RetrievalQuery(
        query_id="rel-query-narrow",
        original_question="Nghị định sửa đổi văn bản nào?",
        normalized_question="nghị định sửa đổi văn bản",
        top_k=5,
        candidate_k=15,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        query_analysis=QueryAnalysis(
            intent=QueryIntent.RELATIONSHIP,
            relationship_cues=["sửa đổi"],
        ),
    )

    RerankingRetriever(base, reranker, config).search(query)

    assert base.calls[0].top_k == 15


def test_reranking_service_rejects_candidate_response_exceeding_effective_candidate_top_k() -> None:
    """Candidate response with more hits than requested candidate_top_k is rejected."""
    candidates = [_hit(f"chunk-{i}", i, RetrievalStrategy.HYBRID) for i in range(1, 22)]
    base = _CandidateRetriever(candidates)
    reranker = _FixtureReranker()
    config = RerankerConfig(max_candidates=40, relationship_candidate_k=20)

    query = RetrievalQuery(
        query_id="rel-query-overflow",
        original_question="Nghị định sửa đổi văn bản nào?",
        normalized_question="nghị định sửa đổi văn bản",
        top_k=10,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        query_analysis=QueryAnalysis(
            intent=QueryIntent.RELATIONSHIP,
            relationship_cues=["sửa đổi"],
        ),
    )

    with pytest.raises(RetrievalError, match="incompatible"):
        RerankingRetriever(base, reranker, config).search(query)

def test_reranking_service_rejects_relationship_candidate_pool_smaller_than_top_k() -> None:
    """When effective candidate_top_k is smaller than requested top_k, fail fast before candidate retrieval."""
    base = _CandidateRetriever([])
    reranker = _FixtureReranker()
    config = RerankerConfig(max_candidates=40, relationship_candidate_k=5)

    query = RetrievalQuery(
        query_id="rel-query-insufficient",
        original_question="Nghị định sửa đổi văn bản nào?",
        normalized_question="nghị định sửa đổi văn bản",
        top_k=10,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        query_analysis=QueryAnalysis(
            intent=QueryIntent.RELATIONSHIP,
            relationship_cues=["sửa đổi"],
        ),
    )

    with pytest.raises(
        RetrievalError,
        match="Effective reranking candidate pool cannot be smaller than top_k",
    ):
        RerankingRetriever(base, reranker, config).search(query)

    assert base.calls == []
