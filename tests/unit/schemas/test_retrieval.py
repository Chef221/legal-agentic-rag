"""Tests for query, filter, hit, response, and trace contracts."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.retrieval import (
    GraphPathStep,
    RetrievalFilters,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-1",
        original_question="Điều kiện là gì?",
        normalized_question="Điều kiện là gì?",
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )


def test_retrieval_query_uses_typed_filters_and_limits() -> None:
    """Query filters are backend-neutral and candidate count covers top-k."""
    query = RetrievalQuery(
        query_id="query-1",
        original_question="Câu hỏi",
        normalized_question="Câu hỏi",
        filters=RetrievalFilters(document_ids=["doc-1"]),
        top_k=5,
        candidate_k=20,
    )
    assert query.filters.document_ids == ["doc-1"]

    with pytest.raises(ValidationError):
        RetrievalQuery(
            query_id="query-2",
            original_question="Câu hỏi",
            normalized_question="Câu hỏi",
            top_k=10,
            candidate_k=5,
        )


def test_retrieval_filters_reject_backend_specific_fields() -> None:
    """Concrete backend filters cannot leak into the unified contract."""
    with pytest.raises(ValidationError):
        RetrievalFilters(index_partition="backend-specific")


def test_retrieval_trace_preserves_branch_contributions() -> None:
    """RRF trace stores separate BM25 and dense contributions."""
    trace = RetrievalTrace(
        bm25_rank=1,
        dense_rank=2,
        bm25_rrf_contribution=0.016,
        dense_rrf_contribution=0.015,
        rrf_score=0.031,
    )
    assert trace.rrf_score == pytest.approx(0.031)


def test_graph_trace_requires_consistent_hop_metadata() -> None:
    """Graph hop count must match the typed path."""
    step = GraphPathStep(
        source_document_id="doc-1",
        target_document_id="doc-2",
        relationship_type="amends",
        hop=1,
    )
    trace = RetrievalTrace(graph_hop=1, graph_path=[step])
    assert trace.graph_path[0].target_document_id == "doc-2"

    with pytest.raises(ValidationError):
        RetrievalTrace(graph_hop=2, graph_path=[step])


def test_retrieval_response_keeps_artifact_versions() -> None:
    """Retrieval output remains traceable to persisted artifacts."""
    query = _query()
    hit = RetrievalHit(
        chunk_id="chunk-1",
        document_id="doc-1",
        rank=1,
        score=0.5,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text="Điều 1. Nội dung.",
    )
    response = RetrievalResponse(
        query=query,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        hits=[hit],
        artifact_versions={"bm25": "v1", "vector": "v2"},
    )
    assert response.hits[0].rank == 1
    assert response.artifact_versions["bm25"] == "v1"
