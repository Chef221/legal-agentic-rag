"""Tests for fail-closed fixed retrieval-to-answer orchestration."""

from dataclasses import dataclass
from collections.abc import Sequence

from legal_agentic_rag.generation import FixedRAGService
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    Evidence,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="rag-query",
        original_question="Quy định này áp dụng thế nào?",
        normalized_question="quy định áp dụng",
        top_k=1,
        candidate_k=2,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )


@dataclass
class _Retriever:
    hits: list[RetrievalHit]

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.HYBRID_RERANK,
            hits=self.hits,
            latency_ms=2.0,
            artifact_versions={"bm25_index": "1.0"},
        )


class _InventingGenerator:
    def generate(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
    ) -> AnswerResponse:
        return AnswerResponse(
            question=query.original_question,
            answer="Câu trả lời có citation giả.",
            citations=[
                Citation(
                    evidence_id="E1",
                    chunk_id="invented",
                    document_id="doc-invented",
                )
            ],
            insufficient_evidence=False,
            retrieval_strategy=retrieval_strategy,
            trace_id=trace_id,
        )


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="chunk-1",
        document_id="doc-1",
        rank=1,
        score=0.9,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text="Không áp dụng trong trường hợp ngoại lệ.",
        metadata={"token_count": 8, "document_number": "01/2026/QH"},
    )


def test_fixed_rag_returns_verified_answer_with_trace_metadata() -> None:
    """A valid context reaches generation and retains retrieval provenance."""
    response = FixedRAGService(_Retriever([_hit()])).answer(_query())

    assert response.insufficient_evidence is False
    assert response.citations[0].chunk_id == "chunk-1"
    assert response.trace_id == "rag-query"
    assert response.metadata["retrieval"]["artifact_versions"] == {
        "bm25_index": "1.0"
    }
    assert response.metadata["selected_evidence_ids"] == ["E1"]
    assert response.metadata["context_grade"]["is_sufficient"] is True
    assert (
        response.metadata["citation_verification"]["is_valid"] is True
    )


def test_fixed_rag_abstains_for_empty_retrieval() -> None:
    """No retrieval evidence short-circuits generation without fabricated law."""
    response = FixedRAGService(_Retriever([])).answer(_query())

    assert response.insufficient_evidence is True
    assert response.citations == []
    assert "insufficient_context" in response.warnings
    assert response.metadata["context_grade"]["is_sufficient"] is False


def test_fixed_rag_replaces_unverifiable_generated_answer_with_abstention() -> None:
    """An invented citation never escapes the fixed service boundary."""
    response = FixedRAGService(
        _Retriever([_hit()]),
        answer_generator=_InventingGenerator(),
    ).answer(_query())

    assert response.insufficient_evidence is True
    assert response.citations == []
    assert "citation_verification_failed" in response.warnings
    assert response.metadata["citation_verification"]["is_valid"] is False
