"""Integration from fixed hybrid retrieval to verified grounded answer."""

from dataclasses import dataclass

import numpy as np

from legal_agentic_rag.generation import FixedRAGService
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import FixedRetriever
from legal_agentic_rag.schemas import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


@dataclass
class _Branch:
    strategy: RetrievalStrategy
    source_artifact_identity: tuple[str, str, str] = (
        "legal_chunks",
        "1.0",
        "chunks-hash",
    )

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        hits = [
            RetrievalHit(
                chunk_id="chunk-applicable",
                document_id="doc-law",
                rank=1,
                score=2.0 if self.strategy == RetrievalStrategy.BM25 else 0.9,
                strategy=self.strategy,
                text="Quy định này chỉ áp dụng khi đáp ứng đủ điều kiện.",
                metadata={
                    "token_count": 11,
                    "document_title": "Luật thử nghiệm",
                    "document_number": "01/2026/QH",
                    "effect_status": "còn hiệu lực",
                    "source_url": "https://example.test/doc-law",
                    "structure": {
                        "article_number": "10",
                        "article_title": "Điều kiện áp dụng",
                    },
                },
            )
        ]
        return RetrievalResponse(
            query=query,
            strategy=self.strategy,
            hits=hits,
            artifact_versions={f"{self.strategy.value}_index": "1.0"},
        )


class _FixtureCrossEncoder:
    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object:
        return np.asarray([1.5 for _ in inputs], dtype=np.float32)


def test_question_flows_through_rrf_reranking_context_and_verification() -> None:
    """The completed fixed pipeline returns only cited retrieved legal content."""
    retriever = FixedRetriever(
        _Branch(RetrievalStrategy.BM25),
        _Branch(RetrievalStrategy.DENSE),
        reranker=CrossEncoderReranker(
            model_loader=lambda config: _FixtureCrossEncoder()
        ),
    )
    query = RetrievalQuery(
        query_id="fixed-rag-integration",
        original_question="Khi nào quy định này được áp dụng?",
        normalized_question="điều kiện áp dụng quy định",
        top_k=1,
        candidate_k=2,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )

    response = FixedRAGService(retriever).answer(query)

    assert response.insufficient_evidence is False
    assert "[E1] Quy định này chỉ áp dụng" in response.answer
    assert response.citations[0].article_number == "10"
    assert response.citations[0].document_number == "01/2026/QH"
    assert response.retrieval_strategy == RetrievalStrategy.HYBRID_RERANK
    trace = response.metadata["evidence_retrieval_trace"]["E1"]
    assert trace["rrf_score"] is not None
    assert trace["reranker_score"] == 1.5
    assert response.metadata["citation_verification"]["is_valid"] is True
