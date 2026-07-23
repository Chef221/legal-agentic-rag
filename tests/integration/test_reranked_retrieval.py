"""Integration from hybrid branches through concrete cross-encoder reranking."""

from dataclasses import dataclass

import numpy as np

from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import FixedRetriever
from legal_agentic_rag.schemas import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


class _SemanticFixtureModel:
    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object:
        return np.asarray(
            [2.0 if "tốc độ" in passage else -1.0 for _, passage in inputs],
            dtype=np.float32,
        )


@dataclass
class _Branch:
    strategy: RetrievalStrategy
    hits: list[RetrievalHit]
    source_artifact_identity: tuple[str, str, str] = (
        "legal_chunks",
        "1.0",
        "shared-hash",
    )

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        return RetrievalResponse(
            query=query,
            strategy=self.strategy,
            hits=self.hits,
            artifact_versions={f"{self.strategy.value}_index": "1.0"},
        )


def _hit(
    chunk_id: str,
    rank: int,
    strategy: RetrievalStrategy,
    text: str,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=strategy,
        text=text,
        metadata={"document_number": f"{chunk_id}/2026/QH"},
    )


def test_fixed_hybrid_rerank_keeps_rrf_and_model_trace() -> None:
    """RRF candidates flow through concrete reranker to final fixed response."""
    bm25 = _Branch(
        RetrievalStrategy.BM25,
        [
            _hit("license", 1, RetrievalStrategy.BM25, "Giấy phép lái xe."),
            _hit("speed", 2, RetrievalStrategy.BM25, "Không chạy quá tốc độ."),
        ],
    )
    dense = _Branch(
        RetrievalStrategy.DENSE,
        [
            _hit("license", 1, RetrievalStrategy.DENSE, "Giấy phép lái xe."),
            _hit("speed", 2, RetrievalStrategy.DENSE, "Không chạy quá tốc độ."),
        ],
    )
    reranker = CrossEncoderReranker(
        model_loader=lambda config: _SemanticFixtureModel()
    )
    response = FixedRetriever(
        bm25,
        dense,
        reranker=reranker,
    ).search(
        RetrievalQuery(
            query_id="query-full-rerank",
            original_question="Quy định chạy xe nhanh?",
            normalized_question="quy định tốc độ xe",
            top_k=1,
            candidate_k=2,
            requested_strategy=RetrievalStrategy.HYBRID_RERANK,
        )
    )

    assert response.strategy == RetrievalStrategy.HYBRID_RERANK
    assert [hit.chunk_id for hit in response.hits] == ["speed"]
    assert response.hits[0].retrieval_trace.rrf_score is not None
    assert response.hits[0].retrieval_trace.reranker_score == 2.0
    assert response.hits[0].metadata["document_number"] == "speed/2026/QH"
    assert response.artifact_versions == {
        "bm25_index": "1.0",
        "dense_index": "1.0",
    }
