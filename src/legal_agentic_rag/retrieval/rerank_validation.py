"""Validation shared by bounded reranking orchestration paths."""

from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


def validate_reranked_response(
    response: RetrievalResponse,
    candidates: list[RetrievalHit],
    query: RetrievalQuery,
) -> None:
    """Reject reranker output that mutates candidates or violates ranking."""
    output_ids = [hit.chunk_id for hit in response.hits]
    candidates_by_id = {hit.chunk_id: hit for hit in candidates}
    candidate_ids = set(candidates_by_id)
    payload_changed = any(
        hit.document_id != candidates_by_id[hit.chunk_id].document_id
        or hit.text != candidates_by_id[hit.chunk_id].text
        or hit.metadata != candidates_by_id[hit.chunk_id].metadata
        for hit in response.hits
        if hit.chunk_id in candidates_by_id
    )
    trace_changed = any(
        hit.retrieval_trace
        != candidates_by_id[hit.chunk_id].retrieval_trace.model_copy(
            update={"reranker_score": hit.score}
        )
        for hit in response.hits
        if hit.chunk_id in candidates_by_id
    )
    if (
        response.strategy != RetrievalStrategy.RERANK
        or response.query != query
        or len(output_ids) > query.top_k
        or len(output_ids) != len(set(output_ids))
        or not set(output_ids).issubset(candidate_ids)
        or payload_changed
        or trace_changed
        or any(hit.strategy != RetrievalStrategy.RERANK for hit in response.hits)
        or [hit.rank for hit in response.hits]
        != list(range(1, len(response.hits) + 1))
    ):
        raise RetrievalError("Reranker returned an incompatible response")
