"""Tests for bounded retrieval-hit to evidence conversion."""

import pytest

from legal_agentic_rag.configuration import GenerationConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.generation import ContextBuilder
from legal_agentic_rag.schemas import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


def _hit(
    chunk_id: str,
    rank: int,
    *,
    token_count: int,
    text: str = "Nội dung căn cứ.",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text=text,
        metadata={
            "token_count": token_count,
            "document_title": f"Văn bản {chunk_id}",
            "document_number": f"{rank}/2026/QH",
            "effect_status": None,
            "source_url": f"https://example.test/{chunk_id}",
            "structure": {
                "article_number": str(rank),
                "article_title": f"Điều {rank}",
            },
        },
    )


def _response(hits: list[RetrievalHit]) -> RetrievalResponse:
    query = RetrievalQuery(
        query_id="context-query",
        original_question="Câu hỏi",
        normalized_question="câu hỏi",
        top_k=3,
        candidate_k=3,
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )
    return RetrievalResponse(
        query=query,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        hits=hits,
    )


def test_context_builder_preserves_metadata_trace_and_complete_chunk_budget() -> None:
    """Ranked chunks become evidence without truncating over-budget legal text."""
    builder = ContextBuilder(
        GenerationConfig(max_context_tokens=5, max_evidence=3)
    )

    result = builder.build(
        _response(
            [
                _hit("first", 1, token_count=3),
                _hit("second", 2, token_count=4),
            ]
        )
    )

    assert [item.evidence_id for item in result.evidence] == ["E1"]
    evidence = result.evidence[0]
    assert evidence.article_number == "1"
    assert evidence.document_number == "1/2026/QH"
    assert evidence.metadata["retrieval_rank"] == 1
    assert evidence.metadata["retrieval_strategy"] == "hybrid_rerank"
    assert result.estimated_token_count == 3
    assert result.omitted_hit_count == 1
    assert result.truncated is True
    assert "context_budget_exhausted" in result.warnings
    assert "effect_status_unknown:E1" in result.warnings


def test_context_builder_deduplicates_exact_hits_and_rejects_changed_payload() -> None:
    """Duplicate chunk IDs cannot silently hide conflicting legal content."""
    hit = _hit("same", 1, token_count=3)
    result = ContextBuilder().build(_response([hit, hit]))

    assert result.selected_count == 1
    assert result.duplicate_hit_count == 1
    assert "duplicate_retrieval_hits_removed:1" in result.warnings

    changed = hit.model_copy(update={"text": "Nội dung khác."})
    with pytest.raises(DataValidationError, match="inconsistent"):
        ContextBuilder().build(_response([hit, changed]))


def test_context_builder_falls_back_to_deterministic_token_estimate() -> None:
    """Backends missing token_count still receive a bounded deterministic estimate."""
    hit = _hit(
        "fallback",
        1,
        token_count=3,
        text="Không áp dụng 10%.",
    )
    hit = hit.model_copy(update={"metadata": {}})

    result = ContextBuilder(
        GenerationConfig(max_context_tokens=10)
    ).build(_response([hit]))

    assert result.selected_count == 1
    assert 1 <= result.estimated_token_count <= 10


def test_context_builder_deprioritizes_only_explicit_inactive_statuses() -> None:
    """Effect-status ranking changes only when an inactive label is configured."""
    inactive = _hit("expired", 1, token_count=3)
    inactive_metadata = dict(inactive.metadata)
    inactive_metadata["effect_status"] = "Hết hiệu lực"
    inactive = inactive.model_copy(update={"metadata": inactive_metadata})
    current = _hit("current", 2, token_count=3)
    current_metadata = dict(current.metadata)
    current_metadata["effect_status"] = "Còn hiệu lực"
    current = current.model_copy(update={"metadata": current_metadata})
    builder = ContextBuilder(
        GenerationConfig(
            max_context_tokens=3,
            max_evidence=1,
            inactive_effect_statuses=frozenset({"hết hiệu lực"}),
        )
    )

    result = builder.build(_response([inactive, current]))

    assert result.evidence[0].chunk_id == "current"
