"""Tests for bounded retrieval-hit to evidence conversion with legacy and V2 representations."""

import copy
import pytest

from legal_agentic_rag.configuration import EvidenceSelectionConfig, GenerationConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.generation import ContextBuilder
from legal_agentic_rag.schemas import (
    QueryAnalysis,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
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


def _v2_hit(
    unit_id: str,
    doc_id: str,
    rank: int,
    *,
    doc_number: str | None = "45/2019/QH14",
    doc_title: str | None = "Bộ luật Lao động 2019",
    article_label: str | None = "113",
    clause_label: str | None = "1",
    point_label: str | None = None,
    heading_path: list[dict] | None = None,
    authority_text: str = "Người lao động được nghỉ hằng năm.",
    retrieval_text: str = "Văn bản: Bộ luật Lao động\n---\nKhoản 1 Điều 113...",
    token_count: int = 4,
    reranker_score: float = 8.5,
) -> RetrievalHit:
    headings = heading_path if heading_path is not None else [
        {"type": "CHAPTER", "label": "VII", "title": "THỜI GIỜ LÀM VIỆC, NGHỈ NGƠI"}
    ]
    trace = RetrievalTrace(
        bm25_rank=rank,
        dense_rank=rank,
        bm25_score=-1.2,
        dense_score=0.88,
        bm25_rrf_contribution=0.016,
        dense_rrf_contribution=0.016,
        rrf_score=0.032,
        reranker_score=reranker_score,
    )
    return RetrievalHit(
        chunk_id=unit_id,
        document_id=doc_id,
        rank=rank,
        score=reranker_score,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text=authority_text,
        metadata={
            "token_count": token_count,
            "provision_id": f"{doc_id}::art:{article_label}",
            "retrieval_text": retrieval_text,
            "document_identity": {
                "document_number": doc_number,
                "title": doc_title,
            },
            "hierarchy": {
                "article_label": article_label,
                "clause_label": clause_label,
                "point_label": point_label,
                "heading_path": headings,
            },
            "strategy": "WHOLE_PROVISION",
            "quality_flags": [],
            "segment_index": 1,
            "segment_count": 1,
        },
        retrieval_trace=trace,
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


def test_context_builder_records_applicability_and_budget_decisions() -> None:
    """Selected and omitted hits retain a typed evidence-selection trace."""
    matching = _hit("matching", 2, token_count=3)
    matching_metadata = dict(matching.metadata)
    matching_metadata["document_number"] = "45/2019/QH14"
    matching_metadata["structure"] = {"article_number": "113"}
    matching = matching.model_copy(update={"metadata": matching_metadata})
    wrong = _hit("wrong", 1, token_count=3)
    response = _response([wrong, matching])
    query = response.query.model_copy(
        update={
            "query_analysis": QueryAnalysis(
                document_numbers=["45/2019/QH14"],
                article_numbers=["113"],
            )
        }
    )
    response = response.model_copy(update={"query": query})

    result = ContextBuilder(
        GenerationConfig(max_evidence=1, max_context_tokens=10)
    ).build(response)

    assert [item.chunk_id for item in result.evidence] == ["matching"]
    assert result.evidence[0].metadata["evidence_selection"][
        "applicability"
    ] == "explicit_match"
    assert [item.chunk_id for item in result.selection_trace] == [
        "matching",
        "wrong",
    ]
    assert result.selection_trace[0].selected is True
    assert result.selection_trace[1].reason == "max_evidence"
    assert "max_evidence_omissions:1" in result.warnings
    assert "context_budget_exhausted" not in result.warnings


def test_context_builder_enforces_document_and_article_diversity() -> None:
    """Repeated chunks cannot consume all context slots in the M45 policy."""
    first = _hit("first", 1, token_count=2)
    repeated = _hit("repeated", 2, token_count=2).model_copy(
        update={"document_id": first.document_id, "metadata": first.metadata}
    )
    other = _hit("other", 3, token_count=2)
    builder = ContextBuilder(
        GenerationConfig(max_context_tokens=10, max_evidence=3),
        EvidenceSelectionConfig(max_per_document=1, max_per_article=1),
    )

    result = builder.build(_response([first, repeated, other]))

    assert [item.chunk_id for item in result.evidence] == ["first", "other"]
    assert result.selection_trace[1].reason == "diversity_limit"
    assert "diversity_limit_omissions:1" in result.warnings


# ==============================================================================
# V2 CONTEXT BUILDER TESTS
# ==============================================================================

def test_v2_evidence_conversion_fields_and_trace() -> None:
    """V2 retrieval units are converted to Evidence with authority text and metadata preservation."""
    v2_hit = _v2_hit(
        "unit_100",
        "doc_labour",
        1,
        doc_number="45/2019/QH14",
        doc_title="Bộ luật Lao động 2019",
        article_label="113",
        authority_text="Người lao động làm việc đủ 12 tháng được nghỉ hằng năm.",
        retrieval_text="Văn bản: Bộ luật Lao động\n---\nKhoản 1 Điều 113...",
        token_count=12,
        reranker_score=9.2,
    )
    original_meta = copy.deepcopy(v2_hit.metadata)

    builder = ContextBuilder(GenerationConfig(max_context_tokens=100, max_evidence=5))
    result = builder.build(_response([v2_hit]))

    assert len(result.evidence) == 1
    ev = result.evidence[0]

    # 1. Identity & text
    assert ev.chunk_id == "unit_100"
    assert ev.document_id == "doc_labour"
    assert ev.text == "Người lao động làm việc đủ 12 tháng được nghỉ hằng năm."
    # 5. retrieval_text is NOT Evidence.text
    assert "retrieval_text" not in ev.text
    assert "Khoản 1 Điều 113..." not in ev.text
    # 2. Document title populated
    assert ev.document_title == "Bộ luật Lao động 2019"
    # 3. Document number populated
    assert ev.document_number == "45/2019/QH14"
    # 4. Article label populated as article_number
    assert ev.article_number == "113"
    # 6. Unavailable fields remain None
    assert ev.document_type is None
    assert ev.effective_date is None
    assert ev.expiry_date is None
    assert ev.effect_status is None
    assert ev.source_url is None
    # 7. chunk_metadata preserves V2 metadata
    assert ev.metadata["chunk_metadata"]["provision_id"] == "doc_labour::art:113"
    # 8. Trace preserved
    assert ev.metadata["retrieval_trace"]["reranker_score"] == 9.2
    assert ev.metadata["retrieval_trace"]["bm25_rank"] == 1
    assert ev.metadata["retrieval_trace"]["dense_rank"] == 1
    # 14. Input hit metadata not mutated
    assert v2_hit.metadata == original_meta


def test_v2_diversity_limits_per_document_and_article() -> None:
    """V2 document and article diversity limits correctly filter excessive chunks."""
    h1 = _v2_hit("u1", "doc_A", 1, doc_number="1/2024/QH", article_label="5")
    # Same doc, same article -> filtered by max_per_article
    h2 = _v2_hit("u2", "doc_A", 2, doc_number="1/2024/QH", article_label="5")
    # Same doc, different article -> filtered by max_per_document if max_per_document=1
    h3 = _v2_hit("u3", "doc_A", 3, doc_number="1/2024/QH", article_label="6")
    # Different doc -> accepted
    h4 = _v2_hit("u4", "doc_B", 4, doc_number="2/2024/QH", article_label="5")

    builder = ContextBuilder(
        GenerationConfig(max_context_tokens=100, max_evidence=5),
        EvidenceSelectionConfig(max_per_document=2, max_per_article=1),
    )

    result = builder.build(_response([h1, h2, h3, h4]))

    assert [ev.chunk_id for ev in result.evidence] == ["u1", "u3", "u4"]
    assert result.selection_trace[1].reason == "diversity_limit"


def test_v2_duplicate_hit_handling_fails_closed_on_payload_mismatch() -> None:
    """V2 duplicate hits with identical payload are deduplicated; conflicting payload fails closed."""
    h1 = _v2_hit("u_dup", "doc_A", 1)
    result = ContextBuilder().build(_response([h1, h1]))
    assert result.selected_count == 1
    assert result.duplicate_hit_count == 1

    h_conflict = h1.model_copy(update={"text": "Nội dung mâu thuẫn."})
    with pytest.raises(DataValidationError, match="inconsistent"):
        ContextBuilder().build(_response([h1, h_conflict]))
