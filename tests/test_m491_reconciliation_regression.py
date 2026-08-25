"""Comprehensive regression tests for M49.1-JINA35 reconciled behavior."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from legal_agentic_rag.schemas.answering import AnswerResponse, Citation, EvidenceApplicability
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    QueryAnalysis,
    RetrievalStrategy,
)
from legal_agentic_rag.generation.evidence_selector import EvidenceSelector
from legal_agentic_rag.configuration import ApplicationConfig, EvidenceSelectionConfig
from legal_agentic_rag.runtime.online import OnlineRuntimeFactory


# =============================================================================
# HOTFIX V1: AnswerResponse Raw-Question Identity Preservation
# =============================================================================

def _make_citation(
    evidence_id: str = "E1",
    chunk_id: str = "chunk_1",
    document_id: str = "doc_1",
) -> Citation:
    return Citation(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        document_id=document_id,
    )


def test_hotfix_v1_ordinary_question_unchanged():
    resp = AnswerResponse(
        question="Điều kiện hưởng lương hưu là gì?",
        answer="Theo quy định...",
        citations=[_make_citation()],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="trace_123",
    )
    assert resp.question == "Điều kiện hưởng lương hưu là gì?"
    assert resp.answer == "Theo quy định..."
    assert resp.trace_id == "trace_123"


def test_hotfix_v1_trailing_space_question_preserved_exactly():
    raw_q = "Nghị định 26/2023/NĐ-CP được áp dụng từ ngày nào?  "
    resp = AnswerResponse(
        question=raw_q,
        answer="Theo Điều 1...",
        citations=[_make_citation()],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="trace_456",
    )
    assert resp.question == raw_q
    assert resp.question.endswith("  ")
    assert len(resp.question) == len(raw_q)


def test_hotfix_v1_leading_and_trailing_whitespace_preserved():
    raw_q = " \t Nghị định số 01/2021/NĐ-CP có hiệu lực khi nào? \n "
    resp = AnswerResponse(
        question=raw_q,
        answer="Từ ngày 04/01/2021...",
        citations=[_make_citation()],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="trace_789",
    )
    assert resp.question == raw_q


def test_hotfix_v1_newline_in_question_preserved():
    raw_q = "Câu hỏi pháp luật:\nQuy định về thời hạn nộp thuế?"
    resp = AnswerResponse(
        question=raw_q,
        answer="Thời hạn nộp thuế...",
        citations=[_make_citation()],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="trace_newline",
    )
    assert resp.question == raw_q
    assert "\n" in resp.question


def test_hotfix_v1_whitespace_only_question_rejected():
    with pytest.raises(ValueError, match="question must not be empty"):
        AnswerResponse(
            question="   \t\n  ",
            answer="Answer text",
            citations=[_make_citation()],
            insufficient_evidence=False,
            retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
            trace_id="trace_empty",
        )


def test_hotfix_v1_empty_question_rejected():
    with pytest.raises(ValueError, match="question must not be empty"):
        AnswerResponse(
            question="",
            answer="Answer text",
            citations=[_make_citation()],
            insufficient_evidence=False,
            retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
            trace_id="trace_empty2",
        )


def test_hotfix_v1_answer_and_trace_id_normalization_remains_intact():
    resp = AnswerResponse(
        question="  Câu hỏi có khoảng trắng?  ",
        answer="  Câu trả lời có khoảng trắng đầu cuối.  ",
        citations=[_make_citation()],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="  trace_padded  ",
    )
    # Question preserves raw string
    assert resp.question == "  Câu hỏi có khoảng trắng?  "
    # Answer and trace_id are stripped by _non_empty validation
    assert resp.answer == "Câu trả lời có khoảng trắng đầu cuối."
    assert resp.trace_id == "trace_padded"


# =============================================================================
# HOTFIX V2: Conservative Dual-Source Anchored Explicit Document Fallback
# =============================================================================

@pytest.fixture
def evidence_selector():
    return EvidenceSelector(EvidenceSelectionConfig(enabled=True))


def _make_hit(
    chunk_id: str = "chunk_1",
    document_id: str = "210540",
    document_number: str | None = None,
    document_title: str | None = None,
    source_url: str | None = None,
    rank: int = 1,
    score: float = 0.9,
    strategy: str = "hybrid_rerank",
) -> RetrievalHit:
    metadata = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "score": score,
    }
    if document_number is not None:
        metadata["document_number"] = document_number
    if document_title is not None:
        metadata["document_title"] = document_title
    if source_url is not None:
        metadata["source_url"] = source_url

    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        rank=rank,
        score=score,
        strategy=strategy,
        text="Nội dung điều luật...",
        metadata=metadata,
    )


def test_hotfix_v2_canonical_nd_identity_and_dual_source_fallback_match(evidence_selector):
    """QID 17789 case: 26/2023/NĐ-CP without document_number matches when title and URL anchor to same slug."""
    hit = _make_hit(
        document_number=None,
        document_title="Nghi-dinh-26-2023-ND-CP-Bieu-thue-xuat-khau-Bieu-thue-nhap-khau-uu-dai-548616",
        source_url="https://thuvienphapluat.vn/van-ban/Xuat-nhap-khau/Nghi-dinh-26-2023-ND-CP-Bieu-thue-xuat-khau-Bieu-thue-nhap-khau-uu-dai-548616.aspx",
    )
    query = RetrievalQuery(
        query_id="q1",
        original_question="Nghị định 26/2023/NĐ-CP được áp dụng từ ngày nào?",
        normalized_question="Nghị định 26/2023/NĐ-CP được áp dụng từ ngày nào?",
        query_analysis=QueryAnalysis(
            document_numbers=["26/2023/NĐ-CP"],
            article_numbers=[],
        ),
    )
    scored = evidence_selector.score(query=query, hits=[hit])
    assert len(scored) == 1
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH
    assert "document_number" not in scored[0].hit.metadata


def test_hotfix_v2_correct_title_wrong_url_fails_closed(evidence_selector):
    """If title matches but URL does not anchor to same identity, no match."""
    hit = _make_hit(
        document_number=None,
        document_title="Nghi-dinh-26-2023-ND-CP-Bieu-thue-xuat-khau",
        source_url="https://thuvienphapluat.vn/van-ban/Xuat-nhap-khau/Nghi-dinh-15-2022-ND-CP.aspx",
    )
    assert not EvidenceSelector._fallback_document_reference_match(
        hit, {"26/2023/nd-cp"}
    )


def test_hotfix_v2_wrong_title_correct_url_fails_closed(evidence_selector):
    """If URL matches but title does not anchor to same identity, no match."""
    hit = _make_hit(
        document_number=None,
        document_title="Nghi-dinh-15-2022-ND-CP-Chinh-sach-mien-giam-thue",
        source_url="https://thuvienphapluat.vn/van-ban/Xuat-nhap-khau/Nghi-dinh-26-2023-ND-CP.aspx",
    )
    assert not EvidenceSelector._fallback_document_reference_match(
        hit, {"26/2023/nd-cp"}
    )


def test_hotfix_v2_missing_title_fails_closed(evidence_selector):
    hit = _make_hit(
        document_number=None,
        document_title=None,
        source_url="https://thuvienphapluat.vn/van-ban/Nghi-dinh-26-2023-ND-CP.aspx",
    )
    assert not EvidenceSelector._fallback_document_reference_match(
        hit, {"26/2023/nd-cp"}
    )


def test_hotfix_v2_missing_url_fails_closed(evidence_selector):
    hit = _make_hit(
        document_number=None,
        document_title="Nghi-dinh-26-2023-ND-CP-Bieu-thue-xuat-khau",
        source_url=None,
    )
    assert not EvidenceSelector._fallback_document_reference_match(
        hit, {"26/2023/nd-cp"}
    )


def test_hotfix_v2_unsupported_document_family_fails_closed(evidence_selector):
    """Unsupported document code prefix returns None and fails closed."""
    prefix = EvidenceSelector._document_reference_identity_prefix("123/2023/UNKNOWN-XYZ")
    assert prefix is None

    hit = _make_hit(
        document_number=None,
        document_title="Unknown-123-2023-UNKNOWN-XYZ",
        source_url="https://thuvienphapluat.vn/van-ban/Unknown-123-2023-UNKNOWN-XYZ.aspx",
    )
    assert not EvidenceSelector._fallback_document_reference_match(
        hit, {"123/2023/unknown-xyz"}
    )


def test_hotfix_v2_identity_mentioned_only_later_in_title_fails_closed(evidence_selector):
    """A document that merely amends or cites 26/2023/ND-CP later in its title must NOT match."""
    hit = _make_hit(
        document_number=None,
        document_title="Thong-tu-01-2024-TT-BTC-huong-dan-Nghi-dinh-26-2023-ND-CP",
        source_url="https://thuvienphapluat.vn/van-ban/Thong-tu-01-2024-TT-BTC.aspx",
    )
    assert not EvidenceSelector._fallback_document_reference_match(
        hit, {"26/2023/nd-cp"}
    )


def test_hotfix_v2_explicit_metadata_document_number_remains_authoritative(evidence_selector):
    """When document_number exists in metadata, normal matching path is used directly."""
    hit = _make_hit(
        document_number="26/2023/NĐ-CP",
        document_title="Custom title",
        source_url="https://custom.url/doc",
    )
    query = RetrievalQuery(
        query_id="q2",
        original_question="Nghị định 26/2023/NĐ-CP",
        normalized_question="Nghị định 26/2023/NĐ-CP",
        query_analysis=QueryAnalysis(
            document_numbers=["26/2023/NĐ-CP"],
            article_numbers=[],
        ),
    )
    scored = evidence_selector.score(query=query, hits=[hit])
    assert len(scored) == 1
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH
    assert scored[0].hit.metadata["document_number"] == "26/2023/NĐ-CP"


# =============================================================================
# RUNTIME & SPAWN SAFETY
# =============================================================================

def test_online_runtime_factory_construction_contract():
    """Verify OnlineRuntimeFactory instantiates cleanly from valid ApplicationConfig."""
    cfg_path = Path("configs/uit-dsc-2026-task2-m491-jina35.example.json")
    app_cfg = ApplicationConfig.model_validate(json.loads(cfg_path.read_text(encoding="utf-8")))

    mock_ep = MagicMock()
    mock_reranker = MagicMock()
    mock_grader = MagicMock()
    mock_gen = MagicMock()
    mock_ver = MagicMock()

    factory = OnlineRuntimeFactory(
        app_cfg,
        embedding_provider=mock_ep,
        reranker=mock_reranker,
        context_grader=mock_grader,
        answer_generator=mock_gen,
        citation_verifier=mock_ver,
    )
    assert factory._config == app_cfg
    assert factory._embedding_provider == mock_ep
    assert factory._reranker == mock_reranker
    assert factory._context_grader == mock_grader
    assert factory._answer_generator == mock_gen
    assert factory._citation_verifier == mock_ver


def test_dual_session_runner_uses_valid_factory_contract():
    """Verify dual_session_runner does not call nonexistent OnlineRuntimeFactory.from_config."""
    from legal_agentic_rag.competition.uit_dsc_2026 import dual_session_runner
    import inspect

    src = inspect.getsource(dual_session_runner)
    assert "OnlineRuntimeFactory.from_config" not in src
    assert "OnlineRuntimeFactory(app_cfg)" in src
