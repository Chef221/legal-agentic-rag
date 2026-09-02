"""Unit tests for build_legal_rerank_text with legacy and V2 metadata representations."""

import copy
import pytest

from legal_agentic_rag.reranking.legal_context import build_legal_rerank_text
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalStrategy,
    RetrievalTrace,
)


def _make_hit(metadata: dict, text: str = "Nội dung quy định tại khoản 1 điều 5.") -> RetrievalHit:
    return RetrievalHit(
        chunk_id="doc:1::art:5::cl:1",
        document_id="doc:1",
        rank=1,
        score=0.85,
        strategy=RetrievalStrategy.HYBRID,
        text=text,
        metadata=metadata,
        retrieval_trace=RetrievalTrace(dense_rank=1, dense_score=0.85),
    )


# 1. Legacy metadata output remains unchanged
def test_legacy_metadata_legal_context():
    legacy_meta = {
        "document_title": "Luật Bảo vệ môi trường 2020",
        "document_number": "72/2020/QH14",
        "document_type": "Luật",
        "issuing_authority": "Quốc hội",
        "legal_field": "Môi trường",
        "effect_status": "Còn hiệu lực",
        "effective_date": "01/01/2022",
        "expiry_date": "31/12/2030",
        "structure": {
            "part": "Phần I",
            "chapter": "Chương II",
            "section": "Mục 1",
            "subsection": "Tiểu mục A",
            "article_number": "5",
            "article_title": "Quyền của công dân",
            "clause_numbers": ["1", "2"],
            "point_numbers": ["a", "b"],
        },
    }
    hit = _make_hit(legacy_meta, text="Quy định quyền bảo vệ môi trường.")
    rendered = build_legal_rerank_text(hit)

    assert "Tên văn bản: Luật Bảo vệ môi trường 2020" in rendered
    assert "Số ký hiệu: 72/2020/QH14" in rendered
    assert "Loại văn bản: Luật" in rendered
    assert "Phần: Phần I" in rendered
    assert "Chương: Chương II" in rendered
    assert "Mục: Mục 1" in rendered
    assert "Tiểu mục: Tiểu mục A" in rendered
    assert "Điều: 5" in rendered
    assert "Tên điều: Quyền của công dân" in rendered
    assert "Khoản: 1, 2" in rendered
    assert "Điểm: a, b" in rendered
    assert "Nội dung:\nQuy định quyền bảo vệ môi trường." in rendered


# 2, 3, 4, 5, 6, 7, 8, 9. V2 metadata representation tests
def test_v2_metadata_legal_context_complete():
    v2_meta = {
        "provision_id": "doc:100::art:10::cl:2::pt:a",
        "retrieval_text": "Văn bản: Luật Đất đai\n---\nKhoản 2 Điều 10...",  # MUST NOT BE IN OUTPUT
        "document_identity": {
            "title": "Luật Đất đai 2024",
            "document_number": "31/2024/QH15",
        },
        "hierarchy": {
            "heading_path": [
                {"type": "PART", "label": "I", "title": "QUY ĐỊNH CHUNG"},
                {"type": "CHAPTER", "label": "II", "title": "QUYỀN VÀ TRÁCH NHIỆM"},
                {"type": "SECTION", "label": "1", "title": "QUYỀN CỦA NHÀ NƯỚC"},
            ],
            "article_label": "10",
            "clause_label": "2",
            "point_label": "a",
        },
        "strategy": "WHOLE_PROVISION",
        "quality_flags": ["NO_ARTICLE_MATCH"],
        "segment_index": 1,
        "segment_count": 1,
    }
    original_meta = copy.deepcopy(v2_meta)
    hit = _make_hit(v2_meta, text="Nhà nước thực hiện quyền đại diện chủ sở hữu về đất đai.")

    rendered = build_legal_rerank_text(hit)

    # 2. V2 document title included
    assert "Tên văn bản: Luật Đất đai 2024" in rendered
    # 3. V2 document number included
    assert "Số ký hiệu: 31/2024/QH15" in rendered
    # 4. V2 article/clause/point hierarchy included
    assert "Điều: 10" in rendered
    assert "Khoản: 2" in rendered
    assert "Điểm: a" in rendered
    # 5. V2 heading_path rendered deterministically
    assert "Phần I: QUY ĐỊNH CHUNG" in rendered
    assert "Chương II: QUYỀN VÀ TRÁCH NHIỆM" in rendered
    assert "Mục 1: QUYỀN CỦA NHÀ NƯỚC" in rendered
    # 7. V2 retrieval_text is NOT duplicated
    assert "retrieval_text" not in rendered
    assert "Khoản 2 Điều 10..." not in rendered
    # Excluded noise fields
    assert "NO_ARTICLE_MATCH" not in rendered
    assert "segment_index" not in rendered
    assert "WHOLE_PROVISION" not in rendered
    # 8. Final body contains hit.text
    assert "Nội dung:\nNhà nước thực hiện quyền đại diện chủ sở hữu về đất đai." in rendered
    # 9. Function does not mutate hit or metadata
    assert hit.metadata == original_meta


def test_v2_metadata_cleanly_skips_missing_optional_fields():
    v2_meta_sparse = {
        "document_identity": {
            "title": "Thông tư thử nghiệm",
            "document_number": None,
        },
        "hierarchy": {
            "heading_path": [
                {"type": "APPENDIX", "label": "1", "title": None},
            ],
            "article_label": "1",
            "clause_label": None,
            "point_label": None,
        },
    }
    hit = _make_hit(v2_meta_sparse, text="Nội dung điều 1.")
    rendered = build_legal_rerank_text(hit)

    assert "Tên văn bản: Thông tư thử nghiệm" in rendered
    assert "Số ký hiệu:" not in rendered
    assert "Phụ lục 1" in rendered
    assert "Điều: 1" in rendered
    assert "Khoản:" not in rendered
    assert "Điểm:" not in rendered
    assert "Nội dung:\nNội dung điều 1." in rendered
