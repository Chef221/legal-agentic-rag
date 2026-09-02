"""Unit tests for M54 Preprocessing V2 legal structure parser."""

from legal_agentic_rag.offline.preprocessing_v2.parser import (
    parse_document_structure_v2,
    parse_provisions_from_document,
)


def test_appendix_hierarchy_reset():
    text = """CHƯƠNG I. QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Nội dung điều 1.

PHỤ LỤC I. DANH MỤC BIỂU MẪU
Điều 1. Biểu mẫu thống kê
Nội dung biểu mẫu.
"""
    provs, unrec = parse_provisions_from_document("doc:uitdsc2026:test", text)
    articles = [p for p in provs if p.provision_type == "ARTICLE"]
    assert len(articles) == 2

    # First article is under Chapter I
    art1 = articles[0]
    assert any(h.type == "CHAPTER" and "I" in h.label for h in art1.heading_path)
    assert not any(h.type == "APPENDIX" for h in art1.heading_path)

    # Second article is under Appendix I, and Chapter I must be reset/absent
    art2 = articles[1]
    assert any(h.type == "APPENDIX" for h in art2.heading_path)
    assert not any(h.type == "CHAPTER" for h in art2.heading_path)


def test_ordinary_body_fallback_gap_materialization():
    text = """Lời nói đầu văn bản không có cấu trúc điều khoản.
BỘ GIAO THÔNG VẬN TẢI
Căn cứ các quy định pháp luật hiện hành.

Điều 1. Phạm vi áp dụng
1. Khoản 1 áp dụng cho mọi đối tượng.
"""
    provs, _ = parse_provisions_from_document("doc:uitdsc2026:test_gap", text)
    # Leading non-whitespace text before Điều 1 must be materialized as ORDINARY_BODY_FALLBACK gap
    fallbacks = [p for p in provs if p.provision_type == "DOCUMENT_FALLBACK"]
    assert len(fallbacks) == 1
    fb = fallbacks[0]
    assert fb.canonical_path == "doc_fallback:1"
    assert fb.parse_rule == "ORDINARY_BODY_FALLBACK"
    assert "ORDINARY_BODY_FALLBACK" in fb.quality_flags
    assert "Lời nói đầu" in fb.authority_text


def test_no_article_document_fallback():
    text = """CHƯƠNG I. QUY ĐỊNH CHUNG
MỤC 1. NGUYÊN TẮC
Văn bản quy định toàn bộ nhưng không hề có điều nào."""
    provs, _ = parse_provisions_from_document("doc:uitdsc2026:test_no_art", text)
    assert len(provs) == 1
    fb = provs[0]
    assert fb.canonical_path == "doc_fallback"
    assert fb.provision_id == "doc:uitdsc2026:test_no_art::doc_fallback"
    assert fb.parse_rule == "DOCUMENT_FALLBACK"
    assert "DOCUMENT_FALLBACK" in fb.quality_flags


def test_duplicate_canonical_path_parent_linkage():
    text = """Điều 1. Quy định chung
1. Khoản 1 của điều 1 lần 1.
a) Điểm a khoản 1.

Điều 1. Quy định bổ sung
1. Khoản 1 của điều 1 lần 2.
a) Điểm a khoản 1 lần 2.
"""
    provs, _ = parse_provisions_from_document("doc:uitdsc2026:test_dup", text)
    art_ids = [p.provision_id for p in provs if p.provision_type == "ARTICLE"]
    assert art_ids == ["doc:uitdsc2026:test_dup::art:1", "doc:uitdsc2026:test_dup::art:1~2"]

    cl_map = {p.provision_id: p.parent_provision_id for p in provs if p.provision_type == "CLAUSE"}
    assert cl_map["doc:uitdsc2026:test_dup::art:1::cl:1"] == "doc:uitdsc2026:test_dup::art:1"
    assert cl_map["doc:uitdsc2026:test_dup::art:1::cl:1~2"] == "doc:uitdsc2026:test_dup::art:1~2"

    pt_map = {p.provision_id: p.parent_provision_id for p in provs if p.provision_type == "POINT"}
    assert pt_map["doc:uitdsc2026:test_dup::art:1::cl:1::pt:a"] == "doc:uitdsc2026:test_dup::art:1::cl:1"
    assert pt_map["doc:uitdsc2026:test_dup::art:1::cl:1::pt:a~2"] == "doc:uitdsc2026:test_dup::art:1::cl:1~2"


def test_unrecognized_marker_detection_word_boundary():
    # Test that non-standard marker lines like 'Chương XXII và Chương XXXIII...' are captured as unrecognized markers
    text = """Chương XXII và Chương XXXIII của Bộ luật này.
Mục II Phụ lục II Nghị định số 26/2023/NĐ-CP ngày 31 tháng 5 năm 2023 của Chính
Điều 1. Điều hợp lệ
Nội dung điều hợp lệ.
"""
    provs, unrec = parse_provisions_from_document("doc:uitdsc2026:test_unrec", text)
    unrec_texts = [u.line_text for u in unrec]
    assert any("Chương XXII" in t for t in unrec_texts)
    assert any("Mục II" in t for t in unrec_texts)
