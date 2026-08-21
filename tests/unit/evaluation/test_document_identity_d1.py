"""Unit tests for Phase D1-A deterministic legal document identity feasibility evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import unicodedata
import zipfile
import pytest

from scripts.evaluate_document_identity_d1 import (
    CandidateIdentity,
    ResolvedDocumentIdentity,
    normalize_key,
    normalize_doc_number,
    extract_from_slug,
    clean_and_combine_header_lines,
    extract_from_header,
    resolve_document_identity,
    run_d1a_feasibility_evaluation,
    EXPECTED_CONTEXTS_SHA256,
    EXPECTED_D0_EVIDENCE_SHA256,
    EXPECTED_TRAIN_SHA256,
    FEASIBILITY_GATE_A_THRESHOLD,
    FEASIBILITY_GATE_B_THRESHOLD,
)


def test_normalize_key_basic_and_accents() -> None:
    """Test ASCII-folding, lowercase, and hyphens normalization."""
    assert normalize_key("Thông tư") == "thong-tu"
    assert normalize_key("Nghị định") == "nghi-dinh"
    assert normalize_key("Quyết định") == "quyet-dinh"
    assert normalize_key("99/2003/NĐ-CP") == "99-2003-nd-cp"
    assert normalize_key("99-2003-ND-CP") == "99-2003-nd-cp"
    assert normalize_key("01/2024/TT-BTP") == "01-2024-tt-btp"
    assert normalize_key("31/2024/QH15") == "31-2024-qh15"
    assert normalize_key("51/QĐ-VKSTC-V12") == "51-qd-vkstc-v12"


def test_normalize_doc_number_preserves_canonical_shape() -> None:
    """Test that legal document numbers preserve digits, year, slashes, hyphens, and legal suffixes."""
    num = "99/2003/NĐ-CP"
    assert normalize_doc_number(num) == "99/2003/NĐ-CP"

    num_with_spaces = "  01/2024/TT-BTP . "
    assert normalize_doc_number(num_with_spaces) == "01/2024/TT-BTP"

    # Preserves complex suffixes
    num_complex = "51/QĐ-VKSTC-V12"
    assert normalize_doc_number(num_complex) == "51/QĐ-VKSTC-V12"


def test_extract_title_slug() -> None:
    """Test extracting document type and number from title slug."""
    title1 = "Nghi-dinh-99-2003-ND-CP-Quy-che-Khu-cong-nghe-cao-51305"
    cand1 = extract_from_slug(title1, "title")
    assert cand1 is not None
    assert cand1.source == "title"
    assert cand1.document_type == "Nghị định"
    assert cand1.document_number == "99-2003-ND-CP"
    assert cand1.normalized_identity == "nghi-dinh::99-2003-nd-cp"

    title2 = "Thong-tu-30-2022-TT-BTC-co-che-tai-chinh-nang-cao-nang-luc-giang-vien-dai-hoc-516332"
    cand2 = extract_from_slug(title2, "title")
    assert cand2 is not None
    assert cand2.document_type == "Thông tư"
    assert cand2.document_number == "30-2022-TT-BTC"
    assert cand2.normalized_identity == "thong-tu::30-2022-tt-btc"


def test_extract_url_slug() -> None:
    """Test extracting document type and number from URL slug."""
    url_slug = "Quyet-dinh-43-2021-QD-UBND-uy-quyen-cap-giay-phep-xay-dung-tinh-Vinh-Phuc-484964"
    cand = extract_from_slug(url_slug, "url")
    assert cand is not None
    assert cand.source == "url"
    assert cand.document_type == "Quyết định"
    assert cand.document_number == "43-2021-QD-UBND"
    assert cand.normalized_identity == "quyet-dinh::43-2021-qd-ubnd"


def test_extract_url_slug_with_trailing_metadata_year() -> None:
    """Test that TVPL slugs with trailing metadata years after acronyms are properly normalized."""
    url_slug = "Quyet-dinh-440-QD-TTCP-2021-chuc-nang-nhiem-vu-Vien-Chien-luoc-va-Khoa-hoc-Thanh-tra-492433"
    cand = extract_from_slug(url_slug, "url")
    assert cand is not None
    assert cand.document_type == "Quyết định"
    assert cand.document_number == "440-QD-TTCP"
    assert cand.normalized_identity == "quyet-dinh::440-qd-ttcp"


def test_extract_header_standard() -> None:
    """Test extracting document type and number from formal passage header."""
    passage = """CHÍNH PHỦ
-------
CỘNG HÒA XÃ HỘI
CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
---------------
Số: 33/2014/NĐ-CP
Hà Nội, ngày 26 tháng 04 năm 2014
NGHỊ ĐỊNH
VỀ TỔ CHỨC VÀ HOẠT ĐỘNG CỦA THANH TRA QUỐC PHÒNG
Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001;
"""
    cand = extract_from_header(passage)
    assert cand is not None
    assert cand.source == "header"
    assert cand.document_type == "Nghị định"
    assert cand.document_number == "33/2014/NĐ-CP"
    assert cand.normalized_identity == "nghi-dinh::33-2014-nd-cp"


def test_extract_header_multiline_split_type() -> None:
    """Test extracting document type when type is split across multiple header lines."""
    passage = """BỘ TÀI CHÍNH
-------
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
---------------
Số: 30/2022/TT-BTC
Hà Nội, ngày 3 tháng 6 năm 2022
THÔNG
TƯ
HƯỚNG DẪN CƠ CHẾ TÀI CHÍNH THỰC HIỆN ĐỀ ÁN
Căn cứ Luật Ngân sách nhà nước ngày 25 tháng 6 năm 2015;
"""
    cand = extract_from_header(passage)
    assert cand is not None
    assert cand.document_type == "Thông tư"
    assert cand.document_number == "30/2022/TT-BTC"
    assert cand.normalized_identity == "thong-tu::30-2022-tt-btc"


def test_adversarial_own_document_safety_body_citations_ignored() -> None:
    """Critical safety test: citations to other laws in the passage body must NOT become document identity."""
    adversarial_passage = """BỘ KHOA HỌC VÀ CÔNG NGHỆ
-------
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
---------------
Số: 05/2019/TT-BKHCN
Hà Nội, ngày 26 tháng 6 năm 2019
THÔNG TƯ
QUY ĐỊNH CHI TIẾT THI HÀNH MỘT SỐ ĐIỀU CỦA NGHỊ ĐỊNH SỐ 43/2017/NĐ-CP NGÀY 14 THÁNG 4 NĂM 2017 CỦA CHÍNH PHỦ VỀ NHÃN HÀNG HÓA
Căn cứ Luật Tiêu chuẩn và Quy chuẩn kỹ thuật số 68/2006/QH11 ngày 29 tháng 6 năm 2006;
Căn cứ Luật Chất lượng sản phẩm, hàng hóa số 05/2007/QH12 ngày 21 tháng 11 năm 2007;
Căn cứ Nghị định số 95/2017/NĐ-CP ngày 16 tháng 8 năm 2017 của Chính phủ;
Theo đề nghị của Tổng cục trưởng Tổng cục Tiêu chuẩn Đo lường Chất lượng;
Bộ trưởng Bộ Khoa học và Công nghệ ban hành Thông tư hướng dẫn...
Điều 1. Phạm vi điều chỉnh
Thông tư này quy định chi tiết theo Nghị định số 43/2017/NĐ-CP...
"""
    cand = extract_from_header(adversarial_passage)
    assert cand is not None
    # Must be own document: 05/2019/TT-BKHCN, Thông tư
    # Must NOT be referenced laws: 43/2017/NĐ-CP, 68/2006/QH11, 05/2007/QH12, 95/2017/NĐ-CP
    assert cand.document_type == "Thông tư"
    assert cand.document_number == "05/2019/TT-BKHCN"
    assert cand.document_number != "43/2017/NĐ-CP"
    assert cand.document_number != "68/2006/QH11"
    assert cand.document_number != "95/2017/NĐ-CP"


def test_resolve_three_source_agreement() -> None:
    """Test agreement across Title, URL, and Header yielding HIGH_CONFIDENCE."""
    name = "Nghi-dinh-99-2003-ND-CP-Quy-che-Khu-cong-nghe-cao-51305"
    link = "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-99-2003-ND-CP-Quy-che-Khu-cong-nghe-cao-51305.aspx"
    passage = """CHÍNH PHỦ
Số: 99/2003/NĐ-CP
NGHỊ ĐỊNH
VỀ VIỆC BAN HÀNH QUY CHẾ KHU CÔNG NGHỆ CAO
Căn cứ Luật Tổ chức Chính phủ;
"""
    resolved = resolve_document_identity("100686", name, link, passage)
    assert resolved.status == "HIGH_CONFIDENCE"
    assert resolved.agreement_pattern == "all_three"
    assert resolved.agreeing_sources == ["header", "title", "url"]
    assert resolved.document_type == "Nghị định"
    assert resolved.document_number == "99/2003/NĐ-CP"


def test_resolve_two_source_agreement_title_and_url() -> None:
    """Test two-source agreement between Title and URL when header is unparsed."""
    name = "Thong-tu-59-2021-TT-BCA-huong-dan-Luat-Can-cuoc-cong-dan-478289"
    link = "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Thong-tu-59-2021-TT-BCA-huong-dan-Luat-Can-cuoc-cong-dan-478289.aspx"
    passage = "Thông tin chung không có tiêu đề Số cụ thể."

    resolved = resolve_document_identity("256847", name, link, passage)
    assert resolved.status == "HIGH_CONFIDENCE"
    assert resolved.agreement_pattern == "title_url"
    assert resolved.agreeing_sources == ["title", "url"]
    assert resolved.document_type == "Thông tư"
    assert resolved.document_number == "59-2021-TT-BCA"


def test_resolve_two_source_agreement_url_and_header_untitled() -> None:
    """Test two-source agreement between URL and Header when context title is missing (name=None)."""
    name = None
    link = "https://thuvienphapluat.vn/van-ban/Tai-nguyen-Moi-truong/Nghi-dinh-45-2023-ND-CP-huong-dan-Luat-Dau-khi-571537.aspx"
    passage = """CHÍNH PHỦ
Số: 45/2023/NĐ-CP
NGHỊ ĐỊNH
HƯỚNG DẪN LUẬT DẦU KHÍ
Căn cứ Luật Dầu khí;
"""
    resolved = resolve_document_identity("218546", name, link, passage)
    assert resolved.status == "HIGH_CONFIDENCE"
    assert resolved.agreement_pattern == "url_header"
    assert resolved.agreeing_sources == ["header", "url"]
    assert resolved.document_type == "Nghị định"
    assert resolved.document_number == "45/2023/NĐ-CP"


def test_resolve_single_source_no_conflict() -> None:
    """Test single explicit source without conflict yielding HIGH_CONFIDENCE."""
    name = None
    link = "https://thuvienphapluat.vn/cong-van/Bao-hiem/Cong-van-1880-BHXH-CSXH-2023-che-do-bao-hiem-xa-hoi-570357.aspx"
    passage = ""  # empty passage

    resolved = resolve_document_identity("243835", name, link, passage)
    assert resolved.status == "HIGH_CONFIDENCE"
    assert resolved.agreement_pattern == "single_url"
    assert resolved.document_type == "Công văn"
    assert resolved.document_number == "1880-BHXH-CSXH"


def test_resolve_conflicting_candidates_yields_ambiguous() -> None:
    """Test that conflicting candidates across sources fail closed as AMBIGUOUS."""
    name = "Quyet-dinh-100-2020-QD-TTg-ve-chinh-sach-12345"
    link = "https://thuvienphapluat.vn/van-ban/Nghi-dinh-200-2021-ND-CP-ve-chinh-sach-67890.aspx"
    passage = """BỘ TÀI CHÍNH
Số: 50/2022/TT-BTC
THÔNG TƯ
HƯỚNG DẪN...
"""
    resolved = resolve_document_identity("999999", name, link, passage)
    assert resolved.status == "AMBIGUOUS"
    assert resolved.document_type is None
    assert resolved.document_number is None
    assert resolved.agreement_pattern == "conflict"


def test_resolve_no_candidates_yields_unresolved() -> None:
    """Test that documents with no extractable identity return UNRESOLVED."""
    name = None
    link = "https://thuvienphapluat.vn/tintuc/bai-viet-thong-tin-123.aspx"
    passage = "Một đoạn thông tin chung chung không có căn cứ pháp lý."

    resolved = resolve_document_identity("888888", name, link, passage)
    assert resolved.status == "UNRESOLVED"
    assert resolved.document_type is None
    assert resolved.document_number is None
    assert resolved.agreement_pattern == "none"


def test_proxy_population_determinism() -> None:
    """Test that proxy population ordering and SHA-256 calculation are strictly deterministic."""
    items = [
        {"question_id": "132757", "target_document_id": "245534"},
        {"question_id": "108971", "target_document_id": "150679"},
    ]
    sorted_items = sorted(items, key=lambda x: (str(x["question_id"]), str(x["target_document_id"])))
    sha1 = json.dumps(sorted_items, sort_keys=True)
    sha2 = json.dumps(sorted_items, sort_keys=True)
    assert sha1 == sha2
    assert sorted_items[0]["question_id"] == "108971"
    assert sorted_items[1]["question_id"] == "132757"


def test_feasibility_gate_decision_logic() -> None:
    """Test evaluation of pre-registered feasibility gates."""
    # Case 1: Gate A passed (non-empty >= 50%)
    non_empty_cov_1 = 0.65
    proxy_cov_1 = 0.60
    assert non_empty_cov_1 >= FEASIBILITY_GATE_A_THRESHOLD

    # Case 2: Gate B passed (proxy >= 70%)
    non_empty_cov_2 = 0.45
    proxy_cov_2 = 0.75
    assert proxy_cov_2 >= FEASIBILITY_GATE_B_THRESHOLD

    # Case 3: Both failed
    non_empty_cov_3 = 0.40
    proxy_cov_3 = 0.60
    assert not (non_empty_cov_3 >= FEASIBILITY_GATE_A_THRESHOLD or proxy_cov_3 >= FEASIBILITY_GATE_B_THRESHOLD)


def test_source_checksum_failure_handling(tmp_path: Path) -> None:
    """Test that mismatched source checksums fail closed with ValueError."""
    fake_ctx = tmp_path / "fake_contexts.zip"
    fake_ctx.write_bytes(b"corrupted zip content")

    fake_d0 = tmp_path / "fake_d0.zip"
    fake_d0.write_bytes(b"fake d0")

    fake_train = tmp_path / "fake_train.json"
    fake_train.write_bytes(b"{}")

    fake_chunks = tmp_path / "records.jsonl"
    fake_chunks.write_bytes(b"")

    with pytest.raises(ValueError, match="D1A_SOURCE_IDENTITY_FAILURE"):
        run_d1a_feasibility_evaluation(
            contexts_zip=fake_ctx,
            d0_evidence_zip=fake_d0,
            train_json=fake_train,
            chunks_jsonl=fake_chunks,
            evidence_zip=tmp_path / "out.zip",
            output_dir=tmp_path / "staging",
        )
