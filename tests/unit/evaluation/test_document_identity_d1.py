"""Unit tests for Phase D1-A deterministic document identity extraction and strict multi-channel feasibility evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import zipfile

from scripts.evaluate_document_identity_d1 import (
    CandidateIdentity,
    clean_and_combine_header_lines,
    extract_from_header,
    extract_from_slug,
    normalize_doc_number,
    normalize_key,
    resolve_document_identity,
    run_d1a_feasibility_evaluation,
)


def test_normalize_key_deterministic() -> None:
    """Test deterministic key normalization with diacritics removal and punctuation folding."""
    assert normalize_key("Nghị định") == "nghi-dinh"
    assert normalize_key("Thông tư liên tịch") == "thong-tu-lien-tich"
    assert normalize_key("99/2003/NĐ-CP") == "99-2003-nd-cp"
    assert normalize_key("440/QĐ-TTCP") == "440-qd-ttcp"
    assert normalize_key("123:QĐ_UBND") == "123-qd-ubnd"


def test_normalize_doc_number() -> None:
    """Test conservative document number whitespace and trailing punctuation cleanup."""
    assert normalize_doc_number(" 99/2003/NĐ-CP. ") == "99/2003/NĐ-CP"
    assert normalize_doc_number("440/QĐ-TTCP;") == "440/QĐ-TTCP"
    assert normalize_doc_number("01/2020/TT-BKHCN") == "01/2020/TT-BKHCN"


def test_extract_from_slug_standard_decree() -> None:
    """Test standard decree slug extraction."""
    cand = extract_from_slug("Nghi-dinh-99-2003-ND-CP-quy-che-khu-cong-nghe-cao", "title")
    assert cand is not None
    assert cand.source == "title"
    assert cand.document_type == "Nghị định"
    assert cand.document_number == "99-2003-ND-CP"
    assert cand.normalized_identity == "nghi-dinh::99-2003-nd-cp"


def test_extract_from_slug_trailing_year_normalization() -> None:
    """Test TVPL slug trailing metadata year normalization."""
    cand = extract_from_slug("Quyet-dinh-440-QD-TTCP-2021-ke-hoach-kiem-tra", "url")
    assert cand is not None
    assert cand.source == "url"
    assert cand.document_type == "Quyết định"
    assert cand.document_number == "440-QD-TTCP"
    assert cand.normalized_identity == "quyet-dinh::440-qd-ttcp"


def test_extract_from_header_standard_passage() -> None:
    """Test extracting type and number from typical Vietnamese legal document preamble."""
    passage = (
        "CHÍNH PHỦ\n"
        "Số: 99/2003/NĐ-CP\n"
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n"
        "NGHỊ ĐỊNH\n"
        "Ban hành Quy chế Khu công nghệ cao\n"
        "Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001;\n"
        "Điều 1. Ban hành kèm theo Nghị định này..."
    )
    cand = extract_from_header(passage)
    assert cand is not None
    assert cand.source == "header"
    assert cand.document_type == "Nghị định"
    assert cand.document_number == "99/2003/NĐ-CP"
    assert cand.normalized_identity == "nghi-dinh::99-2003-nd-cp"


def test_clean_and_combine_header_lines_multiline_uppercase() -> None:
    """Test combining split multi-word uppercase document type tokens."""
    passage = "BỘ TÀI CHÍNH\nTHÔNG\nTƯ\nLIÊN TỊCH\nSố: 01/2020/TTLT\nCăn cứ..."
    combined = clean_and_combine_header_lines(passage)
    assert any(l.lower().startswith("thông tư liên tịch") for l in combined)


def test_extract_from_header_bounds_before_cancu() -> None:
    """Test strict header parsing safety: does NOT extract referenced law from enacting citations."""
    passage = (
        "ỦY BAN NHÂN DÂN\n"
        "Số: 15/2018/QĐ-UBND\n"
        "QUYẾT ĐỊNH\n"
        "Căn cứ Luật Đất đai số 45/2013/QH13;\n"
        "Căn cứ Nghị định số 43/2014/NĐ-CP;\n"
        "Điều 1. Phạm vi điều chỉnh..."
    )
    cand = extract_from_header(passage)
    assert cand is not None
    assert cand.document_type == "Quyết định"
    assert cand.document_number == "15/2018/QĐ-UBND"
    assert cand.normalized_identity == "quyet-dinh::15-2018-qd-ubnd"


def test_resolve_document_identity_all_three_is_strict() -> None:
    """Test unanimous consensus across title, url, and header qualifies as STRICT_MULTI_CHANNEL."""
    name = "Nghi-dinh-99-2003-ND-CP-quy-che"
    link = "https://thuvienphapluat.vn/van-ban/Nghi-dinh-99-2003-ND-CP.aspx"
    passage = "CHÍNH PHỦ\nSố: 99/2003/NĐ-CP\nNGHỊ ĐỊNH\nCăn cứ..."

    res = resolve_document_identity("1001", name, link, passage)
    assert res.status == "HIGH_CONFIDENCE"
    assert res.strict_status == "STRICT_MULTI_CHANNEL"
    assert res.agreement_pattern == "all_three"
    assert res.document_type == "Nghị định"
    assert res.document_number == "99/2003/NĐ-CP"
    assert res.agreeing_sources == ["header", "title", "url"]


def test_resolve_document_identity_url_header_is_strict() -> None:
    """Test consensus between URL slug and passage header qualifies as STRICT_MULTI_CHANNEL."""
    name = None  # Untitled context
    link = "https://thuvienphapluat.vn/van-ban/Thong-tu-01-2020-TT-BTP.aspx"
    passage = "BỘ TƯ PHÁP\nSố: 01/2020/TT-BTP\nTHÔNG TƯ\nCăn cứ..."

    res = resolve_document_identity("1002", name, link, passage)
    assert res.status == "HIGH_CONFIDENCE"
    assert res.strict_status == "STRICT_MULTI_CHANNEL"
    assert res.agreement_pattern == "url_header"
    assert res.document_type == "Thông tư"
    assert res.document_number == "01/2020/TT-BTP"
    assert res.agreeing_sources == ["header", "url"]


def test_resolve_document_identity_title_header_is_strict() -> None:
    """Test consensus between title slug and passage header qualifies as STRICT_MULTI_CHANNEL."""
    name = "Thong-tu-05-2021-TT-BCA"
    link = None  # No link
    passage = "BỘ CÔNG AN\nSố: 05/2021/TT-BCA\nTHÔNG TƯ\nCăn cứ..."

    res = resolve_document_identity("1003", name, link, passage)
    assert res.status == "HIGH_CONFIDENCE"
    assert res.strict_status == "STRICT_MULTI_CHANNEL"
    assert res.agreement_pattern == "title_header"
    assert res.document_type == "Thông tư"
    assert res.document_number == "05/2021/TT-BCA"
    assert res.agreeing_sources == ["header", "title"]


def test_resolve_document_identity_title_url_only_is_provisional() -> None:
    """Test agreement between title and URL alone without header agreement is PROVISIONAL_SINGLE_SOURCE."""
    name = "Quyet-dinh-100-QD-TTg"
    link = "https://thuvienphapluat.vn/van-ban/Quyet-dinh-100-QD-TTg.aspx"
    passage = "Nội dung văn bản không có tiêu đề rõ ràng..."  # No header extracted

    res = resolve_document_identity("1004", name, link, passage)
    assert res.status == "HIGH_CONFIDENCE"
    assert res.strict_status == "PROVISIONAL_SINGLE_SOURCE"
    assert res.agreement_pattern == "title_url_only"
    assert res.agreeing_sources == ["title", "url"]


def test_resolve_document_identity_single_source_is_provisional() -> None:
    """Test single explicit source with no conflict is PROVISIONAL_SINGLE_SOURCE."""
    name = None
    link = "https://thuvienphapluat.vn/van-ban/Quyet-dinh-440-QD-TTCP-2021.aspx"
    passage = "Văn bản không có số hiệu ở đầu..."

    res = resolve_document_identity("1005", name, link, passage)
    assert res.status == "HIGH_CONFIDENCE"
    assert res.strict_status == "PROVISIONAL_SINGLE_SOURCE"
    assert res.agreement_pattern == "single_url"
    assert res.document_type == "Quyết định"
    assert res.document_number == "440-QD-TTCP"


def test_resolve_document_identity_ambiguous_conflict() -> None:
    """Test conflicting candidates between sources fails closed to AMBIGUOUS."""
    name = "Nghi-dinh-99-2003-ND-CP"
    link = "https://thuvienphapluat.vn/van-ban/Thong-tu-01-2020-TT-BTP.aspx"
    passage = "CHÍNH PHỦ\nSố: 99/2003/NĐ-CP\nNGHỊ ĐỊNH\nCăn cứ..."

    res = resolve_document_identity("1006", name, link, passage)
    assert res.status == "AMBIGUOUS"
    assert res.strict_status == "AMBIGUOUS"
    assert res.agreement_pattern == "conflict"
    assert res.document_type is None
    assert res.document_number is None


def test_resolve_document_identity_unresolved_empty() -> None:
    """Test empty/unstructured document fails closed to UNRESOLVED."""
    res = resolve_document_identity("1007", None, None, "")
    assert res.status == "UNRESOLVED"
    assert res.strict_status == "UNRESOLVED"
    assert res.document_type is None
    assert res.document_number is None


def test_run_d1a_feasibility_evaluation_synthetic(tmp_path: Path) -> None:
    """Test full evaluation pipeline with synthetic data verifying strict gates and correct subset numerators."""
    # 1. Create synthetic contexts.zip (4 contexts: 2 strict, 1 provisional, 1 unresolved)
    ctx_zip = tmp_path / "contexts.zip"
    with zipfile.ZipFile(ctx_zip, "w") as z:
        # Context 1: All three agreement -> Strict
        z.writestr(
            "1.json",
            json.dumps({
                "id": 1,
                "name": "Nghi-dinh-99-2003-ND-CP",
                "link": "https://tvpl.vn/Nghi-dinh-99-2003-ND-CP.aspx",
                "passage": "CHÍNH PHỦ\nSố: 99/2003/NĐ-CP\nNGHỊ ĐỊNH\nCăn cứ...",
            }),
        )
        # Context 2: URL + Header agreement -> Strict (untitled)
        z.writestr(
            "2.json",
            json.dumps({
                "id": 2,
                "name": None,
                "link": "https://tvpl.vn/Thong-tu-01-2020-TT-BTP.aspx",
                "passage": "BỘ TƯ PHÁP\nSố: 01/2020/TT-BTP\nTHÔNG TƯ\nCăn cứ...",
            }),
        )
        # Context 3: Title + URL only -> Provisional
        z.writestr(
            "3.json",
            json.dumps({
                "id": 3,
                "name": "Quyet-dinh-10-QD-UBND",
                "link": "https://tvpl.vn/Quyet-dinh-10-QD-UBND.aspx",
                "passage": "Không có số hiệu ở đầu...",
            }),
        )
        # Context 4: Unresolved
        z.writestr(
            "4.json",
            json.dumps({
                "id": 4,
                "name": None,
                "link": None,
                "passage": "",
            }),
        )

    # 2. Create synthetic d0_evidence.zip with 2 proxy links
    d0_zip = tmp_path / "d0_evidence.zip"
    with zipfile.ZipFile(d0_zip, "w") as z:
        z.writestr(
            "train_qa_census.json",
            json.dumps({
                "linkability": {
                    "all_unambiguous_links": [
                        {"question_id": "q1", "target_document_id": "1"},
                        {"question_id": "q2", "target_document_id": "3"},
                    ],
                    "unambiguous_article_links_count": 639,
                }
            }),
        )

    # 3. Create synthetic train.json
    train_json = tmp_path / "train.json"
    train_json.write_text("{}", encoding="utf-8")

    # 4. Create synthetic chunks.jsonl
    chunks_file = tmp_path / "records.jsonl"
    with chunks_file.open("w", encoding="utf-8") as f:
        for i in range(330768):
            doc_id = str((i % 4) + 1)
            f.write(json.dumps({"chunk_id": f"c_{i}", "document_id": doc_id}) + "\n")

    evidence_zip = tmp_path / "evidence_strict.zip"
    staging_dir = tmp_path / "staging"

    # Patch expected checksums for synthetic test
    import scripts.evaluate_document_identity_d1 as d1_mod
    from scripts.evaluate_document_identity_d1 import compute_file_sha256

    orig_ctx_sha = d1_mod.EXPECTED_CONTEXTS_SHA256
    orig_d0_sha = d1_mod.EXPECTED_D0_EVIDENCE_SHA256
    orig_train_sha = d1_mod.EXPECTED_TRAIN_SHA256
    orig_unambig = d1_mod.EXPECTED_UNAMBIGUOUS_LINKS_COUNT

    d1_mod.EXPECTED_CONTEXTS_SHA256 = compute_file_sha256(ctx_zip)
    d1_mod.EXPECTED_D0_EVIDENCE_SHA256 = compute_file_sha256(d0_zip)
    d1_mod.EXPECTED_TRAIN_SHA256 = compute_file_sha256(train_json)
    d1_mod.EXPECTED_UNAMBIGUOUS_LINKS_COUNT = 2

    try:
        res = run_d1a_feasibility_evaluation(
            contexts_zip=ctx_zip,
            d0_evidence_zip=d0_zip,
            train_json=train_json,
            chunks_jsonl=chunks_file,
            evidence_zip=evidence_zip,
            output_dir=staging_dir,
        )

        assert res["feasibility_gate"] == "PASS"
        assert res["final_decision"] == "D1A_STRICT_FEASIBILITY_PASS"

        cov = res["strict_identity_coverage"]
        assert cov["total_contexts"] == 4
        assert cov["non_empty_contexts"] == 3
        assert cov["titled_contexts"] == 2
        assert cov["counts"]["strict_multi_channel"] == 2
        assert cov["counts"]["provisional_single_source"] == 1
        assert cov["counts"]["unresolved"] == 1

        # Check percentages: none may exceed 100%
        for pct in cov["strict_coverage_percentages"].values():
            assert 0.0 <= pct <= 100.0
        for pct in cov["diagnostic_high_confidence_percentages"].values():
            assert 0.0 <= pct <= 100.0

        # Titled subset numerator: context 1 is strict & titled -> 1 / 2 = 50.0%
        assert cov["strict_coverage_percentages"]["titled_contexts_pct"] == 50.0

        # Proxy coverage: q1 (doc 1) is strict, q2 (doc 3) is provisional -> 1 / 2 = 50.0%
        assert res["strict_proxy_coverage"]["covered_proxy_questions"] == 1
        assert res["strict_proxy_coverage"]["proxy_question_coverage_pct"] == 50.0

        # Verify evidence zip contents
        assert evidence_zip.exists()
        with zipfile.ZipFile(evidence_zip) as z:
            names = z.namelist()
            assert "execution/source_identity.json" in names
            assert "execution/strict_identity_policy.json" in names
            assert "results/strict_identity_coverage.json" in names
            assert "results/strict_proxy_coverage.json" in names
            assert "results/d1a_strict_decision.json" in names

    finally:
        d1_mod.EXPECTED_CONTEXTS_SHA256 = orig_ctx_sha
        d1_mod.EXPECTED_D0_EVIDENCE_SHA256 = orig_d0_sha
        d1_mod.EXPECTED_TRAIN_SHA256 = orig_train_sha
        d1_mod.EXPECTED_UNAMBIGUOUS_LINKS_COUNT = orig_unambig
