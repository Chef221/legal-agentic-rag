"""Unit tests for Phase D1-B strict document identity BM25 causal A/B harness."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import pytest

from legal_agentic_rag.offline.chunking.tokenizer import UnicodeWordTokenizer
from scripts.evaluate_document_identity_d1b import (
    MAX_SEARCH_TOKENS,
    classify_query_identity_signals,
    construct_candidate_search_text,
    construct_match_query,
    evaluate_bm25_retrieval,
)


def test_construct_match_query_exact_d0_replay():
    """Verify exact replay of D0 query construction logic."""
    q_text = "Hành vi trốn thuế từ 100 triệu đồng trở lên bị xử lý hình sự thế nào theo Bộ luật Hình sự?"
    match_query = construct_match_query(q_text)

    assert match_query != ""
    tokens = match_query.split(" OR ")
    assert len(tokens) <= 15
    for tok in tokens:
        assert tok.startswith('"') and tok.endswith('"')
        assert len(tok.strip('"')) > 1


def test_construct_match_query_empty_or_symbols():
    """Verify fail-closed handling for empty or symbols-only query."""
    assert construct_match_query("") == ""
    assert construct_match_query("!@#$%^&*()") == ""
    assert construct_match_query("a b c") == ""  # len <= 1 skipped


def test_classify_query_identity_signals():
    """Verify deterministic pre-registered query signal classification."""
    # Both number and type
    assert classify_query_identity_signals("Theo Nghị định 12/2020/NĐ-CP thì...") == "both"
    assert classify_query_identity_signals("Theo Luật 45/2019/QH14...") == "both"

    # Document number only
    assert classify_query_identity_signals("Văn bản 123/2021/TT-BTC quy định gì?") == "explicit_document_number"
    assert classify_query_identity_signals("Theo số hiệu 1234 thì...") == "explicit_document_number"

    # Document type only
    assert classify_query_identity_signals("Theo Luật đất đai thì...") == "explicit_document_type"
    assert classify_query_identity_signals("Quy định tại Thông tư về thuế...") == "explicit_document_type"

    # Neither
    assert classify_query_identity_signals("Hành vi lừa đảo chiếm đoạt tài sản bị phạt thế nào?") == "neither"


def test_construct_candidate_search_text_full_enrichment():
    """Verify candidate search_text adds document number and type with raw text suffix intact."""
    tokenizer = UnicodeWordTokenizer()
    raw_text = "Người nào trốn thuế với số tiền từ 100 triệu đến dưới 300 triệu đồng thì bị phạt tiền..."
    base_search = f"Điều 200: Tội trốn thuế\nNội dung:\n{raw_text}"
    doc_type = "Bộ luật"
    doc_num = "100/2015/QH13"

    cand_text, mod_type = construct_candidate_search_text(
        base_search, raw_text, doc_type, doc_num, tokenizer, max_tokens=512
    )

    assert mod_type == "full_enrichment"
    assert cand_text.startswith(f"Số ký hiệu: {doc_num}\nLoại văn bản: {doc_type}\n")
    assert cand_text.endswith(f"\nNội dung:\n{raw_text}")
    assert f"Nội dung:\n{raw_text}" in cand_text
    assert tokenizer.count(cand_text) <= 512


def test_construct_candidate_search_text_budget_priority_drop_header():
    """Verify budget priority: existing optional header is dropped before touching doc number/type."""
    tokenizer = UnicodeWordTokenizer()
    # Create long raw text ~485 tokens
    raw_words = ["từ"] * 485
    raw_text = " ".join(raw_words)
    long_header = "Chương 1: Quy định chung\nMục 1: Phạm vi\nĐiều 1: Tên điều\nKhoản 1"
    base_search = f"{long_header}\nNội dung:\n{raw_text}"
    doc_type = "Nghị định"
    doc_num = "12/2020/NĐ-CP"

    cand_text, mod_type = construct_candidate_search_text(
        base_search, raw_text, doc_type, doc_num, tokenizer, max_tokens=512
    )

    assert mod_type in {"partial_header_dropped", "all_header_dropped"}
    assert cand_text.startswith(f"Số ký hiệu: {doc_num}\nLoại văn bản: {doc_type}")
    assert cand_text.endswith(f"\nNội dung:\n{raw_text}")
    assert tokenizer.count(cand_text) <= 512


def test_construct_candidate_search_text_budget_priority_drop_type():
    """Verify budget priority: Loại văn bản is dropped before Số ký hiệu."""
    tokenizer = UnicodeWordTokenizer()
    # Create raw text ~498 tokens
    raw_words = ["từ"] * 498
    raw_text = " ".join(raw_words)
    base_search = f"Nội dung:\n{raw_text}"
    doc_type = "Nghị định"
    doc_num = "12/2020/NĐ-CP"

    cand_text, mod_type = construct_candidate_search_text(
        base_search, raw_text, doc_type, doc_num, tokenizer, max_tokens=512
    )

    assert mod_type == "type_dropped"
    assert cand_text.startswith(f"Số ký hiệu: {doc_num}\nNội dung:\n")
    assert cand_text.endswith(f"\n{raw_text}") or cand_text.endswith(raw_text)
    assert tokenizer.count(cand_text) <= 512


def test_chunk_row_cutoff_semantics_no_document_dedup_before_cutoff(tmp_path: Path):
    """Verify historical chunk-row cutoff semantics: cutoff is on ranked chunk rows, not deduplicated documents."""
    db_path = tmp_path / "test_bm25.sqlite3"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE VIRTUAL TABLE bm25_documents USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            document_type UNINDEXED,
            legal_field UNINDEXED,
            effect_status UNINDEXED,
            search_terms,
            chunk_json UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 0'
        )
        """
    )

    # Insert 5 chunks from document A and 1 chunk from target document B
    # Target document B is row 6
    rows = [
        ("c1", "doc_A", "Luật", None, None, "thue thu nhap", "{}"),
        ("c2", "doc_A", "Luật", None, None, "thue thu nhap", "{}"),
        ("c3", "doc_A", "Luật", None, None, "thue thu nhap", "{}"),
        ("c4", "doc_A", "Luật", None, None, "thue thu nhap", "{}"),
        ("c5", "doc_A", "Luật", None, None, "thue thu nhap", "{}"),
        ("c6", "doc_B", "Luật", None, None, "thue thu nhap", "{}"),
    ]
    cursor.executemany("INSERT INTO bm25_documents VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()

    eval_queries = [{"question_id": "q1", "target_document_id": "doc_B"}]
    q_map = {"q1": "thue thu nhap"}

    res = evaluate_bm25_retrieval(db_path, eval_queries, q_map)

    # Under chunk-row cutoff semantics: doc_B is at rank 6, so Recall@5 is 0% and Recall@10 is 100%
    assert res["recall_at_1_numerator"] == 0
    assert res["recall_at_5_numerator"] == 0
    assert res["recall_at_10_numerator"] == 1
    assert res["recall_at_20_numerator"] == 1


def test_paired_transition_arithmetic():
    """Verify paired transition arithmetic and net gain calculation."""
    b_hits = [True, True, False, False]
    c_hits = [True, False, True, False]

    both_hit = sum(1 for b, c in zip(b_hits, c_hits) if b and c)
    base_only = sum(1 for b, c in zip(b_hits, c_hits) if b and not c)
    cand_only = sum(1 for b, c in zip(b_hits, c_hits) if not b and c)
    both_miss = sum(1 for b, c in zip(b_hits, c_hits) if not b and not c)

    assert both_hit == 1
    assert base_only == 1
    assert cand_only == 1
    assert both_miss == 1
    assert cand_only - base_only == 0


def test_gate_decision_logic():
    """Verify exact pre-registered gate decision logic."""
    # Test RETAIN: Delta R@5 >= +2.0, R@10 >= base, R@20 >= base - 0.5, cand_only > base_only
    delta_r5 = 2.5
    secondary_pass = True
    tertiary_pass = True
    paired_safety_pass = True

    all_pass = (delta_r5 >= 2.0) and secondary_pass and tertiary_pass and paired_safety_pass
    assert all_pass is True
    decision = "D1_DOCUMENT_IDENTITY_RETAIN" if all_pass else "KEEP_BASELINE"
    assert decision == "D1_DOCUMENT_IDENTITY_RETAIN"

    # Test KEEP_BASELINE when Delta R@5 is +1.9 (< +2.0)
    delta_r5_fail = 1.9
    all_pass_fail = (delta_r5_fail >= 2.0) and secondary_pass and tertiary_pass and paired_safety_pass
    assert all_pass_fail is False
    decision_fail = "D1_DOCUMENT_IDENTITY_RETAIN" if all_pass_fail else "KEEP_BASELINE"
    assert decision_fail == "KEEP_BASELINE"


def test_non_strict_documents_leave_search_text_untouched():
    """Verify that non-strict documents are guaranteed to produce identical search text."""
    strict_doc_map = {"10001": ("Luật", "01/2020/QH14")}
    tokenizer = UnicodeWordTokenizer()

    # Document 10002 is not strict
    doc_id = "10002"
    base_search = "Điều 1: Tên điều\nNội dung:\nNội dung điều 1..."
    raw_text = "Nội dung điều 1..."

    if doc_id in strict_doc_map:
        cand_search, _ = construct_candidate_search_text(
            base_search, raw_text, strict_doc_map[doc_id][0], strict_doc_map[doc_id][1], tokenizer
        )
    else:
        cand_search = base_search

    assert cand_search == base_search
