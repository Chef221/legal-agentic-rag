"""Unit tests for Phase D0 official data census and retrieval unit audit script."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile
import pytest

from scripts.audit_official_data_d0 import (
    compute_percentiles,
    compute_file_sha256,
    normalize_text_for_dup,
    audit_sources_and_artifacts,
    audit_raw_corpus,
    audit_legal_markers_and_parser,
    audit_legal_chunks,
    audit_train_qa_and_linkability,
    audit_retrieval_proxy,
)
from legal_agentic_rag.competition.uit_dsc_2026.passage_cleaner import UitDsc2026PassageCleaner
from legal_agentic_rag.offline.parsing.structure_parser import LegalStructureParser
from legal_agentic_rag.schemas.competition import CompetitionQuestion


def test_compute_percentiles_edge_cases() -> None:
    # Empty
    p_empty = compute_percentiles([])
    assert p_empty["count"] == 0
    assert p_empty["min"] == 0.0
    assert p_empty["max"] == 0.0

    # Single value
    p_single = compute_percentiles([42])
    assert p_single["count"] == 1
    assert p_single["min"] == 42.0
    assert p_single["max"] == 42.0
    assert p_single["p50"] == 42.0

    # Ordered list
    p_multi = compute_percentiles([10, 20, 30, 40, 50])
    assert p_multi["count"] == 5
    assert p_multi["min"] == 10.0
    assert p_multi["max"] == 50.0
    assert p_multi["p50"] == 30.0
    assert p_multi["p25"] == 20.0
    assert p_multi["p75"] == 40.0


def test_normalize_text_for_dup() -> None:
    raw1 = "  Điều 1.\r\nQuy định   chung  \n"
    raw2 = "điều 1.\nquy định chung"
    assert normalize_text_for_dup(raw1) == normalize_text_for_dup(raw2)


def test_compute_file_sha256(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("Hello Legal RAG", encoding="utf-8")
    sha = compute_file_sha256(f)
    assert len(sha) == 64
    assert isinstance(sha, str)


def test_audit_sources_and_artifacts_synthetic(tmp_path: Path) -> None:
    train_f = tmp_path / "train.json"
    train_f.write_text(json.dumps({"1": {"question": "Q1", "answer": "A1"}}), encoding="utf-8")
    
    public_f = tmp_path / "public.json"
    public_f.write_text(json.dumps({"2": {"question": "Q2", "answer": None}}), encoding="utf-8")
    
    ctx_zip = tmp_path / "contexts.zip"
    with zipfile.ZipFile(ctx_zip, "w") as z:
        z.writestr("selected-contexts/context_1.json", json.dumps({"id": 1, "passage": "Nội dung", "name": "Văn bản 1", "link": "http://example.com"}))
        
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    (serving_root / "dataset_manifest.json").write_text(json.dumps({"code_version": "0.40.0"}), encoding="utf-8")
    
    res = audit_sources_and_artifacts(train_f, public_f, ctx_zip, serving_root)
    assert "official_train" in res["sources"]
    assert res["sources"]["official_train"]["record_count"] == 1
    assert "dataset_manifest" in res["artifacts"]


def test_audit_raw_corpus_synthetic(tmp_path: Path) -> None:
    ctx_zip = tmp_path / "contexts.zip"
    with zipfile.ZipFile(ctx_zip, "w") as z:
        z.writestr("selected-contexts/context_1.json", json.dumps({"id": 1, "passage": "Điều 1. Phạm vi áp dụng.\nLuật này quy định...", "name": "Luật 1", "link": "http://link1"}))
        z.writestr("selected-contexts/context_2.json", json.dumps({"id": 2, "passage": "", "name": None, "link": "http://link2"}))
        z.writestr("selected-contexts/context_3.json", json.dumps({"id": 3, "passage": "Điều 1. Phạm vi áp dụng.\nLuật này quy định...", "name": "Luật 1", "link": "http://link1"}))

    res = audit_raw_corpus(ctx_zip)
    assert res["total_records"] == 3
    assert res["non_empty_records"] == 2
    assert res["empty_records"] == 1
    assert res["without_title_records"] == 1
    assert res["duplication"]["exact_duplicate_clusters"] == 1
    assert res["duplication"]["exact_duplicate_records"] == 2


def test_audit_legal_markers_and_parser_synthetic(tmp_path: Path) -> None:
    ctx_zip = tmp_path / "contexts.zip"
    with zipfile.ZipFile(ctx_zip, "w") as z:
        z.writestr("selected-contexts/context_1.json", json.dumps({
            "id": 1,
            "passage": "CHƯƠNG I\nQUY ĐỊNH CHUNG\n\nĐiều 1. Phạm vi\n1. Khoản 1 quy định.\na) Điểm a quy định.",
            "name": "Luật A",
            "link": "http://a.vn"
        }))
        z.writestr("selected-contexts/context_2.json", json.dumps({
            "id": 2,
            "passage": "Văn bản không có cấu trúc điều khoản rõ ràng.",
            "name": "Thông báo",
            "link": "http://b.vn"
        }))

    cleaner = UitDsc2026PassageCleaner()
    parser = LegalStructureParser()
    res = audit_legal_markers_and_parser(ctx_zip, cleaner, parser)
    assert res["docs_parsed_article_ge_1"] == 1
    assert res["docs_zero_structure"] == 1
    assert res["raw_marker_presence_counts"]["DIEU"] == 1


def test_audit_legal_chunks_synthetic(tmp_path: Path) -> None:
    serving_root = tmp_path / "serving"
    chunks_dir = serving_root / "legal_chunks"
    chunks_dir.mkdir(parents=True)
    records_f = chunks_dir / "records.jsonl"
    
    chunk1 = {
        "chunk_id": "c1",
        "document_id": "doc1",
        "chunk_index": 0,
        "text": "Điều 1. Quy định nếu",
        "search_text": "Văn bản: Luật 1\nĐiều 1: Quy định\nNội dung:\nĐiều 1. Quy định nếu",
        "token_count": 10,
        "structure": {"article_number": "1", "clause_numbers": ["1"], "point_numbers": []},
        "metadata": {"chunk_strategy": "article_chunk"},
        "document_title": "Luật 1",
    }
    chunk2 = {
        "chunk_id": "c2",
        "document_id": "doc1",
        "chunk_index": 1,
        "text": "người sử dụng lao động vi phạm thì bị phạt.",
        "search_text": "Văn bản: Luật 1\nNội dung:\nngười sử dụng lao động vi phạm thì bị phạt.",
        "token_count": 12,
        "structure": {"article_number": "1", "clause_numbers": ["2"], "point_numbers": []},
        "metadata": {"chunk_strategy": "clause_group"},
        "document_title": "Luật 1",
    }
    
    with records_f.open("w", encoding="utf-8") as f:
        f.write(json.dumps(chunk1) + "\n")
        f.write(json.dumps(chunk2) + "\n")
        
    res = audit_legal_chunks(serving_root)
    assert res["total_chunks"] == 2
    assert res["unique_source_documents"] == 1
    assert res["strategy_distribution"]["article_chunk"]["count"] == 1
    assert res["strategy_distribution"]["clause_group"]["count"] == 1
    assert "CONDITION_OPEN_AT_LEFT_BOUNDARY" in res["boundary_risk"]["risk_counts"]


def test_audit_train_qa_and_linkability_synthetic(tmp_path: Path) -> None:
    train_f = tmp_path / "train.json"
    train_data = {
        "1001": {
            "question": "Mức xử phạt theo Nghị định 100/2019/NĐ-CP là bao nhiêu?",
            "answer": "Theo quy định tại Điều 5 Nghị định 100/2019/NĐ-CP, phạt tiền từ 2.000.000 đồng."
        },
        "1002": {
            "question": "Có được chuyển nhượng không?",
            "answer": "Không được phép chuyển nhượng."
        }
    }
    train_f.write_text(json.dumps(train_data), encoding="utf-8")
    
    public_f = tmp_path / "public.json"
    public_data = {
        "2001": {
            "question": "Có được chuyển nhượng không?",
            "answer": None
        }
    }
    public_f.write_text(json.dumps(public_data), encoding="utf-8")
    
    ctx_zip = tmp_path / "contexts.zip"
    with zipfile.ZipFile(ctx_zip, "w") as z:
        z.writestr("selected-contexts/context_50.json", json.dumps({
            "id": 50,
            "passage": "Nghị định 100/2019/NĐ-CP quy định xử phạt vi phạm hành chính...",
            "name": "Nghị định 100/2019/NĐ-CP xử phạt giao thông",
            "link": "http://thuvienphapluat.vn/van-ban/100-2019-ND-CP"
        }))
        
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    
    res = audit_train_qa_and_linkability(train_f, public_f, ctx_zip, serving_root)
    assert res["total_train_questions"] == 2
    assert res["taxonomy_distribution"]["binary_yes_no"]["count"] >= 1
    assert res["legal_reference_signal_counts"]["a_has_doc_number"] == 1
    assert res["linkability"]["unambiguous_doc_links_count"] == 1
    assert res["duplication"]["train_public_exact_overlap_count"] == 1


def test_audit_retrieval_proxy_mock(tmp_path: Path) -> None:
    serving_root = tmp_path / "serving"
    bm25_dir = serving_root / "bm25"
    bm25_dir.mkdir(parents=True)
    db_path = bm25_dir / "index.sqlite3"
    
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, document_id, search_text)")
    cursor.execute("INSERT INTO chunks_fts VALUES ('c1', 'doc_50', 'Nghị định 100 2019 NĐ CP xử phạt')")
    conn.commit()
    conn.close()
    
    links = [{"question_id": "q1", "target_document_id": "doc_50"}]
    questions = [CompetitionQuestion(question_id="q1", question="Nghị định 100 2019 NĐ CP", reference_answer="A")]
    
    res = audit_retrieval_proxy(serving_root, links, questions)
    assert res["status"] == "COMPLETED"
    assert res["document_recall_at_1"] == 100.0
