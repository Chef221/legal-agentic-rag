"""Tests for ArticleAuthorityStore and FirstKFullArticleAnswerAssembler."""

import hashlib
import json
from pathlib import Path
import pytest

from legal_agentic_rag.configuration.online import ArticleAnswerConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.generation.article_authority import (
    ArticleAuthorityStore,
    FirstKFullArticleAnswerAssembler,
)
from legal_agentic_rag.schemas.answering import (
    ContextBuildResult,
    Evidence,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalStrategy,
)


def _create_synthetic_jsonl(tmp_path: Path, records: list[dict[str, str]]) -> tuple[Path, str, int]:
    file_path = tmp_path / "lookup.jsonl"
    hasher = hashlib.sha256()
    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            hasher.update(line.encode("utf-8"))
            f.write(line)
    return file_path, hasher.hexdigest(), len(records)


def test_article_authority_store_valid_load(tmp_path: Path) -> None:
    records = [
        {"document_id": "doc1", "article_identity": "1", "article_text": "Article 1 text"},
        {"document_id": "doc1", "article_identity": "2", "article_text": "Article 2 text"},
    ]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    store = ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)

    assert len(store) == 2
    assert store.sha256 == sha
    assert store.record_count == 2
    assert store.get("doc1", "1") == "Article 1 text"
    assert store.get("doc1", " 1 ") == "Article 1 text"
    assert store.get("doc1", "2") == "Article 2 text"
    assert store.get("doc1", "3") is None
    assert store.has("doc1", "1") is True
    assert store.has("doc2", "1") is False


def test_article_authority_store_sha_mismatch_rejected(tmp_path: Path) -> None:
    records = [{"document_id": "doc1", "article_identity": "1", "article_text": "Text"}]
    path, _, count = _create_synthetic_jsonl(tmp_path, records)
    with pytest.raises(DataValidationError, match="SHA256 mismatch"):
        ArticleAuthorityStore.from_jsonl(path, expected_sha256="0" * 64, expected_record_count=count)


def test_article_authority_store_count_mismatch_rejected(tmp_path: Path) -> None:
    records = [{"document_id": "doc1", "article_identity": "1", "article_text": "Text"}]
    path, sha, _ = _create_synthetic_jsonl(tmp_path, records)
    with pytest.raises(DataValidationError, match="record count mismatch"):
        ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=2)


def test_article_authority_store_duplicate_key_rejected(tmp_path: Path) -> None:
    records = [
        {"document_id": "doc1", "article_identity": "1", "article_text": "Text A"},
        {"document_id": "doc1", "article_identity": "1", "article_text": "Text B"},
    ]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    with pytest.raises(DataValidationError, match="Duplicate exact article key"):
        ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)


@pytest.mark.parametrize(
    "invalid_record",
    [
        {"document_id": "", "article_identity": "1", "article_text": "Text"},
        {"document_id": "doc1", "article_identity": "", "article_text": "Text"},
        {"document_id": "doc1", "article_identity": "1", "article_text": "  "},
        {"document_id": "doc1"},
    ],
)
def test_article_authority_store_malformed_record_rejected(tmp_path: Path, invalid_record: dict[str, str]) -> None:
    path = tmp_path / "malformed.jsonl"
    line = json.dumps(invalid_record) + "\n"
    hasher = hashlib.sha256(line.encode("utf-8"))
    path.write_text(line, encoding="utf-8")
    with pytest.raises(DataValidationError):
        ArticleAuthorityStore.from_jsonl(path, expected_sha256=hasher.hexdigest(), expected_record_count=1)


def test_article_authority_store_malformed_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not valid json\n", encoding="utf-8")
    sha = hashlib.sha256(b"{not valid json\n").hexdigest()
    with pytest.raises(DataValidationError, match="Malformed JSON"):
        ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=1)


def test_article_authority_store_no_fuzzy_or_chunk_id_inference(tmp_path: Path) -> None:
    records = [{"document_id": "doc1", "article_identity": "10", "article_text": "Article 10"}]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    store = ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)

    # Must not match substring, near match, or chunk_id
    assert store.get("doc1", "1") is None
    assert store.get("doc1", "010") is None
    assert store.get("doc1", "Điều 10") is None
    assert store.get("doc1", "chunk-10") is None


def test_assembler_first_2_concatenation_and_deduplication(tmp_path: Path) -> None:
    records = [
        {"document_id": "doc1", "article_identity": "1", "article_text": "Full Article 1 Content"},
        {"document_id": "doc1", "article_identity": "2", "article_text": "Full Article 2 Content"},
        {"document_id": "doc2", "article_identity": "5", "article_text": "Full Article 5 Content"},
    ]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    store = ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)
    assembler = FirstKFullArticleAnswerAssembler(store, config=ArticleAnswerConfig(enabled=True, max_articles=2))

    ev1 = Evidence(
        evidence_id="E1",
        chunk_id="c1",
        document_id="doc1",
        article_number="1",
        text="Fragment of article 1",
    )
    ev2 = Evidence(
        evidence_id="E2",
        chunk_id="c2",
        document_id="doc1",
        article_number="1",  # duplicate article 1
        text="Another fragment of article 1",
    )
    ev3 = Evidence(
        evidence_id="E3",
        chunk_id="c3",
        document_id="doc1",
        article_number="2",  # article 2
        text="Fragment of article 2",
    )
    ev4 = Evidence(
        evidence_id="E4",
        chunk_id="c4",
        document_id="doc2",
        article_number="5",  # article 5 (should be omitted because max_articles=2)
        text="Fragment of article 5",
    )
    context = ContextBuildResult(
        evidence=[ev1, ev2, ev3, ev4],
        input_hit_count=4,
        selected_count=4,
        omitted_hit_count=0,
        duplicate_hit_count=0,
        estimated_token_count=100,
    )
    query = RetrievalQuery(
        query_id="q1",
        original_question="What is the rule?",
        normalized_question="what is the rule",
    )

    response = assembler.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=context)

    assert response.question == "What is the rule?"
    assert response.answer == "Full Article 1 Content\n\nFull Article 2 Content"
    assert response.insufficient_evidence is False
    assert len(response.citations) == 2
    assert response.citations[0].evidence_id == "E1"
    assert response.citations[0].document_id == "doc1"
    assert response.citations[0].article_number == "1"
    assert response.citations[1].evidence_id == "E3"
    assert response.citations[1].document_id == "doc1"
    assert response.citations[1].article_number == "2"

    meta = response.metadata
    assert meta["answer_path"] == "m55_first_k_full_article"
    assert meta["requested_max_articles"] == 2
    assert meta["resolved_article_count"] == 2
    assert meta["structural_fallback_used"] is False
    assert meta["source_evidence_ids"] == ["E1", "E3"]
    assert meta["included_articles"] == [
        {"document_id": "doc1", "article_identity": "1"},
        {"document_id": "doc1", "article_identity": "2"},
    ]


def test_assembler_single_article(tmp_path: Path) -> None:
    records = [{"document_id": "doc1", "article_identity": "1", "article_text": "Single Article Text"}]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    store = ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)
    assembler = FirstKFullArticleAnswerAssembler(store, config=ArticleAnswerConfig(enabled=True, max_articles=2))

    ev1 = Evidence(
        evidence_id="E1",
        chunk_id="c1",
        document_id="doc1",
        article_number="1",
        text="Frag 1",
    )
    context = ContextBuildResult(
        evidence=[ev1],
        input_hit_count=1,
        selected_count=1,
        omitted_hit_count=0,
        duplicate_hit_count=0,
        estimated_token_count=20,
    )
    query = RetrievalQuery(query_id="q1", original_question="Question?", normalized_question="question")

    response = assembler.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=context)
    assert response.answer == "Single Article Text"
    assert len(response.citations) == 1
    assert response.metadata["resolved_article_count"] == 1


def test_assembler_unresolved_evidence_skipped_and_structural_fallback(tmp_path: Path) -> None:
    records = [{"document_id": "doc1", "article_identity": "99", "article_text": "Irrelevant"}]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    store = ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)
    assembler = FirstKFullArticleAnswerAssembler(
        store,
        config=ArticleAnswerConfig(enabled=True, max_articles=2, structural_fallback_max_evidence=3),
    )

    ev1 = Evidence(evidence_id="E1", chunk_id="c1", document_id="docX", article_number="1", text="Fallback text 1")
    ev2 = Evidence(evidence_id="E2", chunk_id="c2", document_id="docX", article_number="2", text="Fallback text 2")
    ev3 = Evidence(evidence_id="E3", chunk_id="c3", document_id="docX", article_number="3", text="Fallback text 3")
    ev4 = Evidence(evidence_id="E4", chunk_id="c4", document_id="docX", article_number="4", text="Fallback text 4")
    context = ContextBuildResult(
        evidence=[ev1, ev2, ev3, ev4],
        input_hit_count=4,
        selected_count=4,
        omitted_hit_count=0,
        duplicate_hit_count=0,
        estimated_token_count=50,
    )
    query = RetrievalQuery(query_id="q1", original_question="Question?", normalized_question="question")

    response = assembler.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=context)

    # Top 3 structural fallback
    assert response.answer == "Fallback text 1\n\nFallback text 2\n\nFallback text 3"
    assert response.insufficient_evidence is False
    assert len(response.citations) == 3
    assert response.citations[0].evidence_id == "E1"
    assert response.citations[1].evidence_id == "E2"
    assert response.citations[2].evidence_id == "E3"
    assert response.metadata["structural_fallback_used"] is True
    assert response.metadata["resolved_article_count"] == 0
    assert "structural_fallback_used" in response.warnings


def test_assembler_empty_evidence_abstains(tmp_path: Path) -> None:
    records = [{"document_id": "doc1", "article_identity": "1", "article_text": "Text"}]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    store = ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)
    assembler = FirstKFullArticleAnswerAssembler(store, config=ArticleAnswerConfig(enabled=True))

    context = ContextBuildResult(
        evidence=[],
        input_hit_count=0,
        selected_count=0,
        omitted_hit_count=0,
        duplicate_hit_count=0,
        estimated_token_count=0,
    )
    query = RetrievalQuery(query_id="q1", original_question="Question?", normalized_question="question")

    response = assembler.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=context)
    assert response.insufficient_evidence is True
    assert len(response.citations) == 0
    assert "no_evidence_available" in response.warnings

def test_abstention_text_exact_match_with_legacy_workflow(tmp_path: Path) -> None:
    from legal_agentic_rag.agent.workflow import _ABSTENTION_TEXT
    from legal_agentic_rag.generation.article_authority import DEFAULT_ABSTENTION_TEXT

    assert DEFAULT_ABSTENTION_TEXT == _ABSTENTION_TEXT
    assert DEFAULT_ABSTENTION_TEXT == "Hệ thống chưa tìm thấy căn cứ pháp luật đủ rõ trong dữ liệu hiện có để trả lời chắc chắn."

    records = [{"document_id": "doc1", "article_identity": "1", "article_text": "Text"}]
    path, sha, count = _create_synthetic_jsonl(tmp_path, records)
    store = ArticleAuthorityStore.from_jsonl(path, expected_sha256=sha, expected_record_count=count)
    assembler = FirstKFullArticleAnswerAssembler(store, config=ArticleAnswerConfig(enabled=True))

    context = ContextBuildResult(
        evidence=[],
        input_hit_count=0,
        selected_count=0,
        omitted_hit_count=0,
        duplicate_hit_count=0,
        estimated_token_count=0,
    )
    query = RetrievalQuery(query_id="q1", original_question="Question?", normalized_question="question")
    response = assembler.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=context)
    assert response.answer == _ABSTENTION_TEXT

def test_article_assembler_preserves_raw_trailing_whitespace(tmp_path: Path) -> None:
    """M55 Article mode must preserve exact raw article text character-for-character."""
    from legal_agentic_rag.configuration.online import ArticleAnswerConfig
    from legal_agentic_rag.schemas.answering import ContextBuildResult, Evidence
    from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy

    lookup_file = tmp_path / "lookup.jsonl"
    records = [
        {"document_id": "doc1", "article_identity": "1", "article_text": "Nội dung Điều 1\n\n"},
        {"document_id": "doc1", "article_identity": "2", "article_text": "Nội dung Điều 2\n\n"},
    ]
    hasher = hashlib.sha256()
    with open(lookup_file, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            hasher.update(line.encode("utf-8"))
            f.write(line)

    store = ArticleAuthorityStore.from_jsonl(
        lookup_file, expected_sha256=hasher.hexdigest(), expected_record_count=2
    )

    # 1. Single article retains trailing \n\n
    assembler_k1 = FirstKFullArticleAnswerAssembler(store, config=ArticleAnswerConfig(enabled=True, max_articles=1))
    ev1 = Evidence(
        evidence_id="E1",
        chunk_id="c1",
        document_id="doc1",
        article_number="1",
        text="fragment 1",
    )
    ctx1 = ContextBuildResult(evidence=[ev1], input_hit_count=1, selected_count=1, omitted_hit_count=0, duplicate_hit_count=0, estimated_token_count=100)
    query = RetrievalQuery(query_id="q1", original_question="q", normalized_question="q")
    resp1 = assembler_k1.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=ctx1)
    assert resp1.answer == "Nội dung Điều 1\n\n"
    assert resp1.answer.endswith("\n\n")

    # 2. Two articles retain every character and insert \n\n separator
    assembler_k2 = FirstKFullArticleAnswerAssembler(store, config=ArticleAnswerConfig(enabled=True, max_articles=2))
    ev2 = Evidence(
        evidence_id="E2",
        chunk_id="c2",
        document_id="doc1",
        article_number="2",
        text="fragment 2",
    )
    ctx2 = ContextBuildResult(evidence=[ev1, ev2], input_hit_count=2, selected_count=2, omitted_hit_count=0, duplicate_hit_count=0, estimated_token_count=200)
    resp2 = assembler_k2.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=ctx2)
    expected_ans2 = "Nội dung Điều 1\n\n\n\nNội dung Điều 2\n\n"
    assert resp2.answer == expected_ans2

    # 3. Structural fallback exact semantics
    ev_unresolved = Evidence(
        evidence_id="E3",
        chunk_id="c3",
        document_id="doc999",
        article_number="999",
        text="Fallback text fragment",
    )
    ctx_fallback = ContextBuildResult(evidence=[ev_unresolved], input_hit_count=1, selected_count=1, omitted_hit_count=0, duplicate_hit_count=0, estimated_token_count=50)
    resp_fallback = assembler_k2.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=ctx_fallback)
    assert resp_fallback.metadata["structural_fallback_used"] is True
    assert resp_fallback.answer == "Fallback text fragment"

    # 4. Abstention exact legacy wording
    ctx_empty = ContextBuildResult(evidence=[], input_hit_count=0, selected_count=0, omitted_hit_count=0, duplicate_hit_count=0, estimated_token_count=0)
    resp_empty = assembler_k2.assemble(query=query, strategy=RetrievalStrategy.HYBRID_RERANK, context=ctx_empty)
    assert resp_empty.insufficient_evidence is True
    assert resp_empty.answer == "Hệ thống chưa tìm thấy căn cứ pháp luật đủ rõ trong dữ liệu hiện có để trả lời chắc chắn."
