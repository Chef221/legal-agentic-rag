"""Tests for deterministic evidence applicability scoring with legacy and V2 metadata."""

import copy
import pytest

from legal_agentic_rag.configuration import (
    EvidenceSelectionConfig,
    GenerationConfig,
)
from legal_agentic_rag.generation import EvidenceSelector
from legal_agentic_rag.schemas import (
    EvidenceApplicability,
    QueryAnalysis,
    RetrievalHit,
    RetrievalQuery,
    RetrievalStrategy,
)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="selection-query",
        original_question=(
            "Điều 113 Luật số 45/2019/QH14 quy định nghỉ hằng năm thế nào?"
        ),
        normalized_question=(
            "Điều 113 Luật số 45/2019/QH14 quy định nghỉ hằng năm thế nào?"
        ),
        query_analysis=QueryAnalysis(
            document_numbers=["45/2019/QH14"],
            article_numbers=["113"],
        ),
        top_k=2,
        candidate_k=2,
    )


def _hit(
    chunk_id: str,
    rank: int,
    *,
    document_number: str,
    article_number: str,
    effect_status: str | None = "còn hiệu lực",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text="Người lao động được nghỉ hằng năm.",
        metadata={
            "document_title": "Bộ luật Lao động",
            "document_number": document_number,
            "effect_status": effect_status,
            "structure": {"article_number": article_number},
        },
    )


def _v2_hit(
    unit_id: str,
    rank: int,
    *,
    doc_number: str | None = "45/2019/QH14",
    doc_title: str = "Bộ luật Lao động 2019",
    article_label: str | None = "113",
    clause_label: str | None = "1",
    point_label: str | None = "a",
    heading_path: list[dict] | None = None,
    text: str = "Người lao động được nghỉ hằng năm theo quy định.",
    retrieval_text: str = "Văn bản: Bộ luật Lao động\n---\nKhoản 1 Điều 113...",
) -> RetrievalHit:
    headings = heading_path if heading_path is not None else [
        {"type": "PART", "label": "I", "title": "QUY ĐỊNH CHUNG"},
        {"type": "CHAPTER", "label": "VII", "title": "THỜI GIỜ LÀM VIỆC, NGHỈ NGƠI"},
    ]
    return RetrievalHit(
        chunk_id=unit_id,
        document_id=f"doc-{unit_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text=text,
        metadata={
            "provision_id": f"doc:{unit_id}::art:{article_label}",
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
    )


def test_explicit_document_and_article_match_outrank_raw_rank() -> None:
    """User-supplied legal references can promote the matching provision."""
    selector = EvidenceSelector()

    scored = selector.score(
        _query(),
        [
            _hit(
                "wrong",
                1,
                document_number="145/2020/NĐ-CP",
                article_number="113",
            ),
            _hit(
                "matching",
                2,
                document_number="45/2019/QH14",
                article_number="113",
            ),
        ],
    )

    assert [item.hit.chunk_id for item in scored] == ["matching", "wrong"]
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH
    assert (
        scored[1].applicability
        == EvidenceApplicability.REFERENCE_MISMATCH
    )
    assert scored[0].document_reference_match is True
    assert scored[0].article_reference_match is True


def test_configured_inactive_status_is_penalized_without_guessing_labels() -> None:
    """Only an explicitly configured inactive label changes selection score."""
    selector = EvidenceSelector(
        EvidenceSelectionConfig(inactive_penalty=2.0),
        GenerationConfig(
            inactive_effect_statuses=frozenset({"hết hiệu lực"})
        ),
    )

    scored = selector.score(
        _query().model_copy(update={"query_analysis": None}),
        [
            _hit(
                "inactive",
                1,
                document_number="1/2020/QH",
                article_number="1",
                effect_status="Hết hiệu lực",
            ),
            _hit(
                "active",
                2,
                document_number="2/2020/QH",
                article_number="2",
            ),
        ],
    )

    assert [item.hit.chunk_id for item in scored] == ["active", "inactive"]
    assert scored[1].applicability == EvidenceApplicability.INACTIVE


def test_selector_is_deterministic_for_equal_candidates() -> None:
    """Stable source rank and chunk ID break equal-score ties."""
    query = _query().model_copy(update={"query_analysis": None})
    hit_b = _hit(
        "b",
        1,
        document_number="1/2020/QH",
        article_number="1",
    )
    hit_a = hit_b.model_copy(update={"chunk_id": "a"})

    first = EvidenceSelector().score(query, [hit_b, hit_a])
    second = EvidenceSelector().score(query, [hit_a, hit_b])

    assert [item.hit.chunk_id for item in first] == ["a", "b"]
    assert [item.hit.chunk_id for item in second] == ["a", "b"]


# ==============================================================================
# V2 EVIDENCE SELECTOR TESTS
# ==============================================================================

def test_v2_explicit_document_and_article_match_and_promotion() -> None:
    """V2 document_identity and hierarchy article_label match and promote over raw rank."""
    selector = EvidenceSelector()

    hit_wrong = _v2_hit("wrong_v2", 1, doc_number="145/2020/ND-CP", article_label="113")
    hit_matching = _v2_hit("match_v2", 2, doc_number="45/2019/QH14", article_label="113")

    scored = selector.score(_query(), [hit_wrong, hit_matching])

    assert [item.hit.chunk_id for item in scored] == ["match_v2", "wrong_v2"]
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH
    assert scored[0].document_reference_match is True
    assert scored[0].article_reference_match is True
    assert scored[1].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_v2_lexical_overlap_incorporates_title_hierarchy_headings() -> None:
    """V2 document title, article/clause/point labels, and heading titles contribute to lexical overlap."""
    query = RetrievalQuery(
        query_id="q_overlap",
        original_question="môi trường tài nguyên khoáng sản quy định chung",
        normalized_question="môi trường tài nguyên khoáng sản quy định chung",
        top_k=2,
    )

    hit_rich = _v2_hit(
        "rich_v2",
        1,
        doc_title="Luật Bảo vệ Môi trường",
        article_label="5",
        clause_label="1",
        point_label="a",
        heading_path=[{"type": "CHAPTER", "label": "I", "title": "Tài nguyên khoáng sản"}],
        text="Quy định chung về quản lý tài nguyên.",
    )

    hit_plain = _v2_hit(
        "plain_v2",
        2,
        doc_title="Luật Đất đai",
        article_label="1",
        clause_label="1",
        point_label="b",
        heading_path=[],
        text="Không chứa từ khoá trên.",
    )

    selector = EvidenceSelector()
    scored = selector.score(query, [hit_rich, hit_plain])

    assert scored[0].hit.chunk_id == "rich_v2"
    assert scored[0].lexical_overlap_score > scored[1].lexical_overlap_score


def test_v2_missing_identity_and_unknown_effect_status() -> None:
    """V2 hits with None document_number or missing effect status are handled conservatively without fabrication."""
    selector = EvidenceSelector()
    hit_sparse = _v2_hit("sparse_v2", 1, doc_number=None, article_label=None)

    query = _query().model_copy(update={"query_analysis": None})
    scored = selector.score(query, [hit_sparse])

    assert scored[0].applicability == EvidenceApplicability.UNKNOWN
    assert scored[0].document_reference_match is None
    assert scored[0].article_reference_match is None


def test_v2_metadata_is_not_mutated() -> None:
    """Scoring does not mutate input V2 hit metadata."""
    hit = _v2_hit("v2_immutable", 1)
    original_meta = copy.deepcopy(hit.metadata)

    selector = EvidenceSelector()
    _ = selector.score(_query(), [hit])

    assert hit.metadata == original_meta
