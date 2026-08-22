"""Tests for deterministic evidence applicability scoring and strict own-document title recovery."""

from typing import Any

from legal_agentic_rag.configuration import (
    ContextGradingConfig,
    EvidenceSelectionConfig,
    GenerationConfig,
)
from legal_agentic_rag.generation import (
    ContextBuilder,
    EvidenceSelector,
    RuleBasedContextGrader,
)
from legal_agentic_rag.schemas import (
    EvidenceApplicability,
    QueryAnalysis,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


def _query(
    document_numbers: list[str] | None = None,
    article_numbers: list[str] | None = None,
) -> RetrievalQuery:
    doc_nums = document_numbers if document_numbers is not None else ["45/2019/QH14"]
    art_nums = article_numbers if article_numbers is not None else ["113"]
    analysis = (
        QueryAnalysis(document_numbers=doc_nums, article_numbers=art_nums)
        if (doc_nums or art_nums)
        else None
    )
    return RetrievalQuery(
        query_id="selection-query",
        original_question="Điều 113 Luật số 45/2019/QH14 quy định nghỉ hằng năm thế nào?",
        normalized_question="Điều 113 Luật số 45/2019/QH14 quy định nghỉ hằng năm thế nào?",
        query_analysis=analysis,
        top_k=2,
        candidate_k=2,
    )


def _hit(
    chunk_id: str,
    rank: int,
    *,
    document_number: Any = None,
    document_title: str | None = "Bo-luat-Lao-dong",
    article_number: str | None = "113",
    effect_status: str | None = "còn hiệu lực",
    text: str = "Người lao động được nghỉ hằng năm.",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text=text,
        metadata={
            "document_title": document_title,
            "document_number": document_number,
            "effect_status": effect_status,
            "structure": {"article_number": article_number} if article_number else {},
        },
    )


def test_explicit_document_and_article_match_outrank_raw_rank() -> None:
    """1. Existing explicit metadata exact match remains True."""
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
    assert scored[1].applicability == EvidenceApplicability.REFERENCE_MISMATCH
    assert scored[0].document_reference_match is True
    assert scored[0].article_reference_match is True


def test_metadata_mismatch_precedence_over_matching_title() -> None:
    """2. Explicit metadata mismatch remains False even when title would otherwise match."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "mismatch_meta",
        1,
        document_number="999/2020/NĐ-CP",
        document_title="Quyet-dinh-17-2023-QD-TTg-sua-doi-Quyet-dinh-31-2007-QD-TTg-tin-dung-doi-voi-ho-gia-dinh-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False
    assert scored[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_missing_metadata_own_title_recovery_q54485() -> None:
    """3. Case A: Missing metadata + recognized prefix + immediate own number -> True."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "q54485_hit",
        1,
        document_number=None,
        document_title="Quyet-dinh-17-2023-QD-TTg-sua-doi-Quyet-dinh-31-2007-QD-TTg-tin-dung-doi-voi-ho-gia-dinh-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is True
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH


def test_same_title_with_referenced_document_query_is_false() -> None:
    """4. Case B: CRITICAL: Query for referenced/amended document 31/2007/QĐ-TTg on 17/2023 title is False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["31/2007/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "q54485_hit",
        1,
        document_number=None,
        document_title="Quyet-dinh-17-2023-QD-TTg-sua-doi-Quyet-dinh-31-2007-QD-TTg-tin-dung-doi-voi-ho-gia-dinh-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False
    assert scored[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_adversarial_descriptive_prose_before_cited_number_is_false() -> None:
    """5. Case C: Recognized legal prefix but descriptive prose BEFORE cited number -> False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "adversarial_c",
        1,
        document_number=None,
        document_title="Huong-dan-thuc-hien-Quyet-dinh-17-2023-QD-TTg",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False
    assert scored[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_adversarial_non_number_prose_after_prefix_is_false() -> None:
    """6. Case D: Another recognized prefix but non-number prose before the number -> False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "adversarial_d",
        1,
        document_number=None,
        document_title="Quyet-dinh-ve-viec-thuc-hien-17-2023-QD-TTg",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False
    assert scored[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_adversarial_unrecognized_prefix_is_false() -> None:
    """7. Case E: Unrecognized prefix with legal-looking number -> False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "adversarial_e",
        1,
        document_number=None,
        document_title="Bao-cao-17-2023-QD-TTg-tong-ket",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False
    assert scored[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_optional_so_prefix_form_is_true() -> None:
    """8. Case F: Optional 'so' form immediately after recognized prefix -> True."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit_so = _hit(
        "hit_so",
        1,
        document_number=None,
        document_title="Quyet-dinh-so-17-2023-QD-TTg-sua-doi-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit_so])
    assert scored[0].document_reference_match is True
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH


def test_prefix_number_mismatch_is_false() -> None:
    """9. Case G: Query 117/2023/QĐ-TTg against 17/2023 title is False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["117/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "test_hit",
        1,
        document_number=None,
        document_title="Quyet-dinh-17-2023-QD-TTg-sua-doi-Quyet-dinh-31-2007-QD-TTg-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False


def test_organ_mismatch_is_false() -> None:
    """10. Case G: Query 17/2023/QĐ-BTC against 17/2023/QĐ-TTg title is False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-BTC"], article_numbers=[])

    hit = _hit(
        "test_hit",
        1,
        document_number=None,
        document_title="Quyet-dinh-17-2023-QD-TTg-sua-doi-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False


def test_year_mismatch_is_false() -> None:
    """11. Case G: Query 17/20230/QĐ-TTg against 17/2023/QĐ-TTg title is False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/20230/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "test_hit",
        1,
        document_number=None,
        document_title="Quyet-dinh-17-2023-QD-TTg-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False


def test_missing_metadata_and_unparseable_title_is_false() -> None:
    """12. Missing metadata and no parseable identity in title -> False."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "no_num_hit",
        1,
        document_number=None,
        document_title="Hien-phap-nuoc-Cong-hoa-xa-hoi-chu-nghia-Viet-Nam",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False


def test_body_citation_does_not_create_document_match() -> None:
    """13. Case H: A document number appearing ONLY in hit.text body does NOT match."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit = _hit(
        "body_only_hit",
        1,
        document_number=None,
        document_title="Quy-che-lam-viec-noi-bo",
        article_number=None,
        text="Căn cứ theo Quyết định số 17/2023/QĐ-TTg của Thủ tướng Chính phủ.",
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is False
    assert scored[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_present_malformed_metadata_fails_closed() -> None:
    """14. Case I: Present but malformed (non-string) metadata must fail closed and NOT use title fallback."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    # Raw metadata has int 17 instead of string, with matching document_title
    hit_int = _hit(
        "int_meta_hit",
        1,
        document_number=17,
        document_title="Quyet-dinh-17-2023-QD-TTg-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit_int])
    assert scored[0].document_reference_match is False
    assert scored[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH

    # Raw metadata has list instead of string
    hit_list = _hit(
        "list_meta_hit",
        1,
        document_number=["17/2023/QĐ-TTg"],
        document_title="Quyet-dinh-17-2023-QD-TTg-568644",
        article_number=None,
    )

    scored_list = selector.score(query, [hit_list])
    assert scored_list[0].document_reference_match is False
    assert scored_list[0].applicability == EvidenceApplicability.REFERENCE_MISMATCH


def test_blank_metadata_string_treated_as_absent() -> None:
    """15. Case J: Blank/whitespace metadata string is treated as absent and may use title fallback."""
    selector = EvidenceSelector()
    query = _query(document_numbers=["17/2023/QĐ-TTg"], article_numbers=[])

    hit_blank = _hit(
        "blank_meta_hit",
        1,
        document_number="   ",
        document_title="Quyet-dinh-17-2023-QD-TTg-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit_blank])
    assert scored[0].document_reference_match is True
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH


def test_no_document_references_in_query_retains_none() -> None:
    """16. Query with no document_numbers retains None document_reference_match."""
    selector = EvidenceSelector()
    query = _query(document_numbers=[], article_numbers=[])

    hit = _hit(
        "no_doc_query_hit",
        1,
        document_number=None,
        document_title="Quyet-dinh-17-2023-QD-TTg-568644",
        article_number=None,
    )

    scored = selector.score(query, [hit])
    assert scored[0].document_reference_match is None
    assert scored[0].applicability == EvidenceApplicability.COMPATIBLE


def test_diacritic_and_separator_normalization() -> None:
    """17. Diacritic, case, slash vs hyphen, and leading-zero normalization."""
    selector = EvidenceSelector()

    # Leading zero in query vs non-leading zero in title
    query_01 = _query(document_numbers=["01/2021/TT-BCA"], article_numbers=[])
    hit_1 = _hit(
        "hit1",
        1,
        document_number=None,
        document_title="Thong-tu-1-2021-TT-BCA-quy-dinh-ve-cu-tru-12345",
        article_number=None,
    )
    assert selector.score(query_01, [hit_1])[0].document_reference_match is True

    # Vietnamese diacritic NĐ-CP vs ND-CP
    query_nd = _query(document_numbers=["15/2022/NĐ-CP"], article_numbers=[])
    hit_nd = _hit(
        "hit_nd",
        1,
        document_number=None,
        document_title="Nghi-dinh-15-2022-ND-CP-chinh-sach-thue",
        article_number=None,
    )
    assert selector.score(query_nd, [hit_nd])[0].document_reference_match is True


def test_context_builder_and_grader_integration_q54485() -> None:
    """18. End-to-end integration test with ContextBuilder and RuleBasedContextGrader for Q54485."""
    # 1. Positive case: Query 17/2023/QĐ-TTg matches Q54485 hit
    query_54485 = RetrievalQuery(
        query_id="qid-54485",
        original_question="Quyết định số 17/2023/QĐ-TTg quy định thế nào?",
        normalized_question="quyết định số 17/2023/qđ-ttg quy định thế nào?",
        query_analysis=QueryAnalysis(document_numbers=["17/2023/QĐ-TTg"]),
        top_k=2,
    )

    hit_301729 = RetrievalHit(
        chunk_id="chunk-301729-1",
        document_id="301729",
        rank=1,
        score=9.5,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text="Quyết định này có hiệu lực thi hành từ ngày 01 tháng 9 năm 2023.",
        metadata={
            "document_title": "Quyet-dinh-17-2023-QD-TTg-sua-doi-Quyet-dinh-31-2007-QD-TTg-tin-dung-doi-voi-ho-gia-dinh-568644",
            "document_number": None,
            "effect_status": "còn hiệu lực",
            "structure": {},
        },
    )

    response = RetrievalResponse(
        query=query_54485,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        hits=[hit_301729],
    )

    builder = ContextBuilder(GenerationConfig(), EvidenceSelectionConfig())
    build_result = builder.build(response)

    assert len(build_result.evidence) == 1
    selected_ev = build_result.evidence[0]
    selection_trace = selected_ev.metadata["evidence_selection"]
    assert selection_trace["document_reference_match"] is True
    assert selection_trace["applicability"] == EvidenceApplicability.EXPLICIT_MATCH.value

    # Test RuleBasedContextGrader
    grader = RuleBasedContextGrader(ContextGradingConfig())
    grade = grader.grade(query_54485, build_result.evidence)
    assert grade.metadata["reference_coverage"]["document"] is True
    assert "document_reference_match" not in grade.missing_aspects
    assert grade.is_sufficient is True

    # 2. Negative safety case: Query 31/2007/QĐ-TTg on the same hit
    query_31_2007 = RetrievalQuery(
        query_id="qid-31-2007",
        original_question="Quyết định số 31/2007/QĐ-TTg quy định thế nào?",
        normalized_question="quyết định số 31/2007/qđ-ttg quy định thế nào?",
        query_analysis=QueryAnalysis(document_numbers=["31/2007/QĐ-TTg"]),
        top_k=2,
    )

    response_31 = RetrievalResponse(
        query=query_31_2007,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        hits=[hit_301729],
    )

    build_result_31 = builder.build(response_31)
    assert len(build_result_31.evidence) == 1
    selected_ev_31 = build_result_31.evidence[0]
    selection_trace_31 = selected_ev_31.metadata["evidence_selection"]
    assert selection_trace_31["document_reference_match"] is False
    assert selection_trace_31["applicability"] == EvidenceApplicability.REFERENCE_MISMATCH.value

    grade_31 = grader.grade(query_31_2007, build_result_31.evidence)
    assert grade_31.metadata["reference_coverage"]["document"] is False
    assert "document_reference_match" in grade_31.missing_aspects
    assert grade_31.is_sufficient is False


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
