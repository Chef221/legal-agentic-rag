"""Unit tests for deterministic legal-query analysis and variant planning."""

from legal_agentic_rag.configuration import QueryUnderstandingConfig
from legal_agentic_rag.retrieval import QueryUnderstandingService
from legal_agentic_rag.schemas import (
    QueryIntent,
    QueryVariantKind,
    RetrievalQuery,
)


def _query(question: str) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-understanding",
        original_question=question,
        normalized_question=question,
    )


def test_query_understanding_extracts_only_explicit_legal_signals() -> None:
    """References, scope, time, and intent come directly from the question."""
    question = (
        "Xin hỏi: Theo Điều 113 khoản 1 của 45/2019/QH14, "
        "trong trường hợp này thời hạn bao lâu?"
    )

    enriched = QueryUnderstandingService().enrich(_query(question))

    analysis = enriched.query_analysis
    assert analysis is not None
    assert analysis.intent == QueryIntent.QUANTITATIVE
    assert analysis.document_numbers == ["45/2019/QH14"]
    assert analysis.article_numbers == ["113"]
    assert analysis.clause_numbers == ["1"]
    assert analysis.year_mentions == ["2019"]
    assert analysis.scope_cues == ["trong trường hợp"]
    assert [item.kind for item in enriched.query_variants] == [
        QueryVariantKind.NORMALIZED,
        QueryVariantKind.FRAMING_STRIPPED,
        QueryVariantKind.LEGAL_REFERENCE,
    ]
    assert enriched.query_variants[1].text.startswith("Theo Điều 113")
    assert enriched.query_variants[2].text == (
        "Điều 113 khoản 1 45/2019/QH14"
    )


def test_query_understanding_detects_relationship_intent_without_rewriting() -> None:
    """Relationship routing signals do not add a legal term to the query."""
    question = "Văn bản nào sửa đổi và thay thế quy định này?"

    enriched = QueryUnderstandingService().enrich(_query(question))

    assert enriched.query_analysis is not None
    assert enriched.query_analysis.intent == QueryIntent.RELATIONSHIP
    assert enriched.query_analysis.relationship_cues == ["sửa đổi", "thay thế"]
    assert [item.text for item in enriched.query_variants] == [question]


def test_query_understanding_can_be_disabled_and_bounds_variants() -> None:
    """Configuration can preserve the legacy query and cap derived forms."""
    question = "Xin hỏi: Điều 10 của 45/2019/QH14 quy định gì?"
    disabled = QueryUnderstandingService(
        QueryUnderstandingConfig(enabled=False)
    ).enrich(_query(question))
    bounded = QueryUnderstandingService(
        QueryUnderstandingConfig(max_variants=2)
    ).enrich(_query(question))

    assert disabled.query_analysis is None
    assert disabled.query_variants == []
    assert len(bounded.query_variants) == 2
