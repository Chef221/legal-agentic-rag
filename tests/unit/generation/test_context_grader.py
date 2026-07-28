"""Tests for transparent structural context grading."""

from legal_agentic_rag.configuration import ContextGradingConfig
from legal_agentic_rag.contracts import ContextGrader
from legal_agentic_rag.generation import RuleBasedContextGrader
from legal_agentic_rag.schemas import Evidence, QueryAnalysis, RetrievalQuery


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="grade-query",
        original_question="Câu hỏi",
        normalized_question="câu hỏi",
        top_k=2,
        candidate_k=2,
    )


def _evidence(index: int, *, article: str | None = "1") -> Evidence:
    return Evidence(
        evidence_id=f"E{index}",
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        text="Nội dung pháp luật.",
        article_number=article,
        document_number="1/2026/QH",
    )


def test_structural_grader_marks_minimum_context_and_discloses_limitation() -> None:
    """A sufficient structural grade never claims semantic relevance checking."""
    grade = RuleBasedContextGrader().grade(_query(), [_evidence(1)])

    assert grade.is_sufficient is True
    assert grade.score == 1.0
    assert "semantic_relevance_not_verified" in grade.warnings
    assert grade.metadata["semantic_relevance_checked"] is False
    assert isinstance(RuleBasedContextGrader(), ContextGrader)


def test_structural_grader_abstains_for_missing_required_evidence_metadata() -> None:
    """Configurable minimum count and article identity fail closed."""
    grader = RuleBasedContextGrader(
        ContextGradingConfig(
            minimum_evidence_count=2,
            require_article_number=True,
        )
    )

    grade = grader.grade(_query(), [_evidence(1, article=None)])

    assert grade.is_sufficient is False
    assert grade.missing_aspects == [
        "minimum_evidence_count",
        "article_number",
    ]
    assert grade.coverage_score == 0.5


def test_grader_requires_user_supplied_reference_to_match_selected_evidence() -> None:
    """An explicit document/article request fails closed when selection mismatches."""
    query = _query().model_copy(
        update={
            "query_analysis": QueryAnalysis(
                document_numbers=["45/2019/QH14"],
                article_numbers=["113"],
            )
        }
    )
    evidence = _evidence(1).model_copy(
        update={
            "metadata": {
                "evidence_selection": {
                    "applicability": "reference_mismatch",
                    "document_reference_match": False,
                    "article_reference_match": False,
                    "lexical_overlap_score": 0.5,
                }
            }
        }
    )

    grade = RuleBasedContextGrader().grade(query, [evidence])

    assert grade.is_sufficient is False
    assert "document_reference_match" in grade.missing_aspects
    assert "article_reference_match" in grade.missing_aspects
    assert "applicable_evidence" in grade.missing_aspects
    assert grade.applicability_score == 0.0
    assert grade.metadata["legal_applicability_interpreted"] is False
