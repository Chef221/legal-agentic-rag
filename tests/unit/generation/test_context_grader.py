"""Tests for transparent structural context grading."""

from legal_agentic_rag.configuration import ContextGradingConfig
from legal_agentic_rag.contracts import ContextGrader
from legal_agentic_rag.generation import RuleBasedContextGrader
from legal_agentic_rag.schemas import Evidence, RetrievalQuery


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
