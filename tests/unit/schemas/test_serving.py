"""Schema tests for public serving requests and errors."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas import (
    ApiErrorDetail,
    ApiErrorResponse,
    LegalQuestionRequest,
    RetrievalStrategy,
)


def test_legal_question_request_preserves_vietnamese_text() -> None:
    """Validation trims boundaries without case-folding or removing accents."""
    request = LegalQuestionRequest(
        question="  Doanh nghiệp không phải nộp thuế khi nào?  ",
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )

    assert request.question == "Doanh nghiệp không phải nộp thuế khi nào?"
    assert request.requested_strategy == RetrievalStrategy.HYBRID_RERANK


@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_legal_question_request_rejects_blank_text(question: str) -> None:
    """Blank input never reaches retrieval or generation."""
    with pytest.raises(ValidationError, match="question"):
        LegalQuestionRequest(question=question)


def test_legal_question_request_rejects_impossible_limits() -> None:
    """An explicit candidate set must be large enough for final results."""
    with pytest.raises(ValidationError, match="candidate_k"):
        LegalQuestionRequest(question="Câu hỏi", top_k=10, candidate_k=5)


def test_legal_question_request_hides_internal_rerank_stage() -> None:
    """The API exposes complete strategies, not an implementation-only stage."""
    with pytest.raises(ValidationError, match="internal stage"):
        LegalQuestionRequest(
            question="Câu hỏi",
            requested_strategy=RetrievalStrategy.RERANK,
        )


def test_api_error_has_one_stable_envelope() -> None:
    """Clients receive a machine-readable category and safe message."""
    response = ApiErrorResponse(
        error=ApiErrorDetail(
            error_type="retrieval_error",
            message="Legal evidence retrieval could not be completed.",
        )
    )

    assert response.model_dump()["error"]["trace_id"] is None
