"""Tests for evidence, answer, citation, and verification contracts."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ContextBuildResult,
    ContextGrade,
    Evidence,
    EvidenceApplicability,
    EvidenceSelectionReason,
    EvidenceSelectionTrace,
)


def _citation() -> Citation:
    return Citation(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        article_number="1",
    )


def test_answer_response_parses_fixture(load_schema_sample: object) -> None:
    """A grounded response preserves citation and retrieval provenance."""
    data = load_schema_sample("valid_answer_response.json")  # type: ignore[operator]
    response = AnswerResponse.model_validate(data)
    assert response.insufficient_evidence is False
    assert response.citations[0].evidence_id == "E1"


def test_answer_response_requires_explicit_sufficiency_flag() -> None:
    """Omitting the safety-critical abstention flag is invalid."""
    with pytest.raises(ValidationError):
        AnswerResponse(
            question="Câu hỏi",
            answer="Câu trả lời",
            retrieval_strategy="bm25",
            trace_id="trace-1",
        )


def test_evidence_id_uses_citation_pattern() -> None:
    """Evidence identifiers are deterministic citation handles."""
    evidence = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Điều 1. Nội dung.",
    )
    assert evidence.evidence_id == "E1"

    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence-one",
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Nội dung",
        )


def test_context_scores_are_bounded() -> None:
    """Grader scores stay within the documented zero-to-one range."""
    with pytest.raises(ValidationError):
        ContextGrade(is_sufficient=True, score=1.1)


def test_evidence_selection_trace_aligns_with_context_evidence() -> None:
    """Selection trace has a typed reason and matches selected evidence order."""
    evidence = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Nội dung.",
    )
    trace = EvidenceSelectionTrace(
        chunk_id="chunk-1",
        source_rank=2,
        selection_rank=1,
        applicability=EvidenceApplicability.EXPLICIT_MATCH,
        document_reference_match=True,
        lexical_overlap_score=0.5,
        selection_score=3.0,
        selected=True,
        reason=EvidenceSelectionReason.SELECTED,
    )

    result = ContextBuildResult(
        evidence=[evidence],
        input_hit_count=1,
        selected_count=1,
        omitted_hit_count=0,
        duplicate_hit_count=0,
        estimated_token_count=2,
        selection_trace=[trace],
    )

    assert result.selection_trace[0].selection_rank == 1
    with pytest.raises(ValidationError):
        EvidenceSelectionTrace(
            chunk_id="chunk-1",
            source_rank=1,
            applicability=EvidenceApplicability.UNKNOWN,
            lexical_overlap_score=0.0,
            selection_score=1.0,
            selected=True,
            reason=EvidenceSelectionReason.SELECTED,
        )


def test_citation_verification_result_is_consistent() -> None:
    """Invalid citations force a failed verification result."""
    valid = CitationVerificationResult(is_valid=True, valid_citations=[_citation()])
    assert valid.errors == []

    failed = CitationVerificationResult(
        is_valid=False,
        invalid_citations=[_citation()],
        errors=["Evidence ID not found"],
    )
    assert failed.is_valid is False

    with pytest.raises(ValidationError):
        CitationVerificationResult(
            is_valid=True,
            invalid_citations=[_citation()],
        )
