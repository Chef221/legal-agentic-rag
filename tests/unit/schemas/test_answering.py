"""Tests for evidence, answer, citation, and verification contracts."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    ClaimVerification,
    ContextBuildResult,
    ContextGrade,
    Evidence,
    EvidenceApplicability,
    EvidenceSelectionReason,
    EvidenceSelectionTrace,
    ModelAnswerClaimDraft,
    ModelAnswerDraft,
    SemanticClaimAssessmentDraft,
    SemanticClaimVerification,
    SemanticSupportLabel,
    SemanticVerificationDraft,
    SemanticVerificationResult,
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


def test_model_answer_draft_requires_explicit_claim_level_evidence() -> None:
    """A grounded model draft links every claim to at least one evidence ID."""
    draft = ModelAnswerDraft(
        claims=[
            ModelAnswerClaimDraft(
                text="Người lao động được nghỉ hằng năm.",
                evidence_ids=["E1"],
            )
        ],
        insufficient_evidence=False,
    )

    assert draft.claims[0].evidence_ids == ["E1"]
    with pytest.raises(ValidationError):
        ModelAnswerDraft(
            claims=[],
            insufficient_evidence=False,
        )


def test_model_answer_draft_abstention_contains_no_claims() -> None:
    """An insufficient draft cannot smuggle a grounded claim into output."""
    with pytest.raises(ValidationError):
        ModelAnswerDraft(
            claims=[
                ModelAnswerClaimDraft(
                    text="Một nhận định.",
                    evidence_ids=["E1"],
                )
            ],
            insufficient_evidence=True,
        )


def test_model_answer_draft_bounds_claim_count_and_length() -> None:
    """Structured generation stays within the compact M49.1 output budget."""
    claim = {"text": "Một nhận định pháp lý.", "evidence_ids": ["E1"]}

    with pytest.raises(ValidationError):
        ModelAnswerDraft(
            claims=[claim] * 5,
            insufficient_evidence=False,
        )
    with pytest.raises(ValidationError):
        ModelAnswerDraft(
            claims=[{"text": "ấ" * 601, "evidence_ids": ["E1"]}],
            insufficient_evidence=False,
        )


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


def test_claim_verification_contract_aligns_support_and_coverage() -> None:
    """Claim records explain failure and require performed result metadata."""
    supported = ClaimVerification(
        claim_id="C1",
        claim_text="Người lao động được nghỉ 12 ngày.",
        evidence_ids=["E1"],
        status=ClaimSupportStatus.SUPPORTED,
        lexical_support_score=0.8,
        numeric_match=True,
        negation_match=True,
    )
    result = CitationVerificationResult(
        is_valid=True,
        valid_citations=[_citation()],
        claim_verifications=[supported],
        claim_coverage_score=1.0,
        claim_level_verification_performed=True,
    )

    assert result.claim_coverage_score == 1.0
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="C2",
            claim_text="Nhận định thiếu căn cứ.",
            evidence_ids=[],
            status=ClaimSupportStatus.UNSUPPORTED,
            lexical_support_score=0.0,
            numeric_match=True,
            negation_match=True,
        )
    with pytest.raises(ValidationError):
        CitationVerificationResult(
            is_valid=True,
            claim_verifications=[supported],
            claim_coverage_score=1.0,
        )


def test_semantic_verification_contract_is_strict_and_provenanced() -> None:
    """Semantic output separates untrusted model labels from trusted links."""
    claim = ClaimVerification(
        claim_id="C1",
        claim_text="Người lao động được nghỉ 12 ngày.",
        evidence_ids=["E1"],
        status=ClaimSupportStatus.SUPPORTED,
        lexical_support_score=1.0,
        numeric_match=True,
        negation_match=True,
    )
    draft = SemanticVerificationDraft(
        assessments=[
            SemanticClaimAssessmentDraft(
                claim_id="C1",
                label=SemanticSupportLabel.SUPPORTED,
            )
        ]
    )
    semantic = SemanticVerificationResult(
        is_valid=True,
        assessments=[
            SemanticClaimVerification(
                claim_id="C1",
                evidence_ids=["E1"],
                label=SemanticSupportLabel.SUPPORTED,
            )
        ],
        provider_name="fixture",
        provider_version="1.0",
        model_name="fixture-model",
        model_revision="fixture-revision",
    )
    result = CitationVerificationResult(
        is_valid=True,
        claim_verifications=[claim],
        claim_coverage_score=1.0,
        claim_level_verification_performed=True,
        semantic_verification=semantic,
    )

    assert draft.assessments[0].claim_id == "C1"
    assert result.semantic_verification is not None
    with pytest.raises(ValidationError):
        SemanticVerificationDraft(
            assessments=[
                {"claim_id": "C1", "label": "supported"},
                {"claim_id": "C1", "label": "insufficient"},
            ]
        )
    with pytest.raises(ValidationError):
        SemanticVerificationResult(
            is_valid=True,
            assessments=[
                {
                    "claim_id": "C1",
                    "evidence_ids": ["E1"],
                    "label": "contradicted",
                }
            ],
            provider_name="fixture",
            provider_version="1.0",
            model_name="fixture-model",
            model_revision="fixture-revision",
        )
    with pytest.raises(ValidationError):
        CitationVerificationResult(
            is_valid=False,
            errors=["semantic_contradicted:C1"],
            claim_verifications=[claim],
            claim_coverage_score=1.0,
            claim_level_verification_performed=True,
            semantic_verification=semantic,
        )
