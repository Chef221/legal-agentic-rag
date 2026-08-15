"""Unit tests for deterministic verifier-supported claim salvage."""

from legal_agentic_rag.generation.claim_salvage import build_supported_claim_salvage
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    ClaimVerification,
    RetrievalStrategy,
)


def _response() -> AnswerResponse:
    return AnswerResponse(
        question="Cau hoi kiem thu?",
        answer="Dieu khoan mot. [E1] Dieu khoan hai 05 ngay. [E2]",
        citations=[
            Citation(evidence_id="E1", chunk_id="chunk-1", document_id="doc-1"),
            Citation(evidence_id="E2", chunk_id="chunk-2", document_id="doc-2"),
        ],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        trace_id="trace-test",
    )


def test_salvage_preserves_supported_claim_text_and_citation_order() -> None:
    """Salvage must preserve the exact supported numeric token, including zeroes."""
    result = build_supported_claim_salvage(
        _response(),
        CitationVerificationResult(
            is_valid=False,
            claim_verifications=[
                ClaimVerification(
                    claim_id="C1",
                    claim_text="Dieu khoan mot.",
                    evidence_ids=["E1"],
                    status=ClaimSupportStatus.SUPPORTED,
                    lexical_support_score=1.0,
                    numeric_match=True,
                    negation_match=True,
                ),
                ClaimVerification(
                    claim_id="C2",
                    claim_text="Dieu khoan hai 05 ngay.",
                    evidence_ids=["E2"],
                    status=ClaimSupportStatus.SUPPORTED,
                    lexical_support_score=1.0,
                    numeric_match=True,
                    negation_match=True,
                ),
                ClaimVerification(
                    claim_id="C3",
                    claim_text="Dieu khoan sai 07 ngay.",
                    evidence_ids=["E2"],
                    status=ClaimSupportStatus.UNSUPPORTED,
                    lexical_support_score=0.5,
                    numeric_match=False,
                    negation_match=True,
                    errors=["numeric_mismatch"],
                ),
            ],
            claim_coverage_score=2 / 3,
            claim_level_verification_performed=True,
            errors=["unsupported_claim:C3"],
        ),
    )

    assert result.outcome == "candidate_ready"
    assert result.retained_claim_count == 2
    assert result.dropped_claim_count == 1
    assert result.response is not None
    assert result.response.answer == "Dieu khoan mot [E1]. Dieu khoan hai 05 ngay [E2]."
    assert [citation.evidence_id for citation in result.response.citations] == ["E1", "E2"]


def test_salvage_rejects_missing_citation_mapping() -> None:
    """A verifier claim without a matching response citation cannot be repaired."""
    result = build_supported_claim_salvage(
        _response(),
        CitationVerificationResult(
            is_valid=False,
            claim_verifications=[
                ClaimVerification(
                    claim_id="C1",
                    claim_text="Dieu khoan mot.",
                    evidence_ids=["E9"],
                    status=ClaimSupportStatus.SUPPORTED,
                    lexical_support_score=1.0,
                    numeric_match=True,
                    negation_match=True,
                ),
                ClaimVerification(
                    claim_id="C2",
                    claim_text="Dieu khoan sai 07 ngay.",
                    evidence_ids=["E2"],
                    status=ClaimSupportStatus.UNSUPPORTED,
                    lexical_support_score=0.5,
                    numeric_match=False,
                    negation_match=True,
                    errors=["numeric_mismatch"],
                ),
            ],
            claim_coverage_score=0.5,
            claim_level_verification_performed=True,
            errors=["unsupported_claim:C2"],
        ),
    )

    assert result.response is None
    assert result.outcome == "contract_mismatch"


def test_salvage_skips_when_no_supported_claim_exists() -> None:
    """No verifier-supported content means the model fallback remains responsible."""
    result = build_supported_claim_salvage(
        _response(),
        CitationVerificationResult(
            is_valid=False,
            claim_verifications=[
                ClaimVerification(
                    claim_id="C1",
                    claim_text="Dieu khoan sai 07 ngay.",
                    evidence_ids=["E2"],
                    status=ClaimSupportStatus.UNSUPPORTED,
                    lexical_support_score=0.5,
                    numeric_match=False,
                    negation_match=True,
                    errors=["numeric_mismatch"],
                )
            ],
            claim_coverage_score=0.0,
            claim_level_verification_performed=True,
            errors=["unsupported_claim:C1"],
        ),
    )

    assert result.response is None
    assert result.outcome == "not_applicable_no_supported_claim"


def test_salvage_reports_negation_mismatch_without_rewriting_claims() -> None:
    """A supported-only response can safely omit a verifier-rejected negation."""
    result = build_supported_claim_salvage(
        _response(),
        CitationVerificationResult(
            is_valid=False,
            claim_verifications=[
                ClaimVerification(
                    claim_id="C1",
                    claim_text="Dieu khoan mot.",
                    evidence_ids=["E1"],
                    status=ClaimSupportStatus.SUPPORTED,
                    lexical_support_score=1.0,
                    numeric_match=True,
                    negation_match=True,
                ),
                ClaimVerification(
                    claim_id="C2",
                    claim_text="Dieu khoan hai khong ap dung.",
                    evidence_ids=["E2"],
                    status=ClaimSupportStatus.UNSUPPORTED,
                    lexical_support_score=1.0,
                    numeric_match=True,
                    negation_match=False,
                    errors=["negation_mismatch"],
                ),
            ],
            claim_coverage_score=0.5,
            claim_level_verification_performed=True,
            errors=["unsupported_claim:C2"],
        ),
    )

    assert result.outcome == "candidate_ready"
    assert result.dropped_error_counts == {"negation_mismatch": 1}
    assert result.response is not None
    assert result.response.answer == "Dieu khoan mot [E1]."
