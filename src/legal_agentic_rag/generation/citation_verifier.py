"""Rule-based structural and referential citation verification."""

from __future__ import annotations

from collections.abc import Sequence

from legal_agentic_rag.configuration.online import ClaimVerificationConfig
from legal_agentic_rag.generation.claim_grounding import (
    RuleBasedClaimGroundingVerifier,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    Evidence,
)


class RuleBasedCitationVerifier:
    """Verify citation identity and deterministic claim-level grounding."""

    def __init__(
        self,
        config: ClaimVerificationConfig | None = None,
    ) -> None:
        self._claim_grounding = RuleBasedClaimGroundingVerifier(config)

    def verify(
        self,
        response: AnswerResponse,
        evidence: Sequence[Evidence],
    ) -> CitationVerificationResult:
        """Classify valid and invalid citations without semantic claim checking."""
        evidence_values = list(evidence)
        evidence_by_id = {item.evidence_id: item for item in evidence_values}
        errors: list[str] = []
        valid: list[Citation] = []
        invalid: list[Citation] = []
        if len(evidence_by_id) != len(evidence_values):
            errors.append("duplicate_evidence_id")
        if len({item.chunk_id for item in evidence_values}) != len(evidence_values):
            errors.append("duplicate_evidence_chunk_id")
        if response.insufficient_evidence and response.citations:
            errors.append("abstention_must_not_include_citations")
        if not response.insufficient_evidence and not response.citations:
            errors.append("grounded_answer_requires_citation")
        for citation in response.citations:
            source = evidence_by_id.get(citation.evidence_id)
            if source is None or not self._matches(citation, source):
                invalid.append(citation)
            else:
                valid.append(citation)
        (
            claim_verifications,
            claim_coverage,
            claim_errors,
            claim_warnings,
        ) = self._claim_grounding.verify(response, evidence_values)
        errors.extend(claim_errors)
        warnings = list(claim_warnings)
        if not claim_verifications:
            warnings.append("semantic_claim_verification_not_performed")
        return CitationVerificationResult(
            is_valid=not invalid and not errors,
            valid_citations=valid,
            invalid_citations=invalid,
            claim_verifications=claim_verifications,
            claim_coverage_score=claim_coverage,
            claim_level_verification_performed=bool(claim_verifications),
            errors=errors,
            warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _matches(citation: Citation, evidence: Evidence) -> bool:
        return (
            citation.chunk_id == evidence.chunk_id
            and citation.document_id == evidence.document_id
            and citation.document_title == evidence.document_title
            and citation.document_number == evidence.document_number
            and citation.article_number == evidence.article_number
            and citation.source_url == evidence.source_url
        )
