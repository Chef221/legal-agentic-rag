"""Rule-based structural and referential citation verification."""

from __future__ import annotations

from collections.abc import Sequence

from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    Evidence,
)


class RuleBasedCitationVerifier:
    """Verify every citation against the exact supplied evidence identity."""

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
        return CitationVerificationResult(
            is_valid=not invalid and not errors,
            valid_citations=valid,
            invalid_citations=invalid,
            errors=errors,
            warnings=["semantic_claim_verification_not_performed"],
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
