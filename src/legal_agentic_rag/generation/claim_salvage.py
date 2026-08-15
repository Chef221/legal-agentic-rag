"""Deterministically salvage verifier-supported claims after numeric rejection."""

from __future__ import annotations

from dataclasses import dataclass

from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
)


@dataclass(frozen=True)
class NumericClaimSalvageResult:
    """One bounded, content-preserving salvage attempt outcome."""

    response: AnswerResponse | None
    retained_claim_count: int
    dropped_claim_count: int
    outcome: str


def build_numeric_claim_salvage(
    response: AnswerResponse,
    verification: CitationVerificationResult,
) -> NumericClaimSalvageResult:
    """Keep only exactly verifier-supported claims and their existing citations.

    The function never rewrites claim text, invents a marker, or calls a model.
    A missing or ambiguous citation mapping is a contract failure, not a reason to
    guess a replacement citation.
    """
    supported_claims = [
        claim
        for claim in verification.claim_verifications
        if claim.status is ClaimSupportStatus.SUPPORTED
    ]
    dropped_claim_count = len(verification.claim_verifications) - len(
        supported_claims
    )
    if not supported_claims:
        return NumericClaimSalvageResult(
            response=None,
            retained_claim_count=0,
            dropped_claim_count=dropped_claim_count,
            outcome="not_applicable_no_supported_claim",
        )

    citations_by_evidence_id: dict[str, Citation] = {}
    for citation in response.citations:
        if citation.evidence_id in citations_by_evidence_id:
            return NumericClaimSalvageResult(
                response=None,
                retained_claim_count=len(supported_claims),
                dropped_claim_count=dropped_claim_count,
                outcome="contract_mismatch",
            )
        citations_by_evidence_id[citation.evidence_id] = citation

    evidence_ids: list[str] = []
    rendered_claims: list[str] = []
    for claim in supported_claims:
        if not claim.evidence_ids or any(
            evidence_id not in citations_by_evidence_id
            for evidence_id in claim.evidence_ids
        ):
            return NumericClaimSalvageResult(
                response=None,
                retained_claim_count=len(supported_claims),
                dropped_claim_count=dropped_claim_count,
                outcome="contract_mismatch",
            )
        rendered_claims.append(_render_claim(claim.claim_text, claim.evidence_ids))
        for evidence_id in claim.evidence_ids:
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)

    return NumericClaimSalvageResult(
        response=response.model_copy(
            update={
                "answer": " ".join(rendered_claims),
                "citations": [
                    citations_by_evidence_id[evidence_id]
                    for evidence_id in evidence_ids
                ],
                "insufficient_evidence": False,
            }
        ),
        retained_claim_count=len(supported_claims),
        dropped_claim_count=dropped_claim_count,
        outcome="candidate_ready",
    )


def _render_claim(claim_text: str, evidence_ids: list[str]) -> str:
    """Render unchanged claim text with the exact verifier-linked markers."""
    markers = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
    text = claim_text.rstrip()
    if text[-1] in ".!?;":
        return f"{text[:-1].rstrip()} {markers}{text[-1]}"
    return f"{text} {markers}"
