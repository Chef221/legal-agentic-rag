"""Structured semantic verification with strict contradiction overlay (Candidate V2-D3.2).

This module implements Candidate V2-D3.2: Frozen D3 Base + Strict Contradiction Confirmation Overlay.
Unlike previous models, V2-D3.2 is an asymmetric multi-stage verifier that uses the frozen D3
semantic verifier as its base predictor and applies an independent, strict contradiction confirmation
filter to override false negative / undercalled contradiction claims.

Architecture (Per Claim):
- Call A (Frozen D3 Base): Evaluates D3 relation (ENTAILS, CONTRADICTS, DOES_NOT_ESTABLISH).
- Call B (Strict Conflict Filter): Evaluates same_material_proposition and cannot_both_be_true.
- Trusted Combination:
  * If conflict check confirms (same_material_proposition=True and cannot_both_be_true=True):
      final_label = SemanticSupportLabel.CONTRADICTED (Override applied)
  * Else:
      final_label = base_d3_label (D3 base label preserved)

No other overrides are permitted. D3's high supported retention (17/18) and support/insufficient
calibration are strictly preserved unless the strict conflict test triggers.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
from legal_agentic_rag.generation.structured_semantic_verifier_d3 import (
    D3StructuredClaimAssessmentDraft,
    StructuredSemanticCitationVerifierD3,
    derive_claim_semantic_label_d3 as derive_d3_base_label,
)
from legal_agentic_rag.generation.structured_semantic_verifier_d32_conflict import (
    D32ConflictTelemetry,
    StrictConflictStatus,
    StructuredClaimConflictAssessmentD32,
    StructuredSemanticConflictDraftD32,
    StructuredSemanticVerifierD32Conflict,
    derive_strict_conflict_status_d32,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    CitationVerificationResult,
    Evidence,
    SemanticClaimVerification,
    SemanticSupportLabel,
    SemanticVerificationResult,
)

_LOGGER = logging.getLogger(__name__)


class D32ClaimVerificationTelemetry(BaseModel):
    """Operational telemetry for verifying a single claim under V2-D3.2."""

    claim_id: str
    d3_base_calls: int = 0
    d3_base_retries: int = 0
    d3_base_rejections: list[str] = Field(default_factory=list)
    conflict_calls: int = 0
    conflict_retries: int = 0
    conflict_rejections: list[str] = Field(default_factory=list)
    total_provider_calls: int = 0
    semantic_execution_error: bool = False
    override_applied: bool = False


class StructuredClaimVerificationD32(BaseModel):
    """Validated structured claim assessment for V2-D3.2."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    evidence_ids: list[str]
    base_d3_label: SemanticSupportLabel
    final_label: SemanticSupportLabel
    override_applied: bool
    d3_assessment: D3StructuredClaimAssessmentDraft | None = None
    conflict_assessment: StructuredClaimConflictAssessmentD32 | None = None
    telemetry: D32ClaimVerificationTelemetry | None = None


class StructuredSemanticVerificationResultD32(BaseModel):
    """Top-level structured verification result across all claims in an answer under V2-D3.2."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    assessments: list[StructuredClaimVerificationD32] = Field(default_factory=list)
    provider_name: str
    provider_version: str
    model_name: str
    model_revision: str
    errors: list[str] = Field(default_factory=list)
    execution_error_claims: list[str] = Field(default_factory=list)
    claim_telemetries: dict[str, D32ClaimVerificationTelemetry] = Field(default_factory=dict)
    overridden_claims_count: int = 0


class StructuredSemanticCitationVerifierD32:
    """V2-D3.2 Frozen D3 Base + Strict Contradiction Confirmation Overlay."""

    def __init__(
        self,
        provider: ChatModelProvider,
        *,
        max_structured_output_retries: int = 1,
    ) -> None:
        self._provider = provider
        self._max_retries = max(0, int(max_structured_output_retries))
        self._rule_verifier = RuleBasedCitationVerifier()
        self._d3_base_verifier = StructuredSemanticCitationVerifierD3(
            provider, max_structured_output_retries=self._max_retries
        )
        self._conflict_verifier = StructuredSemanticVerifierD32Conflict(
            provider, max_structured_output_retries=self._max_retries
        )

    def verify(
        self,
        response: AnswerResponse,
        evidence: Sequence[Evidence],
    ) -> CitationVerificationResult:
        """Verify citations and augment metadata with standard semantic result."""
        base_result, structured_result = self.verify_structured(response, evidence)

        if not structured_result.assessments and not structured_result.execution_error_claims:
            return base_result

        semantic_assessments: list[SemanticClaimVerification] = [
            SemanticClaimVerification(
                claim_id=a.claim_id,
                evidence_ids=a.evidence_ids,
                label=a.final_label,
            )
            for a in structured_result.assessments
        ]

        semantic_res = SemanticVerificationResult(
            is_valid=structured_result.is_valid,
            assessments=semantic_assessments,
            provider_name=structured_result.provider_name,
            provider_version=structured_result.provider_version,
            model_name=structured_result.model_name,
            model_revision=structured_result.model_revision,
            errors=list(structured_result.errors),
        )

        meta = dict(base_result.metadata)
        meta["semantic_verification"] = semantic_res.model_dump()
        meta["v2_d32_structured_verification"] = structured_result.model_dump()

        return base_result.model_copy(
            update={
                "is_valid": structured_result.is_valid,
                "metadata": meta,
                "errors": list(base_result.errors) + list(structured_result.errors),
            }
        )

    def verify_structured(
        self,
        response: AnswerResponse,
        evidence: Sequence[Evidence],
    ) -> tuple[CitationVerificationResult, StructuredSemanticVerificationResultD32]:
        """Verify cited evidence entailment per-claim with two independent model evaluations."""
        evidence_values = list(evidence)
        base_result = self._rule_verifier.verify(response, evidence_values)

        if not base_result.claim_verifications:
            empty_structured = StructuredSemanticVerificationResultD32(
                is_valid=base_result.is_valid,
                assessments=[],
                provider_name=self._provider.provider_name,
                provider_version=self._provider.provider_version,
                model_name=self._provider.model_name,
                model_revision=self._provider.model_revision,
                errors=list(base_result.errors),
                execution_error_claims=[],
                claim_telemetries={},
                overridden_claims_count=0,
            )
            return base_result, empty_structured

        evidence_by_id = {item.evidence_id: item for item in evidence_values}
        assessments: list[StructuredClaimVerificationD32] = []
        errors: list[str] = []
        execution_error_claims: list[str] = []
        claim_telemetries: dict[str, D32ClaimVerificationTelemetry] = {}
        overridden_count = 0

        for claim in base_result.claim_verifications:
            linked_evidence = [
                evidence_by_id[eid]
                for eid in claim.evidence_ids
                if eid in evidence_by_id
            ]

            # 1. Call A: Frozen D3 Base Verification
            d3_draft, d3_telem = self._d3_base_verifier._verify_single_claim(
                question=response.question,
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                evidence=linked_evidence,
            )

            # 2. Call B: Strict Contradiction Confirmation
            conflict_draft, conflict_telem = self._conflict_verifier.evaluate_conflict(
                question=response.question,
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                evidence=linked_evidence,
            )

            # Combine Telemetry
            total_calls = d3_telem.provider_call_count + conflict_telem.provider_call_count
            has_error = d3_telem.semantic_execution_error or conflict_telem.semantic_execution_error

            telemetry = D32ClaimVerificationTelemetry(
                claim_id=claim.claim_id,
                d3_base_calls=d3_telem.provider_call_count,
                d3_base_retries=d3_telem.retry_count,
                d3_base_rejections=list(d3_telem.draft_rejection_categories),
                conflict_calls=conflict_telem.provider_call_count,
                conflict_retries=conflict_telem.retry_count,
                conflict_rejections=list(conflict_telem.draft_rejection_categories),
                total_provider_calls=total_calls,
                semantic_execution_error=has_error,
                override_applied=False,
            )
            claim_telemetries[claim.claim_id] = telemetry

            if has_error or d3_draft is None or conflict_draft is None:
                execution_error_claims.append(claim.claim_id)
                errors.append(
                    f"Semantic verification execution error on claim '{claim.claim_id}' in D3.2"
                )
                assessments.append(
                    StructuredClaimVerificationD32(
                        claim_id=claim.claim_id,
                        evidence_ids=claim.evidence_ids,
                        base_d3_label=SemanticSupportLabel.INSUFFICIENT,
                        final_label=SemanticSupportLabel.INSUFFICIENT,
                        override_applied=False,
                        d3_assessment=d3_draft,
                        conflict_assessment=None,
                        telemetry=telemetry,
                    )
                )
                continue

            # Both drafts succeeded
            base_label = derive_d3_base_label(d3_draft)
            conflict_status = derive_strict_conflict_status_d32(conflict_draft)

            conflict_assessment = StructuredClaimConflictAssessmentD32(
                claim_id=claim.claim_id,
                status=conflict_status,
                same_material_proposition=conflict_draft.same_material_proposition,
                cannot_both_be_true=conflict_draft.cannot_both_be_true,
                telemetry=conflict_telem,
            )

            # Asymmetric Override Rule:
            # Override TO CONTRADICTED if and only if strict conflict confirmed
            if conflict_status == StrictConflictStatus.STRICT_CONTRADICTION_CONFIRMED:
                final_label = SemanticSupportLabel.CONTRADICTED
                override_applied = (base_label != SemanticSupportLabel.CONTRADICTED)
                if override_applied:
                    overridden_count += 1
            else:
                final_label = base_label
                override_applied = False

            telemetry.override_applied = override_applied

            assessments.append(
                StructuredClaimVerificationD32(
                    claim_id=claim.claim_id,
                    evidence_ids=claim.evidence_ids,
                    base_d3_label=base_label,
                    final_label=final_label,
                    override_applied=override_applied,
                    d3_assessment=d3_draft,
                    conflict_assessment=conflict_assessment,
                    telemetry=telemetry,
                )
            )

        all_supported = all(
            a.final_label == SemanticSupportLabel.SUPPORTED for a in assessments
        )
        is_valid = base_result.is_valid and (len(execution_error_claims) == 0) and all_supported

        structured_result = StructuredSemanticVerificationResultD32(
            is_valid=is_valid,
            assessments=assessments,
            provider_name=self._provider.provider_name,
            provider_version=self._provider.provider_version,
            model_name=self._provider.model_name,
            model_revision=self._provider.model_revision,
            errors=errors,
            execution_error_claims=execution_error_claims,
            claim_telemetries=claim_telemetries,
            overridden_claims_count=overridden_count,
        )

        return base_result, structured_result
