"""Unit tests for V2-D2 Structured Semantic Citation Verifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.generation.structured_semantic_verifier_d2 import (
    STRUCTURED_SEMANTIC_D2_SYSTEM_INSTRUCTION,
    D2EvidenceCoverageStatus,
    D2SemanticDimensionStatus,
    D2StructuredClaimAssessmentDraft,
    DraftRejectionCategory,
    StructuredSemanticCitationVerifierD2,
    derive_claim_semantic_label_d2,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    Evidence,
    SemanticSupportLabel,
)


class MockChatProvider(ChatModelProvider):
    """Mock provider with programmed sequential response queue."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = responses
        self._call_count = 0
        self.call_history: list[dict[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "mock_transformers"

    @property
    def provider_version(self) -> str:
        return "4.47.1"

    @property
    def model_name(self) -> str:
        return "Qwen/Qwen2.5-3B-Instruct"

    @property
    def model_revision(self) -> str:
        return "a1d308dfcc03e09da285d49d912439a655a571e8"

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        idx = self._call_count
        self._call_count += 1
        self.call_history.append(
            {"system_instruction": system_instruction, "user_prompt": user_prompt}
        )
        if idx < len(self._responses):
            resp = self._responses[idx]
        else:
            resp = self._responses[-1]
        if isinstance(resp, Exception):
            raise resp
        return resp


def _make_dummy_evidence(eid: str = "E1") -> Evidence:
    return Evidence(
        evidence_id=eid,
        chunk_id=f"chunk_{eid}",
        document_id="doc_001",
        document_title="Luật Doanh nghiệp 2020",
        document_number="59/2020/QH14",
        article_number="10",
        article_title="Tiêu chí doanh nghiệp",
        effect_status="active",
        text="Doanh nghiệp nhà nước bao gồm doanh nghiệp do Nhà nước nắm giữ 100% vốn điều lệ.",
    )


def _make_dummy_response(
    answer: str = "Doanh nghiệp nhà nước do Nhà nước nắm giữ 100% vốn điều lệ [E1].",
) -> AnswerResponse:
    return AnswerResponse(
        question="Doanh nghiệp nhà nước gồm những doanh nghiệp nào?",
        answer=answer,
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_E1",
                document_id="doc_001",
                document_title="Luật Doanh nghiệp 2020",
                document_number="59/2020/QH14",
                article_number="10",
            )
        ],
    )


def test_deterministic_label_derivation_all_cases():
    """Verify all categorical derivation branches of derive_claim_semantic_label_d2."""
    # 1. Any CONFLICT -> CONTRADICTED
    draft_conflict = D2StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=D2SemanticDimensionStatus.CONFLICT,
        action_object=D2SemanticDimensionStatus.ESTABLISHED,
        condition_exception=D2SemanticDimensionStatus.ESTABLISHED,
        quantity_temporal=D2SemanticDimensionStatus.ESTABLISHED,
        negation_modality=D2SemanticDimensionStatus.ESTABLISHED,
        source_article_scope=D2SemanticDimensionStatus.ESTABLISHED,
        evidence_coverage=D2EvidenceCoverageStatus.COMPLETE,
    )
    assert derive_claim_semantic_label_d2(draft_conflict) == SemanticSupportLabel.CONTRADICTED

    # 2. Coverage != COMPLETE -> INSUFFICIENT
    draft_partial = D2StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=D2SemanticDimensionStatus.ESTABLISHED,
        action_object=D2SemanticDimensionStatus.ESTABLISHED,
        condition_exception=D2SemanticDimensionStatus.ESTABLISHED,
        quantity_temporal=D2SemanticDimensionStatus.ESTABLISHED,
        negation_modality=D2SemanticDimensionStatus.ESTABLISHED,
        source_article_scope=D2SemanticDimensionStatus.ESTABLISHED,
        evidence_coverage=D2EvidenceCoverageStatus.PARTIAL,
    )
    assert derive_claim_semantic_label_d2(draft_partial) == SemanticSupportLabel.INSUFFICIENT

    draft_none = D2StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=D2SemanticDimensionStatus.ESTABLISHED,
        action_object=D2SemanticDimensionStatus.ESTABLISHED,
        condition_exception=D2SemanticDimensionStatus.ESTABLISHED,
        quantity_temporal=D2SemanticDimensionStatus.ESTABLISHED,
        negation_modality=D2SemanticDimensionStatus.ESTABLISHED,
        source_article_scope=D2SemanticDimensionStatus.ESTABLISHED,
        evidence_coverage=D2EvidenceCoverageStatus.NONE,
    )
    assert derive_claim_semantic_label_d2(draft_none) == SemanticSupportLabel.INSUFFICIENT

    # 3. Any material dimension NOT_ESTABLISHED -> INSUFFICIENT
    draft_not_est = D2StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=D2SemanticDimensionStatus.ESTABLISHED,
        action_object=D2SemanticDimensionStatus.ESTABLISHED,
        condition_exception=D2SemanticDimensionStatus.NOT_ESTABLISHED,
        quantity_temporal=D2SemanticDimensionStatus.ESTABLISHED,
        negation_modality=D2SemanticDimensionStatus.ESTABLISHED,
        source_article_scope=D2SemanticDimensionStatus.ESTABLISHED,
        evidence_coverage=D2EvidenceCoverageStatus.COMPLETE,
    )
    assert derive_claim_semantic_label_d2(draft_not_est) == SemanticSupportLabel.INSUFFICIENT

    # 4. All material ESTABLISHED + COMPLETE -> SUPPORTED
    draft_supported = D2StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=D2SemanticDimensionStatus.ESTABLISHED,
        action_object=D2SemanticDimensionStatus.ESTABLISHED,
        condition_exception=D2SemanticDimensionStatus.ESTABLISHED,
        quantity_temporal=D2SemanticDimensionStatus.ESTABLISHED,
        negation_modality=D2SemanticDimensionStatus.ESTABLISHED,
        source_article_scope=D2SemanticDimensionStatus.ESTABLISHED,
        evidence_coverage=D2EvidenceCoverageStatus.COMPLETE,
    )
    assert derive_claim_semantic_label_d2(draft_supported) == SemanticSupportLabel.SUPPORTED

    # 5. NOT_MATERIAL does not cause failure if other dimensions are ESTABLISHED + COMPLETE
    draft_not_material = D2StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=D2SemanticDimensionStatus.NOT_MATERIAL,
        action_object=D2SemanticDimensionStatus.ESTABLISHED,
        condition_exception=D2SemanticDimensionStatus.NOT_MATERIAL,
        quantity_temporal=D2SemanticDimensionStatus.NOT_MATERIAL,
        negation_modality=D2SemanticDimensionStatus.ESTABLISHED,
        source_article_scope=D2SemanticDimensionStatus.ESTABLISHED,
        evidence_coverage=D2EvidenceCoverageStatus.COMPLETE,
    )
    assert derive_claim_semantic_label_d2(draft_not_material) == SemanticSupportLabel.SUPPORTED


def test_single_claim_invocation_and_successful_verification():
    """Verify single claim verification performs exactly 1 provider call and constructs valid result."""
    draft_json = json.dumps(
        {
            "claim_id": "C1",
            "actor_role": "ESTABLISHED",
            "action_object": "ESTABLISHED",
            "condition_exception": "NOT_MATERIAL",
            "quantity_temporal": "ESTABLISHED",
            "negation_modality": "ESTABLISHED",
            "source_article_scope": "ESTABLISHED",
            "evidence_coverage": "COMPLETE",
        }
    )
    provider = MockChatProvider([draft_json])
    verifier = StructuredSemanticCitationVerifierD2(provider=provider)

    resp = _make_dummy_response()
    ev = _make_dummy_evidence()

    result, structured_res = verifier.verify_structured(resp, [ev])

    assert len(provider.call_history) == 1
    assert result.is_valid is True
    assert len(structured_res.assessments) == 1
    assessment = structured_res.assessments[0]
    assert assessment.claim_id == "C1"
    assert assessment.label == SemanticSupportLabel.SUPPORTED
    assert assessment.telemetry is not None
    assert assessment.telemetry.provider_call_count == 1
    assert assessment.telemetry.retry_count == 0
    assert assessment.telemetry.draft_rejection_count == 0
    assert assessment.telemetry.semantic_execution_error is False


def test_multi_claim_independent_provider_calls():
    """Verify an answer with 3 claims triggers 3 independent provider calls with isolated contexts."""
    # 3 claims in answer
    answer_text = "Doanh nghiệp nhà nước do Nhà nước nắm giữ 100% [E1]. Vốn điều lệ phải đăng ký [E2]. Không được kinh doanh ngành nghề cấm [E1]."
    citations = [
        Citation(
            evidence_id="E1",
            chunk_id="chunk_E1",
            document_id="doc_001",
            document_title="Luật Doanh nghiệp",
            document_number="59/2020/QH14",
            article_number="10",
        ),
        Citation(
            evidence_id="E2",
            chunk_id="chunk_E2",
            document_id="doc_001",
            document_title="Luật Doanh nghiệp",
            document_number="59/2020/QH14",
            article_number="11",
        ),
    ]
    resp = AnswerResponse(
        question="Q?",
        answer=answer_text,
        insufficient_evidence=False,
        retrieval_strategy="hybrid",
        trace_id="t1",
        citations=citations,
    )
    ev1 = _make_dummy_evidence("E1")
    ev2 = _make_dummy_evidence("E2")

    # Responses for C1, C2, C3
    r1 = json.dumps({
        "claim_id": "C1", "actor_role": "ESTABLISHED", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "ESTABLISHED",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })
    r2 = json.dumps({
        "claim_id": "C2", "actor_role": "CONFLICT", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "NOT_MATERIAL",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })
    r3 = json.dumps({
        "claim_id": "C3", "actor_role": "NOT_ESTABLISHED", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "NOT_MATERIAL",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })

    provider = MockChatProvider([r1, r2, r3])
    verifier = StructuredSemanticCitationVerifierD2(provider=provider)

    result, structured_res = verifier.verify_structured(resp, [ev1, ev2])

    assert len(provider.call_history) == 3
    assert "C1" in provider.call_history[0]["user_prompt"]
    assert "C2" in provider.call_history[1]["user_prompt"]
    assert "C3" in provider.call_history[2]["user_prompt"]

    assert len(structured_res.assessments) == 3
    assert structured_res.assessments[0].label == SemanticSupportLabel.SUPPORTED
    assert structured_res.assessments[1].label == SemanticSupportLabel.CONTRADICTED
    assert structured_res.assessments[2].label == SemanticSupportLabel.INSUFFICIENT
    assert result.is_valid is False


def test_malformed_claim_does_not_poison_sibling_claims():
    """Verify that a single claim failure on retry does not fail sibling claim evaluations."""
    answer_text = "Khái niệm doanh nghiệp [E1]. Đăng ký vốn [E1]. Phạt tiền 10 triệu [E1]."
    citations = [
        Citation(
            evidence_id="E1",
            chunk_id="chunk_E1",
            document_id="doc_001",
            document_title="Luật Doanh nghiệp",
            document_number="59/2020/QH14",
            article_number="10",
        ),
    ]
    resp = AnswerResponse(
        question="Q?",
        answer=answer_text,
        insufficient_evidence=False,
        retrieval_strategy="hybrid",
        trace_id="t1",
        citations=citations,
    )
    ev = _make_dummy_evidence("E1")

    # C1: valid SUPPORTED
    r_c1 = json.dumps({
        "claim_id": "C1", "actor_role": "ESTABLISHED", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "NOT_MATERIAL",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })
    # C2: attempt 1 invalid json, attempt 2 invalid json -> execution error
    r_c2_1 = "This is not JSON at all"
    r_c2_2 = "Still not JSON"
    # C3: valid CONTRADICTED
    r_c3 = json.dumps({
        "claim_id": "C3", "actor_role": "ESTABLISHED", "action_object": "CONFLICT",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "NOT_MATERIAL",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })

    provider = MockChatProvider([r_c1, r_c2_1, r_c2_2, r_c3])
    verifier = StructuredSemanticCitationVerifierD2(provider=provider, max_structured_output_retries=1)

    result, structured_res = verifier.verify_structured(resp, [ev])

    # 4 calls made: C1 (1), C2 (2: attempt 1 + retry), C3 (1)
    assert len(provider.call_history) == 4
    assert len(structured_res.assessments) == 3

    # C1 is SUPPORTED
    assert structured_res.assessments[0].claim_id == "C1"
    assert structured_res.assessments[0].label == SemanticSupportLabel.SUPPORTED

    # C2 has execution error isolated to C2
    assert structured_res.assessments[1].claim_id == "C2"
    assert structured_res.assessments[1].label == SemanticSupportLabel.INSUFFICIENT
    assert structured_res.assessments[1].telemetry.semantic_execution_error is True
    assert structured_res.assessments[1].telemetry.draft_rejection_count == 2
    assert structured_res.assessments[1].telemetry.draft_rejection_categories == [
        DraftRejectionCategory.JSON_PARSE_ERROR.value,
        DraftRejectionCategory.JSON_PARSE_ERROR.value,
    ]

    # C3 is successfully evaluated as CONTRADICTED
    assert structured_res.assessments[2].claim_id == "C3"
    assert structured_res.assessments[2].label == SemanticSupportLabel.CONTRADICTED
    assert structured_res.assessments[2].telemetry.semantic_execution_error is False

    assert "C2" in structured_res.execution_error_claims
    assert "C1" not in structured_res.execution_error_claims
    assert "C3" not in structured_res.execution_error_claims


def test_retry_on_claim_id_mismatch_and_recovers():
    """Verify that a claim_id mismatch on attempt 1 triggers retry and recovers on attempt 2."""
    resp = _make_dummy_response()
    ev = _make_dummy_evidence()

    # Attempt 1: returns claim_id="C99"
    r_bad_id = json.dumps({
        "claim_id": "C99", "actor_role": "ESTABLISHED", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "ESTABLISHED",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })
    # Attempt 2: returns correct claim_id="C1"
    r_good = json.dumps({
        "claim_id": "C1", "actor_role": "ESTABLISHED", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "ESTABLISHED",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })

    provider = MockChatProvider([r_bad_id, r_good])
    verifier = StructuredSemanticCitationVerifierD2(provider=provider, max_structured_output_retries=1)

    result, structured_res = verifier.verify_structured(resp, [ev])

    assert len(provider.call_history) == 2
    assert "STRUCTURAL_CORRECTION_INSTRUCTION" in provider.call_history[1]["user_prompt"]
    assert structured_res.assessments[0].label == SemanticSupportLabel.SUPPORTED
    assert structured_res.assessments[0].telemetry.retry_count == 1
    assert structured_res.assessments[0].telemetry.draft_rejection_count == 1
    assert structured_res.assessments[0].telemetry.draft_rejection_categories == [
        DraftRejectionCategory.CLAIM_ID_MISMATCH.value
    ]
    assert structured_res.assessments[0].telemetry.semantic_execution_error is False


def test_schema_validation_error_categorizations():
    """Verify precise draft rejection categorization for missing, extra, and enum errors."""
    verifier = StructuredSemanticCitationVerifierD2(provider=MockChatProvider([]))

    # 1. Missing field
    missing_json = json.dumps({
        "claim_id": "C1", "actor_role": "ESTABLISHED",
        # action_object missing
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "ESTABLISHED",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })
    _, cat_missing = verifier._parse_and_validate_single_claim_draft(missing_json, expected_claim_id="C1")
    assert cat_missing == DraftRejectionCategory.MISSING_FIELD.value

    # 2. Extra field
    extra_json = json.dumps({
        "claim_id": "C1", "actor_role": "ESTABLISHED", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "ESTABLISHED",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE", "extra_reasoning": "some reasoning"
    })
    _, cat_extra = verifier._parse_and_validate_single_claim_draft(extra_json, expected_claim_id="C1")
    assert cat_extra == DraftRejectionCategory.EXTRA_FIELD.value

    # 3. Invalid enum value
    enum_json = json.dumps({
        "claim_id": "C1", "actor_role": "SOME_UNKNOWN_STATUS", "action_object": "ESTABLISHED",
        "condition_exception": "NOT_MATERIAL", "quantity_temporal": "ESTABLISHED",
        "negation_modality": "ESTABLISHED", "source_article_scope": "ESTABLISHED",
        "evidence_coverage": "COMPLETE"
    })
    _, cat_enum = verifier._parse_and_validate_single_claim_draft(enum_json, expected_claim_id="C1")
    assert cat_enum == DraftRejectionCategory.ENUM_VALUE_INVALID.value
