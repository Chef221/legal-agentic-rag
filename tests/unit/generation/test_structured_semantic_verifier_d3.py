"""Unit tests for V2-D3 Structured Semantic Citation Verifier."""

from __future__ import annotations

import json
from typing import Any
import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.generation.structured_semantic_verifier_d3 import (
    D3EvidenceRelation,
    D3StructuredClaimAssessmentDraft,
    DraftRejectionCategory,
    STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION,
    StructuredSemanticCitationVerifierD3,
    derive_claim_semantic_label_d3,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
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
    citations: list[Citation] | None = None,
) -> AnswerResponse:
    if citations is None:
        citations = [
            Citation(
                evidence_id="E1",
                chunk_id="chunk_E1",
                document_id="doc_001",
                document_title="Luật Doanh nghiệp 2020",
                document_number="59/2020/QH14",
                article_number="10",
            )
        ]
    return AnswerResponse(
        question="Doanh nghiệp nhà nước gồm những doanh nghiệp nào?",
        answer=answer,
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=citations,
    )


def test_derive_claim_semantic_label_d3_entails():
    """ENTAILS maps deterministically to SUPPORTED."""
    draft = D3StructuredClaimAssessmentDraft(
        claim_id="C1",
        relation=D3EvidenceRelation.ENTAILS,
        actor_mismatch=False,
        condition_exception_mismatch=False,
        quantity_temporal_mismatch=False,
        negation_modality_mismatch=False,
        source_scope_mismatch=False,
    )
    assert derive_claim_semantic_label_d3(draft) == SemanticSupportLabel.SUPPORTED


def test_derive_claim_semantic_label_d3_contradicts():
    """CONTRADICTS maps deterministically to CONTRADICTED."""
    draft = D3StructuredClaimAssessmentDraft(
        claim_id="C1",
        relation=D3EvidenceRelation.CONTRADICTS,
        actor_mismatch=True,
        condition_exception_mismatch=False,
        quantity_temporal_mismatch=False,
        negation_modality_mismatch=False,
        source_scope_mismatch=False,
    )
    assert derive_claim_semantic_label_d3(draft) == SemanticSupportLabel.CONTRADICTED


def test_derive_claim_semantic_label_d3_does_not_establish():
    """DOES_NOT_ESTABLISH maps deterministically to INSUFFICIENT."""
    draft = D3StructuredClaimAssessmentDraft(
        claim_id="C1",
        relation=D3EvidenceRelation.DOES_NOT_ESTABLISH,
        actor_mismatch=False,
        condition_exception_mismatch=False,
        quantity_temporal_mismatch=False,
        negation_modality_mismatch=False,
        source_scope_mismatch=False,
    )
    assert derive_claim_semantic_label_d3(draft) == SemanticSupportLabel.INSUFFICIENT


def test_diagnostic_flags_do_not_override_relation():
    """Diagnostic flags must not override the primary relation."""
    draft1 = D3StructuredClaimAssessmentDraft(
        claim_id="C1",
        relation=D3EvidenceRelation.ENTAILS,
        actor_mismatch=True,
        condition_exception_mismatch=True,
        quantity_temporal_mismatch=False,
        negation_modality_mismatch=False,
        source_scope_mismatch=False,
    )
    assert derive_claim_semantic_label_d3(draft1) == SemanticSupportLabel.SUPPORTED

    draft2 = D3StructuredClaimAssessmentDraft(
        claim_id="C2",
        relation=D3EvidenceRelation.CONTRADICTS,
        actor_mismatch=False,
        condition_exception_mismatch=False,
        quantity_temporal_mismatch=False,
        negation_modality_mismatch=False,
        source_scope_mismatch=False,
    )
    assert derive_claim_semantic_label_d3(draft2) == SemanticSupportLabel.CONTRADICTED


def test_one_claim_per_provider_call():
    """Verifier makes one separate provider call per claim."""
    answer_text = "Doanh nghiệp nhà nước do Nhà nước nắm giữ 100% [E1]. Vốn điều lệ phải đăng ký [E2]."
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
    resp = _make_dummy_response(answer=answer_text, citations=citations)
    ev1 = _make_dummy_evidence("E1")
    ev2 = _make_dummy_evidence("E2")

    p1 = json.dumps({
        "claim_id": "C1",
        "relation": "ENTAILS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    p2 = json.dumps({
        "claim_id": "C2",
        "relation": "CONTRADICTS",
        "actor_mismatch": True,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([p1, p2])
    verifier = StructuredSemanticCitationVerifierD3(provider)

    cit_res, struct_res = verifier.verify_structured(resp, [ev1, ev2])
    assert len(provider.call_history) == 2
    assert "Claim ID: C1" in provider.call_history[0]["user_prompt"]
    assert "Claim ID: C2" in provider.call_history[1]["user_prompt"]
    assert len(struct_res.assessments) == 2
    assert struct_res.assessments[0].label == SemanticSupportLabel.SUPPORTED
    assert struct_res.assessments[1].label == SemanticSupportLabel.CONTRADICTED


def test_wrong_actor_generic_case():
    """Wrong actor mismatch with CONTRADICTS."""
    resp = _make_dummy_response("Thủ tướng Chính phủ cấp giấy phép [E1].")
    ev = _make_dummy_evidence("E1")
    p1 = json.dumps({
        "claim_id": "C1",
        "relation": "CONTRADICTS",
        "actor_mismatch": True,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([p1])
    verifier = StructuredSemanticCitationVerifierD3(provider)
    _, struct_res = verifier.verify_structured(resp, [ev])
    assert struct_res.assessments[0].label == SemanticSupportLabel.CONTRADICTED
    assert struct_res.assessments[0].actor_mismatch is True


def test_partial_same_topic_generic_case():
    """Same topic partial evidence with DOES_NOT_ESTABLISH."""
    resp = _make_dummy_response("Lệ phí đăng ký doanh nghiệp là 50.000 VNĐ [E1].")
    ev = _make_dummy_evidence("E1")
    p1 = json.dumps({
        "claim_id": "C1",
        "relation": "DOES_NOT_ESTABLISH",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([p1])
    verifier = StructuredSemanticCitationVerifierD3(provider)
    _, struct_res = verifier.verify_structured(resp, [ev])
    assert struct_res.assessments[0].label == SemanticSupportLabel.INSUFFICIENT
    assert struct_res.assessments[0].relation == D3EvidenceRelation.DOES_NOT_ESTABLISH


def test_condition_omission_generic_case():
    """Condition omission mismatch."""
    resp = _make_dummy_response("Nhà đầu tư nước ngoài được đầu tư tự do [E1].")
    ev = _make_dummy_evidence("E1")
    p1 = json.dumps({
        "claim_id": "C1",
        "relation": "DOES_NOT_ESTABLISH",
        "actor_mismatch": False,
        "condition_exception_mismatch": True,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([p1])
    verifier = StructuredSemanticCitationVerifierD3(provider)
    _, struct_res = verifier.verify_structured(resp, [ev])
    assert struct_res.assessments[0].condition_exception_mismatch is True
    assert struct_res.assessments[0].label == SemanticSupportLabel.INSUFFICIENT


def test_numeric_semantic_role_mismatch_generic_case():
    """Numeric semantic-role mismatch."""
    resp = _make_dummy_response("Thời hạn đăng ký là 15 ngày [E1].")
    ev = _make_dummy_evidence("E1")
    p1 = json.dumps({
        "claim_id": "C1",
        "relation": "CONTRADICTS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": True,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([p1])
    verifier = StructuredSemanticCitationVerifierD3(provider)
    _, struct_res = verifier.verify_structured(resp, [ev])
    assert struct_res.assessments[0].quantity_temporal_mismatch is True
    assert struct_res.assessments[0].label == SemanticSupportLabel.CONTRADICTED


def test_source_scope_mismatch_generic_case():
    """Source-scope mismatch."""
    resp = _make_dummy_response("Theo Nghị định 01, lệ phí được miễn [E1].")
    ev = _make_dummy_evidence("E1")
    p1 = json.dumps({
        "claim_id": "C1",
        "relation": "CONTRADICTS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": True,
    })
    provider = MockChatProvider([p1])
    verifier = StructuredSemanticCitationVerifierD3(provider)
    _, struct_res = verifier.verify_structured(resp, [ev])
    assert struct_res.assessments[0].source_scope_mismatch is True
    assert struct_res.assessments[0].label == SemanticSupportLabel.CONTRADICTED


def test_exact_claim_id_validation_and_retry():
    """Claim ID mismatch triggers retry with targeted error message."""
    resp = _make_dummy_response()
    ev = _make_dummy_evidence()
    bad_resp = json.dumps({
        "claim_id": "WRONG_ID",
        "relation": "ENTAILS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    good_resp = json.dumps({
        "claim_id": "C1",
        "relation": "ENTAILS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([bad_resp, good_resp])
    verifier = StructuredSemanticCitationVerifierD3(provider, max_structured_output_retries=1)
    _, struct_res = verifier.verify_structured(resp, [ev])

    assert len(provider.call_history) == 2
    assert "CLAIM_ID_MISMATCH" in provider.call_history[1]["user_prompt"]
    assert struct_res.assessments[0].label == SemanticSupportLabel.SUPPORTED
    assert struct_res.assessments[0].telemetry is not None
    assert struct_res.assessments[0].telemetry.retry_count == 1
    assert struct_res.assessments[0].telemetry.draft_rejection_categories == ["CLAIM_ID_MISMATCH"]


def test_permanent_failure_isolated_and_preserves_telemetry():
    """Permanent execution error on C1 does not poison C2 and preserves operational telemetry."""
    answer_text = "Doanh nghiệp nhà nước do Nhà nước nắm giữ 100% [E1]. Vốn điều lệ phải đăng ký [E2]."
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
    resp = _make_dummy_response(answer=answer_text, citations=citations)
    ev1 = _make_dummy_evidence("E1")
    ev2 = _make_dummy_evidence("E2")

    bad_resp1 = "NOT JSON"
    bad_resp2 = "STILL NOT JSON"
    good_resp_c2 = json.dumps({
        "claim_id": "C2",
        "relation": "ENTAILS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([bad_resp1, bad_resp2, good_resp_c2])
    verifier = StructuredSemanticCitationVerifierD3(provider, max_structured_output_retries=1)
    cit_res, struct_res = verifier.verify_structured(resp, [ev1, ev2])

    assert len(struct_res.execution_error_claims) == 1
    assert "C1" in struct_res.execution_error_claims
    assert len(struct_res.assessments) == 2

    c1_assess = [a for a in struct_res.assessments if a.claim_id == "C1"][0]
    c2_assess = [a for a in struct_res.assessments if a.claim_id == "C2"][0]

    assert c1_assess.label == SemanticSupportLabel.INSUFFICIENT
    assert c1_assess.telemetry is not None
    assert c1_assess.telemetry.semantic_execution_error is True
    assert c1_assess.telemetry.provider_call_count == 2
    assert c1_assess.telemetry.retry_count == 1
    assert c1_assess.telemetry.draft_rejection_count == 2
    assert c1_assess.telemetry.draft_rejection_categories == ["JSON_PARSE_ERROR", "JSON_PARSE_ERROR"]

    # C2 is completely normal
    assert c2_assess.label == SemanticSupportLabel.SUPPORTED
    assert c2_assess.telemetry is not None
    assert c2_assess.telemetry.semantic_execution_error is False
    assert c2_assess.telemetry.retry_count == 0

    # Top-level claim_telemetries dictionary preserves both
    assert "C1" in struct_res.claim_telemetries
    assert struct_res.claim_telemetries["C1"].semantic_execution_error is True
    assert "C2" in struct_res.claim_telemetries
    assert struct_res.claim_telemetries["C2"].semantic_execution_error is False


def test_no_human_labels_error_tags_reference_answer_in_prompts():
    """Prompts must never contain human benchmark labels, error tags, or reference answers."""
    resp = _make_dummy_response()
    ev = _make_dummy_evidence()
    good_resp = json.dumps({
        "claim_id": "C1",
        "relation": "ENTAILS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    provider = MockChatProvider([good_resp])
    verifier = StructuredSemanticCitationVerifierD3(provider)
    verifier.verify_structured(resp, [ev])

    system_inst = provider.call_history[0]["system_instruction"]
    user_p = provider.call_history[0]["user_prompt"]

    forbidden = [
        "HUMAN_SUPPORTED", "HUMAN_CONTRADICTED", "HUMAN_INSUFFICIENT",
        "MISSING_PREDICATE", "INACCURATE_ACTOR", "CITATION_SCOPE_MISMATCH",
        "REFERENCE_ANSWER", "GOLD_ANSWER", "GROUND_TRUTH",
    ]
    for term in forbidden:
        assert term not in system_inst, f"Forbidden term {term} found in system instruction"
        assert term not in user_p, f"Forbidden term {term} found in user prompt"
