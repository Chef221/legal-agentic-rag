"""Unit tests for V2-D3.2 Structured Semantic Verifier (Frozen D3 Base + Strict Conflict Overlay).

All tests use synthetic payloads only. No external model invocations.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import ModelError
from legal_agentic_rag.generation.structured_semantic_verifier_d3 import (
    D3EvidenceRelation,
    D3StructuredClaimAssessmentDraft,
)
from legal_agentic_rag.generation.structured_semantic_verifier_d32 import (
    StructuredClaimVerificationD32,
    StructuredSemanticCitationVerifierD32,
    StructuredSemanticVerificationResultD32,
)
from legal_agentic_rag.generation.structured_semantic_verifier_d32_conflict import (
    DraftRejectionCategoryD32Conflict,
    StrictConflictStatus,
    StructuredSemanticConflictDraftD32,
    StructuredSemanticVerifierD32Conflict,
    derive_strict_conflict_status_d32,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    Evidence,
    SemanticSupportLabel,
)


class MockD32ChatProvider(ChatModelProvider):
    """Synthetic provider simulating D3 base and D3.2 conflict model responses."""

    def __init__(
        self,
        d3_responses: list[str] | None = None,
        conflict_responses: list[str] | None = None,
    ) -> None:
        self._d3_responses = list(d3_responses or [])
        self._conflict_responses = list(conflict_responses or [])
        self.d3_calls: list[dict[str, str]] = []
        self.conflict_calls: list[dict[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "transformers"

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
        if "conflict verifier" in system_instruction.lower() or "cannot_both_be_true" in system_instruction:
            self.conflict_calls.append({"sys": system_instruction, "prompt": user_prompt})
            if self._conflict_responses:
                return self._conflict_responses.pop(0)
            return json.dumps({"claim_id": "C1", "same_material_proposition": False, "cannot_both_be_true": False})
        else:
            self.d3_calls.append({"sys": system_instruction, "prompt": user_prompt})
            if self._d3_responses:
                return self._d3_responses.pop(0)
            return json.dumps({
                "claim_id": "C1",
                "relation": "ENTAILS",
                "authority_mismatch": False,
                "scope_or_condition_mismatch": False,
                "temporal_or_version_mismatch": False,
                "exception_mismatch": False,
                "negation_or_modality_mismatch": False,
            })


# Test 1: State Machine Derivation for Strict Conflict
def test_derive_strict_conflict_status_confirmed():
    draft = StructuredSemanticConflictDraftD32(
        claim_id="C1", same_material_proposition=True, cannot_both_be_true=True
    )
    assert derive_strict_conflict_status_d32(draft) == StrictConflictStatus.STRICT_CONTRADICTION_CONFIRMED


def test_derive_strict_conflict_status_same_prop_compatible():
    draft = StructuredSemanticConflictDraftD32(
        claim_id="C1", same_material_proposition=True, cannot_both_be_true=False
    )
    assert derive_strict_conflict_status_d32(draft) == StrictConflictStatus.NO_STRICT_CONTRADICTION


def test_derive_strict_conflict_status_diff_prop_compatible():
    draft = StructuredSemanticConflictDraftD32(
        claim_id="C1", same_material_proposition=False, cannot_both_be_true=False
    )
    assert derive_strict_conflict_status_d32(draft) == StrictConflictStatus.NO_STRICT_CONTRADICTION


def test_derive_strict_conflict_status_invalid_state():
    draft = StructuredSemanticConflictDraftD32(
        claim_id="C1", same_material_proposition=False, cannot_both_be_true=True
    )
    with pytest.raises(ValueError, match="logically inconsistent"):
        derive_strict_conflict_status_d32(draft)


# Test 2: Conflict Verifier Parser & Retry on Inconsistent State
def test_conflict_verifier_retry_on_invalid_logical_state():
    # Attempt 1: (False, True) -> Logically inconsistent
    # Attempt 2: (True, True) -> Confirmed contradiction
    resp1 = json.dumps({"claim_id": "C1", "same_material_proposition": False, "cannot_both_be_true": True})
    resp2 = json.dumps({"claim_id": "C1", "same_material_proposition": True, "cannot_both_be_true": True})

    provider = MockD32ChatProvider(conflict_responses=[resp1, resp2])
    verifier = StructuredSemanticVerifierD32Conflict(provider, max_structured_output_retries=1)

    evidence = [Evidence(evidence_id="E1", chunk_id="chk1", document_id="doc1", text="Luật quy định thẩm quyền thuộc Tổng Kiểm toán.")]
    draft, telem = verifier.evaluate_conflict(
        question="Ai có thẩm quyền?",
        claim_id="C1",
        claim_text="Thẩm quyền thuộc Trưởng đoàn.",
        evidence=evidence,
    )

    assert draft is not None
    assert draft.same_material_proposition is True
    assert draft.cannot_both_be_true is True
    assert telem.retry_count == 1
    assert telem.provider_call_count == 2
    assert DraftRejectionCategoryD32Conflict.LOGICALLY_INCONSISTENT_STATE.value in telem.draft_rejection_categories


# Test 3: Conflict Verifier Permanent Failure on Repeated Rejection
def test_conflict_verifier_permanent_failure():
    resp1 = "NOT JSON"
    resp2 = "STILL NOT JSON"
    provider = MockD32ChatProvider(conflict_responses=[resp1, resp2])
    verifier = StructuredSemanticVerifierD32Conflict(provider, max_structured_output_retries=1)

    draft, telem = verifier.evaluate_conflict(
        question="Q",
        claim_id="C1",
        claim_text="Claim",
        evidence=[],
    )

    assert draft is None
    assert telem.semantic_execution_error is True
    assert telem.provider_call_count == 2


# Test 4: D3.2 Orchestrator - D3 Base SUPPORTED + No Conflict -> Preserved SUPPORTED
def test_d32_orchestrator_preserves_supported():
    d3_resp = json.dumps({
        "claim_id": "C1",
        "relation": "ENTAILS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    conflict_resp = json.dumps({
        "claim_id": "C1",
        "same_material_proposition": True,
        "cannot_both_be_true": False,
    })

    provider = MockD32ChatProvider(d3_responses=[d3_resp], conflict_responses=[conflict_resp])
    verifier = StructuredSemanticCitationVerifierD32(provider)

    evidence = [Evidence(evidence_id="E1", chunk_id="chk1", document_id="doc1", text="Người có thẩm quyền là Thủ trưởng.", source_url="http://law/1")]
    response = AnswerResponse(
        question="Ai có thẩm quyền?",
        answer="Thủ trưởng có thẩm quyền [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[Citation(evidence_id="E1", chunk_id="chk1", document_id="doc1", source_url="http://law/1")],
    )

    cit_res, struct_res = verifier.verify_structured(response, evidence)
    assert len(struct_res.assessments) == 1
    assess = struct_res.assessments[0]
    assert assess.base_d3_label == SemanticSupportLabel.SUPPORTED
    assert assess.final_label == SemanticSupportLabel.SUPPORTED
    assert assess.override_applied is False
    assert struct_res.is_valid is True
    assert len(provider.d3_calls) == 1
    assert len(provider.conflict_calls) == 1


# Test 5: D3.2 Orchestrator - D3 Base INSUFFICIENT + No Conflict -> Preserved INSUFFICIENT
def test_d32_orchestrator_preserves_insufficient():
    d3_resp = json.dumps({
        "claim_id": "C1",
        "relation": "DOES_NOT_ESTABLISH",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    conflict_resp = json.dumps({
        "claim_id": "C1",
        "same_material_proposition": False,
        "cannot_both_be_true": False,
    })

    provider = MockD32ChatProvider(d3_responses=[d3_resp], conflict_responses=[conflict_resp])
    verifier = StructuredSemanticCitationVerifierD32(provider)

    evidence = [Evidence(evidence_id="E1", chunk_id="chk1", document_id="doc1", text="Văn bản này quy định về thuế giá trị gia tăng.", source_url="http://law/1")]
    response = AnswerResponse(
        question="Thời hạn tạm giam là bao lâu?",
        answer="Thời hạn tạm giam là 3 tháng [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[Citation(evidence_id="E1", chunk_id="chk1", document_id="doc1", source_url="http://law/1")],
    )

    cit_res, struct_res = verifier.verify_structured(response, evidence)
    assert len(struct_res.assessments) == 1
    assess = struct_res.assessments[0]
    assert assess.base_d3_label == SemanticSupportLabel.INSUFFICIENT
    assert assess.final_label == SemanticSupportLabel.INSUFFICIENT
    assert assess.override_applied is False
    assert struct_res.is_valid is False


# Test 6: D3.2 Orchestrator - D3 Base SUPPORTED + Strict Conflict Confirmed -> Override TO CONTRADICTED
def test_d32_orchestrator_overrides_supported_to_contradicted():
    d3_resp = json.dumps({
        "claim_id": "C1",
        "relation": "ENTAILS",  # D3 erroneously entailed
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    conflict_resp = json.dumps({
        "claim_id": "C1",
        "same_material_proposition": True,
        "cannot_both_be_true": True,  # Strict contradiction confirmed!
    })

    provider = MockD32ChatProvider(d3_responses=[d3_resp], conflict_responses=[conflict_resp])
    verifier = StructuredSemanticCitationVerifierD32(provider)

    evidence = [Evidence(evidence_id="E1", chunk_id="chk1", document_id="doc1", text="Thời hạn bổ nhiệm lại là 5 năm.", source_url="http://law/1")]
    response = AnswerResponse(
        question="Thời hạn bổ nhiệm lại là bao lâu?",
        answer="Thời hạn bổ nhiệm lại là 3 năm [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[Citation(evidence_id="E1", chunk_id="chk1", document_id="doc1", source_url="http://law/1")],
    )

    cit_res, struct_res = verifier.verify_structured(response, evidence)
    assert len(struct_res.assessments) == 1
    assess = struct_res.assessments[0]
    assert assess.base_d3_label == SemanticSupportLabel.SUPPORTED
    assert assess.final_label == SemanticSupportLabel.CONTRADICTED
    assert assess.override_applied is True
    assert struct_res.overridden_claims_count == 1
    assert struct_res.is_valid is False


# Test 7: D3.2 Orchestrator - D3 Base INSUFFICIENT + Strict Conflict Confirmed -> Override TO CONTRADICTED
def test_d32_orchestrator_overrides_insufficient_to_contradicted():
    d3_resp = json.dumps({
        "claim_id": "C1",
        "relation": "DOES_NOT_ESTABLISH",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })
    conflict_resp = json.dumps({
        "claim_id": "C1",
        "same_material_proposition": True,
        "cannot_both_be_true": True,
    })

    provider = MockD32ChatProvider(d3_responses=[d3_resp], conflict_responses=[conflict_resp])
    verifier = StructuredSemanticCitationVerifierD32(provider)

    evidence = [Evidence(evidence_id="E1", chunk_id="chk1", document_id="doc1", text="Thẩm quyền xử phạt thuộc Chủ tịch UBND tỉnh.", source_url="http://law/1")]
    response = AnswerResponse(
        question="Ai có thẩm quyền xử phạt?",
        answer="Thẩm quyền xử phạt thuộc Trưởng đoàn kiểm toán [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[Citation(evidence_id="E1", chunk_id="chk1", document_id="doc1", source_url="http://law/1")],
    )

    cit_res, struct_res = verifier.verify_structured(response, evidence)
    assess = struct_res.assessments[0]
    assert assess.base_d3_label == SemanticSupportLabel.INSUFFICIENT
    assert assess.final_label == SemanticSupportLabel.CONTRADICTED
    assert assess.override_applied is True


# Test 8: Prompt Blindness - Conflict Prompt does not contain D3 prediction or benchmark info
def test_conflict_prompt_blindness():
    provider = MockD32ChatProvider()
    verifier = StructuredSemanticCitationVerifierD32(provider)

    evidence = [Evidence(evidence_id="E1", chunk_id="chk1", document_id="doc1", text="Nội dung luật", source_url="http://law/1")]
    response = AnswerResponse(
        question="Câu hỏi pháp lý?",
        answer="Nội dung câu trả lời [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[Citation(evidence_id="E1", chunk_id="chk1", document_id="doc1", source_url="http://law/1")],
    )

    verifier.verify_structured(response, evidence)
    assert len(provider.conflict_calls) == 1
    conflict_prompt = provider.conflict_calls[0]["prompt"]

    forbidden_tokens = ["ENTAILS", "CONTRADICTS", "DOES_NOT_ESTABLISH", "gold", "human_label", "benchmark", "B-FORENSIC"]
    for tok in forbidden_tokens:
        assert tok not in conflict_prompt, f"Forbidden token '{tok}' found in conflict prompt"
