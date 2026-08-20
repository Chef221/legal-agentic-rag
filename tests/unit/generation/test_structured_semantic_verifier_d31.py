"""Unit tests for V2-D3.1 Hierarchical Two-Gate Semantic Citation Verifier."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import ModelError
from legal_agentic_rag.generation.structured_semantic_verifier_d31 import (
    DraftRejectionCategoryD31,
    STRUCTURED_SEMANTIC_D31_SYSTEM_INSTRUCTION,
    StructuredClaimVerificationD31,
    StructuredSemanticCitationVerifierD31,
    StructuredSemanticVerificationDraftD31,
    derive_claim_semantic_label_d31,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    Evidence,
    SemanticSupportLabel,
)


class DummyChatProvider(ChatModelProvider):
    """Synthetic dummy provider returning predetermined responses."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock_provider"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def model_revision(self) -> str:
        return "mock-rev"

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        self.calls.append({
            "system_instruction": system_instruction,
            "user_prompt": user_prompt,
        })
        if self._call_count >= len(self._responses):
            raise ModelError("Exhausted mock responses")
        resp = self._responses[self._call_count]
        self._call_count += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def _make_dummy_evidence(eid: str = "E1") -> Evidence:
    return Evidence(
        evidence_id=eid,
        chunk_id=f"chunk_{eid}",
        document_id="doc_001",
        document_title="Quyết định 03/2020/QĐ-KTNN",
        document_number="03/2020/QĐ-KTNN",
        article_number="17",
        article_title="Thay thế thành viên Đoàn kiểm toán",
        effect_status="active",
        text="Tổng Kiểm toán nhà nước quyết định thay thế Phó trưởng Đoàn kiểm toán.",
    )


def _make_dummy_response(
    answer: str = "Thủ trưởng đơn vị chủ trì cuộc kiểm toán có thẩm quyền thay thế Phó trưởng Đoàn kiểm toán [E1].",
    citations: list[Citation] | None = None,
) -> AnswerResponse:
    if citations is None:
        citations = [
            Citation(
                evidence_id="E1",
                chunk_id="chunk_E1",
                document_id="doc_001",
                document_title="Quyết định 03/2020/QĐ-KTNN",
                document_number="03/2020/QĐ-KTNN",
                article_number="17",
            )
        ]
    return AnswerResponse(
        question="Ai có thẩm quyền thay thế Phó trưởng Đoàn kiểm toán?",
        answer=answer,
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=citations,
    )


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    return [_make_dummy_evidence("E1")]


@pytest.fixture
def sample_response() -> AnswerResponse:
    return _make_dummy_response()



# Test A: True / False -> CONTRADICTED
def test_deterministic_label_state_a_contradicted():
    draft = StructuredSemanticVerificationDraftD31(
        claim_id="C1",
        is_contradicted=True,
        is_fully_established=False,
    )
    assert derive_claim_semantic_label_d31(draft) == SemanticSupportLabel.CONTRADICTED


# Test B: False / True -> SUPPORTED
def test_deterministic_label_state_b_supported():
    draft = StructuredSemanticVerificationDraftD31(
        claim_id="C1",
        is_contradicted=False,
        is_fully_established=True,
    )
    assert derive_claim_semantic_label_d31(draft) == SemanticSupportLabel.SUPPORTED


# Test C: False / False -> INSUFFICIENT
def test_deterministic_label_state_c_insufficient():
    draft = StructuredSemanticVerificationDraftD31(
        claim_id="C1",
        is_contradicted=False,
        is_fully_established=False,
    )
    assert derive_claim_semantic_label_d31(draft) == SemanticSupportLabel.INSUFFICIENT


# Test D: True / True -> Invalid state raises ValueError
def test_deterministic_label_state_invalid_true_true():
    draft = StructuredSemanticVerificationDraftD31.model_construct(
        claim_id="C1",
        is_contradicted=True,
        is_fully_established=True,
    )
    with pytest.raises(ValueError, match="cannot both be True"):
        derive_claim_semantic_label_d31(draft)


# Test D2: Schema validator rejects True/True in parser
def test_parser_rejects_logically_inconsistent_true_true():
    provider = DummyChatProvider([
        json.dumps({"claim_id": "C1", "is_contradicted": True, "is_fully_established": True})
    ])
    verifier = StructuredSemanticCitationVerifierD31(provider, max_structured_output_retries=0)
    draft, rej_cat, rej_msg = verifier._parse_and_validate_draft(
        raw_completion=json.dumps({"claim_id": "C1", "is_contradicted": True, "is_fully_established": True}),
        expected_claim_id="C1",
    )
    assert draft is None
    assert rej_cat == DraftRejectionCategoryD31.LOGICALLY_INCONSISTENT_STATE


# Test E: True / True on attempt 1, valid retry succeeds
def test_retry_after_logically_inconsistent_state(sample_response, sample_evidence):
    invalid_resp = json.dumps({"claim_id": "C1", "is_contradicted": True, "is_fully_established": True})
    valid_resp = json.dumps({"claim_id": "C1", "is_contradicted": True, "is_fully_established": False})

    provider = DummyChatProvider([invalid_resp, valid_resp])
    verifier = StructuredSemanticCitationVerifierD31(provider, max_structured_output_retries=1)

    base_res, struct_res = verifier.verify_structured(sample_response, sample_evidence)
    assert len(struct_res.assessments) == 1
    assert struct_res.assessments[0].label == SemanticSupportLabel.CONTRADICTED
    assert struct_res.assessments[0].is_contradicted is True
    assert struct_res.assessments[0].is_fully_established is False

    telem = struct_res.claim_telemetries["C1"]
    assert telem.provider_call_count == 2
    assert telem.retry_count == 1
    assert telem.draft_rejection_count == 1
    assert telem.draft_rejection_categories == [DraftRejectionCategoryD31.LOGICALLY_INCONSISTENT_STATE.value]
    assert telem.semantic_execution_error is False


# Test F: True / True twice -> isolated semantic execution error
def test_permanent_execution_error_after_repeated_invalid_states(sample_response, sample_evidence):
    invalid_resp = json.dumps({"claim_id": "C1", "is_contradicted": True, "is_fully_established": True})

    provider = DummyChatProvider([invalid_resp, invalid_resp])
    verifier = StructuredSemanticCitationVerifierD31(provider, max_structured_output_retries=1)

    base_res, struct_res = verifier.verify_structured(sample_response, sample_evidence)
    assert struct_res.is_valid is False
    assert struct_res.execution_error_claims == ["C1"]
    assert struct_res.assessments[0].label == SemanticSupportLabel.INSUFFICIENT

    telem = struct_res.claim_telemetries["C1"]
    assert telem.provider_call_count == 2
    assert telem.retry_count == 1
    assert telem.draft_rejection_count == 2
    assert telem.semantic_execution_error is True


# Test G & N: Unrelated evidence / absence of evidence -> is_contradicted=False, is_fully_established=False -> INSUFFICIENT
def test_unrelated_evidence_insufficient(sample_response, sample_evidence):
    resp = json.dumps({"claim_id": "C1", "is_contradicted": False, "is_fully_established": False})
    provider = DummyChatProvider([resp])
    verifier = StructuredSemanticCitationVerifierD31(provider)

    base_res, struct_res = verifier.verify_structured(sample_response, sample_evidence)
    assert struct_res.assessments[0].label == SemanticSupportLabel.INSUFFICIENT
    assert struct_res.assessments[0].is_contradicted is False
    assert struct_res.assessments[0].is_fully_established is False


# Test H, I, J: Incompatible authority / inverted condition / conflicting quantity -> is_contradicted=True -> CONTRADICTED
def test_material_incompatibility_contradicted(sample_response, sample_evidence):
    resp = json.dumps({"claim_id": "C1", "is_contradicted": True, "is_fully_established": False})
    provider = DummyChatProvider([resp])
    verifier = StructuredSemanticCitationVerifierD31(provider)

    base_res, struct_res = verifier.verify_structured(sample_response, sample_evidence)
    assert struct_res.assessments[0].label == SemanticSupportLabel.CONTRADICTED
    assert struct_res.assessments[0].is_contradicted is True


# Test K, L: Partial evidence / rank mismatch -> is_contradicted=False, is_fully_established=False -> INSUFFICIENT
def test_partial_evidence_insufficient(sample_response, sample_evidence):
    resp = json.dumps({"claim_id": "C1", "is_contradicted": False, "is_fully_established": False})
    provider = DummyChatProvider([resp])
    verifier = StructuredSemanticCitationVerifierD31(provider)

    base_res, struct_res = verifier.verify_structured(sample_response, sample_evidence)
    assert struct_res.assessments[0].label == SemanticSupportLabel.INSUFFICIENT


# Test M: Valid statutory internal cross-reference -> is_fully_established=True -> SUPPORTED
def test_valid_cross_reference_supported(sample_response, sample_evidence):
    resp = json.dumps({"claim_id": "C1", "is_contradicted": False, "is_fully_established": True})
    provider = DummyChatProvider([resp])
    verifier = StructuredSemanticCitationVerifierD31(provider)

    base_res, struct_res = verifier.verify_structured(sample_response, sample_evidence)
    assert struct_res.assessments[0].label == SemanticSupportLabel.SUPPORTED


# Test O: One claim per provider invocation (sibling claims not mixed in same call)
def test_per_claim_isolated_invocation():
    ev1 = _make_dummy_evidence("E1")
    ev2 = _make_dummy_evidence("E2")
    multi_claim_resp = _make_dummy_response(
        answer="Doanh nghiệp nhà nước do Nhà nước nắm giữ 100% [E1]. Vốn điều lệ phải đăng ký [E2].",
        citations=[
            Citation(evidence_id="E1", chunk_id="chunk_E1", document_id="doc_001", document_title="Doc", document_number="01", article_number="1"),
            Citation(evidence_id="E2", chunk_id="chunk_E2", document_id="doc_001", document_title="Doc", document_number="01", article_number="2"),
        ],
    )
    r1 = json.dumps({"claim_id": "C1", "is_contradicted": False, "is_fully_established": True})
    r2 = json.dumps({"claim_id": "C2", "is_contradicted": False, "is_fully_established": True})

    provider = DummyChatProvider([r1, r2])
    verifier = StructuredSemanticCitationVerifierD31(provider)

    base_res, struct_res = verifier.verify_structured(multi_claim_resp, [ev1, ev2])
    assert len(struct_res.assessments) == 2
    assert len(provider.calls) == 2

    # Check that Call 1 only contains C1 and NOT C2
    assert "Claim ID: C1" in provider.calls[0]["user_prompt"]
    assert "Claim ID: C2" not in provider.calls[0]["user_prompt"]

    # Check that Call 2 only contains C2 and NOT C1
    assert "Claim ID: C2" in provider.calls[1]["user_prompt"]
    assert "Claim ID: C1" not in provider.calls[1]["user_prompt"]


# Test P: Sibling claim unaffected by permanent failure on another claim
def test_sibling_claim_unaffected_by_failure():
    ev1 = _make_dummy_evidence("E1")
    ev2 = _make_dummy_evidence("E2")
    multi_claim_resp = _make_dummy_response(
        answer="Doanh nghiệp nhà nước do Nhà nước nắm giữ 100% [E1]. Vốn điều lệ phải đăng ký [E2].",
        citations=[
            Citation(evidence_id="E1", chunk_id="chunk_E1", document_id="doc_001", document_title="Doc", document_number="01", article_number="1"),
            Citation(evidence_id="E2", chunk_id="chunk_E2", document_id="doc_001", document_title="Doc", document_number="01", article_number="2"),
        ],
    )
    # C1 fails repeatedly, C2 succeeds
    invalid_r = json.dumps({"claim_id": "C1", "is_contradicted": True, "is_fully_established": True})
    valid_r2 = json.dumps({"claim_id": "C2", "is_contradicted": False, "is_fully_established": True})

    provider = DummyChatProvider([invalid_r, invalid_r, valid_r2])
    verifier = StructuredSemanticCitationVerifierD31(provider, max_structured_output_retries=1)

    base_res, struct_res = verifier.verify_structured(multi_claim_resp, [ev1, ev2])
    assert struct_res.execution_error_claims == ["C1"]
    assert struct_res.assessments[0].claim_id == "C1"
    assert struct_res.assessments[0].label == SemanticSupportLabel.INSUFFICIENT
    assert struct_res.assessments[1].claim_id == "C2"
    assert struct_res.assessments[1].label == SemanticSupportLabel.SUPPORTED




# Test Q, R, S, T: Human labels, error tags, reference answer, D3 prediction absent from prompt
def test_prompt_blindness_to_labels_and_metadata(sample_response, sample_evidence):
    resp = json.dumps({"claim_id": "C1", "is_contradicted": False, "is_fully_established": True})
    provider = DummyChatProvider([resp])
    verifier = StructuredSemanticCitationVerifierD31(provider)

    verifier.verify_structured(sample_response, sample_evidence)
    call = provider.calls[0]
    prompt = call["user_prompt"]
    sys_inst = call["system_instruction"]

    for forbidden in [
        "human_label",
        "error_tag",
        "reference_answer",
        "gold_label",
        "V1 prediction",
        "D3 prediction",
        "D3.1 prediction",
        "102047:BASE:C1",
        "31883:PRIMARY:C1",
    ]:
        assert forbidden not in prompt
        assert forbidden not in sys_inst


# Test Extra Keys Rejected
def test_parser_rejects_extra_keys():
    verifier = StructuredSemanticCitationVerifierD31(DummyChatProvider([]))
    draft, rej_cat, rej_msg = verifier._parse_and_validate_draft(
        raw_completion=json.dumps({
            "claim_id": "C1",
            "is_contradicted": False,
            "is_fully_established": True,
            "extra_field": "forbidden",
        }),
        expected_claim_id="C1",
    )
    assert draft is None
    assert rej_cat == DraftRejectionCategoryD31.EXTRA_FIELD


# Test Non-Boolean Rejected
def test_parser_rejects_non_boolean_values():
    verifier = StructuredSemanticCitationVerifierD31(DummyChatProvider([]))
    draft, rej_cat, rej_msg = verifier._parse_and_validate_draft(
        raw_completion=json.dumps({
            "claim_id": "C1",
            "is_contradicted": "true",  # string instead of boolean
            "is_fully_established": True,
        }),
        expected_claim_id="C1",
    )
    assert draft is None
    assert rej_cat == DraftRejectionCategoryD31.NON_BOOLEAN_VALUE
