"""Unit tests for StructuredSemanticCitationVerifier and deterministic label derivation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import ModelError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
from legal_agentic_rag.generation.structured_semantic_verifier import (
    EvidenceCoverageStatus,
    SemanticDimensionStatus,
    StructuredClaimAssessmentDraft,
    StructuredSemanticCitationVerifier,
    StructuredSemanticVerificationDraft,
    derive_claim_semantic_label,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
    SemanticSupportLabel,
)


def _make_assessment_draft(
    claim_id: str = "C1",
    actor_role: SemanticDimensionStatus = SemanticDimensionStatus.MATCH,
    action_object: SemanticDimensionStatus = SemanticDimensionStatus.MATCH,
    condition_exception: SemanticDimensionStatus = SemanticDimensionStatus.MATCH,
    quantity_temporal: SemanticDimensionStatus = SemanticDimensionStatus.MATCH,
    negation_modality: SemanticDimensionStatus = SemanticDimensionStatus.MATCH,
    source_article_scope: SemanticDimensionStatus = SemanticDimensionStatus.MATCH,
    evidence_coverage: EvidenceCoverageStatus = EvidenceCoverageStatus.COMPLETE,
) -> StructuredClaimAssessmentDraft:
    return StructuredClaimAssessmentDraft(
        claim_id=claim_id,
        actor_role=actor_role,
        action_object=action_object,
        condition_exception=condition_exception,
        quantity_temporal=quantity_temporal,
        negation_modality=negation_modality,
        source_article_scope=source_article_scope,
        evidence_coverage=evidence_coverage,
    )


class MockChatProvider(ChatModelProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.system_instructions: list[str] = []
        self.user_prompts: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock_provider"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def model_name(self) -> str:
        return "Qwen/Qwen2.5-3B-Instruct"

    @property
    def model_revision(self) -> str:
        return "a1d308dfcc03e09da285d49d912439a655a571e8"

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        self.system_instructions.append(system_instruction)
        self.user_prompts.append(user_prompt)
        res = self.responses[self.call_count]
        self.call_count += 1
        return res


def test_derive_claim_semantic_label_all_match():
    """Verify that all MATCH dimensions and COMPLETE coverage yield SUPPORTED."""
    draft = _make_assessment_draft()
    assert derive_claim_semantic_label(draft) == SemanticSupportLabel.SUPPORTED


def test_derive_claim_semantic_label_not_applicable():
    """Verify that NOT_APPLICABLE dimensions do not prevent SUPPORTED label."""
    draft = _make_assessment_draft(
        actor_role=SemanticDimensionStatus.NOT_APPLICABLE,
        condition_exception=SemanticDimensionStatus.NOT_APPLICABLE,
        quantity_temporal=SemanticDimensionStatus.NOT_APPLICABLE,
        negation_modality=SemanticDimensionStatus.NOT_APPLICABLE,
    )
    assert derive_claim_semantic_label(draft) == SemanticSupportLabel.SUPPORTED


@pytest.mark.parametrize(
    "dimension_field",
    [
        "actor_role",
        "action_object",
        "condition_exception",
        "quantity_temporal",
        "negation_modality",
        "source_article_scope",
    ],
)
def test_derive_claim_semantic_label_conflict_yields_contradicted(dimension_field: str):
    """Verify that ANY CONFLICT dimension yields CONTRADICTED regardless of others."""
    kwargs = {dimension_field: SemanticDimensionStatus.CONFLICT}
    draft = _make_assessment_draft(**kwargs)
    assert derive_claim_semantic_label(draft) == SemanticSupportLabel.CONTRADICTED


@pytest.mark.parametrize(
    "dimension_field",
    [
        "actor_role",
        "action_object",
        "condition_exception",
        "quantity_temporal",
        "negation_modality",
        "source_article_scope",
    ],
)
def test_derive_claim_semantic_label_insufficient_dimension_yields_insufficient(dimension_field: str):
    """Verify that an INSUFFICIENT dimension without CONFLICT yields INSUFFICIENT."""
    kwargs = {dimension_field: SemanticDimensionStatus.INSUFFICIENT}
    draft = _make_assessment_draft(**kwargs)
    assert derive_claim_semantic_label(draft) == SemanticSupportLabel.INSUFFICIENT


@pytest.mark.parametrize(
    "coverage",
    [
        EvidenceCoverageStatus.PARTIAL,
        EvidenceCoverageStatus.NONE,
    ],
)
def test_derive_claim_semantic_label_incomplete_coverage_yields_insufficient(coverage: EvidenceCoverageStatus):
    """Verify that PARTIAL or NONE evidence coverage yields INSUFFICIENT even if all dimensions MATCH."""
    draft = _make_assessment_draft(evidence_coverage=coverage)
    assert derive_claim_semantic_label(draft) == SemanticSupportLabel.INSUFFICIENT


def test_derive_claim_semantic_label_conflict_precedence_over_insufficient():
    """Verify that CONFLICT takes strict precedence over INSUFFICIENT and incomplete coverage."""
    draft = _make_assessment_draft(
        actor_role=SemanticDimensionStatus.CONFLICT,
        condition_exception=SemanticDimensionStatus.INSUFFICIENT,
        evidence_coverage=EvidenceCoverageStatus.NONE,
    )
    # Even though coverage is NONE and condition is INSUFFICIENT, CONFLICT takes highest precedence
    assert derive_claim_semantic_label(draft) == SemanticSupportLabel.CONTRADICTED


def test_derive_claim_semantic_label_numeric_role_mismatch():
    """Verify numeric mismatch (same token but wrong semantic role) marked as CONFLICT/INSUFFICIENT."""
    draft_conflict = _make_assessment_draft(quantity_temporal=SemanticDimensionStatus.CONFLICT)
    assert derive_claim_semantic_label(draft_conflict) == SemanticSupportLabel.CONTRADICTED

    draft_insufficient = _make_assessment_draft(quantity_temporal=SemanticDimensionStatus.INSUFFICIENT)
    assert derive_claim_semantic_label(draft_insufficient) == SemanticSupportLabel.INSUFFICIENT


def test_structured_verifier_verify_success():
    """Verify end-to-end execution of StructuredSemanticCitationVerifier on clean input."""
    base_verifier = RuleBasedCitationVerifier()

    mock_response = AnswerResponse(
        question="Thời hạn cấp giấy phép là bao lâu?",
        answer="Thời hạn cấp giấy phép là 15 ngày làm việc [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_001",
                document_id="doc_001",
                document_title="Luật Đầu tư",
                document_number="61/2020/QH14",
                article_number="15",
            )
        ],
    )
    evidence = [
        Evidence(
            evidence_id="E1",
            chunk_id="chunk_001",
            document_id="doc_001",
            text="Thời hạn cấp giấy phép là 15 ngày làm việc kể từ ngày nhận đủ hồ sơ.",
            document_title="Luật Đầu tư",
            document_number="61/2020/QH14",
            article_number="15",
        )
    ]

    model_output = json.dumps(
        {
            "assessments": [
                {
                    "claim_id": "C1",
                    "actor_role": "MATCH",
                    "action_object": "MATCH",
                    "condition_exception": "MATCH",
                    "quantity_temporal": "MATCH",
                    "negation_modality": "MATCH",
                    "source_article_scope": "MATCH",
                    "evidence_coverage": "COMPLETE",
                }
            ]
        }
    )

    provider = MockChatProvider([model_output])
    verifier = StructuredSemanticCitationVerifier(base_verifier, provider)

    citation_res, structured_res = verifier.verify_structured(mock_response, evidence)

    assert citation_res.is_valid is True
    assert citation_res.semantic_verification is not None
    assert citation_res.semantic_verification.is_valid is True
    assert len(citation_res.semantic_verification.assessments) == 1
    assert citation_res.semantic_verification.assessments[0].label == SemanticSupportLabel.SUPPORTED

    assert structured_res is not None
    assert structured_res.is_valid is True
    assert structured_res.assessments[0].actor_role == SemanticDimensionStatus.MATCH
    assert structured_res.assessments[0].evidence_coverage == EvidenceCoverageStatus.COMPLETE

    # Verify prompt contents: question and claims present, reference answer and human labels strictly absent
    prompt = provider.user_prompts[0]
    assert "QUESTION:\nThời hạn cấp giấy phép là bao lâu?" in prompt
    assert "C1" in prompt
    assert "Luật Đầu tư" in prompt
    assert "reference_answer" not in prompt
    assert "human_label" not in prompt
    assert "error_tag" not in prompt


def test_structured_verifier_retries_on_malformed_json():
    """Verify that verifier retries once on malformed JSON and succeeds if second attempt is valid."""
    base_verifier = RuleBasedCitationVerifier()

    mock_response = AnswerResponse(
        question="Thời hạn cấp giấy phép là bao lâu?",
        answer="Thời hạn cấp giấy phép là 15 ngày làm việc [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_001",
                document_id="doc_001",
                document_title="Luật Đầu tư",
                document_number="61/2020/QH14",
                article_number="15",
            )
        ],
    )
    evidence = [
        Evidence(
            evidence_id="E1",
            chunk_id="chunk_001",
            document_id="doc_001",
            text="Thời hạn cấp giấy phép là 15 ngày làm việc kể từ ngày nhận đủ hồ sơ.",
            document_title="Luật Đầu tư",
            document_number="61/2020/QH14",
            article_number="15",
        )
    ]

    malformed_output = "THIS IS NOT JSON"
    valid_output = json.dumps(
        {
            "assessments": [
                {
                    "claim_id": "C1",
                    "actor_role": "MATCH",
                    "action_object": "MATCH",
                    "condition_exception": "MATCH",
                    "quantity_temporal": "MATCH",
                    "negation_modality": "MATCH",
                    "source_article_scope": "MATCH",
                    "evidence_coverage": "COMPLETE",
                }
            ]
        }
    )

    provider = MockChatProvider([malformed_output, valid_output])
    verifier = StructuredSemanticCitationVerifier(base_verifier, provider, max_structured_output_retries=1)

    citation_res, structured_res = verifier.verify_structured(mock_response, evidence)

    assert provider.call_count == 2
    assert citation_res.is_valid is True
    assert structured_res is not None
    assert structured_res.is_valid is True


def test_structured_verifier_raises_model_error_on_permanent_malformed_json():
    """Verify that ModelError is raised if all retry attempts produce malformed output."""
    base_verifier = RuleBasedCitationVerifier()

    mock_response = AnswerResponse(
        question="Thời hạn cấp giấy phép là bao lâu?",
        answer="Thời hạn cấp giấy phép là 15 ngày làm việc [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_001",
                document_id="doc_001",
                document_title="Luật Đầu tư",
                document_number="61/2020/QH14",
                article_number="15",
            )
        ],
    )
    evidence = [
        Evidence(
            evidence_id="E1",
            chunk_id="chunk_001",
            document_id="doc_001",
            text="Thời hạn cấp giấy phép là 15 ngày làm việc kể từ ngày nhận đủ hồ sơ.",
            document_title="Luật Đầu tư",
            document_number="61/2020/QH14",
            article_number="15",
        )
    ]

    provider = MockChatProvider(["MALFORMED 1", "MALFORMED 2"])
    verifier = StructuredSemanticCitationVerifier(base_verifier, provider, max_structured_output_retries=1)

    with pytest.raises(ModelError, match="does not match the structured semantic verification schema"):
        verifier.verify_structured(mock_response, evidence)


def test_structured_verifier_raises_on_claim_id_or_order_mismatch():
    """Verify that ModelError is raised if claim IDs or order do not match expected claims."""
    base_verifier = RuleBasedCitationVerifier()

    mock_response = AnswerResponse(
        question="Thời hạn cấp giấy phép là bao lâu?",
        answer="Thời hạn cấp giấy phép là 15 ngày làm việc [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_001",
                document_id="doc_001",
                document_title="Luật Đầu tư",
                document_number="61/2020/QH14",
                article_number="15",
            )
        ],
    )
    evidence = [
        Evidence(
            evidence_id="E1",
            chunk_id="chunk_001",
            document_id="doc_001",
            text="Thời hạn cấp giấy phép là 15 ngày làm việc kể từ ngày nhận đủ hồ sơ.",
            document_title="Luật Đầu tư",
            document_number="61/2020/QH14",
            article_number="15",
        )
    ]

    # Model returned assessment for 'C99' instead of 'C1'
    mismatched_output = json.dumps(
        {
            "assessments": [
                {
                    "claim_id": "C99",
                    "actor_role": "MATCH",
                    "action_object": "MATCH",
                    "condition_exception": "MATCH",
                    "quantity_temporal": "MATCH",
                    "negation_modality": "MATCH",
                    "source_article_scope": "MATCH",
                    "evidence_coverage": "COMPLETE",
                }
            ]
        }
    )

    provider = MockChatProvider([mismatched_output, mismatched_output])
    verifier = StructuredSemanticCitationVerifier(base_verifier, provider, max_structured_output_retries=1)

    with pytest.raises(ModelError, match="assess every supplied claim exactly once in supplied order"):
        verifier.verify_structured(mock_response, evidence)
