"""Tests for two-stage model-backed semantic claim verification."""

from dataclasses import dataclass, field

import pytest

from legal_agentic_rag.configuration import (
    ClaimVerificationConfig,
    SemanticVerificationConfig,
)
from legal_agentic_rag.contracts import CitationVerifier
from legal_agentic_rag.exceptions import ModelError
from legal_agentic_rag.generation import (
    ModelBackedCitationVerifier,
    RuleBasedCitationVerifier,
    build_citation_verifier,
)
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    Evidence,
    RetrievalStrategy,
)


@dataclass
class _FixtureProvider:
    completions: list[str]
    provider_name: str = "fixture"
    provider_version: str = "1.0"
    model_name: str = "fixture-verifier"
    model_revision: str = "fixture-revision"
    prompts: list[str] = field(default_factory=list)

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        self.prompts.append(f"{system_instruction}\n{user_prompt}")
        return self.completions.pop(0)


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="E1",
        chunk_id="chunk-semantic",
        document_id="doc-semantic",
        document_title="Bộ luật thử nghiệm",
        document_number="01/2026/QH",
        article_number="10",
        text=(
            "Người lao động làm đủ 12 tháng được nghỉ hằng năm "
            "12 ngày làm việc."
        ),
    )


def _response(
    *,
    answer: str = (
        "Người lao động làm đủ 12 tháng được nghỉ hằng năm "
        "12 ngày làm việc. [E1]"
    ),
) -> AnswerResponse:
    evidence = _evidence()
    return AnswerResponse(
        question="Người lao động được nghỉ hằng năm bao nhiêu ngày?",
        answer=answer,
        citations=[
            Citation(
                evidence_id=evidence.evidence_id,
                chunk_id=evidence.chunk_id,
                document_id=evidence.document_id,
                document_title=evidence.document_title,
                document_number=evidence.document_number,
                article_number=evidence.article_number,
            )
        ],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="semantic-query",
        metadata={"semantic_synthesis": True},
    )


def _verifier(
    provider: _FixtureProvider,
    *,
    retries: int = 0,
) -> ModelBackedCitationVerifier:
    return ModelBackedCitationVerifier(
        RuleBasedCitationVerifier(),
        provider,
        max_structured_output_retries=retries,
    )


def test_semantic_verifier_accepts_supported_claim_with_provenance() -> None:
    """A complete supported assessment preserves trusted evidence identity."""
    provider = _FixtureProvider(
        ['{"assessments":[{"claim_id":"C1","label":"supported"}]}']
    )

    result = _verifier(provider).verify(_response(), [_evidence()])

    assert result.is_valid is True
    assert result.semantic_verification is not None
    assert result.semantic_verification.model_name == "fixture-verifier"
    assessment = result.semantic_verification.assessments[0]
    assert assessment.claim_id == "C1"
    assert assessment.evidence_ids == ["E1"]
    assert assessment.label.value == "supported"
    assert "semantic_entailment_not_verified" not in result.warnings
    assert "CLAIMS_AND_CITED_EVIDENCE_JSON" in provider.prompts[0]


def test_semantic_verifier_rejects_contradicted_or_insufficient_claim() -> None:
    """Any non-supported semantic label makes the aggregate result invalid."""
    for label in ("contradicted", "insufficient"):
        provider = _FixtureProvider(
            [
                (
                    '{"assessments":[{"claim_id":"C1","label":'
                    f'"{label}"'
                    "}]}"
                )
            ]
        )

        result = _verifier(provider).verify(_response(), [_evidence()])

        assert result.is_valid is False
        assert f"semantic_{label}:C1" in result.errors


def test_semantic_verifier_retries_invalid_structured_output_once() -> None:
    """A single bounded retry can repair malformed model JSON."""
    provider = _FixtureProvider(
        [
            "not-json",
            '{"assessments":[{"claim_id":"C1","label":"supported"}]}',
        ]
    )

    result = _verifier(provider, retries=1).verify(
        _response(),
        [_evidence()],
    )

    assert result.is_valid is True
    assert len(provider.prompts) == 2
    assert "previous output was invalid" in provider.prompts[1]


def test_semantic_verifier_rejects_missing_or_reordered_claims() -> None:
    """The model cannot omit, invent, duplicate, or reorder claim identities."""
    provider = _FixtureProvider(['{"assessments":[]}'])

    with pytest.raises(ModelError, match="semantic verification schema"):
        _verifier(provider).verify(_response(), [_evidence()])


def test_semantic_verifier_skips_model_when_hard_checks_fail() -> None:
    """Deterministic failures short-circuit the optional model boundary."""
    provider = _FixtureProvider(
        ['{"assessments":[{"claim_id":"C1","label":"supported"}]}']
    )
    response = _response(answer="Được nghỉ 99 ngày. [E1]")

    result = _verifier(provider).verify(response, [_evidence()])

    assert result.is_valid is False
    assert provider.prompts == []
    assert "semantic_verification_skipped_hard_failure" in result.warnings


def test_factory_keeps_semantic_model_disabled_by_default() -> None:
    """Default assembly remains local, deterministic, and GPU-independent."""
    disabled = build_citation_verifier(
        ClaimVerificationConfig(),
        SemanticVerificationConfig(),
    )
    provider = _FixtureProvider(
        ['{"assessments":[{"claim_id":"C1","label":"supported"}]}']
    )
    enabled = build_citation_verifier(
        ClaimVerificationConfig(),
        SemanticVerificationConfig(
            backend="transformers",
            model_name="fixture-verifier",
            model_revision="fixture-revision",
        ),
        provider=provider,
    )

    assert isinstance(disabled, RuleBasedCitationVerifier)
    assert isinstance(enabled, ModelBackedCitationVerifier)
    assert isinstance(enabled, CitationVerifier)
