"""Unit tests for bounded structural model-answer recovery."""

from __future__ import annotations

from copy import deepcopy

from pydantic import ValidationError

from legal_agentic_rag.generation.schema_recovery import (
    recover_terminal_model_answer_schema,
)
from legal_agentic_rag.schemas import ModelAnswerDraft
from legal_agentic_rag.schemas.tools import (
    StructuredGenerationSchemaIssueCode,
    StructuredGenerationSchemaRecoveryOutcome,
    StructuredGenerationSchemaRepairCode,
)


def _claim(*, evidence_ids: object = None) -> dict[str, object]:
    return {
        "text": "Noi dung phap ly giu nguyen.",
        "evidence_ids": ["E1"] if evidence_ids is None else evidence_ids,
    }


def _validation_error(payload: object) -> ValidationError:
    try:
        ModelAnswerDraft.model_validate(payload)
    except ValidationError as error:
        return error
    raise AssertionError("fixture must violate ModelAnswerDraft")


def test_recovery_normalizes_only_safe_shape_errors_without_mutating_input() -> None:
    """Extra fields, scalar IDs and duplicates are recoverable shape mistakes."""
    payload = {
        "claims": {**_claim(evidence_ids="E1"), "discard": "never persist"},
        "insufficient_evidence": False,
        "warnings": ["warning", "warning"],
        "unexpected": "never persist",
    }
    original = deepcopy(payload)

    result = recover_terminal_model_answer_schema(payload, _validation_error(payload))

    assert result.outcome is StructuredGenerationSchemaRecoveryOutcome.SUCCEEDED
    assert result.draft is not None
    assert result.draft.claims[0].text == "Noi dung phap ly giu nguyen."
    assert result.draft.claims[0].evidence_ids == ["E1"]
    assert result.draft.warnings == ["warning"]
    assert StructuredGenerationSchemaIssueCode.TOP_LEVEL_EXTRA_FIELDS in result.issue_codes
    assert StructuredGenerationSchemaIssueCode.CLAIM_EVIDENCE_ID_SCALAR in result.issue_codes
    assert StructuredGenerationSchemaRepairCode.REMOVED_TOP_LEVEL_EXTRA_FIELDS in result.repair_codes
    assert StructuredGenerationSchemaRepairCode.WRAPPED_SINGLE_CLAIM in result.repair_codes
    assert payload == original


def test_recovery_keeps_only_whole_claims_when_the_schema_limit_is_exceeded() -> None:
    """Recovery never truncates legal claim text, only complete excess records."""
    payload = {
        "claims": [_claim() for _ in range(5)],
        "insufficient_evidence": False,
        "warnings": [],
    }

    result = recover_terminal_model_answer_schema(payload, _validation_error(payload))

    assert result.outcome is StructuredGenerationSchemaRecoveryOutcome.SUCCEEDED
    assert result.draft is not None
    assert len(result.draft.claims) == 4
    assert result.draft.claims[0].text == payload["claims"][0]["text"]
    assert StructuredGenerationSchemaRepairCode.DROPPED_EXCESS_CLAIMS in result.repair_codes


def test_recovery_refuses_to_invent_missing_or_semantic_values() -> None:
    """Missing state, invalid IDs and contradictory states remain fail-closed."""
    payloads = [
        {"claims": [_claim()], "warnings": []},
        {"claims": [_claim(evidence_ids="bad")], "insufficient_evidence": False},
        {"claims": [_claim()], "insufficient_evidence": True, "warnings": []},
    ]

    outcomes = [
        recover_terminal_model_answer_schema(payload, _validation_error(payload))
        for payload in payloads
    ]

    assert all(result.draft is None for result in outcomes)
    assert all(
        result.outcome is StructuredGenerationSchemaRecoveryOutcome.NOT_RECOVERABLE
        for result in outcomes
    )
    assert StructuredGenerationSchemaIssueCode.MISSING_REQUIRED_FIELD in outcomes[0].issue_codes
    assert StructuredGenerationSchemaIssueCode.INVALID_CLAIM_EVIDENCE_IDS in outcomes[1].issue_codes
    assert StructuredGenerationSchemaIssueCode.GROUNDING_STATE_MISMATCH in outcomes[2].issue_codes


def test_recovery_does_not_truncate_invalid_claim_text() -> None:
    """A too-long claim remains rejected after unrelated safe normalization."""
    payload = {
        "claims": [{**_claim(), "text": "x" * 601, "extra": True}],
        "insufficient_evidence": False,
        "warnings": [],
    }

    result = recover_terminal_model_answer_schema(payload, _validation_error(payload))

    assert result.draft is None
    assert result.outcome is StructuredGenerationSchemaRecoveryOutcome.REVALIDATION_FAILED
    assert StructuredGenerationSchemaRepairCode.REMOVED_CLAIM_EXTRA_FIELDS in result.repair_codes
