"""Strict, content-free recovery of safe structural model-output mistakes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import ValidationError

from legal_agentic_rag.schemas.answering import (
    MODEL_ANSWER_MAX_CLAIMS,
    ModelAnswerDraft,
)
from legal_agentic_rag.schemas.tools import (
    StructuredGenerationSchemaIssueCode,
    StructuredGenerationSchemaRecoveryOutcome,
    StructuredGenerationSchemaRepairCode,
)

_TOP_LEVEL_FIELDS = frozenset({"claims", "insufficient_evidence", "warnings"})
_CLAIM_FIELDS = frozenset({"text", "evidence_ids"})
_Value = TypeVar("_Value")


@dataclass(frozen=True)
class ModelAnswerSchemaRecoveryResult:
    """One bounded schema-recovery result with no model-authored content."""

    draft: ModelAnswerDraft | None
    attempted: bool
    outcome: StructuredGenerationSchemaRecoveryOutcome
    issue_codes: tuple[StructuredGenerationSchemaIssueCode, ...]
    repair_codes: tuple[StructuredGenerationSchemaRepairCode, ...]


def recover_terminal_model_answer_schema(
    payload: object,
    validation_error: ValidationError,
) -> ModelAnswerSchemaRecoveryResult:
    """Apply only allow-listed structural repairs then revalidate strictly.

    The caller invokes this after model retries are exhausted.  This function
    never changes claim text, invents field values, or returns raw payload data.
    """
    issue_codes = _classify_issues(payload, validation_error)
    if not isinstance(payload, dict):
        return _result(
            outcome=StructuredGenerationSchemaRecoveryOutcome.NOT_RECOVERABLE,
            issue_codes=issue_codes,
        )

    candidate = deepcopy(payload)
    repair_codes: list[StructuredGenerationSchemaRepairCode] = []

    if any(key not in _TOP_LEVEL_FIELDS for key in candidate):
        candidate = {
            key: value for key, value in candidate.items() if key in _TOP_LEVEL_FIELDS
        }
        repair_codes.append(
            StructuredGenerationSchemaRepairCode.REMOVED_TOP_LEVEL_EXTRA_FIELDS
        )

    claims = candidate.get("claims")
    if isinstance(claims, dict):
        claims = [claims]
        candidate["claims"] = claims
        repair_codes.append(StructuredGenerationSchemaRepairCode.WRAPPED_SINGLE_CLAIM)
    if isinstance(claims, list):
        if len(claims) > MODEL_ANSWER_MAX_CLAIMS:
            claims = claims[:MODEL_ANSWER_MAX_CLAIMS]
            candidate["claims"] = claims
            repair_codes.append(
                StructuredGenerationSchemaRepairCode.DROPPED_EXCESS_CLAIMS
            )
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                continue
            normalized_claim = claim
            if any(key not in _CLAIM_FIELDS for key in normalized_claim):
                normalized_claim = {
                    key: value
                    for key, value in normalized_claim.items()
                    if key in _CLAIM_FIELDS
                }
                claims[index] = normalized_claim
                repair_codes.append(
                    StructuredGenerationSchemaRepairCode.REMOVED_CLAIM_EXTRA_FIELDS
                )
            evidence_ids = normalized_claim.get("evidence_ids")
            if _is_valid_evidence_id(evidence_ids):
                evidence_ids = [evidence_ids]
                normalized_claim["evidence_ids"] = evidence_ids
                repair_codes.append(
                    StructuredGenerationSchemaRepairCode.WRAPPED_SCALAR_EVIDENCE_ID
                )
            if isinstance(evidence_ids, list) and _has_duplicates(evidence_ids):
                normalized_claim["evidence_ids"] = _deduplicate(evidence_ids)
                repair_codes.append(
                    StructuredGenerationSchemaRepairCode.DEDUPLICATED_EVIDENCE_IDS
                )

    warnings = candidate.get("warnings")
    if isinstance(warnings, list) and _has_duplicates(warnings):
        candidate["warnings"] = _deduplicate(warnings)
        repair_codes.append(
            StructuredGenerationSchemaRepairCode.DEDUPLICATED_WARNINGS
        )

    ordered_repairs = _ordered_unique(repair_codes)
    if not ordered_repairs:
        return _result(
            outcome=StructuredGenerationSchemaRecoveryOutcome.NOT_RECOVERABLE,
            issue_codes=issue_codes,
        )
    try:
        draft = ModelAnswerDraft.model_validate(candidate)
    except ValidationError:
        return _result(
            outcome=StructuredGenerationSchemaRecoveryOutcome.REVALIDATION_FAILED,
            issue_codes=issue_codes,
            repair_codes=ordered_repairs,
        )
    return ModelAnswerSchemaRecoveryResult(
        draft=draft,
        attempted=True,
        outcome=StructuredGenerationSchemaRecoveryOutcome.SUCCEEDED,
        issue_codes=issue_codes,
        repair_codes=ordered_repairs,
    )


def _classify_issues(
    payload: object,
    validation_error: ValidationError,
) -> tuple[StructuredGenerationSchemaIssueCode, ...]:
    """Classify only closed shape categories; never expose validation input."""
    codes: list[StructuredGenerationSchemaIssueCode] = []
    if not isinstance(payload, dict):
        codes.append(StructuredGenerationSchemaIssueCode.INVALID_TOP_LEVEL_TYPE)
    else:
        if any(key not in _TOP_LEVEL_FIELDS for key in payload):
            codes.append(StructuredGenerationSchemaIssueCode.TOP_LEVEL_EXTRA_FIELDS)
        if "insufficient_evidence" not in payload:
            codes.append(StructuredGenerationSchemaIssueCode.MISSING_REQUIRED_FIELD)
        claims = payload.get("claims")
        if isinstance(claims, dict):
            codes.append(
                StructuredGenerationSchemaIssueCode.CLAIMS_OBJECT_INSTEAD_OF_LIST
            )
            claim_values: list[object] | None = [claims]
        elif claims is not None and not isinstance(claims, list):
            codes.append(StructuredGenerationSchemaIssueCode.INVALID_CLAIM_TYPE)
            claim_values = None
        elif isinstance(claims, list):
            claim_values = claims
            if len(claims) > MODEL_ANSWER_MAX_CLAIMS:
                codes.append(StructuredGenerationSchemaIssueCode.CLAIM_LIMIT_EXCEEDED)
        else:
            claim_values = None
        if claim_values is not None:
            for claim in claim_values:
                if not isinstance(claim, dict):
                    codes.append(StructuredGenerationSchemaIssueCode.INVALID_CLAIM_TYPE)
                    continue
                if any(key not in _CLAIM_FIELDS for key in claim):
                    codes.append(StructuredGenerationSchemaIssueCode.CLAIM_EXTRA_FIELDS)
                if "text" in claim and not isinstance(claim["text"], str):
                    codes.append(StructuredGenerationSchemaIssueCode.INVALID_CLAIM_TEXT)
                evidence_ids = claim.get("evidence_ids")
                if _is_valid_evidence_id(evidence_ids):
                    codes.append(
                        StructuredGenerationSchemaIssueCode.CLAIM_EVIDENCE_ID_SCALAR
                    )
                elif evidence_ids is not None and not isinstance(evidence_ids, list):
                    codes.append(
                        StructuredGenerationSchemaIssueCode.INVALID_CLAIM_EVIDENCE_IDS
                    )
                elif isinstance(evidence_ids, list):
                    if _has_duplicates(evidence_ids):
                        codes.append(
                            StructuredGenerationSchemaIssueCode.DUPLICATE_CLAIM_EVIDENCE_IDS
                        )
        warnings = payload.get("warnings")
        if warnings is not None and not isinstance(warnings, list):
            codes.append(StructuredGenerationSchemaIssueCode.INVALID_WARNINGS)
        elif isinstance(warnings, list) and _has_duplicates(warnings):
            codes.append(StructuredGenerationSchemaIssueCode.DUPLICATE_WARNINGS)
        if _has_grounding_state_mismatch(payload):
            codes.append(StructuredGenerationSchemaIssueCode.GROUNDING_STATE_MISMATCH)

    for error in validation_error.errors(include_input=False, include_url=False):
        error_type = error.get("type")
        location = error.get("loc")
        if error_type == "missing":
            codes.append(StructuredGenerationSchemaIssueCode.MISSING_REQUIRED_FIELD)
        elif location == () and error_type == "value_error":
            codes.append(StructuredGenerationSchemaIssueCode.GROUNDING_STATE_MISMATCH)

    if not codes:
        codes.append(StructuredGenerationSchemaIssueCode.OTHER_SCHEMA_VALIDATION_ERROR)
    return _ordered_unique(codes)


def _has_grounding_state_mismatch(payload: dict[str, Any]) -> bool:
    """Recognize only explicit boolean/claim-presence disagreement."""
    insufficient = payload.get("insufficient_evidence")
    claims = payload.get("claims")
    if not isinstance(insufficient, bool) or not isinstance(claims, list):
        return False
    return (insufficient and bool(claims)) or (not insufficient and not claims)


def _is_valid_evidence_id(value: object) -> bool:
    """Accept the scalar form only if it already satisfies E<number>."""
    return (
        isinstance(value, str)
        and len(value) > 1
        and value.startswith("E")
        and value[1:].isdigit()
        and value[1] != "0"
    )


def _has_duplicates(values: list[Any]) -> bool:
    """Detect duplicate scalar values without assuming hashability."""
    return len(values) != len(_deduplicate(values))


def _deduplicate(values: list[Any]) -> list[Any]:
    """Keep first occurrences while supporting only JSON-compatible values."""
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _ordered_unique(values: list[_Value]) -> tuple[_Value, ...]:
    """Return deterministic unique enum values in first-observed order."""
    return tuple(dict.fromkeys(values))


def _result(
    *,
    outcome: StructuredGenerationSchemaRecoveryOutcome,
    issue_codes: tuple[StructuredGenerationSchemaIssueCode, ...],
    repair_codes: tuple[StructuredGenerationSchemaRepairCode, ...] = (),
) -> ModelAnswerSchemaRecoveryResult:
    """Construct a terminal no-draft recovery result."""
    return ModelAnswerSchemaRecoveryResult(
        draft=None,
        attempted=True,
        outcome=outcome,
        issue_codes=issue_codes,
        repair_codes=repair_codes,
    )
