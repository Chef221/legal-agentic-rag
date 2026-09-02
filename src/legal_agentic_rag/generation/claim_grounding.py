"""Deterministic claim-to-evidence grounding without semantic overclaiming."""

from __future__ import annotations

from collections.abc import Sequence
import re

from legal_agentic_rag.configuration.online import ClaimVerificationConfig
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
)

_BRACKET_CONTENT_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_EVIDENCE_ID_PATTERN = re.compile(r"\bE[1-9][0-9]*\b")
_SENTENCE_PATTERN = re.compile(r".+?(?:[.!?;](?=\s|$)|$)")
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*%?")
_MARKER_ONLY_PATTERN = re.compile(
    r"^(?:\s*\[(?:[^\[\]]*E[1-9][0-9]*[^\[\]]*)\]\s*)+$"
)
_LEADING_LIST_PATTERN = re.compile(r"^(?:[-•]\s+|\d+[.)]\s+)")
_STOPWORDS = {
    "các",
    "cho",
    "của",
    "là",
    "một",
    "những",
    "theo",
    "trong",
    "tại",
    "và",
    "về",
}
_LEGAL_PREFIX_RE = (
    r"(?:bộ\s+luật|luật|nghị\s+định|"
    r"thông\s+tư|quyết\s+định|"
    r"nghị\s+quyết|pháp\s+lệnh)"
)
_NUMBERED_LEGAL_REF_PATTERN = re.compile(
    rf"\b(?:{_LEGAL_PREFIX_RE}(?:\s+số)?)?\s*([0-9]+/[0-9]+/[A-Za-z0-9_Đđ-]+)\b",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_LEGAL_INSTRUMENT_PATTERN = re.compile(
    rf"\b({_LEGAL_PREFIX_RE}\s+[A-ZÀ-ỹĐ][A-Za-zÀ-ỹĐđ0-9_-]*(?:\s+[A-Za-zÀ-ỹĐđ0-9_-]+){{0,5}})",
    re.UNICODE | re.IGNORECASE,
)
_LEGAL_BOUNDARY_STOP_WORDS = {
    "năm",
    "quy",
    "định",
    "ngày",
    "thì",
    "là",
    "và",
    "hoặc",
    "khi",
    "được",
    "của",
    "tại",
    "về",
    "có",
    "phải",
    "không",
    "cho",
    "theo",
    "nhằm",
    "do",
    "để",
    "sao",
    "nếu",
    "bị",
    "gồm",
    "như",
    "trong",
}
_NEGATION_TERMS = {
    "bãi",
    "cấm",
    "chưa",
    "hủy",
    "không",
    "ngoại",
    "trừ",
}


class RuleBasedClaimGroundingVerifier:
    """Check inline markers and conservative lexical support per answer claim."""

    def __init__(
        self,
        config: ClaimVerificationConfig | None = None,
    ) -> None:
        self._config = config or ClaimVerificationConfig()

    def verify(
        self,
        response: AnswerResponse,
        evidence: Sequence[Evidence],
    ) -> tuple[list[ClaimVerification], float | None, list[str], list[str]]:
        """Return claim results, coverage, errors, and explicit limitations."""
        if response.insufficient_evidence:
            return [], None, [], ["claim_verification_not_applicable_abstention"]
        if not self._config.enabled:
            return [], None, [], ["claim_verification_disabled"]
        if response.metadata.get("semantic_synthesis") is False:
            return [], None, [], ["claim_verification_not_applicable_extractive"]

        evidence_by_id = {item.evidence_id: item for item in evidence}
        citation_ids = {item.evidence_id for item in response.citations}
        segments = self._claims(response.answer)
        errors: list[str] = []
        if len(segments) > self._config.max_claims:
            errors.append("claim_count_exceeded")
            segments = segments[: self._config.max_claims]
        if not segments:
            return (
                [],
                None,
                ["grounded_answer_has_no_claims"],
                ["semantic_entailment_not_verified"],
            )

        results: list[ClaimVerification] = []
        used_evidence_ids: set[str] = set()
        for index, segment in enumerate(segments, start=1):
            result = self._verify_claim(
                claim_id=f"C{index}",
                segment=segment,
                evidence_by_id=evidence_by_id,
                citation_ids=citation_ids,
            )
            results.append(result)
            used_evidence_ids.update(result.evidence_ids)
            if result.status == ClaimSupportStatus.UNSUPPORTED:
                errors.append(f"unsupported_claim:{result.claim_id}")

        for evidence_id in sorted(citation_ids - used_evidence_ids):
            errors.append(f"citation_not_used_in_answer:{evidence_id}")
        supported_count = sum(
            item.status == ClaimSupportStatus.SUPPORTED for item in results
        )
        coverage = supported_count / len(results)
        return (
            results,
            coverage,
            list(dict.fromkeys(errors)),
            ["semantic_entailment_not_verified"],
        )

    def _verify_claim(
        self,
        *,
        claim_id: str,
        segment: str,
        evidence_by_id: dict[str, Evidence],
        citation_ids: set[str],
    ) -> ClaimVerification:
        marker_ids = self._marker_ids(segment)
        claim_text = self._claim_text(segment)
        errors: list[str] = []
        if self._config.require_inline_citations and not marker_ids:
            errors.append("missing_inline_evidence")
        if not marker_ids and not self._config.require_inline_citations:
            marker_ids = sorted(citation_ids)
        unknown_ids = [
            value for value in marker_ids if value not in evidence_by_id
        ]
        uncited_ids = [
            value for value in marker_ids if value not in citation_ids
        ]
        if unknown_ids:
            errors.append("unknown_evidence_marker")
        if uncited_ids:
            errors.append("answer_marker_missing_citation")
        linked = [
            evidence_by_id[value]
            for value in marker_ids
            if value in evidence_by_id and value in citation_ids
        ]
        claim_terms = self._content_terms(claim_text)
        if len(claim_terms) < self._config.minimum_claim_tokens:
            errors.append("claim_too_short")
        evidence_text = " ".join(self._evidence_text(item) for item in linked)
        evidence_terms = self._content_terms(evidence_text)
        lexical_support = (
            len(claim_terms & evidence_terms) / len(claim_terms)
            if claim_terms
            else 0.0
        )
        if (
            linked
            and lexical_support < self._config.minimum_lexical_support
        ):
            errors.append("insufficient_lexical_support")
        if not linked:
            errors.append("no_linked_evidence")

        claim_numbers = set(_NUMBER_PATTERN.findall(claim_text.casefold()))
        evidence_numbers = set(
            _NUMBER_PATTERN.findall(evidence_text.casefold())
        )
        numeric_match = claim_numbers.issubset(evidence_numbers)
        if self._config.require_numeric_match and not numeric_match:
            errors.append("numeric_mismatch")

        claim_negations = self._negations(claim_text)
        evidence_negations = self._negations(evidence_text)
        negation_match = claim_negations.issubset(evidence_negations)
        if self._config.require_negation_match and not negation_match:
            errors.append("negation_mismatch")

        if linked and not self._check_legal_references(claim_text, linked):
            errors.append("legal_reference_mismatch")

        unique_errors = list(dict.fromkeys(errors))
        return ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_ids=marker_ids,
            status=(
                ClaimSupportStatus.SUPPORTED
                if not unique_errors
                else ClaimSupportStatus.UNSUPPORTED
            ),
            lexical_support_score=lexical_support,
            numeric_match=numeric_match,
            negation_match=negation_match,
            errors=unique_errors,
        )

    @staticmethod
    def _claims(answer: str) -> list[str]:
        segments: list[str] = []
        for line in answer.splitlines():
            normalized = line.strip()
            if not normalized:
                continue
            for match in _SENTENCE_PATTERN.finditer(normalized):
                value = match.group(0).strip()
                if not value:
                    continue
                if _MARKER_ONLY_PATTERN.fullmatch(value) and segments:
                    segments[-1] = f"{segments[-1]} {value}"
                else:
                    segments.append(value)
        return segments

    @staticmethod
    def _marker_ids(value: str) -> list[str]:
        markers: list[str] = []
        for content in _BRACKET_CONTENT_PATTERN.findall(value):
            markers.extend(_EVIDENCE_ID_PATTERN.findall(content))
        return list(dict.fromkeys(markers))

    @staticmethod
    def _claim_text(value: str) -> str:
        without_markers = _BRACKET_CONTENT_PATTERN.sub("", value).strip()
        without_prefix = _LEADING_LIST_PATTERN.sub("", without_markers)
        normalized = without_prefix.strip()
        return normalized or "claim_without_visible_text"

    @staticmethod
    def _content_terms(value: str) -> set[str]:
        return {
            token
            for token in (
                raw.casefold() for raw in _TOKEN_PATTERN.findall(value)
            )
            if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
        }

    @staticmethod
    def _negations(value: str) -> set[str]:
        return RuleBasedClaimGroundingVerifier._content_terms(value) & (
            _NEGATION_TERMS
        )

    @staticmethod
    def _extract_legal_references(text: str) -> list[str]:
        references: list[str] = []
        for match in _NUMBERED_LEGAL_REF_PATTERN.finditer(text):
            doc_num = match.group(1).strip().casefold()
            if "/" in doc_num:
                references.append(doc_num)

        for match in _NAMED_LEGAL_INSTRUMENT_PATTERN.finditer(text):
            raw = match.group(1).strip()
            tokens = raw.split()
            p_len = (
                2
                if tokens[0].casefold()
                in {"bộ", "nghị", "thông", "quyết", "pháp"}
                else 1
            )
            if len(tokens) <= p_len:
                continue
            name_first_word = tokens[p_len]
            if not name_first_word[0].isupper():
                continue
            valid_tokens = tokens[:p_len]
            for tok in tokens[p_len:]:
                tok_clean = tok.strip(".,;:()[]\"'?")
                if (
                    tok_clean.casefold() in _LEGAL_BOUNDARY_STOP_WORDS
                    or tok_clean.isdigit()
                ):
                    break
                valid_tokens.append(tok_clean)
            if len(valid_tokens) > p_len:
                references.append(" ".join(valid_tokens).casefold())

        return list(dict.fromkeys(references))

    @staticmethod
    def _check_legal_references(
        claim_text: str,
        linked: list[Evidence],
    ) -> bool:
        refs = RuleBasedClaimGroundingVerifier._extract_legal_references(
            claim_text
        )
        if not refs:
            return True
        haystack_parts: list[str] = []
        for ev in linked:
            for field in (
                ev.document_title,
                ev.document_number,
                ev.document_type,
                ev.text,
            ):
                if field:
                    haystack_parts.append(field.casefold())
        haystack = " ".join(haystack_parts)
        for ref in refs:
            if ref not in haystack:
                return False
        return True

    @staticmethod
    def _evidence_text(evidence: Evidence) -> str:
        return " ".join(
            value
            for value in (
                evidence.document_title,
                evidence.document_number,
                evidence.article_number,
                evidence.article_title,
                evidence.text,
            )
            if value is not None
        )
