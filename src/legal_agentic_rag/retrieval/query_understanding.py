"""Deterministic Vietnamese legal-query analysis without external knowledge."""

from __future__ import annotations

from collections.abc import Iterable
import re

from legal_agentic_rag.configuration.online import QueryUnderstandingConfig
from legal_agentic_rag.schemas.retrieval import (
    QueryAnalysis,
    QueryIntent,
    QueryVariant,
    QueryVariantKind,
    RetrievalQuery,
)

_DOCUMENT_NUMBER_PATTERN = re.compile(
    r"(?<!\w)\d{1,4}/\d{4}/[0-9A-ZĐ-]+(?:/[0-9A-ZĐ-]+)*(?!\w)",
    re.IGNORECASE,
)
_ARTICLE_PATTERN = re.compile(r"\bđiều\s+(\d+[a-zđ]?)\b", re.IGNORECASE)
_CLAUSE_PATTERN = re.compile(r"\bkhoản\s+(\d+[a-zđ]?)\b", re.IGNORECASE)
_POINT_PATTERN = re.compile(r"\bđiểm\s+([a-zđ]|\d+)\b", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

_RELATIONSHIP_CUES = (
    "sửa đổi",
    "bổ sung",
    "thay thế",
    "bãi bỏ",
    "hướng dẫn",
    "dẫn chiếu",
    "còn hiệu lực",
    "hết hiệu lực",
)
_SCOPE_CUES = (
    "đối với",
    "áp dụng cho",
    "không áp dụng",
    "trong trường hợp",
    "trừ trường hợp",
    "ngoại trừ",
)
_FRAMING_PATTERNS = (
    re.compile(r"^xin\s+(?:cho\s+)?hỏi[\s,:;\-–—]+", re.IGNORECASE),
    re.compile(
        r"^cho\s+(?:tôi|mình|em)\s+hỏi[\s,:;\-–—]+",
        re.IGNORECASE,
    ),
    re.compile(r"^tôi\s+muốn\s+hỏi[\s,:;\-–—]+", re.IGNORECASE),
    re.compile(
        r"^theo\s+quy\s+định\s+(?:của\s+)?pháp\s+luật"
        r"(?:\s+hiện\s+hành)?[\s,:;\-–—]+",
        re.IGNORECASE,
    ),
)


class QueryUnderstandingService:
    """Extract explicit legal signals and plan bounded user-derived variants."""

    def __init__(
        self,
        config: QueryUnderstandingConfig | None = None,
    ) -> None:
        self._config = config or QueryUnderstandingConfig()

    def enrich(self, query: RetrievalQuery) -> RetrievalQuery:
        """Return a query with trusted analysis and deterministic variants."""
        if not self._config.enabled:
            return query.model_copy(
                update={"query_analysis": None, "query_variants": []}
            )
        analysis = self.analyze(query.normalized_question)
        variants = self.plan_variants(query.normalized_question, analysis)
        return query.model_copy(
            update={
                "query_analysis": analysis,
                "query_variants": variants,
            }
        )

    def analyze(self, question: str) -> QueryAnalysis:
        """Analyze signals that occur verbatim in one normalized question."""
        normalized = question.strip()
        folded = normalized.casefold()
        document_numbers = self._full_matches(
            _DOCUMENT_NUMBER_PATTERN,
            normalized,
        )
        article_numbers = self._group_matches(_ARTICLE_PATTERN, normalized)
        clause_numbers = self._group_matches(_CLAUSE_PATTERN, normalized)
        point_numbers = self._group_matches(_POINT_PATTERN, normalized)
        year_mentions = self._full_matches(_YEAR_PATTERN, normalized)
        relationship_cues = [
            cue for cue in _RELATIONSHIP_CUES if cue in folded
        ]
        scope_cues = [cue for cue in _SCOPE_CUES if cue in folded]
        return QueryAnalysis(
            intent=self._intent(
                folded,
                has_reference=bool(
                    document_numbers
                    or article_numbers
                    or clause_numbers
                    or point_numbers
                ),
                has_relationship=bool(relationship_cues),
            ),
            document_numbers=document_numbers,
            article_numbers=article_numbers,
            clause_numbers=clause_numbers,
            point_numbers=point_numbers,
            year_mentions=year_mentions,
            scope_cues=scope_cues,
            relationship_cues=relationship_cues,
        )

    def plan_variants(
        self,
        question: str,
        analysis: QueryAnalysis,
    ) -> list[QueryVariant]:
        """Create bounded variants using only text already present in the query."""
        normalized = question.strip()
        candidates: list[tuple[QueryVariantKind, str]] = [
            (QueryVariantKind.NORMALIZED, normalized)
        ]
        stripped = self._strip_framing(normalized)
        if stripped != normalized:
            candidates.append((QueryVariantKind.FRAMING_STRIPPED, stripped))
        reference_text = self._reference_variant(normalized, analysis)
        if reference_text is not None:
            candidates.append(
                (QueryVariantKind.LEGAL_REFERENCE, reference_text)
            )

        variants: list[QueryVariant] = []
        seen: set[str] = set()
        for kind, text in candidates:
            value = text.strip()
            if not value or value.casefold() in seen:
                continue
            seen.add(value.casefold())
            variants.append(
                QueryVariant(
                    variant_id=f"qv{len(variants) + 1}",
                    text=value,
                    kind=kind,
                )
            )
            if len(variants) >= self._config.max_variants:
                break
        return variants

    @staticmethod
    def _intent(
        folded: str,
        *,
        has_reference: bool,
        has_relationship: bool,
    ) -> QueryIntent:
        if has_relationship:
            return QueryIntent.RELATIONSHIP
        if any(cue in folded for cue in ("là gì", "được hiểu là", "định nghĩa")):
            return QueryIntent.DEFINITION
        if any(cue in folded for cue in ("thủ tục", "hồ sơ", "trình tự")):
            return QueryIntent.PROCEDURE
        if any(
            cue in folded
            for cue in ("bao nhiêu", "mức nào", "thời hạn", "bao lâu")
        ):
            return QueryIntent.QUANTITATIVE
        if any(
            cue in folded
            for cue in ("có được", "được phép", "đủ điều kiện")
        ):
            return QueryIntent.ELIGIBILITY
        if any(cue in folded for cue in ("bị cấm", "có bị cấm", "không được")):
            return QueryIntent.PROHIBITION
        if any(cue in folded for cue in ("có phải", "bắt buộc", "phải")):
            return QueryIntent.OBLIGATION
        if has_reference:
            return QueryIntent.REFERENCE_LOOKUP
        return QueryIntent.GENERAL

    @staticmethod
    def _strip_framing(question: str) -> str:
        value = question
        changed = True
        while changed:
            changed = False
            for pattern in _FRAMING_PATTERNS:
                stripped = pattern.sub("", value, count=1).strip()
                if stripped != value:
                    value = stripped
                    changed = True
                    break
        return value

    @staticmethod
    def _reference_variant(
        question: str,
        analysis: QueryAnalysis,
    ) -> str | None:
        if not analysis.has_explicit_legal_reference:
            return None
        matches = [
            *list(_DOCUMENT_NUMBER_PATTERN.finditer(question)),
            *list(_ARTICLE_PATTERN.finditer(question)),
            *list(_CLAUSE_PATTERN.finditer(question)),
            *list(_POINT_PATTERN.finditer(question)),
        ]
        matches.sort(key=lambda item: item.start())
        parts: list[str] = []
        seen: set[str] = set()
        for match in matches:
            value = match.group(0).strip()
            folded = value.casefold()
            if folded not in seen:
                parts.append(value)
                seen.add(folded)
        reference_text = " ".join(parts)
        if not reference_text or reference_text.casefold() == question.casefold():
            return None
        return reference_text

    @staticmethod
    def _full_matches(pattern: re.Pattern[str], value: str) -> list[str]:
        return QueryUnderstandingService._unique(
            match.group(0).strip() for match in pattern.finditer(value)
        )

    @staticmethod
    def _group_matches(pattern: re.Pattern[str], value: str) -> list[str]:
        return QueryUnderstandingService._unique(
            match.group(1).strip() for match in pattern.finditer(value)
        )

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for raw_value in values:
            folded = raw_value.casefold()
            if folded not in seen:
                result.append(raw_value)
                seen.add(folded)
        return result
