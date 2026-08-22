"""Deterministic evidence applicability scoring and ordering."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from legal_agentic_rag.configuration.online import (
    EvidenceSelectionConfig,
    GenerationConfig,
)
from legal_agentic_rag.schemas.answering import EvidenceApplicability
from legal_agentic_rag.schemas.retrieval import RetrievalHit, RetrievalQuery

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

_RECOGNIZED_SLUG_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        [
            "thong-tu-lien-tich",
            "nghi-quyet-lien-tich",
            "nghi-dinh-sua-doi-nghi-dinh",
            "thong-tu-sua-doi-thong-tu",
            "van-ban-hop-nhat",
            "bo-luat",
            "luat",
            "nghi-dinh",
            "thong-tu",
            "nghi-quyet",
            "quyet-dinh",
            "phap-lenh",
            "cong-van",
            "chi-thi",
            "thong-bao",
            "huong-dan",
            "ke-hoach",
            "cong-dien",
            "quy-dinh",
            "quy-che",
            "dieu-le",
            "lenh",
        ],
        key=lambda item: -len(item),
    )
)


def _remove_diacritics(text: str) -> str:
    """Strip Vietnamese diacritics while mapping đ/Đ to d/D."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _tokenize_legal_id(text: str) -> list[str]:
    """Tokenize legal document identifier or title into alphanumeric segments."""
    clean = _remove_diacritics(text).casefold()
    return [t for t in re.split(r"[^a-z0-9]+", clean) if t]


@dataclass(frozen=True, slots=True)
class ScoredEvidenceCandidate:
    """One immutable retrieval hit with transparent selection signals."""

    hit: RetrievalHit
    applicability: EvidenceApplicability
    document_reference_match: bool | None
    article_reference_match: bool | None
    lexical_overlap_score: float
    selection_score: float


class EvidenceSelector:
    """Order hits using query evidence without claiming legal interpretation."""

    def __init__(
        self,
        config: EvidenceSelectionConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self._config = config or EvidenceSelectionConfig()
        generation = generation_config or GenerationConfig()
        self._inactive_statuses = generation.inactive_effect_statuses

    def score(
        self,
        query: RetrievalQuery,
        hits: list[RetrievalHit],
    ) -> list[ScoredEvidenceCandidate]:
        """Return deterministic candidates ordered by applicability signals."""
        query_terms = self._terms(query.normalized_question)
        analysis = query.query_analysis
        raw_document_numbers = (
            analysis.document_numbers if analysis is not None else []
        )
        document_references = {
            self._normalize_reference(value)
            for value in raw_document_numbers
        }
        article_references = {
            self._normalize_reference(value)
            for value in (
                analysis.article_numbers if analysis is not None else []
            )
        }
        scored = [
            self._score_hit(
                hit,
                query_terms=query_terms,
                document_references=document_references,
                raw_document_references=raw_document_numbers,
                article_references=article_references,
            )
            for hit in hits
        ]
        if not self._config.enabled:
            return sorted(
                scored,
                key=lambda item: (
                    item.applicability == EvidenceApplicability.INACTIVE,
                    item.hit.rank,
                    item.hit.chunk_id,
                ),
            )
        return sorted(
            scored,
            key=lambda item: (
                -item.selection_score,
                item.applicability == EvidenceApplicability.INACTIVE,
                item.hit.rank,
                item.hit.chunk_id,
            ),
        )

    def _score_hit(
        self,
        hit: RetrievalHit,
        *,
        query_terms: set[str],
        document_references: set[str],
        raw_document_references: list[str],
        article_references: set[str],
    ) -> ScoredEvidenceCandidate:
        raw_doc_num = hit.metadata.get("document_number")
        if raw_doc_num is not None and not isinstance(raw_doc_num, str):
            # Rule C: Malformed present non-string metadata -> fail closed (False if references exist, else None)
            document_match = False if document_references else None
        else:
            document_number = self._text(raw_doc_num)
            document_title = self._text(hit.metadata.get("document_title"))
            document_match = self._reference_match(
                document_number,
                document_references,
                raw_document_references=raw_document_references,
                document_title=document_title,
            )

        structure = hit.metadata.get("structure")
        hierarchy = structure if isinstance(structure, dict) else {}
        article_number = self._text(hierarchy.get("article_number"))
        article_match = self._reference_match(
            article_number,
            article_references,
        )
        inactive = self._is_inactive(hit)
        applicability = self._applicability(
            hit,
            inactive=inactive,
            document_match=document_match,
            article_match=article_match,
        )
        overlap = self._lexical_overlap(query_terms, hit, hierarchy)
        reference_match_count = sum(
            value is True for value in (document_match, article_match)
        )
        score = 1.0 / hit.rank
        if self._config.enabled:
            score += (
                reference_match_count * self._config.reference_match_boost
            )
            score += overlap * self._config.lexical_overlap_weight
            if inactive:
                score -= self._config.inactive_penalty
        return ScoredEvidenceCandidate(
            hit=hit,
            applicability=applicability,
            document_reference_match=document_match,
            article_reference_match=article_match,
            lexical_overlap_score=overlap,
            selection_score=score,
        )

    def _applicability(
        self,
        hit: RetrievalHit,
        *,
        inactive: bool,
        document_match: bool | None,
        article_match: bool | None,
    ) -> EvidenceApplicability:
        if inactive:
            return EvidenceApplicability.INACTIVE
        reference_matches = [
            value
            for value in (document_match, article_match)
            if value is not None
        ]
        if reference_matches and all(reference_matches):
            return EvidenceApplicability.EXPLICIT_MATCH
        if reference_matches and not all(reference_matches):
            return EvidenceApplicability.REFERENCE_MISMATCH
        if self._text(hit.metadata.get("effect_status")) is None:
            return EvidenceApplicability.UNKNOWN
        return EvidenceApplicability.COMPATIBLE

    def _is_inactive(self, hit: RetrievalHit) -> bool:
        status = self._text(hit.metadata.get("effect_status"))
        return (
            status is not None
            and status.casefold() in self._inactive_statuses
        )

    @classmethod
    def _reference_match(
        cls,
        value: str | None,
        references: set[str],
        *,
        raw_document_references: list[str] | None = None,
        document_title: str | None = None,
    ) -> bool | None:
        if not references:
            return None
        # Rule A: If metadata document_number is present as a non-empty string,
        # compare strictly using existing canonical normalization.
        if value is not None:
            return cls._normalize_reference(value) in references

        # Rule B: Only if metadata document_number is absent (None),
        # attempt strict own-document-number recovery from document_title.
        if document_title is not None:
            raw_refs = raw_document_references or list(references)
            return any(
                cls._match_own_document_title(document_title, ref)
                for ref in raw_refs
            )

        return False

    @classmethod
    def _match_own_document_title(
        cls,
        title: str,
        query_doc_ref: str,
    ) -> bool:
        """Strictly match query document reference against the leading own identity of title.

        Requires the title to start with a recognized legal document slug prefix, followed
        immediately by the own document number (with an optional 'so' token).
        """
        if not title.strip() or not query_doc_ref.strip():
            return False
        ref_tokens = _tokenize_legal_id(query_doc_ref)
        if not ref_tokens:
            return False
        title_tokens = _tokenize_legal_id(title)
        if not title_tokens:
            return False

        # Match longest recognized slug prefix at the beginning of the title
        matched_prefix_token_count: int | None = None
        for prefix in _RECOGNIZED_SLUG_PREFIXES:
            p_tokens = prefix.split("-")
            if len(title_tokens) >= len(p_tokens) and title_tokens[: len(p_tokens)] == p_tokens:
                matched_prefix_token_count = len(p_tokens)
                break

        if matched_prefix_token_count is None:
            return False

        remainder_tokens = title_tokens[matched_prefix_token_count:]

        # Allow optional 'so' immediately after the prefix
        if remainder_tokens and remainder_tokens[0] == "so":
            remainder_tokens = remainder_tokens[1:]

        # Remainder must begin IMMEDIATELY with the query reference tokens
        if len(remainder_tokens) < len(ref_tokens):
            return False

        slice_tokens = remainder_tokens[: len(ref_tokens)]

        # First token is the number: compare integer value if both are numeric (e.g. '01' == '1')
        ref_num = ref_tokens[0]
        title_num = slice_tokens[0]
        if ref_num.isdigit() and title_num.isdigit():
            if int(ref_num) != int(title_num):
                return False
        elif ref_num != title_num:
            return False

        # Compare all remaining tokens (year, organ/type codes) exactly
        for r_tok, t_tok in zip(ref_tokens[1:], slice_tokens[1:]):
            if r_tok != t_tok:
                return False

        return True

    @staticmethod
    def _lexical_overlap(
        query_terms: set[str],
        hit: RetrievalHit,
        hierarchy: dict[str, object],
    ) -> float:
        if not query_terms:
            return 0.0
        values = [
            hit.text,
            EvidenceSelector._text(hit.metadata.get("document_title")) or "",
            EvidenceSelector._text(hit.metadata.get("document_number")) or "",
            EvidenceSelector._text(hierarchy.get("article_title")) or "",
            EvidenceSelector._text(hierarchy.get("article_number")) or "",
        ]
        evidence_terms = EvidenceSelector._terms(" ".join(values))
        return len(query_terms & evidence_terms) / len(query_terms)

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {
            token.casefold()
            for token in _TOKEN_PATTERN.findall(value)
            if len(token) > 1 or token.isdigit()
        }

    @staticmethod
    def _normalize_reference(value: str) -> str:
        return "".join(value.split()).casefold()

    @staticmethod
    def _text(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None
