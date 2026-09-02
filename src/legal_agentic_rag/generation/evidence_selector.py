"""Deterministic evidence applicability scoring and ordering."""

from __future__ import annotations

from dataclasses import dataclass
import re

from legal_agentic_rag.configuration.online import (
    EvidenceSelectionConfig,
    GenerationConfig,
)
from legal_agentic_rag.schemas.answering import EvidenceApplicability
from legal_agentic_rag.schemas.retrieval import RetrievalHit, RetrievalQuery

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


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
        document_references = {
            self._normalize_reference(value)
            for value in (
                analysis.document_numbers if analysis is not None else []
            )
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
        article_references: set[str],
    ) -> ScoredEvidenceCandidate:
        document_number = self._extract_document_number(hit)
        article_number = self._extract_article_number(hit)

        document_match = self._reference_match(
            document_number,
            document_references,
        )

        if (
            document_references
            and document_number is None
            and document_match is False
        ):
            document_match = (
                self._fallback_document_reference_match(
                    hit,
                    document_references,
                )
            )
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
        overlap = self._lexical_overlap(query_terms, hit)
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

    @staticmethod
    def _extract_document_number(hit: RetrievalHit) -> str | None:
        legacy_doc_num = EvidenceSelector._text(hit.metadata.get("document_number"))
        if legacy_doc_num is not None:
            return legacy_doc_num
        doc_identity = hit.metadata.get("document_identity")
        if isinstance(doc_identity, dict):
            return EvidenceSelector._text(doc_identity.get("document_number"))
        return None

    @staticmethod
    def _extract_document_title(hit: RetrievalHit) -> str | None:
        legacy_title = EvidenceSelector._text(hit.metadata.get("document_title"))
        if legacy_title is not None:
            return legacy_title
        doc_identity = hit.metadata.get("document_identity")
        if isinstance(doc_identity, dict):
            return EvidenceSelector._text(doc_identity.get("title"))
        return None

    @staticmethod
    def _extract_article_number(hit: RetrievalHit) -> str | None:
        structure = hit.metadata.get("structure")
        if isinstance(structure, dict):
            legacy_art = EvidenceSelector._text(structure.get("article_number"))
            if legacy_art is not None:
                return legacy_art
        hierarchy = hit.metadata.get("hierarchy")
        if isinstance(hierarchy, dict):
            v2_art = EvidenceSelector._text(hierarchy.get("article_label"))
            if v2_art is not None:
                return v2_art
        return None

    @staticmethod
    def _reference_match(
        value: str | None,
        references: set[str],
    ) -> bool | None:
        if not references:
            return None
        if value is None:
            return False
        return EvidenceSelector._normalize_reference(value) in references

    @staticmethod
    def _lexical_overlap(
        query_terms: set[str],
        hit: RetrievalHit,
    ) -> float:
        if not query_terms:
            return 0.0
        values = [
            hit.text,
            EvidenceSelector._extract_document_title(hit) or "",
            EvidenceSelector._extract_document_number(hit) or "",
        ]
        structure = hit.metadata.get("structure")
        if isinstance(structure, dict):
            values.extend([
                EvidenceSelector._text(structure.get("article_title")) or "",
                EvidenceSelector._text(structure.get("article_number")) or "",
            ])
        hierarchy = hit.metadata.get("hierarchy")
        if isinstance(hierarchy, dict):
            values.extend([
                EvidenceSelector._text(hierarchy.get("article_label")) or "",
                EvidenceSelector._text(hierarchy.get("clause_label")) or "",
                EvidenceSelector._text(hierarchy.get("point_label")) or "",
            ])
            heading_path = hierarchy.get("heading_path")
            if isinstance(heading_path, list):
                for item in heading_path:
                    if isinstance(item, dict):
                        values.extend([
                            EvidenceSelector._text(item.get("label")) or "",
                            EvidenceSelector._text(item.get("title")) or "",
                        ])

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
        """Normalize reference identity without broad text accent folding."""
        import re

        normalized = "".join(value.split()).casefold()

        if re.fullmatch(
            r"\d{1,4}/\d{4}/[0-9a-zđ-]+"
            r"(?:/[0-9a-zđ-]+)*",
            normalized,
        ):
            normalized = normalized.replace("đ", "d")

        return normalized

    @staticmethod
    def _document_reference_identity_prefix(
        reference: str,
    ) -> str | None:
        """Return a conservative TVPL identity prefix for a query reference."""
        import re

        normalized = EvidenceSelector._normalize_reference(
            reference
        )

        match = re.fullmatch(
            r"(?P<number>\d{1,4})/"
            r"(?P<year>\d{4})/"
            r"(?P<suffix>[0-9a-z-]+"
            r"(?:/[0-9a-z-]+)*)",
            normalized,
        )

        if match is None:
            return None

        number = match.group("number")
        year = match.group("year")
        suffix = match.group("suffix").replace("/", "-")

        first_code = suffix.split("-", 1)[0]

        type_prefix_by_code = {
            "nd": "nghi-dinh",
            "tt": "thong-tu",
            "ttlt": "thong-tu",
            "qd": "quyet-dinh",
            "nq": "nghi-quyet",
            "ct": "chi-thi",
            "vbhn": "van-ban-hop-nhat",
        }

        document_type = type_prefix_by_code.get(
            first_code
        )

        if document_type is None:
            return None

        return (
            f"{document_type}-"
            f"{number}-"
            f"{year}-"
            f"{suffix}"
        )

    @staticmethod
    def _canonical_identity_slug(
        value: object,
        *,
        source_url: bool = False,
    ) -> str | None:
        """Canonicalize a title or URL basename for anchored identity matching."""
        import re
        import unicodedata
        from pathlib import PurePosixPath
        from urllib.parse import unquote, urlparse

        if not isinstance(value, str):
            return None

        text = value.strip()

        if not text:
            return None

        if source_url:
            try:
                parsed = urlparse(text)
            except Exception:
                return None

            text = PurePosixPath(
                unquote(parsed.path)
            ).name

            if text.casefold().endswith(".aspx"):
                text = text[:-5]

        text = unicodedata.normalize(
            "NFKD",
            text,
        )

        text = "".join(
            character
            for character in text
            if not unicodedata.combining(
                character
            )
        )

        text = (
            text
            .casefold()
            .replace("đ", "d")
        )

        text = re.sub(
            r"[^a-z0-9]+",
            "-",
            text,
        ).strip("-")

        return text or None

    @staticmethod
    def _fallback_document_reference_match(
        hit: RetrievalHit,
        references: set[str],
    ) -> bool:
        """
        Match explicit document identity without fabricating document_number.

        Requirements:
        - query has a supported explicit document reference;
        - document_title begins with that document identity;
        - source_url basename begins with the same identity.

        A document that merely mentions another legal document later in its
        title/URL therefore does not match.
        """
        if not references:
            return False

        title = EvidenceSelector._canonical_identity_slug(
            EvidenceSelector._extract_document_title(hit)
        )

        source_url = EvidenceSelector._canonical_identity_slug(
            hit.metadata.get("source_url"),
            source_url=True,
        )

        if title is None or source_url is None:
            return False

        for reference in references:

            prefix = (
                EvidenceSelector
                ._document_reference_identity_prefix(
                    reference
                )
            )

            if prefix is None:
                continue

            title_match = (
                title == prefix
                or title.startswith(
                    prefix + "-"
                )
            )

            url_match = (
                source_url == prefix
                or source_url.startswith(
                    prefix + "-"
                )
            )

            if title_match and url_match:
                return True

        return False

    @staticmethod
    def _text(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None
