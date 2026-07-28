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
        document_number = self._text(hit.metadata.get("document_number"))
        structure = hit.metadata.get("structure")
        hierarchy = structure if isinstance(structure, dict) else {}
        article_number = self._text(hierarchy.get("article_number"))
        document_match = self._reference_match(
            document_number,
            document_references,
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
