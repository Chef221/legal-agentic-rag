"""Corpus-aware bounded query planning for the SQLite FTS5 backend."""

from __future__ import annotations

from dataclasses import dataclass

from legal_agentic_rag.configuration.online import BM25RuntimeConfig

_SEMANTIC_MODIFIERS = frozenset(
    {
        "chưa",
        "chỉ",
        "không",
        "ngoại",
        "phải",
        "trừ",
    }
)


@dataclass(frozen=True, slots=True)
class BM25QueryPlan:
    """Selected lexical terms and non-content diagnostics for one query."""

    terms: tuple[str, ...]
    original_unique_term_count: int
    known_term_count: int
    was_limited: bool


class BM25QueryPlanner:
    """Prefer corpus-discriminative terms while preserving legal modifiers."""

    def __init__(self, config: BM25RuntimeConfig | None = None) -> None:
        self._config = config or BM25RuntimeConfig()

    def plan(
        self,
        terms: list[str],
        *,
        document_frequencies: dict[str, int],
        document_count: int,
    ) -> BM25QueryPlan:
        """Return a bounded deterministic subset without static stopword removal."""
        unique_terms = list(dict.fromkeys(terms))
        if not unique_terms:
            return BM25QueryPlan((), 0, 0, False)

        known_terms = [
            term for term in unique_terms if document_frequencies.get(term, 0) > 0
        ]
        if not known_terms:
            selected = unique_terms[: self._config.max_query_terms]
            return BM25QueryPlan(
                tuple(selected),
                len(unique_terms),
                0,
                len(selected) < len(unique_terms),
            )

        maximum_frequency = (
            document_count * self._config.max_document_frequency_ratio
        )
        priority_terms = [
            term
            for term in known_terms
            if self._is_numeric(term) or term in _SEMANTIC_MODIFIERS
        ]
        discriminative_terms = [
            term
            for term in known_terms
            if term not in priority_terms
            and document_frequencies[term] <= maximum_frequency
        ]
        if not discriminative_terms:
            discriminative_terms = [
                term for term in known_terms if term not in priority_terms
            ]
        discriminative_terms.sort(
            key=lambda term: (
                document_frequencies[term],
                unique_terms.index(term),
            )
        )

        ranked_terms = [*priority_terms, *discriminative_terms]
        if not ranked_terms:
            ranked_terms = known_terms
        selected_set = set(ranked_terms[: self._config.max_query_terms])
        selected = [term for term in unique_terms if term in selected_set]
        return BM25QueryPlan(
            tuple(selected),
            len(unique_terms),
            len(known_terms),
            len(selected) < len(unique_terms),
        )

    @staticmethod
    def _is_numeric(term: str) -> bool:
        return any(character.isdigit() for character in term)
