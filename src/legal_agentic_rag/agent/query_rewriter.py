"""Conservative query rewriting that never invents legal terms."""

from __future__ import annotations

from legal_agentic_rag.schemas.retrieval import RetrievalQuery


class ConservativeQueryRewriter:
    """Reuse only user-derived query forms when a retrieval retry is needed."""

    def rewrite(
        self,
        query: RetrievalQuery,
        *,
        current_query: str,
        previously_used: set[str],
    ) -> str | None:
        """Return an unused user-supplied form, or null when none is available."""
        candidates = (
            *(variant.text for variant in query.query_variants),
            query.original_question.strip(),
            query.normalized_question.strip(),
        )
        for candidate in candidates:
            if candidate != current_query and candidate not in previously_used:
                return candidate
        return None
