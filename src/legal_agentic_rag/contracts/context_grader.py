"""Protocol for grading selected legal evidence context."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.answering import ContextGrade, Evidence
from legal_agentic_rag.schemas.retrieval import RetrievalQuery


@runtime_checkable
class ContextGrader(Protocol):
    """Assess whether selected evidence is sufficient for generation."""

    def grade(
        self, query: RetrievalQuery, evidence: Sequence[Evidence]
    ) -> ContextGrade:
        """Return relevance, coverage, consistency, and sufficiency scores."""
        ...
