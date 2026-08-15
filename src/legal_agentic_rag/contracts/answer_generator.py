"""Protocol for grounded Vietnamese legal answer generation."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.answering import AnswerResponse, Evidence
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy
from legal_agentic_rag.schemas.tools import AnswerGenerationCorrectionSignal


@runtime_checkable
class AnswerGenerator(Protocol):
    """Generate an answer using only the selected evidence."""

    def generate(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
        correction_signal: AnswerGenerationCorrectionSignal | None = None,
    ) -> AnswerResponse:
        """Return a grounded answer or explicit abstention."""
        ...
