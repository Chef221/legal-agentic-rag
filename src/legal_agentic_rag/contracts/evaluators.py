"""Protocols for replaceable retrieval and generation evaluators."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.answering import AnswerResponse
from legal_agentic_rag.schemas.evaluation import (
    EvaluationCase,
    GenerationCaseMetrics,
    RetrievalCaseMetrics,
)
from legal_agentic_rag.schemas.retrieval import RetrievalResponse


@runtime_checkable
class RetrievalEvaluator(Protocol):
    """Score one ranked response against explicit stable-ID labels."""

    def evaluate(
        self,
        case: EvaluationCase,
        response: RetrievalResponse,
        cutoffs: Sequence[int],
    ) -> RetrievalCaseMetrics: ...


@runtime_checkable
class GenerationEvaluator(Protocol):
    """Score one answer only for labels present in the case."""

    def evaluate(
        self,
        case: EvaluationCase,
        response: AnswerResponse,
    ) -> GenerationCaseMetrics: ...
