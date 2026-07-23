"""Dependency-free retrieval and generation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from math import log2
import unicodedata

from legal_agentic_rag.schemas import (
    AnswerResponse,
    EvaluationCase,
    EvaluationTargetGranularity,
    GenerationCaseMetrics,
    RetrievalCaseMetrics,
    RetrievalResponse,
)


class StandardRetrievalEvaluator:
    """Compute standard ranked metrics from explicit relevance grades."""

    def evaluate(
        self,
        case: EvaluationCase,
        response: RetrievalResponse,
        cutoffs: Sequence[int],
    ) -> RetrievalCaseMetrics:
        """Compute Recall, Precision, reciprocal rank, and NDCG."""
        ranked_ids = _ranked_ids(case.target_granularity, response)
        grades = case.relevance_grades
        relevant = set(grades)
        first_rank = next(
            (
                rank
                for rank, identity in enumerate(ranked_ids, 1)
                if identity in relevant
            ),
            None,
        )
        recall: dict[int, float] = {}
        precision: dict[int, float] = {}
        ndcg: dict[int, float] = {}
        for cutoff in cutoffs:
            selected = ranked_ids[:cutoff]
            found = sum(identity in relevant for identity in selected)
            recall[cutoff] = found / len(relevant)
            precision[cutoff] = found / cutoff
            actual_dcg = sum(
                (2 ** grades.get(identity, 0) - 1) / log2(rank + 1)
                for rank, identity in enumerate(selected, 1)
            )
            ideal_grades = sorted(grades.values(), reverse=True)[:cutoff]
            ideal_dcg = sum(
                (2**grade - 1) / log2(rank + 1)
                for rank, grade in enumerate(ideal_grades, 1)
            )
            ndcg[cutoff] = actual_dcg / ideal_dcg if ideal_dcg else 0.0
        return RetrievalCaseMetrics(
            recall_at_k=recall,
            precision_at_k=precision,
            ndcg_at_k=ndcg,
            reciprocal_rank=0.0 if first_rank is None else 1 / first_rank,
            first_relevant_rank=first_rank,
        )


class StandardGenerationEvaluator:
    """Compute only automatic metrics supported by available labels."""

    def evaluate(
        self,
        case: EvaluationCase,
        response: AnswerResponse,
    ) -> GenerationCaseMetrics:
        """Score exact answer, abstention, and citation identities."""
        exact_match = None
        if case.reference_answer is not None:
            exact_match = float(
                _normalize_answer(response.answer)
                == _normalize_answer(case.reference_answer)
            )
        abstention = None
        if case.should_abstain is not None:
            abstention = float(
                response.insufficient_evidence == case.should_abstain
            )
        citation_precision = None
        citation_recall = None
        if case.expected_citation_chunk_ids:
            expected = set(case.expected_citation_chunk_ids)
            actual = {citation.chunk_id for citation in response.citations}
            correct = len(expected & actual)
            citation_precision = correct / len(actual) if actual else 0.0
            citation_recall = correct / len(expected)
        return GenerationCaseMetrics(
            exact_match=exact_match,
            abstention_accuracy=abstention,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
        )


def ranked_target_ids(
    case: EvaluationCase,
    response: RetrievalResponse,
) -> list[str]:
    """Expose the deduplicated identities used by metric calculations."""
    return _ranked_ids(case.target_granularity, response)


def _ranked_ids(
    granularity: EvaluationTargetGranularity,
    response: RetrievalResponse,
) -> list[str]:
    values = [
        hit.chunk_id
        if granularity == EvaluationTargetGranularity.CHUNK
        else hit.document_id
        for hit in response.hits
    ]
    return list(dict.fromkeys(values))


def _normalize_answer(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()
