"""Dependency-free retrieval and generation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from math import log2
import re
from typing import Callable
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

    def __init__(
        self,
        *,
        answer_renderer: Callable[[AnswerResponse], str] | None = None,
    ) -> None:
        self._answer_renderer = answer_renderer or (lambda response: response.answer)

    def evaluate(
        self,
        case: EvaluationCase,
        response: AnswerResponse,
    ) -> GenerationCaseMetrics:
        """Score exact answer, abstention, and citation identities."""
        evaluated_answer = self._answer_renderer(response)
        exact_match = None
        meteor = None
        rouge_l = None
        if case.reference_answer is not None:
            text_metrics = score_text_answer(
                evaluated_answer,
                case.reference_answer,
            )
            exact_match = text_metrics.exact_match
            meteor = text_metrics.meteor
            rouge_l = text_metrics.rouge_l
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
            meteor=meteor,
            rouge_l=rouge_l,
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


def score_text_answer(
    prediction: str,
    reference: str,
) -> GenerationCaseMetrics:
    """Score one answer pair with deterministic local competition diagnostics."""
    prediction_tokens = _metric_tokens(prediction)
    reference_tokens = _metric_tokens(reference)
    return GenerationCaseMetrics(
        exact_match=float(
            _normalize_answer(prediction) == _normalize_answer(reference)
        ),
        meteor=_meteor_exact_token(prediction_tokens, reference_tokens),
        rouge_l=_rouge_l_f1(prediction_tokens, reference_tokens),
    )


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


def _metric_tokens(value: str) -> list[str]:
    normalized = _normalize_answer(value)
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _meteor_exact_token(
    prediction: Sequence[str],
    reference: Sequence[str],
) -> float:
    """Compute deterministic exact-token METEOR without stemming or synonyms."""
    if not prediction or not reference:
        return 0.0
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference):
        positions.setdefault(token, []).append(index)
    consumed: dict[str, int] = {}
    matches: list[tuple[int, int]] = []
    for prediction_index, token in enumerate(prediction):
        offset = consumed.get(token, 0)
        candidates = positions.get(token, [])
        if offset >= len(candidates):
            continue
        matches.append((prediction_index, candidates[offset]))
        consumed[token] = offset + 1
    match_count = len(matches)
    if match_count == 0:
        return 0.0
    precision = match_count / len(prediction)
    recall = match_count / len(reference)
    weighted_harmonic = (precision * recall) / (
        0.9 * precision + 0.1 * recall
    )
    chunks = 1 + sum(
        current_prediction != previous_prediction + 1
        or current_reference != previous_reference + 1
        for (previous_prediction, previous_reference),
        (current_prediction, current_reference) in zip(matches, matches[1:])
    )
    penalty = 0.5 * (chunks / match_count) ** 3
    return weighted_harmonic * (1 - penalty)


def _rouge_l_f1(
    prediction: Sequence[str],
    reference: Sequence[str],
) -> float:
    """Compute token-level ROUGE-L F1 from the longest common subsequence."""
    if not prediction or not reference:
        return 0.0
    previous = [0] * (len(reference) + 1)
    for prediction_token in prediction:
        current = [0]
        for reference_index, reference_token in enumerate(reference, 1):
            if prediction_token == reference_token:
                current.append(previous[reference_index - 1] + 1)
            else:
                current.append(max(previous[reference_index], current[-1]))
        previous = current
    lcs_length = previous[-1]
    if lcs_length == 0:
        return 0.0
    precision = lcs_length / len(prediction)
    recall = lcs_length / len(reference)
    return 2 * precision * recall / (precision + recall)
