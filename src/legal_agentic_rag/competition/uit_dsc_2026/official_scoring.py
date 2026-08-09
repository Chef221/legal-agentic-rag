"""Local metric implementation aligned with the audited BTC scorer source."""

from __future__ import annotations

from collections.abc import Callable
import importlib.metadata
import re

from legal_agentic_rag.exceptions import BackendInitializationError
from legal_agentic_rag.schemas.evaluation import GenerationCaseMetrics

OFFICIAL_SCORER_SHA256 = (
    "4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891"
)
OFFICIAL_NLTK_VERSION = "3.7"


def score_official_compatible_answer(
    prediction: str,
    reference: str,
    *,
    meteor_scorer: Callable[[list[list[str]], list[str]], float] | None = None,
) -> GenerationCaseMetrics:
    """Score one pair with the algorithms observed in the audited BTC scorer.

    The optional scorer injection exists for deterministic tests. Production use
    loads NLTK 3.7 lazily and fails closed when its WordNet resources are absent.
    """
    scorer = meteor_scorer or _load_official_meteor_scorer()
    prediction_tokens = prediction.split()
    reference_tokens = reference.split()
    meteor = float(scorer([reference_tokens], prediction_tokens))
    return GenerationCaseMetrics(
        exact_match=float(prediction == reference),
        meteor=meteor,
        rouge_l=_official_rouge_l(prediction, reference),
    )


def _load_official_meteor_scorer() -> Callable[[list[list[str]], list[str]], float]:
    try:
        version = importlib.metadata.version("nltk")
    except importlib.metadata.PackageNotFoundError as error:
        raise BackendInitializationError(
            "Official-compatible METEOR requires optional dependency nltk==3.7"
        ) from error
    if version != OFFICIAL_NLTK_VERSION:
        raise BackendInitializationError(
            "Official-compatible METEOR requires exact nltk version 3.7"
        )
    try:
        from nltk.corpus import wordnet
        from nltk.translate.meteor_score import meteor_score

        wordnet.ensure_loaded()
    except LookupError as error:
        raise BackendInitializationError(
            "Official-compatible METEOR requires local NLTK wordnet resources"
        ) from error
    return meteor_score


def _official_rouge_l(prediction: str, reference: str) -> float:
    """Reproduce the vendored scorer's ASCII tokenizer and ROUGE-L F1."""
    prediction_tokens = _ascii_tokens(prediction)
    reference_tokens = _ascii_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    previous = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        current = [0]
        for reference_index, reference_token in enumerate(reference_tokens, 1):
            if prediction_token == reference_token:
                current.append(previous[reference_index - 1] + 1)
            else:
                current.append(max(previous[reference_index], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _ascii_tokens(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).split()
