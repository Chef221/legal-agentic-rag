"""Tests for the audited organizer-compatible answer metrics."""

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.official_scoring import (
    score_official_compatible_answer,
)


def test_official_compatible_scoring_uses_whitespace_meteor_and_ascii_rouge() -> None:
    calls: list[tuple[list[list[str]], list[str]]] = []

    def meteor(references: list[list[str]], prediction: list[str]) -> float:
        calls.append((references, prediction))
        return 0.25

    metrics = score_official_compatible_answer(
        "Theo luật doanh nghiệp",
        "Theo luật lao động",
        meteor_scorer=meteor,
    )

    assert calls == [
        (["Theo luật lao động".split()], "Theo luật doanh nghiệp".split())
    ]
    assert metrics.meteor == 0.25
    assert metrics.rouge_l == pytest.approx(6 / 11)
    assert metrics.exact_match == 0


def test_official_rouge_matches_vendored_ascii_tokenizer_behavior() -> None:
    metrics = score_official_compatible_answer(
        "người lao động",
        "người sử dụng lao động",
        meteor_scorer=lambda _references, _prediction: 0.0,
    )

    # The organizer's vendored tokenizer strips Vietnamese letters rather than
    # using the Unicode tokenizer of the project's diagnostic metric.
    assert metrics.rouge_l == pytest.approx(8 / 11)
