"""Tests for official-only generator supervision splitting."""

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.generator_training import (
    fixed_dev_sample,
    normalize_supervision_question,
    question_id_digest,
    split_generator_supervision,
)
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas import CompetitionQuestion


def _question(question_id: str, text: str) -> CompetitionQuestion:
    return CompetitionQuestion(
        question_id=question_id,
        question=text,
        reference_answer=f"Câu trả lời thật {question_id}.",
    )


def test_split_keeps_normalized_duplicate_group_together() -> None:
    """Spacing, case and Unicode-equivalent questions cannot cross splits."""
    questions = [
        _question("1", "Điều kiện đăng ký?"),
        _question("2", "  ĐIỀU KIỆN   ĐĂNG KÝ? "),
        *[_question(str(index), f"Câu hỏi {index}?") for index in range(3, 40)],
    ]

    split = split_generator_supervision(questions)
    memberships = {
        item.question_id: name
        for name, values in (
            ("train", split.train),
            ("dev", split.dev),
            ("holdout", split.holdout),
        )
        for item in values
    }

    assert memberships["1"] == memberships["2"]
    assert split.duplicate_group_count == 1
    assert sum(split.record_counts.values()) == len(questions)
    assert normalize_supervision_question(questions[0].question) == (
        normalize_supervision_question(questions[1].question)
    )


def test_split_is_deterministic_and_preserves_real_answers() -> None:
    """Repeated preparation changes neither assignment nor official label text."""
    questions = [_question(str(index), f"Câu hỏi số {index}?") for index in range(80)]

    first = split_generator_supervision(questions)
    second = split_generator_supervision(questions)

    assert first == second
    assert [item.reference_answer for item in first.train] == [
        item.reference_answer for item in second.train
    ]


def test_fixed_dev_sample_has_stable_order_and_digest() -> None:
    """The control sample is bounded and receives one reproducible identity."""
    questions = [_question(str(index), f"Câu hỏi kiểm thử {index}?") for index in range(200)]
    split = split_generator_supervision(questions)
    sample_size = min(5, len(split.dev))

    sample = fixed_dev_sample(split, sample_size=sample_size)

    assert sample == fixed_dev_sample(split, sample_size=sample_size)
    assert len(question_id_digest(sample)) == 64


def test_split_rejects_missing_gold_and_duplicate_id() -> None:
    """Fine-tuning never invents a label or silently overwrites one record."""
    missing = CompetitionQuestion(question_id="1", question="Một câu hỏi?")
    with pytest.raises(DataValidationError, match="reference answers"):
        split_generator_supervision([missing])

    duplicate = [_question("1", "Một?"), _question("1", "Hai?")]
    with pytest.raises(DataValidationError, match="duplicate question IDs"):
        split_generator_supervision(duplicate)
