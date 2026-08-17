"""Tests for deterministic M50 three-way dataset splitter."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from legal_agentic_rag.fine_tuning.splitting import (
    M50_SPLIT_MANIFEST_FILENAME,
    SCREEN_HOLDOUT_FILENAME,
    SFT_TRAIN_FILENAME,
    SFT_VAL_FILENAME,
    M50FineTuningSplitter,
)


@pytest.fixture
def clean_training_file(tmp_path: Path) -> Path:
    # 10 questions with 2 duplicate groups: (q1, q2), (q3, q4)
    data = {
        "1": {"question": "Q_alpha 1?", "answer": "A1."},
        "2": {"question": "Q_alpha 1?", "answer": "A2."},
        "3": {"question": "Q_beta 2?", "answer": "A3."},
        "4": {"question": "Q_beta 2?", "answer": "A4."},
        "5": {"question": "Q_gamma 3?", "answer": "A5."},
        "6": {"question": "Q_delta 4?", "answer": "A6."},
        "7": {"question": "Q_epsilon 5?", "answer": "A7."},
        "8": {"question": "Q_zeta 6?", "answer": "A8."},
        "9": {"question": "Q_eta 7?", "answer": "A9."},
        "10": {"question": "Q_theta 8?", "answer": "A10."},
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_m50_splitter_reproducibility(tmp_path: Path, clean_training_file: Path) -> None:
    out1 = tmp_path / "split_1"
    out2 = tmp_path / "split_2"

    splitter = M50FineTuningSplitter()
    m1 = splitter.split(clean_training_file, out1, val_target=2, screen_target=2, seed=2026)
    m2 = splitter.split(clean_training_file, out2, val_target=2, screen_target=2, seed=2026)

    # Manifests and partition contents must match identically
    assert m1.clean_training_source.sha256 == m2.clean_training_source.sha256
    assert [p.sha256 for p in m1.partitions] == [p.sha256 for p in m2.partitions]
    assert [p.question_count for p in m1.partitions] == [p.question_count for p in m2.partitions]


def test_m50_splitter_zero_overlap_and_group_preservation(
    tmp_path: Path, clean_training_file: Path
) -> None:
    output_dir = tmp_path / "m50_split"
    splitter = M50FineTuningSplitter()
    manifest = splitter.split(
        clean_training_file, output_dir, val_target=2, screen_target=2, seed=2026
    )

    train_p = json.loads((output_dir / SFT_TRAIN_FILENAME).read_text(encoding="utf-8"))
    val_p = json.loads((output_dir / SFT_VAL_FILENAME).read_text(encoding="utf-8"))
    screen_p = json.loads((output_dir / SCREEN_HOLDOUT_FILENAME).read_text(encoding="utf-8"))

    train_ids = set(train_p.keys())
    val_ids = set(val_p.keys())
    screen_ids = set(screen_p.keys())

    # Total records match source
    assert len(train_ids) + len(val_ids) + len(screen_ids) == 10

    # Zero overlap
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & screen_ids) == 0
    assert len(val_ids & screen_ids) == 0

    # Duplicate group ('1', '2') must be in the same partition
    dup_group_1 = {"1", "2"}
    assert (
        dup_group_1.issubset(train_ids)
        or dup_group_1.issubset(val_ids)
        or dup_group_1.issubset(screen_ids)
    )

    # Duplicate group ('3', '4') must be in the same partition
    dup_group_2 = {"3", "4"}
    assert (
        dup_group_2.issubset(train_ids)
        or dup_group_2.issubset(val_ids)
        or dup_group_2.issubset(screen_ids)
    )


class _OverlengthMockTokenizer:
    """Mock tokenizer where certain words expand to a very large number of tokens."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 'EXPAND' word produces 2000 tokens despite having very few characters
        if "EXPAND" in text:
            return [1] * 2000
        return [1] * len(text.split())


def test_find_overlength_question_ids_is_tokenizer_derived() -> None:
    from legal_agentic_rag.fine_tuning.splitting import find_overlength_question_ids
    from legal_agentic_rag.schemas import CompetitionQuestion

    # q1 has short character count but expands to 2000 tokens via tokenizer
    # q2 has long character count (500 chars of single spaces/words) but produces only 50 tokens
    questions = [
        CompetitionQuestion(
            question_id="short_chars_huge_tokens",
            question="Short question?",
            reference_answer="EXPAND answer.",
        ),
        CompetitionQuestion(
            question_id="long_chars_few_tokens",
            question="Long question with many characters but simple words?",
            reference_answer=" ".join(["word"] * 50),
        ),
    ]

    tokenizer = _OverlengthMockTokenizer()
    overlength = find_overlength_question_ids(questions, tokenizer, max_seq_length=1536)

    # Must detect only the tokenizer-expanded question, proving token-based rather than char-based detection
    assert overlength == ["short_chars_huge_tokens"]
