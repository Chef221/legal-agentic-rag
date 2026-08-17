"""Tests for answer-only SFT dataset encoding and prompt loss masking."""

from __future__ import annotations

import pytest

from legal_agentic_rag.fine_tuning.dataset import (
    DEFAULT_MAX_SEQ_LENGTH,
    SYSTEM_PROMPT,
    SFTAnswerOnlyDataset,
    encode_sft_example,
)
from legal_agentic_rag.schemas import CompetitionQuestion


class _MockTokenizer:
    """Deterministic mock tokenizer implementing chat template and encode interfaces."""

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
        # Deterministic dummy tokenization: 1 token per word + special token markers
        tokens = []
        for word in text.split():
            tokens.append(abs(hash(word)) % 10000 + 1)
        return tokens


def test_encode_sft_example_answer_only_masking() -> None:
    tokenizer = _MockTokenizer()
    question = "Thời hiệu xử lý vi phạm kỷ luật là bao lâu?"
    answer = "Thời hiệu xử lý kỷ luật là 02 năm đối với vi phạm ít nghiêm trọng."

    encoded = encode_sft_example(question, answer, tokenizer, max_seq_length=512)

    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    attention_mask = encoded["attention_mask"]

    assert len(input_ids) == len(labels) == len(attention_mask)
    assert all(mask == 1 for mask in attention_mask)

    # Prompt tokens must be masked with -100
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    assert labels[:prompt_len] == [-100] * prompt_len
    # Target tokens must match input_ids
    assert labels[prompt_len:] == input_ids[prompt_len:]
    assert all(lbl != -100 for lbl in labels[prompt_len:])


def test_encode_sft_example_truncation_ceiling() -> None:
    tokenizer = _MockTokenizer()
    question = "Câu hỏi ngắn?"
    long_answer = " ".join([f"từ_{i}" for i in range(100)])

    max_len = 20
    encoded = encode_sft_example(question, long_answer, tokenizer, max_seq_length=max_len)

    assert len(encoded["input_ids"]) == max_len
    assert len(encoded["labels"]) == max_len
    assert len(encoded["attention_mask"]) == max_len


def test_sft_dataset_container() -> None:
    tokenizer = _MockTokenizer()
    questions = [
        CompetitionQuestion(question_id="1", question="Q1?", reference_answer="A1."),
        CompetitionQuestion(question_id="2", question="Q2?", reference_answer="A2."),
    ]

    dataset = SFTAnswerOnlyDataset(questions, tokenizer, max_seq_length=128)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "input_ids" in item0
    assert "labels" in item0
    assert "attention_mask" in item0


def test_sft_dataset_rejects_missing_reference_answer() -> None:
    tokenizer = _MockTokenizer()
    questions = [
        CompetitionQuestion(question_id="1", question="Q1?", reference_answer=None),
    ]

    with pytest.raises(ValueError, match="reference answers"):
        SFTAnswerOnlyDataset(questions, tokenizer)
