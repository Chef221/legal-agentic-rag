"""Tests for answer-only SFT dataset encoding, prompt loss masking, and EOS preservation."""

from __future__ import annotations

import pytest

from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.fine_tuning.dataset import (
    DEFAULT_MAX_SEQ_LENGTH,
    SYSTEM_PROMPT,
    SFTAnswerOnlyDataset,
    encode_sft_example,
)
from legal_agentic_rag.schemas import CompetitionQuestion


class _MockTokenizer:
    """Deterministic mock tokenizer implementing chat template and encode interfaces."""

    def __init__(self, eos_token_id: int = 9999) -> None:
        self.eos_token_id = eos_token_id
        self.eos_token = "<|im_end|>"

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
        tokens = []
        for word in text.split():
            if word == "<|im_end|>":
                tokens.append(self.eos_token_id)
            else:
                tokens.append(abs(hash(word)) % 9000 + 1)
        # If text ends with <|im_end|>\n, ensure terminal token is eos_token_id
        if text.rstrip().endswith("<|im_end|>") and (not tokens or tokens[-1] != self.eos_token_id):
            tokens.append(self.eos_token_id)
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
    assert encoded["was_truncated"] is False

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
    # Final token must be EOS
    assert labels[-1] == tokenizer.eos_token_id
    assert input_ids[-1] == tokenizer.eos_token_id


def test_encode_sft_example_eos_preserving_truncation() -> None:
    tokenizer = _MockTokenizer()
    question = "Câu hỏi ngắn?"
    long_answer = " ".join([f"từ_{i}" for i in range(100)])

    max_len = 25
    encoded = encode_sft_example(question, long_answer, tokenizer, max_seq_length=max_len)

    input_ids = encoded["input_ids"]
    labels = encoded["labels"]

    assert len(input_ids) == max_len
    assert len(labels) == max_len
    assert encoded["was_truncated"] is True
    assert encoded["original_target_token_count"] > encoded["retained_target_token_count"]

    # Critical invariant: final token and final label must remain EOS
    assert input_ids[-1] == tokenizer.eos_token_id
    assert labels[-1] == tokenizer.eos_token_id


def test_encode_sft_example_prompt_consuming_budget_fails_closed() -> None:
    tokenizer = _MockTokenizer()
    # Prompt alone is longer than max_seq_length - 1
    very_long_question = " ".join([f"cau_hoi_dai_{i}" for i in range(50)])
    short_answer = "Tra loi ngan."

    with pytest.raises(DataValidationError, match="leaves no safe assistant token capacity"):
        encode_sft_example(very_long_question, short_answer, tokenizer, max_seq_length=15)


def test_encode_sft_example_malformed_target_fails_closed() -> None:
    class _MalformedTokenizer:
        def __init__(self) -> None:
            self.eos_token_id = 9999
            self.eos_token = "<|im_end|>"

        def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = False) -> str:
            # Buggy template that omits im_end
            text = ""
            for m in messages:
                text += f"{m['role']}: {m['content']}\n"
            return text

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2, 3, 4]  # Does not end with 9999

    malformed_tok = _MalformedTokenizer()
    with pytest.raises(DataValidationError, match="Assistant target does not terminate"):
        encode_sft_example("Q?", "A.", malformed_tok, max_seq_length=128)


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
    assert item0["labels"][-1] == tokenizer.eos_token_id


def test_sft_dataset_rejects_missing_reference_answer() -> None:
    tokenizer = _MockTokenizer()
    questions = [
        CompetitionQuestion(question_id="1", question="Q1?", reference_answer=None),
    ]

    with pytest.raises(DataValidationError, match="reference answers"):
        SFTAnswerOnlyDataset(questions, tokenizer)


def test_encode_sft_example_exact_budget_boundary() -> None:
    tokenizer = _MockTokenizer()
    question = "Cau hoi?"
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    p_len = len(prompt_ids)

    # 4 content words + 1 terminal EOS added by chat template = 5 target tokens
    max_len = p_len + 5
    answer = "mot hai ba bon"
    encoded = encode_sft_example(question, answer, tokenizer, max_seq_length=max_len)

    assert encoded["was_truncated"] is False
    assert len(encoded["input_ids"]) == max_len
    assert encoded["retained_target_token_count"] == 5
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id


def test_encode_sft_example_budget_plus_one_boundary() -> None:
    tokenizer = _MockTokenizer()
    question = "Cau hoi?"
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    p_len = len(prompt_ids)

    # 5 content words + 1 terminal EOS added by chat template = 6 target tokens
    # When max_seq_length is p_len + 5, it must truncate by 1 token and preserve terminal EOS
    max_len = p_len + 5
    answer = "mot hai ba bon nam"
    encoded = encode_sft_example(question, answer, tokenizer, max_seq_length=max_len)

    assert encoded["was_truncated"] is True
    assert len(encoded["input_ids"]) == max_len
    assert encoded["retained_target_token_count"] == 5
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id
    assert encoded["labels"][-1] == tokenizer.eos_token_id


def test_encode_sft_example_determinism() -> None:
    tokenizer = _MockTokenizer()
    question = "Quyen loi cua nguoi lao dong la gi?"
    answer = "Nguoi lao dong duoc huong luong, nghi phep, va cac che do bao hiem xa hoi theo quy dinh. <|im_end|>"

    res1 = encode_sft_example(question, answer, tokenizer, max_seq_length=30)
    res2 = encode_sft_example(question, answer, tokenizer, max_seq_length=30)

    assert res1["input_ids"] == res2["input_ids"]
    assert res1["labels"] == res2["labels"]
    assert res1["attention_mask"] == res2["attention_mask"]
    assert res1["was_truncated"] == res2["was_truncated"]
