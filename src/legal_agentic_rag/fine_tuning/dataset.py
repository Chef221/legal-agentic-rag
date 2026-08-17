"""Dataset and encoding routines for official-data answer-only SFT."""

from __future__ import annotations

from typing import Any

from legal_agentic_rag.schemas import CompetitionQuestion

SYSTEM_PROMPT = "Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam."
DEFAULT_MAX_SEQ_LENGTH = 1536


def encode_sft_example(
    question: str,
    reference_answer: str,
    tokenizer: Any,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, list[int]]:
    """Format and tokenize one QA pair with strict answer-only loss masking."""
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": reference_answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    if not full_text.startswith(prompt_text):
        raise ValueError("Formatted full text must strictly start with the prompt prefix")

    target_text = full_text[len(prompt_text) :]

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    # Enforce sequence length truncation ceiling
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SFTAnswerOnlyDataset:
    """In-memory dataset container converting official questions to supervised SFT features."""

    def __init__(
        self,
        questions: list[CompetitionQuestion],
        tokenizer: Any,
        *,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._system_prompt = system_prompt
        self._records = [q for q in questions if q.reference_answer is not None]
        if len(self._records) != len(questions):
            raise ValueError("All SFT dataset questions must possess non-empty reference answers")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self._records[index]
        assert record.reference_answer is not None
        return encode_sft_example(
            question=record.question,
            reference_answer=record.reference_answer,
            tokenizer=self._tokenizer,
            max_seq_length=self._max_seq_length,
            system_prompt=self._system_prompt,
        )
