"""Dataset and encoding routines for official-data answer-only SFT with EOS preservation."""

from __future__ import annotations

from typing import Any

from legal_agentic_rag.exceptions import DataValidationError
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
) -> dict[str, Any]:
    """Format and tokenize one QA pair with strict answer-only loss masking and terminal EOS preservation."""
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
        raise DataValidationError("Formatted full text must strictly start with the prompt prefix")

    target_text = full_text[len(prompt_text) :]

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if not target_ids:
        raise DataValidationError("Formatted assistant target token sequence is empty")

    # Validate that assistant target terminates with the expected terminal EOS token
    terminal_token_id = target_ids[-1]
    expected_eos_id = getattr(tokenizer, "eos_token_id", None)
    if expected_eos_id is not None and terminal_token_id != expected_eos_id:
        # If tokenizer has im_end or other special tokens, check if target text ends properly
        eos_token_str = getattr(tokenizer, "eos_token", "<|im_end|>")
        if not target_text.endswith("<|im_end|>") and not target_text.endswith(eos_token_str):
            raise DataValidationError(
                f"Assistant target does not terminate with expected EOS token (got id {terminal_token_id}, expected {expected_eos_id})"
            )

    # Validate prompt leaves safe capacity for assistant target
    # Minimum safe capacity: at least 2 tokens (1 content + terminal EOS) or 1 token
    if len(prompt_ids) >= max_seq_length - 1:
        raise DataValidationError(
            f"Prompt length {len(prompt_ids)} leaves no safe assistant token capacity under max_seq_length={max_seq_length}"
        )

    original_target_count = len(target_ids)
    total_untruncated_len = len(prompt_ids) + len(target_ids)

    if total_untruncated_len <= max_seq_length:
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        was_truncated = False
        retained_target_count = original_target_count
    else:
        # Overlength case: preserve prompt, keep leading assistant tokens, and reserve final position for terminal EOS
        target_budget = max_seq_length - len(prompt_ids)
        leading_budget = target_budget - 1
        terminal_eos = target_ids[-1]
        retained_target = target_ids[:leading_budget] + [terminal_eos]

        input_ids = prompt_ids + retained_target
        labels = [-100] * len(prompt_ids) + retained_target
        was_truncated = True
        retained_target_count = len(retained_target)

    # Invariant assertions
    if len(input_ids) > max_seq_length:
        raise DataValidationError(
            f"Encoded sequence length {len(input_ids)} exceeds max_seq_length={max_seq_length}"
        )
    if labels[-1] != terminal_token_id:
        raise DataValidationError("Final non-masked label does not match the terminal EOS token")

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "was_truncated": was_truncated,
        "original_target_token_count": original_target_count,
        "retained_target_token_count": retained_target_count,
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
            raise DataValidationError("All SFT dataset questions must possess non-empty reference answers")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        assert record.reference_answer is not None
        return encode_sft_example(
            question=record.question,
            reference_answer=record.reference_answer,
            tokenizer=self._tokenizer,
            max_seq_length=self._max_seq_length,
            system_prompt=self._system_prompt,
        )
