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
    raw_target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if not raw_target_ids:
        raise DataValidationError("Formatted assistant target token sequence is empty")

    expected_eos_id = getattr(tokenizer, "eos_token_id", None)
    if expected_eos_id is None:
        raise DataValidationError("Tokenizer must define an explicit eos_token_id for SFT encoding")

    # Validate target text ends with terminal EOS marker (e.g. <|im_end|>) allowing only trailing template whitespace
    eos_token_str = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
    if not target_text.rstrip().endswith(eos_token_str):
        raise DataValidationError(
            f"Assistant target text does not end with terminal EOS marker {repr(eos_token_str)}: {repr(target_text[-40:])}"
        )

    last_eos_text_pos = target_text.rfind(eos_token_str)
    text_after_eos = target_text[last_eos_text_pos + len(eos_token_str) :]
    if text_after_eos and not text_after_eos.isspace():
        raise DataValidationError(
            f"Meaningful content found after terminal EOS marker: {repr(text_after_eos)}"
        )

    # Locate the FINAL expected EOS token in raw_target_ids
    if expected_eos_id not in raw_target_ids:
        raise DataValidationError(
            f"Assistant target token IDs do not contain expected EOS token ID {expected_eos_id}"
        )

    last_eos_idx = len(raw_target_ids) - 1 - raw_target_ids[::-1].index(expected_eos_id)
    target_ids = raw_target_ids[: last_eos_idx + 1]

    if not target_ids or target_ids[-1] != expected_eos_id:
        raise DataValidationError(
            f"Canonicalized assistant target does not terminate with expected EOS token ID {expected_eos_id}"
        )

    # Validate prompt leaves safe capacity for assistant target
    # Minimum safe capacity: at least 2 tokens (1 content + terminal EOS)
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
        retained_target = target_ids[:leading_budget] + [expected_eos_id]

        input_ids = prompt_ids + retained_target
        labels = [-100] * len(prompt_ids) + retained_target
        was_truncated = True
        retained_target_count = len(retained_target)

    # Invariant assertions
    if len(input_ids) > max_seq_length:
        raise DataValidationError(
            f"Encoded sequence length {len(input_ids)} exceeds max_seq_length={max_seq_length}"
        )
    if labels[-1] != expected_eos_id:
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


def validate_sft_dataset_encoding(
    train_questions: list[CompetitionQuestion],
    val_questions: list[CompetitionQuestion],
    tokenizer: Any,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Preflight audit validating that every SFT training and validation record encodes with terminal EOS."""
    record_count = len(train_questions) + len(val_questions)
    encoding_success_count = 0
    terminal_eos_verified_count = 0
    truncated_count = 0
    max_encoded_length = 0
    failures: list[dict[str, Any]] = []

    expected_eos_id = getattr(tokenizer, "eos_token_id", None)
    if expected_eos_id is None:
        raise DataValidationError("Tokenizer must define an explicit eos_token_id for SFT preflight")

    all_records = [("train", q) for q in train_questions] + [("val", q) for q in val_questions]

    for split_name, q in all_records:
        if q.reference_answer is None:
            failures.append({"question_id": q.question_id, "split": split_name, "error": "Missing reference answer"})
            continue
        try:
            encoded = encode_sft_example(
                question=q.question,
                reference_answer=q.reference_answer,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                system_prompt=system_prompt,
            )
            input_ids = encoded["input_ids"]
            labels = encoded["labels"]

            if len(input_ids) > max_seq_length:
                failures.append({
                    "question_id": q.question_id,
                    "split": split_name,
                    "error": f"Length {len(input_ids)} > {max_seq_length}",
                })
                continue
            if labels[-1] != expected_eos_id:
                failures.append({
                    "question_id": q.question_id,
                    "split": split_name,
                    "error": f"Final label {labels[-1]} != expected EOS {expected_eos_id}",
                })
                continue

            encoding_success_count += 1
            terminal_eos_verified_count += 1
            if encoded["was_truncated"]:
                truncated_count += 1
            if len(input_ids) > max_encoded_length:
                max_encoded_length = len(input_ids)

        except Exception as err:
            failures.append({"question_id": q.question_id, "split": split_name, "error": str(err)})

    report = {
        "record_count": record_count,
        "encoding_success_count": encoding_success_count,
        "terminal_eos_verified_count": terminal_eos_verified_count,
        "truncated_count": truncated_count,
        "max_encoded_length": max_encoded_length,
        "failure_count": len(failures),
        "failures": failures,
    }

    if failures or encoding_success_count != record_count or terminal_eos_verified_count != record_count:
        raise DataValidationError(
            f"SFT preflight encoding audit failed with {len(failures)} failures out of {record_count} records: {failures[:5]}"
        )

    return report


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
