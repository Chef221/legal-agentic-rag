"""Dynamic padding collator for answer-only SFT batches."""

from __future__ import annotations

from typing import Any

import torch


class SFTDynamicDataCollator:
    """Dynamically pad input batches to the longest example within the batch."""

    def __init__(self, pad_token_id: int, pad_label_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch of features")

        max_len = max(len(feature["input_ids"]) for feature in features)

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [self.pad_label_id] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
