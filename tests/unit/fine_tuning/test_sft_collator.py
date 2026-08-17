"""Tests for dynamic padding SFT collator."""

from __future__ import annotations

import pytest
import torch

from legal_agentic_rag.fine_tuning.collator import SFTDynamicDataCollator


def test_collator_dynamic_padding() -> None:
    collator = SFTDynamicDataCollator(pad_token_id=0, pad_label_id=-100)

    feature1 = {
        "input_ids": [10, 20, 30],
        "attention_mask": [1, 1, 1],
        "labels": [-100, -100, 30],
    }
    feature2 = {
        "input_ids": [40, 50, 60, 70, 80],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, 60, 70, 80],
    }

    batch = collator([feature1, feature2])

    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)

    # Batch shape should be (2, 5) — max length in batch
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)

    # Feature 1 padded elements check
    assert batch["input_ids"][0].tolist() == [10, 20, 30, 0, 0]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["labels"][0].tolist() == [-100, -100, 30, -100, -100]

    # Feature 2 unpadded check
    assert batch["input_ids"][1].tolist() == [40, 50, 60, 70, 80]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]
    assert batch["labels"][1].tolist() == [-100, -100, 60, 70, 80]


def test_collator_rejects_empty_batch() -> None:
    collator = SFTDynamicDataCollator(pad_token_id=0)
    with pytest.raises(ValueError, match="empty batch"):
        collator([])
