"""Tests for QLoRA training runner parameter verification and environment inspection."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from legal_agentic_rag.exceptions import BackendInitializationError
from legal_agentic_rag.fine_tuning.training_runner import (
    M50QLoRATrainer,
    get_environment_dependency_versions,
    verify_trainable_parameters,
)
from legal_agentic_rag.schemas import QLoRACandidateConfig


class _MockLoRAModel(nn.Module):
    def __init__(self, unfrozen_base: bool = False, wrong_target: bool = False) -> None:
        super().__init__()
        self.model = nn.Module()
        # Large frozen base
        self.model.base_frozen = nn.Linear(500, 500, bias=False)  # 250,000 params (always frozen)
        for p in self.model.base_frozen.parameters():
            p.requires_grad = False

        # Small base layer for unfreeze testing
        self.model.embed_tokens = nn.Embedding(10, 10)  # 100 params
        for p in self.model.embed_tokens.parameters():
            p.requires_grad = unfrozen_base

        self.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        self.model.layers[0].self_attn = nn.Module()

        # LoRA adapter parameters
        target_name = "k_proj" if not wrong_target else "unrelated_proj"
        lora_linear = nn.Linear(10, 10, bias=False)  # 100 params (trainable)
        for p in lora_linear.parameters():
            p.requires_grad = True
        setattr(self.model.layers[0].self_attn, f"{target_name}_lora_A", lora_linear)


def test_verify_trainable_parameters_success() -> None:
    model = _MockLoRAModel()
    total, trainable, pct = verify_trainable_parameters(model, allowed_target_modules=["k_proj"])
    assert total == 250200
    assert trainable == 100
    assert pct < 0.1


def test_verify_trainable_parameters_rejects_unfrozen_base_layer() -> None:
    # Model with unfrozen embed_tokens
    model = _MockLoRAModel(unfrozen_base=True)
    with pytest.raises(BackendInitializationError, match="embed_tokens"):
        verify_trainable_parameters(model, allowed_target_modules=["k_proj"])


def test_verify_trainable_parameters_rejects_wrong_target_module() -> None:
    # Model with adapter on unrelated module not in allowed_target_modules
    model = _MockLoRAModel(wrong_target=True)
    with pytest.raises(BackendInitializationError, match="does not match allowed target modules"):
        verify_trainable_parameters(model, allowed_target_modules=["q_proj", "k_proj"])


def test_verify_trainable_parameters_rejects_zero_trainable() -> None:
    model = nn.Linear(10, 10)
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(BackendInitializationError, match="zero trainable parameters"):
        verify_trainable_parameters(model)


def test_verify_trainable_parameters_rejects_all_trainable() -> None:
    model = nn.Linear(10, 10)
    for p in model.parameters():
        p.requires_grad = True
    with pytest.raises(BackendInitializationError, match="trainable parameter ratio"):
        verify_trainable_parameters(model)


def test_get_environment_dependency_versions() -> None:
    versions = get_environment_dependency_versions()
    assert "torch" in versions
    assert versions["torch"] != "not_installed"


def test_candidate_1_lora_parameter_derivation() -> None:
    from legal_agentic_rag.fine_tuning.training_runner import EXPECTED_M50_C1_TRAINABLE_PARAMS

    hidden_size = 2048
    num_layers = 36
    num_attn_heads = 16
    num_kv_heads = 2
    head_dim = hidden_size // num_attn_heads  # 128
    kv_dim = num_kv_heads * head_dim  # 256
    lora_r = 8

    # q_proj: (hidden_size -> hidden_size)
    q_params = (hidden_size * lora_r) + (lora_r * hidden_size)  # 32,768
    # k_proj: (hidden_size -> kv_dim)
    k_params = (hidden_size * lora_r) + (lora_r * kv_dim)  # 18,432
    # v_proj: (hidden_size -> kv_dim)
    v_params = (hidden_size * lora_r) + (lora_r * kv_dim)  # 18,432
    # o_proj: (hidden_size -> hidden_size)
    o_params = (hidden_size * lora_r) + (lora_r * hidden_size)  # 32,768

    layer_params = q_params + k_params + v_params + o_params  # 102,400
    total_lora_params = num_layers * layer_params  # 3,686,400

    assert layer_params == 102_400
    assert total_lora_params == 3_686_400
    assert total_lora_params == EXPECTED_M50_C1_TRAINABLE_PARAMS


def test_verify_trainable_parameters_exact_count_check() -> None:
    model = _MockLoRAModel()  # 100 trainable params
    # Matching exact expectation
    verify_trainable_parameters(model, allowed_target_modules=["k_proj"], expected_trainable_params=100)

    # Mismatching exact expectation raises
    with pytest.raises(BackendInitializationError, match="Trainable parameter mismatch"):
        verify_trainable_parameters(model, allowed_target_modules=["k_proj"], expected_trainable_params=200)
