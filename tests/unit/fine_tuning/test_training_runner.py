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


def test_optimizer_step_accounting_divisible_and_nondivisible() -> None:
    import math

    # Divisible case: 16 batches, accum 8 -> 2 steps
    batches_divisible = 16
    accum = 8
    assert math.ceil(batches_divisible / accum) == 2

    # Non-divisible case: 17 batches, accum 8 -> 3 steps
    batches_nondivisible = 17
    assert math.ceil(batches_nondivisible / accum) == 3

    # Candidate 1 exact calculation:
    train_records = 4500
    microbatch = 2
    total_batches = train_records // microbatch  # 2250
    steps_candidate_1 = math.ceil(total_batches / accum)
    assert steps_candidate_1 == 282
    assert total_batches % accum == 2  # Final partial group has 2 microbatches


def test_partial_gradient_accumulation_group_sizing() -> None:
    batches_in_epoch = 2250
    accum_steps = 8

    # First batch (step 1) in normal group
    step = 1
    group_index = (step - 1) // accum_steps
    is_final_group = group_index == ((batches_in_epoch - 1) // accum_steps)
    assert not is_final_group
    group_size_first = (
        batches_in_epoch % accum_steps
        if is_final_group and (batches_in_epoch % accum_steps != 0)
        else accum_steps
    )
    assert group_size_first == 8

    # Final batch (step 2250) in final partial group
    step = 2250
    group_index = (step - 1) // accum_steps
    is_final_group = group_index == ((batches_in_epoch - 1) // accum_steps)
    assert is_final_group
    group_size_final = (
        batches_in_epoch % accum_steps
        if is_final_group and (batches_in_epoch % accum_steps != 0)
        else accum_steps
    )
    assert group_size_final == 2


def test_seeded_dataloader_reproducible_ordering() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.arange(100))
    gen1 = torch.Generator().manual_seed(2026)
    gen2 = torch.Generator().manual_seed(2026)

    loader1 = DataLoader(dataset, batch_size=10, shuffle=True, generator=gen1)
    loader2 = DataLoader(dataset, batch_size=10, shuffle=True, generator=gen2)

    batches1 = [b[0].tolist() for b in loader1]
    batches2 = [b[0].tolist() for b in loader2]

    assert batches1 == batches2


class _MockRunnerTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        return "<|im_start|>user\nQ1?<|im_end|>\n<|im_start|>assistant\nA1.<|im_end|>\n"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2, 3]


def test_fail_closed_paged_adamw_optimizer_on_invalid_environment(tmp_path: Path) -> None:
    from legal_agentic_rag.schemas import QLoRACandidateConfig

    config = QLoRACandidateConfig(candidate_id="CUSTOM_TEST", optimizer="paged_adamw_8bit")
    trainer = M50QLoRATrainer(config=config)

    train_p = tmp_path / "sft_train.json"
    val_p = tmp_path / "sft_val.json"
    train_p.write_text('{"1": {"question": "Q1?", "answer": "A1."}}', encoding="utf-8")
    val_p.write_text('{"2": {"question": "Q2?", "answer": "A2."}}', encoding="utf-8")

    # In CPU environment where bitsandbytes is unavailable, fail closed
    # with BackendInitializationError
    mock_model = _MockLoRAModel()
    with pytest.raises(BackendInitializationError, match="paged_adamw_8bit"):
        trainer.train(
            train_partition_path=train_p,
            val_partition_path=val_p,
            output_directory=tmp_path / "out",
            model=mock_model,
            tokenizer=_MockRunnerTokenizer(),
        )


def test_git_commit_validation() -> None:
    from legal_agentic_rag.fine_tuning.training_runner import get_git_commit

    # Valid 40-char commit
    valid_sha = "aa4d1106e7abae7c67fa080f0dd097d63ff898c7"
    assert get_git_commit(explicit_commit=valid_sha) == valid_sha

    # Invalid commit raises
    with pytest.raises(BackendInitializationError, match="Invalid explicit commit"):
        get_git_commit(explicit_commit="short_commit")

    # Local repo discovery returns 40-character hex
    discovered = get_git_commit()
    assert len(discovered) == 40


def test_committed_candidate_1_config_parses_with_loader() -> None:
    from legal_agentic_rag.fine_tuning.training_runner import load_qlora_candidate_config

    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "configs" / "m50-c1-qwen3b-qlora-kaggle.example.json"

    assert config_path.exists()
    config, config_sha = load_qlora_candidate_config(config_path)

    assert config.candidate_id == "M50-C1"
    assert config.base_model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert config.base_model_revision == "a1d308dfcc03e09da285d49d912439a655a571e8"
    assert config.lora_r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.learning_rate == 5e-5
    assert config.max_seq_length == 1536
    assert config.per_device_train_batch_size == 2
    assert config.gradient_accumulation_steps == 8
    assert config.seed == 2026
    assert config.system_prompt == "Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam."
    assert config.use_cache is False
    assert config.training_partition == "sft_train.json"
    assert config.validation_partition == "sft_val.json"
    assert config.screening_partition == "screen_holdout.json"
    assert len(config_sha) == 64


def test_pre_training_config_and_manifest_fail_closed(tmp_path: Path) -> None:
    from legal_agentic_rag.exceptions import DataValidationError
    from legal_agentic_rag.schemas import QLoRACandidateConfig

    # 1. Reject partition filename mismatch before model load
    config = QLoRACandidateConfig(candidate_id="CUSTOM_TEST", training_partition="sft_train.json")
    trainer = M50QLoRATrainer(config=config)

    wrong_train = tmp_path / "wrong_train_name.json"
    val_p = tmp_path / "sft_val.json"
    wrong_train.write_text('{"1": {"question": "Q1?", "answer": "A1."}}', encoding="utf-8")
    val_p.write_text('{"2": {"question": "Q2?", "answer": "A2."}}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="Train partition filename mismatch"):
        trainer.train(
            train_partition_path=wrong_train,
            val_partition_path=val_p,
            output_directory=tmp_path / "out",
        )

    # 2. Reject missing split manifest for Candidate 1 before model load
    c1_config = QLoRACandidateConfig(candidate_id="M50-C1")
    c1_trainer = M50QLoRATrainer(config=c1_config)

    correct_train = tmp_path / "sft_train.json"
    correct_train.write_text('{"1": {"question": "Q1?", "answer": "A1."}}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="M50-C1 training requires a valid existing split_manifest_path"):
        c1_trainer.train(
            train_partition_path=correct_train,
            val_partition_path=val_p,
            output_directory=tmp_path / "out2",
            split_manifest_path=None,
        )
