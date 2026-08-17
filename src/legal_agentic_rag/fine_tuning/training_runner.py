"""Candidate-1 QLoRA training execution and reproducible manifest generation."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.development_split import _file_sha256
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    ConfigurationError,
    DataValidationError,
)
from legal_agentic_rag.fine_tuning.collator import SFTDynamicDataCollator
from legal_agentic_rag.fine_tuning.dataset import (
    DEFAULT_MAX_SEQ_LENGTH,
    SFTAnswerOnlyDataset,
)
from legal_agentic_rag.schemas import M50TrainingManifest, QLoRACandidateConfig

TRAINING_MANIFEST_FILENAME = "training_manifest.json"
TRAINING_HISTORY_FILENAME = "training_history.json"
ADAPTER_MODEL_FILENAME = "adapter_model.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"


EXPECTED_M50_C1_TRAINABLE_PARAMS = 3_686_400


def get_environment_dependency_versions() -> dict[str, str]:
    """Inspect and return installed versions of key training dependencies."""
    packages = ["transformers", "peft", "bitsandbytes", "accelerate", "torch", "numpy"]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def verify_trainable_parameters(
    model: Any,
    allowed_target_modules: list[str] | None = None,
    expected_trainable_params: int | None = None,
) -> tuple[int, int, float]:
    """Verify that only intended LoRA adapter parameters require gradients."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params == 0:
        raise BackendInitializationError("Model has zero total parameters")
    if trainable_params == 0:
        raise BackendInitializationError("Model has zero trainable parameters — LoRA setup failed")

    trainable_pct = (trainable_params / total_params) * 100.0

    # Ensure we are not training the whole base model (LoRA should be < 3% of total)
    if trainable_pct > 3.0:
        raise BackendInitializationError(
            f"Unusually high trainable parameter ratio: {trainable_pct:.2f}% ({trainable_params}/{total_params})"
        )

    # Exact expected parameter count assertion if specified
    if expected_trainable_params is not None:
        if trainable_params != expected_trainable_params:
            raise BackendInitializationError(
                f"Trainable parameter mismatch: expected exactly {expected_trainable_params:,}, got {trainable_params:,}"
            )

    # Inspect all named parameters requiring gradients
    forbidden_substrings = ["embed_tokens", "lm_head", "layernorm", "norm.", "norm_"]
    trainable_names: list[str] = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)
            # Ensure parameter is a LoRA adapter parameter
            if "lora_" not in name.lower() and "adapter" not in name.lower():
                raise BackendInitializationError(
                    f"Non-LoRA base parameter '{name}' is marked as trainable"
                )
            # Ensure parameter is not a forbidden layer
            if any(sub in name.lower() for sub in forbidden_substrings):
                raise BackendInitializationError(
                    f"Forbidden layer '{name}' is marked as trainable"
                )
            # If allowed target modules are specified, ensure it belongs to one of them
            if allowed_target_modules:
                if not any(target in name for target in allowed_target_modules):
                    raise BackendInitializationError(
                        f"Trainable parameter '{name}' does not match allowed target modules: {allowed_target_modules}"
                    )

    return total_params, trainable_params, trainable_pct


class M50QLoRATrainer:
    """Execute bounded Candidate 1 QLoRA fine-tuning on official SFT partitions."""

    def __init__(
        self,
        config: QLoRACandidateConfig | None = None,
        loader: UitDsc2026DataLoader | None = None,
    ) -> None:
        self.config = config or QLoRACandidateConfig()
        self.loader = loader or UitDsc2026DataLoader()

    def train(
        self,
        train_partition_path: Path,
        val_partition_path: Path,
        output_directory: Path,
        *,
        split_manifest_path: Path | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        device: torch.device | None = None,
    ) -> M50TrainingManifest:
        """Run 1-epoch QLoRA training and persist the best checkpoint and manifest."""
        output = output_directory.resolve()
        if output.exists() and any(output.iterdir()):
            raise ArtifactCompatibilityError(
                f"Training output directory {output} is not empty"
            )
        output.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now(UTC)

        # 1. Validate dataset sources
        sft_train_sha = _file_sha256(train_partition_path)
        sft_val_sha = _file_sha256(val_partition_path)
        split_manifest_sha = (
            _file_sha256(split_manifest_path) if split_manifest_path and split_manifest_path.exists() else "unknown"
        )

        train_questions = self.loader.load_questions(train_partition_path, require_reference_answers=True)
        val_questions = self.loader.load_questions(val_partition_path, require_reference_answers=True)

        # 2. Setup Device & Dependencies
        training_device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        dep_versions = get_environment_dependency_versions()

        # 3. Model & Tokenizer Initialization
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_id,
                revision=self.config.base_model_revision,
                trust_remote_code=False,
            )

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        if model is None:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=self.config.double_quantization,
                bnb_4bit_compute_dtype=torch.float16 if self.config.compute_dtype == "float16" else torch.bfloat16,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                revision=self.config.base_model_revision,
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if self.config.compute_dtype == "float16" else torch.bfloat16,
                trust_remote_code=False,
            )

            if self.config.gradient_checkpointing:
                base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False

            base_model = prepare_model_for_kbit_training(base_model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, lora_config)

        # 4. Parameter Preflight Check
        expected_params = (
            EXPECTED_M50_C1_TRAINABLE_PARAMS if self.config.candidate_id == "M50-C1" else None
        )
        total_params, trainable_params, trainable_pct = verify_trainable_parameters(
            model,
            allowed_target_modules=self.config.target_modules,
            expected_trainable_params=expected_params,
        )

        # 5. Datasets and DataLoaders
        train_dataset = SFTAnswerOnlyDataset(
            train_questions, tokenizer, max_seq_length=self.config.max_seq_length
        )
        val_dataset = SFTAnswerOnlyDataset(
            val_questions, tokenizer, max_seq_length=self.config.max_seq_length
        )

        collator = SFTDynamicDataCollator(pad_token_id=pad_token_id, pad_label_id=-100)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        # 6. Optimizer and Scheduler
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW8bit(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
            )
        except Exception:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
            )

        total_steps = (len(train_loader) // self.config.gradient_accumulation_steps) * self.config.num_train_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, 1)
        )

        # 7. Training Loop with Periodic Validation
        model.train()
        global_step = 0
        best_val_loss = float("inf")
        best_step = 0
        final_val_loss = float("inf")
        history_entries: list[dict[str, Any]] = []

        optimizer.zero_grad()

        for epoch in range(self.config.num_train_epochs):
            accumulated_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(training_device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                if step % self.config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if hasattr(scheduler, "get_last_lr")
                        else self.config.learning_rate
                    )

                    val_loss_recorded: float | None = None
                    # Periodic Validation Evaluation
                    if global_step % self.config.eval_steps == 0 or step == len(train_loader):
                        val_loss = self._evaluate_loss(model, val_loader, training_device)
                        final_val_loss = val_loss
                        val_loss_recorded = val_loss
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = global_step
                            self._save_checkpoint(model, output / "best_checkpoint")
                        model.train()

                    if (
                        global_step % self.config.logging_steps == 0
                        or val_loss_recorded is not None
                        or step == len(train_loader)
                    ):
                        history_entries.append(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "train_loss": round(accumulated_loss, 6),
                                "learning_rate": current_lr,
                                "eval_loss": (
                                    round(val_loss_recorded, 6)
                                    if val_loss_recorded is not None
                                    else None
                                ),
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )
                    accumulated_loss = 0.0

        # Finalize and Save Checkpoint
        final_checkpoint_dir = output / "final_checkpoint"
        self._save_checkpoint(model, final_checkpoint_dir)

        end_time = datetime.now(UTC)

        # Save project-owned training history
        history_payload = {
            "schema_version": "1.0",
            "created_at": end_time.isoformat(),
            "code_version": __version__,
            "candidate_id": self.config.candidate_id,
            "total_steps": total_steps,
            "num_train_epochs": self.config.num_train_epochs,
            "best_checkpoint_step": best_step,
            "best_validation_loss": (
                best_val_loss if best_val_loss != float("inf") else final_val_loss
            ),
            "final_validation_loss": final_val_loss,
            "history": history_entries,
        }
        (output / TRAINING_HISTORY_FILENAME).write_text(
            json.dumps(history_payload, indent=2), encoding="utf-8"
        )

        config_sha = sha256(self.config.model_dump_json().encode("utf-8")).hexdigest()

        manifest = M50TrainingManifest(
            created_at=end_time,
            code_version=__version__,
            candidate_id=self.config.candidate_id,
            git_commit=self._get_git_commit(),
            source_split_manifest_sha256=split_manifest_sha,
            sft_train_sha256=sft_train_sha,
            sft_val_sha256=sft_val_sha,
            base_model_id=self.config.base_model_id,
            base_model_revision=self.config.base_model_revision,
            tokenizer_id=self.config.base_model_id,
            tokenizer_revision=self.config.base_model_revision,
            training_config_sha256=config_sha,
            lora_config={
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "target_modules": self.config.target_modules,
            },
            dependency_versions=dep_versions,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            seed=self.config.seed,
            max_seq_length=self.config.max_seq_length,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            trainable_percentage=trainable_pct,
            training_start_time=start_time,
            training_end_time=end_time,
            best_checkpoint_step=best_step,
            best_validation_loss=best_val_loss if best_val_loss != float("inf") else final_val_loss,
            final_validation_loss=final_val_loss,
            warnings=[],
        )

        (output / TRAINING_MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _evaluate_loss(model: Any, val_loader: DataLoader, device: torch.device) -> float:
        model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                batches += 1
        return total_loss / max(batches, 1)

    @staticmethod
    def _save_checkpoint(model: Any, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(path))
        else:
            torch.save(model.state_dict(), str(path / "pytorch_model.bin"))

    @staticmethod
    def _get_git_commit() -> str:
        import subprocess

        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "unknown_commit"
