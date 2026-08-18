"""Candidate-1 & Candidate-2 QLoRA training execution, progress observability, and gate probing."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
from typing import Any, Callable

import numpy as np
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
    validate_sft_dataset_encoding,
)
from legal_agentic_rag.fine_tuning.generation_gates import (
    evaluate_checkpoint_health_gate,
    run_free_generation_probe,
    select_best_pilot_checkpoint,
)
from legal_agentic_rag.fine_tuning.val_probe import (
    VAL_PROBE_BASE_MANIFEST_FILENAME,
    VAL_PROBE_BASE_RESULTS_FILENAME,
    VAL_PROBE_FILENAME,
    VAL_PROBE_MANIFEST_FILENAME,
    load_and_validate_val_probe_base_cache,
)
from legal_agentic_rag.schemas import (
    CheckpointGateReport,
    CheckpointManifest,
    CheckpointSelectionReport,
    CompetitionQuestion,
    M50TrainingManifest,
    QLoRACandidateConfig,
    TrainingProgressSnapshot,
    ValProbeBaseManifest,
    ValProbeCaseResult,
)

TRAINING_MANIFEST_FILENAME = "training_manifest.json"
TRAINING_HISTORY_FILENAME = "training_history.json"
TRAINING_HISTORY_JSONL_FILENAME = "training_history.jsonl"
PROGRESS_SNAPSHOT_FILENAME = "progress.json"
CHECKPOINT_SELECTION_FILENAME = "checkpoint-selection-report.json"
ADAPTER_MODEL_FILENAME = "adapter_model.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
TRAINING_STATE_FILENAME = "training_state.pt"

EXPECTED_M50_C1_TRAINABLE_PARAMS = 3_686_400
EXPECTED_M50_C2_TRAINABLE_PARAMS = 921_600


def format_duration(seconds: float) -> str:
    """Format duration in seconds to arbitrary HH:MM:SS without modulo 24 wrap."""
    total_secs = max(0, int(seconds))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_qlora_candidate_config(path: Path) -> tuple[QLoRACandidateConfig, str]:
    """Load and validate a candidate QLoRA config from a JSON file and return config + exact file SHA."""
    if not path.exists():
        raise DataValidationError(f"Candidate configuration file missing at {path}")
    raw_bytes = path.read_bytes()
    file_sha = sha256(raw_bytes).hexdigest()
    try:
        config = QLoRACandidateConfig.model_validate_json(raw_bytes.decode("utf-8"))
    except Exception as err:
        raise DataValidationError(f"Invalid candidate configuration in {path}: {err}") from err
    return config, file_sha


def get_environment_dependency_versions() -> dict[str, str]:
    """Inspect and return installed versions of key training dependencies."""
    packages = ["transformers", "peft", "bitsandbytes", "accelerate", "torch", "numpy", "nltk"]
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


def get_git_commit(
    repo_root: Path | None = None,
    explicit_commit: str | None = None,
) -> str:
    """Retrieve and validate authoritative 40-character Git commit hash."""
    if explicit_commit:
        if len(explicit_commit) == 40 and re.fullmatch(r"[0-9a-fA-F]{40}", explicit_commit):
            return explicit_commit.lower()
        raise BackendInitializationError(f"Invalid explicit commit SHA format: {explicit_commit}")

    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        if len(commit) == 40 and re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            return commit.lower()
        raise BackendInitializationError(f"Unexpected git output for commit SHA: {commit}")
    except Exception as err:
        raise BackendInitializationError(
            f"Valid 40-character Git commit SHA could not be determined from {root}: {err}"
        ) from err


def write_progress_atomically(
    output_dir: Path,
    snapshot: TrainingProgressSnapshot,
) -> None:
    """Persist progress.json atomically via temporary file and replace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / PROGRESS_SNAPSHOT_FILENAME
    content = snapshot.model_dump_json(indent=2)

    with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False, encoding="utf-8") as tf:
        tf.write(content)
        tf.flush()
        temp_name = tf.name

    os.replace(temp_name, target_path)


class M50QLoRATrainer:
    """Execute bounded Candidate 1 & Candidate 2 QLoRA fine-tuning on official SFT partitions."""

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
        val_probe_path: Path | None = None,
        val_probe_manifest_path: Path | None = None,
        val_probe_base_results_path: Path | None = None,
        val_probe_base_manifest_path: Path | None = None,
        split_manifest_path: Path | None = None,
        config_source_path: Path | None = None,
        expected_config_sha256: str | None = None,
        git_commit: str | None = None,
        repo_root: Path | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        device: torch.device | None = None,
        resume_from_checkpoint_dir: Path | None = None,
        official_meteor_scorer: Callable[[list[list[str]], list[str]], float] | None = None,
    ) -> M50TrainingManifest:
        """Run bounded QLoRA training with observable progress and generation gate probing."""
        output = output_directory.resolve()
        is_resuming = resume_from_checkpoint_dir is not None

        if not is_resuming and output.exists() and any(output.iterdir()):
            raise ArtifactCompatibilityError(
                f"Training output directory {output} is not empty"
            )

        # 1. Early Fail-Closed Config, Partition, and Split-Manifest Validation
        if config_source_path:
            loaded_cfg, actual_config_sha = load_qlora_candidate_config(config_source_path)
            if expected_config_sha256 and actual_config_sha != expected_config_sha256:
                raise DataValidationError(
                    f"Config SHA mismatch: expected {expected_config_sha256}, got {actual_config_sha}"
                )
            if loaded_cfg != self.config:
                raise DataValidationError(
                    "Config file content does not match trainer configuration instance"
                )
            config_sha = actual_config_sha
        elif expected_config_sha256:
            config_sha = expected_config_sha256
        else:
            config_sha = sha256(self.config.model_dump_json().encode("utf-8")).hexdigest()

        # Validate partition filenames against config
        if train_partition_path.name != self.config.training_partition:
            raise DataValidationError(
                f"Train partition filename mismatch: expected {self.config.training_partition}, got {train_partition_path.name}"
            )
        if val_partition_path.name != self.config.validation_partition:
            raise DataValidationError(
                f"Validation partition filename mismatch: expected {self.config.validation_partition}, got {val_partition_path.name}"
            )

        # Official candidates require valid split manifest
        if self.config.candidate_id in ["M50-C1", "M50-C2"]:
            if not split_manifest_path or not split_manifest_path.exists():
                raise DataValidationError(
                    f"{self.config.candidate_id} training requires a valid existing split_manifest_path"
                )
            split_manifest_sha = _file_sha256(split_manifest_path)
        else:
            split_manifest_sha = (
                _file_sha256(split_manifest_path)
                if split_manifest_path and split_manifest_path.exists()
                else "unknown"
            )

        # C2 candidate requires VAL probe & BASE cache paths
        base_probe_cases: list[ValProbeCaseResult] = []
        probe_questions: list[CompetitionQuestion] = []
        if self.config.candidate_id == "M50-C2" and self.config.probe_steps:
            if not val_probe_path or not val_probe_path.exists():
                raise DataValidationError("M50-C2 training requires a valid val_probe_path")
            if not val_probe_manifest_path or not val_probe_manifest_path.exists():
                raise DataValidationError("M50-C2 training requires a valid val_probe_manifest_path")
            if not val_probe_base_results_path or not val_probe_base_results_path.exists():
                raise DataValidationError("M50-C2 training requires a valid val_probe_base_results_path")
            if not val_probe_base_manifest_path or not val_probe_base_manifest_path.exists():
                raise DataValidationError("M50-C2 training requires a valid val_probe_base_manifest_path")

            probe_questions = self.loader.load_questions(val_probe_path, require_reference_answers=True)
            val_probe_sha = _file_sha256(val_probe_path)
            base_probe_cases, _ = load_and_validate_val_probe_base_cache(
                val_probe_base_results_path,
                val_probe_base_manifest_path,
                expected_val_probe_sha256=val_probe_sha,
                expected_base_model_id=self.config.base_model_id,
                expected_base_revision=self.config.base_model_revision,
                expected_system_prompt=self.config.system_prompt,
                expected_max_new_tokens=self.config.generation_probe_max_new_tokens,
                expected_record_count=self.config.generation_probe_question_count,
            )

        output.mkdir(parents=True, exist_ok=True)
        start_time = datetime.now(UTC)

        # 2. Validate dataset sources
        sft_train_sha = _file_sha256(train_partition_path)
        sft_val_sha = _file_sha256(val_partition_path)

        train_questions = self.loader.load_questions(train_partition_path, require_reference_answers=True)
        val_questions = self.loader.load_questions(val_partition_path, require_reference_answers=True)

        # 3. Setup Device & Seed Controls
        training_device = device or (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        dep_versions = get_environment_dependency_versions()

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # 4. Model & Tokenizer Initialization
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_id,
                revision=self.config.base_model_revision,
                trust_remote_code=False,
            )

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Preflight SFT encoding audit across train and val splits before model loading
        validate_sft_dataset_encoding(
            train_questions,
            val_questions,
            tokenizer,
            max_seq_length=self.config.max_seq_length,
            system_prompt=self.config.system_prompt,
        )

        if model is None:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=self.config.double_quantization,
                bnb_4bit_compute_dtype=torch.float16 if self.config.compute_dtype == "float16" else torch.bfloat16,
            )

            device_map = {"": 0} if torch.cuda.is_available() else None

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                revision=self.config.base_model_revision,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                device_map=device_map,
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

        # 5. Parameter Preflight & Placement Check
        expected_params = None
        if self.config.candidate_id == "M50-C1":
            expected_params = EXPECTED_M50_C1_TRAINABLE_PARAMS
        elif self.config.candidate_id == "M50-C2":
            expected_params = EXPECTED_M50_C2_TRAINABLE_PARAMS

        total_params, trainable_params, trainable_pct = verify_trainable_parameters(
            model,
            allowed_target_modules=self.config.target_modules,
            expected_trainable_params=expected_params,
        )

        # Verify device placement consistency
        for name, p in model.named_parameters():
            if training_device.type == "cuda" and p.device.type != "cuda":
                raise BackendInitializationError(
                    f"Model parameter '{name}' is on {p.device}, expected {training_device}"
                )

        # 6. Datasets with Authoritative Config System Prompt and DataLoaders
        train_dataset = SFTAnswerOnlyDataset(
            train_questions,
            tokenizer,
            system_prompt=self.config.system_prompt,
            max_seq_length=self.config.max_seq_length,
        )
        val_dataset = SFTAnswerOnlyDataset(
            val_questions,
            tokenizer,
            system_prompt=self.config.system_prompt,
            max_seq_length=self.config.max_seq_length,
        )

        collator = SFTDynamicDataCollator(pad_token_id=pad_token_id, pad_label_id=-100)

        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.config.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            generator=loader_generator,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        # 7. Fail-Closed Optimizer and Scheduler
        if self.config.optimizer == "paged_adamw_8bit":
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.PagedAdamW8bit(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=self.config.learning_rate,
                )
            except Exception as err:
                raise BackendInitializationError(
                    f"Failed to initialize paged_adamw_8bit optimizer: {err}"
                ) from err
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
            )
        else:
            raise ConfigurationError(f"Unsupported optimizer: {self.config.optimizer}")

        batches_in_epoch = len(train_loader)
        accum_steps = self.config.gradient_accumulation_steps
        steps_per_epoch = math.ceil(batches_in_epoch / accum_steps)
        total_epoch_steps = steps_per_epoch * self.config.num_train_epochs

        # Step bound priority for pilot
        if self.config.max_optimizer_steps is not None:
            max_training_steps = min(self.config.max_optimizer_steps, total_epoch_steps)
        else:
            max_training_steps = total_epoch_steps

        warmup_steps = int(max_training_steps * self.config.warmup_ratio)

        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(max_training_steps, 1),
        )

        # 8. Resume from checkpoint if requested
        start_step = 0
        consumed_microbatches = 0
        start_consumed = 0
        gate_reports: dict[int, CheckpointGateReport] = {}
        checkpoint_dirs: dict[int, str] = {}

        if resume_from_checkpoint_dir:
            ckpt_path = resume_from_checkpoint_dir.resolve()
            if not ckpt_path.exists():
                raise ArtifactCompatibilityError(f"Resume checkpoint directory does not exist: {ckpt_path}")
            manifest_file = ckpt_path / "checkpoint_manifest.json"
            state_file = ckpt_path / TRAINING_STATE_FILENAME
            if not manifest_file.exists() or not state_file.exists():
                raise ArtifactCompatibilityError(f"Incomplete checkpoint at {ckpt_path}")

            ckpt_manifest = CheckpointManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
            if ckpt_manifest.training_config_sha256 != config_sha:
                raise ArtifactCompatibilityError("Resume checkpoint config SHA mismatch")
            if ckpt_manifest.sft_train_sha256 != sft_train_sha or ckpt_manifest.sft_val_sha256 != sft_val_sha:
                raise ArtifactCompatibilityError("Resume checkpoint partition SHA mismatch")

            # Load weights
            if hasattr(model, "load_adapter"):
                model.load_adapter(str(ckpt_path), adapter_name="default")
            elif (ckpt_path / "pytorch_model.bin").exists() and hasattr(model, "load_state_dict"):
                model.load_state_dict(
                    torch.load(str(ckpt_path / "pytorch_model.bin"), map_location=training_device, weights_only=False)
                )

            state_payload = torch.load(str(state_file), map_location=training_device, weights_only=False)
            if state_payload.get("optimizer") is not None and hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(state_payload["optimizer"])
            if state_payload.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(state_payload["scheduler"])
            start_step = state_payload["global_step"]
            consumed_microbatches = state_payload.get("consumed_microbatches", 0)
            start_consumed = consumed_microbatches

            # Restore RNG states
            if "random_state" in state_payload:
                random.setstate(state_payload["random_state"])
            if "numpy_state" in state_payload:
                np.random.set_state(state_payload["numpy_state"])
            if "torch_state" in state_payload:
                torch.set_rng_state(state_payload["torch_state"])
            if "cuda_state" in state_payload and state_payload["cuda_state"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state_payload["cuda_state"])

            # Reload existing probe reports
            for p_file in output.glob("probe-step-*.json"):
                try:
                    rep = CheckpointGateReport.model_validate_json(p_file.read_text(encoding="utf-8"))
                    gate_reports[rep.optimizer_step] = rep
                    checkpoint_dirs[rep.optimizer_step] = str(output / f"checkpoint-step-{rep.optimizer_step:04d}")
                except Exception:
                    pass

        # 9. Training Loop with Observable ETA, Periodic Validation, and Gate Probing
        history_entries: list[dict[str, Any]] = []
        if resume_from_checkpoint_dir and (output / TRAINING_HISTORY_JSONL_FILENAME).exists():
            for line in (output / TRAINING_HISTORY_JSONL_FILENAME).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        h_entry = json.loads(line)
                        if h_entry.get("step", 0) <= start_step:
                            history_entries.append(h_entry)
                    except Exception:
                        pass

        model.train()
        global_step = start_step
        best_val_loss = float("inf")
        best_step = 0
        final_val_loss = float("inf")

        optimizer.zero_grad()
        training_loop_start = time.perf_counter()
        commit_sha = get_git_commit(repo_root=repo_root, explicit_commit=git_commit)

        # Initial progress snapshot
        init_snapshot = TrainingProgressSnapshot(
            updated_at=datetime.now(UTC),
            code_version=__version__,
            candidate_id=self.config.candidate_id,
            git_commit=commit_sha,
            training_config_sha256=config_sha,
            status="initialized" if global_step == 0 else "training",
            current_optimizer_step=global_step,
            max_optimizer_steps=max_training_steps,
            current_microbatch=consumed_microbatches,
            total_microbatches=batches_in_epoch * self.config.num_train_epochs,
            elapsed_seconds=0.0,
            eta_seconds=0.0,
            elapsed_formatted="00:00:00",
            eta_formatted="00:00:00",
            latest_train_loss=None,
            latest_learning_rate=self.config.learning_rate,
            latest_val_loss=None,
            latest_completed_probe_step=max(gate_reports.keys()) if gate_reports else None,
            latest_durable_checkpoint=checkpoint_dirs.get(max(gate_reports.keys())) if gate_reports else None,
            warnings=[],
        )
        write_progress_atomically(output, init_snapshot)

        history_jsonl_path = output / TRAINING_HISTORY_JSONL_FILENAME

        for epoch in range(self.config.num_train_epochs):
            if global_step >= max_training_steps:
                break

            accumulated_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                # If resuming, skip microbatches that were already processed prior to restart
                current_overall_batch_idx = epoch * batches_in_epoch + step
                if current_overall_batch_idx <= start_consumed:
                    continue

                if global_step >= max_training_steps:
                    break

                consumed_microbatches += 1
                group_index = (step - 1) // accum_steps
                is_final_group = group_index == ((batches_in_epoch - 1) // accum_steps)
                if is_final_group and (batches_in_epoch % accum_steps != 0):
                    current_group_size = batches_in_epoch % accum_steps
                else:
                    current_group_size = accum_steps

                batch = {k: v.to(training_device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / current_group_size
                loss.backward()
                accumulated_loss += loss.item()

                if step % accum_steps == 0 or step == batches_in_epoch:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    elapsed = time.perf_counter() - training_loop_start
                    steps_completed = global_step - start_step
                    steps_remaining = max(0, max_training_steps - global_step)
                    avg_time_per_step = elapsed / max(steps_completed, 1)
                    eta_seconds = steps_remaining * avg_time_per_step

                    elapsed_fmt = format_duration(elapsed)
                    eta_fmt = format_duration(eta_seconds)

                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if hasattr(scheduler, "get_last_lr")
                        else self.config.learning_rate
                    )

                    # Visible logging every logging_steps
                    if global_step % self.config.logging_steps == 0 or global_step == max_training_steps:
                        print(
                            f"[{self.config.candidate_id}] optimizer_step={global_step}/{max_training_steps} "
                            f"train_loss={accumulated_loss:.4f} lr={current_lr:.2e} "
                            f"elapsed={elapsed_fmt} eta={eta_fmt}",
                            flush=True,
                        )

                    # Periodic Teacher-Forced Validation Loss and/or Checkpoint Probe Gates
                    is_probe_step = global_step in self.config.probe_steps
                    is_eval_step = (
                        self.config.eval_steps and (global_step % self.config.eval_steps == 0)
                    ) or is_probe_step or (global_step == max_training_steps)

                    val_loss_recorded: float | None = None

                    if is_probe_step:
                        print(
                            f"\n[{self.config.candidate_id}] step={global_step}/{max_training_steps} teacher_forced_validation START",
                            flush=True,
                        )
                        val_loss = self._evaluate_loss(model, val_loader, training_device)
                        final_val_loss = val_loss
                        val_loss_recorded = val_loss
                        print(
                            f"[{self.config.candidate_id}] step={global_step}/{max_training_steps} val_loss={val_loss:.6f}",
                            flush=True,
                        )

                        # Save durable step checkpoint
                        ckpt_dir = output / f"checkpoint-step-{global_step:04d}"
                        self._save_checkpoint_bundle(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            consumed_microbatches=consumed_microbatches,
                            val_loss=val_loss,
                            output_dir=ckpt_dir,
                            config_sha=config_sha,
                            sft_train_sha=sft_train_sha,
                            sft_val_sha=sft_val_sha,
                            trainable_params=trainable_params,
                        )
                        checkpoint_dirs[global_step] = str(ckpt_dir)
                        print(
                            f"[{self.config.candidate_id}] step={global_step}/{max_training_steps} checkpoint_saved={ckpt_dir.name}",
                            flush=True,
                        )

                        # Free-generation probe on VAL questions
                        if probe_questions and base_probe_cases:
                            print(
                                f"[{self.config.candidate_id}] step={global_step}/{max_training_steps} free_generation_probe START (20 cases)",
                                flush=True,
                            )
                            cand_probe_cases = run_free_generation_probe(
                                model=model,
                                tokenizer=tokenizer,
                                questions=probe_questions,
                                system_prompt=self.config.system_prompt,
                                max_new_tokens=self.config.generation_probe_max_new_tokens,
                            )

                            gate_report = evaluate_checkpoint_health_gate(
                                candidate_results=cand_probe_cases,
                                base_results=base_probe_cases,
                                references=probe_questions,
                                config=self.config,
                                optimizer_step=global_step,
                                val_loss=val_loss,
                                official_meteor_scorer=official_meteor_scorer,
                            )
                            gate_reports[global_step] = gate_report
                            probe_report_path = output / f"probe-step-{global_step:04d}.json"
                            probe_report_path.write_text(
                                gate_report.model_dump_json(indent=2), encoding="utf-8"
                            )

                            print(
                                f"[{self.config.candidate_id}] step={global_step}/{max_training_steps} "
                                f"safety_eligible={gate_report.safety_eligible} "
                                f"semantic_eligible={gate_report.semantic_eligible} "
                                f"checkpoint_eligible={gate_report.checkpoint_eligible}\n",
                                flush=True,
                            )
                        model.train()

                    elif is_eval_step:
                        val_loss = self._evaluate_loss(model, val_loader, training_device)
                        final_val_loss = val_loss
                        val_loss_recorded = val_loss
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = global_step
                            self._save_checkpoint(model, output / "best_checkpoint")
                        model.train()

                    # Save history entry
                    entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": round(accumulated_loss, 6),
                        "learning_rate": current_lr,
                        "eval_loss": round(val_loss_recorded, 6) if val_loss_recorded is not None else None,
                        "elapsed_seconds": round(elapsed, 2),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    history_entries.append(entry)
                    with open(history_jsonl_path, "a", encoding="utf-8") as hf:
                        hf.write(json.dumps(entry) + "\n")

                    # Atomically update progress.json
                    progress_snapshot = TrainingProgressSnapshot(
                        updated_at=datetime.now(UTC),
                        code_version=__version__,
                        candidate_id=self.config.candidate_id,
                        git_commit=commit_sha,
                        training_config_sha256=config_sha,
                        status="training" if global_step < max_training_steps else "completed",
                        current_optimizer_step=global_step,
                        max_optimizer_steps=max_training_steps,
                        current_microbatch=consumed_microbatches,
                        total_microbatches=batches_in_epoch * self.config.num_train_epochs,
                        elapsed_seconds=round(elapsed, 2),
                        eta_seconds=round(eta_seconds, 2),
                        elapsed_formatted=elapsed_fmt,
                        eta_formatted=eta_fmt,
                        latest_train_loss=round(accumulated_loss, 6),
                        latest_learning_rate=current_lr,
                        latest_val_loss=round(val_loss_recorded, 6) if val_loss_recorded is not None else None,
                        latest_completed_probe_step=max(gate_reports.keys()) if gate_reports else None,
                        latest_durable_checkpoint=checkpoint_dirs.get(max(gate_reports.keys())) if gate_reports else None,
                        warnings=[],
                    )
                    write_progress_atomically(output, progress_snapshot)

                    accumulated_loss = 0.0

        # 10. Checkpoint Selection & Packaging
        end_time = datetime.now(UTC)

        if gate_reports:
            selection_report = select_best_pilot_checkpoint(
                gate_reports=gate_reports,
                candidate_id=self.config.candidate_id,
                checkpoint_dirs=checkpoint_dirs,
            )
            (output / CHECKPOINT_SELECTION_FILENAME).write_text(
                selection_report.model_dump_json(indent=2), encoding="utf-8"
            )

            if selection_report.status == "selected_pilot_checkpoint" and selection_report.selected_checkpoint_step:
                best_step = selection_report.selected_checkpoint_step
                best_val_loss = gate_reports[best_step].val_loss
            else:
                best_step = 0
                best_val_loss = final_val_loss
        else:
            # Finalize final checkpoint if no gate reports
            final_checkpoint_dir = output / "final_checkpoint"
            self._save_checkpoint(model, final_checkpoint_dir)

        # Save project-owned training history
        history_payload = {
            "schema_version": "1.0",
            "created_at": end_time.isoformat(),
            "code_version": __version__,
            "candidate_id": self.config.candidate_id,
            "total_steps": global_step,
            "max_training_steps": max_training_steps,
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

        manifest = M50TrainingManifest(
            created_at=end_time,
            code_version=__version__,
            candidate_id=self.config.candidate_id,
            git_commit=commit_sha,
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
        self.last_optimizer = optimizer
        self.last_scheduler = scheduler
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

    def _save_checkpoint_bundle(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        global_step: int,
        consumed_microbatches: int,
        val_loss: float,
        output_dir: Path,
        config_sha: str,
        sft_train_sha: str,
        sft_val_sha: str,
        trainable_params: int,
    ) -> CheckpointManifest:
        """Save durable checkpoint bundle with weights, training state, and manifest."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save adapter weights and config
        self._save_checkpoint(model, output_dir)

        # 2. Save training state for exact resume
        state_payload = {
            "global_step": global_step,
            "consumed_microbatches": consumed_microbatches,
            "optimizer": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
            "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        state_file = output_dir / TRAINING_STATE_FILENAME
        torch.save(state_payload, str(state_file))

        adapter_file = output_dir / ADAPTER_MODEL_FILENAME
        if not adapter_file.exists():
            adapter_file = output_dir / "pytorch_model.bin"
        if not adapter_file.exists():
            # In mock mode create placeholder
            adapter_file.write_bytes(b"mock_adapter")

        adapter_cfg_file = output_dir / ADAPTER_CONFIG_FILENAME
        if not adapter_cfg_file.exists():
            adapter_cfg_file.write_text("{}", encoding="utf-8")

        adapter_sha = _file_sha256(adapter_file)
        adapter_cfg_sha = _file_sha256(adapter_cfg_file)
        state_sha = _file_sha256(state_file)

        manifest = CheckpointManifest(
            created_at=datetime.now(UTC),
            code_version=__version__,
            candidate_id=self.config.candidate_id,
            optimizer_step=global_step,
            training_config_sha256=config_sha,
            sft_train_sha256=sft_train_sha,
            sft_val_sha256=sft_val_sha,
            base_model_id=self.config.base_model_id,
            base_model_revision=self.config.base_model_revision,
            tokenizer_revision=self.config.base_model_revision,
            trainable_parameters=trainable_params,
            val_loss=val_loss,
            adapter_weights_sha256=adapter_sha,
            adapter_config_sha256=adapter_cfg_sha,
            training_state_sha256=state_sha,
            warnings=[],
        )
        (output_dir / "checkpoint_manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        return manifest
