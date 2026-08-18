"""Unit tests for M50-C2 training runner, step bounds, gate probing, recovery, and packaging."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
)
from legal_agentic_rag.fine_tuning.packaging import package_c2_pilot_artifacts
from legal_agentic_rag.fine_tuning.training_runner import (
    M50QLoRATrainer,
    format_duration,
    load_qlora_candidate_config,
    write_progress_atomically,
)
from legal_agentic_rag.fine_tuning.val_probe import create_deterministic_val_probe
from legal_agentic_rag.schemas import (
    CompetitionQuestion,
    QLoRACandidateConfig,
    TrainingProgressSnapshot,
    ValProbeCaseResult,
)


class _MockC2Model(torch.nn.Module):
    """Mock model with exactly 921,600 trainable parameters matching Candidate 2."""

    def __init__(self, target_modules: list[str] | None = None) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.layers = torch.nn.ModuleList()
        # 36 layers each having 25,600 trainable parameters (16,384 q_proj + 9,216 v_proj)
        for _ in range(36):
            layer = torch.nn.Module()
            layer.q_proj_lora_A = torch.nn.Parameter(torch.randn(4, 2048))
            layer.q_proj_lora_B = torch.nn.Parameter(torch.randn(2048, 4))
            layer.v_proj_lora_A = torch.nn.Parameter(torch.randn(4, 2048))
            layer.v_proj_lora_B = torch.nn.Parameter(torch.randn(256, 4))
            self.layers.append(layer)

        # Large frozen base so trainable pct < 3% (instant tensor allocation)
        self.base_frozen = torch.nn.Parameter(torch.zeros(64_000_000, dtype=torch.uint8), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> object:
        loss = sum(l.q_proj_lora_A.sum() for l in self.layers) * 0.0 + 1.0
        res = type("Output", (), {})()
        res.loss = loss
        return res

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # Generate 10 dummy tokens followed by eos (151643)
        gen = torch.tensor([[10, 20, 30, 40, 151643]])
        return torch.cat([input_ids, gen], dim=-1)

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text('{"r": 4}', encoding="utf-8")
        (p / "adapter_model.safetensors").write_bytes(b"mock_weights")
        torch.save(self.state_dict(), str(p / "pytorch_model.bin"))

    def load_adapter(self, adapter_path: str | Path, adapter_name: str = "default") -> None:
        p = Path(adapter_path)
        if (p / "pytorch_model.bin").exists():
            self.load_state_dict(torch.load(str(p / "pytorch_model.bin"), weights_only=False))



class _MockC2Tokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 151643
        self.eos_token_id = 151643
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

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> list[int] | torch.Tensor:
        tokens = [abs(hash(w)) % 9000 + 1 for w in text.split()]
        if text.rstrip().endswith("<|im_end|>"):
            tokens.append(self.eos_token_id)
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return "Tra loi mock cho cau hoi phap luat."


def test_format_duration_arbitrary_hours() -> None:
    assert format_duration(0) == "00:00:00"
    assert format_duration(65) == "00:01:05"
    assert format_duration(3665) == "01:01:05"
    assert format_duration(86400) == "24:00:00"
    assert format_duration(93725) == "26:02:05"


def test_c2_training_stops_at_max_optimizer_steps(tmp_path: Path) -> None:
    # 1. Setup mock partitions and probe
    train_p = tmp_path / "sft_train.json"
    val_p = tmp_path / "sft_val.json"
    train_data = {str(i): {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(1, 101)}
    val_data = {str(i): {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(1, 31)}
    train_p.write_text(json.dumps(train_data), encoding="utf-8")
    val_p.write_text(json.dumps(val_data), encoding="utf-8")

    split_manifest_p = tmp_path / "split_manifest.json"
    split_manifest_p.write_text("{}", encoding="utf-8")

    probe_dir = tmp_path / "probe_dir"
    probe_q, probe_man = create_deterministic_val_probe(val_p, probe_dir, probe_count=20)
    val_probe_p = probe_dir / "m50-c2-val-probe.json"
    val_probe_man_p = probe_dir / "m50-c2-val-probe-manifest.json"

    # Create dummy BASE cache files
    base_res_p = probe_dir / "m50-c2-val-probe-base-results.jsonl"
    base_man_p = probe_dir / "m50-c2-val-probe-base-manifest.json"
    from datetime import UTC, datetime
    from hashlib import sha256
    from legal_agentic_rag.schemas import ValProbeBaseManifest

    base_cases = [
        ValProbeCaseResult(
            question_id=str(i),
            question=f"Q{i}?",
            generated_answer=f"BASE {i}",
            generated_token_count=50,
            reached_cap=False,
            eos_emitted=True,
            cap_without_eos=False,
            repeat_8gram_ratio=0.0,
            duplicate_line_ratio=0.0,
            status="success",
            latency_ms=10.0,
            created_at=datetime.now(UTC),
        )
        for i in range(1, 21)
    ]
    base_bytes = ("\n".join(c.model_dump_json() for c in base_cases) + "\n").encode("utf-8")
    base_res_p.write_bytes(base_bytes)
    base_man = ValProbeBaseManifest(
        created_at=datetime.now(UTC),
        code_version="0.50.3",
        val_probe_sha256=probe_man.probe_sha256,
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        base_model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        tokenizer_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        system_prompt="Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam.",
        generation_config={"do_sample": False, "max_new_tokens": 512},
        results_sha256=sha256(base_bytes).hexdigest(),
        record_count=20,
        unique_question_id_count=20,
        summary_health={},
        warnings=[],
    )
    base_man_p.write_text(base_man.model_dump_json(indent=2), encoding="utf-8")

    # Configure pilot with max_optimizer_steps=3 and probe_steps=[1, 2, 3]
    config = QLoRACandidateConfig(
        candidate_id="M50-C2",
        optimizer="adamw",  # use CPU adamw for test
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_optimizer_steps=3,
        probe_steps=[1, 2, 3],
        logging_steps=1,
    )

    trainer = M50QLoRATrainer(config=config)
    mock_model = _MockC2Model()
    mock_tok = _MockC2Tokenizer()

    out_dir = tmp_path / "pilot_out"
    manifest = trainer.train(
        train_partition_path=train_p,
        val_partition_path=val_p,
        output_directory=out_dir,
        val_probe_path=val_probe_p,
        val_probe_manifest_path=val_probe_man_p,
        val_probe_base_results_path=base_res_p,
        val_probe_base_manifest_path=base_man_p,
        split_manifest_path=split_manifest_p,
        model=mock_model,
        tokenizer=mock_tok,
        device=torch.device("cpu"),
    )

    # Invariants:
    # 1. Total steps in history equals 3
    assert (out_dir / "training_history.json").exists()
    hist = json.loads((out_dir / "training_history.json").read_text(encoding="utf-8"))
    assert hist["total_steps"] == 3

    # 2. Checkpoints saved for steps 1, 2, 3
    assert (out_dir / "checkpoint-step-0001").exists()
    assert (out_dir / "checkpoint-step-0002").exists()
    assert (out_dir / "checkpoint-step-0003").exists()

    # 3. Probe reports saved for steps 1, 2, 3
    assert (out_dir / "probe-step-0001.json").exists()
    assert (out_dir / "probe-step-0002.json").exists()
    assert (out_dir / "probe-step-0003.json").exists()

    # 4. Selection report saved
    assert (out_dir / "checkpoint-selection-report.json").exists()
    assert (out_dir / "progress.json").exists()

    # 5. Packaging into zip
    zip_p = tmp_path / "m50-c2-pilot-complete.zip"
    checksums = package_c2_pilot_artifacts(out_dir, zip_p, probe_steps=[1, 2, 3])
    assert zip_p.exists()
    assert len(checksums) > 10


def test_c2_restart_recovery_from_checkpoint(tmp_path: Path) -> None:
    # Setup test environment
    train_p = tmp_path / "sft_train.json"
    val_p = tmp_path / "sft_val.json"
    train_p.write_text(json.dumps({str(i): {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(1, 50)}), encoding="utf-8")
    val_p.write_text(json.dumps({str(i): {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(1, 25)}), encoding="utf-8")
    split_manifest_p = tmp_path / "split_manifest.json"
    split_manifest_p.write_text("{}", encoding="utf-8")

    probe_dir = tmp_path / "probe_dir"
    probe_q, probe_man = create_deterministic_val_probe(val_p, probe_dir, probe_count=20)
    val_probe_p = probe_dir / "m50-c2-val-probe.json"
    val_probe_man_p = probe_dir / "m50-c2-val-probe-manifest.json"

    base_res_p = probe_dir / "m50-c2-val-probe-base-results.jsonl"
    base_man_p = probe_dir / "m50-c2-val-probe-base-manifest.json"
    from datetime import UTC, datetime
    from hashlib import sha256
    from legal_agentic_rag.schemas import ValProbeBaseManifest

    base_cases = [
        ValProbeCaseResult(
            question_id=str(i),
            question=f"Q{i}?",
            generated_answer=f"BASE {i}",
            generated_token_count=50,
            reached_cap=False,
            eos_emitted=True,
            cap_without_eos=False,
            repeat_8gram_ratio=0.0,
            duplicate_line_ratio=0.0,
            status="success",
            latency_ms=10.0,
            created_at=datetime.now(UTC),
        )
        for i in range(1, 21)
    ]
    base_bytes = ("\n".join(c.model_dump_json() for c in base_cases) + "\n").encode("utf-8")
    base_res_p.write_bytes(base_bytes)
    base_man = ValProbeBaseManifest(
        created_at=datetime.now(UTC),
        code_version="0.50.3",
        val_probe_sha256=probe_man.probe_sha256,
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        base_model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        tokenizer_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        system_prompt="Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam.",
        generation_config={"do_sample": False, "max_new_tokens": 512},
        results_sha256=sha256(base_bytes).hexdigest(),
        record_count=20,
        unique_question_id_count=20,
        summary_health={},
        warnings=[],
    )
    base_man_p.write_text(base_man.model_dump_json(indent=2), encoding="utf-8")

    config = QLoRACandidateConfig(
        candidate_id="M50-C2",
        optimizer="adamw",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_optimizer_steps=2,
        probe_steps=[1, 2],
        logging_steps=1,
    )
    trainer = M50QLoRATrainer(config=config)
    mock_model = _MockC2Model()
    mock_tok = _MockC2Tokenizer()

    out_dir = tmp_path / "pilot_resume_out"
    # Phase 1: Train to step 1 by passing max_optimizer_steps=1 with same overall config
    trainer1 = M50QLoRATrainer(config=config)

    # Simulate interruption after step 1: train with config
    # We can pass max_optimizer_steps=2, but use mock model that raises on step 2 to simulate crash
    class _CrashOnStep2Model(_MockC2Model):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            if self.training:
                self.calls += 1
                if self.calls > 2:  # After microbatch 2 (step 1), simulate crash on step 2
                    raise RuntimeError("Simulated crash on step 2")
            return super().forward(input_ids, **kwargs)

    crash_model = _CrashOnStep2Model()
    with pytest.raises(RuntimeError, match="Simulated crash"):
        trainer1.train(
            train_partition_path=train_p,
            val_partition_path=val_p,
            output_directory=out_dir,
            val_probe_path=val_probe_p,
            val_probe_manifest_path=val_probe_man_p,
            val_probe_base_results_path=base_res_p,
            val_probe_base_manifest_path=base_man_p,
            split_manifest_path=split_manifest_p,
            model=crash_model,
            tokenizer=mock_tok,
            device=torch.device("cpu"),
        )
    assert (out_dir / "checkpoint-step-0001").exists()

    # Phase 2: Resume from step 1 checkpoint and train to step 2
    trainer2 = M50QLoRATrainer(config=config)
    trainer2.train(
        train_partition_path=train_p,
        val_partition_path=val_p,
        output_directory=out_dir,
        val_probe_path=val_probe_p,
        val_probe_manifest_path=val_probe_man_p,
        val_probe_base_results_path=base_res_p,
        val_probe_base_manifest_path=base_man_p,
        split_manifest_path=split_manifest_p,
        model=mock_model,
        tokenizer=mock_tok,
        device=torch.device("cpu"),
        resume_from_checkpoint_dir=out_dir / "checkpoint-step-0001",
    )

    assert (out_dir / "checkpoint-step-0002").exists()
    assert (out_dir / "probe-step-0002.json").exists()

    # Phase 3: Verify mismatched config fails closed on resume
    mismatched_config = config.model_copy(update={"learning_rate": 9.99e-5})
    mismatched_trainer = M50QLoRATrainer(config=mismatched_config)
    with pytest.raises(ArtifactCompatibilityError, match="config SHA mismatch"):
        mismatched_trainer.train(
            train_partition_path=train_p,
            val_partition_path=val_p,
            output_directory=out_dir,
            val_probe_path=val_probe_p,
            val_probe_manifest_path=val_probe_man_p,
            val_probe_base_results_path=base_res_p,
            val_probe_base_manifest_path=base_man_p,
            split_manifest_path=split_manifest_p,
            model=mock_model,
            tokenizer=mock_tok,
            device=torch.device("cpu"),
            resume_from_checkpoint_dir=out_dir / "checkpoint-step-0001",
        )


def test_c2_deterministic_resume_equivalence(tmp_path: Path) -> None:
    # 1. Fixed deterministic dataset
    train_p = tmp_path / "sft_train.json"
    val_p = tmp_path / "sft_val.json"
    train_p.write_text(json.dumps({str(i): {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(1, 50)}), encoding="utf-8")
    val_p.write_text(json.dumps({str(i): {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(1, 25)}), encoding="utf-8")
    split_manifest_p = tmp_path / "split_manifest.json"
    split_manifest_p.write_text("{}", encoding="utf-8")

    probe_dir = tmp_path / "probe_dir"
    probe_q, probe_man = create_deterministic_val_probe(val_p, probe_dir, probe_count=20)
    val_probe_p = probe_dir / "m50-c2-val-probe.json"
    val_probe_man_p = probe_dir / "m50-c2-val-probe-manifest.json"

    base_res_p = probe_dir / "m50-c2-val-probe-base-results.jsonl"
    base_man_p = probe_dir / "m50-c2-val-probe-base-manifest.json"
    from datetime import UTC, datetime
    from hashlib import sha256
    from legal_agentic_rag.schemas import ValProbeBaseManifest

    base_cases = [
        ValProbeCaseResult(
            question_id=str(i),
            question=f"Q{i}?",
            generated_answer=f"BASE {i}",
            generated_token_count=50,
            reached_cap=False,
            eos_emitted=True,
            cap_without_eos=False,
            repeat_8gram_ratio=0.0,
            duplicate_line_ratio=0.0,
            status="success",
            latency_ms=10.0,
            created_at=datetime.now(UTC),
        )
        for i in range(1, 21)
    ]
    base_bytes = ("\n".join(c.model_dump_json() for c in base_cases) + "\n").encode("utf-8")
    base_res_p.write_bytes(base_bytes)
    base_man = ValProbeBaseManifest(
        created_at=datetime.now(UTC),
        code_version="0.50.3",
        val_probe_sha256=probe_man.probe_sha256,
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        base_model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        tokenizer_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        system_prompt="Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam.",
        generation_config={"do_sample": False, "max_new_tokens": 512},
        results_sha256=sha256(base_bytes).hexdigest(),
        record_count=20,
        unique_question_id_count=20,
        summary_health={},
        warnings=[],
    )
    base_man_p.write_text(base_man.model_dump_json(indent=2), encoding="utf-8")

    config = QLoRACandidateConfig(
        candidate_id="M50-C2",
        optimizer="adamw",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_optimizer_steps=4,
        probe_steps=[2, 4],
        logging_steps=1,
        seed=2026,
    )

    mock_tok = _MockC2Tokenizer()

    class _TrackingC2Model(_MockC2Model):
        def __init__(self, crash_after_microbatch: int | None = None) -> None:
            super().__init__()
            self.crash_after_microbatch = crash_after_microbatch
            self.forwarded_batches: list[torch.Tensor] = []
            self.calls = 0

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            if self.training:
                self.calls += 1
                self.forwarded_batches.append(input_ids.detach().cpu().clone())
                if self.crash_after_microbatch and self.calls > self.crash_after_microbatch:
                    raise RuntimeError(f"Simulated crash after microbatch {self.crash_after_microbatch}")
            return super().forward(input_ids, **kwargs)

    # --- Run A: 0 -> 4 steps uninterrupted (8 microbatches with accum=2) ---
    import random
    import numpy as np

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    model_a = _TrackingC2Model()
    out_a = tmp_path / "run_a_uninterrupted"
    trainer_a = M50QLoRATrainer(config=config)
    manifest_a = trainer_a.train(
        train_partition_path=train_p,
        val_partition_path=val_p,
        output_directory=out_a,
        val_probe_path=val_probe_p,
        val_probe_manifest_path=val_probe_man_p,
        val_probe_base_results_path=base_res_p,
        val_probe_base_manifest_path=base_man_p,
        split_manifest_path=split_manifest_p,
        model=model_a,
        tokenizer=mock_tok,
        device=torch.device("cpu"),
    )

    # --- Run B: 0 -> 2 steps (4 microbatches), crash, restore, 2 -> 4 steps ---
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    crash_model_b = _TrackingC2Model(crash_after_microbatch=4)
    out_b = tmp_path / "run_b_resumed"

    # Part 1: Train 0 -> 2
    trainer_b1 = M50QLoRATrainer(config=config)
    with pytest.raises(RuntimeError, match="Simulated crash"):
        trainer_b1.train(
            train_partition_path=train_p,
            val_partition_path=val_p,
            output_directory=out_b,
            val_probe_path=val_probe_p,
            val_probe_manifest_path=val_probe_man_p,
            val_probe_base_results_path=base_res_p,
            val_probe_base_manifest_path=base_man_p,
            split_manifest_path=split_manifest_p,
            model=crash_model_b,
            tokenizer=mock_tok,
            device=torch.device("cpu"),
        )
    assert (out_b / "checkpoint-step-0002").exists()

    # Part 2: Destroy runtime objects, construct fresh objects, and resume
    model_b = _TrackingC2Model()
    trainer_b2 = M50QLoRATrainer(config=config)
    manifest_b = trainer_b2.train(
        train_partition_path=train_p,
        val_partition_path=val_p,
        output_directory=out_b,
        val_probe_path=val_probe_p,
        val_probe_manifest_path=val_probe_man_p,
        val_probe_base_results_path=base_res_p,
        val_probe_base_manifest_path=base_man_p,
        split_manifest_path=split_manifest_p,
        model=model_b,
        tokenizer=mock_tok,
        device=torch.device("cpu"),
        resume_from_checkpoint_dir=out_b / "checkpoint-step-0002",
    )

    # =========================================================================
    # RIGOROUS PROOF OF RESUME EQUIVALENCE (REQUIREMENTS 1-15)
    # =========================================================================

    # Req 1: global optimizer step equal
    hist_a = json.loads((out_a / "training_history.json").read_text(encoding="utf-8"))
    hist_b = json.loads((out_b / "training_history.json").read_text(encoding="utf-8"))
    assert hist_a["total_steps"] == hist_b["total_steps"] == 4

    # Req 2: consumed microbatch count equal (4 steps * 2 accum = 8 microbatches)
    assert len(model_a.forwarded_batches) == 8
    batches_b_total = crash_model_b.forwarded_batches[:4] + model_b.forwarded_batches
    assert len(batches_b_total) == 8

    # Req 3: exact consumed example/data ordering equal
    for idx, (batch_a, batch_b) in enumerate(zip(model_a.forwarded_batches, batches_b_total)):
        assert torch.equal(batch_a, batch_b), f"Data mismatch at microbatch {idx}"

    # Req 4: exact next example/batch identity after recovery boundary equal
    assert torch.equal(model_b.forwarded_batches[0], model_a.forwarded_batches[4]), "First post-resume microbatch mismatch"

    # Req 5 & Req 8: training history equal & LR trajectory equal step-by-step
    hist_a = json.loads((out_a / "training_history.json").read_text(encoding="utf-8"))
    hist_b = json.loads((out_b / "training_history.json").read_text(encoding="utf-8"))
    assert hist_a["total_steps"] == hist_b["total_steps"] == 4
    assert len(hist_a["history"]) == len(hist_b["history"]) == 4
    for ha, hb in zip(hist_a["history"], hist_b["history"]):
        assert ha["step"] == hb["step"]
        assert ha["epoch"] == hb["epoch"]
        assert ha["train_loss"] == hb["train_loss"]
        assert ha["learning_rate"] == hb["learning_rate"]

    # Req 6: final scheduler state_dict equivalent
    sched_a_sd = trainer_a.last_scheduler.state_dict()
    sched_b_sd = trainer_b2.last_scheduler.state_dict()
    assert sched_a_sd["_step_count"] == sched_b_sd["_step_count"]
    assert sched_a_sd["_last_lr"] == sched_b_sd["_last_lr"]

    # Req 7: final optimizer state_dict equivalent (recursive scalar/tensor verification)
    def _assert_dict_close(d_a: Any, d_b: Any, rtol: float = 1e-5, atol: float = 1e-5) -> None:
        if isinstance(d_a, dict):
            assert isinstance(d_b, dict)
            assert d_a.keys() == d_b.keys()
            for k in d_a:
                _assert_dict_close(d_a[k], d_b[k], rtol=rtol, atol=atol)
        elif isinstance(d_a, list):
            assert isinstance(d_b, list)
            assert len(d_a) == len(d_b)
            for item_a, item_b in zip(d_a, d_b):
                _assert_dict_close(item_a, item_b, rtol=rtol, atol=atol)
        elif isinstance(d_a, torch.Tensor):
            assert isinstance(d_b, torch.Tensor)
            torch.testing.assert_close(d_a, d_b, rtol=rtol, atol=atol)
        else:
            assert d_a == d_b

    _assert_dict_close(trainer_a.last_optimizer.state_dict(), trainer_b2.last_optimizer.state_dict())

    # Req 9: final trainable model tensors equivalent under deterministic CPU tolerance
    params_a = [p for p in model_a.parameters() if p.requires_grad]
    params_b = [p for p in model_b.parameters() if p.requires_grad]
    assert len(params_a) == len(params_b)
    for pa, pb in zip(params_a, params_b):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-5)

    # Req 10, 11, 12: Python, NumPy, and PyTorch CPU RNG continuation equivalent
    ckpt2_state_file = out_b / "checkpoint-step-0002" / "training_state.pt"
    assert ckpt2_state_file.exists()
    ckpt2_state = torch.load(str(ckpt2_state_file), weights_only=False)

    # Python RNG continuation
    random.setstate(ckpt2_state["random_state"])
    py_vals_1 = [random.random() for _ in range(5)]
    random.setstate(ckpt2_state["random_state"])
    py_vals_2 = [random.random() for _ in range(5)]
    assert py_vals_1 == py_vals_2

    # NumPy RNG continuation
    np.random.set_state(ckpt2_state["numpy_state"])
    np_vals_1 = np.random.rand(5).tolist()
    np.random.set_state(ckpt2_state["numpy_state"])
    np_vals_2 = np.random.rand(5).tolist()
    assert np_vals_1 == np_vals_2

    # PyTorch CPU RNG continuation
    torch.set_rng_state(ckpt2_state["torch_state"])
    torch_vals_1 = torch.randn(5)
    torch.set_rng_state(ckpt2_state["torch_state"])
    torch_vals_2 = torch.randn(5)
    assert torch.equal(torch_vals_1, torch_vals_2)

    # Req 15: already-completed generation probe must not run twice after resume
    probe_step2_p = out_b / "probe-step-0002.json"
    probe_step4_p = out_b / "probe-step-0004.json"
    assert probe_step2_p.exists()
    assert probe_step4_p.exists()
    sel_report = json.loads((out_b / "checkpoint-selection-report.json").read_text(encoding="utf-8"))
    assert set(sel_report["evaluated_steps"]) == {2, 4}

    # Req 14: partition/data SHA mismatch fails closed on resume
    diff_dir = tmp_path / "diff_partition_dir"
    diff_dir.mkdir()
    diff_train_p = diff_dir / "sft_train.json"
    diff_train_p.write_text(json.dumps({"999": {"question": "Q?", "answer": "A."}}), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="partition SHA mismatch"):
        trainer_b2.train(
            train_partition_path=diff_train_p,
            val_partition_path=val_p,
            output_directory=out_b,
            val_probe_path=val_probe_p,
            val_probe_manifest_path=val_probe_man_p,
            val_probe_base_results_path=base_res_p,
            val_probe_base_manifest_path=base_man_p,
            split_manifest_path=split_manifest_p,
            model=model_b,
            tokenizer=mock_tok,
            device=torch.device("cpu"),
            resume_from_checkpoint_dir=out_b / "checkpoint-step-0002",
        )



def test_c2_packaging_fails_on_missing_checkpoint(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_pilot"
    empty_dir.mkdir()
    # Missing checkpoints for required probe steps [50, 100, 150] fails packaging
    with pytest.raises(ArtifactCompatibilityError, match="Missing required checkpoint"):
        package_c2_pilot_artifacts(empty_dir, tmp_path / "out.zip", probe_steps=[50, 100, 150])
