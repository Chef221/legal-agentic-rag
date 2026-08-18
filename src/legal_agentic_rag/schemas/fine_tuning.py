"""Schemas for M50 fine-tuning data preparation, QLoRA training, and direct screening."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from legal_agentic_rag.schemas.competition_execution import (
    CompetitionSplitPartition,
    CompetitionSplitSource,
    _validate_sha256,
    _validate_timestamp,
)


class M50SplitManifest(BaseModel):
    """Reproducible proof of an M50 three-way clean training split."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    clean_training_source: CompetitionSplitSource
    seed: int = 2026
    near_duplicate_threshold: float = Field(ge=0.5, le=1.0)
    exact_duplicate_pair_count: int = Field(ge=0)
    near_duplicate_pair_count: int = Field(ge=0)
    val_target: int = Field(gt=0)
    screen_target: int = Field(gt=0)
    partitions: list[CompetitionSplitPartition] = Field(min_length=3, max_length=3)
    overlength_question_ids_at_1536: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)

    @model_validator(mode="after")
    def validate_partitions(self) -> "M50SplitManifest":
        names = [partition.filename for partition in self.partitions]
        if len(names) != len(set(names)):
            raise ValueError("M50 split partition filenames must be unique")
        identities = [
            identity
            for partition in self.partitions
            for identity in partition.question_ids
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("question IDs cannot cross M50 split partitions")
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("M50 split warnings must be unique")
        return self


class ScreenTokenAuditReport(BaseModel):
    """Audited token length distribution and deterministic generation cap for screening."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    screen_holdout_sha256: str
    tokenizer_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8"
    system_prompt: str
    question_count: int = Field(gt=0)
    min_tokens: int = Field(ge=0)
    p50_tokens: int = Field(ge=0)
    p75_tokens: int = Field(ge=0)
    p90_tokens: int = Field(ge=0)
    p95_tokens: int = Field(ge=0)
    p99_tokens: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    counts_exceeding: dict[str, int]
    selected_max_new_tokens: int = Field(ge=256)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class DirectQACaseResult(BaseModel):
    """Output from direct question-to-answer generation without RAG."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    generated_answer: str
    generated_token_count: int = Field(ge=0)
    hit_max_tokens: bool = False
    status: Literal["success", "error"] = "success"
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0, default=0.0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class DirectQAPairedCaseScore(BaseModel):
    """Per-question paired comparison between BASE and TREATMENT direct generation."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    base_meteor: float = Field(ge=0.0, le=1.0)
    treatment_meteor: float = Field(ge=0.0, le=1.0)
    delta_meteor: float
    base_rouge_l: float = Field(ge=0.0, le=1.0)
    treatment_rouge_l: float = Field(ge=0.0, le=1.0)
    delta_rouge_l: float
    base_answer_length: int = Field(ge=0)
    treatment_answer_length: int = Field(ge=0)
    reference_answer_length: int = Field(ge=0)


class PairedBootstrapInterval(BaseModel):
    """Bootstrap confidence interval for a paired difference."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1)
    mean_delta: float
    median_delta: float
    ci_lower_95: float
    ci_upper_95: float
    resamples: int = Field(ge=100, default=1000)
    seed: int = 2026


class PairedMetricSummary(BaseModel):
    """Summary of paired comparison for a single evaluation metric."""

    model_config = ConfigDict(extra="forbid")

    base_mean: float = Field(ge=0.0, le=1.0)
    treatment_mean: float = Field(ge=0.0, le=1.0)
    mean_delta: float
    median_delta: float
    win_count: int = Field(ge=0)
    tie_count: int = Field(ge=0)
    loss_count: int = Field(ge=0)
    bootstrap_ci_95: PairedBootstrapInterval


class DirectQAPairedComparisonReport(BaseModel):
    """Durable report comparing direct QA screening performance between BASE and TREATMENT."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    treatment_model_id: str = Field(min_length=1)
    treatment_model_revision: str = Field(min_length=1)
    question_count: int = Field(gt=0)
    screen_holdout_sha256: str | None = None
    base_results_sha256: str | None = None
    treatment_results_sha256: str | None = None
    training_manifest_sha256: str | None = None
    adapter_config_sha256: str | None = None
    adapter_weights_sha256: str | None = None
    best_checkpoint_step: int | None = None
    base_hit_max_tokens_count: int = 0
    treatment_hit_max_tokens_count: int = 0
    meteor: PairedMetricSummary
    rouge_l: PairedMetricSummary
    length_summary: dict[str, float]
    cases: list[DirectQAPairedCaseScore] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class DirectQABaseCacheManifest(BaseModel):
    """Identity and provenance proof for cached BASE direct QA screening predictions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    screen_holdout_sha256: str
    base_model_id: str
    base_model_revision: str
    tokenizer_revision: str
    system_prompt: str
    generation_config: dict[str, Any]
    results_sha256: str
    record_count: int = Field(ge=1)
    unique_question_id_count: int = Field(ge=1)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class ValProbeManifest(BaseModel):
    """Identity and deterministic selection manifest for the 20-question VAL generation probe."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    source_val_sha256: str
    selection_algorithm: str = "sha256_salted_hash_v1"
    salt: str = "m50-c2-val-probe-v1"
    question_count: int = Field(ge=1, default=20)
    selected_question_ids: list[str] = Field(min_length=1)
    probe_sha256: str
    tokenizer_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8"
    system_prompt: str
    diagnostic_generation_config: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class ValProbeCaseResult(BaseModel):
    """Output and generation health diagnostics for one probe question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    generated_answer: str
    generated_token_count: int = Field(ge=0)
    reached_cap: bool = False
    eos_emitted: bool = True
    cap_without_eos: bool = False
    repeat_8gram_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    duplicate_line_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    status: Literal["success", "error"] = "success"
    latency_ms: float = Field(ge=0.0, default=0.0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class ValProbeBaseManifest(BaseModel):
    """Identity and provenance proof for cached BASE direct QA predictions on VAL probe."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    val_probe_sha256: str
    base_model_id: str
    base_model_revision: str
    tokenizer_revision: str
    system_prompt: str
    generation_config: dict[str, Any]
    results_sha256: str
    record_count: int = Field(ge=1, default=20)
    unique_question_id_count: int = Field(ge=1, default=20)
    summary_health: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class CheckpointGateReport(BaseModel):
    """Comprehensive generation health and semantic gate evaluation for a candidate checkpoint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    optimizer_step: int = Field(ge=1)
    val_loss: float
    base_probe_sha256: str
    candidate_probe_sha256: str

    # Raw Candidate metrics
    candidate_eos_emitted_count: int
    candidate_reached_cap_count: int
    candidate_cap_without_eos_count: int
    candidate_cap_without_eos_rate: float
    candidate_repeat8_high_count: int
    candidate_duplicate_line_high_count: int
    candidate_mean_generated_token_count: float
    candidate_median_generated_token_count: float
    candidate_generation_error_count: int

    # BASE metrics for comparison
    base_eos_emitted_count: int
    base_reached_cap_count: int
    base_cap_without_eos_count: int
    base_repeat8_high_count: int
    base_duplicate_line_high_count: int
    base_mean_generated_token_count: float
    base_median_generated_token_count: float

    # Paired semantic deltas
    mean_rouge_l_delta: float
    median_rouge_l_delta: float
    mean_meteor_delta: float | None = None
    median_meteor_delta: float | None = None
    meteor_available: bool = True
    combined_semantic_delta: float

    # Gate eligibility
    safety_eligible: bool
    safety_failure_reasons: list[str] = Field(default_factory=list)
    semantic_eligible: bool
    semantic_failure_reasons: list[str] = Field(default_factory=list)
    checkpoint_eligible: bool

    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class CheckpointSelectionReport(BaseModel):
    """Final selection report ranking eligible candidate checkpoints."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    status: Literal["selected_pilot_checkpoint", "no_promotable_checkpoint"]
    selected_checkpoint_step: int | None = None
    selected_checkpoint_dir: str | None = None
    evaluated_steps: list[int]
    eligible_steps: list[int]
    ranked_steps: list[int]
    ranking_explanation: list[str]
    gate_reports: dict[str, CheckpointGateReport]
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class CheckpointManifest(BaseModel):
    """Reproducible proof and provenance for an intermediate checkpoint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    optimizer_step: int = Field(ge=1)
    training_config_sha256: str
    sft_train_sha256: str
    sft_val_sha256: str
    base_model_id: str
    base_model_revision: str
    tokenizer_revision: str
    trainable_parameters: int
    val_loss: float
    adapter_weights_sha256: str
    adapter_config_sha256: str
    training_state_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class TrainingProgressSnapshot(BaseModel):
    """Atomic real-time progress record of fine-tuning pilot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    updated_at: datetime
    code_version: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    git_commit: str | None = None
    training_config_sha256: str
    status: Literal[
        "initialized",
        "training",
        "validating",
        "checkpointing",
        "probing",
        "completed",
        "failed",
        "no_promotable_checkpoint",
    ]
    current_optimizer_step: int = Field(ge=0)
    max_optimizer_steps: int = Field(ge=1)
    current_microbatch: int = Field(ge=0)
    total_microbatches: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0.0)
    eta_seconds: float = Field(ge=0.0)
    elapsed_formatted: str
    eta_formatted: str
    latest_train_loss: float | None = None
    latest_learning_rate: float | None = None
    latest_val_loss: float | None = None
    latest_completed_probe_step: int | None = None
    latest_durable_checkpoint: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("updated_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)


class QLoRACandidateConfig(BaseModel):
    """Pinned specification for one QLoRA fine-tuning experiment."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = "M50-C1"
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    base_model_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8"
    quantization_type: Literal["4bit_nf4"] = "4bit_nf4"
    double_quantization: bool = True
    compute_dtype: Literal["float16", "bfloat16"] = "float16"
    lora_r: int = Field(ge=1, default=8)
    lora_alpha: int = Field(ge=1, default=16)
    lora_dropout: float = Field(ge=0.0, le=0.5, default=0.05)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    max_seq_length: int = Field(ge=256, le=4096, default=1536)
    system_prompt: str = "Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam."
    learning_rate: float = Field(gt=0.0, default=5e-5)
    lr_scheduler_type: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(ge=0.0, le=0.5, default=0.05)
    num_train_epochs: int = Field(ge=1, default=1)
    per_device_train_batch_size: int = Field(ge=1, default=2)
    gradient_accumulation_steps: int = Field(ge=1, default=8)
    gradient_checkpointing: bool = True
    use_cache: Literal[False] = False
    optimizer: str = "paged_adamw_8bit"
    logging_steps: int = 10
    eval_steps: int = 75
    save_steps: int = 75
    seed: int = 2026
    training_partition: str = "sft_train.json"
    validation_partition: str = "sft_val.json"
    screening_partition: str = "screen_holdout.json"

    # M50-C2 Pilot extensions
    max_optimizer_steps: int | None = Field(default=None, ge=1)
    probe_steps: list[int] = Field(default_factory=list)
    generation_probe_question_count: int = Field(ge=1, default=20)
    generation_probe_max_new_tokens: int = Field(ge=64, default=512)
    repetition_ngram_size: int = Field(ge=2, default=8)
    repetition_high_threshold: float = Field(ge=0.0, le=1.0, default=0.25)
    duplicate_line_high_threshold: float = Field(ge=0.0, le=1.0, default=0.25)
    health_count_slack: int = Field(ge=0, default=1)
    max_mean_length_ratio: float = Field(gt=1.0, default=1.35)
    semantic_regression_tolerance: float = Field(default=-0.01)

    @model_validator(mode="after")
    def validate_candidate_config(self) -> "QLoRACandidateConfig":
        if self.probe_steps:
            if sorted(self.probe_steps) != self.probe_steps or len(self.probe_steps) != len(set(self.probe_steps)):
                raise ValueError("probe_steps must be strictly sorted in ascending order and unique")
            if any(s <= 0 for s in self.probe_steps):
                raise ValueError("all probe_steps must be positive integers")
            if self.max_optimizer_steps is not None and self.probe_steps[-1] > self.max_optimizer_steps:
                raise ValueError(
                    f"Final probe step ({self.probe_steps[-1]}) cannot exceed max_optimizer_steps ({self.max_optimizer_steps})"
                )
        return self


class M50TrainingManifest(BaseModel):
    """Reproducible proof of one completed M50 QLoRA training run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    code_version: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    git_commit: str = Field(min_length=7)
    source_split_manifest_sha256: str
    sft_train_sha256: str
    sft_val_sha256: str
    base_model_id: str
    base_model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    training_config_sha256: str
    lora_config: dict[str, Any]
    dependency_versions: dict[str, str]
    cuda_version: str | None = None
    gpu_name: str | None = None
    seed: int
    max_seq_length: int
    total_parameters: int
    trainable_parameters: int
    trainable_percentage: float
    training_start_time: datetime
    training_end_time: datetime
    best_checkpoint_step: int
    best_validation_loss: float
    final_validation_loss: float
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at", "training_start_time", "training_end_time")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_timestamp(value)
