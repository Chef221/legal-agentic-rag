"""Fine-tuning domain infrastructure for UIT DSC 2026 Task 2."""

from legal_agentic_rag.fine_tuning.collator import SFTDynamicDataCollator
from legal_agentic_rag.fine_tuning.dataset import (
    DEFAULT_MAX_SEQ_LENGTH,
    SYSTEM_PROMPT,
    SFTAnswerOnlyDataset,
    encode_sft_example,
)
from legal_agentic_rag.fine_tuning.paired_metrics import (
    DirectQAPairedScorer,
    compute_paired_bootstrap_ci,
)
from legal_agentic_rag.fine_tuning.screening import (
    BASE_CACHE_JSONL_FILENAME,
    BASE_CACHE_MANIFEST_FILENAME,
    DEFAULT_DIRECT_QA_MAX_NEW_TOKENS,
    SCREEN_TOKEN_AUDIT_FILENAME,
    DirectQAScreeningRunner,
    create_and_save_screen_token_audit,
    load_and_validate_base_direct_qa_cache,
    load_and_validate_screen_token_audit,
    load_cached_direct_qa_results,
    save_base_direct_qa_cache,
)
from legal_agentic_rag.fine_tuning.splitting import (
    M50_SPLIT_MANIFEST_FILENAME,
    SCREEN_HOLDOUT_FILENAME,
    SFT_TRAIN_FILENAME,
    SFT_VAL_FILENAME,
    M50FineTuningSplitter,
    find_overlength_question_ids,
)
from legal_agentic_rag.fine_tuning.training_runner import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_MODEL_FILENAME,
    EXPECTED_M50_C1_TRAINABLE_PARAMS,
    TRAINING_HISTORY_FILENAME,
    TRAINING_MANIFEST_FILENAME,
    M50QLoRATrainer,
    get_environment_dependency_versions,
    load_qlora_candidate_config,
    verify_trainable_parameters,
)

__all__ = [
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_DIRECT_QA_MAX_NEW_TOKENS",
    "SCREEN_TOKEN_AUDIT_FILENAME",
    "EXPECTED_M50_C1_TRAINABLE_PARAMS",
    "TRAINING_HISTORY_FILENAME",
    "SYSTEM_PROMPT",
    "M50_SPLIT_MANIFEST_FILENAME",
    "SFT_TRAIN_FILENAME",
    "SFT_VAL_FILENAME",
    "SCREEN_HOLDOUT_FILENAME",
    "BASE_CACHE_JSONL_FILENAME",
    "BASE_CACHE_MANIFEST_FILENAME",
    "TRAINING_MANIFEST_FILENAME",
    "ADAPTER_MODEL_FILENAME",
    "ADAPTER_CONFIG_FILENAME",
    "SFTDynamicDataCollator",
    "SFTAnswerOnlyDataset",
    "encode_sft_example",
    "DirectQAScreeningRunner",
    "load_cached_direct_qa_results",
    "save_base_direct_qa_cache",
    "load_and_validate_base_direct_qa_cache",
    "create_and_save_screen_token_audit",
    "load_and_validate_screen_token_audit",
    "DirectQAPairedScorer",
    "compute_paired_bootstrap_ci",
    "M50FineTuningSplitter",
    "find_overlength_question_ids",
    "M50QLoRATrainer",
    "load_qlora_candidate_config",
    "get_environment_dependency_versions",
    "verify_trainable_parameters",
]
