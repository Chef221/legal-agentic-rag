"""Tests for frozen M54 production runtime configuration."""

from legal_agentic_rag.configuration import (
    EmbeddingConfig,
    M54_PARAMETER_LIMIT,
    M54_PRODUCTION_SCHEMA_VERSION,
    M54_TOTAL_ACTIVE_MODEL_PARAMETERS,
    OnlineConfig,
    build_m54_embedding_config,
    build_m54_online_config,
)
from legal_agentic_rag.schemas.retrieval import RetrievalStrategy


def test_build_m54_embedding_config_returns_valid_config() -> None:
    """Builder produces a fully validated EmbeddingConfig instance with exact identities."""
    cfg = build_m54_embedding_config()
    assert isinstance(cfg, EmbeddingConfig)
    assert cfg.model_name == "AITeamVN/Vietnamese_Embedding"
    assert cfg.model_revision == "dea33aa1ab339f38d66ae0a40e6c40e0a9249568"
    assert cfg.expected_dimension == 1024
    assert cfg.max_sequence_length == 2048
    assert cfg.device == "cuda:0"
    assert cfg.torch_dtype == "float32"
    assert cfg.normalize_embeddings is True
    assert cfg.document_prefix == ""
    assert cfg.query_prefix == ""
    assert cfg.query_prompt_name is None
    assert cfg.query_instruction is None
    assert cfg.local_files_only is False


def test_build_m54_online_config_returns_valid_config() -> None:
    """Builder produces a fully validated OnlineConfig instance."""
    cfg = build_m54_online_config()
    assert isinstance(cfg, OnlineConfig)


def test_m54_retrieval_and_evidence_selection_policy_is_exact() -> None:
    """Retrieval parameters, RRF constant, and evidence selection match accepted M54 freeze."""
    cfg = build_m54_online_config()

    # Retrieval policy
    assert cfg.retrieval.default_strategy == RetrievalStrategy.HYBRID_RERANK
    assert cfg.retrieval.candidate_k == 40
    assert cfg.retrieval.top_k == 10
    assert cfg.retrieval.rrf_constant == 60

    # Evidence selection policy
    assert cfg.evidence_selection.enabled is True
    assert cfg.evidence_selection.reference_match_boost == 3.0
    assert cfg.evidence_selection.lexical_overlap_weight == 1.0
    assert cfg.evidence_selection.inactive_penalty == 2.0
    assert cfg.evidence_selection.max_per_document == 3
    assert cfg.evidence_selection.max_per_article == 2

    # Query understanding disabled
    assert cfg.query_understanding.enabled is False
    assert cfg.query_understanding.multi_query_enabled is False

    # Context grading
    assert cfg.context_grading.minimum_evidence_count == 1
    assert cfg.context_grading.require_document_number is False
    assert cfg.context_grading.require_article_number is False


def test_m54_reranker_exact_base_identities_and_no_o2_projector() -> None:
    """Reranker is pinned to Jina 3.5 base with all O2 projector fields set to None."""
    cfg = build_m54_online_config()
    reranker = cfg.reranker

    assert reranker.backend == "jina_native_listwise"
    assert reranker.model_name == "jinaai/jina-reranker-v3.5"
    assert reranker.model_revision == "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
    assert reranker.device == "cuda:0"
    assert reranker.torch_dtype == "float16"
    assert reranker.native_context_cap == 12288
    assert reranker.expected_parameter_count == 596836352
    assert reranker.input_mode == "legal_context"
    assert reranker.local_files_only is False
    assert reranker.max_candidates == 40

    # Strict O2 Projector absence
    assert reranker.projector_checkpoint_path is None
    assert reranker.projector_checkpoint_sha256 is None
    assert reranker.expected_projector_state_sha256 is None
    assert reranker.expected_projector_parameter_count is None


def test_m54_generator_stock_identities_and_compact_policy() -> None:
    """Candidate generator is pinned to stock Qwen3.5-2B with compact schema mode."""
    cfg = build_m54_online_config()
    gen = cfg.generation

    assert gen.backend == "transformers"
    assert gen.model_name == "Qwen/Qwen3.5-2B"
    assert gen.model_revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert gen.model_loader == "image_text_to_text"
    assert gen.device == "cuda:1"
    assert gen.torch_dtype == "float16"
    assert gen.local_files_only is False

    # Generation hyperparameters
    assert gen.max_context_tokens == 6144
    assert gen.max_evidence == 10
    assert gen.max_input_tokens == 8192
    assert gen.temperature == 0.0
    assert gen.max_output_tokens == 1536
    assert gen.repetition_penalty == 1.08
    assert gen.no_repeat_ngram_size == 8

    # Format & styles
    assert gen.prompt_schema_mode == "compact_example"
    assert gen.answer_style == "competition_reference"

    # Retry and fallback policy
    assert gen.max_structured_output_retries == 1
    assert gen.max_model_error_retries == 1
    assert gen.model_failure_policy == "top_evidence"
    assert gen.max_grounding_repair_retries == 1
    assert gen.grounding_failure_policy == "supported_claims_or_top_evidence"
    assert gen.extractive_fallback_max_evidence == 3
    assert gen.salvage_rendering == "standalone"


def test_m54_verification_policy_and_disabled_semantic_verifier() -> None:
    """Claim verification is deterministic with safety gates, semantic verifier disabled."""
    cfg = build_m54_online_config()

    # Claim verifier
    cv = cfg.claim_verification
    assert cv.enabled is True
    assert cv.require_inline_citations is False
    assert cv.minimum_lexical_support == 0.2
    assert cv.minimum_claim_tokens == 2
    assert cv.require_numeric_match is True
    assert cv.require_negation_match is True
    assert cv.max_claims == 60

    # Semantic verifier
    assert cfg.semantic_verification.backend == "disabled"


def test_m54_parameter_authority_constants() -> None:
    """Parameter constants satisfy strict competition limit inequality."""
    assert M54_PRODUCTION_SCHEMA_VERSION == "m54_production_v1"
    assert M54_TOTAL_ACTIVE_MODEL_PARAMETERS == 3_377_832_768
    assert M54_PARAMETER_LIMIT == 4_000_000_000
    assert M54_TOTAL_ACTIVE_MODEL_PARAMETERS < M54_PARAMETER_LIMIT


def test_m54_builders_produce_independent_equivalent_objects() -> None:
    """Repeated calls produce independent objects with equivalent values."""
    emb1 = build_m54_embedding_config()
    emb2 = build_m54_embedding_config()
    assert emb1 is not emb2
    assert emb1 == emb2

    onl1 = build_m54_online_config()
    onl2 = build_m54_online_config()
    assert onl1 is not onl2
    assert onl1 == onl2


def test_no_protected_split_or_kaggle_paths_in_production_config() -> None:
    """Configuration contains no hardcoded filesystem/Kaggle or protected split paths."""
    emb_json = build_m54_embedding_config().model_dump_json()
    onl_json = build_m54_online_config().model_dump_json()
    combined = emb_json + onl_json

    forbidden = ["/kaggle/", "C:\\", "dev.json", "test.json", "public.json", "holdout.json"]
    for word in forbidden:
        assert word not in combined
