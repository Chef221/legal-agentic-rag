"""Frozen production runtime configuration for M54."""

from __future__ import annotations

from legal_agentic_rag.configuration.offline import EmbeddingConfig
from legal_agentic_rag.configuration.online import (
    AgentConfig,
    BM25RuntimeConfig,
    ClaimVerificationConfig,
    ContextGradingConfig,
    EvidenceSelectionConfig,
    GenerationConfig,
    OnlineConfig,
    QueryUnderstandingConfig,
    RerankerConfig,
    RetrievalConfig,
    SemanticVerificationConfig,
    StartupValidationConfig,
    VectorRuntimeConfig,
)
from legal_agentic_rag.schemas.retrieval import RetrievalStrategy

M54_PRODUCTION_SCHEMA_VERSION = "m54_production_v1"
M54_TOTAL_ACTIVE_MODEL_PARAMETERS = 3_377_832_768
M54_PARAMETER_LIMIT = 4_000_000_000


def build_m54_embedding_config() -> EmbeddingConfig:
    """Return the frozen M54 production query embedding configuration."""
    return EmbeddingConfig(
        model_name="AITeamVN/Vietnamese_Embedding",
        model_revision="dea33aa1ab339f38d66ae0a40e6c40e0a9249568",
        expected_dimension=1024,
        max_sequence_length=2048,
        device="cuda:0",
        torch_dtype="float32",
        document_prefix="",
        query_prefix="",
        query_prompt_name=None,
        query_instruction=None,
        normalize_embeddings=True,
        local_files_only=False,
    )


def build_m54_online_config() -> OnlineConfig:
    """Return the frozen M54 production online service configuration."""
    return OnlineConfig(
        retrieval=RetrievalConfig(
            top_k=10,
            candidate_k=40,
            default_strategy=RetrievalStrategy.HYBRID_RERANK,
            rrf_constant=60,
        ),
        bm25_runtime=BM25RuntimeConfig(),
        startup_validation=StartupValidationConfig(),
        vector_runtime=VectorRuntimeConfig(),
        query_understanding=QueryUnderstandingConfig(
            enabled=False,
            multi_query_enabled=False,
        ),
        evidence_selection=EvidenceSelectionConfig(
            enabled=True,
            reference_match_boost=3.0,
            lexical_overlap_weight=1.0,
            inactive_penalty=2.0,
            max_per_document=3,
            max_per_article=2,
        ),
        context_grading=ContextGradingConfig(
            minimum_evidence_count=1,
            require_document_number=False,
            require_article_number=False,
        ),
        claim_verification=ClaimVerificationConfig(
            enabled=True,
            require_inline_citations=False,
            minimum_lexical_support=0.2,
            minimum_claim_tokens=2,
            require_numeric_match=True,
            require_negation_match=True,
            max_claims=60,
        ),
        semantic_verification=SemanticVerificationConfig(backend="disabled"),
        reranker=RerankerConfig(
            backend="jina_native_listwise",
            model_name="jinaai/jina-reranker-v3.5",
            model_revision="e8a93f33f0b22108f8c2364f8484ce3422552fbc",
            device="cuda:0",
            torch_dtype="float16",
            native_context_cap=12288,
            expected_parameter_count=596836352,
            input_mode="legal_context",
            local_files_only=False,
            max_candidates=40,
            projector_checkpoint_path=None,
            projector_checkpoint_sha256=None,
            expected_projector_state_sha256=None,
            expected_projector_parameter_count=None,
        ),
        generation=GenerationConfig(
            backend="transformers",
            model_name="Qwen/Qwen3.5-2B",
            model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
            model_loader="image_text_to_text",
            device="cuda:1",
            torch_dtype="float16",
            local_files_only=False,
            max_context_tokens=6144,
            max_evidence=10,
            max_input_tokens=8192,
            temperature=0.0,
            max_output_tokens=1536,
            repetition_penalty=1.08,
            no_repeat_ngram_size=8,
            max_structured_output_retries=1,
            max_model_error_retries=1,
            model_failure_policy="top_evidence",
            max_grounding_repair_retries=1,
            grounding_failure_policy="supported_claims_or_top_evidence",
            extractive_fallback_max_evidence=3,
            salvage_rendering="standalone",
            prompt_schema_mode="compact_example",
            answer_style="competition_reference",
        ),
        agent=AgentConfig(),
    )
