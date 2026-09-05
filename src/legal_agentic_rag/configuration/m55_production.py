"""Frozen production runtime configuration for M55 with deterministic First-2 Article answering."""

from __future__ import annotations

from legal_agentic_rag.configuration.m54_production import (
    M54_PARAMETER_LIMIT,
    M54_TOTAL_ACTIVE_MODEL_PARAMETERS,
    build_m54_embedding_config,
    build_m54_online_config,
)
from legal_agentic_rag.configuration.offline import EmbeddingConfig
from legal_agentic_rag.configuration.online import (
    ArticleAnswerConfig,
    OnlineConfig,
    RetrievalArtifactMode,
)

M55_PRODUCTION_SCHEMA_VERSION = "m55_production_v1"
M55_LOOKUP_FILENAME = "m55_a4_full_article_lookup_v1.jsonl"
M55_LOOKUP_SHA256 = (
    "202ba43b02403aa89ad1994a9979702efe1e7436e3b5178e964f65663c926dae"
)
M55_EXPECTED_RECORD_COUNT = 139073
M55_MAX_ARTICLES = 2
M55_STRUCTURAL_FALLBACK_MAX_EVIDENCE = 3

# Active model parameters in M55 Article mode:
# Only SentenceTransformer (Vietnamese_Embedding: ~135M) + Jina Reranker v3.5 (~597M).
# Qwen generation and citation verification are omitted.
# We retain the global parameter limit constant for reference.
M55_PARAMETER_LIMIT = M54_PARAMETER_LIMIT


def build_m55_embedding_config() -> EmbeddingConfig:
    """Return the frozen query embedding configuration (identical to M54)."""
    return build_m54_embedding_config()


M55_RETRIEVAL_TIMEOUT_SECONDS = 60.0


def build_m55_online_config() -> OnlineConfig:
    """Return the frozen M55 production online configuration with deterministic Article answering."""
    m54_online = build_m54_online_config()
    return m54_online.model_copy(
        update={
            "retrieval_artifact_mode": RetrievalArtifactMode.V2_PRECOMPUTED,
            "retrieval": m54_online.retrieval.model_copy(
                update={"timeout_seconds": M55_RETRIEVAL_TIMEOUT_SECONDS}
            ),
            "article_answer": ArticleAnswerConfig(
                enabled=True,
                max_articles=M55_MAX_ARTICLES,
                lookup_filename=M55_LOOKUP_FILENAME,
                lookup_sha256=M55_LOOKUP_SHA256,
                expected_record_count=M55_EXPECTED_RECORD_COUNT,
                structural_fallback_max_evidence=M55_STRUCTURAL_FALLBACK_MAX_EVIDENCE,
            ),
        }
    )