"""Factory functions for instantiating configured reranker backends."""

from __future__ import annotations

from legal_agentic_rag.configuration.online import RerankerConfig
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import ConfigurationError
from legal_agentic_rag.reranking.cross_encoder import CrossEncoderReranker
from legal_agentic_rag.reranking.jina_native import JinaNativeReranker


def build_reranker(config: RerankerConfig | None = None) -> Reranker:
    """Build the configured reranker backend, defaulting to sentence_transformers."""
    resolved_config = config or RerankerConfig()
    backend = resolved_config.backend

    if backend == "sentence_transformers_cross_encoder":
        return CrossEncoderReranker(resolved_config)
    if backend == "jina_native_listwise":
        return JinaNativeReranker(resolved_config)

    raise ConfigurationError(f"Unsupported reranker backend: {backend}")
