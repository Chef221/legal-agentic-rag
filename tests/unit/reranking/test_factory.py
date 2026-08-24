"""Tests for the reranker factory."""

import pytest

from legal_agentic_rag.configuration.online import RerankerConfig
from legal_agentic_rag.exceptions import ConfigurationError
from legal_agentic_rag.reranking.cross_encoder import CrossEncoderReranker
from legal_agentic_rag.reranking.factory import build_reranker
from legal_agentic_rag.reranking.jina_native import JinaNativeReranker


def test_build_reranker_defaults_to_sentence_transformers() -> None:
    """Ensure default config instantiates CrossEncoderReranker without altering legacy behavior."""
    reranker = build_reranker()
    assert isinstance(reranker, CrossEncoderReranker)


def test_build_reranker_explicit_sentence_transformers() -> None:
    """Ensure explicit sentence_transformers backend instantiates CrossEncoderReranker."""
    config = RerankerConfig(backend="sentence_transformers_cross_encoder")
    reranker = build_reranker(config)
    assert isinstance(reranker, CrossEncoderReranker)


def test_build_reranker_jina_native() -> None:
    """Ensure jina_native_listwise backend instantiates JinaNativeReranker."""
    config = RerankerConfig(
        backend="jina_native_listwise",
        model_name="jinaai/jina-reranker-v3.5",
        model_revision="e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        native_context_cap=12288,
    )
    reranker = build_reranker(config)
    assert isinstance(reranker, JinaNativeReranker)
    assert reranker.model_name == "jinaai/jina-reranker-v3.5"
    assert reranker.model_revision == "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
