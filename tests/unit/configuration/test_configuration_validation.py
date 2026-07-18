"""Validation tests for bounded offline and online settings."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    AgentConfig,
    ChunkingConfig,
    LoggingConfig,
    RetrievalConfig,
)


def test_retrieval_config_validates_candidate_and_graph_limits() -> None:
    """Invalid retrieval bounds fail before any backend is called."""
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=20, candidate_k=10)
    with pytest.raises(ValidationError):
        RetrievalConfig(graph_hop_limit=3)


def test_chunking_config_validates_token_relationships() -> None:
    """Token fallback configuration cannot contain impossible limits."""
    with pytest.raises(ValidationError):
        ChunkingConfig(
            max_tokens=128,
            min_tokens=256,
            overlap_tokens=16,
            tokenizer_name="fixture",
        )


def test_agent_retry_is_capped_at_two() -> None:
    """Configuration enforces accepted decision D014."""
    with pytest.raises(ValidationError):
        AgentConfig(max_retry=3)


def test_logging_level_uses_standard_library_names() -> None:
    """Logging configuration accepts standard names and rejects unknown levels."""
    assert LoggingConfig(level="warning").level == "WARNING"
    with pytest.raises(ValidationError):
        LoggingConfig(level="verbose")
