"""Tests for framework-neutral configuration composition."""

from pathlib import Path

from legal_agentic_rag.configuration import (
    ApplicationConfig,
    ArtifactConfig,
    ChunkingConfig,
    GenerationConfig,
    OfflineConfig,
    OnlineConfig,
)


def _application_config() -> ApplicationConfig:
    return ApplicationConfig(
        artifacts=ArtifactConfig(root_path=Path("artifacts")),
        offline=OfflineConfig(
            chunking=ChunkingConfig(
                max_tokens=384,
                max_search_tokens=448,
                min_tokens=32,
                overlap_tokens=32,
                tokenizer_name="unicode_word_v1",
            ),
        ),
        online=OnlineConfig(generation=GenerationConfig(max_context_tokens=4096)),
    )


def test_application_config_composes_without_external_framework() -> None:
    """Typed config can be constructed directly from Python or serialized data."""
    config = _application_config()
    assert config.competition.data_policy == "competition_only"
    assert config.online.agent.max_retry == 2
    assert config.online.reranker.max_candidates == 100
    assert config.logging.level == "INFO"


def test_application_config_has_stable_json_representation() -> None:
    """Config can be serialized before a processing hash is computed later."""
    config = _application_config()
    payload = config.model_dump_json()
    assert '"root_path":"artifacts"' in payload
