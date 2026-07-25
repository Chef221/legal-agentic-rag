"""Tests for framework-neutral configuration composition."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    ApplicationConfig,
    ArtifactConfig,
    BuildValidationConfig,
    ChunkingConfig,
    DatasetSourceConfig,
    GenerationConfig,
    OfflineConfig,
    OnlineConfig,
)


def _application_config() -> ApplicationConfig:
    return ApplicationConfig(
        artifacts=ArtifactConfig(root_path=Path("artifacts")),
        offline=OfflineConfig(
            dataset=DatasetSourceConfig(dataset_name="fixture"),
            chunking=ChunkingConfig(
                max_tokens=512,
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
    assert config.offline.dataset.dataset_name == "fixture"
    assert config.online.agent.max_retry == 2
    assert config.online.reranker.max_candidates == 100
    assert config.logging.level == "INFO"


def test_application_config_has_stable_json_representation() -> None:
    """Config can be serialized before a processing hash is computed later."""
    config = _application_config()
    payload = config.model_dump_json()
    assert '"root_path":"artifacts"' in payload


def test_application_config_rejects_sample_full_corpus_profile() -> None:
    """Full-corpus validation cannot be attached to a sampled dataset source."""
    base = _application_config()
    with pytest.raises(ValidationError, match="sample_limit"):
        ApplicationConfig(
            artifacts=base.artifacts,
            offline=OfflineConfig(
                dataset=DatasetSourceConfig(
                    dataset_name="fixture",
                    dataset_revision="pinned",
                    sample_limit=100,
                )
            ),
            online=base.online,
            build_validation=BuildValidationConfig(
                require_pinned_dataset_revision=True,
                require_full_corpus=True,
                expected_record_counts={
                    "metadata": 1,
                    "content": 1,
                    "relationships": 1,
                },
            ),
        )
