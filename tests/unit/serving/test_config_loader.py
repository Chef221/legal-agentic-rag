"""Tests for explicit JSON application configuration loading."""

from pathlib import Path

import pytest

from legal_agentic_rag.exceptions import ConfigurationError
from legal_agentic_rag.serving.config_loader import load_application_config


def test_example_configuration_loads() -> None:
    """The committed baseline example is a complete ApplicationConfig."""
    path = Path(__file__).parents[3] / "configs" / "baseline.example.json"

    config = load_application_config(path)

    assert config.serving.api_prefix == "/api/v1"
    assert config.competition.data_policy == "competition_only"
    assert config.competition.allow_external_data is False


def test_colab_m45_qwen3_configuration_loads_strong_local_pipeline() -> None:
    """M45 pins all local models and enables bounded hybrid reranking."""
    path = (
        Path(__file__).parents[3]
        / "configs"
        / "uit-dsc-2026-task2-m45-qwen3-colab.example.json"
    )

    config = load_application_config(path)

    assert config.offline.embedding.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert config.offline.embedding.expected_dimension == 1024
    assert config.online.retrieval.default_strategy == "hybrid_rerank"
    assert config.online.reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert config.online.generation.model_name == "Qwen/Qwen3.5-2B"
    assert config.online.generation.model_loader == "image_text_to_text"
    assert config.online.semantic_verification.backend == "disabled"
    assert config.competition.allow_external_data is False


def test_m48_configuration_reuses_models_and_enables_safe_answer_recovery() -> None:
    """M48 changes only prompt/recovery behavior on the immutable M45 DB."""
    path = (
        Path(__file__).parents[3]
        / "configs"
        / "uit-dsc-2026-task2-m48-qwen3-dev.example.json"
    )

    config = load_application_config(path)

    assert config.offline.embedding.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert config.online.reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert config.online.generation.model_name == "Qwen/Qwen3.5-2B"
    assert config.online.generation.answer_style == "competition_reference"
    assert config.online.generation.prompt_schema_mode == "compact_example"
    assert config.online.generation.max_output_tokens == 1536
    assert config.online.generation.model_failure_policy == "top_evidence"
    assert (
        config.online.generation.grounding_failure_policy
        == "supported_claims_or_top_evidence"
    )
    assert config.online.generation.salvage_rendering == "standalone"
    assert config.online.semantic_verification.backend == "disabled"


def test_m49_configuration_uses_the_local_merged_generator() -> None:
    """M49 changes generator identity without rebuilding M45 retrieval."""
    path = (
        Path(__file__).parents[3]
        / "configs"
        / "uit-dsc-2026-task2-m49-qwen3-dev.example.json"
    )

    config = load_application_config(path)

    assert config.offline.embedding.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert config.online.reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert config.online.generation.model_name == "/kaggle/working/m49-generator-merged"
    assert config.online.generation.local_files_only is True
    assert config.online.generation.prompt_schema_mode == "compact_example"


def test_m491_configuration_aligns_output_and_repetition_controls() -> None:
    """M49.1 retains M49 weights and changes only bounded online behavior."""
    path = (
        Path(__file__).parents[3]
        / "configs"
        / "uit-dsc-2026-task2-m491-qwen3-dev.example.json"
    )

    config = load_application_config(path)

    assert config.online.generation.model_name == "/kaggle/working/m49-generator-merged"
    assert config.online.generation.prompt_schema_mode == "plain_text_markers"
    assert config.online.generation.repetition_penalty == 1.08
    assert config.online.generation.no_repeat_ngram_size == 8
    assert config.online.reranker.relationship_candidate_k == 20
    assert config.online.reranker.max_candidates == 40
    assert config.online.retrieval.candidate_k == 40
    assert config.online.retrieval.top_k == 10


def test_config_loader_wraps_invalid_json_without_leaking_details(
    tmp_path: Path,
) -> None:
    """Malformed configuration becomes the project exception taxonomy."""
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(
        ConfigurationError,
        match="could not be loaded",
    ):
        load_application_config(path)
