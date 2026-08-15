"""Tests for explicit JSON application configuration loading."""

from copy import deepcopy
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


def test_kaggle_qwen_configuration_is_bounded_and_local() -> None:
    """The M43 profile pins one local model and bounded hybrid generation."""
    path = (
        Path(__file__).parents[3]
        / "configs"
        / "uit-dsc-2026-task2-qwen3b-kaggle.example.json"
    )

    config = load_application_config(path)

    assert config.online.agent.strategy_order == ["hybrid"]
    assert config.online.generation.backend == "transformers"
    assert config.online.generation.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert config.online.generation.max_output_tokens == 256
    assert config.online.semantic_verification.backend == "disabled"
    assert config.competition.allow_external_data is False


def test_kaggle_qwen_reranker_ablation_changes_only_candidate_k() -> None:
    """M44.4 end-to-end profiles must isolate the candidate-pool variable."""
    root = Path(__file__).parents[3] / "configs"
    control = load_application_config(
        root / "uit-dsc-2026-task2-qwen3b-rerank-k20-kaggle.example.json"
    )
    treatment = load_application_config(
        root / "uit-dsc-2026-task2-qwen3b-rerank-k40-kaggle.example.json"
    )

    assert control.online.retrieval.top_k == treatment.online.retrieval.top_k == 8
    assert control.online.retrieval.candidate_k == 20
    assert treatment.online.retrieval.candidate_k == 40
    assert control.online.agent.strategy_order == ["hybrid_rerank"]
    assert treatment.online.agent.strategy_order == ["hybrid_rerank"]
    assert control.online.reranker.device == treatment.online.reranker.device == "cuda"
    assert control.online.generation.model_revision == (
        treatment.online.generation.model_revision
    )

    control_payload = deepcopy(control.model_dump(mode="json"))
    treatment_payload = deepcopy(treatment.model_dump(mode="json"))
    for payload in (control_payload, treatment_payload):
        payload["online"]["retrieval"]["candidate_k"] = 0
        payload["evaluation"]["candidate_k"] = 0

    assert control_payload == treatment_payload


def test_m49_2_smoke_profile_uses_bounded_larger_output_budget() -> None:
    """The doc-cap smoke profile isolates the measured structured-output fix."""
    path = (
        Path(__file__).parents[3]
        / "configs"
        / "uit-dsc-2026-task2-qwen3b-rerank-k40-doccap2-kaggle.example.json"
    )

    config = load_application_config(path)

    assert config.online.retrieval.candidate_k == 40
    assert config.online.evidence_selection.max_evidence_per_document == 2
    assert config.online.generation.max_output_tokens == 384
    assert config.online.generation.max_structured_output_retries == 1
    assert config.online.agent.max_numeric_mismatch_repairs == 1


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
