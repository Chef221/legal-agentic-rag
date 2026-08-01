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
