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
    assert config.offline.dataset.dataset_name == (
        "th1nhng0/vietnamese-legal-documents"
    )


def test_full_corpus_configuration_is_pinned_and_unsampled() -> None:
    """The committed full build profile makes its completeness claim measurable."""
    path = Path(__file__).parents[3] / "configs" / "full-corpus.example.json"

    config = load_application_config(path)

    assert config.offline.dataset.dataset_revision == (
        "0a39ad7eae8e6c188cb225c4b1443c3b346461d8"
    )
    assert config.offline.dataset.sample_limit is None
    assert config.build_validation.require_full_corpus is True
    assert config.build_validation.expected_record_counts["metadata"] == 153_420


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
