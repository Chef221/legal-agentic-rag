"""Tests for fail-closed competition data provenance policy."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    CompetitionConfig,
    OFFICIAL_CORPUS_DATASET_NAME,
)


def test_competition_config_defaults_to_official_only() -> None:
    """The active application cannot opt into external corpus by omission."""
    config = CompetitionConfig()

    assert config.corpus_dataset_name == OFFICIAL_CORPUS_DATASET_NAME
    assert config.data_policy == "competition_only"
    assert config.allow_external_data is False
    assert config.require_official_artifact_lineage is True


def test_competition_config_rejects_external_data_override() -> None:
    """A runtime config cannot weaken the accepted data-scope decision."""
    with pytest.raises(ValidationError):
        CompetitionConfig(allow_external_data=True)
