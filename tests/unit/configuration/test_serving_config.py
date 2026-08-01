"""Validation tests for the local HTTP and UI serving policy."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    ApplicationConfig,
    ArtifactConfig,
    OfflineConfig,
    OnlineConfig,
    RetrievalConfig,
    RerankerConfig,
    ServingConfig,
)


def test_evaluation_defaults_fit_online_reranker() -> None:
    """Default evaluation can run against the default online runtime."""
    config = _application_config()

    assert config.evaluation.cutoffs == [1, 5, 10]
    assert config.evaluation.candidate_k == 100


def _application_config(
    *,
    retrieval: RetrievalConfig | None = None,
    reranker: RerankerConfig | None = None,
    serving: ServingConfig | None = None,
) -> ApplicationConfig:
    return ApplicationConfig(
        artifacts=ArtifactConfig(root_path=Path("artifacts")),
        offline=OfflineConfig(),
        online=OnlineConfig(
            retrieval=retrieval or RetrievalConfig(),
            reranker=reranker or RerankerConfig(),
        ),
        serving=serving or ServingConfig(),
    )


def test_serving_config_defaults_to_local_bounded_process() -> None:
    """Defaults expose local serving while keeping query sizes bounded."""
    config = ServingConfig()

    assert config.host == "127.0.0.1"
    assert config.api_prefix == "/api/v1"
    assert config.ui_path == "/ui"
    assert config.max_top_k == 100
    assert config.max_candidate_k == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_prefix", "api/v1"),
        ("api_prefix", "/api/v1/"),
        ("ui_path", "/"),
        ("ui_path", "//ui"),
    ],
)
def test_serving_config_rejects_ambiguous_mount_paths(
    field: str,
    value: str,
) -> None:
    """API and UI paths must be unambiguous absolute mount points."""
    with pytest.raises(ValidationError):
        ServingConfig(**{field: value})


def test_serving_config_rejects_route_collision() -> None:
    """The mounted UI cannot replace the versioned API namespace."""
    with pytest.raises(ValidationError, match="must be different"):
        ServingConfig(api_prefix="/same", ui_path="/same")


def test_application_config_requires_defaults_to_fit_serving_policy() -> None:
    """A valid process cannot start with a default request it would reject."""
    with pytest.raises(ValidationError, match="candidate_k"):
        _application_config(
            retrieval=RetrievalConfig(top_k=5, candidate_k=20),
            serving=ServingConfig(max_top_k=10, max_candidate_k=10),
        )


def test_application_config_requires_defaults_to_fit_reranker() -> None:
    """The default candidate pool must fit the configured reranker bound."""
    with pytest.raises(ValidationError, match="candidate_k"):
        _application_config(
            retrieval=RetrievalConfig(top_k=5, candidate_k=20),
            reranker=RerankerConfig(max_candidates=10),
            serving=ServingConfig(max_top_k=20, max_candidate_k=20),
        )
