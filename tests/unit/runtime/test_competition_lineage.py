"""Tests for the competition-only online artifact boundary."""

from datetime import UTC, datetime

import pytest

from legal_agentic_rag.configuration import (
    CompetitionConfig,
    OFFICIAL_CORPUS_DATASET_NAME,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.runtime.startup_validation import (
    validate_competition_artifact_lineage,
)
from legal_agentic_rag.schemas import ArtifactManifest, ArtifactType


def _manifest(
    artifact_type: ArtifactType,
    *,
    dataset_name: str = OFFICIAL_CORPUS_DATASET_NAME,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=artifact_type,
        artifact_version="1.0",
        dataset_name=dataset_name,
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        record_count=1,
        processing_config_hash=f"{artifact_type.value}-hash",
    )


def test_official_artifact_lineage_is_accepted() -> None:
    """A complete artifact family may use the configured BTC identity."""
    manifests = tuple(
        _manifest(artifact_type)
        for artifact_type in (
            ArtifactType.LEGAL_CHUNKS,
            ArtifactType.BM25_INDEX,
            ArtifactType.VECTOR_INDEX,
            ArtifactType.GRAPH_INDEX,
        )
    )

    validate_competition_artifact_lineage(manifests, CompetitionConfig())


def test_external_artifact_lineage_is_rejected() -> None:
    """Legacy or third-party corpus artifacts cannot enter competition runtime."""
    manifests = (
        _manifest(ArtifactType.LEGAL_CHUNKS, dataset_name="external-corpus"),
    )

    with pytest.raises(ArtifactCompatibilityError, match="approved competition"):
        validate_competition_artifact_lineage(manifests, CompetitionConfig())


def test_mixed_artifact_lineage_is_rejected() -> None:
    """One official-looking manifest cannot hide a mixed corpus build."""
    manifests = (
        _manifest(ArtifactType.LEGAL_CHUNKS),
        _manifest(ArtifactType.BM25_INDEX, dataset_name="external-corpus"),
    )

    with pytest.raises(ArtifactCompatibilityError, match="different datasets"):
        validate_competition_artifact_lineage(manifests, CompetitionConfig())
