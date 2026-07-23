"""Schema tests for normalized document run results."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    DocumentNormalizationResult,
)


def _artifact_manifest(record_count: int) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=record_count,
        processing_config_hash="hash",
    )


def test_normalization_result_rejects_manifest_count_mismatch() -> None:
    """A persisted manifest cannot claim documents absent from the result."""
    with pytest.raises(ValidationError, match="record_count"):
        DocumentNormalizationResult(
            documents=[],
            manifest=_artifact_manifest(record_count=1),
            input_metadata_count=0,
            input_content_count=0,
            rejected_metadata_count=0,
            orphan_content_count=0,
            ambiguous_content_count=0,
        )
