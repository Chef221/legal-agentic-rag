"""Validation tests for HTML cleaning result contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    HtmlCleaningResult,
    LegalDocument,
)


def _manifest(artifact_type: ArtifactType, record_count: int) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=artifact_type,
        artifact_version="1.0",
        dataset_name="fixture",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=record_count,
        processing_config_hash="hash",
    )


def test_cleaning_result_requires_consistent_counts() -> None:
    """Every input record must be classified by exactly one outcome."""
    document = LegalDocument(
        document_id="doc-1",
        clean_text="Điều 1.",
        has_content=True,
        source_dataset="fixture",
    )

    with pytest.raises(ValidationError, match="one cleaning outcome"):
        HtmlCleaningResult(
            documents=[document],
            manifest=_manifest(ArtifactType.CLEANED_DOCUMENTS, 1),
            input_document_count=1,
            cleaned_document_count=0,
            missing_content_count=0,
            empty_output_count=0,
        )


def test_cleaning_result_requires_cleaned_artifact_manifest() -> None:
    """A result cannot masquerade as an upstream normalized artifact."""
    with pytest.raises(ValidationError, match="cleaned documents"):
        HtmlCleaningResult(
            documents=[],
            manifest=_manifest(ArtifactType.NORMALIZED_DOCUMENTS, 0),
            input_document_count=0,
            cleaned_document_count=0,
            missing_content_count=0,
            empty_output_count=0,
        )
