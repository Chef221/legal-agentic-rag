"""Validation tests for legal structure parsing result contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    DocumentParsingDiagnostic,
    LegalStructureParsingResult,
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


def test_document_diagnostic_validates_coverage_ratio() -> None:
    """Coverage cannot disagree with its source and covered character counts."""
    with pytest.raises(ValidationError, match="text_coverage"):
        DocumentParsingDiagnostic(
            document_id="doc-1",
            block_count=1,
            recognized_structure_count=0,
            source_non_whitespace_characters=10,
            covered_non_whitespace_characters=5,
            text_coverage=1.0,
            has_recognized_structure=False,
        )


def test_parsing_result_requires_legal_blocks_manifest() -> None:
    """Parsing output cannot use the identity of a cleaned document artifact."""
    with pytest.raises(ValidationError, match="legal blocks"):
        LegalStructureParsingResult(
            manifest=_manifest(ArtifactType.CLEANED_DOCUMENTS, 0),
            input_document_count=0,
            parsed_document_count=0,
            missing_clean_text_count=0,
            structured_document_count=0,
            unstructured_document_count=0,
        )
