"""Validation tests for legal chunking result contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    DocumentChunkingDiagnostic,
    LegalChunkingResult,
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


def test_chunking_diagnostic_validates_block_coverage() -> None:
    """Coverage ratio must be derived from source and covered block counts."""
    with pytest.raises(ValidationError, match="block_coverage"):
        DocumentChunkingDiagnostic(
            document_id="doc-1",
            source_block_count=2,
            covered_block_count=1,
            chunk_count=1,
            article_unit_count=0,
            token_fallback_chunk_count=0,
            block_coverage=1.0,
            has_chunks=True,
        )


def test_chunking_result_requires_legal_chunks_manifest() -> None:
    """Chunking output cannot masquerade as a legal-block artifact."""
    with pytest.raises(ValidationError, match="legal chunks"):
        LegalChunkingResult(
            manifest=_manifest(ArtifactType.LEGAL_BLOCKS, 0),
            input_document_count=0,
            input_block_count=0,
            documents_with_chunks_count=0,
            documents_without_chunks_count=0,
            article_chunk_count=0,
            clause_fallback_chunk_count=0,
            token_fallback_chunk_count=0,
            standalone_chunk_count=0,
        )
