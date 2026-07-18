"""Tests for legal document, hierarchy, block, and chunk schemas."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalBlockType,
    LegalChunk,
    LegalDocument,
    LegalStructure,
)


def test_legal_document_parses_fixture_and_preserves_provenance(
    load_schema_sample: object,
) -> None:
    """A normalized document must serialize with its source provenance."""
    data = load_schema_sample("valid_legal_document.json")  # type: ignore[operator]
    document = LegalDocument.model_validate(data)

    assert document.document_id == "doc-001"
    assert document.source_dataset == "aio"
    assert document.effective_date.isoformat() == "2026-01-01"


def test_legal_document_requires_source_dataset() -> None:
    """Core schemas must not silently default to the current dataset."""
    with pytest.raises(ValidationError):
        LegalDocument(document_id="doc-001", has_content=False)


def test_optional_empty_document_metadata_becomes_null() -> None:
    """Nullable textual metadata follows the unified empty-string convention."""
    document = LegalDocument(
        document_id="doc-001",
        title="   ",
        has_content=False,
        source_dataset="fixture",
    )
    assert document.title is None


def test_legal_structure_supports_documents_without_articles() -> None:
    """Unstructured legal documents may carry an empty hierarchy."""
    structure = LegalStructure()
    assert structure.structure_path == []
    assert structure.article_number is None


def test_legal_block_rejects_self_parent() -> None:
    """A parsed block cannot be its own hierarchy parent."""
    with pytest.raises(ValidationError):
        LegalBlock(
            block_id="block-1",
            document_id="doc-1",
            block_type=LegalBlockType.ARTICLE,
            text="Điều 1.",
            parent_block_id="block-1",
            order_index=0,
        )


def test_legal_chunk_parses_fixture_and_validates_token_count(
    load_schema_sample: object,
) -> None:
    """A retrieval chunk must retain hierarchy and have a positive token count."""
    data = load_schema_sample("valid_legal_chunk.json")  # type: ignore[operator]
    chunk = LegalChunk.model_validate(data)

    assert chunk.structure.article_number == "1"
    assert chunk.token_count == 12

    with pytest.raises(ValidationError):
        LegalChunk.model_validate({**data, "token_count": 0})


def test_schema_rejects_unknown_top_level_fields() -> None:
    """Extension data must use metadata instead of leaking raw fields."""
    with pytest.raises(ValidationError):
        LegalDocument(
            document_id="doc-001",
            has_content=False,
            source_dataset="fixture",
            unknown_raw_field="value",
        )
