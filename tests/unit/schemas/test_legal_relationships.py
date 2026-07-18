"""Tests for normalized legal relationship contracts."""

from legal_agentic_rag.schemas.legal_relationships import LegalRelationship


def test_unmapped_relationship_uses_null_canonical_type() -> None:
    """Unknown raw labels remain auditable without guessed canonical values."""
    relationship = LegalRelationship(
        source_document_id="doc-1",
        target_document_id="doc-2",
        raw_relationship="nhãn chưa chuẩn hóa",
        source_dataset="fixture",
    )

    assert relationship.relationship_type is None
    assert relationship.raw_relationship == "nhãn chưa chuẩn hóa"
    assert relationship.is_directed is True


def test_empty_canonical_relationship_normalizes_to_null() -> None:
    """An empty canonical label must not be treated as a valid mapping."""
    relationship = LegalRelationship(
        source_document_id="doc-1",
        target_document_id="doc-2",
        relationship_type="  ",
        raw_relationship="liên quan",
        source_dataset="fixture",
    )
    assert relationship.relationship_type is None
