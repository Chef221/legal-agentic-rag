"""Tests for AIO-boundary legal relationship normalization."""

from datetime import UTC, datetime

from legal_agentic_rag.configuration import RelationshipNormalizationConfig
from legal_agentic_rag.offline.datasets.aio import AioRelationshipNormalizer
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalDocument,
)


def _documents() -> list[LegalDocument]:
    return [
        LegalDocument(
            document_id=f"doc-{index}",
            has_content=True,
            source_dataset="aio",
        )
        for index in range(1, 4)
    ]


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        record_count=3,
        processing_config_hash="documents-hash",
    )


def test_relationship_normalizer_maps_only_explicit_labels_and_rejects_bad_edges() -> None:
    """Raw labels stay preserved while duplicates, orphans, and self-loops fail."""
    records = [
        {"doc_id": "doc-1", "other_doc_id": "doc-2", "relationship": "Sửa đổi"},
        {"doc_id": "doc-1", "other_doc_id": "doc-2", "relationship": "Sửa đổi"},
        {"doc_id": "doc-2", "other_doc_id": "doc-3", "relationship": "Liên quan"},
        {"doc_id": "doc-1", "other_doc_id": "doc-1", "relationship": "Liên quan"},
        {"doc_id": "doc-1", "other_doc_id": "ghost", "relationship": "Hướng dẫn"},
        {"doc_id": "doc-3", "other_doc_id": "doc-1", "relationship": " "},
    ]
    normalizer = AioRelationshipNormalizer(
        RelationshipNormalizationConfig(
            relationship_type_mapping={"Sửa đổi": "amends"}
        ),
        clock=lambda: datetime(2026, 1, 2, tzinfo=UTC),
    )

    result = normalizer.normalize(
        relationship_records=records,
        documents=_documents(),
        document_manifest=_manifest(),
    )

    assert result.input_count == 6
    assert result.rejected_count == 4
    assert result.duplicate_count == 1
    assert [item.raw_relationship for item in result.relationships] == [
        "Sửa đổi",
        "Liên quan",
    ]
    assert [item.relationship_type for item in result.relationships] == [
        "amends",
        None,
    ]
    assert {issue.issue_type for issue in result.issues} == {
        "duplicate_relationship",
        "relationship_self_loop",
        "orphan_relationship_endpoint",
        "missing_relationship_label",
    }
    assert result.manifest.record_count == 2
    assert result.manifest.metadata["source_processing_config_hash"] == (
        "documents-hash"
    )
    assert "unmapped_relationship_labels_preserved" in result.manifest.warnings


def test_relationship_normalization_is_deterministic_for_input_order() -> None:
    """Accepted graph edge ordering does not depend on raw record order."""
    records = [
        {"doc_id": "doc-2", "other_doc_id": "doc-3", "relationship": "B"},
        {"doc_id": "doc-1", "other_doc_id": "doc-3", "relationship": "A"},
    ]
    normalizer = AioRelationshipNormalizer()

    first = normalizer.normalize(
        relationship_records=records,
        documents=_documents(),
        document_manifest=_manifest(),
    )
    second = normalizer.normalize(
        relationship_records=reversed(records),
        documents=_documents(),
        document_manifest=_manifest(),
    )

    first_edges = [
        (item.source_document_id, item.target_document_id)
        for item in first.relationships
    ]
    second_edges = [
        (item.source_document_id, item.target_document_id)
        for item in second.relationships
    ]
    assert first_edges == second_edges == [("doc-1", "doc-3"), ("doc-2", "doc-3")]
