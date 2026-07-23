"""Tests for deterministic persisted directed graph indexing."""

from datetime import UTC, datetime

import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, RetrievalError
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalDocument,
    LegalRelationship,
)


def _documents() -> list[LegalDocument]:
    return [
        LegalDocument(
            document_id=f"doc-{index}",
            has_content=True,
            source_dataset="aio",
        )
        for index in range(1, 5)
    ]


def _relationships() -> list[LegalRelationship]:
    return [
        LegalRelationship(
            source_document_id="doc-1",
            target_document_id="doc-2",
            relationship_type="amends",
            raw_relationship="Sửa đổi",
            source_dataset="aio",
        ),
        LegalRelationship(
            source_document_id="doc-1",
            target_document_id="doc-3",
            relationship_type="guides",
            raw_relationship="Hướng dẫn",
            source_dataset="aio",
        ),
        LegalRelationship(
            source_document_id="doc-2",
            target_document_id="doc-4",
            relationship_type=None,
            raw_relationship="Liên quan",
            source_dataset="aio",
        ),
    ]


def _document_manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture",
        dataset_revision="revision",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        record_count=4,
        processing_config_hash="documents-hash",
    )


def _relationship_manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.RELATIONSHIP_MAPPING,
        artifact_version="1.0",
        dataset_name="fixture",
        dataset_revision="revision",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        record_count=3,
        processing_config_hash="relationships-hash",
        metadata={"source_processing_config_hash": "documents-hash"},
    )


def _built_backend() -> AdjacencyGraphBackend:
    backend = AdjacencyGraphBackend(
        clock=lambda: datetime(2026, 1, 3, tzinfo=UTC)
    )
    backend.build(
        _documents(),
        _relationships(),
        document_manifest=_document_manifest(),
        relationship_manifest=_relationship_manifest(),
    )
    return backend


def test_graph_build_and_bounded_directed_traversal() -> None:
    """One-hop and two-hop BFS preserve direction, labels, and hop values."""
    backend = _built_backend()

    one_hop = backend.traverse(["doc-1"], 1)
    two_hops = backend.traverse(["doc-1"], 2)

    assert [
        (step.target_document_id, step.relationship_type, step.hop)
        for step in one_hop
    ] == [
        ("doc-2", "amends", 1),
        ("doc-3", "guides", 1),
    ]
    assert [
        (step.target_document_id, step.relationship_type, step.hop)
        for step in two_hops
    ] == [
        ("doc-2", "amends", 1),
        ("doc-3", "guides", 1),
        ("doc-4", "Liên quan", 2),
    ]
    assert [
        step.target_document_id
        for step in backend.traverse(["doc-1"], 2, ["amends"])
    ] == ["doc-2"]
    assert backend.traverse(["doc-4"], 2) == []


def test_graph_persistence_reload_and_compatibility(tmp_path: object) -> None:
    """A persisted graph reloads exactly and refuses checksum corruption."""
    backend = _built_backend()
    destination = tmp_path / "graph"
    manifest = backend.persist(destination)
    loaded = AdjacencyGraphBackend()
    loaded.load(destination, manifest)

    assert loaded.manifest == manifest
    assert list(loaded.traverse(["doc-1"], 2)) == list(
        backend.traverse(["doc-1"], 2)
    )
    graph_path = destination / "graph.json"
    graph_path.write_text(
        graph_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactCompatibilityError, match="checksum"):
        AdjacencyGraphBackend().load(destination, manifest)


def test_graph_traversal_rejects_unbounded_or_unknown_requests() -> None:
    """Online callers cannot expand beyond two hops or unknown graph nodes."""
    backend = _built_backend()

    with pytest.raises(RetrievalError, match="between 1 and 2"):
        backend.traverse(["doc-1"], 3)
    with pytest.raises(RetrievalError, match="absent"):
        backend.traverse(["ghost"], 1)
