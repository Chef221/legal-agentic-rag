"""Integration from raw AIO relationships to a reloadable graph artifact."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.offline.datasets.aio import AioRelationshipNormalizer
from legal_agentic_rag.offline.relationships import (
    load_relationship_artifact,
    persist_relationship_artifact,
)
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalDocument,
)


def test_raw_relationships_flow_to_persisted_graph(
    tmp_path: Path,
    load_raw_aio_fixture: Any,
) -> None:
    """Invalid AIO edges are audited before accepted edges reach graph traversal."""
    documents = [
        LegalDocument(
            document_id=document_id,
            has_content=True,
            source_dataset="aio",
        )
        for document_id in ("doc-1", "doc-2")
    ]
    document_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        record_count=2,
        processing_config_hash="documents-hash",
    )
    result = AioRelationshipNormalizer().normalize(
        relationship_records=load_raw_aio_fixture("relationships"),
        documents=documents,
        document_manifest=document_manifest,
    )
    relationship_destination = tmp_path / "relationships"
    relationship_manifest = persist_relationship_artifact(
        relationships=result.relationships,
        destination=relationship_destination,
        manifest=result.manifest,
    )
    relationships, loaded_relationship_manifest = load_relationship_artifact(
        source=relationship_destination,
        supplied_manifest=relationship_manifest,
    )
    graph = AdjacencyGraphBackend()
    graph.build(
        documents,
        relationships,
        document_manifest=document_manifest,
        relationship_manifest=loaded_relationship_manifest,
    )
    graph_destination = tmp_path / "graph"
    graph_manifest = graph.persist(graph_destination)
    loaded_graph = AdjacencyGraphBackend()
    loaded_graph.load(graph_destination, graph_manifest)

    assert result.input_count == 6
    assert result.manifest.record_count == 2
    assert result.rejected_count == 4
    assert len(loaded_graph.traverse(["doc-1"], 1)) == 1
