"""Integration from persisted graph through fixed hybrid graph retrieval."""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

from legal_agentic_rag.configuration import RetrievalConfig
from legal_agentic_rag.exceptions import RetrievalError
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import FixedRetriever
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalDocument,
    LegalRelationship,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


def _manifest(
    artifact_type: ArtifactType,
    count: int,
    config_hash: str,
    metadata: dict[str, object] | None = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=artifact_type,
        artifact_version="1.0",
        dataset_name="fixture",
        dataset_revision="revision",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        record_count=count,
        processing_config_hash=config_hash,
        metadata=metadata or {},
    )


@dataclass
class _FilteredBranch:
    strategy: RetrievalStrategy
    source_artifact_identity: tuple[str, str, str] = (
        "legal_chunks",
        "1.0",
        "chunks-hash",
    )

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        document_id = (
            query.filters.document_ids[0]
            if query.filters.document_ids
            else "doc-seed"
        )
        chunk_id = "related" if document_id == "doc-related" else "seed"
        hit = RetrievalHit(
            chunk_id=chunk_id,
            document_id=document_id,
            rank=1,
            score=2.0 if self.strategy == RetrievalStrategy.BM25 else 0.8,
            strategy=self.strategy,
            text=(
                "Văn bản sửa đổi trực tiếp quy định."
                if chunk_id == "related"
                else "Quy định gốc."
            ),
        )
        return RetrievalResponse(
            query=query,
            strategy=self.strategy,
            hits=[hit],
            artifact_versions={f"{self.strategy.value}_index": "1.0"},
        )


class _SemanticModel:
    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object:
        return np.asarray(
            [2.0 if "sửa đổi trực tiếp" in passage else 0.1 for _, passage in inputs],
            dtype=np.float32,
        )


def test_fixed_graph_strategy_reloads_graph_and_reranks_related_chunk(
    tmp_path: object,
) -> None:
    """Persisted graph expansion is available through fixed routing without Agent."""
    documents = [
        LegalDocument(
            document_id=document_id,
            has_content=True,
            source_dataset="fixture-corpus",
        )
        for document_id in ("doc-seed", "doc-related")
    ]
    relationships = [
        LegalRelationship(
            source_document_id="doc-seed",
            target_document_id="doc-related",
            relationship_type="amends",
            raw_relationship="Sửa đổi",
            source_dataset="fixture-corpus",
        )
    ]
    graph = AdjacencyGraphBackend()
    graph.build(
        documents,
        relationships,
        document_manifest=_manifest(
            ArtifactType.NORMALIZED_DOCUMENTS, 2, "documents-hash"
        ),
        relationship_manifest=_manifest(
            ArtifactType.RELATIONSHIP_MAPPING,
            1,
            "relationships-hash",
            {"source_processing_config_hash": "documents-hash"},
        ),
    )
    destination = tmp_path / "graph"
    graph_manifest = graph.persist(destination)
    loaded_graph = AdjacencyGraphBackend()
    loaded_graph.load(destination, graph_manifest)
    chunk_manifest = _manifest(ArtifactType.LEGAL_CHUNKS, 2, "chunks-hash")
    retriever = FixedRetriever(
        _FilteredBranch(RetrievalStrategy.BM25),
        _FilteredBranch(RetrievalStrategy.DENSE),
        RetrievalConfig(
            graph_seed_chunk_k=1,
            graph_seed_document_k=1,
            graph_related_document_k=1,
        ),
        reranker=CrossEncoderReranker(
            model_loader=lambda config: _SemanticModel()
        ),
        graph_backend=loaded_graph,
        chunk_manifest=chunk_manifest,
    )

    response = retriever.search(
        RetrievalQuery(
            query_id="graph-integration",
            original_question="Văn bản nào sửa đổi?",
            normalized_question="văn bản sửa đổi",
            top_k=1,
            candidate_k=2,
            requested_strategy=RetrievalStrategy.GRAPH,
        )
    )

    assert response.strategy == RetrievalStrategy.GRAPH
    assert response.hits[0].chunk_id == "related"
    assert response.hits[0].retrieval_trace.graph_hop == 1
    assert response.hits[0].retrieval_trace.reranker_score == 2.0
    assert response.artifact_versions == {
        "bm25_index": "1.0",
        "dense_index": "1.0",
        "graph_index": "1.0",
    }

def test_fixed_graph_strategy_fails_when_graph_runtime_disabled(
    tmp_path: object,
) -> None:
    """When graph_runtime_enabled is False, FixedRetriever rejects explicit graph strategy."""
    documents = [
        LegalDocument(
            document_id="doc-seed",
            has_content=True,
            source_dataset="fixture-corpus",
        )
    ]
    graph = AdjacencyGraphBackend()
    graph.build(
        documents,
        [],
        document_manifest=_manifest(
            ArtifactType.NORMALIZED_DOCUMENTS, 1, "documents-hash"
        ),
        relationship_manifest=_manifest(
            ArtifactType.RELATIONSHIP_MAPPING,
            0,
            "relationships-hash",
            {"source_processing_config_hash": "documents-hash"},
        ),
    )
    destination = tmp_path / "graph"
    graph_manifest = graph.persist(destination)
    loaded_graph = AdjacencyGraphBackend()
    loaded_graph.load(destination, graph_manifest)
    chunk_manifest = _manifest(ArtifactType.LEGAL_CHUNKS, 1, "chunks-hash")

    retriever = FixedRetriever(
        _FilteredBranch(RetrievalStrategy.BM25),
        _FilteredBranch(RetrievalStrategy.DENSE),
        RetrievalConfig(graph_runtime_enabled=False),
        reranker=CrossEncoderReranker(
            model_loader=lambda config: _SemanticModel()
        ),
        graph_backend=loaded_graph,
        chunk_manifest=chunk_manifest,
    )

    with pytest.raises(
        RetrievalError,
        match="Fixed graph strategy requires graph, chunks, and reranker",
    ):
        retriever.search(
            RetrievalQuery(
                query_id="graph-disabled",
                original_question="Văn bản nào sửa đổi?",
                normalized_question="văn bản sửa đổi",
                top_k=1,
                candidate_k=2,
                requested_strategy=RetrievalStrategy.GRAPH,
            )
        )
