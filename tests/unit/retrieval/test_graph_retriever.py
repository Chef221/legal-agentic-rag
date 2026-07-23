"""Tests for bounded graph expansion and final reranking."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections.abc import Sequence

import pytest

from legal_agentic_rag.configuration import RetrievalConfig
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.retrieval import GraphExpandedRetriever
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
    *,
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


def _graph() -> AdjacencyGraphBackend:
    documents = [
        LegalDocument(
            document_id=f"doc-{index}",
            has_content=True,
            source_dataset="aio",
        )
        for index in range(1, 4)
    ]
    relationships = [
        LegalRelationship(
            source_document_id="doc-1",
            target_document_id="doc-2",
            relationship_type="amends",
            raw_relationship="Sửa đổi",
            source_dataset="aio",
        ),
        LegalRelationship(
            source_document_id="doc-2",
            target_document_id="doc-3",
            relationship_type="guides",
            raw_relationship="Hướng dẫn",
            source_dataset="aio",
        ),
    ]
    backend = AdjacencyGraphBackend()
    backend.build(
        documents,
        relationships,
        document_manifest=_manifest(
            ArtifactType.NORMALIZED_DOCUMENTS, 3, "documents-hash"
        ),
        relationship_manifest=_manifest(
            ArtifactType.RELATIONSHIP_MAPPING,
            2,
            "relationships-hash",
            metadata={"source_processing_config_hash": "documents-hash"},
        ),
    )
    return backend


def _hit(chunk_id: str, document_id: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        rank=rank,
        score=1 / (60 + rank),
        strategy=RetrievalStrategy.HYBRID,
        text=f"Legal text {chunk_id}",
    )


@dataclass
class _HybridRetriever:
    source_artifact_identity: tuple[str, str, str] = (
        "legal_chunks",
        "1.0",
        "chunks-hash",
    )
    calls: list[RetrievalQuery] = field(default_factory=list)

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.calls.append(query)
        document_ids = query.filters.document_ids
        if document_ids:
            available = {
                "doc-2": _hit("related-2", "doc-2", 1),
                "doc-3": _hit("related-3", "doc-3", 2),
            }
            hits = [available[item] for item in document_ids if item in available]
        else:
            hits = [_hit("seed-1", "doc-1", 1)]
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            hits=hits[: query.top_k],
            artifact_versions={"bm25_index": "1.0", "vector_index": "1.0"},
        )


class _FixtureReranker:
    provider_name = "fixture"
    provider_version = "1"
    model_name = "fixture-reranker"
    model_revision = "revision"

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> RetrievalResponse:
        ordered = sorted(
            candidates,
            key=lambda item: (item.document_id != "doc-2", item.chunk_id),
        )[: query.top_k]
        hits = [
            item.model_copy(
                update={
                    "rank": rank,
                    "score": float(10 - rank),
                    "strategy": RetrievalStrategy.RERANK,
                    "retrieval_trace": item.retrieval_trace.model_copy(
                        update={"reranker_score": float(10 - rank)}
                    ),
                }
            )
            for rank, item in enumerate(ordered, start=1)
        ]
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.RERANK,
            hits=hits,
        )


def _chunk_manifest() -> ArtifactManifest:
    return _manifest(ArtifactType.LEGAL_CHUNKS, 3, "chunks-hash")


def test_graph_retriever_expands_related_documents_and_preserves_path() -> None:
    """Related chunks are filtered by reached documents and reranked with trace."""
    hybrid = _HybridRetriever()
    retriever = GraphExpandedRetriever(
        hybrid,
        _graph(),
        _FixtureReranker(),
        _chunk_manifest(),
        RetrievalConfig(
            graph_hop_limit=1,
            graph_seed_chunk_k=1,
            graph_seed_document_k=1,
            graph_related_document_k=2,
        ),
    )
    query = RetrievalQuery(
        query_id="graph-query",
        original_question="Văn bản nào sửa đổi quy định này?",
        normalized_question="văn bản sửa đổi quy định",
        top_k=1,
        candidate_k=3,
        requested_strategy=RetrievalStrategy.GRAPH,
    )

    response = retriever.search(query)

    assert response.strategy == RetrievalStrategy.GRAPH
    assert [hit.chunk_id for hit in response.hits] == ["related-2"]
    trace = response.hits[0].retrieval_trace
    assert trace.graph_hop == 1
    assert [
        (
            step.source_document_id,
            step.target_document_id,
            step.relationship_type,
        )
        for step in trace.graph_path
    ] == [("doc-1", "doc-2", "amends")]
    assert trace.reranker_score == 9.0
    assert hybrid.calls[1].filters.document_ids == ["doc-2"]
    assert response.artifact_versions["graph_index"] == "1.0"


def test_graph_retriever_applies_relationship_and_document_filters() -> None:
    """Explicit filters prevent graph expansion from widening caller scope."""
    hybrid = _HybridRetriever()
    retriever = GraphExpandedRetriever(
        hybrid,
        _graph(),
        _FixtureReranker(),
        _chunk_manifest(),
        RetrievalConfig(graph_relationship_types=["guides"]),
    )
    query = RetrievalQuery(
        query_id="filtered-query",
        original_question="Question",
        normalized_question="question",
        top_k=1,
        candidate_k=3,
        requested_strategy=RetrievalStrategy.GRAPH,
    )

    response = retriever.search(query)

    assert len(hybrid.calls) == 1
    assert response.hits[0].chunk_id == "seed-1"
    assert "no_graph_expansion" in response.warnings


def test_graph_retriever_rejects_mismatched_chunk_artifact() -> None:
    """Text index provenance must match the supplied legal-chunks manifest."""
    hybrid = _HybridRetriever(
        source_artifact_identity=("legal_chunks", "1.0", "other-hash")
    )
    retriever = GraphExpandedRetriever(
        hybrid,
        _graph(),
        _FixtureReranker(),
        _chunk_manifest(),
    )
    query = RetrievalQuery(
        query_id="bad-artifact",
        original_question="Question",
        normalized_question="question",
        top_k=1,
        candidate_k=2,
        requested_strategy=RetrievalStrategy.GRAPH,
    )

    with pytest.raises(ArtifactCompatibilityError, match="chunk artifact"):
        retriever.search(query)
