"""Structural tests for minimum protocol capabilities."""

from inspect import signature

from legal_agentic_rag.contracts import (
    AgentWorkflow,
    AnswerGenerator,
    BM25Backend,
    ChatModelProvider,
    CitationVerifier,
    ContextGrader,
    DatasetSource,
    EmbeddingProvider,
    GenerationEvaluator,
    GraphBackend,
    Reranker,
    RetrievalEvaluator,
    VectorBackend,
)


def test_contracts_expose_only_domain_capabilities() -> None:
    """Each backend boundary exposes the documented minimum methods."""
    expected_methods = {
        DatasetSource: {"iter_records", "dataset_manifest"},
        BM25Backend: {"build", "search", "persist", "load"},
        ChatModelProvider: {"complete"},
        EmbeddingProvider: {"embed_documents", "embed_query"},
        VectorBackend: {
            "build",
            "build_persisted",
            "search",
            "persist",
            "load",
        },
        Reranker: {"rerank"},
        GraphBackend: {"build", "traverse", "persist", "load"},
        AnswerGenerator: {"generate"},
        ContextGrader: {"grade"},
        CitationVerifier: {"verify"},
        AgentWorkflow: {"run"},
        RetrievalEvaluator: {"evaluate"},
        GenerationEvaluator: {"evaluate"},
    }
    for contract, method_names in expected_methods.items():
        assert method_names.issubset(contract.__dict__)


def test_no_generic_base_backend_exists() -> None:
    """Independent protocols are not coupled through a generic base backend."""
    import legal_agentic_rag.contracts as contracts

    assert not hasattr(contracts, "BaseBackend")


def test_bm25_build_contract_requires_source_manifest() -> None:
    """Index provenance is explicit at the backend boundary."""
    assert "source_artifact_identity" in BM25Backend.__dict__
    assert list(signature(BM25Backend.build).parameters) == [
        "self",
        "chunks",
        "source_manifest",
    ]


def test_vector_contract_exposes_provenance_and_embedding_identity() -> None:
    """Dense build/load boundaries expose compatibility-critical identity."""
    assert {
        "embedding_provider_name",
        "embedding_provider_version",
        "source_artifact_identity",
        "model_name",
        "model_revision",
        "dimension",
    }.issubset(
        VectorBackend.__dict__
    )
    assert list(signature(VectorBackend.build).parameters) == [
        "self",
        "chunks",
        "vectors",
        "source_manifest",
        "model_name",
        "model_revision",
        "embedding_provider_name",
        "embedding_provider_version",
        "dimension",
        "embedding_batch_size",
    ]
    assert list(signature(VectorBackend.build_persisted).parameters) == [
        "self",
        "batches",
        "source_manifest",
        "destination",
        "model_name",
        "model_revision",
        "embedding_provider_name",
        "embedding_provider_version",
        "dimension",
        "embedding_batch_size",
    ]
    assert list(signature(EmbeddingProvider.embed_documents).parameters) == [
        "self",
        "texts",
        "batch_size",
    ]
    assert {
        "provider_name",
        "provider_version",
        "model_name",
        "model_revision",
        "dimension",
    }.issubset(EmbeddingProvider.__dict__)


def test_reranker_contract_exposes_provider_and_model_identity() -> None:
    """Reranker observability includes reproducibility-critical identity."""
    assert {
        "provider_name",
        "provider_version",
        "model_name",
        "model_revision",
        "rerank",
    }.issubset(Reranker.__dict__)
    assert list(signature(Reranker.rerank).parameters) == [
        "self",
        "query",
        "candidates",
    ]


def test_chat_model_contract_exposes_reproducible_identity() -> None:
    """Generator providers retain model identity without a shared base backend."""
    assert {
        "provider_name",
        "provider_version",
        "model_name",
        "model_revision",
        "complete",
    }.issubset(ChatModelProvider.__dict__)
    assert list(signature(ChatModelProvider.complete).parameters) == [
        "self",
        "system_instruction",
        "user_prompt",
    ]


def test_graph_contract_requires_source_manifests_and_exposes_manifest() -> None:
    """Graph builds retain both document and relationship provenance."""
    assert "manifest" in GraphBackend.__dict__
    assert list(signature(GraphBackend.build).parameters) == [
        "self",
        "documents",
        "relationships",
        "document_manifest",
        "relationship_manifest",
    ]
