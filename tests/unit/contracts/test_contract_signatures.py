"""Structural tests for minimum protocol capabilities."""

from legal_agentic_rag.contracts import (
    AnswerGenerator,
    BM25Backend,
    CitationVerifier,
    ContextGrader,
    DatasetSource,
    EmbeddingProvider,
    GraphBackend,
    Reranker,
    VectorBackend,
)


def test_contracts_expose_only_domain_capabilities() -> None:
    """Each backend boundary exposes the documented minimum methods."""
    expected_methods = {
        DatasetSource: {"iter_records", "dataset_manifest"},
        BM25Backend: {"build", "search", "persist", "load"},
        EmbeddingProvider: {"embed_documents", "embed_query"},
        VectorBackend: {"build", "search", "persist", "load"},
        Reranker: {"rerank"},
        GraphBackend: {"build", "traverse", "persist", "load"},
        AnswerGenerator: {"generate"},
        ContextGrader: {"grade"},
        CitationVerifier: {"verify"},
    }
    for contract, method_names in expected_methods.items():
        assert method_names.issubset(contract.__dict__)


def test_no_generic_base_backend_exists() -> None:
    """Independent protocols are not coupled through a generic base backend."""
    import legal_agentic_rag.contracts as contracts

    assert not hasattr(contracts, "BaseBackend")
