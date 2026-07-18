"""Import tests for all approved backend-neutral protocols."""

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


def test_all_approved_contracts_are_protocols() -> None:
    """Milestone 1 exposes exactly the approved backend boundaries."""
    contracts = (
        DatasetSource,
        BM25Backend,
        EmbeddingProvider,
        VectorBackend,
        Reranker,
        GraphBackend,
        AnswerGenerator,
        ContextGrader,
        CitationVerifier,
    )
    assert all(getattr(contract, "_is_protocol", False) for contract in contracts)
