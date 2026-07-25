"""Import tests for all approved backend-neutral protocols."""

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


def test_all_approved_contracts_are_protocols() -> None:
    """Milestone 1 exposes exactly the approved backend boundaries."""
    contracts = (
        DatasetSource,
        AgentWorkflow,
        BM25Backend,
        ChatModelProvider,
        EmbeddingProvider,
        VectorBackend,
        Reranker,
        GraphBackend,
        AnswerGenerator,
        ContextGrader,
        CitationVerifier,
        RetrievalEvaluator,
        GenerationEvaluator,
    )
    assert all(getattr(contract, "_is_protocol", False) for contract in contracts)
