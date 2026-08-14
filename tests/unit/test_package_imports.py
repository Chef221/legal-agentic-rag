"""Basic import tests for the Milestone 1 package surface."""


def test_package_imports_without_backend_side_effects() -> None:
    """The package and its public contract groups must import successfully."""
    import legal_agentic_rag
    import legal_agentic_rag.agent
    import legal_agentic_rag.configuration
    import legal_agentic_rag.contracts
    import legal_agentic_rag.embeddings
    import legal_agentic_rag.evaluation
    import legal_agentic_rag.generation
    import legal_agentic_rag.indexing.bm25
    import legal_agentic_rag.indexing.graph
    import legal_agentic_rag.indexing.vector
    import legal_agentic_rag.observability
    import legal_agentic_rag.offline.chunking
    import legal_agentic_rag.offline.cleaning
    import legal_agentic_rag.competition.uit_dsc_2026
    import legal_agentic_rag.offline.parsing
    import legal_agentic_rag.offline.relationships
    import legal_agentic_rag.reranking
    import legal_agentic_rag.retrieval
    import legal_agentic_rag.runtime
    import legal_agentic_rag.schemas
    import legal_agentic_rag.serving
    import legal_agentic_rag.tools

    assert legal_agentic_rag.__version__ == "0.49.1"
