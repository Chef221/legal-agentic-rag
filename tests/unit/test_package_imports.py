"""Basic import tests for the Milestone 1 package surface."""


def test_package_imports_without_backend_side_effects() -> None:
    """The package and its public contract groups must import successfully."""
    import legal_agentic_rag
    import legal_agentic_rag.configuration
    import legal_agentic_rag.contracts
    import legal_agentic_rag.observability
    import legal_agentic_rag.schemas

    assert legal_agentic_rag.__version__ == "0.1.0"
