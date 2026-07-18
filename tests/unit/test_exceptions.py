"""Tests for the explicit application exception taxonomy."""

from legal_agentic_rag.exceptions import (
    ConfigurationError,
    LegalAgenticRAGError,
    OperationTimeoutError,
)


def test_domain_exceptions_share_a_safe_base_type() -> None:
    """Callers can catch package errors without catching unrelated exceptions."""
    assert issubclass(ConfigurationError, LegalAgenticRAGError)
    assert issubclass(OperationTimeoutError, LegalAgenticRAGError)
