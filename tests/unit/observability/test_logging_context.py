"""Tests for explicit, privacy-aware standard-library logging setup."""

import logging

from legal_agentic_rag.configuration import LoggingConfig
from legal_agentic_rag.observability import LoggingContext, configure_logging, get_logger


def test_logging_context_contains_identifiers_not_legal_content() -> None:
    """The base context carries operational identifiers and metrics only."""
    context = LoggingContext(
        trace_id="trace-1",
        document_id="doc-1",
        strategy="bm25",
        latency_ms=12.5,
    )
    extra = context.as_log_extra()
    assert extra["trace_id"] == "trace-1"
    assert "legal_text" not in extra
    assert "query_text" not in extra


def test_logging_is_configured_only_when_explicitly_called(capsys: object) -> None:
    """Package logging avoids print and does not require a tracing backend."""
    package_logger = configure_logging(LoggingConfig(level="INFO"))
    adapter = get_logger(
        "legal_agentic_rag.test", LoggingContext(trace_id="trace-1")
    )
    adapter.info("schema validated")

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "trace_id=trace-1" in captured.err
    assert "schema validated" in captured.err
    assert package_logger.propagate is False
    assert logging.getLogger().level != logging.NOTSET or package_logger.handlers
