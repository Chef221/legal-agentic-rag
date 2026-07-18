"""Standard-library logging foundation."""

from legal_agentic_rag.observability.logging_context import (
    LoggingContext,
    get_logger,
)
from legal_agentic_rag.observability.logging_setup import configure_logging

__all__ = ["LoggingContext", "configure_logging", "get_logger"]
