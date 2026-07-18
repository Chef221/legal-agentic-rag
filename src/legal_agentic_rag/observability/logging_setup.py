"""Explicit standard-library logging configuration for this package."""

import logging

from legal_agentic_rag.configuration.observability import LoggingConfig


class _ContextDefaultsFilter(logging.Filter):
    """Populate structured fields not supplied by a specific log event."""

    _DEFAULT_FIELDS = (
        "trace_id",
        "query_id",
        "document_id",
        "chunk_id",
        "strategy",
        "latency_ms",
        "error_type",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Add placeholder values required by the package formatter."""
        for field_name in self._DEFAULT_FIELDS:
            if not hasattr(record, field_name):
                setattr(record, field_name, "-")
        return True


def configure_logging(config: LoggingConfig) -> logging.Logger:
    """Configure and return the package logger without mutating the root logger."""
    package_logger = logging.getLogger("legal_agentic_rag")
    package_logger.setLevel(config.level)
    package_logger.propagate = False

    handler = logging.StreamHandler()
    handler.addFilter(_ContextDefaultsFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s "
            "trace_id=%(trace_id)s query_id=%(query_id)s "
            "document_id=%(document_id)s chunk_id=%(chunk_id)s "
            "strategy=%(strategy)s latency_ms=%(latency_ms)s "
            "error_type=%(error_type)s %(message)s"
        )
    )

    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    return package_logger
