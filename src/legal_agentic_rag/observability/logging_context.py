"""Privacy-aware structured context for standard-library logging."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LoggingContext:
    """Optional identifiers and metrics attached to one log event."""

    trace_id: str | None = None
    query_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    strategy: str | None = None
    latency_ms: float | None = None
    error_type: str | None = None

    def as_log_extra(self) -> dict[str, object]:
        """Return non-null context values accepted by ``LoggerAdapter``."""
        return {key: value for key, value in asdict(self).items() if value is not None}


def get_logger(
    name: str, context: LoggingContext | None = None
) -> logging.LoggerAdapter[logging.Logger]:
    """Return a named logger adapter without configuring global logging."""
    logger = logging.getLogger(name)
    extra = context.as_log_extra() if context is not None else {}
    return logging.LoggerAdapter(logger, extra)
