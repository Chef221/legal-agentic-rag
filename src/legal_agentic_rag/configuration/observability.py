"""Typed configuration for the standard-library logging foundation."""

import logging

from pydantic import BaseModel, ConfigDict, field_validator


class LoggingConfig(BaseModel):
    """Logging level and privacy-safe content inclusion policy."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    include_query_text: bool = False
    include_legal_text: bool = False

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        """Accept only standard named logging levels."""
        normalized = value.strip().upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError("unsupported logging level")
        return normalized
