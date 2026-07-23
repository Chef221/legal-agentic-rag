"""Minimal JSON configuration loading for build and serving commands."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.exceptions import ConfigurationError


def load_application_config(path: Path) -> ApplicationConfig:
    """Load one explicit JSON config without environment or composition magic."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ApplicationConfig.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ConfigurationError(
            "Application configuration could not be loaded"
        ) from error
