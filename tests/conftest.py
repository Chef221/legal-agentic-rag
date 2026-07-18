"""Shared fixtures for deterministic Milestone 1 tests."""

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def load_schema_sample() -> Any:
    """Return a loader for small unified-schema JSON fixtures."""
    fixture_root = Path(__file__).parent / "fixtures" / "schema_samples"

    def load(filename: str) -> dict[str, object]:
        return json.loads((fixture_root / filename).read_text(encoding="utf-8"))

    return load
