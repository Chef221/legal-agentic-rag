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


@pytest.fixture
def load_html_fixture() -> Any:
    """Return a loader for small legal HTML cleaning fixtures."""
    fixture_root = Path(__file__).parent / "fixtures" / "html_samples"

    def load(filename: str) -> str:
        return (fixture_root / filename).read_text(encoding="utf-8")

    return load


@pytest.fixture
def load_clean_text_fixture() -> Any:
    """Return a loader for small cleaned legal text fixtures."""
    fixture_root = Path(__file__).parent / "fixtures" / "clean_text"

    def load(filename: str) -> str:
        return (fixture_root / filename).read_text(encoding="utf-8")

    return load
