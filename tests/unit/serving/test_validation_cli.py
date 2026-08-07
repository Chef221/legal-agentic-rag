"""Tests for immutable validation-report CLI persistence."""

from pathlib import Path

from legal_agentic_rag.serving.cli import _validation_parser


def test_validation_parser_defaults_to_read_only() -> None:
    arguments = _validation_parser().parse_args(["--config", "config.json"])

    assert arguments.config == Path("config.json")
    assert arguments.persist is False


def test_validation_parser_accepts_explicit_persistence() -> None:
    arguments = _validation_parser().parse_args(
        ["--config", "config.json", "--persist"]
    )

    assert arguments.persist is True
