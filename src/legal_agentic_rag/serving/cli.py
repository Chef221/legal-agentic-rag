"""Command-line entry points for immutable build and local serving."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from legal_agentic_rag.observability import configure_logging
from legal_agentic_rag.evaluation import (
    EvaluationRunner,
    load_benchmark,
    persist_report,
)
from legal_agentic_rag.runtime import OfflineBuildRuntime, OnlineRuntimeFactory
from legal_agentic_rag.serving.api import create_app
from legal_agentic_rag.serving.config_loader import load_application_config

_LOGGER = logging.getLogger(__name__)


def build_main() -> None:
    """Build the configured immutable artifact set."""
    arguments = _parser("Build legal RAG artifacts").parse_args()
    config = load_application_config(arguments.config)
    configure_logging(config.logging)
    result = OfflineBuildRuntime(config).build()
    _LOGGER.info(
        "build_command_completed",
        extra={
            "artifact_count": len(result.artifact_manifests),
            "error_type": "-",
        },
    )


def serve_main() -> None:
    """Start the configured FastAPI and optional Gradio application."""
    arguments = _parser("Serve legal RAG API and UI").parse_args()
    config = load_application_config(arguments.config)
    app = create_app(config)
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Uvicorn is required for serving") from error
    uvicorn.run(
        app,
        host=config.serving.host,
        port=config.serving.port,
        log_level=config.logging.level.casefold(),
    )


def evaluate_main() -> None:
    """Run one labeled JSONL benchmark against loaded artifacts."""
    arguments = _evaluation_parser().parse_args()
    config = load_application_config(arguments.config)
    configure_logging(config.logging)
    cases, benchmark_hash = load_benchmark(arguments.benchmark)
    runtime = OnlineRuntimeFactory(config).build()
    result = EvaluationRunner(runtime, config.evaluation).run(
        cases,
        benchmark_name=arguments.benchmark.name,
        benchmark_sha256=benchmark_hash,
    )
    persist_report(result, arguments.output)
    _LOGGER.info(
        "evaluation_command_completed",
        extra={
            "case_count": result.summary.case_count,
            "failed_case_count": result.summary.failed_case_count,
            "output_path": str(arguments.output),
        },
    )


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an ApplicationConfig JSON file.",
    )
    return parser


def _evaluation_parser() -> argparse.ArgumentParser:
    parser = _parser("Evaluate legal RAG against a labeled JSONL benchmark")
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to labeled EvaluationCase JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for immutable evaluation reports.",
    )
    return parser
