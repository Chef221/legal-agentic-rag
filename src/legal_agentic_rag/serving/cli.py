"""Command-line entry points for immutable build and local serving."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from legal_agentic_rag.observability import configure_logging
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.evaluation import (
    EvaluationRunner,
    load_benchmark,
    persist_report,
)
from legal_agentic_rag.indexing.vector import prepare_vector_serving_metadata
from legal_agentic_rag.runtime import (
    ArtifactSetValidator,
    OfflineBuildRuntime,
    OnlineRuntimeFactory,
)
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.schemas import ArtifactType
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


def validate_main() -> None:
    """Revalidate an existing artifact set without rebuilding or changing it."""
    arguments = _parser("Validate legal RAG artifacts").parse_args()
    config = load_application_config(arguments.config)
    configure_logging(config.logging)
    report = ArtifactSetValidator(
        config.artifacts,
        config.build_validation,
    ).validate()
    _LOGGER.info(
        "validation_command_completed",
        extra={
            "artifact_count": len(report.artifact_results),
            "is_full_corpus": report.is_full_corpus,
            "is_valid": report.is_valid,
        },
    )
    if not report.is_valid:
        raise ArtifactCompatibilityError("Artifact set failed validation")


def prepare_serving_main() -> None:
    """Prepare immutable online metadata from existing validated artifacts."""
    arguments = _parser("Prepare legal RAG serving metadata").parse_args()
    config = load_application_config(arguments.config)
    configure_logging(config.logging)
    vector_directory = config.artifacts.directory("vector_directory")
    manifest = load_artifact_manifest(
        vector_directory,
        expected_type=ArtifactType.VECTOR_INDEX,
    )
    result = prepare_vector_serving_metadata(
        vector_directory=vector_directory,
        destination=config.artifacts.directory("vector_serving_directory"),
        vector_manifest=manifest,
        batch_size=(
            config.online.vector_runtime.serving_metadata_build_batch_size
        ),
        progress_interval_records=(
            config.online.vector_runtime.load_progress_interval_records
        ),
    )
    _LOGGER.info(
        "prepare_serving_command_completed",
        extra={
            "artifact_count": 1,
            "chunk_count": result.record_count,
            "backend": result.backend,
        },
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
