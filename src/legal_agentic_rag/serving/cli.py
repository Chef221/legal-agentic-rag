"""Command-line entry points for validation, evaluation, and local serving."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from legal_agentic_rag.competition.uit_dsc_2026 import (
    CompetitionBatchRunner,
    render_competition_answer,
)
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.observability import configure_logging
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.evaluation import (
    EvaluationComparisonService,
    EvaluationRunner,
    RetrievalDiagnosticsRunner,
    StandardGenerationEvaluator,
    evaluation_runtime_provenance,
    load_benchmark_bundle,
    load_comparison_config,
    load_evaluation_summary,
    persist_comparison_report,
    persist_report,
)
from legal_agentic_rag.indexing.vector import prepare_vector_serving_metadata
from legal_agentic_rag.runtime import (
    ArtifactSetValidator,
    OnlineRuntimeFactory,
    persist_build_validation_report,
)
from legal_agentic_rag.runtime.competition_offline import (
    CompetitionOfflineBuildRuntime,
)
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.schemas import ArtifactType, CompetitionBuildStage
from legal_agentic_rag.serving.api import create_app
from legal_agentic_rag.serving.config_loader import load_application_config
from legal_agentic_rag.serving.query_service import ServingService

_LOGGER = logging.getLogger(__name__)


def competition_build_main() -> None:
    """Build or resume official UIT DSC corpus artifacts."""
    arguments = _competition_build_parser().parse_args()
    config = load_application_config(arguments.config)
    configure_logging(config.logging)
    result = CompetitionOfflineBuildRuntime(config, arguments.source).build(
        through=arguments.through
    )
    _LOGGER.info(
        "competition_build_command_completed",
        extra={
            "stage_count": len(result.completed_stages),
            "through_stage": arguments.through.value,
            "resumed": result.resumed,
            "is_valid": (
                result.validation_report.is_valid
                if result.validation_report is not None
                else None
            ),
        },
    )


def competition_batch_main() -> None:
    """Run or resume submission-neutral official question inference."""
    arguments = _competition_batch_parser().parse_args()
    config = load_application_config(arguments.config)
    configure_logging(config.logging)
    runtime = OnlineRuntimeFactory(config).build()
    service = ServingService(runtime, config.serving, config.online)
    manifest = CompetitionBatchRunner(
        service,
        application_config_hash=canonical_sha256(config),
    ).run(arguments.questions, arguments.output)
    _LOGGER.info(
        "competition_batch_command_completed",
        extra={
            "question_count": manifest.record_count,
            "output_path": str(arguments.output),
        },
    )


def serve_main() -> None:
    """Start the configured FastAPI API and optional diagnostic UI."""
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
    arguments = _validation_parser().parse_args()
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
    if arguments.persist:
        persist_build_validation_report(
            report,
            config.artifacts.root_path,
            config.build_validation.report_filename,
        )


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
    cases, benchmark_manifest, benchmark_manifest_hash = (
        load_benchmark_bundle(
            arguments.benchmark,
            arguments.benchmark_manifest,
        )
    )
    runtime = OnlineRuntimeFactory(config).build()
    runtime_hash, provenance = evaluation_runtime_provenance(config)
    result = EvaluationRunner(
        runtime,
        config.evaluation,
        generation_evaluator=StandardGenerationEvaluator(
            answer_renderer=render_competition_answer,
        ),
        runtime_config_sha256=runtime_hash,
        component_provenance=provenance,
    ).run(
        cases,
        benchmark_manifest=benchmark_manifest,
        benchmark_manifest_sha256=benchmark_manifest_hash,
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


def compare_main() -> None:
    """Compare immutable evaluation reports under one explicit policy."""
    arguments = _comparison_parser().parse_args()
    config = load_comparison_config(arguments.comparison)
    summaries = {
        candidate.candidate_id: load_evaluation_summary(
            candidate.report_directory
        )
        for candidate in config.candidates
    }
    report = EvaluationComparisonService().compare(config, summaries)
    persist_comparison_report(report, arguments.output)
    _LOGGER.info(
        "evaluation_comparison_completed",
        extra={
            "candidate_count": len(report.candidates),
            "eligible_candidate_count": sum(
                candidate.eligible for candidate in report.candidates
            ),
            "selected_candidate_id": report.selected_candidate_id or "-",
            "output_path": str(arguments.output),
        },
    )


def retrieval_diagnostics_main() -> None:
    """Run non-gold BM25/dense/hybrid diagnostics over official questions."""
    arguments = _retrieval_diagnostics_parser().parse_args()
    config = load_application_config(arguments.config)
    configure_logging(config.logging)
    runtime = OnlineRuntimeFactory(config).build()
    report = RetrievalDiagnosticsRunner(
        runtime,
        application_config_sha256=canonical_sha256(config),
    ).run(
        arguments.questions,
        arguments.output,
        top_k=arguments.top_k,
        candidate_k=arguments.candidate_k,
        max_cases=arguments.max_cases,
        include_reranker=arguments.include_reranker,
        low_document_diversity_threshold=(
            arguments.low_document_diversity_threshold
        ),
        low_answer_term_coverage_threshold=(
            arguments.low_answer_term_coverage_threshold
        ),
    )
    _LOGGER.info(
        "retrieval_diagnostics_command_completed",
        extra={
            "question_count": report.question_count,
            "failed_case_count": report.failed_case_count,
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
        "--benchmark-manifest",
        type=Path,
        required=True,
        help="Path to the benchmark identity and label-provenance manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for immutable evaluation reports.",
    )
    return parser


def _validation_parser() -> argparse.ArgumentParser:
    parser = _parser("Validate legal RAG artifacts")
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Persist the immutable report using build_validation.report_filename; "
            "an existing report is never overwritten."
        ),
    )
    return parser


def _competition_build_parser() -> argparse.ArgumentParser:
    parser = _parser("Build official UIT DSC 2026 Task 2 corpus artifacts")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Official selected-contexts ZIP or extracted directory.",
    )
    parser.add_argument(
        "--through",
        type=CompetitionBuildStage,
        choices=list(CompetitionBuildStage),
        default=CompetitionBuildStage.VALIDATION,
        help=(
            "Last durable stage to build. Use document_processing to stop "
            "after parser/chunker artifacts; default: validation."
        ),
    )
    return parser


def _competition_batch_parser() -> argparse.ArgumentParser:
    parser = _parser("Run official UIT DSC 2026 Task 2 question batch")
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Official warm-up, public-test, or private-test question JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or compatible resumable internal batch directory.",
    )
    return parser


def _comparison_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare reproducible legal RAG evaluation reports"
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help="Path to typed EvaluationComparisonConfig JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for the immutable comparison report.",
    )
    return parser


def _retrieval_diagnostics_parser() -> argparse.ArgumentParser:
    parser = _parser("Diagnose BM25, dense, and hybrid retrieval without gold labels")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--include-reranker",
        action="store_true",
        help="Also run hybrid_rerank for a retrieval-only M44.3 ablation.",
    )
    parser.add_argument(
        "--low-document-diversity-threshold", type=float, default=0.4
    )
    parser.add_argument(
        "--low-answer-term-coverage-threshold", type=float, default=0.25
    )
    return parser
