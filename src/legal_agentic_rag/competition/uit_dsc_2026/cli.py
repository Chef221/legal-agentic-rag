"""Lightweight competition packaging and diagnostic scoring commands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from legal_agentic_rag.competition.uit_dsc_2026.submission import (
    CodabenchSubmissionFormatter,
)
from legal_agentic_rag.competition.uit_dsc_2026.warmup_scoring import (
    CompetitionWarmupScorer,
)
from legal_agentic_rag.competition.uit_dsc_2026.development_split import (
    CompetitionDevelopmentSplitter,
)
from legal_agentic_rag.evaluation.competition_quality import (
    CompetitionBatchReadinessService,
    CompetitionWarmupScoreComparisonService,
    load_batch_readiness_policy,
    persist_batch_readiness_report,
    persist_warmup_score_comparison_report,
    require_ready_competition_batch,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import CompetitionMetricMode
from legal_agentic_rag.configuration.observability import LoggingConfig
from legal_agentic_rag.observability import configure_logging

_LOGGER = logging.getLogger(__name__)


def submission_main() -> None:
    """Create one validated official Codabench submission archive."""
    arguments = _submission_parser().parse_args()
    configure_logging(LoggingConfig())
    readiness = require_ready_competition_batch(
        questions_source=arguments.questions,
        batch_directory=arguments.batch,
        readiness_report_directory=arguments.readiness_report,
    )
    result = CodabenchSubmissionFormatter().format(
        arguments.questions,
        arguments.batch,
        arguments.output,
    )
    _LOGGER.info(
        "competition_submission_command_completed",
        extra={
            "question_count": result.question_count,
            "output_path": result.output_path,
            "archive_sha256": result.archive_sha256,
            "readiness_policy_sha256": readiness.policy_sha256,
        },
    )


def warmup_score_main() -> None:
    """Score one exact submission against official warm-up references."""
    arguments = _warmup_score_parser().parse_args()
    configure_logging(LoggingConfig())
    report = CompetitionWarmupScorer().score(
        arguments.references,
        arguments.submission,
        arguments.output,
        metric_mode=CompetitionMetricMode(arguments.metric_mode),
    )
    _LOGGER.info(
        "competition_warmup_score_completed",
        extra={
            "question_count": report.question_count,
            "meteor": report.meteor,
            "rouge_l": report.rouge_l,
            "exact_match": report.exact_match,
            "output_path": str(arguments.output),
        },
    )


def development_split_main() -> None:
    """Create one immutable leakage-aware local development split."""
    arguments = _development_split_parser().parse_args()
    configure_logging(LoggingConfig())
    manifest = CompetitionDevelopmentSplitter().split(
        arguments.train,
        arguments.holdout,
        arguments.output,
        dev_fraction=arguments.dev_fraction,
        seed=arguments.seed,
        near_duplicate_threshold=arguments.near_duplicate_threshold,
    )
    _LOGGER.info(
        "competition_development_split_completed",
        extra={
            "question_count": manifest.training_source.question_count,
            "output_path": str(arguments.output),
        },
    )


def batch_readiness_main() -> None:
    """Fail closed when a completed batch violates its explicit quality policy."""
    arguments = _batch_readiness_parser().parse_args()
    configure_logging(LoggingConfig())
    policy = load_batch_readiness_policy(arguments.policy)
    report = CompetitionBatchReadinessService().check(
        questions_source=arguments.questions,
        batch_directory=arguments.batch,
        policy=policy,
    )
    persist_batch_readiness_report(report, arguments.output)
    _LOGGER.info(
        "competition_batch_readiness_checked",
        extra={
            "record_count": report.record_count,
            "is_ready": report.is_ready,
            "violation_count": len(report.violations),
            "output_path": str(arguments.output),
        },
    )
    if not report.is_ready:
        raise ArtifactCompatibilityError(
            "Completed batch did not pass the explicit submission-readiness policy"
        )


def warmup_score_comparison_main() -> None:
    """Compare two compatible local warm-up score reports without loading models."""
    arguments = _warmup_score_comparison_parser().parse_args()
    configure_logging(LoggingConfig())
    report = CompetitionWarmupScoreComparisonService().compare(
        arguments.baseline,
        arguments.candidate,
    )
    persist_warmup_score_comparison_report(report, arguments.output)
    _LOGGER.info(
        "competition_warmup_score_comparison_completed",
        extra={
            "question_count": report.question_count,
            "meteor_mean_delta": report.meteor.mean_delta,
            "rouge_l_mean_delta": report.rouge_l.mean_delta,
            "output_path": str(arguments.output),
        },
    )


def _submission_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create UIT DSC 2026 Task 2 Codabench submission.zip"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Exact official question JSON used by the completed batch.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Completed internal batch directory created by legal-rag-batch.",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        required=True,
        help="Immutable passed report directory created by legal-rag-check-batch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output path whose filename must be submission.zip.",
    )
    return parser


def _warmup_score_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score submission.zip against official warmup.json"
    )
    parser.add_argument(
        "--metric-mode",
        choices=[mode.value for mode in CompetitionMetricMode],
        default=CompetitionMetricMode.DIAGNOSTIC.value,
        help="Diagnostic metrics or audited BTC scorer-compatible metrics.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Official warmup JSON containing question and reference answer.",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Exact submission.zip to score.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for immutable warm-up score output.",
    )
    return parser


def _development_split_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a leakage-aware UIT DSC 2026 Task 2 development split"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--holdout", type=Path, action="append", default=[],
        help="Repeat for each warm-up/public source excluded from local dev.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    return parser


def _batch_readiness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a completed competition batch against an explicit policy"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Exact official question JSON used to create the batch.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Completed internal batch directory created by legal-rag-batch.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Explicit JSON quality thresholds selected for this experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for the immutable readiness report.",
    )
    return parser


def _warmup_score_comparison_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two compatible local UIT DSC warm-up score reports"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Directory containing the control warmup_score.json report.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Directory containing the candidate warmup_score.json report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for the immutable paired score comparison report.",
    )
    return parser
