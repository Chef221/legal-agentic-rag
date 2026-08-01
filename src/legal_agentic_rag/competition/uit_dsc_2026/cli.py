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
from legal_agentic_rag.configuration.observability import LoggingConfig
from legal_agentic_rag.observability import configure_logging

_LOGGER = logging.getLogger(__name__)


def submission_main() -> None:
    """Create one validated official Codabench submission archive."""
    arguments = _submission_parser().parse_args()
    configure_logging(LoggingConfig())
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
