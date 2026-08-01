"""Answer-only local scoring for official warm-up references."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from typing import Callable

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.competition.uit_dsc_2026.submission import (
    load_submission_archive,
)
from legal_agentic_rag.evaluation import score_text_answer
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.schemas import (
    CompetitionWarmupCaseScore,
    CompetitionWarmupScoreReport,
)

WARMUP_SCORE_REPORT_FILENAME = "warmup_score.json"
_DIAGNOSTIC_WARNING = (
    "competition_text_metrics_are_diagnostic_not_official_equivalent"
)


class CompetitionWarmupScorer:
    """Validate and score one complete submission against official references."""

    def __init__(
        self,
        loader: UitDsc2026DataLoader | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._loader = loader or UitDsc2026DataLoader()
        self._clock = clock or (lambda: datetime.now(UTC))

    def score(
        self,
        reference_source: Path,
        submission_archive: Path,
        output_directory: Path,
    ) -> CompetitionWarmupScoreReport:
        """Persist a content-free immutable diagnostic scoring report."""
        reference_hash = self._sha256_file(reference_source)
        references = self._loader.load_questions(
            reference_source,
            require_reference_answers=True,
        )
        if self._sha256_file(reference_source) != reference_hash:
            raise DataValidationError("Warm-up reference source changed while loading")
        submissions, submission_payload = load_submission_archive(
            submission_archive
        )
        reference_ids = [reference.question_id for reference in references]
        submission_ids = [submission.id for submission in submissions]
        if submission_ids != reference_ids:
            raise DataValidationError(
                "Submission IDs must exactly match warm-up references in source order"
            )

        case_scores: list[CompetitionWarmupCaseScore] = []
        for reference, submission in zip(references, submissions, strict=True):
            if reference.reference_answer is None:
                raise DataValidationError("Warm-up reference answer is missing")
            metrics = score_text_answer(
                submission.answer,
                reference.reference_answer,
            )
            if (
                metrics.exact_match is None
                or metrics.meteor is None
                or metrics.rouge_l is None
            ):
                raise DataValidationError("Warm-up text metrics are incomplete")
            case_scores.append(
                CompetitionWarmupCaseScore(
                    question_id=reference.question_id,
                    exact_match=metrics.exact_match,
                    meteor=metrics.meteor,
                    rouge_l=metrics.rouge_l,
                )
            )

        report = CompetitionWarmupScoreReport(
            created_at=self._clock(),
            code_version=__version__,
            reference_source_sha256=reference_hash,
            submission_archive_sha256=self._sha256_file(submission_archive),
            submission_json_sha256=sha256(submission_payload).hexdigest(),
            question_count=len(case_scores),
            exact_match=fmean(score.exact_match for score in case_scores),
            meteor=fmean(score.meteor for score in case_scores),
            rouge_l=fmean(score.rouge_l for score in case_scores),
            cases=case_scores,
            warnings=[_DIAGNOSTIC_WARNING],
        )
        self._persist_report(output_directory, report)
        return report

    @staticmethod
    def _persist_report(
        output_directory: Path,
        report: CompetitionWarmupScoreReport,
    ) -> None:
        output = output_directory.resolve()
        if output.exists():
            raise ArtifactCompatibilityError(
                "Warm-up score output already exists"
            )
        temporary = output.with_name(f".{output.name}.tmp")
        if temporary.exists():
            raise ArtifactCompatibilityError(
                "Warm-up score temporary output already exists"
            )
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            (temporary / WARMUP_SCORE_REPORT_FILENAME).write_text(
                json.dumps(
                    report.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(output)
        except OSError as error:
            report_path = temporary / WARMUP_SCORE_REPORT_FILENAME
            report_path.unlink(missing_ok=True)
            try:
                temporary.rmdir()
            except OSError:
                pass
            raise ArtifactCompatibilityError(
                "Warm-up score report could not be persisted"
            ) from error

    @staticmethod
    def _sha256_file(path: Path) -> str:
        try:
            digest = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except OSError as error:
            raise ArtifactCompatibilityError(
                "Warm-up scoring input is unreadable"
            ) from error
