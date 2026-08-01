"""JSONL benchmark loading and immutable evaluation report persistence."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path
import shutil

from pydantic import BaseModel, ValidationError

from legal_agentic_rag.configuration import EvaluationComparisonConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    ConfigurationError,
    DataValidationError,
)
from legal_agentic_rag.schemas import (
    EvaluationBenchmarkManifest,
    EvaluationCase,
    EvaluationComparisonReport,
    EvaluationRunResult,
    EvaluationSummary,
)


def load_benchmark(path: Path) -> tuple[list[EvaluationCase], str]:
    """Load a UTF-8 JSONL benchmark and return its SHA-256 identity."""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DataValidationError("Evaluation benchmark cannot be read") from error
    cases: list[EvaluationCase] = []
    try:
        for line in payload.decode("utf-8").splitlines():
            if line.strip():
                cases.append(EvaluationCase.model_validate_json(line))
    except (UnicodeError, ValidationError, ValueError) as error:
        raise DataValidationError("Evaluation benchmark is invalid") from error
    if not cases:
        raise DataValidationError("Evaluation benchmark must not be empty")
    identities = [case.case_id for case in cases]
    if len(identities) != len(set(identities)):
        raise DataValidationError("Evaluation case IDs must be unique")
    return cases, sha256(payload).hexdigest()


def load_benchmark_bundle(
    benchmark_path: Path,
    manifest_path: Path,
) -> tuple[list[EvaluationCase], EvaluationBenchmarkManifest, str]:
    """Load and cross-check benchmark bytes against their typed manifest."""
    cases, benchmark_digest = load_benchmark(benchmark_path)
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = EvaluationBenchmarkManifest.model_validate_json(
            manifest_payload
        )
    except (OSError, ValidationError, ValueError) as error:
        raise DataValidationError(
            "Evaluation benchmark manifest is invalid"
        ) from error
    if manifest.benchmark_sha256 != benchmark_digest:
        raise DataValidationError(
            "Evaluation benchmark does not match manifest SHA-256"
        )
    if manifest.case_count != len(cases):
        raise DataValidationError(
            "Evaluation benchmark does not match manifest case count"
        )
    granularities = {case.target_granularity for case in cases}
    if set(manifest.target_granularities) != granularities:
        raise DataValidationError(
            "Evaluation benchmark does not match manifest granularities"
        )
    return cases, manifest, sha256(manifest_payload).hexdigest()


def persist_report(result: EvaluationRunResult, destination: Path) -> Path:
    """Persist summary, cases, and failures without overwriting a prior run."""
    if destination.exists():
        raise ArtifactCompatibilityError(
            "Evaluation report destination already exists"
        )
    try:
        destination.mkdir(parents=True)
        (destination / "summary.json").write_text(
            json.dumps(
                result.summary.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_jsonl(destination / "cases.jsonl", result.cases)
        _write_jsonl(
            destination / "errors.jsonl",
            [case for case in result.cases if not case.success],
        )
    except OSError as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise ArtifactCompatibilityError(
            "Evaluation report could not be persisted"
        ) from error
    return destination


def load_evaluation_summary(directory: Path) -> EvaluationSummary:
    """Load one immutable evaluation summary for cross-run comparison."""
    path = directory / "summary.json"
    try:
        return EvaluationSummary.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise DataValidationError(
            "Evaluation summary cannot be loaded"
        ) from error


def load_comparison_config(path: Path) -> EvaluationComparisonConfig:
    """Load comparison policy and resolve report paths relative to its file."""
    try:
        config = EvaluationComparisonConfig.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ConfigurationError(
            "Evaluation comparison configuration is invalid"
        ) from error
    candidates = [
        candidate.model_copy(
            update={
                "report_directory": (
                    candidate.report_directory
                    if candidate.report_directory.is_absolute()
                    else (path.parent / candidate.report_directory).resolve()
                )
            }
        )
        for candidate in config.candidates
    ]
    return config.model_copy(update={"candidates": candidates})


def persist_comparison_report(
    report: EvaluationComparisonReport,
    destination: Path,
) -> Path:
    """Persist one comparison immutably without overwriting prior evidence."""
    if destination.exists():
        raise ArtifactCompatibilityError(
            "Evaluation comparison destination already exists"
        )
    try:
        destination.mkdir(parents=True)
        (destination / "comparison.json").write_text(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise ArtifactCompatibilityError(
            "Evaluation comparison could not be persisted"
        ) from error
    return destination


def _write_jsonl(path: Path, values: Sequence[BaseModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    value.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
