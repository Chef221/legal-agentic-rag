"""JSONL benchmark loading and immutable evaluation report persistence."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path
import shutil

from pydantic import BaseModel, ValidationError

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas import EvaluationCase, EvaluationRunResult


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
