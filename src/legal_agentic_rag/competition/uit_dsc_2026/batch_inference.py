"""Checkpointed, submission-neutral inference over official questions."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from typing import Callable, Protocol

from pydantic import ValidationError

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.schemas import (
    AnswerResponse,
    CompetitionBatchManifest,
    CompetitionBatchRecord,
    CompetitionBatchState,
    LegalQuestionRequest,
)

BATCH_RECORDS_FILENAME = "results.jsonl"
BATCH_STATE_FILENAME = "batch_state.json"
BATCH_MANIFEST_FILENAME = "manifest.json"
_PROGRESS_INTERVAL = 25
_LOGGER = logging.getLogger(__name__)


class CompetitionQuestionAnswerer(Protocol):
    """Narrow serving boundary needed by batch inference."""

    def answer(self, request: LegalQuestionRequest) -> AnswerResponse:
        """Return one grounded public answer contract."""
        ...


class CompetitionBatchRunner:
    """Run or resume one exact ordered official question batch."""

    def __init__(
        self,
        answerer: CompetitionQuestionAnswerer,
        *,
        application_config_hash: str,
        loader: UitDsc2026DataLoader | None = None,
        clock: Callable[[], datetime] | None = None,
        progress_interval: int = _PROGRESS_INTERVAL,
    ) -> None:
        if progress_interval <= 0:
            raise ValueError("progress_interval must be positive")
        self._answerer = answerer
        self._config_hash = application_config_hash
        self._loader = loader or UitDsc2026DataLoader()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._progress_interval = progress_interval

    def run(
        self,
        questions_source: Path,
        output_directory: Path,
    ) -> CompetitionBatchManifest:
        """Persist every answer once and publish a completeness manifest."""
        source_hash = self._sha256_file(questions_source)
        questions = self._loader.load_questions(
            questions_source,
            require_reference_answers=False,
        )
        if self._sha256_file(questions_source) != source_hash:
            raise DataValidationError("Official question source changed while loading")
        question_ids = [question.question_id for question in questions]
        output = output_directory.resolve()
        output.mkdir(parents=True, exist_ok=True)
        records_path = output / BATCH_RECORDS_FILENAME
        state_path = output / BATCH_STATE_FILENAME
        manifest_path = output / BATCH_MANIFEST_FILENAME

        if manifest_path.exists():
            manifest = self._load_manifest(manifest_path)
            records = self._load_records(records_path)
            self._validate_complete(
                records,
                question_ids,
                manifest,
                source_hash,
                records_path,
            )
            return manifest

        state = self._prepare_state(
            state_path,
            source_hash=source_hash,
            question_count=len(questions),
            output=output,
        )
        records = self._load_records(records_path) if records_path.exists() else []
        self._validate_prefix(records, question_ids)
        _LOGGER.info(
            "competition_batch_started",
            extra={
                "question_count": len(questions),
                "completed_question_count": len(records),
            },
        )
        record_ids = [record.question_id for record in records]
        if state.completed_question_ids != record_ids:
            state = state.model_copy(
                update={
                    "completed_question_ids": record_ids,
                    "updated_at": self._clock(),
                }
            )
            self._write_json_atomic(state_path, state.model_dump(mode="json"))

        with records_path.open("a", encoding="utf-8", newline="\n") as stream:
            for question in questions[len(records) :]:
                request = LegalQuestionRequest(question=question.question)
                response = self._answerer.answer(request)
                if response.question != request.question:
                    raise DataValidationError(
                        "Batch answer question differs from official input"
                    )
                record = CompetitionBatchRecord(
                    question_id=question.question_id,
                    response=response,
                )
                stream.write(record.model_dump_json())
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(record)
                state = state.model_copy(
                    update={
                        "completed_question_ids": [
                            *state.completed_question_ids,
                            question.question_id,
                        ],
                        "updated_at": self._clock(),
                    }
                )
                self._write_json_atomic(
                    state_path, state.model_dump(mode="json")
                )
                if (
                    len(records) % self._progress_interval == 0
                    or len(records) == len(questions)
                ):
                    _LOGGER.info(
                        "competition_batch_progress",
                        extra={
                            "question_count": len(questions),
                            "completed_question_count": len(records),
                        },
                    )

        self._validate_prefix(records, question_ids)
        if len(records) != len(question_ids):
            raise DataValidationError("Batch inference did not produce every answer")
        manifest = CompetitionBatchManifest(
            question_source_sha256=source_hash,
            application_config_hash=self._config_hash,
            code_version=__version__,
            created_at=self._clock(),
            record_count=len(records),
            records_sha256=self._sha256_file(records_path),
        )
        self._write_json_exclusive(manifest_path, manifest.model_dump(mode="json"))
        _LOGGER.info(
            "competition_batch_completed",
            extra={
                "question_count": len(questions),
                "completed_question_count": len(records),
            },
        )
        return manifest

    def _prepare_state(
        self,
        path: Path,
        *,
        source_hash: str,
        question_count: int,
        output: Path,
    ) -> CompetitionBatchState:
        if path.exists():
            state = self._load_state(path)
            if (
                state.question_source_sha256 != source_hash
                or state.application_config_hash != self._config_hash
                or state.code_version != __version__
                or state.question_count != question_count
            ):
                raise ArtifactCompatibilityError(
                    "Competition batch recovery identity is incompatible"
                )
            return state
        unexpected = [
            path
            for path in output.iterdir()
            if path.name not in {BATCH_RECORDS_FILENAME}
        ]
        if unexpected or (output / BATCH_RECORDS_FILENAME).exists():
            raise ArtifactCompatibilityError(
                "Batch output has no compatible recovery state"
            )
        now = self._clock()
        state = CompetitionBatchState(
            question_source_sha256=source_hash,
            application_config_hash=self._config_hash,
            code_version=__version__,
            question_count=question_count,
            created_at=now,
            updated_at=now,
        )
        self._write_json_atomic(path, state.model_dump(mode="json"))
        return state

    def _validate_complete(
        self,
        records: list[CompetitionBatchRecord],
        question_ids: list[str],
        manifest: CompetitionBatchManifest,
        source_hash: str,
        records_path: Path,
    ) -> None:
        if (
            manifest.question_source_sha256 != source_hash
            or manifest.application_config_hash != self._config_hash
            or manifest.code_version != __version__
        ):
            raise ArtifactCompatibilityError(
                "Completed competition batch is incompatible"
            )
        self._validate_prefix(records, question_ids)
        if (
            len(records) != len(question_ids)
            or manifest.record_count != len(records)
            or manifest.records_sha256 != self._sha256_file(records_path)
        ):
            raise ArtifactCompatibilityError(
                "Completed competition batch is incompatible"
            )

    @staticmethod
    def _validate_prefix(
        records: list[CompetitionBatchRecord], question_ids: list[str]
    ) -> None:
        record_ids = [record.question_id for record in records]
        if record_ids != question_ids[: len(record_ids)]:
            raise ArtifactCompatibilityError(
                "Batch records are not an ordered prefix of official questions"
            )

    @staticmethod
    def _load_records(path: Path) -> list[CompetitionBatchRecord]:
        records: list[CompetitionBatchRecord] = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        raise ArtifactCompatibilityError(
                            "Batch checkpoint contains a blank record"
                        )
                    records.append(CompetitionBatchRecord.model_validate_json(line))
        except ArtifactCompatibilityError:
            raise
        except (OSError, ValidationError) as error:
            raise ArtifactCompatibilityError(
                "Batch checkpoint is missing or invalid"
            ) from error
        return records

    @staticmethod
    def _load_state(path: Path) -> CompetitionBatchState:
        try:
            return CompetitionBatchState.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ArtifactCompatibilityError("Batch state is invalid") from error

    @staticmethod
    def _load_manifest(path: Path) -> CompetitionBatchManifest:
        try:
            return CompetitionBatchManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ArtifactCompatibilityError("Batch manifest is invalid") from error

    @staticmethod
    def _sha256_file(path: Path) -> str:
        try:
            digest = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except OSError as error:
            raise ArtifactCompatibilityError("Batch source is unreadable") from error

    @staticmethod
    def _write_json_atomic(path: Path, payload: object) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ArtifactCompatibilityError("Batch state could not be persisted") from error

    @classmethod
    def _write_json_exclusive(cls, path: Path, payload: object) -> None:
        if path.exists():
            raise ArtifactCompatibilityError("Batch manifest already exists")
        cls._write_json_atomic(path, payload)
