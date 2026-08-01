"""Fail-closed Codabench submission packaging for UIT DSC 2026 Task 2."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from pydantic import ValidationError

from legal_agentic_rag.competition.uit_dsc_2026.batch_inference import (
    BATCH_MANIFEST_FILENAME,
    BATCH_RECORDS_FILENAME,
)
from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
    render_competition_answer,
)
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.schemas import (
    CompetitionBatchManifest,
    CompetitionBatchRecord,
    CompetitionSubmissionItem,
    CompetitionSubmissionResult,
)

SUBMISSION_ARCHIVE_FILENAME = "submission.zip"
SUBMISSION_JSON_FILENAME = "submission.json"


def load_submission_archive(
    path: Path,
) -> tuple[list[CompetitionSubmissionItem], bytes]:
    """Load one exact official submission ZIP without accepting extra members."""
    if path.name != SUBMISSION_ARCHIVE_FILENAME:
        raise DataValidationError("Codabench input filename must be submission.zip")
    return _read_submission_archive(path)


class CodabenchSubmissionFormatter:
    """Convert one complete internal batch into the exact organizer ZIP contract."""

    def __init__(self, loader: UitDsc2026DataLoader | None = None) -> None:
        self._loader = loader or UitDsc2026DataLoader()

    def format(
        self,
        questions_source: Path,
        batch_directory: Path,
        output_path: Path,
    ) -> CompetitionSubmissionResult:
        """Validate lineage and write a deterministic one-member submission ZIP."""
        if output_path.name != SUBMISSION_ARCHIVE_FILENAME:
            raise DataValidationError(
                "Codabench output filename must be submission.zip"
            )
        if output_path.exists():
            raise ArtifactCompatibilityError("Submission output already exists")

        question_source_hash = self._sha256_file(questions_source)
        questions = self._loader.load_questions(
            questions_source,
            require_reference_answers=False,
        )
        if self._sha256_file(questions_source) != question_source_hash:
            raise DataValidationError("Official question source changed while loading")

        batch = batch_directory.resolve()
        records_path = batch / BATCH_RECORDS_FILENAME
        manifest = self._load_manifest(batch / BATCH_MANIFEST_FILENAME)
        records = self._load_records(records_path)
        question_ids = [question.question_id for question in questions]
        self._validate_batch(
            records=records,
            question_ids=question_ids,
            manifest=manifest,
            question_source_hash=question_source_hash,
            records_path=records_path,
        )

        items = [
            CompetitionSubmissionItem(
                id=record.question_id,
                answer=render_competition_answer(record.response),
            )
            for record in records
        ]
        payload = (
            json.dumps(
                {item.id: {"answer": item.answer} for item in items},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ArtifactCompatibilityError(
                "Submission destination could not be prepared"
            ) from error
        temporary_path = self._write_archive_atomic(output_path, payload)
        try:
            self._validate_archive(temporary_path, question_ids)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        try:
            temporary_path.replace(output_path)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise ArtifactCompatibilityError(
                "Submission archive could not be published"
            ) from error

        return CompetitionSubmissionResult(
            output_path=str(output_path.resolve()),
            question_count=len(items),
            submission_json_sha256=sha256(payload).hexdigest(),
            archive_sha256=self._sha256_file(output_path),
        )

    @staticmethod
    def _validate_batch(
        *,
        records: list[CompetitionBatchRecord],
        question_ids: list[str],
        manifest: CompetitionBatchManifest,
        question_source_hash: str,
        records_path: Path,
    ) -> None:
        record_ids = [record.question_id for record in records]
        if record_ids != question_ids:
            raise ArtifactCompatibilityError(
                "Batch must contain every official question exactly once "
                "in source order"
            )
        if (
            manifest.question_source_sha256 != question_source_hash
            or manifest.output_format != "internal_answer_response_jsonl_v1"
            or manifest.record_count != len(records)
            or manifest.records_sha256
            != CodabenchSubmissionFormatter._sha256_file(records_path)
        ):
            raise ArtifactCompatibilityError(
                "Completed competition batch is incompatible with submission input"
            )

    @staticmethod
    def _write_archive_atomic(output_path: Path, payload: bytes) -> Path:
        try:
            with NamedTemporaryFile(
                mode="wb",
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
            member = ZipInfo(SUBMISSION_JSON_FILENAME, (1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            with ZipFile(temporary_path, mode="w") as archive:
                archive.writestr(member, payload, compresslevel=9)
            return temporary_path
        except OSError as error:
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)
            raise ArtifactCompatibilityError(
                "Submission archive could not be persisted"
            ) from error

    @staticmethod
    def _validate_archive(path: Path, question_ids: list[str]) -> None:
        items, _ = _read_submission_archive(path)
        if [item.id for item in items] != question_ids:
            raise DataValidationError(
                "Generated submission IDs do not exactly match official questions"
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
            raise ArtifactCompatibilityError(
                "Submission input is unreadable"
            ) from error


def _read_submission_archive(
    path: Path,
) -> tuple[list[CompetitionSubmissionItem], bytes]:
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != SUBMISSION_JSON_FILENAME:
                raise DataValidationError(
                    "Submission ZIP must contain only submission.json"
                )
            if members[0].is_dir() or members[0].flag_bits & 0x1:
                raise DataValidationError(
                    "Submission JSON member must be a regular unencrypted file"
                )
            payload_bytes = archive.read(members[0])
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_unique_submission_object,
        )
        if not isinstance(payload, dict):
            raise DataValidationError(
                "Submission JSON root must be an object keyed by question ID"
            )
        items: list[CompetitionSubmissionItem] = []
        for question_id, value in payload.items():
            if not isinstance(value, dict):
                raise DataValidationError(
                    "Each submission question ID must map to an answer object"
                )
            if set(value) != {"answer"}:
                raise DataValidationError(
                    "Each submission answer object must contain only 'answer'"
                )
            items.append(
                CompetitionSubmissionItem.model_validate(
                    {"id": question_id, **value}
                )
            )
    except DataValidationError:
        raise
    except (
        BadZipFile,
        KeyError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
    ) as error:
        raise DataValidationError("Submission archive is invalid") from error
    if not items:
        raise DataValidationError("Submission JSON object must not be empty")
    identities = [item.id for item in items]
    if len(identities) != len(set(identities)):
        raise DataValidationError("Submission question IDs must be unique")
    return items, payload_bytes


def _unique_submission_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(
                f"Submission JSON contains duplicate key '{key}'"
            )
        result[key] = value
    return result
