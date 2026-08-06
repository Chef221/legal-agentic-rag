"""Strict local loaders for currently documented UIT DSC 2026 formats."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from pydantic import ValidationError

from legal_agentic_rag.competition.uit_dsc_2026 import raw_schema
from legal_agentic_rag.exceptions import DatasetSchemaError
from legal_agentic_rag.schemas.competition import (
    CompetitionContext,
    CompetitionQuestion,
)

_CONTEXT_PATTERN = "context_*.json"


@dataclass(frozen=True)
class ContextSourceIdentity:
    """Deterministic identity of one official context source."""

    source_kind: Literal["directory", "zip"]
    revision: str
    member_count: int


class UitDsc2026DataLoader:
    """Read official local JSON without leaking raw names into core modules."""

    def load_questions(
        self,
        source: Path,
        *,
        require_reference_answers: bool,
    ) -> list[CompetitionQuestion]:
        """Load an ordered question-ID mapping from one official JSON file."""
        payload = self._read_json(source)
        if not isinstance(payload, Mapping):
            raise DatasetSchemaError("Competition question root must be an object")

        records: list[CompetitionQuestion] = []
        for question_id, raw_record in payload.items():
            if not isinstance(question_id, str) or not question_id.strip():
                raise DatasetSchemaError("Competition question ID must be text")
            if not isinstance(raw_record, Mapping):
                raise DatasetSchemaError(
                    f"Competition question '{question_id}' must be an object"
                )
            keys = set(raw_record)
            allowed = raw_schema.QUESTION_FIELDS
            if keys - allowed:
                raise DatasetSchemaError(
                    f"Competition question '{question_id}' has unknown fields"
                )
            if raw_schema.QUESTION_FIELD not in raw_record:
                raise DatasetSchemaError(
                    f"Competition question '{question_id}' is missing question"
                )
            answer = raw_record.get(raw_schema.ANSWER_FIELD)
            if require_reference_answers and answer is None:
                raise DatasetSchemaError(
                    f"Competition question '{question_id}' is missing answer"
                )
            try:
                records.append(
                    CompetitionQuestion(
                        question_id=question_id,
                        question=raw_record[raw_schema.QUESTION_FIELD],
                        reference_answer=answer,
                    )
                )
            except ValidationError as error:
                raise DatasetSchemaError(
                    f"Competition question '{question_id}' is invalid"
                ) from error
        if not records:
            raise DatasetSchemaError("Competition question file must not be empty")
        return records

    def load_contexts(self, source: Path) -> list[CompetitionContext]:
        """Load one documented context object per file from a directory or ZIP."""
        return list(self.iter_contexts(source))

    def iter_contexts(self, source: Path) -> Iterator[CompetitionContext]:
        """Yield official contexts in deterministic filename order."""
        if source.is_dir():
            raw_records = self._iter_directory_contexts(source)
        elif source.is_file() and source.suffix.casefold() == ".zip":
            raw_records = self._iter_zip_contexts(source)
        else:
            raise DatasetSchemaError(
                "Competition context source must be a directory or ZIP file"
            )
        seen_ids: set[str] = set()
        yielded = False
        for source_name, raw_record in raw_records:
            record = self._context_record(raw_record, source_name)
            if record.context_id in seen_ids:
                raise DatasetSchemaError(
                    f"Duplicate competition context ID '{record.context_id}'"
            )
            seen_ids.add(record.context_id)
            yielded = True
            yield record
        if not yielded:
            raise DatasetSchemaError("No context_*.json files were found")

    def inspect_context_source(self, source: Path) -> ContextSourceIdentity:
        """Hash ordered official filenames and exact bytes without extracting ZIPs."""
        digest = sha256()
        count = 0
        if source.is_dir():
            members = self._directory_paths(source)
            kind: Literal["directory", "zip"] = "directory"
            try:
                for path in members:
                    self._update_source_digest(digest, path.name, path.read_bytes())
                    count += 1
            except OSError as error:
                raise DatasetSchemaError(
                    "Competition context source cannot be read"
                ) from error
        elif source.is_file() and source.suffix.casefold() == ".zip":
            kind = "zip"
            try:
                with ZipFile(source) as archive:
                    members = self._zip_members(archive)
                    for member in members:
                        self._update_source_digest(
                            digest,
                            PurePosixPath(member.filename).name,
                            archive.read(member),
                        )
                        count += 1
            except (BadZipFile, OSError, RuntimeError) as error:
                raise DatasetSchemaError(
                    "Competition context ZIP cannot be read"
                ) from error
        else:
            raise DatasetSchemaError(
                "Competition context source must be a directory or ZIP file"
            )
        if count == 0:
            raise DatasetSchemaError("No context_*.json files were found")
        return ContextSourceIdentity(
            source_kind=kind,
            revision=f"sha256:{digest.hexdigest()}",
            member_count=count,
        )

    @staticmethod
    def _context_record(
        raw_record: object,
        source_name: str,
    ) -> CompetitionContext:
        if not isinstance(raw_record, Mapping):
            raise DatasetSchemaError(
                f"Context file '{source_name}' must contain one object"
            )
        keys = set(raw_record)
        missing = raw_schema.CONTEXT_REQUIRED_FIELDS - keys
        unknown = keys - raw_schema.CONTEXT_FIELDS
        if missing:
            raise DatasetSchemaError(
                f"Context file '{source_name}' is missing required fields"
            )
        if unknown:
            raise DatasetSchemaError(
                f"Context file '{source_name}' has unknown fields"
            )
        context_id = _canonical_context_id(
            raw_record[raw_schema.CONTEXT_ID_FIELD],
            source_name,
        )
        try:
            return CompetitionContext(
                context_id=context_id,
                title=raw_record.get(raw_schema.CONTEXT_TITLE_FIELD),
                source_url=raw_record[raw_schema.CONTEXT_URL_FIELD],
                passage=raw_record[raw_schema.CONTEXT_PASSAGE_FIELD],
            )
        except ValidationError as error:
            raise DatasetSchemaError(
                f"Context file '{source_name}' is invalid"
            ) from error

    def _iter_directory_contexts(
        self, source: Path
    ) -> Iterator[tuple[str, Any]]:
        for path in self._directory_paths(source):
            try:
                payload = self._decode_json(path.read_bytes(), path.name)
            except OSError as error:
                raise DatasetSchemaError(
                    "Competition context source cannot be read"
                ) from error
            yield path.name, payload

    def _iter_zip_contexts(self, source: Path) -> Iterator[tuple[str, Any]]:
        try:
            with ZipFile(source) as archive:
                for member in self._zip_members(archive):
                    name = PurePosixPath(member.filename).name
                    yield name, self._decode_json(archive.read(member), name)
        except DatasetSchemaError:
            raise
        except (BadZipFile, OSError, RuntimeError) as error:
            raise DatasetSchemaError(
                "Competition context ZIP cannot be read"
            ) from error

    @staticmethod
    def _directory_paths(source: Path) -> list[Path]:
        return sorted(source.glob(_CONTEXT_PATTERN), key=lambda path: path.name)

    @staticmethod
    def _zip_members(archive: ZipFile) -> list[ZipInfo]:
        selected: list[ZipInfo] = []
        seen_names: set[str] = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            basename = path.name
            if member.is_dir() or not fnmatchcase(basename, _CONTEXT_PATTERN):
                continue
            if path.is_absolute() or ".." in path.parts:
                raise DatasetSchemaError(
                    "Competition context ZIP contains an unsafe member path"
                )
            if member.flag_bits & 0x1:
                raise DatasetSchemaError(
                    "Competition context ZIP must not contain encrypted members"
                )
            if basename in seen_names:
                raise DatasetSchemaError(
                    f"Duplicate competition context member '{basename}'"
                )
            seen_names.add(basename)
            selected.append(member)
        return sorted(selected, key=lambda member: PurePosixPath(member.filename).name)

    @staticmethod
    def _update_source_digest(digest: Any, name: str, payload: bytes) -> None:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    @staticmethod
    def _decode_json(payload: bytes, source_name: str) -> Any:
        try:
            text = payload.decode("utf-8")
            return json.loads(text, object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise DatasetSchemaError(
                f"Competition JSON '{source_name}' cannot be read"
            ) from error

    @staticmethod
    def _read_json(source: Path) -> Any:
        if not source.is_file():
            raise DatasetSchemaError("Competition JSON source does not exist")
        try:
            return UitDsc2026DataLoader._decode_json(
                source.read_bytes(), source.name
            )
        except OSError as error:
            raise DatasetSchemaError("Competition JSON cannot be read") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetSchemaError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _canonical_context_id(value: object, source_name: str) -> str:
    """Canonicalize the two organizer ID types observed in the data contract."""
    if isinstance(value, bool):
        raise DatasetSchemaError(
            f"Context file '{source_name}' has invalid context ID"
        )
    if isinstance(value, int):
        if value < 0:
            raise DatasetSchemaError(
                f"Context file '{source_name}' has invalid context ID"
            )
        return str(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    raise DatasetSchemaError(
        f"Context file '{source_name}' has invalid context ID"
    )
