"""Validation and streaming equivalence gates for M54 Preprocessing V2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any

from pydantic import ValidationError

from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    LegalProvisionV2,
    LegalReferenceV2,
    RetrievalUnitV2,
    UnrecognizedMarkerV2,
)

CANONICAL_REVISION = "sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e"
ZIP_AUTHORITY_SHA256 = "ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL record at {path}:{line_number}")
            yield line_number, value


def _canonical_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _merged_nonwhitespace_gaps(text: str, intervals: list[tuple[int, int]]) -> int:
    """Return non-whitespace characters outside the union of valid intervals."""
    cursor = 0
    uncovered = 0
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        if start > cursor:
            uncovered += sum(not char.isspace() for char in text[cursor:start])
        cursor = max(cursor, end)
    return uncovered + sum(not char.isspace() for char in text[cursor:])


def _source_identity(source: Path) -> dict[str, Any]:
    loader = UitDsc2026DataLoader()
    identity = loader.inspect_context_source(source)
    revision = str(identity.revision)
    canonical_revision = revision if revision.startswith("sha256:") else f"sha256:{revision}"
    return {
        "raw_zip_sha256": _sha256_file(source) if source.is_file() else None,
        "canonical_revision": canonical_revision,
        "source_members": identity.member_count,
    }


def validate_preprocessing_v2(
    *,
    source: Path | None,
    docs_path: Path,
    provs_path: Path,
    rus_path: Path,
    refs_path: Path,
    unrec_path: Path,
    val_out_path: Path,
    expected_zip_sha256: str = ZIP_AUTHORITY_SHA256,
    expected_canonical_revision: str = CANONICAL_REVISION,
    source_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate persisted V2 artifacts and write computed, non-static gates."""
    required = (docs_path, provs_path, rus_path, refs_path, unrec_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V2 artifact payloads: {missing}")

    if source_identity is None:
        if source is None:
            raise ValueError("source is required when source_identity is not supplied")
        source_data = _source_identity(source)
    else:
        source_data = source_identity
    gates: dict[str, Any] = {
        "RAW_ZIP_SHA256_MATCH": source_data["raw_zip_sha256"] == expected_zip_sha256,
        "CANONICAL_REVISION_MATCH": source_data["canonical_revision"] == expected_canonical_revision,
        "SOURCE_MEMBERS": source_data["source_members"],
        "DOCUMENT_SCHEMA_FAILURES": 0,
        "PROVISION_SCHEMA_FAILURES": 0,
        "RETRIEVAL_UNIT_SCHEMA_FAILURES": 0,
        "LEGAL_REFERENCE_SCHEMA_FAILURES": 0,
        "UNRECOGNIZED_MARKER_SCHEMA_FAILURES": 0,
        "AMBIGUOUS_IDENTITY_WITH_DOCUMENT_NUMBER": 0,
        "UNRESOLVED_IDENTITY_WITH_DOCUMENT_NUMBER": 0,
        "RETRIEVAL_UNITS_WITH_UNVERIFIED_DOCUMENT_NUMBER": 0,
        "REFERENCE_UNIQUE_RESOLUTION_IDENTITY_VIOLATIONS": 0,
        "INVALID_PROVISION_SPANS": 0,
        "INVALID_RETRIEVAL_UNIT_SPANS": 0,
        "RETRIEVAL_AUTHORITY_TEXT_MISMATCHES": 0,
        "ORPHAN_PROVISIONS": 0,
        "ORPHAN_RETRIEVAL_UNITS": 0,
        "ORPHAN_REFERENCES": 0,
        "DUPLICATE_IDS": 0,
        "PROVISION_UNCOVERED_NONWHITESPACE_CHARS": 0,
        "PROVISION_COVERAGE_FAILURE_DOCUMENTS": 0,
        "RETRIEVAL_UNCOVERED_NONWHITESPACE_CHARS": 0,
        "RETRIEVAL_COVERAGE_FAILURE_DOCUMENTS": 0,
        "PARENT_UNIQUE_CONTENT_UNREPRESENTED": 0,
    }
    diagnostics: dict[str, int] = defaultdict(int)
    documents: dict[str, tuple[Path, int, str | None, str]] = {}
    provisions: dict[str, tuple[str, int, int]] = {}
    provision_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    retrieval_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    confirmed_index: dict[str, list[str]] = defaultdict(list)
    ambiguous_index: dict[str, list[str]] = defaultdict(list)
    seen_ids: dict[str, set[str]] = defaultdict(set)

    with tempfile.TemporaryDirectory(prefix="m54-v2-validation-") as directory_name:
        temp_root = Path(directory_name)
        for _, data in _iter_jsonl(docs_path):
            try:
                document = CanonicalDocumentV2.model_validate(data)
            except ValidationError:
                gates["DOCUMENT_SCHEMA_FAILURES"] += 1
                continue
            if document.document_id in seen_ids["documents"]:
                gates["DUPLICATE_IDS"] += 1
                continue
            seen_ids["documents"].add(document.document_id)
            if document.source.corpus_revision != expected_canonical_revision:
                gates["DOCUMENT_SCHEMA_FAILURES"] += 1
            if document.identity.status in {"AMBIGUOUS", "UNRESOLVED"} and document.identity.document_number is not None:
                gates[f"{document.identity.status}_IDENTITY_WITH_DOCUMENT_NUMBER"] += 1
            text_path = temp_root / f"{len(documents):05d}.txt"
            text_path.write_text(document.authority_text, encoding="utf-8", newline="")
            documents[document.document_id] = (
                text_path,
                len(document.authority_text),
                document.identity.document_number,
                document.identity.status,
            )
            diagnostics[f"IDENTITY_{document.identity.status}"] += 1
            if document.identity.status in {"EXPLICIT", "DERIVED_FROM_NAME"} and document.identity.document_number:
                confirmed_index[document.identity.document_number.strip().upper()].append(document.document_id)
            elif document.identity.status == "AMBIGUOUS":
                for candidate in document.identity.candidate_document_numbers:
                    ambiguous_index[candidate.strip().upper()].append(document.document_id)

        cached_document_id: str | None = None
        cached_document_text = ""
        for _, data in _iter_jsonl(provs_path):
            try:
                provision = LegalProvisionV2.model_validate(data)
            except ValidationError:
                gates["PROVISION_SCHEMA_FAILURES"] += 1
                continue
            if provision.provision_id in seen_ids["provisions"]:
                gates["DUPLICATE_IDS"] += 1
                continue
            seen_ids["provisions"].add(provision.provision_id)
            document = documents.get(provision.document_id)
            if document is None:
                gates["ORPHAN_PROVISIONS"] += 1
                continue
            start, end = provision.authority_span.start, provision.authority_span.end
            if start < 0 or end < start or end > document[1]:
                gates["INVALID_PROVISION_SPANS"] += 1
                continue
            if provision.document_id != cached_document_id:
                cached_document_id = provision.document_id
                cached_document_text = document[0].read_text(encoding="utf-8")
            if cached_document_text[start:end] != provision.authority_text:
                gates["INVALID_PROVISION_SPANS"] += 1
                continue
            provisions[provision.provision_id] = (provision.document_id, start, end)
            provision_intervals[provision.document_id].append((start, end))
            diagnostics[f"PROVISION_{provision.provision_type}"] += 1

        cached_document_id = None
        cached_document_text = ""
        for _, data in _iter_jsonl(rus_path):
            try:
                unit = RetrievalUnitV2.model_validate(data)
            except ValidationError:
                gates["RETRIEVAL_UNIT_SCHEMA_FAILURES"] += 1
                continue
            if unit.retrieval_unit_id in seen_ids["retrieval_units"]:
                gates["DUPLICATE_IDS"] += 1
                continue
            seen_ids["retrieval_units"].add(unit.retrieval_unit_id)
            provision = provisions.get(unit.provision_id)
            if provision is None or provision[0] != unit.document_id:
                gates["ORPHAN_RETRIEVAL_UNITS"] += 1
                continue
            relative_start = unit.authority_span_in_provision.start
            relative_end = unit.authority_span_in_provision.end
            provision_length = provision[2] - provision[1]
            if relative_start < 0 or relative_end < relative_start or relative_end > provision_length:
                gates["INVALID_RETRIEVAL_UNIT_SPANS"] += 1
                continue
            absolute_start = provision[1] + relative_start
            absolute_end = provision[1] + relative_end
            document = documents[unit.document_id]
            if unit.document_id != cached_document_id:
                cached_document_id = unit.document_id
                cached_document_text = document[0].read_text(encoding="utf-8")
            if cached_document_text[absolute_start:absolute_end] != unit.authority_text:
                gates["RETRIEVAL_AUTHORITY_TEXT_MISMATCHES"] += 1
                continue
            if document[3] in {"AMBIGUOUS", "UNRESOLVED"} and unit.document_identity.document_number is not None:
                gates["RETRIEVAL_UNITS_WITH_UNVERIFIED_DOCUMENT_NUMBER"] += 1
            retrieval_intervals[unit.document_id].append((absolute_start, absolute_end))
            diagnostics[f"RETRIEVAL_{unit.strategy}"] += 1

        for _, data in _iter_jsonl(refs_path):
            try:
                reference = LegalReferenceV2.model_validate(data)
            except ValidationError:
                gates["LEGAL_REFERENCE_SCHEMA_FAILURES"] += 1
                continue
            if reference.reference_id in seen_ids["references"]:
                gates["DUPLICATE_IDS"] += 1
                continue
            seen_ids["references"].add(reference.reference_id)
            if reference.source_document_id not in documents or (
                reference.source_provision_id is not None and reference.source_provision_id not in provisions
            ):
                gates["ORPHAN_REFERENCES"] += 1
                continue
            target_number = reference.target.document_number_normalized
            expected_status, expected_target, expected_candidates = "UNRESOLVED", None, []
            if target_number:
                confirmed = confirmed_index.get(target_number, [])
                ambiguous = ambiguous_index.get(target_number, [])
                if len(confirmed) == 1 and not ambiguous:
                    expected_status, expected_target = "RESOLVED_UNIQUE", confirmed[0]
                elif confirmed or ambiguous:
                    expected_status = "RESOLVED_AMBIGUOUS"
                    expected_candidates = sorted(set(confirmed + ambiguous))
            resolution = reference.resolution
            if (
                resolution.status != expected_status
                or resolution.target_document_id != expected_target
                or sorted(resolution.candidate_document_ids) != expected_candidates
            ):
                gates["REFERENCE_UNIQUE_RESOLUTION_IDENTITY_VIOLATIONS"] += 1
            diagnostics[f"REFERENCE_{resolution.status}"] += 1

        for _, data in _iter_jsonl(unrec_path):
            try:
                marker = UnrecognizedMarkerV2.model_validate(data)
            except ValidationError:
                gates["UNRECOGNIZED_MARKER_SCHEMA_FAILURES"] += 1
                continue
            document = documents.get(marker.document_id)
            if document is None or marker.character_span.end > document[1]:
                gates["UNRECOGNIZED_MARKER_SCHEMA_FAILURES"] += 1
            diagnostics["UNRECOGNIZED_MARKERS"] += 1

        for document_id, (text_path, _, _, _) in documents.items():
            text = text_path.read_text(encoding="utf-8")
            provision_uncovered = _merged_nonwhitespace_gaps(text, provision_intervals[document_id])
            retrieval_uncovered = _merged_nonwhitespace_gaps(text, retrieval_intervals[document_id])
            gates["PROVISION_UNCOVERED_NONWHITESPACE_CHARS"] += provision_uncovered
            gates["RETRIEVAL_UNCOVERED_NONWHITESPACE_CHARS"] += retrieval_uncovered
            gates["PROVISION_COVERAGE_FAILURE_DOCUMENTS"] += int(provision_uncovered > 0)
            gates["RETRIEVAL_COVERAGE_FAILURE_DOCUMENTS"] += int(retrieval_uncovered > 0)

    gates.update(
        DOCUMENTS=len(documents),
        PROVISIONS=len(seen_ids["provisions"]),
        RETRIEVAL_UNITS=len(seen_ids["retrieval_units"]),
        LEGAL_REFERENCES=len(seen_ids["references"]),
        UNRECOGNIZED_MARKERS=diagnostics["UNRECOGNIZED_MARKERS"],
    )
    count_keys = {
        "SOURCE_MEMBERS", "DOCUMENTS", "PROVISIONS", "RETRIEVAL_UNITS", "LEGAL_REFERENCES", "UNRECOGNIZED_MARKERS",
    }
    failed_gates = [
        key for key, value in gates.items()
        if (key.endswith("_MATCH") and value is not True)
        or (key not in count_keys and isinstance(value, int) and not isinstance(value, bool) and value != 0)
    ]
    result = {
        "schema": "m54_preprocessing_v2_production_validation_v1",
        "overall_pass": not failed_gates,
        "failed_gates": failed_gates,
        "source": source_data,
        "gates": gates,
        "diagnostics": dict(sorted(diagnostics.items())),
    }
    val_out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def compare_preprocessing_v2_equivalence(
    *,
    accepted_paths: dict[str, Path],
    production_paths: dict[str, Path],
    report_path: Path,
) -> dict[str, Any]:
    """Stream-compare canonical JSONL object streams without bulk loading."""
    reports: dict[str, Any] = {}
    overall_pass = True
    for artifact_name, accepted_path in accepted_paths.items():
        production_path = production_paths[artifact_name]
        accepted_digest, production_digest = sha256(), sha256()
        first_mismatch: dict[str, Any] | None = None
        accepted_count = production_count = 0
        with accepted_path.open("r", encoding="utf-8") as accepted, production_path.open("r", encoding="utf-8") as production:
            while True:
                accepted_line = accepted.readline()
                production_line = production.readline()
                if not accepted_line and not production_line:
                    break
                if accepted_line:
                    accepted_count += 1
                    accepted_value = json.loads(accepted_line)
                    accepted_serialized = _canonical_line(accepted_value)
                    accepted_digest.update(accepted_serialized)
                else:
                    accepted_value = None
                    accepted_serialized = b""
                if production_line:
                    production_count += 1
                    production_value = json.loads(production_line)
                    production_serialized = _canonical_line(production_value)
                    production_digest.update(production_serialized)
                else:
                    production_value = None
                    production_serialized = b""
                if first_mismatch is None and accepted_serialized != production_serialized:
                    first_mismatch = {
                        "row": max(accepted_count, production_count),
                        "accepted_id": _record_id(accepted_value),
                        "production_id": _record_id(production_value),
                    }
        artifact_pass = first_mismatch is None and accepted_count == production_count and accepted_digest.digest() == production_digest.digest()
        overall_pass = overall_pass and artifact_pass
        reports[artifact_name] = {
            "pass": artifact_pass,
            "accepted_row_count": accepted_count,
            "production_row_count": production_count,
            "accepted_canonical_stream_sha256": accepted_digest.hexdigest(),
            "production_canonical_stream_sha256": production_digest.hexdigest(),
            "first_mismatch": first_mismatch,
        }
    result = {"schema": "m54_preprocessing_v2_equivalence_v1", "overall_pass": overall_pass, "artifacts": reports}
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _record_id(record: dict[str, Any] | None) -> str | None:
    if record is None:
        return None
    for field in ("document_id", "provision_id", "retrieval_unit_id", "reference_id"):
        if field in record:
            return str(record[field])
    return None
