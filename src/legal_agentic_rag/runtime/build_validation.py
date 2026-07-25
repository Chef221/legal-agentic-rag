"""Integrity and lineage validation for one complete offline artifact set."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    ArtifactConfig,
    BuildValidationConfig,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    ArtifactValidationResult,
    BuildValidationReport,
    DatasetManifest,
)

Clock = Callable[[], datetime]
_DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
_MANIFEST_FILENAME = "manifest.json"
_ARTIFACT_DIRECTORIES: tuple[tuple[ArtifactType, str], ...] = (
    (ArtifactType.NORMALIZED_DOCUMENTS, "normalized_documents_directory"),
    (ArtifactType.CLEANED_DOCUMENTS, "cleaned_documents_directory"),
    (ArtifactType.LEGAL_BLOCKS, "legal_blocks_directory"),
    (ArtifactType.LEGAL_CHUNKS, "legal_chunks_directory"),
    (ArtifactType.RELATIONSHIP_MAPPING, "relationships_directory"),
    (ArtifactType.BM25_INDEX, "bm25_directory"),
    (ArtifactType.VECTOR_INDEX, "vector_directory"),
    (ArtifactType.GRAPH_INDEX, "graph_directory"),
)


class ArtifactSetValidator:
    """Validate manifests, payloads, counts, and cross-artifact lineage."""

    def __init__(
        self,
        artifacts: ArtifactConfig,
        policy: BuildValidationConfig,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._artifacts = artifacts
        self._policy = policy
        self._clock = clock or (lambda: datetime.now(UTC))

    def validate(self) -> BuildValidationReport:
        """Return a typed report without mutating the artifact set."""
        checked_at = self._clock()
        errors: list[str] = []
        passed_checks: list[str] = []
        warnings: list[str] = []
        dataset_manifest = self._load_dataset_manifest(errors)
        if dataset_manifest is not None:
            passed_checks.append("dataset_manifest_schema")
            warnings.extend(dataset_manifest.warnings)
            self._validate_dataset_policy(
                dataset_manifest,
                errors=errors,
                passed_checks=passed_checks,
            )

        artifact_results: dict[str, ArtifactValidationResult] = {}
        manifests: dict[ArtifactType, ArtifactManifest] = {}
        for artifact_type, directory_field in _ARTIFACT_DIRECTORIES:
            directory = self._artifacts.directory(directory_field)
            manifest = self._load_artifact_manifest(
                directory,
                artifact_type,
                errors,
            )
            if manifest is None:
                continue
            manifests[artifact_type] = manifest
            result = self._validate_artifact(
                directory=directory,
                manifest=manifest,
                dataset_manifest=dataset_manifest,
                checked_at=checked_at,
            )
            artifact_results[artifact_type.value] = result

        lineage_errors = self._validate_lineage(manifests)
        errors.extend(lineage_errors)
        if not lineage_errors and len(manifests) == len(_ARTIFACT_DIRECTORIES):
            passed_checks.append("artifact_lineage")

        is_full_corpus = self._is_full_corpus(dataset_manifest)
        if self._policy.require_full_corpus and not is_full_corpus:
            errors.append("artifact set does not satisfy the full-corpus policy")
        elif is_full_corpus:
            passed_checks.append("full_corpus_record_counts")

        artifact_failed = any(
            not result.is_valid for result in artifact_results.values()
        )
        return BuildValidationReport(
            checked_at=checked_at,
            dataset_manifest=dataset_manifest,
            artifact_results=artifact_results,
            expected_record_counts=self._policy.expected_record_counts,
            is_full_corpus=is_full_corpus,
            is_valid=not errors and not artifact_failed,
            passed_checks=passed_checks,
            errors=errors,
            warnings=list(dict.fromkeys(warnings)),
        )

    def _load_dataset_manifest(
        self,
        errors: list[str],
    ) -> DatasetManifest | None:
        path = self._artifacts.root_path / _DATASET_MANIFEST_FILENAME
        try:
            return DatasetManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError):
            errors.append("dataset manifest is missing or invalid")
            return None

    def _validate_dataset_policy(
        self,
        manifest: DatasetManifest,
        *,
        errors: list[str],
        passed_checks: list[str],
    ) -> None:
        if (
            self._policy.require_pinned_dataset_revision
            and manifest.dataset_revision is None
        ):
            errors.append("dataset revision is not pinned")
        elif manifest.dataset_revision is not None:
            passed_checks.append("dataset_revision_pinned")

        mismatched = {
            component: {
                "expected": expected,
                "actual": manifest.record_counts.get(component),
            }
            for component, expected in self._policy.expected_record_counts.items()
            if manifest.record_counts.get(component) != expected
        }
        if mismatched:
            components = ", ".join(sorted(mismatched))
            errors.append(f"dataset record counts differ for: {components}")
        elif self._policy.expected_record_counts:
            passed_checks.append("dataset_record_counts")

    def _load_artifact_manifest(
        self,
        directory: Path,
        expected_type: ArtifactType,
        errors: list[str],
    ) -> ArtifactManifest | None:
        try:
            manifest = ArtifactManifest.model_validate_json(
                (directory / _MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError):
            errors.append(f"{expected_type.value} manifest is missing or invalid")
            return None
        if manifest.artifact_type != expected_type:
            errors.append(f"{expected_type.value} manifest declares another type")
            return None
        return manifest

    def _validate_artifact(
        self,
        *,
        directory: Path,
        manifest: ArtifactManifest,
        dataset_manifest: DatasetManifest | None,
        checked_at: datetime,
    ) -> ArtifactValidationResult:
        errors: list[str] = []
        passed_checks = ["manifest_schema", "artifact_type"]
        if dataset_manifest is None or (
            manifest.dataset_name != dataset_manifest.dataset_name
            or manifest.dataset_revision != dataset_manifest.dataset_revision
        ):
            errors.append("artifact dataset identity differs from dataset manifest")
        else:
            passed_checks.append("dataset_identity")

        try:
            passed_checks.extend(
                self._validate_payload(directory=directory, manifest=manifest)
            )
        except ArtifactCompatibilityError as error:
            errors.append(str(error))
        return ArtifactValidationResult(
            manifest=manifest,
            is_valid=not errors,
            checked_at=checked_at,
            passed_checks=list(dict.fromkeys(passed_checks)),
            errors=errors,
        )

    def _validate_payload(
        self,
        *,
        directory: Path,
        manifest: ArtifactManifest,
    ) -> list[str]:
        artifact_type = manifest.artifact_type
        if artifact_type in {
            ArtifactType.NORMALIZED_DOCUMENTS,
            ArtifactType.CLEANED_DOCUMENTS,
            ArtifactType.LEGAL_BLOCKS,
            ArtifactType.LEGAL_CHUNKS,
        }:
            return self._validate_jsonl_payload(
                directory,
                manifest,
                filename_key="payload_file",
                checksum_key="payload_sha256",
            )
        if artifact_type == ArtifactType.RELATIONSHIP_MAPPING:
            return self._validate_jsonl_payload(
                directory,
                manifest,
                filename_key="relationships_filename",
                checksum_key="relationships_sha256",
            )
        if artifact_type == ArtifactType.BM25_INDEX:
            return self._validate_bm25(directory, manifest)
        if artifact_type == ArtifactType.VECTOR_INDEX:
            return self._validate_vector(directory, manifest)
        if artifact_type == ArtifactType.GRAPH_INDEX:
            passes = self._validate_checksum(
                directory,
                manifest,
                filename_key="graph_filename",
                checksum_key="graph_sha256",
            )
            if manifest.metadata.get("edge_count") != manifest.record_count:
                raise ArtifactCompatibilityError(
                    "graph edge count differs from its manifest"
                )
            return [*passes, "record_count"]
        raise ArtifactCompatibilityError("artifact type has no validation policy")

    def _validate_jsonl_payload(
        self,
        directory: Path,
        manifest: ArtifactManifest,
        *,
        filename_key: str,
        checksum_key: str,
    ) -> list[str]:
        passes = self._validate_checksum(
            directory,
            manifest,
            filename_key=filename_key,
            checksum_key=checksum_key,
        )
        filename = manifest.metadata.get(filename_key)
        assert isinstance(filename, str)
        if _count_jsonl_records(directory / filename) != manifest.record_count:
            raise ArtifactCompatibilityError(
                f"{manifest.artifact_type.value} record count differs from payload"
            )
        return [*passes, "record_count"]

    def _validate_bm25(
        self,
        directory: Path,
        manifest: ArtifactManifest,
    ) -> list[str]:
        passes = self._validate_checksum(
            directory,
            manifest,
            filename_key="index_filename",
            checksum_key="index_sha256",
        )
        filename = manifest.metadata.get("index_filename")
        assert isinstance(filename, str)
        try:
            connection = sqlite3.connect(
                f"{(directory / filename).resolve().as_uri()}?mode=ro",
                uri=True,
            )
            try:
                integrity = connection.execute("PRAGMA quick_check").fetchone()
                count = connection.execute(
                    "SELECT COUNT(*) FROM bm25_documents"
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise ArtifactCompatibilityError(
                "BM25 SQLite payload is unreadable"
            ) from error
        if (
            integrity is None
            or integrity[0] != "ok"
            or count is None
            or count[0] != manifest.record_count
        ):
            raise ArtifactCompatibilityError(
                "BM25 integrity or record count validation failed"
            )
        return [*passes, "sqlite_integrity", "record_count"]

    def _validate_vector(
        self,
        directory: Path,
        manifest: ArtifactManifest,
    ) -> list[str]:
        passes: list[str] = []
        for filename_key, checksum_key in (
            ("vectors_filename", "vectors_sha256"),
            ("chunks_filename", "chunks_sha256"),
        ):
            passes.extend(
                self._validate_checksum(
                    directory,
                    manifest,
                    filename_key=filename_key,
                    checksum_key=checksum_key,
                )
            )
        vectors_filename = manifest.metadata.get("vectors_filename")
        chunks_filename = manifest.metadata.get("chunks_filename")
        dimension = manifest.metadata.get("dimension")
        if (
            not isinstance(vectors_filename, str)
            or not isinstance(chunks_filename, str)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ArtifactCompatibilityError("vector manifest metadata is incomplete")
        try:
            vectors = np.load(
                directory / vectors_filename,
                allow_pickle=False,
                mmap_mode="r",
            )
        except (OSError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "vector matrix is unreadable"
            ) from error
        if (
            vectors.dtype != np.float32
            or vectors.shape != (manifest.record_count, dimension)
            or _count_jsonl_records(directory / chunks_filename)
            != manifest.record_count
        ):
            raise ArtifactCompatibilityError(
                "vector shape, dtype, or chunk count differs from manifest"
            )
        return [*passes, "vector_shape", "record_count"]

    @staticmethod
    def _validate_checksum(
        directory: Path,
        manifest: ArtifactManifest,
        *,
        filename_key: str,
        checksum_key: str,
    ) -> list[str]:
        filename = manifest.metadata.get(filename_key)
        expected = manifest.metadata.get(checksum_key)
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise ArtifactCompatibilityError(
                f"{manifest.artifact_type.value} checksum metadata is incomplete"
            )
        path = directory / filename
        try:
            actual = _sha256_file(path)
        except OSError as error:
            raise ArtifactCompatibilityError(
                f"{manifest.artifact_type.value} payload is missing or unreadable"
            ) from error
        if actual != expected:
            raise ArtifactCompatibilityError(
                f"{manifest.artifact_type.value} payload checksum mismatch"
            )
        return ["payload_checksum"]

    def _validate_lineage(
        self,
        manifests: dict[ArtifactType, ArtifactManifest],
    ) -> list[str]:
        if len(manifests) != len(_ARTIFACT_DIRECTORIES):
            return ["artifact set is incomplete"]
        normalized = manifests[ArtifactType.NORMALIZED_DOCUMENTS]
        cleaned = manifests[ArtifactType.CLEANED_DOCUMENTS]
        blocks = manifests[ArtifactType.LEGAL_BLOCKS]
        chunks = manifests[ArtifactType.LEGAL_CHUNKS]
        relationships = manifests[ArtifactType.RELATIONSHIP_MAPPING]
        bm25 = manifests[ArtifactType.BM25_INDEX]
        vector = manifests[ArtifactType.VECTOR_INDEX]
        graph = manifests[ArtifactType.GRAPH_INDEX]
        errors: list[str] = []

        expectations = (
            (cleaned, "source_processing_config_hash", normalized),
            (blocks, "source_processing_config_hash", cleaned),
            (chunks, "source_processing_config_hash", blocks),
            (
                chunks,
                "runtime_normalized_processing_config_hash",
                normalized,
            ),
            (relationships, "source_processing_config_hash", normalized),
            (bm25, "source_processing_config_hash", chunks),
            (vector, "source_processing_config_hash", chunks),
            (
                graph,
                "source_document_processing_config_hash",
                normalized,
            ),
            (
                graph,
                "source_relationship_processing_config_hash",
                relationships,
            ),
        )
        for consumer, metadata_key, source in expectations:
            if (
                consumer.metadata.get(metadata_key)
                != source.processing_config_hash
            ):
                errors.append(
                    f"{consumer.artifact_type.value} lineage differs from "
                    f"{source.artifact_type.value}"
                )

        count_expectations = (
            (cleaned.record_count, normalized.record_count, "cleaned documents"),
            (bm25.record_count, chunks.record_count, "BM25 chunks"),
            (vector.record_count, chunks.record_count, "vector chunks"),
            (
                graph.metadata.get("node_count"),
                normalized.record_count,
                "graph nodes",
            ),
            (
                graph.record_count,
                relationships.record_count,
                "graph relationships",
            ),
        )
        for actual, expected, label in count_expectations:
            if actual != expected:
                errors.append(f"{label} count differs from its source artifact")
        return errors

    def _is_full_corpus(
        self,
        manifest: DatasetManifest | None,
    ) -> bool:
        expected = self._policy.expected_record_counts
        return bool(
            manifest is not None
            and manifest.dataset_revision is not None
            and expected
            and all(
                manifest.record_counts.get(component) == count
                for component, count in expected.items()
            )
            and not any(
                "sample mode" in warning.casefold()
                for warning in manifest.warnings
            )
        )


def persist_build_validation_report(
    report: BuildValidationReport,
    root: Path,
    filename: str,
) -> Path:
    """Persist one validation report without silently replacing an older result."""
    path = root / filename
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    report.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    except FileExistsError as error:
        raise ArtifactCompatibilityError(
            "build validation report already exists"
        ) from error
    except OSError as error:
        raise ArtifactCompatibilityError(
            "build validation report could not be persisted"
        ) from error
    return path


def _count_jsonl_records(path: Path) -> int:
    try:
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise ArtifactCompatibilityError(
                        "JSONL payload contains a blank record"
                    )
                count += 1
        return count
    except UnicodeError as error:
        raise ArtifactCompatibilityError(
            "JSONL payload is not valid UTF-8"
        ) from error
    except OSError as error:
        raise ArtifactCompatibilityError(
            "JSONL payload is missing or unreadable"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
