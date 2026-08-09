"""Leakage-aware local development split for official answer supervision."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import unicodedata

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.schemas import (
    CompetitionDevelopmentSplitManifest,
    CompetitionSplitPartition,
    CompetitionSplitSource,
)

SPLIT_MANIFEST_FILENAME = "split_manifest.json"
TRAINING_FILENAME = "training.json"
DEVELOPMENT_FILENAME = "development.json"
QUARANTINED_FILENAME = "quarantined.json"


class CompetitionDevelopmentSplitter:
    """Create deterministic group-wise splits while quarantining holdout overlap."""

    def __init__(self, loader: UitDsc2026DataLoader | None = None) -> None:
        self._loader = loader or UitDsc2026DataLoader()

    def split(
        self,
        training_source: Path,
        holdout_sources: list[Path],
        output_directory: Path,
        *,
        dev_fraction: float = 0.15,
        seed: int = 2026,
        near_duplicate_threshold: float = 0.92,
    ) -> CompetitionDevelopmentSplitManifest:
        """Persist one immutable split without modifying any official record."""
        if not 0 < dev_fraction < 1:
            raise DataValidationError("Development fraction must be between 0 and 1")
        if not 0.5 <= near_duplicate_threshold <= 1:
            raise DataValidationError("Near-duplicate threshold must be in [0.5, 1]")
        training_hash = _file_sha256(training_source)
        training = self._loader.load_questions(
            training_source, require_reference_answers=True
        )
        if _file_sha256(training_source) != training_hash:
            raise DataValidationError("Training source changed while loading")

        holdouts = []
        source_models: list[CompetitionSplitSource] = []
        for source in holdout_sources:
            source_hash = _file_sha256(source)
            records = self._loader.load_questions(
                source, require_reference_answers=False
            )
            if _file_sha256(source) != source_hash:
                raise DataValidationError("Holdout source changed while loading")
            holdouts.extend(records)
            source_models.append(
                CompetitionSplitSource(
                    filename=source.name,
                    sha256=source_hash,
                    question_count=len(records),
                )
            )

        all_questions = training + holdouts
        groups, exact_pair_count, near_pair_count = _duplicate_groups(
            [record.question for record in all_questions],
            near_duplicate_threshold,
        )
        training_count = len(training)
        holdout_group_ids = {
            groups[index] for index in range(training_count, len(all_questions))
        }
        quarantined_ids = {
            training[index].question_id
            for index in range(training_count)
            if groups[index] in holdout_group_ids
        }

        eligible_groups: dict[int, list[str]] = defaultdict(list)
        for index, record in enumerate(training):
            if record.question_id not in quarantined_ids:
                eligible_groups[groups[index]].append(record.question_id)
        ordered_groups = sorted(
            eligible_groups.values(),
            key=lambda ids: sha256(
                f"{seed}:{'|'.join(sorted(ids))}".encode("utf-8")
            ).hexdigest(),
        )
        target = round(sum(map(len, ordered_groups)) * dev_fraction)
        development_ids: set[str] = set()
        for group_ids in ordered_groups:
            if len(development_ids) >= target:
                break
            development_ids.update(group_ids)
        training_ids = {
            record.question_id for record in training
        } - quarantined_ids - development_ids

        raw_payload = _read_mapping(training_source)
        partitions = {
            TRAINING_FILENAME: _select_raw(raw_payload, training_ids),
            DEVELOPMENT_FILENAME: _select_raw(raw_payload, development_ids),
            QUARANTINED_FILENAME: _select_raw(raw_payload, quarantined_ids),
        }
        partition_models = []
        for filename, payload in partitions.items():
            encoded = _encode_json(payload)
            partition_models.append(
                CompetitionSplitPartition(
                    filename=filename,
                    question_count=len(payload),
                    sha256=sha256(encoded).hexdigest(),
                    question_ids=list(payload),
                )
            )
        manifest = CompetitionDevelopmentSplitManifest(
            created_at=datetime.now(UTC),
            code_version=__version__,
            training_source=CompetitionSplitSource(
                filename=training_source.name,
                sha256=training_hash,
                question_count=len(training),
            ),
            holdout_sources=source_models,
            seed=seed,
            dev_fraction=dev_fraction,
            near_duplicate_threshold=near_duplicate_threshold,
            exact_duplicate_pair_count=exact_pair_count,
            near_duplicate_pair_count=near_pair_count,
            partitions=partition_models,
            warnings=[
                "near_duplicate_detection_is_a_deterministic_heuristic",
                "split_is_for_local_development_not_official_retrieval_evaluation",
            ],
        )
        self._persist(output_directory, partitions, manifest)
        return manifest

    @staticmethod
    def _persist(
        output_directory: Path,
        partitions: dict[str, dict[str, object]],
        manifest: CompetitionDevelopmentSplitManifest,
    ) -> None:
        output = output_directory.resolve()
        temporary = output.with_name(f".{output.name}.tmp")
        if output.exists() or temporary.exists():
            raise ArtifactCompatibilityError("Development split output already exists")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            for filename, payload in partitions.items():
                (temporary / filename).write_bytes(_encode_json(payload))
            (temporary / SPLIT_MANIFEST_FILENAME).write_bytes(
                _encode_json(manifest.model_dump(mode="json"))
            )
            temporary.replace(output)
        except OSError as error:
            for path in temporary.glob("*") if temporary.exists() else []:
                path.unlink(missing_ok=True)
            if temporary.exists():
                temporary.rmdir()
            raise ArtifactCompatibilityError(
                "Development split could not be persisted"
            ) from error


def _duplicate_groups(
    questions: list[str], threshold: float
) -> tuple[list[int], int, int]:
    normalized = [_normalize_question(question) for question in questions]
    parent = list(range(len(questions)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    exact_buckets: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(normalized):
        exact_buckets[value].append(index)
    exact_pairs = 0
    for bucket in exact_buckets.values():
        exact_pairs += len(bucket) * (len(bucket) - 1) // 2
        for index in bucket[1:]:
            union(bucket[0], index)

    shingles = [_token_shingles(value) for value in normalized]
    shingle_frequency = Counter(
        shingle for question_shingles in shingles for shingle in question_shingles
    )
    anchors: dict[str, list[int]] = defaultdict(list)
    for index, values in enumerate(shingles):
        selected = sorted(
            values,
            key=lambda value: (
                shingle_frequency[value],
                sha256(value.encode("utf-8")).hexdigest(),
            ),
        )[:8]
        for anchor in selected:
            anchors[sha256(anchor.encode("utf-8")).hexdigest()].append(index)
    candidates: set[tuple[int, int]] = set()
    for bucket in anchors.values():
        if len(bucket) > 200:
            continue
        for position, left in enumerate(bucket):
            for right in bucket[position + 1 :]:
                if normalized[left] != normalized[right]:
                    candidates.add((min(left, right), max(left, right)))
    near_pairs = 0
    for left, right in candidates:
        union_size = len(shingles[left] | shingles[right])
        similarity = (
            len(shingles[left] & shingles[right]) / union_size if union_size else 0
        )
        if similarity >= threshold:
            union(left, right)
            near_pairs += 1
    roots = [find(index) for index in range(len(questions))]
    canonical = {root: position for position, root in enumerate(sorted(set(roots)))}
    return [canonical[root] for root in roots], exact_pairs, near_pairs


def _normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def _token_shingles(value: str) -> set[str]:
    tokens = value.split()
    if len(tokens) < 3:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)}


def _file_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactCompatibilityError("Development split input is unreadable") from error


def _read_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataValidationError("Training source cannot be preserved") from error
    if not isinstance(payload, dict):
        raise DataValidationError("Training source root must be an object")
    return payload


def _select_raw(payload: dict[str, object], identities: set[str]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key in identities}


def _encode_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
