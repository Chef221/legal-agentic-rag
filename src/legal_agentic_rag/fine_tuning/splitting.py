"""Deterministic group-wise three-way splitter for M50 official-data fine-tuning."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.development_split import (
    _duplicate_groups,
    _encode_json,
    _file_sha256,
    _read_mapping,
    _select_raw,
)
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.fine_tuning.dataset import SYSTEM_PROMPT
from legal_agentic_rag.schemas import (
    CompetitionQuestion,
    CompetitionSplitPartition,
    CompetitionSplitSource,
    M50SplitManifest,
)

M50_SPLIT_MANIFEST_FILENAME = "m50_split_manifest.json"
SFT_TRAIN_FILENAME = "sft_train.json"
SFT_VAL_FILENAME = "sft_val.json"
SCREEN_HOLDOUT_FILENAME = "screen_holdout.json"


def find_overlength_question_ids(
    questions: list[CompetitionQuestion],
    tokenizer: Any,
    *,
    max_seq_length: int = 1536,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[str]:
    """Identify questions whose formatted assistant target sequence exceeds max_seq_length."""
    overlength_ids: list[str] = []
    for q in questions:
        if q.reference_answer is None:
            continue
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question},
                {"role": "assistant", "content": q.reference_answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full_text.startswith(prompt_text):
            continue
        target_text = full_text[len(prompt_text) :]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if len(prompt_ids) + len(target_ids) > max_seq_length:
            overlength_ids.append(q.question_id)
    return overlength_ids


class M50FineTuningSplitter:
    """Create deterministic M50 TRAIN / VAL / SCREEN partitions while preserving duplicate groups."""

    def __init__(self, loader: UitDsc2026DataLoader | None = None) -> None:
        self._loader = loader or UitDsc2026DataLoader()

    def split(
        self,
        clean_training_source: Path,
        output_directory: Path,
        *,
        val_target: int = 500,
        screen_target: int = 617,
        seed: int = 2026,
        near_duplicate_threshold: float = 0.92,
        tokenizer: Any | None = None,
        overlength_question_ids: dict[str, list[str]] | None = None,
    ) -> M50SplitManifest:
        """Persist deterministic three-way SFT split without leaking duplicate groups."""
        if val_target <= 0 or screen_target <= 0:
            raise DataValidationError("Partition targets must be positive integers")
        if not 0.5 <= near_duplicate_threshold <= 1.0:
            raise DataValidationError("Near-duplicate threshold must be in [0.5, 1.0]")

        source_hash = _file_sha256(clean_training_source)
        questions = self._loader.load_questions(
            clean_training_source, require_reference_answers=True
        )
        if _file_sha256(clean_training_source) != source_hash:
            raise DataValidationError("Clean training source changed while loading")

        total_count = len(questions)
        if val_target + screen_target >= total_count:
            raise DataValidationError("Validation and screening targets exceed total question count")

        question_texts = [record.question for record in questions]
        question_ids = [record.question_id for record in questions]

        groups, exact_pairs, near_pairs = _duplicate_groups(
            question_texts, near_duplicate_threshold
        )

        group_to_ids: dict[int, list[str]] = defaultdict(list)
        for index, qid in enumerate(question_ids):
            group_to_ids[groups[index]].append(qid)

        ordered_groups = sorted(
            group_to_ids.values(),
            key=lambda ids: sha256(
                f"{seed}:{'|'.join(sorted(ids))}".encode("utf-8")
            ).hexdigest(),
        )

        val_ids: set[str] = set()
        screen_ids: set[str] = set()
        train_ids: set[str] = set()

        for g_ids in ordered_groups:
            if len(val_ids) < val_target:
                val_ids.update(g_ids)
            elif len(screen_ids) < screen_target:
                screen_ids.update(g_ids)
            else:
                train_ids.update(g_ids)

        if val_ids & screen_ids or val_ids & train_ids or screen_ids & train_ids:
            raise DataValidationError("M50 partition overlap detected across group boundaries")

        raw_payload = _read_mapping(clean_training_source)
        partitions_data = {
            SFT_TRAIN_FILENAME: _select_raw(raw_payload, train_ids),
            SFT_VAL_FILENAME: _select_raw(raw_payload, val_ids),
            SCREEN_HOLDOUT_FILENAME: _select_raw(raw_payload, screen_ids),
        }

        # Tokenizer-derived overlength calculation
        computed_overlength: dict[str, list[str]] = {}
        if tokenizer is not None:
            q_by_id = {q.question_id: q for q in questions}
            for filename, p_data in partitions_data.items():
                p_questions = [q_by_id[qid] for qid in p_data]
                computed_overlength[filename] = find_overlength_question_ids(
                    p_questions, tokenizer, max_seq_length=1536
                )
        elif overlength_question_ids is not None:
            computed_overlength = overlength_question_ids

        partition_models: list[CompetitionSplitPartition] = []
        for filename, payload in partitions_data.items():
            encoded = _encode_json(payload)
            partition_models.append(
                CompetitionSplitPartition(
                    filename=filename,
                    question_count=len(payload),
                    sha256=sha256(encoded).hexdigest(),
                    question_ids=list(payload),
                )
            )

        manifest = M50SplitManifest(
            created_at=datetime.now(UTC),
            code_version=__version__,
            clean_training_source=CompetitionSplitSource(
                filename=clean_training_source.name,
                sha256=source_hash,
                question_count=total_count,
            ),
            seed=seed,
            near_duplicate_threshold=near_duplicate_threshold,
            exact_duplicate_pair_count=exact_pairs,
            near_duplicate_pair_count=near_pairs,
            val_target=val_target,
            screen_target=screen_target,
            partitions=partition_models,
            overlength_question_ids_at_1536=computed_overlength,
            warnings=[
                "near_duplicate_detection_is_a_deterministic_heuristic",
                "screen_holdout_is_strictly_for_direct_qa_model_selection",
            ],
        )

        self._persist(output_directory, partitions_data, manifest)
        return manifest

    @staticmethod
    def _persist(
        output_directory: Path,
        partitions: dict[str, dict[str, object]],
        manifest: M50SplitManifest,
    ) -> None:
        output = output_directory.resolve()
        temporary = output.with_name(f".{output.name}.tmp")
        if output.exists() or temporary.exists():
            raise ArtifactCompatibilityError("M50 split output already exists")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            for filename, payload in partitions.items():
                (temporary / filename).write_bytes(_encode_json(payload))
            (temporary / M50_SPLIT_MANIFEST_FILENAME).write_bytes(
                _encode_json(manifest.model_dump(mode="json"))
            )
            temporary.replace(output)
        except OSError as error:
            for path in temporary.glob("*") if temporary.exists() else []:
                path.unlink(missing_ok=True)
            if temporary.exists():
                temporary.rmdir()
            raise ArtifactCompatibilityError(
                "M50 split could not be persisted"
            ) from error
