"""Leakage-safe official supervision splits for generator fine-tuning."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from hashlib import sha256

from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas import CompetitionQuestion

GENERATOR_SPLIT_SEED = "uit-dsc-2026-m46-v1"
DEV_SAMPLE_SEED = "m46-dev-sample-v1"
QUESTION_NORMALIZATION = "NFC+casefold+whitespace"


@dataclass(frozen=True, slots=True)
class GeneratorSupervisionSplit:
    """Official question-answer records partitioned by normalized question."""

    train: tuple[CompetitionQuestion, ...]
    dev: tuple[CompetitionQuestion, ...]
    holdout: tuple[CompetitionQuestion, ...]
    normalized_question_group_count: int
    duplicate_group_count: int
    split_seed: str = GENERATOR_SPLIT_SEED

    @property
    def record_counts(self) -> dict[str, int]:
        """Return deterministic record counts for the artifact manifest."""
        return {
            "train": len(self.train),
            "dev": len(self.dev),
            "holdout": len(self.holdout),
        }


def normalize_supervision_question(value: str) -> str:
    """Normalize only for grouping; never replace the official question text."""
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def split_generator_supervision(
    questions: list[CompetitionQuestion],
    *,
    split_seed: str = GENERATOR_SPLIT_SEED,
) -> GeneratorSupervisionSplit:
    """Create the frozen 8/1/1 group-safe split from real official labels."""
    if not questions:
        raise DataValidationError("Generator supervision must not be empty")
    if not split_seed.strip():
        raise DataValidationError("Generator split seed must not be blank")
    question_ids = [item.question_id for item in questions]
    if len(question_ids) != len(set(question_ids)):
        raise DataValidationError(
            "Generator supervision contains duplicate question IDs"
        )
    if any(item.reference_answer is None for item in questions):
        raise DataValidationError(
            "Generator supervision requires real reference answers"
        )

    groups: dict[str, list[CompetitionQuestion]] = {}
    for question in questions:
        normalized = normalize_supervision_question(question.question)
        groups.setdefault(normalized, []).append(question)

    partitions: dict[str, list[CompetitionQuestion]] = {
        "train": [],
        "dev": [],
        "holdout": [],
    }
    for normalized, members in groups.items():
        bucket = int(
            sha256(f"{split_seed}\0{normalized}".encode()).hexdigest(),
            16,
        ) % 10
        partition = "dev" if bucket == 0 else "holdout" if bucket == 1 else "train"
        partitions[partition].extend(members)

    return GeneratorSupervisionSplit(
        train=tuple(partitions["train"]),
        dev=tuple(partitions["dev"]),
        holdout=tuple(partitions["holdout"]),
        normalized_question_group_count=len(groups),
        duplicate_group_count=sum(len(members) > 1 for members in groups.values()),
        split_seed=split_seed,
    )


def fixed_dev_sample(
    split: GeneratorSupervisionSplit,
    *,
    sample_size: int = 200,
    sample_seed: str = DEV_SAMPLE_SEED,
) -> tuple[CompetitionQuestion, ...]:
    """Select the same deterministic dev subset used by M46 through M48."""
    if sample_size <= 0 or sample_size > len(split.dev):
        raise DataValidationError("Dev sample size is outside the dev split")
    if not sample_seed.strip():
        raise DataValidationError("Dev sample seed must not be blank")
    ordered = sorted(
        split.dev,
        key=lambda item: sha256(
            f"{sample_seed}\0{item.question_id}".encode()
        ).hexdigest(),
    )
    return tuple(ordered[:sample_size])


def question_id_digest(questions: tuple[CompetitionQuestion, ...]) -> str:
    """Hash an ordered question-ID sequence for split lineage checks."""
    return sha256(
        "\n".join(item.question_id for item in questions).encode()
    ).hexdigest()
