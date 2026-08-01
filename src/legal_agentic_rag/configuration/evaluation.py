"""Typed configuration for reproducible evaluation and run comparison."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legal_agentic_rag.schemas.evaluation import (
    EvaluationMetricDirection,
    EvaluationSelectionMode,
)
from legal_agentic_rag.schemas.retrieval import RetrievalStrategy


class EvaluationConfig(BaseModel):
    """Metric cutoffs and bounded runner behavior."""

    model_config = ConfigDict(extra="forbid")

    cutoffs: list[int] = Field(default_factory=lambda: [1, 5, 10], min_length=1)
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK
    candidate_k: int = Field(default=100, gt=0, le=100)
    run_generation: bool = True
    fail_fast: bool = False
    max_cases: int | None = Field(default=None, gt=0)

    @field_validator("cutoffs")
    @classmethod
    def validate_cutoffs(cls, values: list[int]) -> list[int]:
        """Require sorted, unique, positive retrieval cutoffs."""
        if any(value <= 0 or value > 100 for value in values):
            raise ValueError("evaluation cutoffs must be between 1 and 100")
        if values != sorted(set(values)):
            raise ValueError("evaluation cutoffs must be sorted and unique")
        return values

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> "EvaluationConfig":
        """Ensure candidate pool can supply the largest metric cutoff."""
        if self.candidate_k < max(self.cutoffs):
            raise ValueError("candidate_k must be at least the largest cutoff")
        if self.strategy == RetrievalStrategy.RERANK:
            raise ValueError("rerank is not a complete evaluation strategy")
        return self


class EvaluationCandidateConfig(BaseModel):
    """One explicit candidate and its immutable evaluation report directory."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    report_directory: Path

    @field_validator("report_directory", mode="before")
    @classmethod
    def validate_report_directory(cls, value: object) -> object:
        """Reject an empty path before pathlib normalizes it to the current dir."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("candidate report directory must not be empty")
        return value


class EvaluationObjectiveConfig(BaseModel):
    """One user-declared quality, latency, failure, or resource objective."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    metric: str = Field(min_length=1)
    direction: EvaluationMetricDirection
    eligibility_threshold: float | None = None
    maximum_regression: float | None = Field(default=None, ge=0)


class EvaluationComparisonConfig(BaseModel):
    """Policy for comparing only directly comparable immutable reports."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    comparison_name: str = Field(min_length=1)
    candidates: list[EvaluationCandidateConfig] = Field(min_length=2)
    objectives: list[EvaluationObjectiveConfig] = Field(min_length=1)
    selection_mode: EvaluationSelectionMode = (
        EvaluationSelectionMode.PARETO_ONLY
    )
    baseline_candidate_id: str | None = Field(default=None, min_length=1)
    require_zero_failures: bool = True

    @field_validator("candidates")
    @classmethod
    def validate_candidates(
        cls,
        values: list[EvaluationCandidateConfig],
    ) -> list[EvaluationCandidateConfig]:
        """Require stable unique candidate identities."""
        identities = [value.candidate_id for value in values]
        if len(identities) != len(set(identities)):
            raise ValueError("comparison candidate IDs must be unique")
        return values

    @field_validator("objectives")
    @classmethod
    def validate_objectives(
        cls,
        values: list[EvaluationObjectiveConfig],
    ) -> list[EvaluationObjectiveConfig]:
        """Reject duplicate metrics that would make ordering ambiguous."""
        metrics = [value.metric for value in values]
        if len(metrics) != len(set(metrics)):
            raise ValueError("comparison objective metrics must be unique")
        return values

    @model_validator(mode="after")
    def validate_regression_policy(self) -> "EvaluationComparisonConfig":
        """Require every regression threshold to reference a known baseline."""
        candidate_ids = {
            candidate.candidate_id for candidate in self.candidates
        }
        if (
            self.baseline_candidate_id is not None
            and self.baseline_candidate_id not in candidate_ids
        ):
            raise ValueError("regression baseline must reference a candidate")
        if (
            any(
                objective.maximum_regression is not None
                for objective in self.objectives
            )
            and self.baseline_candidate_id is None
        ):
            raise ValueError(
                "maximum_regression requires baseline_candidate_id"
            )
        return self
