"""Typed configuration for reproducible local evaluation runs."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
