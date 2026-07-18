"""Minimal typed configuration for future online pipeline consumers."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalConfig(BaseModel):
    """Backend-neutral retrieval, fusion, reranking, and graph limits."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=100, gt=0)
    rrf_constant: int = Field(default=60, gt=0)
    graph_hop_limit: int = Field(default=1, ge=1, le=2)
    timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> "RetrievalConfig":
        """Ensure the candidate pool can supply the final result count."""
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class GenerationConfig(BaseModel):
    """Backend-neutral generation resource limits and model identity."""

    model_config = ConfigDict(extra="forbid")

    max_context_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    model_name: str | None = None
    model_revision: str | None = None


class AgentConfig(BaseModel):
    """Bounded retry configuration reserved for the later Agent workflow."""

    model_config = ConfigDict(extra="forbid")

    max_retry: int = Field(default=2, ge=0, le=2)


class OnlineConfig(BaseModel):
    """Top-level typed configuration for future online consumers."""

    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
