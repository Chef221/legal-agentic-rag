"""Minimal typed configuration for future offline pipeline consumers."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSourceConfig(BaseModel):
    """Dataset identity and bounded loading options."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1)
    dataset_revision: str | None = None
    sample_limit: int | None = Field(default=None, gt=0)
    streaming: bool = False


class ChunkingConfig(BaseModel):
    """Token fallback limits without implementing legal chunking."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0)
    min_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    tokenizer_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_token_limits(self) -> "ChunkingConfig":
        """Ensure minimum and overlap limits fit the maximum chunk size."""
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        return self


class IndexBuildConfig(BaseModel):
    """Shared resource limits and backend identity for artifact builds."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    backend_name: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    device: str | None = None


class OfflineConfig(BaseModel):
    """Top-level typed configuration for future offline consumers."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSourceConfig
    chunking: ChunkingConfig
    index_build: IndexBuildConfig = Field(default_factory=IndexBuildConfig)
