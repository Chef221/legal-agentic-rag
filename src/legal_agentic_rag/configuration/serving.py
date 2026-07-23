"""Typed HTTP and local UI serving configuration."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ServingConfig(BaseModel):
    """Bounded local serving policy without secrets or deployment assumptions."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api/v1"
    ui_enabled: bool = True
    ui_path: str = "/ui"
    title: str = Field(default="Vietnamese Legal Agentic RAG", min_length=1)
    docs_enabled: bool = True
    max_question_characters: int = Field(default=4_000, gt=0, le=20_000)
    max_top_k: int = Field(default=100, gt=0, le=100)
    max_candidate_k: int = Field(default=100, gt=0, le=1_000)

    @field_validator("api_prefix", "ui_path")
    @classmethod
    def validate_mount_path(cls, value: str) -> str:
        """Require one absolute URL path without a trailing slash."""
        if (
            not value.startswith("/")
            or value == "/"
            or value.endswith("/")
            or "//" in value
        ):
            raise ValueError("serving path must be absolute without trailing slash")
        return value

    @model_validator(mode="after")
    def validate_limits_and_paths(self) -> "ServingConfig":
        """Prevent route collisions and impossible retrieval limits."""
        if self.api_prefix == self.ui_path:
            raise ValueError("API prefix and UI path must be different")
        if self.max_candidate_k < self.max_top_k:
            raise ValueError("max_candidate_k must be at least max_top_k")
        return self
