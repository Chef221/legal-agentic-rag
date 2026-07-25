"""Schemas for reproducible validation of one offline artifact set."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legal_agentic_rag.schemas.manifests import (
    ArtifactValidationResult,
    DatasetManifest,
)


class BuildValidationReport(BaseModel):
    """Completeness, integrity, and lineage result for one immutable build."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    checked_at: datetime
    dataset_manifest: DatasetManifest | None = None
    artifact_results: dict[str, ArtifactValidationResult] = Field(
        default_factory=dict
    )
    expected_record_counts: dict[str, int] = Field(default_factory=dict)
    is_full_corpus: bool
    is_valid: bool
    passed_checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("checked_at")
    @classmethod
    def validate_checked_at(cls, value: datetime) -> datetime:
        """Require an unambiguous validation timestamp."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must include timezone information")
        return value

    @field_validator("expected_record_counts")
    @classmethod
    def validate_expected_counts(cls, values: dict[str, int]) -> dict[str, int]:
        """Reject invalid component names and negative expectations."""
        if any(not key.strip() or count < 0 for key, count in values.items()):
            raise ValueError("expected record counts must be named and non-negative")
        return values

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "BuildValidationReport":
        """Keep the top-level status aligned with all reported failures."""
        artifact_failed = any(
            not result.is_valid for result in self.artifact_results.values()
        )
        if self.is_valid and (self.errors or artifact_failed):
            raise ValueError("valid build report cannot contain failures")
        if not self.is_valid and not (self.errors or artifact_failed):
            raise ValueError("invalid build report must identify a failure")
        if self.is_full_corpus and self.dataset_manifest is None:
            raise ValueError("full-corpus status requires a dataset manifest")
        return self


class OfflineBuildState(BaseModel):
    """Immutable recovery identity written before the first offline stage."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: str = "1.0"
    application_config_hash: str = Field(min_length=64, max_length=64)
    code_version: str = Field(min_length=1)
    created_at: datetime

    @field_validator("application_config_hash")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        """Require a lowercase SHA-256 digest."""
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("application_config_hash must be lowercase SHA-256")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        """Require an unambiguous build-start timestamp."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return value
