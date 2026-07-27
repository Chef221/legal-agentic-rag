"""Versioned dataset and artifact manifest schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include timezone information")
    return value


class ArtifactType(StrEnum):
    """Persisted artifact categories produced by the offline pipeline."""

    NORMALIZED_DOCUMENTS = "normalized_documents"
    CLEANED_DOCUMENTS = "cleaned_documents"
    LEGAL_BLOCKS = "legal_blocks"
    LEGAL_CHUNKS = "legal_chunks"
    BM25_INDEX = "bm25_index"
    EMBEDDING_OUTPUT = "embedding_output"
    VECTOR_INDEX = "vector_index"
    VECTOR_SERVING_METADATA = "vector_serving_metadata"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    GRAPH_INDEX = "graph_index"


class DatasetManifest(BaseModel):
    """Provenance and counts for one reproducible dataset ingestion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    dataset_name: str
    dataset_revision: str | None = None
    loaded_at: datetime
    configs: list[str]
    record_counts: dict[str, int]
    processing_config_hash: str
    code_version: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("schema_version", "dataset_name", "processing_config_hash")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Require stable manifest identity values."""
        return _non_empty(value)

    @field_validator("dataset_revision", "code_version", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize empty optional version values to null."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("loaded_at")
    @classmethod
    def validate_loaded_at(cls, value: datetime) -> datetime:
        """Require an unambiguous ingestion timestamp."""
        return _aware(value)

    @field_validator("configs")
    @classmethod
    def validate_configs(cls, values: list[str]) -> list[str]:
        """Require at least one unique logical dataset config."""
        normalized = [_non_empty(value) for value in values]
        if not normalized:
            raise ValueError("configs must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("configs must not contain duplicates")
        return normalized

    @field_validator("record_counts")
    @classmethod
    def validate_record_counts(cls, values: dict[str, int]) -> dict[str, int]:
        """Reject invalid component names and negative record counts."""
        normalized: dict[str, int] = {}
        for key, count in values.items():
            normalized[_non_empty(key)] = count
            if count < 0:
                raise ValueError("record counts must be non-negative")
        return normalized


class ArtifactManifest(BaseModel):
    """Compatibility contract for one persisted processed artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    artifact_type: ArtifactType
    artifact_version: str
    dataset_name: str
    dataset_revision: str | None = None
    created_at: datetime
    record_count: int = Field(ge=0)
    processing_config_hash: str
    code_version: str | None = None
    backend: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "schema_version",
        "artifact_version",
        "dataset_name",
        "processing_config_hash",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Require stable artifact identity values."""
        return _non_empty(value)

    @field_validator(
        "dataset_revision",
        "code_version",
        "backend",
        "model_name",
        "model_revision",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize empty optional artifact metadata to null."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        """Require an unambiguous artifact build timestamp."""
        return _aware(value)

    @model_validator(mode="after")
    def validate_model_revision(self) -> "ArtifactManifest":
        """Disallow a model revision without an identified model."""
        if self.model_revision is not None and self.model_name is None:
            raise ValueError("model_revision requires model_name")
        return self


class ArtifactValidationResult(BaseModel):
    """Result of validating a persisted artifact and its compatibility."""

    model_config = ConfigDict(extra="forbid")

    manifest: ArtifactManifest
    is_valid: bool
    checked_at: datetime
    passed_checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("checked_at")
    @classmethod
    def validate_checked_at(cls, value: datetime) -> datetime:
        """Require an unambiguous validation timestamp."""
        return _aware(value)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "ArtifactValidationResult":
        """Keep the validity flag consistent with reported errors."""
        if self.is_valid == bool(self.errors):
            raise ValueError("is_valid must be false exactly when errors are present")
        return self
