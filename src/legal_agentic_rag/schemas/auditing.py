"""Structured reports emitted by offline dataset auditing."""

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

from legal_agentic_rag.schemas.manifests import DatasetManifest


class AuditSeverity(StrEnum):
    """Severity levels for deterministic audit reporting."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AuditIssue(BaseModel):
    """One structured issue associated with a raw or processed record."""

    model_config = ConfigDict(extra="forbid")

    issue_type: str
    severity: AuditSeverity
    record_id: str | None = None
    message: str
    raw_value: JsonValue | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("issue_type", "message")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject issues without a category or explanation."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("record_id", mode="before")
    @classmethod
    def normalize_record_id(cls, value: object) -> object:
        """Normalize an absent record identifier to null."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value


class AuditFieldProfile(BaseModel):
    """Observed shape and nullability of one raw dataset field."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    present_count: int = Field(ge=0)
    null_count: int = Field(ge=0)
    observed_types: dict[str, int] = Field(default_factory=dict)

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, value: str) -> str:
        """Reject empty field names in persisted profiles."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("field_name must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_counts(self) -> "AuditFieldProfile":
        """Keep field presence, null, and observed-type counts consistent."""
        if self.null_count > self.present_count:
            raise ValueError("null_count must not exceed present_count")
        if any(count < 0 for count in self.observed_types.values()):
            raise ValueError("observed type counts must be non-negative")
        if sum(self.observed_types.values()) != self.present_count:
            raise ValueError("observed type counts must equal present_count")
        return self


class ComponentAuditSummary(BaseModel):
    """Counts and schema profile for one logical dataset component."""

    model_config = ConfigDict(extra="forbid")

    component: str
    total_records: int = Field(ge=0)
    unique_ids: int = Field(ge=0)
    duplicate_ids: int = Field(ge=0)
    empty_ids: int = Field(ge=0)
    malformed_ids: int = Field(ge=0)
    field_profiles: list[AuditFieldProfile] = Field(default_factory=list)


class JoinAuditSummary(BaseModel):
    """Coverage of metadata/content joins and relationship endpoints."""

    model_config = ConfigDict(extra="forbid")

    metadata_with_content: int = Field(ge=0)
    metadata_without_content: int = Field(ge=0)
    orphan_content_ids: int = Field(ge=0)
    metadata_with_multiple_content_records: int = Field(ge=0)
    invalid_relationship_sources: int = Field(ge=0)
    invalid_relationship_targets: int = Field(ge=0)


class DatasetAuditReport(BaseModel):
    """Versioned, reproducible result of auditing one dataset ingestion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    audit_config_hash: str
    dataset_manifest: DatasetManifest
    created_at: datetime
    components: dict[str, ComponentAuditSummary]
    joins: JoinAuditSummary
    effect_status_distribution: dict[str, int] = Field(default_factory=dict)
    relationship_distribution: dict[str, int] = Field(default_factory=dict)
    issues: list[AuditIssue] = Field(default_factory=list)

    @field_validator("schema_version", "audit_config_hash")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        """Require non-empty report identity values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("schema_version must not be empty")
        return normalized

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        """Require an unambiguous audit timestamp."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return value

    @field_validator("effect_status_distribution", "relationship_distribution")
    @classmethod
    def validate_distributions(cls, values: dict[str, int]) -> dict[str, int]:
        """Reject empty labels and negative distribution counts."""
        if any(not label.strip() for label in values):
            raise ValueError("distribution labels must not be empty")
        if any(count < 0 for count in values.values()):
            raise ValueError("distribution counts must be non-negative")
        return values

    @model_validator(mode="after")
    def validate_components(self) -> "DatasetAuditReport":
        """Require exactly the three logical AIO streams with matching names."""
        required = {"metadata", "content", "relationships"}
        if set(self.components) != required:
            raise ValueError("components must contain metadata, content, relationships")
        if any(key != summary.component for key, summary in self.components.items()):
            raise ValueError("component map keys must match summary component names")
        return self
