"""Structured issues emitted by offline audit and validation stages."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


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
