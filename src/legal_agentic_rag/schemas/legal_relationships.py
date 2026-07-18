"""Unified schema for directed relationships between legal documents."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class LegalRelationship(BaseModel):
    """A normalized document-level legal relationship."""

    model_config = ConfigDict(extra="forbid")

    source_document_id: str
    target_document_id: str
    relationship_type: str | None = None
    raw_relationship: str
    is_directed: bool = True
    source_dataset: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "source_document_id", "target_document_id", "raw_relationship", "source_dataset"
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject relationships without identity, label, or provenance."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("relationship_type", mode="before")
    @classmethod
    def normalize_relationship_type(cls, value: object) -> object:
        """Use null, rather than a guessed label, for unmapped relationships."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value
