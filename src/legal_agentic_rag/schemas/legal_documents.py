"""Unified schemas for legal documents, structures, blocks, and chunks."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


def _required_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


def _optional_text(value: object) -> object:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return value


def _ordered_non_empty_strings(values: list[str]) -> list[str]:
    normalized = [_required_text(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("values must not contain duplicates")
    return normalized


class LegalBlockType(StrEnum):
    """Supported parser block categories."""

    DOCUMENT = "document"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    ARTICLE = "article"
    CLAUSE = "clause"
    POINT = "point"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    APPENDIX = "appendix"


class LegalDocument(BaseModel):
    """Dataset-independent representation of a legal document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str | None = None
    document_number: str | None = None
    document_type: str | None = None
    issuance_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    effect_status: str | None = None
    issuing_authority: str | None = None
    position_title: str | None = None
    signer: str | None = None
    sector: str | None = None
    legal_field: str | None = None
    scope: str | None = None
    application_info: str | None = None
    publication_date: date | None = None
    source_url: str | None = None
    content_html: str | None = None
    clean_text: str | None = None
    has_content: bool
    source_dataset: str
    raw_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("document_id", "source_dataset")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject empty identifiers and provenance values."""
        return _required_text(value)

    @field_validator(
        "title",
        "document_number",
        "document_type",
        "effect_status",
        "issuing_authority",
        "position_title",
        "signer",
        "sector",
        "legal_field",
        "scope",
        "application_info",
        "source_url",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize empty optional strings to null."""
        return _optional_text(value)

    @field_validator("content_html", "clean_text", mode="before")
    @classmethod
    def preserve_optional_legal_text(cls, value: object) -> object:
        """Map whitespace-only text to null without altering non-empty content."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class LegalStructure(BaseModel):
    """Legal hierarchy attached to a parsed block or retrieval chunk."""

    model_config = ConfigDict(extra="forbid")

    part: str | None = None
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    clause_numbers: list[str] = Field(default_factory=list)
    point_numbers: list[str] = Field(default_factory=list)
    structure_path: list[str] = Field(default_factory=list)

    @field_validator(
        "part",
        "chapter",
        "section",
        "subsection",
        "article_number",
        "article_title",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize absent hierarchy labels consistently."""
        return _optional_text(value)

    @field_validator("clause_numbers", "point_numbers", "structure_path")
    @classmethod
    def validate_ordered_labels(cls, values: list[str]) -> list[str]:
        """Require non-empty, non-duplicated hierarchy labels."""
        return _ordered_non_empty_strings(values)


class LegalBlock(BaseModel):
    """Intermediate legal block emitted by structure parsing."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    document_id: str
    block_type: LegalBlockType
    block_number: str | None = None
    title: str | None = None
    text: str
    parent_block_id: str | None = None
    order_index: int = Field(ge=0)
    structure: LegalStructure = Field(default_factory=LegalStructure)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("block_id", "document_id", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject blocks without stable identity or content."""
        return _required_text(value)

    @field_validator("block_number", "title", "parent_block_id", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize empty optional block labels to null."""
        return _optional_text(value)

    @field_validator("parent_block_id")
    @classmethod
    def validate_parent_id(cls, value: str | None, info: object) -> str | None:
        """Reject an explicitly self-referencing parent when detectable."""
        data = getattr(info, "data", {})
        if value is not None and value == data.get("block_id"):
            raise ValueError("parent_block_id must differ from block_id")
        return value


class LegalChunk(BaseModel):
    """Validated retrieval unit derived from legal structure boundaries."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str
    search_text: str
    token_count: int = Field(ge=1)
    structure: LegalStructure = Field(default_factory=LegalStructure)
    document_title: str | None = None
    document_number: str | None = None
    document_type: str | None = None
    issuance_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    effect_status: str | None = None
    issuing_authority: str | None = None
    legal_field: str | None = None
    source_url: str | None = None
    source_dataset: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "chunk_id", "document_id", "text", "search_text", "source_dataset"
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject chunks without identity, content, or provenance."""
        return _required_text(value)

    @field_validator(
        "document_title",
        "document_number",
        "document_type",
        "effect_status",
        "issuing_authority",
        "legal_field",
        "source_url",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize empty optional chunk metadata to null."""
        return _optional_text(value)
