"""Evidence, grading, answer, citation, and verification schemas."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from legal_agentic_rag.schemas.retrieval import RetrievalStrategy


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


def _optional_text(value: object) -> object:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return value


class Evidence(BaseModel):
    """Selected legal chunk packaged as grounded generation context."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    chunk_id: str
    document_id: str
    text: str
    article_number: str | None = None
    article_title: str | None = None
    document_title: str | None = None
    document_number: str | None = None
    document_type: str | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    effect_status: str | None = None
    source_url: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("evidence_id", "chunk_id", "document_id", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject evidence without stable identity or legal text."""
        return _non_empty(value)

    @field_validator(
        "article_number",
        "article_title",
        "document_title",
        "document_number",
        "document_type",
        "effect_status",
        "source_url",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize empty optional evidence metadata to null."""
        return _optional_text(value)


class EvidenceApplicability(StrEnum):
    """Conservative evidence applicability classification."""

    EXPLICIT_MATCH = "explicit_match"
    COMPATIBLE = "compatible"
    UNKNOWN = "unknown"
    INACTIVE = "inactive"
    REFERENCE_MISMATCH = "reference_mismatch"


class EvidenceSelectionReason(StrEnum):
    """Reason one unique retrieval hit was selected or omitted."""

    SELECTED = "selected"
    MAX_EVIDENCE = "max_evidence"
    TOKEN_BUDGET = "token_budget"


class EvidenceSelectionTrace(BaseModel):
    """Explain deterministic ranking and selection of one retrieval hit."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    chunk_id: str
    source_rank: int = Field(ge=1)
    selection_rank: int | None = Field(default=None, ge=1)
    applicability: EvidenceApplicability
    document_reference_match: bool | None = None
    article_reference_match: bool | None = None
    lexical_overlap_score: float = Field(ge=0, le=1)
    selection_score: float
    selected: bool
    reason: EvidenceSelectionReason

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        """Require the selected or omitted chunk identity."""
        return _non_empty(value)

    @model_validator(mode="after")
    def validate_selection_state(self) -> "EvidenceSelectionTrace":
        """Align selected flag, selection rank, and reason."""
        if self.selected:
            if (
                self.reason != EvidenceSelectionReason.SELECTED
                or self.selection_rank is None
            ):
                raise ValueError(
                    "selected evidence requires selected reason and rank"
                )
        elif (
            self.reason == EvidenceSelectionReason.SELECTED
            or self.selection_rank is not None
        ):
            raise ValueError(
                "omitted evidence must not have selected reason or rank"
            )
        return self


class ContextGrade(BaseModel):
    """Structured assessment of whether selected evidence is sufficient."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    is_sufficient: bool
    score: float = Field(default=0.0, ge=0, le=1)
    relevance_score: float = Field(default=0.0, ge=0, le=1)
    coverage_score: float = Field(default=0.0, ge=0, le=1)
    consistency_score: float = Field(default=0.0, ge=0, le=1)
    applicability_score: float = Field(default=0.0, ge=0, le=1)
    missing_aspects: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ContextBuildResult(BaseModel):
    """Bounded evidence selection result produced from ranked retrieval hits."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[Evidence] = Field(default_factory=list)
    input_hit_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    omitted_hit_count: int = Field(ge=0)
    duplicate_hit_count: int = Field(ge=0)
    estimated_token_count: int = Field(ge=0)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    selection_trace: list[EvidenceSelectionTrace] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "ContextBuildResult":
        """Keep evidence, omission, duplicate, and truncation counts aligned."""
        if self.selected_count != len(self.evidence):
            raise ValueError("selected_count must equal evidence count")
        if (
            self.selected_count
            + self.omitted_hit_count
            + self.duplicate_hit_count
            != self.input_hit_count
        ):
            raise ValueError("every retrieval hit must be selected or classified")
        if self.truncated != bool(self.omitted_hit_count):
            raise ValueError("truncated must be true exactly when hits were omitted")
        evidence_ids = [item.evidence_id for item in self.evidence]
        chunk_ids = [item.chunk_id for item in self.evidence]
        if (
            len(evidence_ids) != len(set(evidence_ids))
            or len(chunk_ids) != len(set(chunk_ids))
        ):
            raise ValueError("selected evidence identities must be unique")
        if self.selection_trace:
            trace_chunk_ids = [item.chunk_id for item in self.selection_trace]
            if len(trace_chunk_ids) != len(set(trace_chunk_ids)):
                raise ValueError("selection trace chunk IDs must be unique")
            if len(self.selection_trace) != (
                self.selected_count + self.omitted_hit_count
            ):
                raise ValueError(
                    "selection trace must classify every unique retrieval hit"
                )
            selected_trace = [
                item for item in self.selection_trace if item.selected
            ]
            if [item.chunk_id for item in selected_trace] != chunk_ids:
                raise ValueError(
                    "selected trace order must match selected evidence"
                )
        return self


class Citation(BaseModel):
    """Reference from an answer to an existing evidence record."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    chunk_id: str
    document_id: str
    document_title: str | None = None
    document_number: str | None = None
    article_number: str | None = None
    source_url: str | None = None

    @field_validator("evidence_id", "chunk_id", "document_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject citations without complete identity links."""
        return _non_empty(value)

    @field_validator(
        "document_title", "document_number", "article_number", "source_url", mode="before"
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Normalize empty optional citation metadata to null."""
        return _optional_text(value)


class ModelAnswerDraft(BaseModel):
    """Strict model output before trusted citation metadata is attached."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        """Reject an empty model answer."""
        return _non_empty(value)

    @field_validator("cited_evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: list[str]) -> list[str]:
        """Require valid, unique evidence references."""
        normalized = [_non_empty(value) for value in values]
        if any(
            not value.startswith("E")
            or not value[1:].isdigit()
            or value[1] == "0"
            for value in normalized
        ):
            raise ValueError("cited evidence IDs must use the E<number> format")
        if len(normalized) != len(set(normalized)):
            raise ValueError("cited evidence IDs must be unique")
        return normalized

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: list[str]) -> list[str]:
        """Normalize non-empty warnings while preserving their order."""
        normalized = [_non_empty(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("model warnings must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_grounding_state(self) -> "ModelAnswerDraft":
        """Align citation presence with the model's sufficiency decision."""
        if self.insufficient_evidence and self.cited_evidence_ids:
            raise ValueError("an insufficient draft must not cite evidence")
        if not self.insufficient_evidence and not self.cited_evidence_ids:
            raise ValueError("a grounded draft requires cited evidence")
        return self


class AnswerResponse(BaseModel):
    """Final grounded answer response or explicit abstention."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool
    warnings: list[str] = Field(default_factory=list)
    retrieval_strategy: RetrievalStrategy
    trace_id: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("question", "answer", "trace_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject responses without question, answer text, or trace identity."""
        return _non_empty(value)

    @field_validator("citations")
    @classmethod
    def validate_unique_citations(cls, values: list[Citation]) -> list[Citation]:
        """Reject exact duplicate citations in a packaged answer."""
        identities = [
            (value.evidence_id, value.chunk_id, value.document_id) for value in values
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("citations must not contain exact duplicates")
        return values


class CitationVerificationResult(BaseModel):
    """Rule-based structural and referential citation verification result."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    valid_citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[Citation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "CitationVerificationResult":
        """Keep the result flag consistent with invalid citations and errors."""
        has_failures = bool(self.invalid_citations or self.errors)
        if self.is_valid == has_failures:
            raise ValueError(
                "is_valid must be false exactly when invalid citations or errors exist"
            )
        return self
