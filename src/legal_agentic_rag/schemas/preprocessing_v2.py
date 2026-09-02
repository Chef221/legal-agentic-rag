"""Pydantic schemas for M54 Preprocessing V2 canonical artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentSourceV2(BaseModel):
    """Source provenance and cryptographic digest for a raw context."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = "uit-dsc-2026-task2"
    context_id: str
    member_name: str
    corpus_revision: str = "sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e"
    raw_passage_sha256: str


class DocumentRawV2(BaseModel):
    """Exact raw text and organizer metadata without mutation."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    link: str | None = None
    text: str


class DocumentCleanerV2(BaseModel):
    """Audited cleaner identification and policy digest."""

    model_config = ConfigDict(extra="forbid")

    name: str = "UitDsc2026PassageCleaner"
    version: str = "1.2"
    policy_identity: dict[str, Any] = Field(default_factory=dict)


class DocumentIdentityEvidenceV2(BaseModel):
    """Deterministic lineage evidence item for document identity."""

    model_config = ConfigDict(extra="forbid")

    source: str
    matched_text: str


class DocumentIdentityV2(BaseModel):
    """Categorical legal identity resolution status and metadata."""

    model_config = ConfigDict(extra="forbid")

    instrument_type: str | None = None
    document_number: str | None = None
    title: str | None = None
    issue_date: str | None = None
    status: Literal["EXPLICIT", "DERIVED_FROM_NAME", "AMBIGUOUS", "UNRESOLVED"]
    candidate_document_numbers: list[str] = Field(default_factory=list)
    evidence: list[DocumentIdentityEvidenceV2] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity_invariants(self) -> DocumentIdentityV2:
        if self.status in ("AMBIGUOUS", "UNRESOLVED") and self.document_number is not None:
            raise ValueError(
                f"Identity status {self.status} must have document_number=None, got {self.document_number!r}"
            )
        return self


class CanonicalDocumentV2(BaseModel):
    """Canonical document representation derived from official raw source."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "m54-preprocessing-v2.1"
    document_id: str
    source: DocumentSourceV2
    raw: DocumentRawV2
    identity: DocumentIdentityV2
    cleaner: DocumentCleanerV2 = Field(default_factory=DocumentCleanerV2)
    authority_text: str
    authority_text_sha256: str
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_hashes(self) -> CanonicalDocumentV2:
        expected_raw_sha = sha256(self.raw.text.encode("utf-8")).hexdigest()
        if self.source.raw_passage_sha256 != expected_raw_sha:
            raise ValueError(
                f"raw_passage_sha256 mismatch: {self.source.raw_passage_sha256} vs {expected_raw_sha}"
            )
        expected_auth_sha = sha256(self.authority_text.encode("utf-8")).hexdigest()
        if self.authority_text_sha256 != expected_auth_sha:
            raise ValueError(
                f"authority_text_sha256 mismatch: {self.authority_text_sha256} vs {expected_auth_sha}"
            )
        return self


class HeadingPathItemV2(BaseModel):
    """Structural parent heading context."""

    model_config = ConfigDict(extra="forbid")

    type: str
    label: str
    title: str | None = None


class TextSpanV2(BaseModel):
    """Half-open character index span [start, end)."""

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int

    @model_validator(mode="after")
    def validate_span(self) -> TextSpanV2:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid character span: [{self.start}, {self.end})")
        return self


class LegalProvisionV2(BaseModel):
    """Canonical legal provision hierarchy unit."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "m54-preprocessing-v2.1"
    provision_id: str
    document_id: str
    canonical_path: str = ""
    parent_provision_id: str | None = None
    provision_type: Literal["ARTICLE", "CLAUSE", "POINT", "DOCUMENT_FALLBACK"]
    article_label: str | None = None
    clause_label: str | None = None
    point_label: str | None = None
    heading_path: list[HeadingPathItemV2] = Field(default_factory=list)
    authority_span: TextSpanV2
    header_span: TextSpanV2 = Field(default_factory=lambda: TextSpanV2(start=0, end=0))
    raw_marker: str | None = None
    parse_status: Literal["EXPLICIT", "CONTROLLED_FALLBACK", "AMBIGUOUS"]
    parse_rule: str
    authority_text: str
    quality_flags: list[str] = Field(default_factory=list)


class RetrievalUnitDocumentIdentityV2(BaseModel):
    """Document identity view in retrieval unit."""

    model_config = ConfigDict(extra="forbid")

    document_number: str | None = None
    title: str | None = None


class RetrievalUnitHierarchyV2(BaseModel):
    """Hierarchy path view in retrieval unit."""

    model_config = ConfigDict(extra="forbid")

    article_label: str | None = None
    clause_label: str | None = None
    point_label: str | None = None
    heading_path: list[HeadingPathItemV2] = Field(default_factory=list)


class RetrievalUnitV2(BaseModel):
    """Retrieval view derived from exactly one legal provision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "m54-preprocessing-v2.1"
    retrieval_unit_id: str
    document_id: str
    provision_id: str
    segment_index: int = 1
    segment_count: int = 1
    authority_span_in_provision: TextSpanV2
    authority_text: str
    retrieval_text: str
    document_identity: RetrievalUnitDocumentIdentityV2
    hierarchy: RetrievalUnitHierarchyV2
    strategy: str
    token_count_authority: int = 0
    token_count_retrieval: int = 0
    quality_flags: list[str] = Field(default_factory=list)


class LegalReferenceTargetV2(BaseModel):
    """Parsed reference target components."""

    model_config = ConfigDict(extra="forbid")

    document_number_raw: str | None = None
    document_number_normalized: str | None = None
    article_label: str | None = None
    clause_label: str | None = None
    point_label: str | None = None


class LegalReferenceResolutionV2(BaseModel):
    """Deterministic two-index reference resolution status."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["RESOLVED_UNIQUE", "RESOLVED_AMBIGUOUS", "UNRESOLVED"]
    target_document_id: str | None = None
    candidate_document_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution(self) -> LegalReferenceResolutionV2:
        if self.status != "RESOLVED_UNIQUE" and self.target_document_id is not None:
            raise ValueError(
                f"Non-unique resolution status {self.status} must have target_document_id=None"
            )
        return self


class LegalReferenceV2(BaseModel):
    """Flat legal reference record."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "m54-preprocessing-v2.1"
    reference_id: str
    source_document_id: str
    source_provision_id: str | None = None
    source_span: TextSpanV2
    evidence_text: str
    relation_type: Literal["CITES", "AMENDS", "SUPPLEMENTS", "REPEALS", "REPLACES", "EFFECTIVITY", "OTHER"]
    target: LegalReferenceTargetV2
    resolution: LegalReferenceResolutionV2
    extraction_rule: str


class UnrecognizedMarkerV2(BaseModel):
    """Diagnostic unrecognized layout marker record."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    line_index: int
    character_span: TextSpanV2
    line_text: str


class PreprocessingV2SegmentationProfile(BaseModel):
    """Typed configuration for controlled retrieval unit segmentation."""

    model_config = ConfigDict(extra="forbid")

    profile_name: str = "m54_shadow_equivalence_v1"
    tokenizer_name: str = "unicode_word_v1"
    max_tokens: int = 512
    overlap_tokens: int = 0


class PreprocessingV2BuildResult(BaseModel):
    """Result manifest of PreprocessingV2Builder execution."""

    model_config = ConfigDict(extra="forbid")

    root_path: Path
    documents_path: Path
    provisions_path: Path
    retrieval_units_path: Path
    legal_references_path: Path
    unrecognized_markers_path: Path
    validation_path: Path
    document_count: int
    provision_count: int
    retrieval_unit_count: int
    legal_reference_count: int
    unrecognized_marker_count: int
    overall_pass: bool
