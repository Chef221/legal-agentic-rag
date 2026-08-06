"""Dataset-neutral records emitted by the competition input boundary."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    ArtifactType,
    DatasetManifest,
)


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must contain non-whitespace text")
    return value


class CompetitionQuestion(BaseModel):
    """One official question and an optional reference answer."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None = None

    @field_validator("question_id", "question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject empty values without modifying official text bytes."""
        return _require_non_blank(value)

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, value: str | None) -> str | None:
        """A present gold answer must contain meaningful text."""
        if value is None:
            return None
        return _require_non_blank(value)


class CompetitionContext(BaseModel):
    """One official legal context before unified legal normalization."""

    model_config = ConfigDict(extra="forbid")

    context_id: str = Field(min_length=1)
    title: str | None = None
    source_url: str = Field(min_length=1)
    passage: str

    @field_validator("context_id", "source_url")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject blank official fields without cleaning legal content."""
        return _require_non_blank(value)

    @field_validator("title")
    @classmethod
    def validate_optional_title(cls, value: str | None) -> str | None:
        """Allow an omitted title but reject a present blank organizer value."""
        if value is None:
            return None
        return _require_non_blank(value)


class CompetitionCorpusAuditReport(BaseModel):
    """Deterministic inventory of one strict official context ingestion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: str
    dataset_revision: str
    source_kind: Literal["directory", "zip"]
    member_count: int = Field(ge=1)
    record_count: int = Field(ge=1)
    unique_context_count: int = Field(ge=1)
    content_context_count: int = Field(ge=0)
    blank_passage_count: int = Field(ge=0)
    missing_title_count: int = Field(ge=0)
    total_passage_characters: int = Field(ge=0)
    total_cleaned_characters: int = Field(ge=0)
    minimum_passage_characters: int = Field(ge=0)
    maximum_passage_characters: int = Field(ge=0)
    duplicate_title_count: int = Field(ge=0)
    duplicate_source_url_count: int = Field(ge=0)
    duplicate_passage_count: int = Field(ge=0)
    html_markup_context_count: int = Field(ge=0)
    boilerplate_context_count: int = Field(ge=0)
    boilerplate_occurrence_count: int = Field(ge=0)
    modified_context_count: int = Field(ge=0)
    passed_checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("dataset_name", "dataset_revision")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        """Require explicit dataset provenance."""
        return _require_non_blank(value)

    @field_validator("passed_checks")
    @classmethod
    def validate_checks(cls, values: list[str]) -> list[str]:
        """Require unique named checks in execution order."""
        normalized = [_require_non_blank(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("passed checks must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_counts(self) -> "CompetitionCorpusAuditReport":
        """One official file must map to one unique context record."""
        if not (
            self.member_count
            == self.record_count
            == self.unique_context_count
        ):
            raise ValueError("context member and record counts must match")
        if self.content_context_count + self.blank_passage_count != self.record_count:
            raise ValueError("content and blank context counts must match records")
        bounded_counts = (
            self.missing_title_count,
            self.duplicate_passage_count,
            self.html_markup_context_count,
            self.boilerplate_context_count,
            self.modified_context_count,
        )
        if any(value > self.record_count for value in bounded_counts):
            raise ValueError("audit subset count cannot exceed record count")
        if self.minimum_passage_characters > self.maximum_passage_characters:
            raise ValueError("passage character bounds are inconsistent")
        return self


class CompetitionCorpusIngestionResult(BaseModel):
    """Unified documents plus reproducible source and artifact provenance."""

    model_config = ConfigDict(extra="forbid")

    normalized_documents: list[LegalDocument]
    cleaned_documents: list[LegalDocument]
    dataset_manifest: DatasetManifest
    normalized_manifest: ArtifactManifest
    cleaned_manifest: ArtifactManifest
    audit: CompetitionCorpusAuditReport

    @model_validator(mode="after")
    def validate_lineage(self) -> "CompetitionCorpusIngestionResult":
        """Keep result records, manifests, and audit on one exact lineage."""
        record_count = len(self.normalized_documents)
        if len(self.cleaned_documents) != record_count:
            raise ValueError("normalized and cleaned document counts must match")
        normalized_ids = [
            document.document_id for document in self.normalized_documents
        ]
        cleaned_ids = [document.document_id for document in self.cleaned_documents]
        if normalized_ids != cleaned_ids:
            raise ValueError("normalized and cleaned document order must match")
        if record_count == 0 or self.normalized_manifest.record_count != record_count:
            raise ValueError("normalized manifest count must match documents")
        if self.audit.record_count != record_count:
            raise ValueError("audit count must match documents")
        if self.dataset_manifest.record_counts.get("contexts") != record_count:
            raise ValueError("dataset manifest count must match documents")
        if self.normalized_manifest.artifact_type != ArtifactType.NORMALIZED_DOCUMENTS:
            raise ValueError("ingestion must emit a normalized-document manifest")
        if self.cleaned_manifest.artifact_type != ArtifactType.CLEANED_DOCUMENTS:
            raise ValueError("ingestion must emit a cleaned-document manifest")
        if self.cleaned_manifest.record_count != record_count:
            raise ValueError("cleaned manifest count must match documents")
        identities = {
            (
                self.dataset_manifest.dataset_name,
                self.dataset_manifest.dataset_revision,
            ),
            (
                self.normalized_manifest.dataset_name,
                self.normalized_manifest.dataset_revision,
            ),
            (
                self.cleaned_manifest.dataset_name,
                self.cleaned_manifest.dataset_revision,
            ),
            (self.audit.dataset_name, self.audit.dataset_revision),
        }
        if len(identities) != 1:
            raise ValueError("ingestion lineage must be identical")
        if any(
            document.source_dataset != self.dataset_manifest.dataset_name
            for document in [
                *self.normalized_documents,
                *self.cleaned_documents,
            ]
        ):
            raise ValueError("document provenance must match dataset manifest")
        return self
