"""Contracts emitted by deterministic legal HTML cleaning."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_agentic_rag.schemas.auditing import AuditIssue
from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType


class HtmlCleaningResult(BaseModel):
    """Cleaned documents, findings, and provenance for one completed run."""

    model_config = ConfigDict(extra="forbid")

    documents: list[LegalDocument] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    manifest: ArtifactManifest
    input_document_count: int = Field(ge=0)
    cleaned_document_count: int = Field(ge=0)
    missing_content_count: int = Field(ge=0)
    empty_output_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "HtmlCleaningResult":
        """Keep cleaning counts and artifact identity mutually consistent."""
        if self.manifest.artifact_type != ArtifactType.CLEANED_DOCUMENTS:
            raise ValueError("manifest must describe cleaned documents")
        if self.manifest.record_count != len(self.documents):
            raise ValueError("manifest record_count must equal document count")
        if self.input_document_count != len(self.documents):
            raise ValueError("cleaning must preserve every input document")
        classified_count = (
            self.cleaned_document_count
            + self.missing_content_count
            + self.empty_output_count
        )
        if classified_count != self.input_document_count:
            raise ValueError("every input document must have one cleaning outcome")
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("cleaned document IDs must be unique")
        actual_cleaned_count = sum(
            document.clean_text is not None for document in self.documents
        )
        if actual_cleaned_count != self.cleaned_document_count:
            raise ValueError("cleaned count must match documents with clean text")
        return self
