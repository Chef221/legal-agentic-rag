"""Contracts emitted by document normalization."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_agentic_rag.schemas.auditing import AuditIssue
from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType


class DocumentNormalizationResult(BaseModel):
    """Normalized documents, issues, and provenance for one completed run."""

    model_config = ConfigDict(extra="forbid")

    documents: list[LegalDocument] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    manifest: ArtifactManifest
    input_metadata_count: int = Field(ge=0)
    input_content_count: int = Field(ge=0)
    rejected_metadata_count: int = Field(ge=0)
    orphan_content_count: int = Field(ge=0)
    ambiguous_content_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "DocumentNormalizationResult":
        """Keep result counts aligned with the normalized artifact manifest."""
        if self.manifest.artifact_type != ArtifactType.NORMALIZED_DOCUMENTS:
            raise ValueError("manifest must describe normalized documents")
        if self.manifest.record_count != len(self.documents):
            raise ValueError("manifest record_count must equal document count")
        if self.rejected_metadata_count > self.input_metadata_count:
            raise ValueError("rejected metadata count exceeds input count")
        if (
            len(self.documents) + self.rejected_metadata_count
            != self.input_metadata_count
        ):
            raise ValueError("every metadata record must be normalized or rejected")
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("normalized document IDs must be unique")
        if self.orphan_content_count > self.input_content_count:
            raise ValueError("orphan content count exceeds input count")
        if self.ambiguous_content_count > self.input_content_count:
            raise ValueError("ambiguous content count exceeds input count")
        return self
