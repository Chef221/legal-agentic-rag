"""Contracts emitted by legal relationship normalization."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_agentic_rag.schemas.auditing import AuditIssue
from legal_agentic_rag.schemas.legal_relationships import LegalRelationship
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType


class RelationshipNormalizationResult(BaseModel):
    """Accepted relationships, rejected findings, and artifact provenance."""

    model_config = ConfigDict(extra="forbid")

    relationships: list[LegalRelationship] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    manifest: ArtifactManifest
    input_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "RelationshipNormalizationResult":
        """Keep relationship counts and identities internally consistent."""
        if self.manifest.artifact_type != ArtifactType.RELATIONSHIP_MAPPING:
            raise ValueError("manifest must describe a relationship mapping")
        if self.manifest.record_count != len(self.relationships):
            raise ValueError("manifest record_count must equal relationship count")
        if len(self.relationships) + self.rejected_count != self.input_count:
            raise ValueError("every input relationship must be accepted or rejected")
        if self.duplicate_count > self.rejected_count:
            raise ValueError("duplicate count must not exceed rejected count")
        identities = [
            (
                relationship.source_document_id,
                relationship.target_document_id,
                relationship.raw_relationship,
                relationship.relationship_type,
            )
            for relationship in self.relationships
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("normalized relationships must be unique")
        return self
