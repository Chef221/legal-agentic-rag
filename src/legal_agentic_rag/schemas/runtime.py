"""Typed results produced by offline and online runtime assembly."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    DatasetManifest,
)


class OfflineBuildResult(BaseModel):
    """Summary of one complete, persisted offline runtime build."""

    model_config = ConfigDict(extra="forbid")

    dataset_manifest: DatasetManifest
    artifact_manifests: dict[str, ArtifactManifest]
    output_paths: dict[str, str]
    audit_issue_count: int = Field(ge=0)
    processing_issue_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_artifact_keys(self) -> "OfflineBuildResult":
        """Keep artifact map keys aligned with their declared artifact types."""
        for key, manifest in self.artifact_manifests.items():
            if key != manifest.artifact_type.value:
                raise ValueError("artifact map key must equal artifact type")
            if (
                manifest.dataset_name != self.dataset_manifest.dataset_name
                or manifest.dataset_revision
                != self.dataset_manifest.dataset_revision
            ):
                raise ValueError("runtime artifacts must originate from one dataset")
        return self
