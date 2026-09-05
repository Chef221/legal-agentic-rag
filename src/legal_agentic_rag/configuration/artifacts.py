"""Typed configuration for artifact location and compatibility policy."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ArtifactConfig(BaseModel):
    """Backend-neutral artifact storage and load policy."""

    model_config = ConfigDict(extra="forbid")

    root_path: Path
    require_compatible: bool = True
    allow_overwrite: bool = False
    audit_directory: str = "audit"
    normalized_documents_directory: str = "normalized_documents"
    cleaned_documents_directory: str = "cleaned_documents"
    legal_blocks_directory: str = "legal_blocks"
    legal_chunks_directory: str = "legal_chunks"
    relationships_directory: str = "relationships"
    bm25_directory: str = "bm25"
    vector_directory: str = "vector"
    vector_serving_directory: str = "vector_serving"
    graph_directory: str = "graph"
    article_authority_directory: str = "article_authority"
    retrieval_units_v2_directory: str = "retrieval_units_v2"
    bm25_v2_directory: str = "bm25_v2"
    dense_v2_directory: str = "dense_v2"

    @field_validator(
        "audit_directory",
        "normalized_documents_directory",
        "cleaned_documents_directory",
        "legal_blocks_directory",
        "legal_chunks_directory",
        "relationships_directory",
        "bm25_directory",
        "vector_directory",
        "vector_serving_directory",
        "graph_directory",
        "article_authority_directory",
        "retrieval_units_v2_directory",
        "bm25_v2_directory",
        "dense_v2_directory",
    )
    @classmethod
    def validate_directory_name(cls, value: str) -> str:
        """Require one safe relative directory segment under the artifact root."""
        normalized = value.strip()
        path = Path(normalized)
        if (
            not normalized
            or path.is_absolute()
            or len(path.parts) != 1
            or normalized in {".", ".."}
        ):
            raise ValueError("artifact directory must be one relative segment")
        return normalized

    @model_validator(mode="after")
    def validate_unique_directories(self) -> "ArtifactConfig":
        """Prevent two artifact types from writing into the same directory."""
        names = [
            self.audit_directory,
            self.normalized_documents_directory,
            self.cleaned_documents_directory,
            self.legal_blocks_directory,
            self.legal_chunks_directory,
            self.relationships_directory,
            self.bm25_directory,
            self.vector_directory,
            self.vector_serving_directory,
            self.graph_directory,
            self.article_authority_directory,
            self.retrieval_units_v2_directory,
            self.bm25_v2_directory,
            self.dense_v2_directory,
        ]
        if len(names) != len(set(names)):
            raise ValueError("artifact directories must be unique")
        return self

    def directory(self, name: str) -> Path:
        """Resolve a configured artifact directory by field name."""
        value = getattr(self, name)
        if not isinstance(value, str):
            raise ValueError("artifact directory field is not configured")
        return self.root_path / value
