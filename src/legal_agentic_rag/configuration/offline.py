"""Typed configuration for offline pipeline consumers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HtmlCleaningConfig(BaseModel):
    """Conservative HTML-to-text cleaning policy for legal documents."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    unicode_normalization_form: Literal["NFC"] = "NFC"
    remove_tags: frozenset[str] = Field(
        default_factory=lambda: frozenset(
            {"iframe", "nav", "noscript", "object", "script", "style", "template"}
        )
    )
    noise_class_tokens: frozenset[str] = Field(
        default_factory=lambda: frozenset(
            {
                "advertisement",
                "breadcrumb",
                "breadcrumbs",
                "cookie-banner",
                "navigation",
                "pagination",
                "sidebar",
                "social-share",
                "tracking",
            }
        )
    )
    noise_id_tokens: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("remove_tags", "noise_class_tokens", "noise_id_tokens")
    @classmethod
    def normalize_html_tokens(cls, values: frozenset[str]) -> frozenset[str]:
        """Normalize exact-match HTML tokens and reject blank configuration."""
        normalized = frozenset(value.strip().casefold() for value in values)
        if "" in normalized:
            raise ValueError("HTML cleaning tokens must not contain empty text")
        return normalized

    @field_validator("remove_tags")
    @classmethod
    def require_mandatory_noise_tags(cls, values: frozenset[str]) -> frozenset[str]:
        """Prevent configuration from retaining executable or navigation text."""
        mandatory_tags = {"nav", "script", "style"}
        if not mandatory_tags <= values:
            raise ValueError("remove_tags must include nav, script, and style")
        return values


class LegalStructureParserConfig(BaseModel):
    """Deterministic boundaries for conservative legal structure parsing."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    maximum_title_characters: int = Field(default=200, ge=1, le=1_000)
    maximum_title_words: int = Field(default=20, ge=1, le=100)
    emit_unrecognized_marker_issues: bool = True


class ChunkingConfig(BaseModel):
    """Deterministic legal boundary and token fallback limits."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    max_tokens: int = Field(default=384, gt=0)
    max_search_tokens: int = Field(default=448, gt=0)
    min_tokens: int = Field(default=50, gt=0)
    overlap_tokens: int = Field(default=50, ge=0)
    tokenizer_name: Literal["unicode_word_v1"] = "unicode_word_v1"

    @model_validator(mode="after")
    def validate_token_limits(self) -> "ChunkingConfig":
        """Ensure minimum and overlap limits fit the maximum chunk size."""
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        if self.max_search_tokens <= self.max_tokens:
            raise ValueError("max_search_tokens must exceed max_tokens")
        return self


class BM25IndexConfig(BaseModel):
    """SQLite FTS5 reference index identity and lexical matching policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    analyzer_name: Literal["unicode_word_casefold_v1"] = (
        "unicode_word_casefold_v1"
    )
    match_mode: Literal["any", "all"] = "any"
    write_batch_size: int = Field(default=1_000, gt=0)


class EmbeddingConfig(BaseModel):
    """Pinned pretrained embedding model and deterministic encoding policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_name: str = Field(
        default="intfloat/multilingual-e5-small", min_length=1
    )
    model_revision: str = Field(
        default="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        min_length=1,
    )
    expected_dimension: int = Field(default=384, gt=0)
    max_sequence_length: int = Field(default=512, gt=0)
    device: str = Field(default="cpu", min_length=1)
    local_files_only: bool = False
    document_prefix: str = Field(default="passage:", min_length=1)
    query_prefix: str = Field(default="query:", min_length=1)
    normalize_embeddings: Literal[True] = True


class VectorIndexConfig(BaseModel):
    """Exact NumPy vector index format and offline batching policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    backend_name: Literal["numpy_flat"] = "numpy_flat"
    distance_metric: Literal["cosine"] = "cosine"
    dtype: Literal["float32"] = "float32"
    embedding_batch_size: int = Field(default=16, gt=0)
    checkpoint_interval_batches: int = Field(default=100, gt=0, exclude=True)


class GraphIndexConfig(BaseModel):
    """Deterministic persisted adjacency graph identity."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    backend_name: Literal["adjacency_json"] = "adjacency_json"


class IndexBuildConfig(BaseModel):
    """Shared resource limits and backend identity for artifact builds."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    backend_name: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    device: str | None = None


class OfflineConfig(BaseModel):
    """Dataset-independent processing and backend configuration."""

    model_config = ConfigDict(extra="forbid")

    html_cleaning: HtmlCleaningConfig = Field(default_factory=HtmlCleaningConfig)
    legal_structure_parser: LegalStructureParserConfig = Field(
        default_factory=LegalStructureParserConfig
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    bm25: BM25IndexConfig = Field(default_factory=BM25IndexConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    graph_index: GraphIndexConfig = Field(default_factory=GraphIndexConfig)
    index_build: IndexBuildConfig = Field(default_factory=IndexBuildConfig)
