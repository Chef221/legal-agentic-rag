"""Typed configuration for offline pipeline consumers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetSourceConfig(BaseModel):
    """Dataset identity and bounded loading options."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_name: str = Field(min_length=1)
    dataset_revision: str | None = None
    metadata_config: str = Field(default="metadata", min_length=1)
    content_config: str = Field(default="content", min_length=1)
    relationships_config: str = Field(default="relationships", min_length=1)
    split: str = Field(default="data", min_length=1)
    sample_limit: int | None = Field(default=None, gt=0)
    streaming: bool = False

    @field_validator("dataset_revision", mode="before")
    @classmethod
    def normalize_optional_revision(cls, value: object) -> object:
        """Treat a blank revision as unpinned instead of passing it downstream."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_component_configs(self) -> "DatasetSourceConfig":
        """Require distinct names for the three logical dataset streams."""
        values = (
            self.metadata_config,
            self.content_config,
            self.relationships_config,
        )
        if len(set(values)) != len(values):
            raise ValueError("dataset component config names must be distinct")
        return self


class DatasetAuditConfig(BaseModel):
    """Policies used to classify raw dataset findings without changing data."""

    model_config = ConfigDict(extra="forbid")

    minimum_content_characters: int = Field(default=50, ge=0)
    maximum_content_characters: int = Field(default=2_000_000, gt=0)
    known_effect_statuses: frozenset[str] = Field(default_factory=frozenset)
    known_relationship_labels: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("known_effect_statuses", "known_relationship_labels")
    @classmethod
    def validate_known_values(cls, values: frozenset[str]) -> frozenset[str]:
        """Normalize configured accepted values and reject empty labels."""
        normalized = frozenset(value.strip() for value in values)
        if "" in normalized:
            raise ValueError("known values must not contain empty strings")
        return normalized

    @model_validator(mode="after")
    def validate_content_limits(self) -> "DatasetAuditConfig":
        """Keep content length thresholds internally consistent."""
        if self.minimum_content_characters > self.maximum_content_characters:
            raise ValueError(
                "minimum_content_characters must not exceed "
                "maximum_content_characters"
            )
        return self


class DocumentNormalizationConfig(BaseModel):
    """Dataset-independent labels and explicit canonicalization mappings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_dataset: str = Field(default="aio", min_length=1)
    artifact_version: str = Field(default="1.0", min_length=1)
    effect_status_mapping: dict[str, str] = Field(default_factory=dict)
    document_type_mapping: dict[str, str] = Field(default_factory=dict)

    @field_validator("effect_status_mapping", "document_type_mapping")
    @classmethod
    def validate_mapping(cls, values: dict[str, str]) -> dict[str, str]:
        """Normalize mapping boundaries and reject empty or conflicting keys."""
        normalized: dict[str, str] = {}
        for raw_key, raw_value in values.items():
            key = raw_key.strip()
            value = raw_value.strip()
            if not key or not value:
                raise ValueError("normalization mappings must not contain empty text")
            if key in normalized and normalized[key] != value:
                raise ValueError("normalization mappings contain conflicting keys")
            normalized[key] = value
        return normalized


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
    max_tokens: int = Field(default=512, gt=0)
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
        return self


class BM25IndexConfig(BaseModel):
    """SQLite FTS5 reference index identity and lexical matching policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    analyzer_name: Literal["unicode_word_casefold_v1"] = (
        "unicode_word_casefold_v1"
    )
    match_mode: Literal["any", "all"] = "any"


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


class RelationshipNormalizationConfig(BaseModel):
    """Explicit AIO relationship mapping and rejection policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artifact_version: str = Field(default="1.0", min_length=1)
    source_dataset: str = Field(default="aio", min_length=1)
    relationship_type_mapping: dict[str, str] = Field(default_factory=dict)
    reject_self_loops: Literal[True] = True

    @field_validator("relationship_type_mapping")
    @classmethod
    def validate_relationship_mapping(cls, values: dict[str, str]) -> dict[str, str]:
        """Require explicit non-empty raw-to-canonical relationship labels."""
        normalized: dict[str, str] = {}
        for raw_key, canonical_value in values.items():
            key = raw_key.strip()
            value = canonical_value.strip()
            if not key or not value:
                raise ValueError("relationship mappings must not contain empty text")
            if key in normalized and normalized[key] != value:
                raise ValueError("relationship mappings contain conflicting keys")
            normalized[key] = value
        return normalized


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
    """Top-level typed configuration for future offline consumers."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSourceConfig
    audit: DatasetAuditConfig = Field(default_factory=DatasetAuditConfig)
    normalization: DocumentNormalizationConfig = Field(
        default_factory=DocumentNormalizationConfig
    )
    html_cleaning: HtmlCleaningConfig = Field(default_factory=HtmlCleaningConfig)
    legal_structure_parser: LegalStructureParserConfig = Field(
        default_factory=LegalStructureParserConfig
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    bm25: BM25IndexConfig = Field(default_factory=BM25IndexConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    relationship_normalization: RelationshipNormalizationConfig = Field(
        default_factory=RelationshipNormalizationConfig
    )
    graph_index: GraphIndexConfig = Field(default_factory=GraphIndexConfig)
    index_build: IndexBuildConfig = Field(default_factory=IndexBuildConfig)
