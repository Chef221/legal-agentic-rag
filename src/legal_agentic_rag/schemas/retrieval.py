"""Unified query, result, filter, and retrieval trace schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


def _unique_strings(values: list[str]) -> list[str]:
    normalized = [_non_empty(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("values must not contain duplicates")
    return normalized


class RetrievalStrategy(StrEnum):
    """Public retrieval strategies and traceable retrieval stages."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    RERANK = "rerank"
    GRAPH = "graph"


class RetrievalFilters(BaseModel):
    """Backend-neutral filters supported by the unified chunk schema."""

    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    legal_fields: list[str] = Field(default_factory=list)
    effect_statuses: list[str] = Field(default_factory=list)

    @field_validator(
        "document_ids", "document_types", "legal_fields", "effect_statuses"
    )
    @classmethod
    def validate_filter_values(cls, values: list[str]) -> list[str]:
        """Require unique, non-empty filter values."""
        return _unique_strings(values)


class GraphPathStep(BaseModel):
    """One directed edge in a bounded graph expansion path."""

    model_config = ConfigDict(extra="forbid")

    source_document_id: str
    target_document_id: str
    relationship_type: str
    hop: int = Field(ge=1, le=2)

    @field_validator(
        "source_document_id", "target_document_id", "relationship_type"
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject graph path steps without identity or relationship type."""
        return _non_empty(value)


class RetrievalTrace(BaseModel):
    """Per-hit contributions from retrieval, fusion, reranking, and graph search."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    bm25_rrf_contribution: float | None = Field(default=None, ge=0)
    dense_rrf_contribution: float | None = Field(default=None, ge=0)
    rrf_score: float | None = Field(default=None, ge=0)
    reranker_score: float | None = None
    graph_hop: int | None = Field(default=None, ge=1, le=2)
    graph_path: list[GraphPathStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_trace(self) -> "RetrievalTrace":
        """Keep graph hop metadata consistent with the recorded path."""
        if self.graph_path:
            maximum_hop = max(step.hop for step in self.graph_path)
            if self.graph_hop != maximum_hop:
                raise ValueError("graph_hop must equal the maximum graph path hop")
        elif self.graph_hop is not None:
            raise ValueError("graph_hop requires a non-empty graph_path")
        return self


class RetrievalQuery(BaseModel):
    """Validated, normalized request passed to retrieval backends."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    original_question: str
    normalized_question: str
    rewritten_question: str | None = None
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=100, gt=0)
    requested_strategy: RetrievalStrategy | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("query_id", "original_question", "normalized_question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject queries without identity or usable text."""
        return _non_empty(value)

    @field_validator("rewritten_question", mode="before")
    @classmethod
    def normalize_rewritten_question(cls, value: object) -> object:
        """Normalize an empty rewrite to null."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> "RetrievalQuery":
        """Ensure the candidate pool can satisfy the requested result count."""
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class RetrievalHit(BaseModel):
    """Backend-neutral ranked legal chunk returned by retrieval."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    chunk_id: str
    document_id: str
    rank: int = Field(ge=1)
    score: float
    strategy: RetrievalStrategy
    text: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    retrieval_trace: RetrievalTrace = Field(default_factory=RetrievalTrace)

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject hits without identity or candidate text."""
        return _non_empty(value)


class RetrievalResponse(BaseModel):
    """Uniform response returned by every retrieval strategy."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    query: RetrievalQuery
    strategy: RetrievalStrategy
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    artifact_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("artifact_versions")
    @classmethod
    def validate_artifact_versions(cls, values: dict[str, str]) -> dict[str, str]:
        """Require non-empty artifact names and versions when provided."""
        return {_non_empty(key): _non_empty(value) for key, value in values.items()}
