"""Minimal typed configuration for future online pipeline consumers."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legal_agentic_rag.schemas.retrieval import RetrievalStrategy


class RetrievalConfig(BaseModel):
    """Backend-neutral retrieval, fusion, reranking, and graph limits."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=100, gt=0)
    default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    rrf_constant: int = Field(default=60, gt=0)
    graph_hop_limit: int = Field(default=1, ge=1, le=2)
    graph_seed_chunk_k: int = Field(default=20, gt=0, le=100)
    graph_seed_document_k: int = Field(default=5, gt=0, le=100)
    graph_related_document_k: int = Field(default=20, gt=0, le=100)
    graph_relationship_types: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> "RetrievalConfig":
        """Ensure the candidate pool can supply the final result count."""
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.default_strategy not in {
            RetrievalStrategy.BM25,
            RetrievalStrategy.DENSE,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.HYBRID_RERANK,
            RetrievalStrategy.GRAPH,
        }:
            raise ValueError(
                "default_strategy must be bm25, dense, hybrid, hybrid_rerank, or graph"
            )
        normalized_types = [
            relationship_type.strip()
            for relationship_type in self.graph_relationship_types
        ]
        if any(not value for value in normalized_types):
            raise ValueError("graph relationship types must not contain empty text")
        if len(normalized_types) != len(set(normalized_types)):
            raise ValueError("graph relationship types must not contain duplicates")
        self.graph_relationship_types = normalized_types
        return self


class RerankerConfig(BaseModel):
    """Pinned multilingual cross-encoder and bounded inference policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_name: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        min_length=1,
    )
    model_revision: str = Field(
        default="1427fd652930e4ba29e8149678df786c240d8825",
        min_length=1,
    )
    device: str = Field(default="cpu", min_length=1)
    batch_size: int = Field(default=8, gt=0, le=128)
    max_length: int = Field(default=512, gt=0, le=512)
    max_candidates: int = Field(default=100, gt=0, le=100)
    local_files_only: bool = False


class GenerationConfig(BaseModel):
    """Backend-neutral generation resource limits and model identity."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    max_context_tokens: int = Field(default=4096, gt=0)
    max_evidence: int = Field(default=8, gt=0, le=100)
    inactive_effect_statuses: frozenset[str] = Field(default_factory=frozenset)
    timeout_seconds: float = Field(default=30.0, gt=0)
    model_name: str | None = Field(default=None, min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)

    @field_validator("inactive_effect_statuses")
    @classmethod
    def validate_inactive_statuses(cls, values: frozenset[str]) -> frozenset[str]:
        """Normalize explicit inactive-status labels without guessing semantics."""
        normalized = frozenset(value.strip().casefold() for value in values)
        if "" in normalized:
            raise ValueError("inactive effect statuses must not contain empty text")
        return normalized

    @model_validator(mode="after")
    def validate_model_identity(self) -> "GenerationConfig":
        """Require model name and revision together when a model is configured."""
        if (self.model_name is None) != (self.model_revision is None):
            raise ValueError("generator model name and revision must be set together")
        return self


class ContextGradingConfig(BaseModel):
    """Transparent structural sufficiency policy for the fixed baseline."""

    model_config = ConfigDict(extra="forbid")

    minimum_evidence_count: int = Field(default=1, gt=0, le=100)
    require_document_number: bool = False
    require_article_number: bool = False


class AgentConfig(BaseModel):
    """Bounded deterministic strategy and retry policy for the Agent workflow."""

    model_config = ConfigDict(extra="forbid")

    max_retry: int = Field(default=2, ge=0, le=2)
    strategy_order: list[RetrievalStrategy] = Field(
        default_factory=lambda: [
            RetrievalStrategy.HYBRID_RERANK,
            RetrievalStrategy.GRAPH,
            RetrievalStrategy.HYBRID,
        ],
        min_length=1,
        max_length=3,
    )
    rewrite_query_on_retry: bool = True

    @field_validator("strategy_order")
    @classmethod
    def validate_strategy_order(
        cls,
        values: list[RetrievalStrategy],
    ) -> list[RetrievalStrategy]:
        """Allow only unique public retrieval tools in the bounded route plan."""
        allowed = {
            RetrievalStrategy.BM25,
            RetrievalStrategy.DENSE,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.HYBRID_RERANK,
            RetrievalStrategy.GRAPH,
        }
        if any(value not in allowed for value in values):
            raise ValueError("agent strategy order contains an unsupported strategy")
        if len(values) != len(set(values)):
            raise ValueError("agent strategy order must not contain duplicates")
        return values


class OnlineConfig(BaseModel):
    """Top-level typed configuration for future online consumers."""

    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    context_grading: ContextGradingConfig = Field(
        default_factory=ContextGradingConfig
    )
    agent: AgentConfig = Field(default_factory=AgentConfig)
