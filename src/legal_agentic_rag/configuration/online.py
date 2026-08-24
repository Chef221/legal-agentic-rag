"""Typed configuration for online retrieval and grounded generation."""

from typing import Literal
from urllib.parse import urlsplit

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


class BM25RuntimeConfig(BaseModel):
    """Bounded full-corpus lexical query planning without changing artifacts."""

    model_config = ConfigDict(extra="forbid")

    max_query_terms: int = Field(default=8, gt=0, le=64)
    max_document_frequency_ratio: float = Field(default=0.25, gt=0, le=1)


class StartupValidationConfig(BaseModel):
    """Choose deep scans or reuse one immutable full-validation report."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["full", "validated_report"] = "full"


class VectorRuntimeConfig(BaseModel):
    """Memory bounds and progress cadence for online vector loading/search."""

    model_config = ConfigDict(extra="forbid")

    validation_batch_size: int = Field(default=8_192, gt=0)
    search_batch_size: int = Field(default=32_768, gt=0)
    load_progress_interval_records: int = Field(default=100_000, gt=0)
    checksum_progress_interval_bytes: int = Field(
        default=256 * 1024 * 1024,
        gt=0,
    )
    prefer_serving_metadata: bool = True
    require_serving_metadata: bool = False
    serving_metadata_build_batch_size: int = Field(default=10_000, gt=0)
    search_device: Literal["cpu", "cuda"] = "cpu"
    device_transfer_batch_size: int = Field(default=32_768, gt=0)


class QueryUnderstandingConfig(BaseModel):
    """Bound deterministic query analysis and multi-query execution."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    multi_query_enabled: bool = True
    adaptive_routing_enabled: bool = True
    max_variants: int = Field(default=3, ge=1, le=5)


class EvidenceSelectionConfig(BaseModel):
    """Deterministic evidence applicability and ranking policy."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = True
    reference_match_boost: float = Field(default=2.0, ge=0, le=10)
    lexical_overlap_weight: float = Field(default=1.0, ge=0, le=10)
    inactive_penalty: float = Field(default=2.0, ge=0, le=10)
    max_per_document: int = Field(default=100, ge=1, le=100)
    max_per_article: int = Field(default=100, ge=1, le=100)


class ClaimVerificationConfig(BaseModel):
    """Bound deterministic claim-to-evidence grounding checks."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = True
    require_inline_citations: bool = True
    minimum_lexical_support: float = Field(default=0.25, ge=0, le=1)
    minimum_claim_tokens: int = Field(default=2, ge=1, le=100)
    require_numeric_match: bool = True
    require_negation_match: bool = True
    max_claims: int = Field(default=20, ge=1, le=100)


class SemanticVerificationConfig(BaseModel):
    """Optional model-backed semantic claim verification policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    backend: Literal[
        "disabled",
        "openai_compatible",
        "transformers",
    ] = "disabled"
    endpoint_url: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    model_name: str | None = Field(default=None, min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)
    device: str = Field(default="cpu", min_length=1)
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    local_files_only: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_input_tokens: int = Field(default=8192, gt=0, le=131072)
    max_output_tokens: int = Field(default=512, gt=0, le=4096)
    max_structured_output_retries: int = Field(default=1, ge=0, le=1)

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, value: str | None) -> str | None:
        """Accept only an explicit HTTP(S) semantic-verifier endpoint."""
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "semantic verifier endpoint_url must be an HTTP(S) URL"
            )
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_environment_name(
        cls,
        value: str | None,
    ) -> str | None:
        """Store an environment-variable name rather than a secret value."""
        if value is None:
            return None
        if not value.replace("_", "").isalnum() or value[0].isdigit():
            raise ValueError("api_key_env must be an environment-variable name")
        return value

    @model_validator(mode="after")
    def validate_model_identity(self) -> "SemanticVerificationConfig":
        """Require a pinned model only when semantic verification is enabled."""
        if (self.model_name is None) != (self.model_revision is None):
            raise ValueError(
                "semantic verifier model name and revision must be set together"
            )
        if self.backend == "openai_compatible":
            if self.endpoint_url is None:
                raise ValueError(
                    "openai_compatible semantic verifier requires endpoint_url"
                )
            if self.model_name is None or self.model_revision is None:
                raise ValueError(
                    "openai_compatible semantic verifier requires pinned model identity"
                )
        elif self.backend == "transformers":
            if self.model_name is None or self.model_revision is None:
                raise ValueError(
                    "transformers semantic verifier requires pinned model identity"
                )
            if self.endpoint_url is not None or self.api_key_env is not None:
                raise ValueError(
                    "transformers semantic verifier must not contain endpoint settings"
                )
            if (
                self.device.casefold().startswith("cpu")
                and self.torch_dtype != "float32"
            ):
                raise ValueError(
                    "CPU transformers semantic verification requires float32"
                )
        elif any(
            value is not None
            for value in (
                self.endpoint_url,
                self.api_key_env,
                self.model_name,
                self.model_revision,
            )
        ):
            raise ValueError(
                "disabled semantic verifier must not contain model backend settings"
            )
        return self

    def as_generation_config(self) -> "GenerationConfig":
        """Translate an enabled verifier model into the shared provider config."""
        if self.backend == "disabled":
            raise ValueError(
                "disabled semantic verifier has no chat-model configuration"
            )
        return GenerationConfig(
            backend=self.backend,
            endpoint_url=self.endpoint_url,
            api_key_env=self.api_key_env,
            model_name=self.model_name,
            model_revision=self.model_revision,
            device=self.device,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
            timeout_seconds=self.timeout_seconds,
            max_input_tokens=self.max_input_tokens,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            max_structured_output_retries=self.max_structured_output_retries,
        )


class RerankerConfig(BaseModel):
    """Pinned multilingual cross-encoder and bounded inference policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    backend: Literal[
        "sentence_transformers_cross_encoder",
        "jina_native_listwise",
    ] = "sentence_transformers_cross_encoder"
    model_name: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        min_length=1,
    )
    model_revision: str = Field(
        default="1427fd652930e4ba29e8149678df786c240d8825",
        min_length=1,
    )
    device: str = Field(default="cpu", min_length=1)
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    batch_size: int = Field(default=8, gt=0, le=128)
    max_length: int = Field(default=512, gt=0, le=8192)
    max_candidates: int = Field(default=100, gt=0, le=100)
    local_files_only: bool = False
    input_mode: Literal["text_only", "legal_context"] = "legal_context"
    prompt_name: str | None = Field(default=None, min_length=1)
    instruction: str | None = Field(default=None, min_length=1)
    native_context_cap: int | None = Field(default=None, gt=0, le=32768)
    expected_parameter_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_prompt_policy(self) -> "RerankerConfig":
        """Require a stable prompt name whenever a custom instruction is used."""
        if self.backend == "jina_native_listwise":
            if self.native_context_cap is None:
                object.__setattr__(self, "native_context_cap", 12288)
            if self.device.casefold().startswith("cuda"):
                if self.torch_dtype != "float16":
                    raise ValueError("CUDA Jina reranking requires float16")
            elif self.device.casefold().startswith("cpu"):
                if self.torch_dtype != "float32":
                    raise ValueError("CPU Jina reranking requires float32")
            else:
                raise ValueError(f"Unsupported device family for Jina reranker: {self.device}")
        elif self.backend == "sentence_transformers_cross_encoder":
            if (self.prompt_name is None) != (self.instruction is None):
                raise ValueError(
                    "reranker prompt_name and instruction must be set together"
                )
            if (
                self.device.casefold().startswith("cpu")
                and self.torch_dtype != "float32"
            ):
                raise ValueError("CPU reranking requires float32")
        return self


class GenerationConfig(BaseModel):
    """Backend-neutral generation resource limits and model identity."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    max_context_tokens: int = Field(default=4096, gt=0)
    max_evidence: int = Field(default=8, gt=0, le=100)
    inactive_effect_statuses: frozenset[str] = Field(default_factory=frozenset)
    timeout_seconds: float = Field(default=30.0, gt=0)
    backend: Literal[
        "extractive",
        "openai_compatible",
        "transformers",
    ] = "extractive"
    endpoint_url: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    model_name: str | None = Field(default=None, min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)
    device: str = Field(default="cpu", min_length=1)
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    model_loader: Literal["causal_lm", "image_text_to_text"] = "causal_lm"
    local_files_only: bool = False
    max_input_tokens: int = Field(default=8192, gt=0, le=131072)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1024, gt=0, le=8192)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0, le=32)
    max_structured_output_retries: int = Field(default=1, ge=0, le=1)
    max_model_error_retries: int = Field(default=0, ge=0, le=1)
    model_failure_policy: Literal[
        "abstain",
        "top_evidence",
    ] = "abstain"
    max_grounding_repair_retries: int = Field(default=0, ge=0, le=1)
    grounding_failure_policy: Literal[
        "abstain",
        "supported_claims",
        "supported_claims_or_top_evidence",
    ] = "abstain"
    extractive_fallback_max_evidence: int = Field(default=1, ge=1, le=3)
    salvage_rendering: Literal[
        "verbatim",
        "standalone",
    ] = "verbatim"
    prompt_schema_mode: Literal[
        "json_schema",
        "compact_example",
        "plain_text_markers",
    ] = "json_schema"
    answer_style: Literal[
        "concise_grounded",
        "reference_complete",
        "competition_reference",
    ] = "concise_grounded"

    @field_validator("inactive_effect_statuses")
    @classmethod
    def validate_inactive_statuses(cls, values: frozenset[str]) -> frozenset[str]:
        """Normalize explicit inactive-status labels without guessing semantics."""
        normalized = frozenset(value.strip().casefold() for value in values)
        if "" in normalized:
            raise ValueError("inactive effect statuses must not contain empty text")
        return normalized

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, value: str | None) -> str | None:
        """Accept only an explicit HTTP(S) model endpoint."""
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("generator endpoint_url must be an HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_environment_name(
        cls,
        value: str | None,
    ) -> str | None:
        """Store an environment-variable name, never a secret value."""
        if value is None:
            return None
        if not value.replace("_", "").isalnum() or value[0].isdigit():
            raise ValueError("api_key_env must be an environment-variable name")
        return value

    @model_validator(mode="after")
    def validate_model_identity(self) -> "GenerationConfig":
        """Require complete model settings only for the model-backed mode."""
        if (self.model_name is None) != (self.model_revision is None):
            raise ValueError("generator model name and revision must be set together")
        if self.backend == "openai_compatible":
            if self.endpoint_url is None:
                raise ValueError(
                    "openai_compatible generator requires endpoint_url"
                )
            if self.model_name is None or self.model_revision is None:
                raise ValueError(
                    "openai_compatible generator requires pinned model identity"
                )
        elif self.backend == "transformers":
            if self.model_name is None or self.model_revision is None:
                raise ValueError(
                    "transformers generator requires pinned model identity"
                )
            if self.endpoint_url is not None or self.api_key_env is not None:
                raise ValueError(
                    "transformers generator must not contain endpoint settings"
                )
            if (
                self.device.casefold().startswith("cpu")
                and self.torch_dtype != "float32"
            ):
                raise ValueError(
                    "CPU transformers generation requires float32"
                )
        elif any(
            value is not None
            for value in (
                self.endpoint_url,
                self.api_key_env,
                self.model_name,
                self.model_revision,
            )
        ):
            raise ValueError(
                "extractive generator must not contain model backend settings"
            )
        if (
            self.grounding_failure_policy
            in {"supported_claims", "supported_claims_or_top_evidence"}
            and self.max_grounding_repair_retries == 0
        ):
            raise ValueError(
                "grounding recovery requires grounding repair"
            )
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
    bm25_runtime: BM25RuntimeConfig = Field(default_factory=BM25RuntimeConfig)
    startup_validation: StartupValidationConfig = Field(
        default_factory=StartupValidationConfig
    )
    vector_runtime: VectorRuntimeConfig = Field(
        default_factory=VectorRuntimeConfig
    )
    query_understanding: QueryUnderstandingConfig = Field(
        default_factory=QueryUnderstandingConfig
    )
    evidence_selection: EvidenceSelectionConfig = Field(
        default_factory=EvidenceSelectionConfig
    )
    claim_verification: ClaimVerificationConfig = Field(
        default_factory=ClaimVerificationConfig
    )
    semantic_verification: SemanticVerificationConfig = Field(
        default_factory=SemanticVerificationConfig
    )
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    context_grading: ContextGradingConfig = Field(
        default_factory=ContextGradingConfig
    )
    agent: AgentConfig = Field(default_factory=AgentConfig)
