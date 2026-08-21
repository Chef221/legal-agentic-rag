"""Validation tests for bounded offline and online settings."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    AgentConfig,
    ArtifactConfig,
    BM25IndexConfig,
    BM25RuntimeConfig,
    BuildValidationConfig,
    ClaimVerificationConfig,
    ChunkingConfig,
    ContextGradingConfig,
    EmbeddingConfig,
    EvidenceSelectionConfig,
    GenerationConfig,
    GraphIndexConfig,
    HtmlCleaningConfig,
    LegalStructureParserConfig,
    LoggingConfig,
    OfflineConfig,
    QueryUnderstandingConfig,
    RetrievalConfig,
    RerankerConfig,
    SemanticVerificationConfig,
    StartupValidationConfig,
    VectorIndexConfig,
    VectorRuntimeConfig,
)
from legal_agentic_rag.schemas import RetrievalStrategy


def test_bm25_index_config_rejects_unknown_analyzer_or_match_mode() -> None:
    """Only the implemented deterministic lexical policy is accepted."""
    with pytest.raises(ValidationError):
        BM25IndexConfig(analyzer_name="unknown")
    with pytest.raises(ValidationError):
        BM25IndexConfig(match_mode="phrase")
    with pytest.raises(ValidationError):
        BM25IndexConfig(write_batch_size=0)


def test_bm25_runtime_config_bounds_full_corpus_query_planning() -> None:
    """Lexical term count and corpus-frequency threshold are explicit."""
    config = BM25RuntimeConfig()

    assert config.max_query_terms == 8
    assert config.max_document_frequency_ratio == 0.25
    with pytest.raises(ValidationError):
        BM25RuntimeConfig(max_query_terms=0)
    with pytest.raises(ValidationError):
        BM25RuntimeConfig(max_document_frequency_ratio=0)


def test_startup_validation_requires_an_explicit_supported_mode() -> None:
    """Deep validation remains default and report reuse is opt-in."""
    assert StartupValidationConfig().mode == "full"
    assert StartupValidationConfig(mode="validated_report").mode == (
        "validated_report"
    )
    with pytest.raises(ValidationError):
        StartupValidationConfig(mode="skip")


def test_embedding_and_vector_defaults_are_pinned_and_bounded() -> None:
    """Dense artifacts have reproducible model, dimension, metric, and batching."""
    embedding = EmbeddingConfig()
    vector = VectorIndexConfig()

    assert embedding.model_name == "intfloat/multilingual-e5-small"
    assert embedding.model_revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert embedding.expected_dimension == 384
    assert embedding.device == "cpu"
    assert embedding.query_prompt_name is None
    assert vector.backend_name == "numpy_flat"
    assert vector.distance_metric == "cosine"
    assert vector.embedding_batch_size == 16
    with pytest.raises(ValidationError):
        EmbeddingConfig(model_revision="")
    with pytest.raises(ValidationError, match="mutually exclusive"):
        EmbeddingConfig(query_prompt_name="query", query_instruction="instruction")
    with pytest.raises(ValidationError, match="requires float32"):
        EmbeddingConfig(torch_dtype="float16")
    with pytest.raises(ValidationError):
        VectorIndexConfig(embedding_batch_size=0)


def test_retrieval_config_validates_candidate_and_graph_limits() -> None:
    """Invalid retrieval bounds fail before any backend is called."""
    assert RetrievalConfig().default_strategy.value == "hybrid"
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=20, candidate_k=10)
    with pytest.raises(ValidationError):
        RetrievalConfig(graph_hop_limit=3)
    assert (
        RetrievalConfig(default_strategy="hybrid_rerank").default_strategy.value
        == "hybrid_rerank"
    )
    assert RetrievalConfig(default_strategy="graph").default_strategy.value == "graph"
    with pytest.raises(ValidationError):
        RetrievalConfig(graph_relationship_types=["amends", " amends "])
    with pytest.raises(ValidationError):
        RetrievalConfig(graph_seed_document_k=101)
    with pytest.raises(ValidationError):
        RetrievalConfig(default_strategy="rerank")


def test_vector_runtime_config_bounds_online_load_and_search_batches() -> None:
    """Online dense execution is explicitly bounded without changing artifacts."""
    config = VectorRuntimeConfig()

    assert config.validation_batch_size == 8_192
    assert config.search_batch_size == 32_768
    assert config.load_progress_interval_records == 100_000
    assert config.checksum_progress_interval_bytes == 256 * 1024 * 1024
    assert config.prefer_serving_metadata is True
    assert config.require_serving_metadata is False
    assert config.serving_metadata_build_batch_size == 10_000
    assert config.search_device == "cpu"
    assert config.device_transfer_batch_size == 32_768
    with pytest.raises(ValidationError):
        VectorRuntimeConfig(validation_batch_size=0)
    with pytest.raises(ValidationError):
        VectorRuntimeConfig(search_batch_size=0)
    with pytest.raises(ValidationError):
        VectorRuntimeConfig(load_progress_interval_records=0)
    with pytest.raises(ValidationError):
        VectorRuntimeConfig(search_device="tpu")
    with pytest.raises(ValidationError):
        VectorRuntimeConfig(device_transfer_batch_size=0)


def test_query_understanding_config_is_enabled_and_bounded() -> None:
    """Query analysis and multi-query behavior are explicit runtime policy."""
    config = QueryUnderstandingConfig()

    assert config.enabled is True
    assert config.multi_query_enabled is True
    assert config.adaptive_routing_enabled is True
    assert config.max_variants == 3
    with pytest.raises(ValidationError):
        QueryUnderstandingConfig(max_variants=0)
    with pytest.raises(ValidationError):
        QueryUnderstandingConfig(max_variants=6)


def test_graph_index_configuration_is_explicit() -> None:
    """The reference graph backend remains typed without raw label policy."""
    assert GraphIndexConfig().backend_name == "adjacency_json"


def test_generation_and_context_grading_defaults_are_bounded() -> None:
    """Fixed RAG context limits and structural sufficiency are typed."""
    generation = GenerationConfig()
    grading = ContextGradingConfig()

    assert generation.max_context_tokens == 4096
    assert generation.max_evidence == 8
    assert generation.backend == "extractive"
    assert generation.max_structured_output_retries == 1
    assert generation.model_failure_policy == "abstain"
    assert generation.grounding_failure_policy == "abstain"
    assert generation.prompt_schema_mode == "json_schema"
    assert generation.salvage_rendering == "verbatim"
    assert generation.repetition_penalty == 1.0
    assert generation.no_repeat_ngram_size == 0
    assert grading.minimum_evidence_count == 1
    with pytest.raises(ValidationError):
        GenerationConfig(max_evidence=101)
    with pytest.raises(ValidationError):
        GenerationConfig(max_structured_output_retries=2)
    with pytest.raises(ValidationError):
        GenerationConfig(repetition_penalty=0.99)
    with pytest.raises(ValidationError):
        GenerationConfig(no_repeat_ngram_size=33)
    with pytest.raises(ValidationError, match="grounding recovery"):
        GenerationConfig(
            grounding_failure_policy="supported_claims_or_top_evidence"
        )
    with pytest.raises(ValidationError):
        GenerationConfig(model_name="model-without-revision")
    with pytest.raises(ValidationError, match="endpoint_url"):
        GenerationConfig(
            backend="openai_compatible",
            model_name="fixture-model",
            model_revision="fixture-revision",
        )
    model_generation = GenerationConfig(
        backend="openai_compatible",
        endpoint_url="http://127.0.0.1:8001/v1/chat/completions",
        api_key_env="LEGAL_RAG_MODEL_API_KEY",
        model_name="fixture-model",
        model_revision="fixture-revision",
    )
    assert model_generation.endpoint_url.endswith("/v1/chat/completions")
    local_generation = GenerationConfig(
        backend="transformers",
        model_name="fixture-model",
        model_revision="fixture-revision",
        device="cuda",
        torch_dtype="float16",
    )
    assert local_generation.max_input_tokens == 8192
    with pytest.raises(ValidationError, match="pinned model identity"):
        GenerationConfig(backend="transformers")
    with pytest.raises(ValidationError, match="requires float32"):
        GenerationConfig(
            backend="transformers",
            model_name="fixture-model",
            model_revision="fixture-revision",
            device="cpu",
            torch_dtype="float16",
        )
    with pytest.raises(ValidationError, match="endpoint settings"):
        GenerationConfig(
            backend="transformers",
            endpoint_url="http://127.0.0.1:8001/v1/chat/completions",
            model_name="fixture-model",
            model_revision="fixture-revision",
        )
    with pytest.raises(ValidationError, match="must not contain"):
        GenerationConfig(
            endpoint_url="http://127.0.0.1:8001/v1/chat/completions"
        )
    with pytest.raises(ValidationError, match="HTTP"):
        GenerationConfig(
            backend="openai_compatible",
            endpoint_url="file:///tmp/model",
            model_name="fixture-model",
            model_revision="fixture-revision",
        )
    with pytest.raises(ValidationError):
        ContextGradingConfig(minimum_evidence_count=0)


def test_evidence_selection_defaults_are_bounded_and_optional() -> None:
    """Applicability ranking weights cannot become unbounded or negative."""
    selection = EvidenceSelectionConfig()

    assert selection.enabled is True
    assert selection.reference_match_boost == 2.0
    assert selection.lexical_overlap_weight == 1.0
    assert selection.inactive_penalty == 2.0
    assert selection.max_per_document == 100
    assert selection.max_per_article == 100
    with pytest.raises(ValidationError):
        EvidenceSelectionConfig(reference_match_boost=-1)
    with pytest.raises(ValidationError):
        EvidenceSelectionConfig(lexical_overlap_weight=11)
    with pytest.raises(ValidationError):
        EvidenceSelectionConfig(inactive_penalty=float("inf"))
    with pytest.raises(ValidationError):
        EvidenceSelectionConfig(max_per_document=0)


def test_claim_verification_defaults_are_fail_closed_and_bounded() -> None:
    """Claim grounding requires inline markers, quantities, and negations."""
    config = ClaimVerificationConfig()

    assert config.enabled is True
    assert config.require_inline_citations is True
    assert config.minimum_lexical_support == 0.25
    assert config.require_numeric_match is True
    assert config.require_negation_match is True
    assert config.max_claims == 20
    with pytest.raises(ValidationError):
        ClaimVerificationConfig(minimum_lexical_support=1.1)
    with pytest.raises(ValidationError):
        ClaimVerificationConfig(minimum_claim_tokens=0)
    with pytest.raises(ValidationError):
        ClaimVerificationConfig(max_claims=101)


def test_semantic_verification_is_disabled_and_pinned_when_enabled() -> None:
    """Semantic checking is opt-in and requires complete model provenance."""
    disabled = SemanticVerificationConfig()

    assert disabled.backend == "disabled"
    assert disabled.model_name is None
    assert disabled.max_structured_output_retries == 1
    with pytest.raises(ValidationError, match="pinned model identity"):
        SemanticVerificationConfig(backend="transformers")
    with pytest.raises(ValidationError, match="must not contain"):
        SemanticVerificationConfig(
            model_name="unexpected",
            model_revision="unexpected-revision",
        )
    with pytest.raises(ValidationError, match="endpoint_url"):
        SemanticVerificationConfig(
            backend="openai_compatible",
            model_name="fixture",
            model_revision="revision",
        )
    enabled = SemanticVerificationConfig(
        backend="transformers",
        model_name="fixture",
        model_revision="revision",
        device="cuda",
        torch_dtype="float16",
    )

    provider_config = enabled.as_generation_config()

    assert provider_config.backend == "transformers"
    assert provider_config.temperature == 0.0
    assert provider_config.max_output_tokens == 512
    with pytest.raises(ValueError, match="disabled"):
        disabled.as_generation_config()


def test_reranker_defaults_are_revision_pinned_and_bounded() -> None:
    """The multilingual baseline is reproducible and candidate-bounded."""
    config = RerankerConfig()

    assert config.model_name == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert config.model_revision == "1427fd652930e4ba29e8149678df786c240d8825"
    assert config.device == "cpu"
    assert config.torch_dtype == "float32"
    assert config.batch_size == 8
    assert config.max_length == 512
    assert config.max_candidates == 100
    assert config.input_mode == "legal_context"
    with pytest.raises(ValidationError):
        RerankerConfig(input_mode="unknown")
    with pytest.raises(ValidationError):
        RerankerConfig(model_revision="")
    with pytest.raises(ValidationError):
        RerankerConfig(max_candidates=0)
    with pytest.raises(ValidationError):
        RerankerConfig(max_candidates=101)
    assert RerankerConfig(max_length=2048).max_length == 2048
    with pytest.raises(ValidationError):
        RerankerConfig(max_length=8193)
    with pytest.raises(ValidationError, match="must be set together"):
        RerankerConfig(prompt_name="legal")
    with pytest.raises(ValidationError, match="requires float32"):
        RerankerConfig(torch_dtype="float16")


def test_chunking_config_validates_token_relationships() -> None:
    """Token fallback configuration cannot contain impossible limits."""
    with pytest.raises(ValidationError):
        ChunkingConfig(
            max_tokens=128,
            min_tokens=256,
            overlap_tokens=16,
            tokenizer_name="fixture",
        )


def test_chunking_config_has_bounded_dependency_free_defaults() -> None:
    """Baseline chunking limits and tokenizer identity are explicit."""
    config = ChunkingConfig()
    assert config.max_tokens == 384
    assert config.max_search_tokens == 448
    assert config.min_tokens == 50
    assert config.overlap_tokens == 50
    assert config.tokenizer_name == "unicode_word_v1"
    with pytest.raises(ValidationError):
        ChunkingConfig(tokenizer_name="unknown")
    with pytest.raises(ValidationError):
        ChunkingConfig(max_tokens=128, max_search_tokens=128)


def test_agent_retry_is_capped_at_two() -> None:
    """Configuration enforces accepted decision D014."""
    config = AgentConfig()
    assert config.strategy_order == [
        RetrievalStrategy.HYBRID_RERANK,
        RetrievalStrategy.GRAPH,
        RetrievalStrategy.HYBRID,
    ]
    with pytest.raises(ValidationError):
        AgentConfig(max_retry=3)
    with pytest.raises(ValidationError):
        AgentConfig(strategy_order=["hybrid", "hybrid"])
    with pytest.raises(ValidationError):
        AgentConfig(strategy_order=["rerank"])


def test_artifact_layout_requires_unique_safe_relative_directories(
    tmp_path,
) -> None:
    """Runtime artifact paths cannot escape or collide under their root."""
    with pytest.raises(ValidationError):
        ArtifactConfig(root_path=tmp_path, bm25_directory="../bm25")
    with pytest.raises(ValidationError):
        ArtifactConfig(
            root_path=tmp_path,
            vector_directory="vector",
            vector_serving_directory="vector",
        )
    with pytest.raises(ValidationError):
        ArtifactConfig(
            root_path=tmp_path,
            bm25_directory="same",
            vector_directory="same",
        )


def test_logging_level_uses_standard_library_names() -> None:
    """Logging configuration accepts standard names and rejects unknown levels."""
    assert LoggingConfig(level="warning").level == "WARNING"
    with pytest.raises(ValidationError):
        LoggingConfig(level="verbose")


def test_full_corpus_validation_requires_pinned_expected_counts() -> None:
    """A build cannot claim full-corpus coverage without measurable provenance."""
    with pytest.raises(ValidationError, match="pinned revision"):
        BuildValidationConfig(require_full_corpus=True)
    with pytest.raises(ValidationError, match="expected record counts"):
        BuildValidationConfig(
            require_full_corpus=True,
            require_pinned_dataset_revision=True,
        )
    policy = BuildValidationConfig(
        require_full_corpus=True,
        require_pinned_dataset_revision=True,
        expected_record_counts={
            " metadata ": 153_420,
            "content": 178_665,
            "relationships": 897_890,
        },
    )
    assert policy.expected_record_counts["metadata"] == 153_420
    single_component = BuildValidationConfig(
        require_full_corpus=True,
        require_pinned_dataset_revision=True,
        expected_record_counts={"contexts": 8_500},
    )
    assert single_component.expected_record_counts == {"contexts": 8_500}
    with pytest.raises(ValidationError):
        BuildValidationConfig(report_filename="../validation.json")


def test_vector_checkpoint_interval_is_positive_execution_tuning() -> None:
    """Checkpoint cadence is configurable without changing artifact identity."""
    from legal_agentic_rag.configuration.hashing import canonical_sha256

    frequent = VectorIndexConfig(checkpoint_interval_batches=1)
    sparse = VectorIndexConfig(checkpoint_interval_batches=100)

    assert canonical_sha256(frequent) == canonical_sha256(sparse)
    with pytest.raises(ValidationError):
        VectorIndexConfig(checkpoint_interval_batches=0)


def test_html_cleaning_config_normalizes_exact_noise_tokens() -> None:
    """HTML policy tokens are case-insensitive and cannot be blank."""
    config = HtmlCleaningConfig(noise_class_tokens=frozenset({" Navigation "}))
    assert config.noise_class_tokens == frozenset({"navigation"})
    with pytest.raises(ValidationError):
        HtmlCleaningConfig(remove_tags=frozenset({"script", " "}))
    with pytest.raises(ValidationError, match="nav, script, and style"):
        HtmlCleaningConfig(remove_tags=frozenset({"script", "style"}))


def test_structure_parser_config_bounds_title_lookahead() -> None:
    """Title heuristics stay explicitly bounded before parsing starts."""
    with pytest.raises(ValidationError):
        LegalStructureParserConfig(maximum_title_characters=0)
    with pytest.raises(ValidationError):
        LegalStructureParserConfig(maximum_title_words=101)
