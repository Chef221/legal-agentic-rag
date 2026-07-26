"""Validation tests for bounded offline and online settings."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.configuration import (
    AgentConfig,
    ArtifactConfig,
    BM25IndexConfig,
    BuildValidationConfig,
    ChunkingConfig,
    ContextGradingConfig,
    DatasetAuditConfig,
    DatasetSourceConfig,
    DocumentNormalizationConfig,
    EmbeddingConfig,
    GenerationConfig,
    GraphIndexConfig,
    HtmlCleaningConfig,
    LegalStructureParserConfig,
    LoggingConfig,
    OfflineConfig,
    OfflineExecutionConfig,
    RetrievalConfig,
    RelationshipNormalizationConfig,
    RerankerConfig,
    VectorIndexConfig,
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


def test_embedding_and_vector_defaults_are_pinned_and_bounded() -> None:
    """Dense artifacts have reproducible model, dimension, metric, and batching."""
    embedding = EmbeddingConfig()
    vector = VectorIndexConfig()

    assert embedding.model_name == "intfloat/multilingual-e5-small"
    assert embedding.model_revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert embedding.expected_dimension == 384
    assert embedding.device == "cpu"
    assert vector.backend_name == "numpy_flat"
    assert vector.distance_metric == "cosine"
    assert vector.embedding_batch_size == 16
    with pytest.raises(ValidationError):
        EmbeddingConfig(model_revision="")
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


def test_relationship_and_graph_index_configuration_is_explicit() -> None:
    """Graph labels are never guessed and the reference backend is typed."""
    relationship = RelationshipNormalizationConfig(
        relationship_type_mapping={" Sửa đổi ": " amends "}
    )
    assert relationship.relationship_type_mapping == {"Sửa đổi": "amends"}
    assert relationship.reject_self_loops is True
    assert GraphIndexConfig().backend_name == "adjacency_json"
    with pytest.raises(ValidationError):
        RelationshipNormalizationConfig(
            relationship_type_mapping={"Liên quan": " "}
        )


def test_generation_and_context_grading_defaults_are_bounded() -> None:
    """Fixed RAG context limits and structural sufficiency are typed."""
    generation = GenerationConfig()
    grading = ContextGradingConfig()

    assert generation.max_context_tokens == 4096
    assert generation.max_evidence == 8
    assert generation.backend == "extractive"
    assert grading.minimum_evidence_count == 1
    with pytest.raises(ValidationError):
        GenerationConfig(max_evidence=101)
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


def test_reranker_defaults_are_revision_pinned_and_bounded() -> None:
    """The multilingual baseline is reproducible and candidate-bounded."""
    config = RerankerConfig()

    assert config.model_name == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert config.model_revision == "1427fd652930e4ba29e8149678df786c240d8825"
    assert config.device == "cpu"
    assert config.batch_size == 8
    assert config.max_length == 512
    assert config.max_candidates == 100
    with pytest.raises(ValidationError):
        RerankerConfig(model_revision="")
    with pytest.raises(ValidationError):
        RerankerConfig(max_candidates=0)
    with pytest.raises(ValidationError):
        RerankerConfig(max_candidates=101)
    with pytest.raises(ValidationError):
        RerankerConfig(max_length=513)


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
    assert config.max_tokens == 512
    assert config.min_tokens == 50
    assert config.overlap_tokens == 50
    assert config.tokenizer_name == "unicode_word_v1"
    with pytest.raises(ValidationError):
        ChunkingConfig(tokenizer_name="unknown")


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
            bm25_directory="same",
            vector_directory="same",
        )


def test_logging_level_uses_standard_library_names() -> None:
    """Logging configuration accepts standard names and rejects unknown levels."""
    assert LoggingConfig(level="warning").level == "WARNING"
    with pytest.raises(ValidationError):
        LoggingConfig(level="verbose")


def test_dataset_source_requires_distinct_component_configs() -> None:
    """Two logical streams cannot silently point at the same HF config."""
    with pytest.raises(ValidationError):
        DatasetSourceConfig(
            dataset_name="th1nhng0/vietnamese-legal-documents",
            metadata_config="metadata",
            content_config="metadata",
        )


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
    with pytest.raises(ValidationError, match="metadata, content"):
        BuildValidationConfig(
            require_full_corpus=True,
            require_pinned_dataset_revision=True,
            expected_record_counts={"metadata": 153_420},
        )
    with pytest.raises(ValidationError):
        BuildValidationConfig(report_filename="../validation.json")


def test_offline_execution_only_resumes_when_explicitly_enabled() -> None:
    """Partial reuse is opt-in while stage-memory release remains mandatory."""
    assert OfflineExecutionConfig().resume_partial_build is False
    assert OfflineExecutionConfig(
        resume_partial_build=True
    ).release_stage_memory is True
    assert (
        OfflineExecutionConfig().document_processing_progress_interval
        == 1_000
    )
    with pytest.raises(ValidationError):
        OfflineExecutionConfig(release_stage_memory=False)
    with pytest.raises(ValidationError):
        OfflineExecutionConfig(document_processing_progress_interval=0)
    with pytest.raises(ValidationError, match="pinned revision"):
        OfflineConfig(
            dataset=DatasetSourceConfig(dataset_name="fixture"),
            execution=OfflineExecutionConfig(bounded_source_passes=True),
        )


def test_vector_checkpoint_interval_is_positive_execution_tuning() -> None:
    """Checkpoint cadence is configurable without changing artifact identity."""
    from legal_agentic_rag.configuration.hashing import canonical_sha256

    frequent = VectorIndexConfig(checkpoint_interval_batches=1)
    sparse = VectorIndexConfig(checkpoint_interval_batches=100)

    assert canonical_sha256(frequent) == canonical_sha256(sparse)
    with pytest.raises(ValidationError):
        VectorIndexConfig(checkpoint_interval_batches=0)


def test_audit_config_validates_content_thresholds() -> None:
    """Raw content classification thresholds must be internally consistent."""
    with pytest.raises(ValidationError):
        DatasetAuditConfig(
            minimum_content_characters=100,
            maximum_content_characters=10,
        )


def test_normalization_config_strips_and_validates_explicit_mappings() -> None:
    """Canonical mappings cannot contain blank source or target labels."""
    config = DocumentNormalizationConfig(
        effect_status_mapping={" Còn hiệu lực ": " effective "}
    )
    assert config.effect_status_mapping == {"Còn hiệu lực": "effective"}
    with pytest.raises(ValidationError):
        DocumentNormalizationConfig(document_type_mapping={"Luật": " "})


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
