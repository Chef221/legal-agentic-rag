from pathlib import Path
"""Tests for frozen M55 production runtime configuration."""

from legal_agentic_rag.configuration import (
    EmbeddingConfig,
    M54_PRODUCTION_SCHEMA_VERSION,
    M55_EXPECTED_RECORD_COUNT,
    M55_LOOKUP_FILENAME,
    M55_LOOKUP_SHA256,
    M55_MAX_ARTICLES,
    M55_PRODUCTION_SCHEMA_VERSION,
    M55_RETRIEVAL_TIMEOUT_SECONDS,
    M55_STRUCTURAL_FALLBACK_MAX_EVIDENCE,
    OnlineConfig,
    build_m54_online_config,
    build_m55_embedding_config,
    build_m55_online_config,
)


def test_build_m55_embedding_config_identical_to_m54() -> None:
    m54_emb = build_m54_online_config()
    m55_emb = build_m55_embedding_config()
    assert isinstance(m55_emb, EmbeddingConfig)
    assert m55_emb.model_name == "AITeamVN/Vietnamese_Embedding"
    assert m55_emb.model_revision == "dea33aa1ab339f38d66ae0a40e6c40e0a9249568"


def test_build_m55_online_config_constants_and_version() -> None:
    assert M55_PRODUCTION_SCHEMA_VERSION == "m55_production_v1"
    assert M55_LOOKUP_FILENAME == "m55_a4_full_article_lookup_v1.jsonl"
    assert M55_LOOKUP_SHA256 == "202ba43b02403aa89ad1994a9979702efe1e7436e3b5178e964f65663c926dae"
    assert M55_EXPECTED_RECORD_COUNT == 139073
    assert M55_MAX_ARTICLES == 2
    assert M55_STRUCTURAL_FALLBACK_MAX_EVIDENCE == 3
    assert M55_RETRIEVAL_TIMEOUT_SECONDS == 60.0


def test_build_m55_online_config_article_answer_enabled() -> None:
    cfg = build_m55_online_config()
    assert isinstance(cfg, OnlineConfig)
    art = cfg.article_answer
    assert art.enabled is True
    assert art.max_articles == 2
    assert art.lookup_filename == "m55_a4_full_article_lookup_v1.jsonl"
    assert art.lookup_sha256 == "202ba43b02403aa89ad1994a9979702efe1e7436e3b5178e964f65663c926dae"
    assert art.expected_record_count == 139073
    assert art.structural_fallback_max_evidence == 3


def test_m54_default_article_answer_is_disabled() -> None:
    m54 = build_m54_online_config()
    assert m54.article_answer.enabled is False


def test_m55_retrieval_timeout_is_60s_and_m54_is_30s() -> None:
    m54 = build_m54_online_config()
    m55 = build_m55_online_config()

    assert m54.retrieval.timeout_seconds == 30.0
    assert m55.retrieval.timeout_seconds == 60.0
    # Retrieval configs differ ONLY in timeout_seconds
    assert m55.retrieval.model_copy(update={"timeout_seconds": 30.0}) == m54.retrieval


def test_m55_and_m54_other_subconfigs_and_retrieval_scoring_are_identical() -> None:
    m54 = build_m54_online_config()
    m55 = build_m55_online_config()

    # Retrieval scoring parameters identical
    assert m55.retrieval.top_k == m54.retrieval.top_k == 10
    assert m55.retrieval.candidate_k == m54.retrieval.candidate_k == 40
    assert m55.retrieval.rrf_constant == m54.retrieval.rrf_constant == 60
    assert m55.retrieval.default_strategy == m54.retrieval.default_strategy

    # Assert all sub-configs other than article_answer and retrieval timeout are strictly equal
    assert m55.bm25_runtime == m54.bm25_runtime
    assert m55.vector_runtime == m54.vector_runtime
    assert m55.reranker == m54.reranker
    assert m55.evidence_selection == m54.evidence_selection
    assert m55.context_grading == m54.context_grading
    assert m55.query_understanding == m54.query_understanding
    assert m55.agent == m54.agent
    assert m55.startup_validation == m54.startup_validation

def test_retrieval_artifact_mode_defaults_and_m55_setting() -> None:
    from legal_agentic_rag.configuration.online import RetrievalArtifactMode
    m54 = build_m54_online_config()
    m55 = build_m55_online_config()

    assert m54.retrieval_artifact_mode == RetrievalArtifactMode.LEGACY
    assert m55.retrieval_artifact_mode == RetrievalArtifactMode.V2_PRECOMPUTED


def test_v2_artifact_directories_in_artifact_config(tmp_path: Path) -> None:
    from legal_agentic_rag.configuration.artifacts import ArtifactConfig
    cfg = ArtifactConfig(root_path=tmp_path)

    assert cfg.retrieval_units_v2_directory == "retrieval_units_v2"
    assert cfg.bm25_v2_directory == "bm25_v2"
    assert cfg.dense_v2_directory == "dense_v2"
    assert cfg.directory("retrieval_units_v2_directory") == tmp_path / "retrieval_units_v2"
    assert cfg.directory("bm25_v2_directory") == tmp_path / "bm25_v2"
    assert cfg.directory("dense_v2_directory") == tmp_path / "dense_v2"
