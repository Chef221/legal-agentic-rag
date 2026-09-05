"""Unit tests for Phase 6: V2 production retrieval runtime reconciliation and legacy regression."""

import hashlib
import re
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.configuration.m54_production import build_m54_online_config
from legal_agentic_rag.configuration.m55_production import build_m55_online_config
from legal_agentic_rag.configuration.online import (
    ArticleAnswerConfig,
    OnlineConfig,
    RetrievalArtifactMode,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, ConfigurationError
from legal_agentic_rag.indexing.bm25.v2_backend import V2SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.vector.v2_precomputed_backend import (
    V2PrecomputedDenseBackend,
)
from legal_agentic_rag.retrieval.v2_branches import build_v2_fixed_retriever
from legal_agentic_rag.runtime.online import OnlineRuntime, OnlineRuntimeFactory


def _build_test_app_config(
    tmp_path: Path,
    *,
    retrieval_artifact_mode: RetrievalArtifactMode = RetrievalArtifactMode.LEGACY,
    article_answer_enabled: bool = False,
) -> ApplicationConfig:
    cfg_path = Path("configs/uit-dsc-2026-task2-m491-jina35.example.json")
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw_cfg["artifacts"]["root_path"] = str(tmp_path)

    app_cfg = ApplicationConfig.model_validate(raw_cfg)
    app_cfg.online.retrieval_artifact_mode = retrieval_artifact_mode

    if article_answer_enabled:
        art_dir = tmp_path / app_cfg.artifacts.article_authority_directory
        art_dir.mkdir(parents=True, exist_ok=True)
        lookup_file = art_dir / "lookup.jsonl"
        records = [
            {"document_id": "doc1", "article_identity": "1", "article_text": "Article 1 text"}
        ]
        hasher = hashlib.sha256()
        with open(lookup_file, "w", encoding="utf-8", newline="\n") as f:
            for r in records:
                line = json.dumps(r, ensure_ascii=False) + "\n"
                hasher.update(line.encode("utf-8"))
                f.write(line)

        app_cfg.online.article_answer = ArticleAnswerConfig(
            enabled=True,
            max_articles=2,
            lookup_filename="lookup.jsonl",
            lookup_sha256=hasher.hexdigest(),
            expected_record_count=1,
            structural_fallback_max_evidence=3,
        )
    else:
        app_cfg.online.article_answer = ArticleAnswerConfig(enabled=False)

    return app_cfg


def test_legacy_mode_calls_legacy_backends_and_skips_v2(tmp_path: Path) -> None:
    """In LEGACY mode, build() delegates to legacy loader without calling V2 loaders."""
    app_cfg = _build_test_app_config(tmp_path, retrieval_artifact_mode=RetrievalArtifactMode.LEGACY)

    with (
        patch.object(OnlineRuntimeFactory, "_build_legacy") as mock_legacy,
        patch.object(OnlineRuntimeFactory, "_build_v2") as mock_v2,
    ):
        mock_legacy.return_value = MagicMock(spec=OnlineRuntime)
        factory = OnlineRuntimeFactory(
            app_cfg,
            embedding_provider=MagicMock(),
            reranker=MagicMock(),
            context_grader=MagicMock(),
            answer_generator=MagicMock(),
            citation_verifier=MagicMock(),
        )
        runtime = factory.build()

        assert mock_legacy.called
        assert not mock_v2.called
        assert runtime == mock_legacy.return_value


def test_m55_v2_path_wiring_and_isolation(tmp_path: Path) -> None:
    """In V2_PRECOMPUTED mode with article answering, V2 backends are loaded and legacy ones are NOT created."""
    app_cfg = _build_test_app_config(
        tmp_path,
        retrieval_artifact_mode=RetrievalArtifactMode.V2_PRECOMPUTED,
        article_answer_enabled=True,
    )

    mock_bm25_backend = MagicMock(spec=V2SQLiteFTS5BM25Backend)
    mock_dense_backend = MagicMock(spec=V2PrecomputedDenseBackend)
    mock_retriever = MagicMock()

    with (
        patch("legal_agentic_rag.runtime.online.V2SQLiteFTS5BM25Backend.load", return_value=mock_bm25_backend) as mock_bm25_load,
        patch("legal_agentic_rag.runtime.online.V2PrecomputedDenseBackend.load", return_value=mock_dense_backend) as mock_dense_load,
        patch("legal_agentic_rag.runtime.online.build_v2_fixed_retriever", return_value=mock_retriever) as mock_build_v2,
        patch("legal_agentic_rag.runtime.online.SQLiteFTS5BM25Backend") as mock_legacy_bm25,
        patch("legal_agentic_rag.runtime.online.NumpyVectorBackend") as mock_legacy_vector,
        patch("legal_agentic_rag.runtime.online.AdjacencyGraphBackend") as mock_legacy_graph,
        patch("legal_agentic_rag.runtime.online.build_generation_components") as mock_gen,
    ):
        embed_provider = MagicMock()
        reranker = MagicMock()
        reranker.model_name = "jina-reranker-v3"
        context_grader = MagicMock()

        factory = OnlineRuntimeFactory(
            app_cfg,
            embedding_provider=embed_provider,
            reranker=reranker,
            context_grader=context_grader,
        )

        runtime = factory.build()

        # 1. V2 loaders called with expected paths
        expected_units = tmp_path / app_cfg.artifacts.retrieval_units_v2_directory / "records.jsonl"
        expected_bm25_dir = tmp_path / app_cfg.artifacts.bm25_v2_directory
        expected_dense_dir = tmp_path / app_cfg.artifacts.dense_v2_directory

        mock_bm25_load.assert_called_once_with(
            artifact_dir=expected_bm25_dir,
            units_path=expected_units,
            verify_db_sha=False,
            runtime_config=app_cfg.online.bm25_runtime,
            strict_manifest=True,
        )
        mock_dense_load.assert_called_once_with(
            matrix_dir=expected_dense_dir,
            units_path=expected_units,
            verify_integrity=False,
            strict_manifest=True,
        )

        # 2. build_v2_fixed_retriever called with exact components and configs
        mock_build_v2.assert_called_once_with(
            bm25_backend=mock_bm25_backend,
            dense_backend=mock_dense_backend,
            embedding_provider=embed_provider,
            retrieval_config=app_cfg.online.retrieval,
            query_understanding_config=app_cfg.online.query_understanding,
            reranker=reranker,
            reranker_config=app_cfg.online.reranker,
        )

        # 3. Negative assertions: legacy backends and Qwen are NOT instantiated
        assert not mock_legacy_bm25.called
        assert not mock_legacy_vector.called
        assert not mock_legacy_graph.called
        assert not mock_gen.called

        # 4. Manifests in V2 mode are truthful and empty without fake manifests
        assert runtime.manifests == {}
        assert runtime._retriever == mock_retriever
        assert factory._article_assembler is not None


def test_v2_path_propagates_backend_load_failure(tmp_path: Path) -> None:
    """If a V2 backend manifest or integrity check fails, factory.build() fails fast."""
    app_cfg = _build_test_app_config(
        tmp_path,
        retrieval_artifact_mode=RetrievalArtifactMode.V2_PRECOMPUTED,
        article_answer_enabled=True,
    )

    with patch(
        "legal_agentic_rag.runtime.online.V2SQLiteFTS5BM25Backend.load",
        side_effect=ArtifactCompatibilityError("V2 BM25 manifest invalid"),
    ):
        factory = OnlineRuntimeFactory(
            app_cfg,
            embedding_provider=MagicMock(),
            reranker=MagicMock(),
            context_grader=MagicMock(),
        )
        with pytest.raises(ArtifactCompatibilityError, match="V2 BM25 manifest invalid"):
            factory.build()


def test_a4_authority_wiring_identity() -> None:
    """Verify that M55 production runtime uses the exact same 3 authority components as A4."""
    import inspect
    from legal_agentic_rag.runtime.online import OnlineRuntimeFactory

    # Verify source code of _build_v2 references the exact A4 components
    src = inspect.getsource(OnlineRuntimeFactory._build_v2)

    assert "V2SQLiteFTS5BM25Backend.load" in src
    assert "V2PrecomputedDenseBackend.load" in src
    assert "build_v2_fixed_retriever" in src
    assert "AdjacencyGraphBackend" not in src
    assert not re.search(r"(?<!V2)SQLiteFTS5BM25Backend", src)
    assert "NumpyVectorBackend.load" not in src


def test_m55_online_config_retrieval_parameters_unchanged_from_m54() -> None:
    """M55 switches retrieval_artifact_mode to V2_PRECOMPUTED and sets 60s timeout while keeping all scoring parameters identical to M54."""
    m54 = build_m54_online_config()
    m55 = build_m55_online_config()

    assert m54.retrieval_artifact_mode == RetrievalArtifactMode.LEGACY
    assert m55.retrieval_artifact_mode == RetrievalArtifactMode.V2_PRECOMPUTED

    # Retrieval timeouts: M54 is 30s, M55 is 60s
    assert m54.retrieval.timeout_seconds == 30.0
    assert m55.retrieval.timeout_seconds == 60.0
    assert m55.retrieval.model_copy(update={"timeout_seconds": 30.0}) == m54.retrieval

    # Exact parameter identity across all other sub-configs
    assert m55.bm25_runtime == m54.bm25_runtime
    assert m55.vector_runtime == m54.vector_runtime
    assert m55.reranker == m54.reranker
    assert m55.evidence_selection == m54.evidence_selection
    assert m55.context_grading == m54.context_grading
    assert m55.query_understanding == m54.query_understanding
    assert m55.agent == m54.agent


def test_online_runtime_factory_passes_retrieval_timeout_to_registry(tmp_path: Path) -> None:
    """OnlineRuntimeFactory passes the configured retrieval timeout (60s in M55, 30s in M54) to the tool registry."""
    from legal_agentic_rag.configuration.artifacts import ArtifactConfig
    from legal_agentic_rag.configuration.evaluation import EvaluationConfig
    from legal_agentic_rag.configuration.offline import OfflineConfig
    from legal_agentic_rag.configuration.m55_production import build_m55_embedding_config
    from legal_agentic_rag.generation.article_authority import ArticleAuthorityStore
    from legal_agentic_rag.schemas import ToolName

    # 1. M55 production config (60.0s)
    app_cfg_m55 = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=tmp_path),
        offline=OfflineConfig(embedding=build_m55_embedding_config()),
        online=build_m55_online_config(),
        evaluation=EvaluationConfig(candidate_k=40),
    )

    with (
        patch("legal_agentic_rag.runtime.online.V2SQLiteFTS5BM25Backend.load", return_value=MagicMock(spec=V2SQLiteFTS5BM25Backend)),
        patch("legal_agentic_rag.runtime.online.V2PrecomputedDenseBackend.load", return_value=MagicMock(spec=V2PrecomputedDenseBackend)),
        patch("legal_agentic_rag.runtime.online.build_v2_fixed_retriever", return_value=MagicMock()),
        patch("legal_agentic_rag.runtime.online.ArticleAuthorityStore.from_jsonl", return_value=MagicMock(spec=ArticleAuthorityStore)),
    ):
        factory_m55 = OnlineRuntimeFactory(
            app_cfg_m55,
            embedding_provider=MagicMock(),
            reranker=MagicMock(),
            context_grader=MagicMock(),
        )
        runtime_m55 = factory_m55.build()
        for tool_name in [ToolName.BM25_SEARCH, ToolName.DENSE_SEARCH, ToolName.HYBRID_SEARCH]:
            tool = runtime_m55._registry._tools.get(tool_name)
            assert tool is not None
            assert tool.timeout_seconds == 60.0

    # 2. M54 production config (30.0s)
    app_cfg_m54 = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=tmp_path),
        offline=OfflineConfig(embedding=build_m55_embedding_config()),
        online=build_m54_online_config(),
        evaluation=EvaluationConfig(candidate_k=40),
    )

    with (
        patch.object(OnlineRuntimeFactory, "_validate_manifests"),
        patch.object(OnlineRuntimeFactory, "_validate_embedding_provider"),
        patch("legal_agentic_rag.runtime.online.SQLiteFTS5BM25Backend"),
        patch("legal_agentic_rag.runtime.online.NumpyVectorBackend"),
        patch("legal_agentic_rag.runtime.online.AdjacencyGraphBackend"),
        patch("legal_agentic_rag.runtime.online.FixedRetriever", return_value=MagicMock()),
        patch("legal_agentic_rag.runtime.online.build_generation_components", return_value=(MagicMock(), MagicMock())),
        patch("legal_agentic_rag.runtime.online.load_artifact_manifest", return_value=MagicMock()),
        patch("legal_agentic_rag.runtime.online.validate_competition_artifact_lineage"),
        patch("legal_agentic_rag.runtime.online.validate_startup_report"),
    ):
        factory_m54 = OnlineRuntimeFactory(
            app_cfg_m54,
            embedding_provider=MagicMock(),
            reranker=MagicMock(),
            context_grader=MagicMock(),
        )
        runtime_m54 = factory_m54.build()
        for tool_name in [ToolName.BM25_SEARCH, ToolName.DENSE_SEARCH, ToolName.HYBRID_SEARCH]:
            tool = runtime_m54._registry._tools.get(tool_name)
            assert tool is not None
            assert tool.timeout_seconds == 30.0
