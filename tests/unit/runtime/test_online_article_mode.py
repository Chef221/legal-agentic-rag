"""Unit tests for OnlineRuntimeFactory feature gate: legacy vs M55 Article mode."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.configuration.m55_production import build_m55_online_config
from legal_agentic_rag.configuration.online import ArticleAnswerConfig
from legal_agentic_rag.exceptions import ConfigurationError
from legal_agentic_rag.generation.article_authority import ArticleAuthorityStore
from legal_agentic_rag.runtime.online import OnlineRuntimeFactory
from legal_agentic_rag.schemas import ToolName


def _build_test_app_config(tmp_path: Path, *, article_answer_enabled: bool = False) -> ApplicationConfig:
    cfg_path = Path("configs/uit-dsc-2026-task2-m491-jina35.example.json")
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Point artifacts root to tmp_path
    raw_cfg["artifacts"]["root_path"] = str(tmp_path)

    app_cfg = ApplicationConfig.model_validate(raw_cfg)

    if article_answer_enabled:
        # Create synthetic lookup in the article authority directory
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


def test_legacy_mode_factory_constructs_generation_components(tmp_path: Path) -> None:
    app_cfg = _build_test_app_config(tmp_path, article_answer_enabled=False)

    with patch("legal_agentic_rag.runtime.online.build_generation_components") as mock_build_gen:
        mock_gen = MagicMock()
        mock_ver = MagicMock()
        mock_build_gen.return_value = (mock_gen, mock_ver)

        factory = OnlineRuntimeFactory(
            app_cfg,
            embedding_provider=MagicMock(),
            reranker=MagicMock(),
            context_grader=MagicMock(),
        )

        assert mock_build_gen.called
        assert factory._answer_generator == mock_gen
        assert factory._citation_verifier == mock_ver
        assert factory._article_answer_enabled is False
        assert factory._article_store is None
        assert factory._article_assembler is None


def test_article_mode_factory_skips_qwen_generation_components(tmp_path: Path) -> None:
    app_cfg = _build_test_app_config(tmp_path, article_answer_enabled=True)

    with (
        patch("legal_agentic_rag.runtime.online.build_generation_components") as mock_build_gen,
        patch("legal_agentic_rag.runtime.online.build_answer_generator") as mock_build_ans,
        patch("legal_agentic_rag.runtime.online.build_citation_verifier") as mock_build_cit,
    ):
        factory = OnlineRuntimeFactory(
            app_cfg,
            embedding_provider=MagicMock(),
            reranker=MagicMock(),
            context_grader=MagicMock(),
        )

        # Must NOT call any Qwen / generation component builders
        mock_build_gen.assert_not_called()
        mock_build_ans.assert_not_called()
        mock_build_cit.assert_not_called()

        assert factory._answer_generator is None
        assert factory._citation_verifier is None
        assert factory._article_answer_enabled is True
        assert factory._article_store is not None
        assert len(factory._article_store) == 1
        assert factory._article_assembler is not None


def test_article_mode_rejects_ambiguous_generator_injection(tmp_path: Path) -> None:
    app_cfg = _build_test_app_config(tmp_path, article_answer_enabled=True)

    with pytest.raises(ConfigurationError, match="Cannot provide answer_generator or citation_verifier"):
        OnlineRuntimeFactory(
            app_cfg,
            embedding_provider=MagicMock(),
            reranker=MagicMock(),
            context_grader=MagicMock(),
            answer_generator=MagicMock(),
        )
