"""Sanitized and deterministic evaluation runtime provenance tests."""

import json
from pathlib import Path

from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.evaluation import evaluation_runtime_provenance


def test_runtime_provenance_is_stable_and_excludes_artifact_paths() -> None:
    """Reports identify model/runtime choices without leaking local paths."""
    payload = json.loads(
        Path("configs/baseline.example.json").read_text(encoding="utf-8")
    )
    config = ApplicationConfig.model_validate(payload)

    first_hash, first = evaluation_runtime_provenance(config)
    second_hash, second = evaluation_runtime_provenance(config)

    assert first_hash == second_hash
    assert len(first_hash) == 64
    assert first == second
    assert "artifacts" not in first
    assert first["embedding"]["model_name"] == (
        config.offline.embedding.model_name
    )
    assert first["reranker"]["model_revision"] == (
        config.online.reranker.model_revision
    )
    assert "endpoint_url" not in first["generation"]
    assert "api_key_env" not in first["generation"]
