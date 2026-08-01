"""Sanitized runtime identity recorded with every evaluation run."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pydantic import BaseModel, JsonValue

from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.configuration.hashing import canonical_sha256


def evaluation_runtime_provenance(
    config: ApplicationConfig,
) -> tuple[str, dict[str, JsonValue]]:
    """Return a stable hash and non-secret identities used by evaluation."""
    generation = _model_config(config.online.generation)
    semantic = _model_config(config.online.semantic_verification)
    components: dict[str, JsonValue] = {
        "embedding": config.offline.embedding.model_dump(mode="json"),
        "bm25": config.offline.bm25.model_dump(mode="json"),
        "vector_index": config.offline.vector_index.model_dump(mode="json"),
        "vector_runtime": config.online.vector_runtime.model_dump(mode="json"),
        "reranker": config.online.reranker.model_dump(mode="json"),
        "generation": generation,
        "semantic_verification": semantic,
        "retrieval": config.online.retrieval.model_dump(mode="json"),
        "query_understanding": config.online.query_understanding.model_dump(
            mode="json"
        ),
        "evidence_selection": config.online.evidence_selection.model_dump(
            mode="json"
        ),
        "claim_verification": config.online.claim_verification.model_dump(
            mode="json"
        ),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "sentence-transformers", "transformers")
        },
    }
    return canonical_sha256(components), components


def _model_config(value: BaseModel) -> dict[str, JsonValue]:
    payload = value.model_dump(mode="json")
    payload.pop("endpoint_url", None)
    payload.pop("api_key_env", None)
    return payload


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
