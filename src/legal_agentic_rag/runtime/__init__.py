"""Offline build and online application composition roots."""

from legal_agentic_rag.runtime.artifact_store import (
    load_artifact_manifest,
    persist_dataset_manifest,
    persist_model_artifact,
)
from legal_agentic_rag.runtime.offline import OfflineBuildRuntime
from legal_agentic_rag.runtime.online import (
    OnlineRuntime,
    OnlineRuntimeFactory,
)

__all__ = [
    "load_artifact_manifest",
    "OfflineBuildRuntime",
    "OnlineRuntime",
    "OnlineRuntimeFactory",
    "persist_dataset_manifest",
    "persist_model_artifact",
]
