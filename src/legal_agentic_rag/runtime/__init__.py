"""Offline build and online application composition roots."""

from legal_agentic_rag.runtime.artifact_store import (
    ModelArtifactWriter,
    load_artifact_manifest,
    load_dataset_manifest,
    load_model_artifact,
    persist_dataset_manifest,
    persist_model_artifact,
    stream_model_artifact,
)
from legal_agentic_rag.runtime.build_validation import (
    ArtifactSetValidator,
    persist_build_validation_report,
)
from legal_agentic_rag.runtime.offline import OfflineBuildRuntime
from legal_agentic_rag.runtime.online import (
    OnlineRuntime,
    OnlineRuntimeFactory,
)

__all__ = [
    "load_artifact_manifest",
    "load_dataset_manifest",
    "load_model_artifact",
    "ModelArtifactWriter",
    "ArtifactSetValidator",
    "OfflineBuildRuntime",
    "OnlineRuntime",
    "OnlineRuntimeFactory",
    "persist_dataset_manifest",
    "persist_build_validation_report",
    "persist_model_artifact",
    "stream_model_artifact",
]
