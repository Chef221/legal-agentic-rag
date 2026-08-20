"""Artifact validation and online application composition roots."""

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
    COMPETITION_REQUIRED_ARTIFACT_TYPES,
    ArtifactSetValidator,
    persist_build_validation_report,
)
from legal_agentic_rag.runtime.online import (
    OnlineRuntime,
    OnlineRuntimeFactory,
)
from legal_agentic_rag.runtime.competition_offline import (
    CompetitionOfflineBuildRuntime,
)

__all__ = [
    "load_artifact_manifest",
    "load_dataset_manifest",
    "load_model_artifact",
    "ModelArtifactWriter",
    "ArtifactSetValidator",
    "COMPETITION_REQUIRED_ARTIFACT_TYPES",
    "CompetitionOfflineBuildRuntime",
    "OnlineRuntime",
    "OnlineRuntimeFactory",
    "persist_dataset_manifest",
    "persist_build_validation_report",
    "persist_model_artifact",
    "stream_model_artifact",
]
