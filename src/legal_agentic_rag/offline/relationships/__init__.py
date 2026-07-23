"""Persistence for dataset-independent normalized legal relationships."""

from legal_agentic_rag.offline.relationships.artifact_store import (
    load_relationship_artifact,
    persist_relationship_artifact,
)

__all__ = ["load_relationship_artifact", "persist_relationship_artifact"]
