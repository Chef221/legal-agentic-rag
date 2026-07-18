"""Protocol for a persistent directed document-level legal graph."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.legal_relationships import LegalRelationship
from legal_agentic_rag.schemas.manifests import ArtifactManifest
from legal_agentic_rag.schemas.retrieval import GraphPathStep


@runtime_checkable
class GraphBackend(Protocol):
    """Persist document nodes and traverse directed legal relationships."""

    def build(
        self,
        documents: Iterable[LegalDocument],
        relationships: Iterable[LegalRelationship],
    ) -> ArtifactManifest:
        """Build a document-level graph from normalized inputs."""
        ...

    def traverse(
        self,
        seed_document_ids: Sequence[str],
        max_hops: int,
        relationship_types: Sequence[str] | None = None,
    ) -> Sequence[GraphPathStep]:
        """Return bounded graph paths starting from text-retrieval seeds."""
        ...

    def persist(self, destination: Path) -> ArtifactManifest:
        """Persist the graph and return its manifest."""
        ...

    def load(self, source: Path, manifest: ArtifactManifest) -> None:
        """Load a compatible persisted graph."""
        ...
