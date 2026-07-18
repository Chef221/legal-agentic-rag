"""Protocol for persistent vector storage and similarity search."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalResponse


@runtime_checkable
class VectorBackend(Protocol):
    """Store supplied vectors and search them without creating embeddings."""

    def build(
        self,
        chunks: Sequence[LegalChunk],
        vectors: Sequence[Sequence[float]],
    ) -> ArtifactManifest:
        """Build a vector index from aligned chunks and vectors."""
        ...

    def search(
        self, query: RetrievalQuery, query_vector: Sequence[float]
    ) -> RetrievalResponse:
        """Return ranked dense hits for a validated query vector."""
        ...

    def persist(self, destination: Path) -> ArtifactManifest:
        """Persist the current vector index and return its manifest."""
        ...

    def load(self, source: Path, manifest: ArtifactManifest) -> None:
        """Load a compatible persisted vector index."""
        ...
