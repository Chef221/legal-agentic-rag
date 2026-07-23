"""Protocol for a persistent BM25 indexing and search backend."""

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalResponse


@runtime_checkable
class BM25Backend(Protocol):
    """Build, persist, load, and query a BM25 index over legal chunks."""

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return source artifact type, version, and processing hash."""
        ...

    def build(
        self,
        chunks: Iterable[LegalChunk],
        source_manifest: ArtifactManifest,
    ) -> ArtifactManifest:
        """Build from a validated legal-chunks artifact with provenance."""
        ...

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        """Return ranked BM25 hits for a validated query."""
        ...

    def persist(self, destination: Path) -> ArtifactManifest:
        """Persist the current index and return its final manifest."""
        ...

    def load(self, source: Path, manifest: ArtifactManifest) -> None:
        """Load an index only when the supplied manifest is compatible."""
        ...
