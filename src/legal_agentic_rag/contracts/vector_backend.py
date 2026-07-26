"""Protocol for persistent vector storage and similarity search."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.manifests import ArtifactManifest
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalResponse


@dataclass(frozen=True)
class VectorBuildBatch:
    """One bounded aligned batch passed from an embedder to vector storage."""

    chunks: Sequence[LegalChunk]
    vectors: Sequence[Sequence[float]]


VectorBuildBatchFactory = Callable[[int], Iterable[VectorBuildBatch]]


@runtime_checkable
class VectorBackend(Protocol):
    """Store supplied vectors and search them without creating embeddings."""

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        """Return source artifact type, version, and processing hash."""
        ...

    @property
    def embedding_provider_name(self) -> str:
        """Return the provider that created the persisted document vectors."""
        ...

    @property
    def embedding_provider_version(self) -> str:
        """Return the provider version that created document vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Return the embedding model identity associated with the index."""
        ...

    @property
    def model_revision(self) -> str | None:
        """Return the embedding model revision associated with the index."""
        ...

    @property
    def dimension(self) -> int:
        """Return the vector dimension required by the index."""
        ...

    def build(
        self,
        chunks: Sequence[LegalChunk],
        vectors: Sequence[Sequence[float]],
        source_manifest: ArtifactManifest,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_provider_name: str,
        embedding_provider_version: str,
        dimension: int,
        embedding_batch_size: int,
    ) -> ArtifactManifest:
        """Build an aligned vector index with source and model provenance."""
        ...

    def search(
        self, query: RetrievalQuery, query_vector: Sequence[float]
    ) -> RetrievalResponse:
        """Return ranked dense hits for a validated query vector."""
        ...

    def persist(self, destination: Path) -> ArtifactManifest:
        """Persist the current vector index and return its manifest."""
        ...

    def build_persisted(
        self,
        batch_factory: VectorBuildBatchFactory,
        source_manifest: ArtifactManifest,
        destination: Path,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_provider_name: str,
        embedding_provider_version: str,
        dimension: int,
        embedding_batch_size: int,
    ) -> ArtifactManifest:
        """Resume or persist an index from bounded batches after a committed offset."""
        ...

    def load(self, source: Path, manifest: ArtifactManifest) -> None:
        """Load a compatible persisted vector index."""
        ...
