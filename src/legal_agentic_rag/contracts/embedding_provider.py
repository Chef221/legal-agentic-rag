"""Protocol for backend-neutral document and query embedding."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Create vectors while exposing the identity required by manifests."""

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""
        ...

    @property
    def model_revision(self) -> str | None:
        """Return the pinned embedding model revision when available."""
        ...

    @property
    def dimension(self) -> int:
        """Return the fixed output vector dimension."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed a batch of legal chunk search texts."""
        ...

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed one normalized retrieval query."""
        ...
