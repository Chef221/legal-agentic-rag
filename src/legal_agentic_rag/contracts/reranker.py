"""Protocol for reranking a bounded retrieval candidate set."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
)


@runtime_checkable
class Reranker(Protocol):
    """Rerank only candidates produced by retrieval."""

    @property
    def model_name(self) -> str:
        """Return the configured reranker model name."""
        ...

    @property
    def model_revision(self) -> str | None:
        """Return the pinned reranker revision when available."""
        ...

    def rerank(
        self, query: RetrievalQuery, candidates: Sequence[RetrievalHit]
    ) -> RetrievalResponse:
        """Return a reranked response for the bounded candidate set."""
        ...
