"""Protocol for reading a dataset without exposing dataset-specific fields."""

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.manifests import DatasetManifest


@runtime_checkable
class DatasetSource(Protocol):
    """Read raw logical streams and report reproducible source provenance."""

    @property
    def dataset_name(self) -> str:
        """Return the stable dataset name."""
        ...

    @property
    def dataset_revision(self) -> str | None:
        """Return the pinned source revision when available."""
        ...

    def iter_records(
        self, limit: int | None = None
    ) -> Iterable[Mapping[str, object]]:
        """Yield raw source records without modifying legal content."""
        ...

    def dataset_manifest(self) -> DatasetManifest:
        """Return source provenance and counts for the completed load."""
        ...
