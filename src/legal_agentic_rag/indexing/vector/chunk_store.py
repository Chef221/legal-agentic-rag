"""Memory-bounded random access to aligned vector chunk metadata."""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence
from hashlib import sha256
import logging
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas.legal_documents import LegalChunk
from legal_agentic_rag.schemas.retrieval import RetrievalFilters

_LOGGER = logging.getLogger(__name__)
_FILTER_FIELDS = (
    ("document_ids", "document_id"),
    ("document_types", "document_type"),
    ("legal_fields", "legal_field"),
    ("effect_statuses", "effect_status"),
)


class JsonlChunkStore(Sequence[LegalChunk]):
    """Validated JSONL metadata indexed by compact byte offsets."""

    def __init__(
        self,
        *,
        path: Path,
        offsets: array,
        chunk_ids: list[str],
        filter_indexes: dict[str, dict[str, array]],
    ) -> None:
        self._path = path
        self._offsets = offsets
        self._chunk_ids = chunk_ids
        self._filter_indexes = filter_indexes

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_count: int,
        expected_checksum: object,
        require_sorted_chunk_ids: bool,
        progress_interval_records: int,
        verify_integrity: bool = True,
    ) -> "JsonlChunkStore":
        """Validate JSONL once while retaining only lookup/filter indexes."""
        offsets = array("Q")
        chunk_ids: list[str] = []
        seen_chunk_ids: set[str] = set()
        filter_indexes: dict[str, dict[str, array]] = {
            chunk_field: {} for _, chunk_field in _FILTER_FIELDS
        }
        digest = sha256() if verify_integrity else None
        first_payload_error: Exception | None = None
        previous_chunk_id: str | None = None
        _LOGGER.info(
            "vector_chunk_metadata_load_started",
            extra={"total_chunk_count": expected_count},
        )
        try:
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if digest is not None:
                        digest.update(line)
                    if first_payload_error is not None:
                        continue
                    try:
                        if not line.strip():
                            raise ValueError("blank chunk record")
                        chunk = LegalChunk.model_validate_json(line)
                        if chunk.chunk_id in seen_chunk_ids:
                            raise ValueError("duplicate chunk ID")
                        if (
                            require_sorted_chunk_ids
                            and previous_chunk_id is not None
                            and chunk.chunk_id < previous_chunk_id
                        ):
                            raise ValueError("chunk IDs are not sorted")
                        row_index = len(offsets)
                        offsets.append(offset)
                        chunk_ids.append(chunk.chunk_id)
                        seen_chunk_ids.add(chunk.chunk_id)
                        previous_chunk_id = chunk.chunk_id
                        for _, chunk_field in _FILTER_FIELDS:
                            value = getattr(chunk, chunk_field)
                            if value is None:
                                continue
                            postings = filter_indexes[chunk_field].setdefault(
                                value,
                                array("I"),
                            )
                            postings.append(row_index)
                        if (
                            len(offsets) % progress_interval_records == 0
                        ):
                            _LOGGER.info(
                                "vector_chunk_metadata_load_progress",
                                extra={
                                    "chunk_count": len(offsets),
                                    "total_chunk_count": expected_count,
                                },
                            )
                    except (ValidationError, ValueError) as error:
                        first_payload_error = error
        except OSError as error:
            raise ArtifactCompatibilityError(
                "Vector chunk metadata cannot be read"
            ) from error

        if verify_integrity and (
            not isinstance(expected_checksum, str)
            or digest is None
            or digest.hexdigest() != expected_checksum
        ):
            raise ArtifactCompatibilityError(
                "Vector chunk metadata checksum mismatch"
            )
        if first_payload_error is not None:
            raise ArtifactCompatibilityError(
                "Vector chunk metadata payload is invalid"
            ) from first_payload_error
        if len(offsets) != expected_count:
            raise ArtifactCompatibilityError(
                "Vector artifact shape or record count is incompatible"
            )
        _LOGGER.info(
            "vector_chunk_metadata_load_completed",
            extra={
                "chunk_count": len(offsets),
                "total_chunk_count": expected_count,
            },
        )
        return cls(
            path=path,
            offsets=offsets,
            chunk_ids=chunk_ids,
            filter_indexes=filter_indexes,
        )

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int | slice) -> LegalChunk | list[LegalChunk]:
        if isinstance(index, slice):
            return self.get_many(range(*index.indices(len(self))))
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("chunk index out of range")
        return self.get_many([normalized])[0]

    def __iter__(self) -> Iterator[LegalChunk]:
        try:
            with self._path.open("rb") as stream:
                for line in stream:
                    yield LegalChunk.model_validate_json(line)
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Vector chunk metadata payload is invalid"
            ) from error

    def chunk_id(self, index: int) -> str:
        """Return one retained chunk ID without parsing its full record."""
        return self._chunk_ids[index]

    def get_many(self, indexes: Sequence[int]) -> list[LegalChunk]:
        """Read selected records through one local handle for thread safety."""
        values: list[LegalChunk] = []
        try:
            with self._path.open("rb") as stream:
                for index in indexes:
                    normalized = int(index)
                    if normalized < 0 or normalized >= len(self):
                        raise IndexError("chunk index out of range")
                    stream.seek(self._offsets[normalized])
                    line = stream.readline()
                    if not line:
                        raise ValueError("missing chunk record")
                    values.append(LegalChunk.model_validate_json(line))
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Vector chunk metadata payload is invalid"
            ) from error
        return values

    def filtered_indexes(self, filters: RetrievalFilters) -> np.ndarray | None:
        """Return compact row indexes, or ``None`` when no filter is active."""
        active_filters = [
            (getattr(filters, filter_field), chunk_field)
            for filter_field, chunk_field in _FILTER_FIELDS
            if getattr(filters, filter_field)
        ]
        if not active_filters:
            return None
        combined = np.ones(len(self), dtype=np.bool_)
        for requested_values, chunk_field in active_filters:
            field_mask = np.zeros(len(self), dtype=np.bool_)
            postings_by_value = self._filter_indexes[chunk_field]
            for value in requested_values:
                postings = postings_by_value.get(value)
                if postings:
                    field_mask[np.frombuffer(postings, dtype=np.uint32)] = True
            combined &= field_mask
            if not combined.any():
                return np.empty(0, dtype=np.int64)
        return np.flatnonzero(combined)
