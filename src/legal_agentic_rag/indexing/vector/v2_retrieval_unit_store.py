"""Memory-bounded random access to aligned V2 retrieval units."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.schemas.preprocessing_v2 import RetrievalUnitV2

_LOGGER = logging.getLogger(__name__)


class V2RetrievalUnitStore(Sequence[RetrievalUnitV2]):
    """Memory-bounded store for RetrievalUnitV2 JSONL indexed by compact byte offsets."""

    def __init__(
        self,
        *,
        path: Path,
        offsets: np.ndarray,
    ) -> None:
        self._path = Path(path).resolve()
        self._offsets = offsets
        self._len = len(offsets)

    def __len__(self) -> int:
        return self._len

    def get(self, index: int) -> RetrievalUnitV2:
        """Fetch and validate a single RetrievalUnitV2 by row index."""
        if not 0 <= index < self._len:
            raise IndexError(f"Index {index} out of range [0, {self._len})")
        offset = int(self._offsets[index])
        with open(self._path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        if not line:
            raise DataValidationError(f"Unexpected EOF reading row {index} at offset {offset}")
        return RetrievalUnitV2.model_validate_json(line)

    def get_many(self, indexes: Sequence[int]) -> list[RetrievalUnitV2]:
        """Fetch multiple RetrievalUnitV2 records in requested order."""
        for idx in indexes:
            if not 0 <= idx < self._len:
                raise IndexError(f"Index {idx} out of range [0, {self._len})")
        results: list[RetrievalUnitV2] = []
        with open(self._path, "rb") as f:
            for idx in indexes:
                f.seek(int(self._offsets[idx]))
                line = f.readline()
                if not line:
                    raise DataValidationError(f"Unexpected EOF reading row {idx}")
                results.append(RetrievalUnitV2.model_validate_json(line))
        return results

    def __getitem__(self, index: int | slice) -> RetrievalUnitV2 | list[RetrievalUnitV2]:
        if isinstance(index, slice):
            indices = range(*index.indices(self._len))
            return self.get_many(indices)
        return self.get(index)

    def __iter__(self) -> Iterator[RetrievalUnitV2]:
        with open(self._path, "rb") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield RetrievalUnitV2.model_validate_json(s)

    @classmethod
    def load(
        cls,
        units_path: Path,
        *,
        ids_path: Path | None = None,
        expected_count: int | None = None,
        verify_alignment: bool = True,
    ) -> "V2RetrievalUnitStore":
        """Build offset index and perform streaming alignment validation."""
        units_path = Path(units_path).resolve()
        if not units_path.is_file():
            raise ArtifactCompatibilityError(f"Retrieval units file not found at {units_path}")

        offsets: list[int] = []

        if verify_alignment and ids_path is not None:
            ids_path = Path(ids_path).resolve()
            if not ids_path.is_file():
                raise ArtifactCompatibilityError(f"Retrieval unit IDs file not found at {ids_path}")

            with open(units_path, "rb") as f_units, open(ids_path, "rb") as f_ids:
                row_idx = 0
                while True:
                    offset = f_units.tell()
                    u_line = f_units.readline()
                    id_line = f_ids.readline()

                    if not u_line and not id_line:
                        break
                    if bool(u_line) != bool(id_line):
                        raise DataValidationError(
                            f"Stream length mismatch during alignment check at row {row_idx}: "
                            f"units_eof={not u_line}, ids_eof={not id_line}"
                        )

                    u_str = u_line.strip()
                    id_str = id_line.strip()
                    if not u_str and not id_str:
                        continue

                    offsets.append(offset)

                    # Extract retrieval_unit_id from JSON without full object retention
                    try:
                        u_obj = json.loads(u_str.decode("utf-8"))
                        u_id = u_obj.get("retrieval_unit_id")
                    except Exception as e:
                        raise DataValidationError(f"Malformed JSON in units file at row {row_idx}") from e

                    try:
                        expected_id = json.loads(id_str.decode("utf-8"))
                    except Exception as e:
                        raise DataValidationError(f"Malformed JSON in IDs file at row {row_idx}") from e

                    if u_id != expected_id:
                        raise DataValidationError(
                            f"Alignment mismatch at row {row_idx}: units ID '{u_id}' != ids ID '{expected_id}'"
                        )

                    row_idx += 1
        else:
            with open(units_path, "rb") as f_units:
                while True:
                    offset = f_units.tell()
                    line = f_units.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(offset)

        total_count = len(offsets)
        if expected_count is not None and total_count != expected_count:
            raise DataValidationError(
                f"Expected {expected_count} retrieval units, found {total_count}"
            )

        offsets_array = np.array(offsets, dtype=np.int64)
        return cls(path=units_path, offsets=offsets_array)
