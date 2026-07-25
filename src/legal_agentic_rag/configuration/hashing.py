"""Deterministic canonical serialization for configuration identities."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel

CanonicalValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["CanonicalValue"]
    | dict[str, "CanonicalValue"]
)


def canonical_sha256(value: object) -> str:
    """Return SHA-256 over a stable JSON representation of typed config data."""
    payload = canonical_json(value)
    return sha256(payload.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    """Serialize mappings, models, and unordered sets deterministically."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonicalize(value: object) -> CanonicalValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("canonical configuration cannot contain NaN or infinity")
        return value
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        normalized: dict[str, CanonicalValue] = {}
        for key, item in value.items():
            normalized_key = _mapping_key(key)
            if normalized_key in normalized:
                raise ValueError("canonical configuration contains duplicate keys")
            normalized[normalized_key] = _canonicalize(item)
        return normalized
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise TypeError(
        f"Unsupported canonical configuration value: {type(value).__name__}"
    )


def _mapping_key(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str):
        raise TypeError("canonical configuration mapping keys must be strings")
    return value
