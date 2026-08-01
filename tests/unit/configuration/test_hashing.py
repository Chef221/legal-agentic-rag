"""Tests for deterministic configuration identity serialization."""

import os
from pathlib import Path
import subprocess
import sys

from legal_agentic_rag.configuration.hashing import (
    canonical_json,
    canonical_sha256,
)


def test_canonical_hash_sorts_unordered_values_but_preserves_lists() -> None:
    """Set construction order is irrelevant while declared list order matters."""
    first = {
        "labels": frozenset({"zeta", "alpha", "beta"}),
        "route": ["bm25", "dense"],
    }
    second = {
        "route": ["bm25", "dense"],
        "labels": frozenset({"beta", "zeta", "alpha"}),
    }

    assert canonical_json(first) == canonical_json(second)
    assert canonical_sha256(first) == canonical_sha256(second)
    assert canonical_sha256(first) != canonical_sha256(
        {**second, "route": ["dense", "bm25"]}
    )


def test_full_application_config_hash_is_stable_across_hash_seeds() -> None:
    """Independent Python processes must agree on one full-profile identity."""
    root = Path(__file__).parents[3]
    config_path = root / "configs" / "baseline.example.json"
    script = """
from pathlib import Path
import sys
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.serving.config_loader import load_application_config

config = load_application_config(Path(sys.argv[1]))
print(canonical_sha256(config))
"""
    hashes: set[str] = set()
    for seed in ("1", "2", "3", "17"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [sys.executable, "-c", script, str(config_path)],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        hashes.add(result.stdout.strip())

    assert len(hashes) == 1
    assert len(hashes.pop()) == 64
