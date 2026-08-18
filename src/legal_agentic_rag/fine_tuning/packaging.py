"""Artifact packaging and SHA256 checksum generation for M50-C2 pilot."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.development_split import _file_sha256
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)

PILOT_PACKAGE_FILENAME = "m50-c2-pilot-complete.zip"
CHECKSUMS_FILENAME = "checksums.sha256.json"


def package_c2_pilot_artifacts(
    pilot_directory: Path,
    output_zip_path: Path,
    *,
    probe_steps: list[int] | None = None,
    allow_no_promotable: bool = True,
) -> dict[str, str]:
    """Validate all required pilot artifacts, compute checksums, and package into complete ZIP."""
    p_dir = pilot_directory.resolve()
    steps = probe_steps or [50, 100, 150]

    # Required file list
    required_files = [
        "training_manifest.json",
        "training_history.json",
        "progress.json",
        "checkpoint-selection-report.json",
    ]

    # Checkpoint directories and probe reports for each step
    for s in steps:
        required_files.append(f"probe-step-{s:04d}.json")
        ckpt_dir = p_dir / f"checkpoint-step-{s:04d}"
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            raise ArtifactCompatibilityError(f"Missing required checkpoint directory: {ckpt_dir.name}")
        if not (ckpt_dir / "checkpoint_manifest.json").exists():
            raise ArtifactCompatibilityError(f"Missing checkpoint_manifest.json in {ckpt_dir.name}")

    # Check required top-level files
    for fname in required_files:
        fpath = p_dir / fname
        if not fpath.exists():
            raise ArtifactCompatibilityError(f"Missing required pilot artifact: {fname}")

    # Collect all files to package and compute SHA256
    checksums: dict[str, str] = {}
    files_to_zip: list[Path] = []

    for item in sorted(p_dir.rglob("*")):
        if item.is_file() and not item.name.endswith(".zip") and item.name != CHECKSUMS_FILENAME:
            rel_path = item.relative_to(p_dir).as_posix()
            file_sha = _file_sha256(item)
            checksums[rel_path] = file_sha
            files_to_zip.append(item)

    # Persist checksum manifest
    checksums_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "code_version": __version__,
        "file_count": len(checksums),
        "checksums": checksums,
    }
    checksums_path = p_dir / CHECKSUMS_FILENAME
    checksums_path.write_text(json.dumps(checksums_manifest, indent=2), encoding="utf-8")
    files_to_zip.append(checksums_path)
    checksums[CHECKSUMS_FILENAME] = _file_sha256(checksums_path)

    # Create ZIP archive
    out_zip = output_zip_path.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_zip:
            rel_arc = f.relative_to(p_dir).as_posix()
            zf.write(f, arcname=rel_arc)

    return checksums
