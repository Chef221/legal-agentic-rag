"""Tests for dependency-light competition command entry points."""

import os
from pathlib import Path
import subprocess
import sys


def test_competition_cli_import_does_not_load_fastapi() -> None:
    """Answer packaging and scoring must not initialize the serving stack."""
    repository_root = Path(__file__).resolve().parents[3]
    command = (
        "import sys; "
        "import legal_agentic_rag.competition.uit_dsc_2026.cli; "
        "assert not any(name == 'fastapi' or name.startswith('fastapi.') "
        "for name in sys.modules)"
    )
    environment = os.environ.copy()
    source_root = str(repository_root / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing}" if existing else source_root
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
