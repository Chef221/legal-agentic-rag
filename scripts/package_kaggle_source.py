"""Create a deterministic Kaggle source ZIP from a clean Git checkout."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETAINED_CANDIDATES = {"m48", "m49", "m491"}
ARCHIVE_ROOT = "legal-agentic-rag"


def git_output(*arguments: str) -> bytes:
    """Run a read-only Git query against the project checkout."""
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def tracked_files() -> list[Path]:
    """Return existing tracked files in stable repository-relative order."""
    entries = git_output("ls-files", "-z").split(b"\0")
    paths = [Path(entry.decode("utf-8")) for entry in entries if entry]
    files = [path for path in paths if (PROJECT_ROOT / path).is_file()]
    return sorted(files, key=lambda path: path.as_posix())


def require_clean_checkout() -> None:
    """Refuse to package an uncommitted or untracked source snapshot."""
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError(
            "Git checkout is not clean. Review and commit the intended source first."
        )


def package(candidate: str, output: Path) -> str:
    """Write a deterministic source archive and return its SHA-256."""
    normalized = candidate.casefold()
    if normalized not in RETAINED_CANDIDATES:
        raise ValueError(f"Unsupported retained candidate: {candidate}")
    require_clean_checkout()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in tracked_files():
            data = (PROJECT_ROOT / relative).read_bytes()
            member = ZipInfo(f"{ARCHIVE_ROOT}/{relative.as_posix()}")
            member.date_time = (1980, 1, 1, 0, 0, 0)
            member.compress_type = ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, data, compresslevel=9)

    return sha256(output.read_bytes()).hexdigest()


def main() -> None:
    """Parse CLI arguments and create one retained-candidate source ZIP."""
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=sorted(RETAINED_CANDIDATES))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or (
        PROJECT_ROOT
        / "dist"
        / f"legal-agentic-rag-{arguments.candidate}-source.zip"
    )
    digest = package(arguments.candidate, output.resolve())
    print(output.resolve())
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
