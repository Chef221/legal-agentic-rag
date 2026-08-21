"""Shared resumable public runner for retained Kaggle candidates."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile


INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
REPO = WORKING / "legal-agentic-rag"
MILESTONE = os.environ.get("LEGAL_RAG_PUBLIC_CANDIDATE", "").casefold()
if MILESTONE not in {"m48", "m491"}:
    raise RuntimeError(
        "LEGAL_RAG_PUBLIC_CANDIDATE must identify a retained public runner: m48 or m491"
    )
CONFIG_TEMPLATE = (
    REPO / f"configs/uit-dsc-2026-task2-{MILESTONE}-qwen3-dev.example.json"
)
CONFIG = WORKING / f"{MILESTONE}-public-config.json"
ARTIFACTS = WORKING / "uit-dsc-2026-task2-m45-artifacts"
BATCH = WORKING / f"{MILESTONE}-public-qwen3-v1"
SUBMISSION = WORKING / "submission.zip"
PUBLIC_SHA256 = "5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5"
_ORPHAN_ENUMERATION_PATTERN = re.compile(
    r":\s*(?:\d+|[a-zđ])\s*\[E\d+\]",
    flags=re.IGNORECASE,
)


def file_sha256(path: Path) -> str:
    """Hash a local input without loading it entirely into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matching_file(filename: str, expected_sha256: str | None = None) -> Path:
    """Select one Kaggle input by exact filename and optional content hash."""
    matches = sorted(INPUT_ROOT.rglob(filename))
    if expected_sha256 is not None:
        matches = [
            path for path in matches if file_sha256(path) == expected_sha256
        ]
    if not matches:
        raise AssertionError(f"Không thấy input hợp lệ: {filename}")
    preferred = [path for path in matches if MILESTONE in str(path).casefold()]
    return (preferred or matches)[0]


def install_source() -> None:
    """Restore and install the active candidate source from Kaggle Inputs."""
    source_zips = sorted(
        INPUT_ROOT.rglob(f"legal-agentic-rag-{MILESTONE}-source.zip")
    )
    source_directories = [
        path.parent
        for path in INPUT_ROOT.rglob("pyproject.toml")
        if (path.parent / "src/legal_agentic_rag").is_dir()
        and (
            path.parent
            / f"configs/uit-dsc-2026-task2-{MILESTONE}-qwen3-dev.example.json"
        ).is_file()
    ]
    if not REPO.is_dir():
        if source_zips:
            with ZipFile(source_zips[0]) as archive:
                archive.extractall(WORKING)
        elif source_directories:
            shutil.copytree(sorted(source_directories)[0], REPO)
        else:
            raise AssertionError("Không thấy source M48 trong Kaggle Inputs")
    if not CONFIG_TEMPLATE.is_file():
        raise AssertionError(CONFIG_TEMPLATE)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "transformers==5.15.0",
            "sentence-transformers==5.4.1",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            str(REPO),
            "--no-deps",
        ],
        check=True,
    )
    repo_src = str(REPO / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)


def restore_artifacts() -> Path:
    """Verify and restore the immutable M45 artifact without rebuilding it."""
    archives = sorted(
        INPUT_ROOT.rglob("uit-dsc-2026-task2-m45-artifacts.tar.gz")
    )
    valid_archives: list[Path] = []
    for archive in archives:
        checksum = archive.with_name(f"{archive.name}.sha256")
        if not checksum.is_file():
            continue
        expected = checksum.read_text(encoding="utf-8").split()[0].casefold()
        if file_sha256(archive) == expected:
            valid_archives.append(archive)
    if valid_archives:
        archive = valid_archives[0]
        print("Artifact SHA-256:", file_sha256(archive))
        if not ARTIFACTS.is_dir():
            subprocess.run(
                ["tar", "-xzf", str(archive), "-C", str(WORKING)],
                check=True,
            )
        root = ARTIFACTS
    else:
        directories = [
            path.parent
            for path in INPUT_ROOT.rglob("build_validation_full_corpus.json")
            if (path.parent / "vector/vectors.npy").is_file()
            and (path.parent / "bm25/index.sqlite3").is_file()
        ]
        if not directories:
            raise AssertionError("Không thấy artifact M45 hợp lệ")
        root = sorted(directories)[0]
        print("Dùng artifact directory:", root)
    required = [
        "dataset_manifest.json",
        "legal_chunks/manifest.json",
        "bm25/index.sqlite3",
        "bm25/manifest.json",
        "vector/vectors.npy",
        "vector/chunks.jsonl",
        "vector/manifest.json",
        "vector_serving/metadata.sqlite3",
        "vector_serving/manifest.json",
        "build_validation_full_corpus.json",
    ]
    for relative in required:
        if not (root / relative).is_file():
            raise AssertionError(root / relative)
    validation = json.loads(
        (root / "build_validation_full_corpus.json").read_text(
            encoding="utf-8"
        )
    )
    if validation["is_valid"] is not True:
        raise AssertionError("Artifact validation report không hợp lệ")
    return root


def create_config(artifact_root: Path) -> None:
    """Point the active online profile at the immutable artifact."""
    payload = json.loads(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    payload["artifacts"]["root_path"] = str(artifact_root)
    CONFIG.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generation = payload["online"]["generation"]
    print(
        "Models:",
        payload["offline"]["embedding"]["model_name"],
        payload["online"]["reranker"]["model_name"],
        generation["model_name"],
    )
    print(
        f"{MILESTONE.upper()} style/output/recovery:",
        generation["answer_style"],
        generation["max_output_tokens"],
        generation["grounding_failure_policy"],
        generation["model_failure_policy"],
    )


def validate_batch() -> dict[str, int]:
    """Require all public IDs and summarize recovery outcomes."""
    records = [
        json.loads(line)
        for line in (BATCH / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    manifest = json.loads(
        (BATCH / "manifest.json").read_text(encoding="utf-8")
    )
    if len(records) != 1000:
        raise AssertionError(f"Batch mới có {len(records)}/1000 câu")
    ids = {record["question_id"] for record in records}
    if len(ids) != 1000 or manifest["record_count"] != 1000:
        raise AssertionError("Public batch thiếu hoặc trùng question ID")
    responses = [record["response"] for record in records]
    warnings = Counter(
        warning for response in responses for warning in response["warnings"]
    )
    summary = {
        "records": len(records),
        "fallbacks": sum(
            response["insufficient_evidence"] for response in responses
        ),
        "citation_failures": warnings["citation_verification_failed"],
        "salvaged": warnings["supported_claim_salvage_applied"],
        "extractive_fallbacks": (
            warnings["extractive_fallback_applied"]
            + warnings["generator_model_error_fallback"]
        ),
        "model_retries": warnings["generator_model_error_retried"],
        "unresolved_repairs": warnings["grounding_repair_unresolved"],
        "generator_errors": warnings["generator:model_error"],
        "orphan_enumerations": sum(
            bool(_ORPHAN_ENUMERATION_PATTERN.search(response["answer"]))
            for response in responses
        ),
    }
    if not all(response["answer"].strip() for response in responses):
        raise AssertionError("Batch chứa answer rỗng")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def restore_batch_checkpoint() -> None:
    """Restore a saved checkpoint from a prior Kaggle notebook output."""
    if BATCH.is_dir():
        return
    saved = [
        path.parent
        for path in INPUT_ROOT.rglob("manifest.json")
        if path.parent.name == BATCH.name
        and (path.parent / "results.jsonl").is_file()
    ]
    if saved:
        source = sorted(saved)[0]
        print(f"Khôi phục checkpoint {MILESTONE.upper()}:", source)
        shutil.copytree(source, BATCH)


def main() -> None:
    """Install the candidate, resume its batch and package submission.zip."""
    import torch

    if not torch.cuda.is_available():
        raise AssertionError("Hãy bật GPU Accelerator trong Kaggle Settings")
    torch.ones(1, device="cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    questions = matching_file("public-official.json", PUBLIC_SHA256)
    install_source()

    import legal_agentic_rag
    import sentence_transformers
    import transformers

    if legal_agentic_rag.__version__ != "0.45.0":
        raise AssertionError(legal_agentic_rag.__version__)
    if sentence_transformers.__version__ != "5.4.1":
        raise AssertionError(sentence_transformers.__version__)
    if transformers.__version__ != "5.15.0":
        raise AssertionError(transformers.__version__)
    artifact_root = restore_artifacts()
    create_config(artifact_root)
    restore_batch_checkpoint()
    print(f"Đang chạy/resume {MILESTONE.upper()} public 1.000 câu...")
    subprocess.run(
        [
            "legal-rag-batch",
            "--config",
            str(CONFIG),
            "--questions",
            str(questions),
            "--output",
            str(BATCH),
        ],
        cwd=REPO,
        check=True,
    )
    validate_batch()
    if SUBMISSION.exists():
        raise AssertionError(
            "submission.zip đã tồn tại; hãy đổi tên/xóa file cũ rồi chạy lại"
        )
    subprocess.run(
        [
            "legal-rag-submit",
            "--questions",
            str(questions),
            "--batch",
            str(BATCH),
            "--output",
            str(SUBMISSION),
        ],
        cwd=REPO,
        check=True,
    )
    with ZipFile(SUBMISSION) as archive:
        if archive.namelist() != ["submission.json"]:
            raise AssertionError(archive.namelist())
    print("SUBMISSION:", SUBMISSION)
    print("SHA-256:", file_sha256(SUBMISSION))


if __name__ == "__main__":
    main()
