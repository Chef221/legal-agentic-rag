"""Resume M49.1 on official public questions and package submission.zip."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

os.environ["LEGAL_RAG_PUBLIC_CANDIDATE"] = "m491"

import kaggle_public_submission_common as public
from m491_kaggle_candidate_dev import restore_generator

public.CONFIG_TEMPLATE = (
    public.REPO / "configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json"
)
public.CONFIG = public.WORKING / "m491-public-config.json"
public.BATCH = public.WORKING / "m491-public-qwen3-v1"


def install_source() -> None:
    """Restore exact M49.1 source without selecting embedded older copies."""
    source_zips = sorted(
        public.INPUT_ROOT.rglob("legal-agentic-rag-m491-source.zip")
    )
    source_directories = [
        path.parent
        for path in public.INPUT_ROOT.rglob("pyproject.toml")
        if (path.parent / "src/legal_agentic_rag").is_dir()
        and (
            path.parent
            / "configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json"
        ).is_file()
    ]
    if not public.REPO.is_dir():
        if source_zips:
            with ZipFile(source_zips[0]) as archive:
                archive.extractall(public.WORKING)
        elif source_directories:
            shutil.copytree(sorted(source_directories)[0], public.REPO)
        else:
            raise AssertionError("Không thấy source M49.1 trong Kaggle Inputs")
    if not public.CONFIG_TEMPLATE.is_file():
        raise AssertionError(public.CONFIG_TEMPLATE)
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
            str(public.REPO),
            "--no-deps",
        ],
        check=True,
    )
    repo_src = str(public.REPO / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)


def create_config(artifact_root: Path) -> None:
    """Bind M49.1 runtime to immutable M45 artifacts and M49 weights."""
    model_path, manifest = restore_generator()
    payload = json.loads(public.CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    payload["artifacts"]["root_path"] = str(artifact_root)
    generation = payload["online"]["generation"]
    generation["model_name"] = str(model_path)
    generation["model_revision"] = manifest["merged_model_sha256"]
    generation["local_files_only"] = True
    public.CONFIG.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("DB/index: M45; generator weights: M49", flush=True)
    print(
        "M49.1 output/repetition:",
        generation["prompt_schema_mode"],
        generation["repetition_penalty"],
        generation["no_repeat_ngram_size"],
        flush=True,
    )


def main() -> None:
    """Patch the measured public runner with the M49.1 runtime policy."""
    public.install_source = install_source
    public.create_config = create_config
    public.main()


if __name__ == "__main__":
    main()
