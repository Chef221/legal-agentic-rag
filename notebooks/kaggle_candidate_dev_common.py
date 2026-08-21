"""Shared frozen dev-200 evaluator for the retained Kaggle candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata
from zipfile import ZipFile


INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
REPO = WORKING / "legal-agentic-rag"
MILESTONE = os.environ.get("LEGAL_RAG_CANDIDATE", "").casefold()
if MILESTONE not in {"m48", "m49", "m491"}:
    raise RuntimeError(
        "LEGAL_RAG_CANDIDATE must identify a retained candidate: m48, m49, or m491"
    )
CONFIG_TEMPLATE = (
    REPO / f"configs/uit-dsc-2026-task2-{MILESTONE}-qwen3-dev.example.json"
)
CONFIG = WORKING / f"{MILESTONE}-candidate-dev-config.json"
ARTIFACTS = WORKING / "uit-dsc-2026-task2-m45-artifacts"
DEV_QUESTIONS = WORKING / f"{MILESTONE}-dev200.json"
BATCH = WORKING / f"{MILESTONE}-candidate-dev200"
REPORT_PATH = WORKING / f"{MILESTONE}-candidate-dev200-report.json"

TRAIN_SHA256 = "2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988"
SCORER_SHA256 = "4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891"
CONTROL_SCORES = {
    "rouge": 0.258831461359759,
    "meteor": 0.151953642792214,
}
CONTROL_FALLBACK_COUNT = 30


def file_sha256(path: Path) -> str:
    """Hash one local input without loading it entirely into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matching_file(
    filename: str,
    expected_sha256: str | None = None,
) -> Path | None:
    """Select one matching Kaggle input, preferring the active candidate."""
    matches = sorted(INPUT_ROOT.rglob(filename))
    if expected_sha256 is not None:
        matches = [
            path for path in matches if file_sha256(path) == expected_sha256
        ]
    if not matches:
        return None
    preferred = [
        path for path in matches if MILESTONE in str(path).casefold()
    ]
    return (preferred or matches)[0]


def normalize_question(text: str) -> str:
    """Use the frozen leakage-control normalization from the M45 control."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def discover_inputs() -> dict[str, Path | None]:
    """Find official evaluation inputs and the immutable M45 artifact."""
    train = matching_file("train.json", TRAIN_SHA256)
    if train is None:
        raise AssertionError("Không thấy đúng train.json chính thức")
    source_zip = matching_file(f"legal-agentic-rag-{MILESTONE}-source.zip")
    source_directories = [
        path.parent
        for path in INPUT_ROOT.rglob("pyproject.toml")
        if (path.parent / "src/legal_agentic_rag").is_dir()
        and (
            path.parent
            / f"configs/uit-dsc-2026-task2-{MILESTONE}-qwen3-dev.example.json"
        ).is_file()
    ]
    if (
        REPO.is_dir()
        and (REPO / "src/legal_agentic_rag").is_dir()
        and CONFIG_TEMPLATE.is_file()
    ):
        source_directories.append(REPO)
    if source_zip is None and not source_directories:
        raise AssertionError(f"Không thấy source {MILESTONE.upper()} hợp lệ")
    scorer_zip = matching_file(
        "Scoring-Program-Task-LegalQA.zip",
        SCORER_SHA256,
    )
    scorer_directories = [
        path.parent
        for path in INPUT_ROOT.rglob("scoring.py")
        if (path.parent / "metadata.yaml").is_file()
        and (path.parent / "rouge_score/rouge_scorer.py").is_file()
    ]
    if scorer_zip is None and not scorer_directories:
        raise AssertionError("Không thấy scorer BTC")
    artifact_archive = matching_file(
        "uit-dsc-2026-task2-m45-artifacts.tar.gz"
    )
    artifact_checksum = matching_file(
        "uit-dsc-2026-task2-m45-artifacts.tar.gz.sha256"
    )
    artifact_directories = [
        path.parent
        for path in INPUT_ROOT.rglob("build_validation_full_corpus.json")
        if (path.parent / "vector/vectors.npy").is_file()
        and (path.parent / "bm25/index.sqlite3").is_file()
    ]
    if artifact_archive is None and not artifact_directories:
        raise AssertionError("Không thấy artifact M45")
    if artifact_archive is not None and artifact_checksum is None:
        raise AssertionError("Thiếu checksum artifact M45")
    return {
        "train": train,
        "source_zip": source_zip,
        "source_directory": (
            sorted(source_directories)[0] if source_zip is None else None
        ),
        "scorer_zip": scorer_zip,
        "scorer_directory": (
            sorted(scorer_directories)[0] if scorer_zip is None else None
        ),
        "artifact_archive": artifact_archive,
        "artifact_checksum": artifact_checksum,
        "artifact_directory": (
            sorted(artifact_directories)[0]
            if artifact_archive is None
            else None
        ),
    }


def install_source(inputs: dict[str, Path | None]) -> None:
    """Restore the active candidate source and install pinned runtime packages."""
    if not REPO.is_dir():
        source_zip = inputs["source_zip"]
        if source_zip is not None:
            with ZipFile(source_zip) as archive:
                archive.extractall(WORKING)
        else:
            source_directory = inputs["source_directory"]
            if source_directory is None:
                raise AssertionError(f"Source {MILESTONE.upper()} không hợp lệ")
            shutil.copytree(source_directory, REPO)
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
            "nltk==3.7",
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


def restore_artifacts(inputs: dict[str, Path | None]) -> Path:
    """Verify and restore the immutable M45 DB/index without rebuilding it."""
    artifact_archive = inputs["artifact_archive"]
    if artifact_archive is not None:
        checksum = inputs["artifact_checksum"]
        if checksum is None:
            raise AssertionError("Thiếu checksum artifact")
        expected = checksum.read_text(encoding="utf-8").split()[0].casefold()
        actual = file_sha256(artifact_archive)
        print("Artifact SHA-256:", actual)
        if actual != expected:
            raise AssertionError("Artifact checksum không khớp")
        if not ARTIFACTS.is_dir():
            subprocess.run(
                [
                    "tar",
                    "-xzf",
                    str(artifact_archive),
                    "-C",
                    str(WORKING),
                ],
                check=True,
            )
        root = ARTIFACTS
    else:
        root = inputs["artifact_directory"]
        if root is None:
            raise AssertionError("Artifact directory không hợp lệ")
    required = [
        "dataset_manifest.json",
        "legal_chunks/manifest.json",
        "bm25/index.sqlite3",
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


def create_config(artifact_root: Path) -> dict[str, object]:
    """Point the selected online candidate at the restored M45 artifact."""
    payload = json.loads(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    payload["artifacts"]["root_path"] = str(artifact_root)
    CONFIG.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generation = payload["online"]["generation"]
    print("Models giữ nguyên:", payload["offline"]["embedding"]["model_name"])
    print("Reranker:", payload["online"]["reranker"]["model_name"])
    print("Generator:", generation["model_name"])
    print(
        f"{MILESTONE.upper()} style/output/repair:",
        generation["answer_style"],
        generation["max_output_tokens"],
        generation["max_grounding_repair_retries"],
    )
    return payload


def create_dev200(train: Path) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    """Recreate the exact group-safe dev sample used by the M45 control."""
    from legal_agentic_rag.competition.uit_dsc_2026.loader import (
        UitDsc2026DataLoader,
    )

    questions = UitDsc2026DataLoader().load_questions(
        train,
        require_reference_answers=True,
    )
    if len(questions) != 7000:
        raise AssertionError("Train record count thay đổi")
    groups: dict[str, list[object]] = defaultdict(list)
    for question in questions:
        groups[normalize_question(question.question)].append(question)
    split_groups: dict[str, list[object]] = {
        "train": [],
        "dev": [],
        "holdout": [],
    }
    # Historical seed values are immutable experiment identities, not imports of
    # a retired executable candidate.
    split_seed = "uit-dsc-2026-m46-v1"
    for normalized, members in groups.items():
        bucket = int(
            sha256(
                f"{split_seed}\0{normalized}".encode("utf-8")
            ).hexdigest(),
            16,
        ) % 10
        split = "dev" if bucket == 0 else "holdout" if bucket == 1 else "train"
        split_groups[split].extend(members)
    counts = {name: len(values) for name, values in split_groups.items()}
    if counts != {"train": 5617, "dev": 678, "holdout": 705}:
        raise AssertionError(counts)
    dev_sorted = sorted(
        split_groups["dev"],
        key=lambda item: sha256(
            f"m46-dev-sample-v1\0{item.question_id}".encode("utf-8")
        ).hexdigest(),
    )
    dev_payload = {
        item.question_id: {
            "question": item.question,
            "answer": item.reference_answer,
        }
        for item in dev_sorted[:200]
    }
    DEV_QUESTIONS.write_text(
        json.dumps(dev_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source_sha256": file_sha256(train),
        "normalization": "NFC+casefold+whitespace",
        "split_seed": split_seed,
        "record_counts": counts,
        "normalized_question_group_count": len(groups),
        "duplicate_group_count": sum(
            len(values) > 1 for values in groups.values()
        ),
        "dev_sample_count": len(dev_payload),
        "dev_sample_ids_sha256": sha256(
            "\n".join(dev_payload).encode("utf-8")
        ).hexdigest(),
    }
    expected_sample = "694825b5961a90a284ad0364ac4f31a1a85f446519c92274a784c8e2be9a48ad"
    if manifest["dev_sample_ids_sha256"] != expected_sample:
        raise AssertionError("Dev-200 không khớp control M45")
    return dev_payload, manifest


def scorer_directory(inputs: dict[str, Path | None]) -> Path:
    """Resolve the audited official scorer ZIP or its Kaggle extraction."""
    scorer_zip = inputs["scorer_zip"]
    if scorer_zip is not None:
        target = WORKING / "official-scorer"
        if not target.is_dir():
            with ZipFile(scorer_zip) as archive:
                archive.extractall(target)
        return target
    target = inputs["scorer_directory"]
    if target is None:
        raise AssertionError("Scorer directory không hợp lệ")
    expected_members = {
        "scoring.py": "f04843fbfad26d41356506d8e49692a7c8a0ed1b9f065a3a8472fa6398a5aa95",
        "rouge_score/rouge_scorer.py": "9484c5fd05e22cd28b5053bf9de586b3620cf65b0c38052b8690badd51f31d1a",
        "rouge_score/tokenize.py": "dc91cea8f09507f744549160c458031d0956e35fa230c80d01f990eba20a7403",
        "rouge_score/tokenizers.py": "2b7b9dae505ce8739064ba8e4791b871b13aa1f65bc54e25692f1a9ba8600508",
    }
    for relative, expected in expected_members.items():
        if file_sha256(target / relative) != expected:
            raise AssertionError(relative)
    return target


def score_batch(
    inputs: dict[str, Path | None],
    dev_payload: dict[str, dict[str, str]],
    split_manifest: dict[str, object],
) -> dict[str, object]:
    """Score rendered candidate answers with the audited BTC scorer source."""
    import nltk

    scorer = scorer_directory(inputs)
    for module_name in list(sys.modules):
        if module_name == "rouge_score" or module_name.startswith("rouge_score."):
            del sys.modules[module_name]
    sys.path.insert(0, str(scorer))
    spec = importlib.util.spec_from_file_location(
        "uit_dsc_official_scoring",
        scorer / "scoring.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Không load được scorer BTC")
    official_scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_scoring)

    from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
        render_competition_answer,
    )
    from legal_agentic_rag.schemas import CompetitionBatchRecord

    records = [
        CompetitionBatchRecord.model_validate_json(line)
        for line in (BATCH / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(records) != 200:
        raise AssertionError(f"Batch {MILESTONE.upper()} chưa đủ 200 câu")
    predictions = {
        item.question_id: {
            "answer": render_competition_answer(item.response)
        }
        for item in records
    }
    truth = {
        question_id: record["answer"]
        for question_id, record in dev_payload.items()
    }
    score_values = official_scoring.eval_qa(predictions, truth)
    scores = {name: float(value) for name, value in score_values.items()}
    warnings = Counter(
        warning
        for item in records
        for warning in item.response.warnings
    )
    fallback_count = sum(
        item.response.insufficient_evidence for item in records
    )
    answer_lengths = [
        len(predictions[item.question_id]["answer"]) for item in records
    ]
    report = {
        "experiment": f"{MILESTONE}-candidate-dev200",
        "scorer_zip_sha256": SCORER_SHA256,
        "nltk_version": nltk.__version__,
        "scores": scores,
        "control_scores": CONTROL_SCORES,
        "score_delta": {
            name: scores[name] - CONTROL_SCORES[name]
            for name in CONTROL_SCORES
        },
        "record_count": len(records),
        "fallback_count": fallback_count,
        "fallback_delta": fallback_count - CONTROL_FALLBACK_COUNT,
        "fallback_rate": fallback_count / len(records),
        "mean_answer_characters": sum(answer_lengths) / len(answer_lengths),
        "grounding_repair_count": warnings["grounding_repair_attempted"],
        "supported_claim_salvage_count": warnings[
            "supported_claim_salvage_applied"
        ],
        "model_error_retry_count": warnings[
            "generator_model_error_retried"
        ],
        "grounding_repair_unresolved_count": warnings[
            "grounding_repair_unresolved"
        ],
        "extractive_fallback_count": (
            warnings["extractive_fallback_applied"]
            + warnings["generator_model_error_fallback"]
        ),
        "warning_counts": dict(warnings.most_common()),
        "split_manifest": split_manifest,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Execute setup, resumable candidate inference and official scoring."""
    import torch

    inputs = discover_inputs()
    if not torch.cuda.is_available():
        raise AssertionError("Hãy bật GPU Accelerator trong Kaggle Settings")
    torch.ones(1, device="cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    install_source(inputs)

    import legal_agentic_rag
    import nltk
    import sentence_transformers
    import transformers

    if legal_agentic_rag.__version__ != "0.45.0":
        raise AssertionError(legal_agentic_rag.__version__)
    if nltk.__version__ != "3.7":
        raise AssertionError(nltk.__version__)
    if sentence_transformers.__version__ != "5.4.1":
        raise AssertionError(sentence_transformers.__version__)
    if transformers.__version__ != "5.15.0":
        raise AssertionError(transformers.__version__)
    artifact_root = restore_artifacts(inputs)
    create_config(artifact_root)
    train = inputs["train"]
    if train is None:
        raise AssertionError("Train input không hợp lệ")
    dev_payload, split_manifest = create_dev200(train)
    print(f"Đang chạy/resume {MILESTONE.upper()} trên đúng 200 câu dev...")
    subprocess.run(
        [
            "legal-rag-batch",
            "--config",
            str(CONFIG),
            "--questions",
            str(DEV_QUESTIONS),
            "--output",
            str(BATCH),
        ],
        cwd=REPO,
        check=True,
    )
    report = score_batch(inputs, dev_payload, split_manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("REPORT:", REPORT_PATH)
    print("RESULTS:", BATCH / "results.jsonl")


if __name__ == "__main__":
    main()
