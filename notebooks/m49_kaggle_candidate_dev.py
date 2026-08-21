"""Evaluate the official-only M49 fine-tuned generator on frozen dev-200."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

os.environ["LEGAL_RAG_CANDIDATE"] = "m49"

import kaggle_candidate_dev_common as candidate

MODEL_LINK = candidate.WORKING / "m49-generator-merged"
TRAINING_MANIFEST_NAME = "m49-training-manifest.json"


def directory_sha256(path: Path) -> str:
    """Recompute the M49 merged-model tree identity before evaluation."""
    digest = sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise AssertionError(f"Merged model M49 rỗng: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(candidate.file_sha256(item)))
    return digest.hexdigest()


def restore_generator() -> tuple[Path, dict[str, object]]:
    """Resolve one complete trained model and expose it at a stable path."""
    manifests = sorted(candidate.INPUT_ROOT.rglob(TRAINING_MANIFEST_NAME))
    valid: list[tuple[Path, dict[str, object]]] = []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = manifest_path.parent / "merged"
        if (
            payload.get("complete") is True
            and payload.get("official_data_only") is True
            and payload.get("synthetic_data_used") is False
            and merged.is_dir()
        ):
            valid.append((merged, payload))
    if len(valid) != 1:
        raise AssertionError(f"Cần đúng 1 trained generator M49, tìm thấy: {len(valid)}")
    merged, manifest = valid[0]
    actual = directory_sha256(merged)
    if actual != manifest.get("merged_model_sha256"):
        raise AssertionError("Merged generator M49 checksum không khớp manifest")
    if MODEL_LINK.exists() or MODEL_LINK.is_symlink():
        if MODEL_LINK.resolve() != merged.resolve():
            raise AssertionError("Stable M49 model path đang trỏ sai artifact")
    else:
        MODEL_LINK.symlink_to(merged, target_is_directory=True)
    print("M49 merged generator SHA-256:", actual, flush=True)
    return MODEL_LINK, manifest


def create_config(artifact_root: Path) -> dict[str, object]:
    """Bind M48 answer pipeline to the newly trained M49 generator revision."""
    model_path, training_manifest = restore_generator()
    payload = json.loads(candidate.CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    payload["artifacts"]["root_path"] = str(artifact_root)
    generation = payload["online"]["generation"]
    generation["model_name"] = str(model_path)
    generation["model_revision"] = training_manifest["merged_model_sha256"]
    generation["local_files_only"] = True
    candidate.CONFIG.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Embedding:", payload["offline"]["embedding"]["model_name"], flush=True)
    print("Reranker:", payload["online"]["reranker"]["model_name"], flush=True)
    print("Generator M49:", generation["model_revision"], flush=True)
    print("DB/index: giữ nguyên artifact M45", flush=True)
    return payload


def main() -> None:
    """Run M49 against the exact M48 dev control and official scorer."""
    candidate.CONTROL_SCORES = {
        "rouge": 0.3660979621381608,
        "meteor": 0.26762720229432313,
    }
    candidate.CONTROL_FALLBACK_COUNT = 2
    candidate.create_config = create_config
    candidate.main()
    report = json.loads(candidate.REPORT_PATH.read_text(encoding="utf-8"))
    promoted = (
        report["score_delta"]["meteor"] > 0
        and report["fallback_count"] <= 5
        and report["warning_counts"].get("generator:model_error", 0) == 0
    )
    print("M49 DEV PROMOTION GATE:", promoted, flush=True)
    if not promoted:
        print("Không chạy public M49; giữ M48 làm control.", flush=True)


if __name__ == "__main__":
    main()
