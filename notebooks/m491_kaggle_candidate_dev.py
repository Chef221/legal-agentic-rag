"""Evaluate the M49.1 runtime repair on the frozen official dev-200 split."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

os.environ["LEGAL_RAG_CANDIDATE"] = "m491"

import kaggle_candidate_dev_common as candidate

MODEL_LINK = candidate.WORKING / "m49-generator-merged"
TRAINING_MANIFEST_NAME = "m49-training-manifest.json"
EXPECTED_MERGED_SHA256 = (
    "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b"
)


def directory_sha256(path: Path) -> str:
    """Recompute the immutable M49 merged-model tree identity."""
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
    """Resolve exactly one complete official-only M49 trained model."""
    valid: list[tuple[Path, dict[str, object]]] = []
    for manifest_path in sorted(
        candidate.INPUT_ROOT.rglob(TRAINING_MANIFEST_NAME)
    ):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = manifest_path.parent / "merged"
        if (
            payload.get("complete") is True
            and payload.get("official_data_only") is True
            and payload.get("synthetic_data_used") is False
            and payload.get("merged_model_sha256") == EXPECTED_MERGED_SHA256
            and merged.is_dir()
        ):
            valid.append((merged, payload))
    if len(valid) != 1:
        raise AssertionError(
            f"Cần đúng 1 trained generator M49, tìm thấy: {len(valid)}"
        )
    merged, manifest = valid[0]
    actual = directory_sha256(merged)
    if actual != EXPECTED_MERGED_SHA256:
        raise AssertionError("Merged generator M49 không khớp checksum")
    if MODEL_LINK.exists() or MODEL_LINK.is_symlink():
        if MODEL_LINK.resolve() != merged.resolve():
            raise AssertionError("Stable M49 model path đang trỏ sai artifact")
    else:
        MODEL_LINK.symlink_to(merged, target_is_directory=True)
    print("M49 merged generator SHA-256:", actual, flush=True)
    return MODEL_LINK, manifest


def create_config(artifact_root: Path) -> dict[str, object]:
    """Bind the M49.1 runtime policy to M45 retrieval and M49 weights."""
    model_path, manifest = restore_generator()
    payload = json.loads(candidate.CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    payload["artifacts"]["root_path"] = str(artifact_root)
    generation = payload["online"]["generation"]
    generation["model_name"] = str(model_path)
    generation["model_revision"] = manifest["merged_model_sha256"]
    generation["local_files_only"] = True
    candidate.CONFIG.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("DB/index: giữ nguyên artifact M45", flush=True)
    print("Generator: giữ nguyên merged M49", flush=True)
    print(
        "M49.1 output/repetition:",
        generation["prompt_schema_mode"],
        generation["repetition_penalty"],
        generation["no_repeat_ngram_size"],
        flush=True,
    )
    return payload


def main() -> None:
    """Run exact dev-200 and promote only on official METEOR improvement."""
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
    print("M49.1 DEV PROMOTION GATE:", promoted, flush=True)
    if not promoted:
        print("Chưa chạy public M49.1; giữ M48 là control.", flush=True)


if __name__ == "__main__":
    main()
