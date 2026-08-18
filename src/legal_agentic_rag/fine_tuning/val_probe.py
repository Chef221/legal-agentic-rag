"""Deterministic VAL generation probe selection, BASE cache generation, and validation."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.development_split import _file_sha256
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.fine_tuning.dataset import SYSTEM_PROMPT
from legal_agentic_rag.schemas import (
    CompetitionQuestion,
    ValProbeBaseManifest,
    ValProbeCaseResult,
    ValProbeManifest,
)

CANONICAL_M50_TRAINING_SHA256 = "0834091ea06dce76d45b693b679b92002c6cf17f82fc8e23f6d413d5155a38c3"
CANONICAL_M50_SFT_TRAIN_SHA256 = "39ae95060c76dce63083d747ce8a12d82d0587907ffe8a093a21ab69c8b19be9"
CANONICAL_M50_SFT_VAL_SHA256 = "545dcbf6119db077373ce3cb8dee0c0da74cb7465dde4a03ea488788e52a715f"
CANONICAL_M50_SCREEN_HOLDOUT_SHA256 = "a165d4a6fba2e2ec460f856a2a67580607d72648f1012fb6dbd5b779c1eb7367"
CANONICAL_M50_SPLIT_MANIFEST_SHA256 = "5dc52813ddcda3124a51aea848058f6008481a65f5b8a167686603416ec82fb7"

VAL_PROBE_FILENAME = "m50-c2-val-probe.json"
VAL_PROBE_MANIFEST_FILENAME = "m50-c2-val-probe-manifest.json"
VAL_PROBE_BASE_RESULTS_FILENAME = "m50-c2-val-probe-base-results.jsonl"
VAL_PROBE_BASE_MANIFEST_FILENAME = "m50-c2-val-probe-base-manifest.json"
VAL_PROBE_SALT = "m50-c2-val-probe-v1"
DEFAULT_VAL_PROBE_QUESTION_COUNT = 20
DEFAULT_VAL_PROBE_MAX_NEW_TOKENS = 512


def compute_repeat_ngram_ratio(text: str, n: int = 8) -> float:
    """Compute fraction of repeated n-grams in text whitespace token sequence."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    total = len(ngrams)
    unique = len(set(ngrams))
    return float((total - unique) / total)


def compute_duplicate_line_ratio(text: str) -> float:
    """Compute fraction of duplicated non-empty lines in text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0
    total = len(lines)
    unique = len(set(lines))
    return float((total - unique) / total)


def create_deterministic_val_probe(
    sft_val_path: Path,
    output_directory: Path,
    *,
    probe_count: int = DEFAULT_VAL_PROBE_QUESTION_COUNT,
    expected_sft_val_sha256: str | None = None,
    split_manifest_path: Path | None = None,
    enforce_canonical_split: bool = True,
    salt: str = VAL_PROBE_SALT,
    system_prompt: str = SYSTEM_PROMPT,
    tokenizer_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    tokenizer_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    generation_max_new_tokens: int = DEFAULT_VAL_PROBE_MAX_NEW_TOKENS,
    loader: UitDsc2026DataLoader | None = None,
) -> tuple[list[CompetitionQuestion], ValProbeManifest]:
    """Deterministically select 20 questions from sft_val.json using salted SHA-256 keys."""
    source_val_sha = _file_sha256(sft_val_path)

    # 1. Content/Lineage Holdout Safeguards
    if enforce_canonical_split:
        if source_val_sha == CANONICAL_M50_SCREEN_HOLDOUT_SHA256:
            raise DataValidationError(
                f"Strict holdout violation: file contents match canonical screen_holdout SHA256 ({source_val_sha})"
            )
        if source_val_sha == CANONICAL_M50_SFT_TRAIN_SHA256:
            raise DataValidationError(
                f"Strict dataset isolation violation: file contents match canonical sft_train SHA256 ({source_val_sha})"
            )

    if split_manifest_path is not None:
        from legal_agentic_rag.schemas import M50SplitManifest

        if not split_manifest_path.exists():
            raise ArtifactCompatibilityError(f"Split manifest not found at {split_manifest_path}")
        s_manifest = M50SplitManifest.model_validate_json(split_manifest_path.read_text(encoding="utf-8"))
        part_map = {p.filename: p.sha256 for p in s_manifest.partitions}
        screen_sha = next((sha for name, sha in part_map.items() if "screen" in name), None)
        val_sha = next((sha for name, sha in part_map.items() if "val" in name), None)
        if screen_sha and source_val_sha == screen_sha:
            raise DataValidationError("Strict holdout violation: file contents match split manifest screen_holdout partition")
        if val_sha and source_val_sha != val_sha:
            raise DataValidationError(
                f"Validation partition SHA mismatch against split manifest: expected {val_sha}, got {source_val_sha}"
            )

    if expected_sft_val_sha256 is not None:
        if source_val_sha != expected_sft_val_sha256:
            raise DataValidationError(
                f"Validation partition SHA256 mismatch: expected {expected_sft_val_sha256}, got {source_val_sha}"
            )

    # Pathname checks as defense-in-depth
    if "screen" in sft_val_path.name.lower() or "holdout" in sft_val_path.name.lower():
        raise DataValidationError(
            f"Strict holdout violation: cannot derive C2 VAL probe from SCREEN dataset {sft_val_path}"
        )
    if "train" in sft_val_path.name.lower() and "sft_val" not in sft_val_path.name.lower():
        raise DataValidationError(
            f"Strict dataset isolation violation: cannot derive VAL probe from training partition {sft_val_path}"
        )

    data_loader = loader or UitDsc2026DataLoader()
    questions = data_loader.load_questions(sft_val_path, require_reference_answers=True)

    if len(questions) < probe_count:
        raise DataValidationError(
            f"Source validation partition {sft_val_path} contains only {len(questions)} questions, required at least {probe_count}"
        )

    # Check question ID uniqueness
    all_ids = [q.question_id for q in questions]
    if len(all_ids) != len(set(all_ids)):
        raise DataValidationError(f"Duplicate question IDs detected in validation partition {sft_val_path}")

    # Compute deterministic salted rank key for each question
    ranked: list[tuple[str, CompetitionQuestion]] = []
    for q in questions:
        key = sha256(f"{salt}:{q.question_id}".encode("utf-8")).hexdigest()
        ranked.append((key, q))

    # Sort ascending by salted hash key to ensure stable order regardless of file ordering
    ranked.sort(key=lambda item: item[0])
    selected = [item[1] for item in ranked[:probe_count]]

    # Ensure output directory exists
    out = output_directory.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Format JSON payload matching official question format
    probe_payload: dict[str, dict[str, str]] = {}
    for q in selected:
        assert q.reference_answer is not None
        probe_payload[q.question_id] = {
            "question": q.question,
            "answer": q.reference_answer,
        }

    probe_json_bytes = json.dumps(probe_payload, indent=2, ensure_ascii=False).encode("utf-8")
    probe_path = out / VAL_PROBE_FILENAME
    probe_path.write_bytes(probe_json_bytes)

    source_val_sha = _file_sha256(sft_val_path)
    probe_sha = sha256(probe_json_bytes).hexdigest()

    manifest = ValProbeManifest(
        created_at=datetime.now(UTC),
        code_version=__version__,
        source_val_sha256=source_val_sha,
        selection_algorithm="sha256_salted_hash_v1",
        salt=salt,
        question_count=len(selected),
        selected_question_ids=[q.question_id for q in selected],
        probe_sha256=probe_sha,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_revision=tokenizer_revision,
        system_prompt=system_prompt,
        diagnostic_generation_config={
            "do_sample": False,
            "max_new_tokens": generation_max_new_tokens,
            "pad_token_id": 151643,
            "eos_token_id": 151643,
        },
        warnings=[],
    )

    (out / VAL_PROBE_MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return selected, manifest


def create_m50_c2_canonical_val_probe(
    sft_val_path: Path,
    output_directory: Path,
    *,
    split_manifest_path: Path | None = None,
    probe_count: int = DEFAULT_VAL_PROBE_QUESTION_COUNT,
    salt: str = VAL_PROBE_SALT,
    system_prompt: str = SYSTEM_PROMPT,
    tokenizer_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    tokenizer_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    generation_max_new_tokens: int = DEFAULT_VAL_PROBE_MAX_NEW_TOKENS,
    loader: UitDsc2026DataLoader | None = None,
) -> tuple[list[CompetitionQuestion], ValProbeManifest]:
    """Deterministically create C2 VAL probe hard-enforcing canonical sft_val SHA-256 and split lineage."""
    return create_deterministic_val_probe(
        sft_val_path=sft_val_path,
        output_directory=output_directory,
        probe_count=probe_count,
        expected_sft_val_sha256=CANONICAL_M50_SFT_VAL_SHA256,
        split_manifest_path=split_manifest_path,
        enforce_canonical_split=True,
        salt=salt,
        system_prompt=system_prompt,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_revision=tokenizer_revision,
        generation_max_new_tokens=generation_max_new_tokens,
        loader=loader,
    )


def generate_and_save_val_probe_base_cache(
    val_probe_path: Path,
    val_probe_manifest_path: Path,
    output_directory: Path,
    *,
    model: Any,
    tokenizer: Any,
    system_prompt: str = SYSTEM_PROMPT,
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    base_model_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    max_new_tokens: int = DEFAULT_VAL_PROBE_MAX_NEW_TOKENS,
    loader: UitDsc2026DataLoader | None = None,
) -> tuple[list[ValProbeCaseResult], ValProbeBaseManifest]:
    """Generate and persist immutable BASE greedy predictions on the 20 VAL probe questions."""
    data_loader = loader or UitDsc2026DataLoader()
    questions = data_loader.load_questions(val_probe_path, require_reference_answers=True)

    manifest_data = json.loads(val_probe_manifest_path.read_text(encoding="utf-8"))
    expected_probe_sha = manifest_data.get("probe_sha256")
    actual_probe_sha = _file_sha256(val_probe_path)
    if expected_probe_sha and actual_probe_sha != expected_probe_sha:
        raise DataValidationError(
            f"VAL probe SHA mismatch: expected {expected_probe_sha}, got {actual_probe_sha}"
        )

    out = output_directory.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / VAL_PROBE_BASE_RESULTS_FILENAME

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    results: list[ValProbeCaseResult] = []
    lines: list[str] = []

    for q in questions:
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)

        t0 = time.perf_counter()
        import torch

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = outputs[0][input_ids.shape[-1] :].tolist()
        gen_token_count = len(gen_tokens)
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        reached_cap = gen_token_count >= max_new_tokens
        eos_emitted = (eos_id in gen_tokens) if eos_id is not None else False
        cap_without_eos = reached_cap and not eos_emitted

        rep8 = compute_repeat_ngram_ratio(gen_text, n=8)
        dup_lines = compute_duplicate_line_ratio(gen_text)

        case = ValProbeCaseResult(
            question_id=q.question_id,
            question=q.question,
            generated_answer=gen_text,
            generated_token_count=gen_token_count,
            reached_cap=reached_cap,
            eos_emitted=eos_emitted,
            cap_without_eos=cap_without_eos,
            repeat_8gram_ratio=rep8,
            duplicate_line_ratio=dup_lines,
            status="success",
            latency_ms=elapsed_ms,
            created_at=datetime.now(UTC),
        )
        results.append(case)
        lines.append(case.model_dump_json())

    # Write incremental/complete JSONL
    results_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    results_path.write_bytes(results_bytes)
    results_sha = sha256(results_bytes).hexdigest()

    # Compute summary health
    cap_without_eos_count = sum(1 for r in results if r.cap_without_eos)
    repeat8_high_count = sum(1 for r in results if r.repeat_8gram_ratio >= 0.25)
    dup_line_high_count = sum(1 for r in results if r.duplicate_line_ratio >= 0.25)
    eos_count = sum(1 for r in results if r.eos_emitted)
    mean_tokens = float(sum(r.generated_token_count for r in results) / len(results))

    manifest = ValProbeBaseManifest(
        created_at=datetime.now(UTC),
        code_version=__version__,
        val_probe_sha256=actual_probe_sha,
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        tokenizer_revision=base_model_revision,
        system_prompt=system_prompt,
        generation_config={
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        },
        results_sha256=results_sha,
        record_count=len(results),
        unique_question_id_count=len(set(r.question_id for r in results)),
        summary_health={
            "cap_without_eos_count": cap_without_eos_count,
            "repeat8_high_count": repeat8_high_count,
            "dup_line_high_count": dup_line_high_count,
            "eos_emitted_count": eos_count,
            "mean_generated_token_count": round(mean_tokens, 2),
        },
        warnings=[],
    )

    manifest_path = out / VAL_PROBE_BASE_MANIFEST_FILENAME
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return results, manifest


def load_and_validate_val_probe_base_cache(
    results_path: Path,
    manifest_path: Path,
    *,
    expected_val_probe_sha256: str,
    expected_base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    expected_base_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    expected_system_prompt: str = SYSTEM_PROMPT,
    expected_max_new_tokens: int = DEFAULT_VAL_PROBE_MAX_NEW_TOKENS,
    expected_record_count: int = DEFAULT_VAL_PROBE_QUESTION_COUNT,
) -> tuple[list[ValProbeCaseResult], ValProbeBaseManifest]:
    """Load and validate cached BASE predictions on the 20 VAL probe questions."""
    if not results_path.exists():
        raise ArtifactCompatibilityError(f"BASE VAL probe results missing at {results_path}")
    if not manifest_path.exists():
        raise ArtifactCompatibilityError(f"BASE VAL probe manifest missing at {manifest_path}")

    manifest = ValProbeBaseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    # Fail closed on any metadata/parameter mismatch
    if manifest.val_probe_sha256 != expected_val_probe_sha256:
        raise ArtifactCompatibilityError(
            f"BASE VAL probe cache SHA mismatch: expected {expected_val_probe_sha256}, got {manifest.val_probe_sha256}"
        )
    if manifest.base_model_id != expected_base_model_id:
        raise ArtifactCompatibilityError(
            f"BASE model ID mismatch: expected {expected_base_model_id}, got {manifest.base_model_id}"
        )
    if manifest.base_model_revision != expected_base_revision:
        raise ArtifactCompatibilityError(
            f"BASE revision mismatch: expected {expected_base_revision}, got {manifest.base_model_revision}"
        )
    if manifest.system_prompt != expected_system_prompt:
        raise ArtifactCompatibilityError(
            f"System prompt mismatch: expected '{expected_system_prompt}', got '{manifest.system_prompt}'"
        )
    if manifest.generation_config.get("max_new_tokens") != expected_max_new_tokens:
        raise ArtifactCompatibilityError(
            f"max_new_tokens mismatch: expected {expected_max_new_tokens}, got {manifest.generation_config.get('max_new_tokens')}"
        )
    if manifest.record_count != expected_record_count:
        raise ArtifactCompatibilityError(
            f"Record count mismatch: expected {expected_record_count}, got {manifest.record_count}"
        )

    # Validate file integrity
    actual_results_sha = _file_sha256(results_path)
    if actual_results_sha != manifest.results_sha256:
        raise ArtifactCompatibilityError(
            f"BASE results SHA corrupted: expected {manifest.results_sha256}, got {actual_results_sha}"
        )

    results: list[ValProbeCaseResult] = []
    seen_ids: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = ValProbeCaseResult.model_validate_json(line)
        if case.question_id in seen_ids:
            raise ArtifactCompatibilityError(f"Duplicate question ID in BASE results: {case.question_id}")
        seen_ids.add(case.question_id)
        results.append(case)

    if len(results) != expected_record_count:
        raise ArtifactCompatibilityError(
            f"Loaded {len(results)} BASE probe cases, expected {expected_record_count}"
        )

    return results, manifest
