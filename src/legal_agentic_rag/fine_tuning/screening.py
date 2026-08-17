"""Direct question-to-answer generation without RAG for cheap semantic-quality screening."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Callable

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.development_split import _file_sha256
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.fine_tuning.dataset import SYSTEM_PROMPT
from legal_agentic_rag.schemas import (
    CompetitionQuestion,
    DirectQABaseCacheManifest,
    DirectQACaseResult,
    ScreenTokenAuditReport,
)

BASE_CACHE_MANIFEST_FILENAME = "m50-screen-base-manifest.json"
BASE_CACHE_JSONL_FILENAME = "m50-screen-base-results.jsonl"
SCREEN_TOKEN_AUDIT_FILENAME = "m50-screen-token-audit.json"
DEFAULT_DIRECT_QA_MAX_NEW_TOKENS = 1536
SUPPORTED_SCREEN_CAPS = [1536, 1792, 2048, 2560]


def create_and_save_screen_token_audit(
    output_path: Path,
    screen_holdout_path: Path,
    questions: list[CompetitionQuestion],
    tokenizer: Any,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    tokenizer_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    tokenizer_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    supported_caps: list[int] | None = None,
) -> ScreenTokenAuditReport:
    """Audit screening reference lengths and select the smallest 100%-coverage generation cap."""
    caps = supported_caps or SUPPORTED_SCREEN_CAPS
    screen_sha = _file_sha256(screen_holdout_path)

    target_tokens: list[int] = []
    for q in questions:
        if q.reference_answer is None:
            continue
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question},
                {"role": "assistant", "content": q.reference_answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert full_text.startswith(prompt_text)
        target_text = full_text[len(prompt_text):]
        toks = tokenizer.encode(target_text, add_special_tokens=False)
        target_tokens.append(len(toks))

    if not target_tokens:
        raise DataValidationError("No reference answers found to audit in screening dataset")

    s = sorted(target_tokens)
    n = len(s)
    max_tok = s[-1]

    counts_exceeding: dict[str, int] = {}
    for threshold in [512, 768, 1024, 1280, 1536, 1792, 2048, 2560]:
        counts_exceeding[str(threshold)] = sum(1 for tok in s if tok > threshold)

    selected_cap: int | None = None
    for cap in caps:
        if max_tok <= cap:
            selected_cap = cap
            break

    if selected_cap is None:
        raise DataValidationError(
            f"Maximum reference target token length ({max_tok}) exceeds all supported caps: {caps}"
        )

    report = ScreenTokenAuditReport(
        created_at=datetime.now(UTC),
        code_version=__version__,
        screen_holdout_sha256=screen_sha,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_revision=tokenizer_revision,
        system_prompt=system_prompt,
        question_count=len(target_tokens),
        min_tokens=s[0],
        p50_tokens=s[int(n * 0.50)],
        p75_tokens=s[int(n * 0.75)],
        p90_tokens=s[int(n * 0.90)],
        p95_tokens=s[int(n * 0.95)],
        p99_tokens=s[int(n * 0.99)],
        max_tokens=max_tok,
        counts_exceeding=counts_exceeding,
        selected_max_new_tokens=selected_cap,
        warnings=[],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def load_and_validate_screen_token_audit(
    audit_path: Path,
    screen_holdout_path: Path,
    *,
    expected_tokenizer_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    expected_question_count: int = 617,
) -> int:
    """Load and verify token audit against screening data hash and return the validated cap."""
    if not audit_path.exists():
        raise ArtifactCompatibilityError(f"Screen token audit missing at {audit_path}")
    if not screen_holdout_path.exists():
        raise ArtifactCompatibilityError(f"Screen holdout dataset missing at {screen_holdout_path}")

    report = ScreenTokenAuditReport.model_validate_json(
        audit_path.read_text(encoding="utf-8")
    )

    actual_screen_sha = _file_sha256(screen_holdout_path)
    if report.screen_holdout_sha256 != actual_screen_sha:
        raise ArtifactCompatibilityError(
            f"Screen token audit mismatch with screen_holdout.json: expected {report.screen_holdout_sha256}, got {actual_screen_sha}"
        )

    if report.tokenizer_revision != expected_tokenizer_revision:
        raise ArtifactCompatibilityError(
            f"Screen token audit tokenizer revision mismatch: expected {expected_tokenizer_revision}, got {report.tokenizer_revision}"
        )

    if report.question_count != expected_question_count:
        raise ArtifactCompatibilityError(
            f"Screen token audit question count mismatch: expected {expected_question_count}, got {report.question_count}"
        )

    if report.selected_max_new_tokens not in SUPPORTED_SCREEN_CAPS:
        raise ArtifactCompatibilityError(
            f"Invalid selected_max_new_tokens {report.selected_max_new_tokens}; must be in {SUPPORTED_SCREEN_CAPS}"
        )

    return report.selected_max_new_tokens


def load_cached_direct_qa_results(path: Path) -> list[DirectQACaseResult]:
    """Load cached direct QA generation results from a JSONL file."""
    if not path.exists():
        raise ArtifactCompatibilityError(f"Cached direct QA results not found at: {path}")

    results: list[DirectQACaseResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                results.append(DirectQACaseResult.model_validate(data))
            except Exception as err:
                raise DataValidationError(
                    f"Corrupted record at line {line_num} in {path}: {err}"
                ) from err
    return results


def save_base_direct_qa_cache(
    jsonl_output_path: Path,
    manifest_output_path: Path,
    results: list[DirectQACaseResult],
    *,
    screen_holdout_path: Path,
    base_model_id: str,
    base_model_revision: str,
    tokenizer_revision: str,
    system_prompt: str = SYSTEM_PROMPT,
    generation_config: dict[str, Any] | None = None,
) -> DirectQABaseCacheManifest:
    """Persist validated BASE direct QA predictions and an immutable provenance manifest."""
    jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    results_sha = _file_sha256(jsonl_output_path)
    screen_sha = _file_sha256(screen_holdout_path)
    unique_ids = {r.question_id for r in results}

    manifest = DirectQABaseCacheManifest(
        created_at=datetime.now(UTC),
        code_version=__version__,
        screen_holdout_sha256=screen_sha,
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        tokenizer_revision=tokenizer_revision,
        system_prompt=system_prompt,
        generation_config=generation_config or {"max_new_tokens": DEFAULT_DIRECT_QA_MAX_NEW_TOKENS, "do_sample": False},
        results_sha256=results_sha,
        record_count=len(results),
        unique_question_id_count=len(unique_ids),
        warnings=[],
    )

    manifest_output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_and_validate_base_direct_qa_cache(
    jsonl_path: Path,
    manifest_path: Path,
    screen_holdout_path: Path,
    *,
    expected_base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    expected_base_model_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    expected_tokenizer_revision: str = "a1d308dfcc03e09da285d49d912439a655a571e8",
    expected_system_prompt: str = SYSTEM_PROMPT,
    expected_max_new_tokens: int | None = None,
    expected_do_sample: bool = False,
    expected_record_count: int = 617,
) -> list[DirectQACaseResult]:
    """Load and verify BASE cache against screen_holdout.json hash and manifest contract."""
    if not manifest_path.exists():
        raise ArtifactCompatibilityError(f"BASE cache manifest missing at {manifest_path}")
    if not jsonl_path.exists():
        raise ArtifactCompatibilityError(f"BASE cache JSONL missing at {jsonl_path}")
    if not screen_holdout_path.exists():
        raise ArtifactCompatibilityError(f"Screen holdout dataset missing at {screen_holdout_path}")

    manifest = DirectQABaseCacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    # Verify screen_holdout hash matches
    actual_screen_sha = _file_sha256(screen_holdout_path)
    if manifest.screen_holdout_sha256 != actual_screen_sha:
        raise ArtifactCompatibilityError(
            f"BASE cache was generated from different screening data: expected {manifest.screen_holdout_sha256}, got {actual_screen_sha}"
        )

    # Verify results JSONL hash matches manifest
    actual_results_sha = _file_sha256(jsonl_path)
    if manifest.results_sha256 != actual_results_sha:
        raise ArtifactCompatibilityError(
            f"BASE cache JSONL corrupted: expected {manifest.results_sha256}, got {actual_results_sha}"
        )

    # Verify model and tokenizer identity
    if manifest.base_model_id != expected_base_model_id:
        raise ArtifactCompatibilityError(
            f"BASE cache model ID mismatch: expected {expected_base_model_id}, got {manifest.base_model_id}"
        )
    if manifest.base_model_revision != expected_base_model_revision:
        raise ArtifactCompatibilityError(
            f"BASE cache model revision mismatch: expected {expected_base_model_revision}, got {manifest.base_model_revision}"
        )
    if manifest.tokenizer_revision != expected_tokenizer_revision:
        raise ArtifactCompatibilityError(
            f"BASE cache tokenizer revision mismatch: expected {expected_tokenizer_revision}, got {manifest.tokenizer_revision}"
        )
    if manifest.system_prompt != expected_system_prompt:
        raise ArtifactCompatibilityError("BASE cache system prompt mismatch")

    # Verify generation config parameters
    gen_cfg = manifest.generation_config or {}
    if expected_max_new_tokens is not None:
        if gen_cfg.get("max_new_tokens") != expected_max_new_tokens:
            raise ArtifactCompatibilityError(
                f"BASE cache max_new_tokens mismatch: expected {expected_max_new_tokens}, got {gen_cfg.get('max_new_tokens')}"
            )
    if gen_cfg.get("do_sample", False) != expected_do_sample:
        raise ArtifactCompatibilityError(
            f"BASE cache do_sample mismatch: expected {expected_do_sample}, got {gen_cfg.get('do_sample')}"
        )

    # Verify record counts
    if manifest.record_count != expected_record_count:
        raise ArtifactCompatibilityError(
            f"BASE cache manifest record count mismatch: expected {expected_record_count}, got {manifest.record_count}"
        )
    if manifest.unique_question_id_count != expected_record_count:
        raise ArtifactCompatibilityError(
            f"BASE cache manifest unique ID count mismatch: expected {expected_record_count}, got {manifest.unique_question_id_count}"
        )

    results = load_cached_direct_qa_results(jsonl_path)
    if len(results) != expected_record_count:
        raise ArtifactCompatibilityError(
            f"BASE cache count mismatch: expected {expected_record_count}, loaded {len(results)}"
        )
    unique_loaded_ids = {r.question_id for r in results}
    if len(unique_loaded_ids) != expected_record_count:
        raise ArtifactCompatibilityError(
            f"BASE cache duplicate IDs detected: expected {expected_record_count} unique, found {len(unique_loaded_ids)}"
        )

    return results


class DirectQAScreeningRunner:
    """Execute deterministic direct QA generation for BASE or TREATMENT candidates."""

    def __init__(
        self,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = DEFAULT_DIRECT_QA_MAX_NEW_TOKENS,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

    def run(
        self,
        questions: list[CompetitionQuestion],
        generator_fn: Callable[[str], tuple[str, int, bool]],
        output_path: Path,
        *,
        model_id: str,
        model_revision: str,
        overwrite: bool = False,
    ) -> list[DirectQACaseResult]:
        """Generate and persist direct QA answers without retrieval or reranking."""
        if output_path.exists() and not overwrite:
            raise ArtifactCompatibilityError(
                f"Direct QA output file already exists at {output_path}. Use overwrite=True to replace."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results: list[DirectQACaseResult] = []

        with open(output_path, "w", encoding="utf-8") as f:
            for q in questions:
                start_time = time.perf_counter()
                try:
                    generated_text, token_count, hit_max = generator_fn(q.question)
                    status = "success"
                except Exception as err:
                    generated_text = f"GENERATION_ERROR: {err}"
                    token_count = 0
                    hit_max = False
                    status = "error"

                latency_ms = (time.perf_counter() - start_time) * 1000.0

                result = DirectQACaseResult(
                    question_id=q.question_id,
                    question=q.question,
                    generated_answer=generated_text,
                    generated_token_count=token_count,
                    hit_max_tokens=hit_max,
                    status=status,
                    model_id=model_id,
                    model_revision=model_revision,
                    latency_ms=latency_ms,
                    created_at=datetime.now(UTC),
                )
                results.append(result)
                f.write(result.model_dump_json() + "\n")
                f.flush()

        return results
