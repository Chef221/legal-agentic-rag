"""T5-6B Controlled Output-Contract Measurement Runner and Telemetry Framework.

Executes same-run controlled measurement across 3 candidate arms:
- control: plain_text_markers
- compact: compact_example
- json_schema: json_schema

Strictly implements Design-B execution, transactional per-QID persistence, deep resume integrity,
and mandatory official scorer execution authority without model weights or inference in PREP phase.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import difflib
from hashlib import sha1, sha256
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import unicodedata
import subprocess
import sys
import tempfile
import threading
from typing import Any
from zipfile import ZipFile

import nltk
from nltk.translate.meteor_score import meteor_score
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
    render_competition_answer,
)
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.online import (
    ClaimVerificationConfig,
    GenerationConfig,
)
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    ModelError,
)
from legal_agentic_rag.generation.citation_verifier import (
    RuleBasedCitationVerifier,
)
from legal_agentic_rag.generation.model_generator import (
    ModelBackedAnswerGenerator,
)
from legal_agentic_rag.generation.transformers_provider import (
    TransformersChatProvider,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
)
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy

logger = logging.getLogger(__name__)

# ==============================================================================
# AUTHORITATIVE CONSTANTS & IMMUTABLE CHECKSUMS
# ==============================================================================

CANONICAL_TUNE20_ORDERED_QIDS: tuple[str, ...] = (
    "89271",
    "39207",
    "31523",
    "116553",
    "113579",
    "83501",
    "102061",
    "94975",
    "56533",
    "17179",
    "89881",
    "140337",
    "36411",
    "46497",
    "58651",
    "150817",
    "150207",
    "21011",
    "84363",
    "102303",
)

EXPECTED_FAST30_ARCHIVE_SHA256: str = (
    "be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f"
)
T5_6B_TUNE20_ORDERED_QIDS_SHA256: str = (
    "9cb88a00c2bcf9fbc0f24411de2f427d6a30f5da0f57feaaafb629f9fcd60b28"
)
T5_6B_FROZEN_GENERATOR_INPUT_SHA256: str = (
    "2fefbb03125f9927edf67c8bc8c165bdd856e1dd2eef0c737aefc7387a2cbbf2"
)
PREREGISTERED_PRODUCTION_GENERATOR_BLOB_SHA: str = (
    "1543eac766c0cf24ccb7904d8bfa2b802547e3c5"
)
EXPECTED_M49_GENERATOR_TREE_SHA256: str = (
    "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b"
)
OFFICIAL_SCORER_ARCHIVE_SHA256: str = (
    "4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891"
)
OFFICIAL_SCORING_PY_SHA256: str = (
    "f04843fbfad26d41356506d8e49692a7c8a0ed1b9f065a3a8472fa6398a5aa95"
)

T5_6B_CONTROL_GENERATION_CONFIG_SHA256: str = (
    "657ee87bdeac212857e9ec199c9fe34d6f7975ff5078c2371e1e6c2dba8738a7"
)
T5_6B_COMPACT_GENERATION_CONFIG_SHA256: str = (
    "810142a8ebacca5331ec13f1777be7edb6d4357b61a1c155d36751049b91bab2"
)
T5_6B_JSON_GENERATION_CONFIG_SHA256: str = (
    "8c930f08131b9cc9e07f1427d21b1d5e96c38431ca2d65f1e080abf04989596f"
)
T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256: str = (
    "fcb8cd2e65b74407be42a312f80624bb2be996e1a79d6a9228758d0893f23988"
)

EXPECTED_GENERATION_CONFIG_HASHES: dict[str, str] = {
    "control": T5_6B_CONTROL_GENERATION_CONFIG_SHA256,
    "compact": T5_6B_COMPACT_GENERATION_CONFIG_SHA256,
    "json_schema": T5_6B_JSON_GENERATION_CONFIG_SHA256,
}

EXECUTION_ARM_ORDER: tuple[str, ...] = ("control", "compact", "json_schema")
ARM_CONTRACT_MAP: dict[str, str] = {
    "control": "plain_text_markers",
    "compact": "compact_example",
    "json_schema": "json_schema",
}

PROVIDER_RELEVANT_CONFIG_FIELDS: tuple[str, ...] = (
    "backend",
    "model_name",
    "model_revision",
    "device",
    "torch_dtype",
    "model_loader",
    "local_files_only",
    "max_input_tokens",
    "temperature",
    "max_output_tokens",
    "repetition_penalty",
    "no_repeat_ngram_size",
)

STRUCTURED_RETRY_SENTINEL = "OUTPUT TRƯỚC KHÔNG HỢP LỆ. Hãy tạo lại từ đầu."
GROUNDING_REPAIR_SENTINEL = "BẢN NHÁP TRƯỚC KHÔNG QUA KIỂM TRA GROUNDING:"

_LEASE_LOCK = threading.Lock()
_RUNNER_LOCK = threading.Lock()

# ==============================================================================
# PREREGISTERED CONFIGURATION CONSTRUCTORS
# ==============================================================================


def get_preregistered_generation_config(
    prompt_schema_mode: str = "plain_text_markers",
) -> GenerationConfig:
    """Construct validated GenerationConfig for the preregistered T5-6B experiment."""
    return GenerationConfig(
        max_context_tokens=6144,
        max_evidence=10,
        timeout_seconds=360.0,
        backend="transformers",
        model_name="/kaggle/working/m49-generator-merged",
        model_revision="e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b",
        device="cuda",
        torch_dtype="float16",
        model_loader="image_text_to_text",
        local_files_only=True,
        max_input_tokens=8192,
        temperature=0.0,
        max_output_tokens=1536,
        repetition_penalty=1.08,
        no_repeat_ngram_size=8,
        max_structured_output_retries=1,
        max_model_error_retries=1,
        model_failure_policy="top_evidence",
        max_grounding_repair_retries=1,
        grounding_failure_policy="supported_claims_or_top_evidence",
        extractive_fallback_max_evidence=1,
        salvage_rendering="standalone",
        prompt_schema_mode=prompt_schema_mode,  # type: ignore[arg-type]
        answer_style="competition_reference",
    )


def get_preregistered_claim_verification_config() -> ClaimVerificationConfig:
    """Construct validated ClaimVerificationConfig for the preregistered T5-6B experiment."""
    return ClaimVerificationConfig(
        enabled=True,
        require_inline_citations=False,
        minimum_lexical_support=0.2,
        minimum_claim_tokens=2,
        require_numeric_match=True,
        require_negation_match=True,
        max_claims=60,
    )


# ==============================================================================
# HASHING & RECONSTRUCTION HELPERS
# ==============================================================================


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a local file."""
    h = sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_sha256(path: Path) -> str:
    """Compute deterministic tree SHA-256 over all files in directory.

    ALGORITHM AUTHORITY: Exact implementation from ``notebooks/m491_kaggle_candidate_dev.py``
    at repository commit ``10681c8`` (``directory_sha256`` function), which produced the
    canonical M49 merged generator tree hash
    ``e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b``.

    Serialisation contract (must not deviate):
      1. ``sorted(path.rglob("*") if is_file)``  -- global sort, NOT per-directory
      2. ``len(relative_utf8).to_bytes(8, "big")`` -- 8-byte big-endian length prefix
      3. ``relative_utf8``                         -- raw UTF-8 path bytes
      4. ``bytes.fromhex(file_sha256_hex)``        -- 32 raw hash bytes (NOT ASCII hex)
    """
    if not path.is_dir():
        raise ArtifactCompatibilityError(f"Model path {path} is not a directory")
    digest = sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(compute_file_sha256(item)))
    return digest.hexdigest()


def compute_tune20_ordered_qids_sha256(qids: Sequence[str]) -> str:
    """Compute canonical SHA-256 over ordered list of Tune20 QIDs."""
    payload = json.dumps(list(qids), separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def compute_frozen_generator_input_sha256(
    packets: Sequence[FrozenGeneratorInputPacket | dict[str, Any]],
) -> str:
    """Compute canonical SHA-256 over ordered frozen generator inputs."""
    payload = json.dumps(
        [
            p.model_dump(mode="json") if isinstance(p, BaseModel) else dict(p)
            for p in packets
        ],
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def git_blob_sha(path: Path) -> str:
    """Compute Git blob SHA-1 of a local file matching git hash-object."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    header = f"blob {len(data)}\0".encode("ascii")
    return sha1(header + data).hexdigest()


def verify_production_generator_blob(repo_root: Path) -> str:
    """Verify production generator file matches preregistered Git blob SHA."""
    generator_file = (
        repo_root
        / "src"
        / "legal_agentic_rag"
        / "generation"
        / "model_generator.py"
    )
    if not generator_file.is_file():
        raise ArtifactCompatibilityError(
            f"Production generator file not found: {generator_file}"
        )
    actual_blob_sha = git_blob_sha(generator_file)
    if actual_blob_sha != PREREGISTERED_PRODUCTION_GENERATOR_BLOB_SHA:
        raise ArtifactCompatibilityError(
            f"PREREGISTERED_PRODUCTION_GENERATOR_BLOB_MISMATCH: expected {PREREGISTERED_PRODUCTION_GENERATOR_BLOB_SHA}, got {actual_blob_sha}"
        )
    return actual_blob_sha


def reconstruct_query(question: str, question_id: str = "") -> RetrievalQuery:
    """Reconstruct valid RetrievalQuery for AnswerGenerator input.

    Normalization is identical to ``ServingService.create_query``:
      1. Strip leading/trailing whitespace.
      2. NFC Unicode normalization of collapsed whitespace.
    ``query_id`` follows the preregistered T5-6B convention ``t5-6b:<question_id>``.
    Retrieval and query-rewrite are never invoked.
    """
    original_question = question.strip()
    normalized_question = unicodedata.normalize(
        "NFC",
        " ".join(original_question.split()),
    )
    query_id = f"t5-6b:{question_id}" if question_id else "t5-6b:unknown"
    return RetrievalQuery(
        query_id=query_id,
        original_question=original_question,
        normalized_question=normalized_question,
    )


# ==============================================================================
# SCHEMAS & TELEMETRY CONTRACTS
# ==============================================================================


class FrozenGeneratorInputPacket(BaseModel):
    """Immutable input packet extracted from FAST30 clean baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    question: str
    selected_evidence: list[Evidence]


class GeneratorCallTelemetry(BaseModel):
    """Detailed telemetry record for a single generator provider call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    candidate_contract: str
    call_stage: str
    call_index: int
    provider_attempt_index: int
    provider_call_success: bool
    input_token_count: int | None = None
    output_token_count: int | None = None
    system_prompt_sha256: str
    user_prompt_sha256: str
    parse_result: str = "PENDING"
    rejection_error_type: str | None = None
    rejection_attempt_index: int | None = None
    raw_completion_text: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None


class GroundingCallTelemetry(BaseModel):
    """Telemetry record for a citation verification call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    candidate_contract: str
    verification_index: int
    grounded: bool
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    unverifiable_claims: int
    supported_ratio: float
    selected_evidence_count: int


class QuestionMeasurementResult(BaseModel):
    """Comprehensive measurement result for a single question under a single contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    candidate_contract: str
    response: AnswerResponse
    final_generator_path: str
    citation_identity_valid: bool
    parser_accepted: bool
    had_contract_rejection: bool
    calls: list[GeneratorCallTelemetry]
    grounding_calls: list[GroundingCallTelemetry]


class ArmMeasurementSummary(BaseModel):
    """Aggregated metrics summary for one candidate contract arm on Tune20."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_contract: str
    record_count: int
    rouge_l: float | None = None
    meteor: float | None = None
    parser_acceptance_rate: float
    citation_identity_validity: float
    contract_rejection_fallback_count: int
    insufficient_evidence_count: int
    total_structured_output_rejections: int
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    final_path_distribution: dict[str, int] = Field(default_factory=dict)
    per_question_scores: dict[str, dict[str, float]] = Field(default_factory=dict)


class CandidateAdvancementResult(BaseModel):
    """Advancement decision and gate checks for one candidate arm against control."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_name: str
    candidate_contract: str
    parser_acceptance_passed: bool
    meteor_passed: bool
    rouge_l_passed: bool
    citation_validity_passed: bool
    fallback_count_passed: bool
    insufficient_count_passed: bool
    all_passed: bool


class T56BRuntimeEnvironmentFingerprint(BaseModel):
    """Recorded hardware and software runtime environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str
    torch_version: str | None
    transformers_version: str | None
    cuda_available: bool
    cuda_runtime_version: str | None
    gpu_count: int
    gpu_name: str | None
    device: str
    torch_dtype: str
    model_loader: str
    model_path: str
    model_tree_sha256: str
    production_generator_blob_sha: str
    measurement_source_sha: str


def compute_environment_fingerprint_sha256(
    fp: T56BRuntimeEnvironmentFingerprint,
) -> str:
    """Compute canonical SHA-256 digest of runtime environment fingerprint."""
    payload = json.dumps(
        fp.model_dump(), separators=(",", ":"), ensure_ascii=False
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class T56BGenerationManifest(BaseModel):
    """Closed manifest produced by the generation execution phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "t5_generator_generation_manifest_v1"
    created_at: str
    measurement_source_sha: str
    production_generator_blob_sha: str
    model_artifact_tree_sha256: str
    resolved_model_path: str
    runtime_environment_sha256: str
    runtime_environment: T56BRuntimeEnvironmentFingerprint
    control_generation_config_sha256: str
    compact_generation_config_sha256: str
    json_generation_config_sha256: str
    claim_verification_config_sha256: str
    frozen_generator_input_sha256: str
    tune20_ordered_qids_sha256: str
    execution_order: list[str]
    record_count: int
    generation_closed_at: str
    generation_artifact_hashes: dict[str, str]


class T56BMeasurementManifest(BaseModel):
    """Final manifest produced by scoring closed generation artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "t5_generator_measurement_manifest_v1"
    created_at: str
    measurement_source_sha: str
    production_generator_blob_sha: str
    model_artifact_tree_sha256: str
    runtime_environment_sha256: str
    runtime_environment: T56BRuntimeEnvironmentFingerprint
    control_generation_config_sha256: str
    compact_generation_config_sha256: str
    json_generation_config_sha256: str
    claim_verification_config_sha256: str
    frozen_generator_input_sha256: str
    tune20_ordered_qids_sha256: str
    official_scorer_sha256: str
    execution_order: list[str]
    record_count: int
    generation_manifest_hash: str
    generation_artifact_hashes: dict[str, str]
    scoring_artifact_hashes: dict[str, str]
    arm_summaries: dict[str, ArmMeasurementSummary]
    advancement_results: dict[str, CandidateAdvancementResult]
    advancement_winner: str | None
    phase_decision: str


# ==============================================================================
# INPUT EXTRACTION & VALIDATION
# ==============================================================================


def extract_frozen_generator_inputs(
    diagnostics_records: Sequence[dict[str, Any]],
) -> list[FrozenGeneratorInputPacket]:
    """Extract and validate frozen generator input packets from FAST30 clean baseline."""
    diag_by_qid: dict[str, dict[str, Any]] = {}
    for r in diagnostics_records:
        qid = str(r.get("question_id", ""))
        if qid:
            diag_by_qid[qid] = r

    packets: list[FrozenGeneratorInputPacket] = []
    for qid in CANONICAL_TUNE20_ORDERED_QIDS:
        if qid not in diag_by_qid:
            raise DataValidationError(
                f"Missing canonical Tune20 QID in diagnostics records: {qid}"
            )
        rec = diag_by_qid[qid]
        question = str(rec.get("question", ""))
        if not question:
            raise DataValidationError(f"Empty question for QID {qid}")

        raw_ev = rec.get("selected_evidence", [])
        if not isinstance(raw_ev, list):
            raise DataValidationError(
                f"Invalid selected_evidence for QID {qid}"
            )

        evidence_list: list[Evidence] = []
        for e in raw_ev:
            evidence_list.append(Evidence.model_validate(e))

        packets.append(
            FrozenGeneratorInputPacket(
                question_id=qid,
                question=question,
                selected_evidence=evidence_list,
            )
        )
    return packets


def load_frozen_tune20_packets(archive_path: Path) -> list[FrozenGeneratorInputPacket]:
    """Load, verify, and parse frozen generator input packets from FAST30 ZIP archive."""
    if not archive_path.is_file():
        raise ArtifactCompatibilityError(f"Archive not found: {archive_path}")

    actual_sha = compute_file_sha256(archive_path)
    if actual_sha != EXPECTED_FAST30_ARCHIVE_SHA256:
        raise ArtifactCompatibilityError(
            f"EXPECTED_FAST30_ARCHIVE_SHA256 mismatch: expected {EXPECTED_FAST30_ARCHIVE_SHA256}, got {actual_sha}"
        )

    with ZipFile(archive_path, "r") as z:
        if "diagnostics.jsonl" not in z.namelist():
            raise ArtifactCompatibilityError(
                "diagnostics.jsonl not found in FAST30 zip archive"
            )
        raw_lines = [
            line
            for line in z.read("diagnostics.jsonl")
            .decode("utf-8")
            .splitlines()
            if line.strip()
        ]
        records = [json.loads(l) for l in raw_lines]

    packets = extract_frozen_generator_inputs(records)

    actual_qids_sha = compute_tune20_ordered_qids_sha256(
        [p.question_id for p in packets]
    )
    if actual_qids_sha != T5_6B_TUNE20_ORDERED_QIDS_SHA256:
        raise ArtifactCompatibilityError(
            f"T5_6B_TUNE20_ORDERED_QIDS_SHA256 mismatch: expected {T5_6B_TUNE20_ORDERED_QIDS_SHA256}, got {actual_qids_sha}"
        )

    actual_inputs_sha = compute_frozen_generator_input_sha256(packets)
    if actual_inputs_sha != T5_6B_FROZEN_GENERATOR_INPUT_SHA256:
        raise ArtifactCompatibilityError(
            f"T5_6B_FROZEN_GENERATOR_INPUT_SHA256 mismatch: expected {T5_6B_FROZEN_GENERATOR_INPUT_SHA256}, got {actual_inputs_sha}"
        )

    return packets


# ==============================================================================
# LOGGING, PROXY, AND OBSERVABILITY OBSERVERS
# ==============================================================================


class ModelGeneratorLoggingLease:
    """Thread-safe context manager capturing logger events from ModelBackedAnswerGenerator."""

    def __init__(self, target_logger_name: str, handler: logging.Handler) -> None:
        self.target_logger_name = target_logger_name
        self.handler = handler
        self.target_logger = logging.getLogger(target_logger_name)
        self._orig_handlers: list[logging.Handler] = []

    def __enter__(self) -> ModelGeneratorLoggingLease:
        if not _LEASE_LOCK.acquire(blocking=False):
            raise DataValidationError(
                "T5_6B_OVERLAPPING_LEASE: A concurrent logging lease is already active. "
                "Overlapping leases are rejected to preserve logger handler integrity."
            )
        self._orig_handlers = list(self.target_logger.handlers)
        self.target_logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self.target_logger.removeHandler(self.handler)
            self.target_logger.handlers = list(self._orig_handlers)
        finally:
            _LEASE_LOCK.release()


class ModelGeneratorRejectionObserver(logging.Handler):
    """Captures structured output rejection and parsing failure events from model generator."""

    def __init__(self) -> None:
        super().__init__()
        self.rejections: list[dict[str, Any]] = []
        self._active_context: tuple[str, str] | None = None
        self._active_logical_call_index: int = 0

    def set_active_context(
        self, question_id: str, candidate_contract: str
    ) -> None:
        self._active_context = (question_id, candidate_contract)
        self._active_logical_call_index = 0

    def set_active_logical_call(self, logical_call_index: int) -> None:
        """Update the current logical call index so rejections are attributed correctly."""
        self._active_logical_call_index = logical_call_index

    def clear_active_context(self) -> None:
        self._active_context = None
        self._active_logical_call_index = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if (
            "model_answer_draft_rejected" in msg
            or "structured_output_schema" in msg
            or "structured_output_missing_fields" in msg
        ):
            # structured_output_attempt is kept as a separate telemetry property
            structured_attempt = getattr(record, "structured_output_attempt", 1)
            if self._active_context is None:
                return
            self.rejections.append(
                {
                    "message": msg,
                    "error_type": getattr(record, "error_type", "structured_output_schema"),
                    "structured_output_attempt": structured_attempt,
                    # context uses logical_call_index, NOT structured_output_attempt
                    "context": (
                        self._active_context[0],
                        self._active_context[1],
                        self._active_logical_call_index,
                    ),
                }
            )

    def clear(self) -> None:
        self.rejections.clear()
        self._active_context = None
        self._active_logical_call_index = 0


class _GenerateProxy:
    """Transparent proxy intercepting model.generate() to capture token telemetry."""

    def __init__(self, target_generate: Any, provider: ObservableTransformersChatProvider) -> None:
        self._target_generate = target_generate
        self._provider = provider

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        input_tokens = (
            int(input_ids.shape[-1])
            if input_ids is not None and hasattr(input_ids, "shape")
            else None
        )
        # Record input tokens before calling generate so even on exception
        # the observer has partial telemetry (output_tokens=None on failure).
        self._provider.record_token_counts(input_tokens, None)
        try:
            out = self._target_generate(*args, **kwargs)
        except Exception:
            # Input tokens already recorded; output remains None. Re-raise.
            raise
        output_tokens = None
        if out is not None and hasattr(out, "shape"):
            output_tokens = int(out.shape[-1])
            if input_tokens is not None and output_tokens >= input_tokens:
                output_tokens = output_tokens - input_tokens

        # Update with both input and output tokens on success.
        self._provider.record_token_counts(input_tokens, output_tokens)
        return out


class ObservableTransformersChatProvider(TransformersChatProvider):
    """TransformersChatProvider instrumented with non-invasive token observation proxy.

    The generate proxy is installed exactly once by overriding ``_load_runtime()``.
    ``complete`` is NOT overridden: ``ObservableTransformersChatProvider.complete``
    is exactly ``TransformersChatProvider.complete``, so the first real provider call
    is always observed. No double-wrapping can occur because ``_require_runtime()``
    caches the runtime after the first ``_load_runtime()`` call.
    """

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        self.token_observer: MeasurementProviderObserver | None = None

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        """Load runtime and install generate proxy before first use."""
        torch, tokenizer, model = super()._load_runtime()
        model.generate = _GenerateProxy(model.generate, self)
        return torch, tokenizer, model

    def record_token_counts(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> None:
        if self.token_observer is not None:
            self.token_observer.record_token_counts(input_tokens, output_tokens)


class MeasurementProviderObserver(ChatModelProvider):
    """Observing decorator for ChatModelProvider recording per-call metadata and tokens."""

    def __init__(self, wrapped_provider: ChatModelProvider) -> None:
        self.wrapped_provider = wrapped_provider
        self.call_telemetry: list[GeneratorCallTelemetry] = []
        self._active_question_id: str | None = None
        self._active_contract: str | None = None
        self._call_counter: int = 0
        self._provider_attempt_index: int = 0
        self._last_prompt_key: tuple[str, str, str] | None = None
        self._last_token_counts: tuple[int | None, int | None] = (None, None)
        self._rejection_observer: ModelGeneratorRejectionObserver | None = None

    @property
    def provider_name(self) -> str:
        return self.wrapped_provider.provider_name

    @property
    def provider_version(self) -> str:
        return self.wrapped_provider.provider_version

    def set_active_question(
        self, question_id: str, candidate_contract: str
    ) -> None:
        self._active_question_id = question_id
        self._active_contract = candidate_contract
        self._call_counter = 0
        self._provider_attempt_index = 0
        self._last_prompt_key: tuple[str, str, str] | None = None

    def clear_active_question(self) -> None:
        self._active_question_id = None
        self._active_contract = None
        self._call_counter = 0
        self._provider_attempt_index = 0
        self._last_prompt_key = None

    def record_token_counts(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> None:
        self._last_token_counts = (input_tokens, output_tokens)

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        """Record per-call telemetry using a deterministic logical call / provider attempt state machine.

        State machine semantics:
          - ``call_index`` (logical call): identifies a distinct prompt+stage.
          - ``provider_attempt_index``: counts retries within the same logical call
            (i.e. ModelError retries on the same prompt via ``_complete_with_retry``).
          - Same (system_sha, user_sha, stage) -> same ``call_index``, increment attempt.
          - Different stage or prompt -> new logical call, reset attempt to 1.
        """
        if self._active_question_id is None or self._active_contract is None:
            raise DataValidationError("Active question context not set on provider observer")

        sys_sha = sha256(system_instruction.encode("utf-8")).hexdigest()
        user_sha = sha256(user_prompt.encode("utf-8")).hexdigest()
        call_stage = classify_prompt_call_stage(user_prompt)

        # Determine logical call index and provider attempt index
        current_prompt_key = (sys_sha, user_sha, call_stage)
        if current_prompt_key == self._last_prompt_key:
            # Same prompt repeated: model error retry within same logical call
            self._provider_attempt_index += 1
        else:
            # New prompt or new stage: new logical call
            self._call_counter += 1
            self._provider_attempt_index = 1
            self._last_prompt_key = current_prompt_key

        call_index = self._call_counter
        provider_attempt_index = self._provider_attempt_index

        # Update rejection observer with current logical call index before execution
        if self._rejection_observer is not None:
            self._rejection_observer.set_active_logical_call(call_index)

        self._last_token_counts = (None, None)
        try:
            raw_completion = self.wrapped_provider.complete(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            )
            inp_t, out_t = self._last_token_counts
            self.call_telemetry.append(
                GeneratorCallTelemetry(
                    question_id=self._active_question_id,
                    candidate_contract=self._active_contract,
                    call_stage=call_stage,
                    call_index=call_index,
                    provider_attempt_index=provider_attempt_index,
                    provider_call_success=True,
                    input_token_count=inp_t,
                    output_token_count=out_t,
                    system_prompt_sha256=sys_sha,
                    user_prompt_sha256=user_sha,
                    raw_completion_text=raw_completion,
                )
            )
            return raw_completion
        except Exception as exc:
            inp_t, _ = self._last_token_counts
            self.call_telemetry.append(
                GeneratorCallTelemetry(
                    question_id=self._active_question_id,
                    candidate_contract=self._active_contract,
                    call_stage=call_stage,
                    call_index=call_index,
                    provider_attempt_index=provider_attempt_index,
                    provider_call_success=False,
                    input_token_count=inp_t,
                    output_token_count=None,
                    system_prompt_sha256=sys_sha,
                    user_prompt_sha256=user_sha,
                    raw_completion_text=None,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            )
            raise


class ObservableCitationVerifier(CitationVerifier):
    """Observing decorator for CitationVerifier recording grounding telemetry."""

    def __init__(self, wrapped_verifier: CitationVerifier) -> None:
        self.wrapped_verifier = wrapped_verifier
        self.grounding_telemetry: list[GroundingCallTelemetry] = []
        self._active_question_id: str | None = None
        self._active_contract: str | None = None
        self._verification_counter: int = 0

    @property
    def verifier_name(self) -> str:
        return self.wrapped_verifier.verifier_name

    @property
    def verifier_version(self) -> str:
        return self.wrapped_verifier.verifier_version

    def set_active_question(
        self, question_id: str, candidate_contract: str
    ) -> None:
        self._active_question_id = question_id
        self._active_contract = candidate_contract
        self._verification_counter = 0

    def clear_active_question(self) -> None:
        self._active_question_id = None
        self._active_contract = None
        self._verification_counter = 0

    def verify(
        self, response: AnswerResponse, evidence: Sequence[Evidence]
    ) -> CitationVerificationResult:
        if self._active_question_id is None or self._active_contract is None:
            raise DataValidationError("Active question context not set on verifier observer")

        self._verification_counter += 1
        ver_index = self._verification_counter

        res = self.wrapped_verifier.verify(response, evidence)

        supp_count = sum(1 for c in res.claim_verifications if c.status == ClaimSupportStatus.SUPPORTED)
        unsupp_count = sum(1 for c in res.claim_verifications if c.status == ClaimSupportStatus.UNSUPPORTED)
        unver_count = 0
        supp_ratio = (supp_count / len(res.claim_verifications)) if res.claim_verifications else 1.0

        self.grounding_telemetry.append(
            GroundingCallTelemetry(
                question_id=self._active_question_id,
                candidate_contract=self._active_contract,
                verification_index=ver_index,
                grounded=res.is_valid,
                total_claims=len(res.claim_verifications),
                supported_claims=supp_count,
                unsupported_claims=unsupp_count,
                unverifiable_claims=unver_count,
                supported_ratio=supp_ratio,
                selected_evidence_count=len(evidence),
            )
        )
        return res


def measurement_transformers_provider_factory(
    config: GenerationConfig,
) -> ChatModelProvider:
    """Construct the accepted measurement Transformers chat model provider."""
    return ObservableTransformersChatProvider(config)


# ==============================================================================
# CLASSIFICATION & CORRELATION HELPERS
# ==============================================================================


def classify_prompt_call_stage(user_prompt: str) -> str:
    """Classify generator call stage based on sentinel prompt markers.

    Both sentinels present simultaneously is a DataValidationError (AMBIGUOUS state
    that must not occur in a well-formed generation run).
    """
    has_grounding = GROUNDING_REPAIR_SENTINEL in user_prompt
    has_structured = STRUCTURED_RETRY_SENTINEL in user_prompt
    if has_grounding and has_structured:
        raise DataValidationError(
            "AMBIGUOUS_CALL_STAGE: Both grounding and structured retry sentinels present in same prompt"
        )
    if has_grounding:
        return "GROUNDING_REPAIR"
    if has_structured:
        return "STRUCTURED_RETRY"
    return "INITIAL_DRAFT"


def classify_final_generator_path(response: AnswerResponse) -> str:
    """Classify the terminal generator execution path from production warnings and status."""
    if response.insufficient_evidence:
        return "INSUFFICIENT_EVIDENCE"

    warnings = set(response.warnings)
    has_model_err = "generator_model_error_fallback" in warnings
    has_repair = "grounding_repair_attempted" in warnings
    has_salvage = "supported_claim_salvage_applied" in warnings
    has_extractive = "extractive_fallback_applied" in warnings

    if has_model_err and (has_repair or has_salvage or has_extractive):
        return "AMBIGUOUS"

    if has_model_err:
        return "MODEL_ERROR_FALLBACK"

    if has_repair:
        if has_salvage and has_extractive:
            return "AMBIGUOUS"
        if has_salvage:
            return "SUPPORTED_CLAIM_SALVAGE"
        if has_extractive:
            return "GROUNDING_EXTRACTIVE_FALLBACK"
        return "GROUNDING_REPAIR_SUCCESS"

    if has_salvage or has_extractive:
        return "AMBIGUOUS"

    return "SEMANTIC_SYNTHESIS"


_EVIDENCE_MARKER_RE = re.compile(r"\[E(\d+)\]")


def evaluate_citation_identity_validity(
    response: AnswerResponse, selected_evidence: Sequence[Evidence]
) -> bool:
    """Strict citation identity validation for non-abstaining responses.

    Checks (all must pass for True):
      1. Each citation.evidence_id exists in supplied selected_evidence.
      2. citation.chunk_id exactly equals the chunk_id of that evidence item.
      3. No duplicate (evidence_id, chunk_id) pairs across citations.
      4. No [E#] markers in answer text referencing unknown supplied evidence IDs.
    """
    if response.insufficient_evidence:
        return True

    evidence_map: dict[str, Evidence] = {e.evidence_id: e for e in selected_evidence}

    # Check citations list
    seen_pairs: set[tuple[str, str]] = set()
    for cit in response.citations:
        ev = evidence_map.get(cit.evidence_id)
        if ev is None:
            return False  # unknown evidence_id
        if cit.chunk_id != ev.chunk_id:
            return False  # chunk_id mismatch
        pair = (cit.evidence_id, cit.chunk_id)
        if pair in seen_pairs:
            return False  # duplicate citation
        seen_pairs.add(pair)

    # Check [E#] markers in answer text
    answer_text = response.answer or ""
    for marker_num in _EVIDENCE_MARKER_RE.findall(answer_text):
        marker_id = f"E{marker_num}"
        if marker_id not in evidence_map:
            return False  # answer references unknown evidence

    return True


def correlate_rejection_events(
    calls: list[GeneratorCallTelemetry],
    rejection_records: list[dict[str, Any]],
) -> None:
    """Correlate structured output rejection records strictly with corresponding provider calls."""
    rejections = list(rejection_records)

    for i, call in enumerate(calls):
        if not call.provider_call_success or call.raw_completion_text is None:
            calls[i] = call.model_copy(
                update={"parse_result": "NOT_APPLICABLE_PROVIDER_ERROR"}
            )
            continue

        matched_idx: int | None = None
        for r_idx, rej in enumerate(rejections):
            ctx = rej.get("context")
            if ctx is None:
                raise DataValidationError(
                    f"AMBIGUOUS_REJECTION_CONTEXT: Rejection event missing mandatory context tuple: {rej}"
                )
            qid, contract, c_idx = ctx
            if (
                qid == call.question_id
                and contract == call.candidate_contract
                and c_idx == call.call_index
            ):
                matched_idx = r_idx
                break

        if matched_idx is not None:
            rej = rejections.pop(matched_idx)
            calls[i] = call.model_copy(
                update={
                    "parse_result": "REJECTED",
                    "rejection_error_type": rej.get("error_type"),
                    "rejection_attempt_index": rej.get("structured_output_attempt"),
                }
            )
        else:
            calls[i] = call.model_copy(update={"parse_result": "ACCEPTED"})

    if rejections:
        raise DataValidationError(
            f"AMBIGUOUS_REJECTION_CONTEXT: Uncorrelated rejection events remaining: {rejections}"
        )


# ==============================================================================
# OFFICIAL SCORING PARITY & EXECUTION ENGINE
# ==============================================================================


def _diagnostic_rouge_l_tokenize(text: str) -> list[str]:
    """DIAGNOSTIC / PARITY-TEST ONLY: ASCII-only lowercase whitespace tokenizer."""
    return re.sub(r"[^a-z0-9]", " ", text.lower()).split()


def _diagnostic_lcs_table(ref: list[str], pred: list[str]) -> list[list[int]]:
    """DIAGNOSTIC / PARITY-TEST ONLY: Longest Common Subsequence dynamic programming table."""
    m, n = len(ref), len(pred)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == pred[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table


def _diagnostic_compute_rouge_l(prediction: str, reference: str) -> float:
    """DIAGNOSTIC / PARITY-TEST ONLY: Offline LCS F1 score."""
    pred_tokens = _diagnostic_rouge_l_tokenize(prediction)
    ref_tokens = _diagnostic_rouge_l_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs_len = _diagnostic_lcs_table(ref_tokens, pred_tokens)[len(ref_tokens)][len(pred_tokens)]
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _diagnostic_compute_meteor(prediction: str, reference: str) -> float:
    """DIAGNOSTIC / PARITY-TEST ONLY: Offline whitespace METEOR score."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    return float(meteor_score([ref_tokens], pred_tokens))


def score_tune20_answers(
    predicted_answers: dict[str, str],
    reference_answers: dict[str, str],
    scorer_path: Path,
) -> tuple[float, float, dict[str, dict[str, float]], str]:
    """Score Tune20 predictions against reference answers by executing the verified official scorer program.

    ENTRYPOINT AUTHORITY: Pinned to the official ``eval_qa(y_pred, y_true)`` function inside
    ``scoring.py`` extracted from the verified official scorer archive (``OFFICIAL_SCORER_ARCHIVE_SHA256``).

    Payload contract:
      - ``y_pred``: dict mapping each QID to ``{"answer": str}``
      - ``y_true``: dict mapping each QID to ``str`` (reference text)

    Validation rules:
      - ``set(predicted_answers.keys()) == set(CANONICAL_TUNE20_ORDERED_QIDS)`` (exactly 20 QIDs)
      - ``set(reference_answers.keys()) == set(CANONICAL_TUNE20_ORDERED_QIDS)`` (exactly 20 QIDs)
      - Missing, extra, or mismatched QIDs fail closed immediately.
      - Official macro values from ``eval_qa`` are the ONLY metrics used by advancement gates.
      - Per-question scores are evaluated via the exact same official entrypoint per single QID.
    """
    if not scorer_path.is_file():
        raise ArtifactCompatibilityError(
            f"OFFICIAL_SCORER_AUTHORITY_FAILED: Scorer archive not found: {scorer_path}"
        )

    actual_archive_sha = compute_file_sha256(scorer_path)
    if actual_archive_sha != OFFICIAL_SCORER_ARCHIVE_SHA256:
        raise ArtifactCompatibilityError(
            f"OFFICIAL_SCORER_AUTHORITY_FAILED: Scorer archive SHA mismatch: expected {OFFICIAL_SCORER_ARCHIVE_SHA256}, got {actual_archive_sha}"
        )

    canonical_set = set(CANONICAL_TUNE20_ORDERED_QIDS)
    if set(predicted_answers.keys()) != canonical_set:
        raise DataValidationError(
            f"Predicted answers QID set mismatch: expected {canonical_set}, got {set(predicted_answers.keys())}"
        )
    if set(reference_answers.keys()) != canonical_set:
        raise DataValidationError(
            f"Reference answers QID set mismatch: expected {canonical_set}, got {set(reference_answers.keys())}"
        )

    with ZipFile(scorer_path, "r") as z:
        if "scoring.py" not in z.namelist():
            raise ArtifactCompatibilityError(
                "OFFICIAL_SCORER_AUTHORITY_FAILED: scoring.py not found in scorer archive"
            )
        member_bytes = z.read("scoring.py")
        actual_py_sha = sha256(member_bytes).hexdigest()
        if actual_py_sha != OFFICIAL_SCORING_PY_SHA256:
            raise ArtifactCompatibilityError(
                f"OFFICIAL_SCORER_AUTHORITY_FAILED: scoring.py member SHA mismatch: expected {OFFICIAL_SCORING_PY_SHA256}, got {actual_py_sha}"
            )

    with tempfile.TemporaryDirectory() as td:
        with ZipFile(scorer_path, "r") as z:
            z.extractall(td)

        sys.path.insert(0, str(td))
        try:
            spec = importlib.util.spec_from_file_location(
                "official_scoring_program", Path(td) / "scoring.py"
            )
            if spec is None or spec.loader is None:
                raise ArtifactCompatibilityError(
                    "OFFICIAL_SCORER_AUTHORITY_FAILED: Unable to load module spec for scoring.py"
                )
            official_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(official_mod)

            if not hasattr(official_mod, "eval_qa"):
                raise ArtifactCompatibilityError(
                    "OFFICIAL_SCORER_AUTHORITY_FAILED: eval_qa entrypoint not found in scoring.py"
                )

            # Construct exact official macro payload
            y_pred = {
                qid: {"answer": str(predicted_answers[qid])}
                for qid in CANONICAL_TUNE20_ORDERED_QIDS
            }
            y_true = {
                qid: str(reference_answers[qid])
                for qid in CANONICAL_TUNE20_ORDERED_QIDS
            }

            macro_scores = official_mod.eval_qa(y_pred, y_true)
            if not isinstance(macro_scores, dict) or "rouge" not in macro_scores or "meteor" not in macro_scores:
                raise ArtifactCompatibilityError(
                    f"OFFICIAL_SCORER_AUTHORITY_FAILED: eval_qa returned invalid score output format: {macro_scores}"
                )

            mean_rouge = float(macro_scores["rouge"])
            mean_meteor = float(macro_scores["meteor"])

            # Per-question scores evaluated via same official entrypoint on single-QID dicts
            per_q: dict[str, dict[str, float]] = {}
            for qid in CANONICAL_TUNE20_ORDERED_QIDS:
                single_pred = {qid: {"answer": str(predicted_answers[qid])}}
                single_ref = {qid: str(reference_answers[qid])}
                single_scores = official_mod.eval_qa(single_pred, single_ref)
                per_q[qid] = {
                    "rouge_l": float(single_scores["rouge"]),
                    "meteor": float(single_scores["meteor"]),
                }

            return mean_rouge, mean_meteor, per_q, actual_archive_sha
        finally:
            if str(td) in sys.path:
                sys.path.remove(str(td))


# ==============================================================================
# ADVANCEMENT GATE & TIE-BREAK EVALUATION
# ==============================================================================


def evaluate_advancement_gate(
    summaries: dict[str, ArmMeasurementSummary],
) -> tuple[dict[str, CandidateAdvancementResult], str | None, str]:
    """Evaluate candidate advancement under the 6-part cumulative gate against the same-run control."""
    if "control" not in summaries:
        raise DataValidationError("Control arm missing from summaries")
    control = summaries["control"]

    results: dict[str, CandidateAdvancementResult] = {}
    passing_candidates: list[tuple[str, float, float, float, str]] = []

    for name in ("compact", "json_schema"):
        if name not in summaries:
            continue
        summary = summaries[name]
        p_acc_pass = summary.parser_acceptance_rate >= 0.80
        meteor_pass = (summary.meteor or 0.0) >= (control.meteor or 0.0)
        rouge_pass = (summary.rouge_l or 0.0) >= (control.rouge_l or 0.0)
        cit_pass = summary.citation_identity_validity == 1.0
        fallback_pass = (
            summary.contract_rejection_fallback_count
            <= control.contract_rejection_fallback_count
        )
        insufficient_pass = (
            summary.insufficient_evidence_count
            <= control.insufficient_evidence_count
        )

        all_passed = bool(
            p_acc_pass
            and meteor_pass
            and rouge_pass
            and cit_pass
            and fallback_pass
            and insufficient_pass
        )

        cand_res = CandidateAdvancementResult(
            candidate_name=name,
            candidate_contract=summary.candidate_contract,
            parser_acceptance_passed=p_acc_pass,
            meteor_passed=meteor_pass,
            rouge_l_passed=rouge_pass,
            citation_validity_passed=cit_pass,
            fallback_count_passed=fallback_pass,
            insufficient_count_passed=insufficient_pass,
            all_passed=all_passed,
        )
        results[name] = cand_res

        if all_passed:
            passing_candidates.append(
                (
                    name,
                    summary.meteor or 0.0,
                    summary.rouge_l or 0.0,
                    summary.parser_acceptance_rate,
                    name,
                )
            )

    if not passing_candidates:
        return results, None, "NO_GENERATOR_CONTRACT_CANDIDATE_JUSTIFIED"

    passing_candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
    winner = passing_candidates[0][0]
    return results, winner, "NEW_CLEAN_VALIDATION_POPULATION_REQUIRED"


# ==============================================================================
# RUNTIME ENVIRONMENT DETECTION
# ==============================================================================


def get_runtime_environment_fingerprint(
    *,
    repo_root: Path,
    measurement_source_sha: str,
    model_path: Path | None = None,
    model_tree_sha256: str | None = None,
) -> tuple[T56BRuntimeEnvironmentFingerprint, str]:
    """Capture detailed runtime hardware and software environment."""
    prod_blob_sha = verify_production_generator_blob(repo_root)

    torch_ver: str | None = None
    cuda_avail = False
    gpu_count = 0
    gpu_name: str | None = None
    cuda_runtime_ver: str | None = None

    try:
        import torch

        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else None
            cuda_runtime_ver = torch.version.cuda
    except ImportError:
        pass

    transformers_ver: str | None = None
    try:
        import transformers

        transformers_ver = transformers.__version__
    except ImportError:
        pass

    m_path_str = str(model_path.resolve().as_posix()) if model_path else "/kaggle/working/m49-generator-merged"
    m_tree_sha = model_tree_sha256 or EXPECTED_M49_GENERATOR_TREE_SHA256

    fp = T56BRuntimeEnvironmentFingerprint(
        python_version=sys.version.split()[0],
        torch_version=torch_ver,
        transformers_version=transformers_ver,
        cuda_available=cuda_avail,
        cuda_runtime_version=cuda_runtime_ver,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        device="cuda" if cuda_avail else "cpu",
        torch_dtype="float16",
        model_loader="image_text_to_text",
        model_path=m_path_str,
        model_tree_sha256=m_tree_sha,
        production_generator_blob_sha=prod_blob_sha,
        measurement_source_sha=measurement_source_sha,
    )
    return fp, compute_environment_fingerprint_sha256(fp)


# ==============================================================================
# MEASUREMENT RUNNER IMPLEMENTATION
# ==============================================================================


class T5GeneratorContractMeasurementRunner:
    """Executes the controlled T5-6B Design-B measurement across the three preregistered arms."""

    def __init__(
        self,
        *,
        repo_root: Path,
        archive_path: Path,
        output_dir: Path,
        provider_factory: Callable[[GenerationConfig], ChatModelProvider] | None = None,
        model_path: Path | None = None,
        expected_model_tree_sha256: str = EXPECTED_M49_GENERATOR_TREE_SHA256,
        expected_measurement_source_sha: str | None = None,
        is_preflight_only: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.archive_path = archive_path.resolve()
        self.output_dir = output_dir.resolve()
        self.provider_factory = provider_factory
        self.model_path = model_path.resolve() if model_path else None
        self.expected_model_tree_sha256 = expected_model_tree_sha256
        self.expected_measurement_source_sha = expected_measurement_source_sha
        self.is_preflight_only = is_preflight_only

    def verify_pre_execution_authorities(self) -> str:
        """Verify repository blob, configs, clean source tree, and source commit SHA before execution."""
        verify_production_generator_blob(self.repo_root)

        claim_cfg = get_preregistered_claim_verification_config()
        actual_claim_sha = canonical_sha256(claim_cfg)
        if actual_claim_sha != T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256:
            raise ArtifactCompatibilityError(
                f"T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256 mismatch: expected {T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256}, got {actual_claim_sha}"
            )

        base_cfg = get_preregistered_generation_config("plain_text_markers")
        for arm_name in EXECUTION_ARM_ORDER:
            prompt_mode = ARM_CONTRACT_MAP[arm_name]
            cfg = get_preregistered_generation_config(prompt_mode)
            actual_sha = canonical_sha256(cfg)
            expected_sha = EXPECTED_GENERATION_CONFIG_HASHES[arm_name]
            if actual_sha != expected_sha:
                raise ArtifactCompatibilityError(
                    f"GenerationConfig hash mismatch for {arm_name}: expected {expected_sha}, got {actual_sha}"
                )

            for field in PROVIDER_RELEVANT_CONFIG_FIELDS:
                base_val = getattr(base_cfg, field)
                cand_val = getattr(cfg, field)
                if base_val != cand_val:
                    raise ArtifactCompatibilityError(
                        f"Provider-relevant config field '{field}' differs across arms: control={base_val}, {arm_name}={cand_val}"
                    )

        if not self.is_preflight_only:
            try:
                diff_cmd = [
                    "git",
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    "src",
                    "configs",
                    "scripts/t5_generator_contract_measurement.py",
                ]
                res = subprocess.run(
                    diff_cmd, cwd=self.repo_root, capture_output=True
                )
                if res.returncode != 0:
                    raise ArtifactCompatibilityError(
                        "T5_6B_DIRTY_EXECUTION_SOURCE: Tracked core source files contain uncommitted changes"
                    )

                untracked_cmd = [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "src",
                    "configs",
                ]
                res_untracked = subprocess.run(
                    untracked_cmd,
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                )
                if res_untracked.stdout.strip():
                    raise ArtifactCompatibilityError(
                        "T5_6B_DIRTY_EXECUTION_SOURCE: Untracked files found in src/ or configs/"
                    )

                cmd = ["git", "rev-parse", "HEAD"]
                current_sha = (
                    subprocess.check_output(
                        cmd, cwd=self.repo_root, text=True
                    )
                    .strip()
                )
            except Exception as e:
                raise ArtifactCompatibilityError(
                    f"Failed to query git source commit SHA: {e}"
                ) from e

            if not self.expected_measurement_source_sha:
                raise ArtifactCompatibilityError(
                    "T5_6B_MISSING_EXPECTED_MEASUREMENT_SOURCE_SHA: --expected-measurement-source-sha is mandatory in execution mode"
                )

            if current_sha != self.expected_measurement_source_sha:
                raise ArtifactCompatibilityError(
                    f"T5_6B_MEASUREMENT_SOURCE_SHA_MISMATCH: expected {self.expected_measurement_source_sha}, got {current_sha}"
                )

            return current_sha

        return self.expected_measurement_source_sha or "preflight_uncommitted"

    def preflight_check(self) -> dict[str, Any]:
        """Run strict offline verification without loading models or executing inference."""
        packets = load_frozen_tune20_packets(self.archive_path)
        source_sha = self.verify_pre_execution_authorities()

        claim_cfg = get_preregistered_claim_verification_config()
        claim_sha = canonical_sha256(claim_cfg)
        gen_hashes = {
            name: canonical_sha256(
                get_preregistered_generation_config(
                    ARM_CONTRACT_MAP[name]
                )
            )
            for name in EXECUTION_ARM_ORDER
        }

        return {
            "preflight_status": "SUCCESS",
            "archive_sha256": compute_file_sha256(self.archive_path),
            "tune20_qids_sha256": compute_tune20_ordered_qids_sha256(
                [p.question_id for p in packets]
            ),
            "frozen_inputs_sha256": compute_frozen_generator_input_sha256(
                packets
            ),
            "production_generator_blob_sha": verify_production_generator_blob(
                self.repo_root
            ),
            "claim_config_sha256": claim_sha,
            "generation_config_hashes": gen_hashes,
            "records_validated": len(packets),
            "evidence_count_total": sum(
                len(p.selected_evidence) for p in packets
            ),
            "model_instantiated": False,
            "provider_called": False,
            "inference_run": False,
            "model_tree_gate": "MODEL_TREE_GATE_DEFERRED_TO_KAGGLE_PRE_EXECUTION",
            "cuda_gate": "CUDA_GATE_DEFERRED_TO_KAGGLE_PRE_EXECUTION",
            "scorer_gate": "NOT_APPLICABLE_TO_GENERATION_PHASE",
        }

    def _validate_and_reload_arm_resume_state(
        self,
        arm_name: str,
        completed_qids: list[str],
        arm_dir: Path,
    ) -> list[QuestionMeasurementResult]:
        """Validate all resume artifacts deeply across 5 files and reload completed QuestionMeasurementResults."""
        if not completed_qids:
            return []

        prompt_mode = ARM_CONTRACT_MAP[arm_name]
        resp_file = arm_dir / "responses.jsonl"
        res_file = arm_dir / "question_results.jsonl"
        call_file = arm_dir / "call_telemetry.jsonl"
        raw_file = arm_dir / "raw_completions.jsonl"
        grd_file = arm_dir / "grounding_telemetry.jsonl"

        for p in (resp_file, res_file, call_file, raw_file, grd_file):
            if not p.is_file():
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Missing artifact file {p.name} on resume"
                )

        canonical_prefix = list(
            CANONICAL_TUNE20_ORDERED_QIDS[: len(completed_qids)]
        )
        if completed_qids != canonical_prefix:
            raise ArtifactCompatibilityError(
                "PARTIAL_QID_ARTIFACT_STATE: Out-of-order completed QIDs on resume"
            )

        # 1. Question Results
        results_lines = [
            line
            for line in res_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        q_results: list[QuestionMeasurementResult] = []
        for line in results_lines:
            q_results.append(
                QuestionMeasurementResult.model_validate(json.loads(line))
            )

        loaded_qids = [r.question_id for r in q_results]
        if loaded_qids != completed_qids:
            raise ArtifactCompatibilityError(
                f"PARTIAL_QID_ARTIFACT_STATE: QIDs in {res_file.name} do not match completed_qids"
            )

        # 2. Responses
        resp_records = [
            json.loads(line)
            for line in resp_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(resp_records) != len(completed_qids):
            raise ArtifactCompatibilityError(
                "PARTIAL_QID_ARTIFACT_STATE: Response record count mismatch"
            )
        for expected_qid, resp in zip(completed_qids, resp_records, strict=True):
            if resp.get("question_id") != expected_qid:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Response QID mismatch: expected {expected_qid}, got {resp.get('question_id')}"
                )

        # 3. Call Telemetry
        call_records = [
            GeneratorCallTelemetry.model_validate(json.loads(line))
            for line in call_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        calls_by_qid: dict[str, list[GeneratorCallTelemetry]] = {}
        for c in call_records:
            if c.question_id not in completed_qids:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Call telemetry contains uncompleted/future QID {c.question_id}"
                )
            if c.candidate_contract != prompt_mode:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Call telemetry candidate contract mismatch: {c.candidate_contract}"
                )
            calls_by_qid.setdefault(c.question_id, []).append(c)

        # 4. Raw Completions
        raw_records = [
            json.loads(line)
            for line in raw_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        raw_by_qid: dict[str, list[dict[str, Any]]] = {}
        for r in raw_records:
            qid = r.get("question_id")
            if qid not in completed_qids:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Raw completions contain uncompleted QID {qid}"
                )
            raw_by_qid.setdefault(qid, []).append(r)

        # 5. Grounding Telemetry
        grd_records = [
            GroundingCallTelemetry.model_validate(json.loads(line))
            for line in grd_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        grd_by_qid: dict[str, list[GroundingCallTelemetry]] = {}
        for g in grd_records:
            if g.question_id not in completed_qids:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Grounding telemetry contains uncompleted QID {g.question_id}"
                )
            if g.candidate_contract != prompt_mode:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Grounding telemetry contract mismatch: {g.candidate_contract}"
                )
            grd_by_qid.setdefault(g.question_id, []).append(g)

        # Cross-validate embedded structures for each question
        for q_res in q_results:
            qid = q_res.question_id
            ext_calls = calls_by_qid.get(qid, [])
            if [c.model_dump() for c in q_res.calls] != [
                c.model_dump() for c in ext_calls
            ]:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Embedded calls mismatch for QID {qid}"
                )

            # Check uniqueness of call identities
            call_identities = [
                (c.question_id, c.call_index, c.provider_attempt_index)
                for c in ext_calls
            ]
            if len(call_identities) != len(set(call_identities)):
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Duplicate call identity for QID {qid}"
                )

            succ_calls = [
                c
                for c in ext_calls
                if c.provider_call_success and c.raw_completion_text is not None
            ]
            ext_raw = raw_by_qid.get(qid, [])
            if len(succ_calls) != len(ext_raw):
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Raw completions count mismatch for QID {qid}: expected {len(succ_calls)}, got {len(ext_raw)}"
                )

            # Check uniqueness of raw identities
            raw_identities = [
                (
                    r.get("question_id"),
                    r.get("call_index"),
                    r.get("provider_attempt_index"),
                )
                for r in ext_raw
            ]
            if len(raw_identities) != len(set(raw_identities)):
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Duplicate raw completion record for QID {qid}"
                )

            for c in succ_calls:
                matching_raw = [
                    r
                    for r in ext_raw
                    if r.get("question_id") == c.question_id
                    and r.get("call_index") == c.call_index
                    and r.get("provider_attempt_index") == c.provider_attempt_index
                ]
                if len(matching_raw) != 1:
                    raise ArtifactCompatibilityError(
                        f"PARTIAL_QID_ARTIFACT_STATE: Missing or ambiguous raw completion record for call {c.call_index} in QID {qid}"
                    )
                r_item = matching_raw[0]
                if r_item.get("raw_completion_text") != c.raw_completion_text:
                    raise ArtifactCompatibilityError(
                        f"PARTIAL_QID_ARTIFACT_STATE: Raw completion text mismatch for call {c.call_index} in QID {qid}"
                    )
                if (
                    "raw_completion_sha256" in r_item
                    and r_item["raw_completion_sha256"]
                    != sha256(c.raw_completion_text.encode("utf-8")).hexdigest()
                ):
                    raise ArtifactCompatibilityError(
                        f"PARTIAL_QID_ARTIFACT_STATE: Raw completion SHA mismatch for call {c.call_index} in QID {qid}"
                    )

            ext_grd = grd_by_qid.get(qid, [])
            if [g.model_dump() for g in q_res.grounding_calls] != [
                g.model_dump() for g in ext_grd
            ]:
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Embedded grounding calls mismatch for QID {qid}"
                )

            grd_identities = [
                (g.question_id, g.verification_index) for g in ext_grd
            ]
            if len(grd_identities) != len(set(grd_identities)):
                raise ArtifactCompatibilityError(
                    f"PARTIAL_QID_ARTIFACT_STATE: Duplicate grounding verification index for QID {qid}"
                )

        return q_results

    def run_generation(self) -> T56BGenerationManifest:
        """Execute closed Design-B generation across all 3 candidate arms (strictly zero reference answers)."""
        if not _RUNNER_LOCK.acquire(blocking=False):
            raise BackendInitializationError(
                "T5_6B_OVERLAPPING_RUNNER: Another measurement runner is already active. "
                "Overlapping runner contexts are rejected to prevent artifact mutation races."
            )
        try:
            return self._run_generation_internal()
        finally:
            _RUNNER_LOCK.release()

    def _run_generation_internal(self) -> T56BGenerationManifest:
        packets = load_frozen_tune20_packets(self.archive_path)
        verified_source_sha = self.verify_pre_execution_authorities()

        if not self.is_preflight_only:
            if not self.model_path or not self.model_path.is_dir():
                raise ArtifactCompatibilityError(
                    f"Model directory not found: {self.model_path}"
                )

            base_generation_config = get_preregistered_generation_config("plain_text_markers")
            expected_model_path = Path(base_generation_config.model_name).resolve()
            if self.model_path.resolve() != expected_model_path:
                raise ArtifactCompatibilityError(
                    f"T5_6B_MODEL_PATH_AUTHORITY_MISMATCH: expected {expected_model_path}, got {self.model_path.resolve()}"
                )

            actual_tree_sha = compute_directory_sha256(self.model_path)
            if actual_tree_sha != self.expected_model_tree_sha256:
                raise ArtifactCompatibilityError(
                    f"EXPECTED_M49_GENERATOR_TREE_SHA256 mismatch: expected {self.expected_model_tree_sha256}, got {actual_tree_sha}"
                )

        env_fp, env_fp_sha = get_runtime_environment_fingerprint(
            repo_root=self.repo_root,
            measurement_source_sha=verified_source_sha,
            model_path=self.model_path,
            model_tree_sha256=self.expected_model_tree_sha256,
        )

        if not self.is_preflight_only:
            if not env_fp.cuda_available or env_fp.gpu_count < 1:
                raise ArtifactCompatibilityError(
                    "T5_6B_ENVIRONMENT_AUTHORITY_FAILED: CUDA GPU runtime is required for measurement execution"
                )

        if self.provider_factory is None:
            raise BackendInitializationError(
                "provider_factory must be provided for generation run"
            )

        base_cfg = get_preregistered_generation_config("plain_text_markers")
        claim_cfg = get_preregistered_claim_verification_config()

        # SINGLE RUNTIME: raw provider constructed exactly once and shared across all 3 arms
        raw_provider = self.provider_factory(base_cfg)
        provider_observer = MeasurementProviderObserver(raw_provider)
        if isinstance(raw_provider, ObservableTransformersChatProvider):
            raw_provider.token_observer = provider_observer

        raw_verifier = RuleBasedCitationVerifier(claim_cfg)
        verifier_observer = ObservableCitationVerifier(raw_verifier)

        logger_observer = ModelGeneratorRejectionObserver()
        # Wire rejection observer to provider observer so set_active_logical_call
        # is called before each provider execution for correct rejection attribution
        provider_observer._rejection_observer = logger_observer
        target_logger_name = ModelBackedAnswerGenerator.__module__

        self.output_dir.mkdir(parents=True, exist_ok=True)
        batch_state_file = self.output_dir / "batch_state.json"

        completed_state: dict[str, list[str]] = {
            "control": [],
            "compact": [],
            "json_schema": [],
        }
        if batch_state_file.is_file():
            saved_state = json.loads(
                batch_state_file.read_text(encoding="utf-8")
            )
            if (
                saved_state.get("frozen_generator_input_sha256")
                != T5_6B_FROZEN_GENERATOR_INPUT_SHA256
            ):
                raise ArtifactCompatibilityError(
                    "Batch state input authority mismatch on resume"
                )
            if (
                saved_state.get("runtime_environment_sha256")
                != env_fp_sha
            ):
                raise ArtifactCompatibilityError(
                    "T5_6B_RESUME_ENVIRONMENT_MISMATCH: Saved batch state runtime environment differs from current"
                )
            completed_state.update(saved_state.get("completed_qids", {}))

            # Arm-order fail-closed: later arms must not have progress ahead of earlier arms
            ctrl_done = len(completed_state.get("control", []))
            compact_done = len(completed_state.get("compact", []))
            json_done = len(completed_state.get("json_schema", []))
            if compact_done > 0 and ctrl_done < 20:
                raise ArtifactCompatibilityError(
                    f"T5_6B_ARM_ORDER_VIOLATED: compact has {compact_done} completed QIDs "
                    f"while control has only {ctrl_done} / 20"
                )
            if json_done > 0 and compact_done < 20:
                raise ArtifactCompatibilityError(
                    f"T5_6B_ARM_ORDER_VIOLATED: json_schema has {json_done} completed QIDs "
                    f"while compact has only {compact_done} / 20"
                )

        results_for_arm: dict[str, list[QuestionMeasurementResult]] = {}
        generation_artifact_hashes: dict[str, str] = {}
        arm_generation_bundle_hashes: dict[str, str] = {}

        for arm_name in EXECUTION_ARM_ORDER:
            prompt_mode = ARM_CONTRACT_MAP[arm_name]
            gen_cfg = get_preregistered_generation_config(prompt_mode)

            generator = ModelBackedAnswerGenerator(
                provider=provider_observer,
                max_structured_output_retries=gen_cfg.max_structured_output_retries,
                max_model_error_retries=gen_cfg.max_model_error_retries,
                model_failure_policy=gen_cfg.model_failure_policy,
                answer_style=gen_cfg.answer_style,
                prompt_schema_mode=gen_cfg.prompt_schema_mode,
                grounding_verifier=verifier_observer,
                max_grounding_repair_retries=gen_cfg.max_grounding_repair_retries,
                grounding_failure_policy=gen_cfg.grounding_failure_policy,
                extractive_fallback_max_evidence=gen_cfg.extractive_fallback_max_evidence,
                salvage_rendering=gen_cfg.salvage_rendering,
            )

            arm_dir = self.output_dir / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)

            already_done = list(completed_state.get(arm_name, []))
            reloaded_results = self._validate_and_reload_arm_resume_state(
                arm_name, already_done, arm_dir
            )
            results_for_arm[arm_name] = list(reloaded_results)

            resp_path = arm_dir / "responses.jsonl"
            res_path = arm_dir / "question_results.jsonl"
            call_path = arm_dir / "call_telemetry.jsonl"
            grd_path = arm_dir / "grounding_telemetry.jsonl"
            raw_path = arm_dir / "raw_completions.jsonl"

            for packet in packets:
                if packet.question_id in already_done:
                    continue

                logger_observer.clear()
                provider_observer.set_active_question(
                    packet.question_id, prompt_mode
                )
                verifier_observer.set_active_question(
                    packet.question_id, prompt_mode
                )
                logger_observer.set_active_context(
                    packet.question_id, prompt_mode
                )

                call_start_idx = len(provider_observer.call_telemetry)
                grounding_start_idx = len(
                    verifier_observer.grounding_telemetry
                )

                query = reconstruct_query(packet.question, question_id=packet.question_id)

                with ModelGeneratorLoggingLease(
                    target_logger_name, logger_observer
                ):
                    response = generator.generate(
                        query=query,
                        evidence=packet.selected_evidence,
                        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
                        trace_id=f"t5-6b-{arm_name}-{packet.question_id}",
                    )

                logger_observer.clear_active_context()

                captured_calls = list(
                    provider_observer.call_telemetry[call_start_idx:]
                )
                for call_rec in captured_calls:
                    if call_rec.question_id != packet.question_id:
                        raise DataValidationError(
                            f"Call telemetry question_id mismatch: expected {packet.question_id}, got {call_rec.question_id}"
                        )
                    if call_rec.candidate_contract != prompt_mode:
                        raise DataValidationError(
                            f"Call telemetry candidate_contract mismatch: expected {prompt_mode}, got {call_rec.candidate_contract}"
                        )

                correlate_rejection_events(
                    captured_calls, logger_observer.rejections
                )

                captured_grounding = list(
                    verifier_observer.grounding_telemetry[
                        grounding_start_idx:
                    ]
                )
                for grd_rec in captured_grounding:
                    if grd_rec.question_id != packet.question_id:
                        raise DataValidationError(
                            f"Grounding telemetry question_id mismatch: expected {packet.question_id}, got {grd_rec.question_id}"
                        )
                    if grd_rec.candidate_contract != prompt_mode:
                        raise DataValidationError(
                            f"Grounding telemetry candidate_contract mismatch: expected {prompt_mode}, got {grd_rec.candidate_contract}"
                        )

                final_path = classify_final_generator_path(response)
                cit_valid = evaluate_citation_identity_validity(
                    response, packet.selected_evidence
                )
                # parser_accepted: True if at least one successful provider completion
                # was parsed and validated into a ModelAnswerDraft before model-error
                # top-evidence fallback. Extractive fallback does NOT count.
                parser_acc = any(
                    c.parse_result == "ACCEPTED"
                    for c in captured_calls
                    if c.provider_call_success
                )
                # had_contract_rejection: True only if at least one call had
                # rejection_error_type == "structured_output_schema"
                had_rej = any(
                    c.rejection_error_type == "structured_output_schema"
                    for c in captured_calls
                )

                q_result = QuestionMeasurementResult(
                    question_id=packet.question_id,
                    candidate_contract=prompt_mode,
                    response=response,
                    final_generator_path=final_path,
                    citation_identity_valid=cit_valid,
                    parser_accepted=parser_acc,
                    had_contract_rejection=had_rej,
                    calls=captured_calls,
                    grounding_calls=captured_grounding,
                )

                results_for_arm[arm_name].append(q_result)

                # Append atomic JSONL records
                with resp_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "question_id": packet.question_id,
                                "answer": response.answer,
                                "citations": [
                                    c.model_dump() for c in response.citations
                                ],
                                "warnings": response.warnings,
                                "insufficient_evidence": response.insufficient_evidence,
                                "final_generator_path": final_path,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()
                    os.fsync(f.fileno())

                with res_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            q_result.model_dump(), ensure_ascii=False
                        )
                        + "\n"
                    )
                    f.flush()
                    os.fsync(f.fileno())

                with call_path.open("a", encoding="utf-8") as f:
                    for c in q_result.calls:
                        f.write(
                            json.dumps(c.model_dump(), ensure_ascii=False)
                            + "\n"
                        )
                    f.flush()
                    os.fsync(f.fileno())

                with grd_path.open("a", encoding="utf-8") as f:
                    for g in q_result.grounding_calls:
                        f.write(
                            json.dumps(g.model_dump(), ensure_ascii=False)
                            + "\n"
                        )
                    f.flush()
                    os.fsync(f.fileno())

                with raw_path.open("a", encoding="utf-8") as f:
                    for c in q_result.calls:
                        if c.raw_completion_text is not None:
                            raw_sha = sha256(
                                c.raw_completion_text.encode("utf-8")
                            ).hexdigest()
                            f.write(
                                json.dumps(
                                    {
                                        "question_id": c.question_id,
                                        "call_index": c.call_index,
                                        "provider_attempt_index": c.provider_attempt_index,
                                        "raw_completion_text": c.raw_completion_text,
                                        "raw_completion_sha256": raw_sha,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    f.flush()
                    os.fsync(f.fileno())

                completed_state[arm_name].append(packet.question_id)
                tmp_state_file = self.output_dir / "batch_state.json.tmp"
                tmp_state_file.write_text(
                    json.dumps(
                        {
                            "frozen_generator_input_sha256": T5_6B_FROZEN_GENERATOR_INPUT_SHA256,
                            "runtime_environment_sha256": env_fp_sha,
                            "completed_qids": completed_state,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                tmp_state_file.replace(batch_state_file)

            generation_artifact_hashes[
                f"{arm_name}/responses.jsonl"
            ] = compute_file_sha256(resp_path)
            generation_artifact_hashes[
                f"{arm_name}/question_results.jsonl"
            ] = compute_file_sha256(res_path)
            generation_artifact_hashes[
                f"{arm_name}/call_telemetry.jsonl"
            ] = compute_file_sha256(call_path)
            generation_artifact_hashes[
                f"{arm_name}/grounding_telemetry.jsonl"
            ] = compute_file_sha256(grd_path)
            generation_artifact_hashes[
                f"{arm_name}/raw_completions.jsonl"
            ] = compute_file_sha256(raw_path)

            arm_bundle_digest = sha256()
            for key in (
                f"{arm_name}/responses.jsonl",
                f"{arm_name}/question_results.jsonl",
                f"{arm_name}/call_telemetry.jsonl",
                f"{arm_name}/grounding_telemetry.jsonl",
                f"{arm_name}/raw_completions.jsonl",
            ):
                arm_bundle_digest.update(
                    generation_artifact_hashes[key].encode("ascii")
                )
            arm_generation_bundle_hashes[arm_name] = (
                arm_bundle_digest.hexdigest()
            )

        gen_closed_at = datetime.now(timezone.utc).isoformat()
        generation_manifest = T56BGenerationManifest(
            created_at=datetime.now(timezone.utc).isoformat(),
            measurement_source_sha=verified_source_sha,
            production_generator_blob_sha=verify_production_generator_blob(
                self.repo_root
            ),
            model_artifact_tree_sha256=self.expected_model_tree_sha256,
            resolved_model_path=str(
                self.model_path.resolve().as_posix()
            )
            if self.model_path
            else "/kaggle/working/m49-generator-merged",
            runtime_environment_sha256=env_fp_sha,
            runtime_environment=env_fp,
            control_generation_config_sha256=T5_6B_CONTROL_GENERATION_CONFIG_SHA256,
            compact_generation_config_sha256=T5_6B_COMPACT_GENERATION_CONFIG_SHA256,
            json_generation_config_sha256=T5_6B_JSON_GENERATION_CONFIG_SHA256,
            claim_verification_config_sha256=T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256,
            frozen_generator_input_sha256=T5_6B_FROZEN_GENERATOR_INPUT_SHA256,
            tune20_ordered_qids_sha256=T5_6B_TUNE20_ORDERED_QIDS_SHA256,
            execution_order=list(EXECUTION_ARM_ORDER),
            record_count=len(packets),
            generation_closed_at=gen_closed_at,
            generation_artifact_hashes=generation_artifact_hashes,
        )

        gen_manifest_path = self.output_dir / "generation_manifest.json"
        gen_manifest_path.write_text(
            json.dumps(generation_manifest.model_dump(), indent=2),
            encoding="utf-8",
        )

        return generation_manifest

    def score_closed_generation(
        self,
        reference_answers: dict[str, str],
        scorer_path: Path,
    ) -> T56BMeasurementManifest:
        """Score closed generation artifacts against official references using official scorer program."""
        gen_manifest_path = self.output_dir / "generation_manifest.json"
        if not gen_manifest_path.is_file():
            raise ArtifactCompatibilityError(
                f"generation_manifest.json not found in {self.output_dir}"
            )

        gen_manifest = T56BGenerationManifest.model_validate(
            json.loads(gen_manifest_path.read_text(encoding="utf-8"))
        )

        # Verify generation artifact hashes before reading
        for rel_path, exp_hash in gen_manifest.generation_artifact_hashes.items():
            full_path = self.output_dir / rel_path
            if not full_path.is_file():
                raise ArtifactCompatibilityError(
                    f"Generation artifact file missing: {full_path}"
                )
            act_hash = compute_file_sha256(full_path)
            if act_hash != exp_hash:
                raise ArtifactCompatibilityError(
                    f"Generation artifact hash mismatch for {rel_path}: expected {exp_hash}, got {act_hash}"
                )

        arm_summaries: dict[str, ArmMeasurementSummary] = {}
        scoring_artifact_hashes: dict[str, str] = {}
        verified_scorer_sha = OFFICIAL_SCORER_ARCHIVE_SHA256

        for arm_name in gen_manifest.execution_order:
            arm_dir = self.output_dir / arm_name
            prompt_mode = ARM_CONTRACT_MAP[arm_name]

            res_lines = [
                line
                for line in (arm_dir / "question_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            q_results = [
                QuestionMeasurementResult.model_validate(json.loads(line))
                for line in res_lines
            ]

            preds = {
                r.question_id: render_competition_answer(r.response)
                for r in q_results
            }

            mean_rouge, mean_meteor, per_q_scores, verified_scorer_sha = (
                score_tune20_answers(
                    predicted_answers=preds,
                    reference_answers=reference_answers,
                    scorer_path=scorer_path,
                )
            )

            p_acc_count = sum(1 for r in q_results if r.parser_accepted)
            cit_valid_count = sum(
                1 for r in q_results if r.citation_identity_valid
            )
            # contract_rejection_fallback_count: questions where MODEL_ERROR_FALLBACK
            # AND had_contract_rejection (structured_output_schema errors).
            # SUPPORTED_CLAIM_SALVAGE and GROUNDING_EXTRACTIVE_FALLBACK are excluded.
            fallback_count = sum(
                1
                for r in q_results
                if r.final_generator_path == "MODEL_ERROR_FALLBACK"
                and r.had_contract_rejection
            )
            insufficient_count = sum(
                1
                for r in q_results
                if r.response.insufficient_evidence
            )
            # total_structured_output_rejections: count calls with
            # rejection_error_type == "structured_output_schema" only
            total_rejections = sum(
                sum(
                    1
                    for c in r.calls
                    if c.rejection_error_type == "structured_output_schema"
                )
                for r in q_results
            )

            inp_tokens = [
                c.input_token_count
                for r in q_results
                for c in r.calls
                if c.input_token_count is not None
            ]
            out_tokens = [
                c.output_token_count
                for r in q_results
                for c in r.calls
                if c.output_token_count is not None
            ]

            mean_inp = float(np.mean(inp_tokens)) if inp_tokens else None
            mean_out = float(np.mean(out_tokens)) if out_tokens else None

            path_dist: dict[str, int] = {}
            for r in q_results:
                path_dist[r.final_generator_path] = (
                    path_dist.get(r.final_generator_path, 0) + 1
                )

            summary = ArmMeasurementSummary(
                candidate_contract=prompt_mode,
                record_count=len(q_results),
                rouge_l=mean_rouge,
                meteor=mean_meteor,
                parser_acceptance_rate=p_acc_count / len(q_results),
                citation_identity_validity=cit_valid_count
                / len(q_results),
                contract_rejection_fallback_count=fallback_count,
                insufficient_evidence_count=insufficient_count,
                total_structured_output_rejections=total_rejections,
                mean_input_tokens=mean_inp,
                mean_output_tokens=mean_out,
                final_path_distribution=path_dist,
                per_question_scores=per_q_scores,
            )
            arm_summaries[arm_name] = summary

            summary_path = arm_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary.model_dump(), indent=2),
                encoding="utf-8",
            )
            scoring_artifact_hashes[
                f"{arm_name}/summary.json"
            ] = compute_file_sha256(summary_path)

        adv_results, winner, phase_decision = evaluate_advancement_gate(
            arm_summaries
        )

        comp_path = self.output_dir / "comparison.json"
        comp_payload = {
            "execution_order": gen_manifest.execution_order,
            "arm_summaries": {
                k: v.model_dump() for k, v in arm_summaries.items()
            },
            "advancement_results": {
                k: v.model_dump() for k, v in adv_results.items()
            },
            "advancement_winner": winner,
            "phase_decision": phase_decision,
        }
        comp_path.write_text(
            json.dumps(comp_payload, indent=2), encoding="utf-8"
        )
        scoring_artifact_hashes[
            "comparison.json"
        ] = compute_file_sha256(comp_path)

        measurement_manifest = T56BMeasurementManifest(
            created_at=datetime.now(timezone.utc).isoformat(),
            measurement_source_sha=gen_manifest.measurement_source_sha,
            production_generator_blob_sha=gen_manifest.production_generator_blob_sha,
            model_artifact_tree_sha256=gen_manifest.model_artifact_tree_sha256,
            runtime_environment_sha256=gen_manifest.runtime_environment_sha256,
            runtime_environment=gen_manifest.runtime_environment,
            control_generation_config_sha256=gen_manifest.control_generation_config_sha256,
            compact_generation_config_sha256=gen_manifest.compact_generation_config_sha256,
            json_generation_config_sha256=gen_manifest.json_generation_config_sha256,
            claim_verification_config_sha256=gen_manifest.claim_verification_config_sha256,
            frozen_generator_input_sha256=gen_manifest.frozen_generator_input_sha256,
            tune20_ordered_qids_sha256=gen_manifest.tune20_ordered_qids_sha256,
            official_scorer_sha256=verified_scorer_sha,
            execution_order=gen_manifest.execution_order,
            record_count=gen_manifest.record_count,
            generation_manifest_hash=compute_file_sha256(
                gen_manifest_path
            ),
            generation_artifact_hashes=gen_manifest.generation_artifact_hashes,
            scoring_artifact_hashes=scoring_artifact_hashes,
            arm_summaries=arm_summaries,
            advancement_results=adv_results,
            advancement_winner=winner,
            phase_decision=phase_decision,
        )

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(measurement_manifest.model_dump(), indent=2),
            encoding="utf-8",
        )

        return measurement_manifest


# ==============================================================================
# CLI ARGUMENT PARSER & MAIN ENTRYPOINT
# ==============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with mutually exclusive phase modes."""
    parser = argparse.ArgumentParser(
        description="T5-6B Generator Contract Measurement Runner"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run strict offline verification without loading models or executing inference",
    )
    mode_group.add_argument(
        "--execute-generation",
        action="store_true",
        help="Execute controlled Design-B generation across 3 arms (zero reference answers)",
    )
    mode_group.add_argument(
        "--score-closed-generation",
        action="store_true",
        help="Score closed generation artifacts against official references (zero model execution)",
    )

    # Generation phase parameters
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to t5-1c-fast30-clean1-evidence.zip archive",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Path to repository root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/t5_6b_measurement"),
        help="Path to output measurement directory",
    )
    parser.add_argument(
        "--expected-measurement-source-sha",
        type=str,
        default=None,
        help="Expected measurement source Git commit SHA",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to merged M49 generator model directory",
    )

    # Scoring phase parameters
    parser.add_argument(
        "--scorer-path",
        type=Path,
        default=None,
        help="Path to official scorer archive (Scoring-Program-Task-LegalQA.zip)",
    )
    parser.add_argument(
        "--reference-answers-file",
        type=Path,
        default=None,
        help="Path to JSON file mapping Tune20 QIDs to reference answers",
    )
    return parser


def parse_cli_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI arguments, strictly enforcing structural firewall boundaries."""
    parser = build_arg_parser()
    parsed = parser.parse_args(args)

    if parsed.preflight_only:
        if not parsed.archive:
            parser.error("--archive is required for --preflight-only")
        if parsed.reference_answers_file is not None or parsed.scorer_path is not None:
            parser.error("Reference answers and scorer path are strictly forbidden in --preflight-only mode")

    elif parsed.execute_generation:
        if not parsed.archive:
            parser.error("--archive is required for --execute-generation")
        if parsed.reference_answers_file is not None or parsed.scorer_path is not None:
            parser.error("Reference answers and scorer path are strictly forbidden in --execute-generation mode")

    elif parsed.score_closed_generation:
        if not parsed.reference_answers_file:
            parser.error("--reference-answers-file is required for --score-closed-generation")
        if not parsed.scorer_path:
            parser.error("--scorer-path is required for --score-closed-generation")
        if parsed.model_path is not None:
            parser.error("--model-path is strictly forbidden in --score-closed-generation mode")

    return parsed


def main() -> None:
    """CLI entrypoint for separated preflight, generation execution, and closed scoring modes."""
    args = parse_cli_args()

    if args.preflight_only:
        runner = T5GeneratorContractMeasurementRunner(
            repo_root=args.repo_root,
            archive_path=args.archive,
            output_dir=args.output_dir,
            model_path=args.model_path,
            expected_measurement_source_sha=args.expected_measurement_source_sha,
            is_preflight_only=True,
        )
        report = runner.preflight_check()
        print(json.dumps(report, indent=2))
        return

    if args.execute_generation:
        runner = T5GeneratorContractMeasurementRunner(
            repo_root=args.repo_root,
            archive_path=args.archive,
            output_dir=args.output_dir,
            provider_factory=measurement_transformers_provider_factory,
            model_path=args.model_path,
            expected_measurement_source_sha=args.expected_measurement_source_sha,
            is_preflight_only=False,
        )
        gen_manifest = runner.run_generation()
        print(f"Generation phase closed successfully! Manifest: {gen_manifest.schema_version}")
        return

    if args.score_closed_generation:
        runner = T5GeneratorContractMeasurementRunner(
            repo_root=args.repo_root,
            archive_path=args.archive or Path("dummy.zip"),
            output_dir=args.output_dir,
            is_preflight_only=True,
        )
        raw_refs = json.loads(args.reference_answers_file.read_text(encoding="utf-8"))
        ref_answers: dict[str, str] = {}
        for qid, val in raw_refs.items():
            if isinstance(val, dict):
                ref_answers[str(qid)] = str(val.get("answer", ""))
            else:
                ref_answers[str(qid)] = str(val)

        meas_manifest = runner.score_closed_generation(
            reference_answers=ref_answers,
            scorer_path=args.scorer_path,
        )
        print(f"Scoring phase completed successfully! Final winner: {meas_manifest.advancement_winner}")
        return


if __name__ == "__main__":
    main()
