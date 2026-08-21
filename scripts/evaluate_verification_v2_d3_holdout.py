#!/usr/bin/env python3
"""V2-D3 Structured Semantic Verifier Fresh Holdout Benchmark Evaluation Harness.

This script executes the controlled offline evaluation of the frozen candidate V2-D3
over the pre-registered fresh holdout dataset (Task B-HOLDOUT-SEALED / B-HOLDOUT-EXEC).

Key Principles & Invariants:
- Candidate Under Evaluation: V2-D3 ONLY (D3.1 and D3.2 are CLOSED as KEEP_D3; NO D3.3).
- Frozen D3 Semantic Identity: structured_semantic_verifier_d3.py source SHA, system
  instruction SHA, output schema, prompt structure, and deterministic label derivation
  must match the canonical frozen D3 identities (0d95aeee...).
- Two-Pass Stability: repeat_count = 2. Pass 1 is authoritative primary evaluation;
  Pass 2 is stability evaluation only.
- Strict Error-Aware Denominators: Execution errors are never counted as correct
  rejections, contradictions, or valid answer retractions.
- Content-Safe Telemetry: Observational proxy records call hashes and timings without
  raw prompt/completion exposure.
- Fail-Closed Label Verification: Zero fallbacks (no packet entailment fallback, no
  INSUFFICIENT fallback). Strict set equality required between packet claims and human labels.
- Non-Vacuous Coverage Gates: Supported claim denominator > 0, negative claim denominator > 0,
  valid answer denominator > 0, invalid answer denominator > 0 required for promotion eligibility.
- Zero-Denominator Semantics: Rate metrics report None when denominator is zero, not 1.0.
- Pre-Registered Rate-Based Promotion Gate: Pre-registered rate thresholds (supported
  retention >= 88%, negative catch >= 50%, valid answer retention >= 80%, full answer
  accuracy >= 60%, claim binary accuracy >= 70%) evaluated on Phase H-EXEC.
- Fail-Closed Governance: promotion_authorized is ALWAYS False in harness outputs;
  actual production promotion requires subsequent human sign-off and wiring tasks.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any
import zipfile

import legal_agentic_rag
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
import legal_agentic_rag.generation.structured_semantic_verifier_d3
from legal_agentic_rag.generation.structured_semantic_verifier_d3 import (
    D3EvidenceRelation,
    D3StructuredClaimAssessmentDraft,
    DraftRejectionCategory,
    STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION,
    StructuredClaimVerificationD3,
    StructuredSemanticCitationVerifierD3,
    StructuredSemanticVerificationResultD3,
    derive_claim_semantic_label_d3,
)
from legal_agentic_rag.generation.transformers_provider import TransformersChatProvider
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    CitationVerificationResult,
    ClaimSupportStatus,
    Evidence,
    SemanticSupportLabel,
)

_LOGGER = logging.getLogger(__name__)

# Canonical Frozen V2-D3 Implementation & Prompt Checksums
CANONICAL_D3_EVIDENCE_ZIP_SHA256 = (
    "0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff"
)
CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256 = (
    "546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b"
)
CANONICAL_D3_IMPLEMENTATION_SHA256 = (
    "a6e8bca15ad14d869e103e1f94fe94bb9a81f9ddc8bc650b280b69b7d57e9826"
)
CANONICAL_D3_SCHEMA_SHA256 = (
    "3591144a40b0519d5da9dd262e8edf8814531d798b69deea94fd81fae39f5f61"
)

# Canonical Pre-Registered Holdout Outer Signatures
CANONICAL_HOLDOUT_SELECTION_SHA256 = (
    "08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b"
)
CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256 = (
    "a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4"
)
GOVERNANCE_STATUS_EXTERNALLY_REVIEWED = "EXTERNALLY_REVIEWED_FOR_H_EXEC"

# Canonical Pinned V2-D3 Model & Execution Parameters
CANONICAL_PACKAGE_VERSION = "0.50.7"
CANONICAL_CANDIDATE_ID = "V2-D3"
CANONICAL_V3_BACKEND = "transformers"
CANONICAL_V3_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CANONICAL_V3_MODEL_REVISION = "a1d308dfcc03e09da285d49d912439a655a571e8"
CANONICAL_V3_PROVIDER_VERSION = "4.47.1"
CANONICAL_V3_DEVICE = "cuda"
CANONICAL_V3_TORCH_DTYPE = "float16"
CANONICAL_V3_TEMPERATURE = 0.0
CANONICAL_V3_MAX_INPUT_TOKENS = 8192
CANONICAL_V3_MAX_OUTPUT_TOKENS = 512
CANONICAL_V3_MAX_STRUCTURED_RETRIES = 1
CANONICAL_V3_TIMEOUT_SECONDS = 180.0
CANONICAL_REPEAT_COUNT = 2

# Pre-Registered Promotion Gate Rate Thresholds
GATE_MIN_SUPPORTED_RETENTION_RATE = 0.88  # Dev: 17/18 = 94.44%
GATE_MIN_NEGATIVE_CATCH_RATE = 0.50       # Dev: 11/20 = 55.00%
GATE_MIN_VALID_ANSWER_RETENTION_RATE = 0.80  # Dev: 6/7 = 85.71%
GATE_MIN_FULL_ANSWER_ACCURACY_RATE = 0.60   # Dev: 14/22 = 63.64%
GATE_MIN_CLAIM_BINARY_ACCURACY_RATE = 0.70  # Dev: 28/38 = 73.68%


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object pairs hook that raises DataValidationError on duplicate object keys."""
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for k, v in pairs:
        if k in seen:
            raise DataValidationError(f"Duplicate JSON key detected: '{k}'")
        seen.add(k)
        result[k] = v
    return result


class HumanEntailment(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class BinaryPrediction(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(frozen=True)
class BenchmarkClaimTarget:
    slice_id: str
    question_id: str
    arm_id: str
    claim_id: str
    claim_text: str
    claim_text_sha256: str
    human_label: HumanEntailment
    error_tags: list[str]
    diagnostic_note: str | None
    stratum: str


@dataclass(frozen=True)
class BenchmarkArmTarget:
    slice_id: str
    question_id: str
    arm_id: str
    historical_stop_reason: str
    stratum: str
    question_text: str
    answer_response: AnswerResponse
    evidence_list: list[Evidence]
    historical_verification: dict[str, Any]
    claims: list[BenchmarkClaimTarget]


class ObservationalChatModelProviderWrapper(ChatModelProvider):
    """Transparent observational proxy recording all provider interactions with content safety."""

    def __init__(self, inner: ChatModelProvider) -> None:
        self._inner = inner
        self._call_history: list[dict[str, Any]] = []
        self._total_calls = 0
        self._total_errors = 0

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def provider_version(self) -> str:
        return self._inner.provider_version

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def model_revision(self) -> str | None:
        return self._inner.model_revision

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_errors(self) -> int:
        return self._total_errors

    @property
    def call_history(self) -> list[dict[str, Any]]:
        return list(self._call_history)

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        self._total_calls += 1
        call_idx = self._total_calls
        sys_sha = sha256(system_instruction.encode("utf-8")).hexdigest()
        usr_sha = sha256(user_prompt.encode("utf-8")).hexdigest()
        prompt_len = len(user_prompt)

        start_time = perf_counter()
        call_record: dict[str, Any] = {
            "call_index": call_idx,
            "system_instruction_sha256": sys_sha,
            "user_prompt_sha256": usr_sha,
            "user_prompt_length_chars": prompt_len,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            raw_completion = self._inner.complete(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            )
            dur = perf_counter() - start_time
            completion_sha = sha256(raw_completion.encode("utf-8")).hexdigest()
            call_record.update({
                "duration_seconds": dur,
                "completion_sha256": completion_sha,
                "completion_length_chars": len(raw_completion),
                "success": True,
                "error_type": None,
            })
            self._call_history.append(call_record)
            return raw_completion
        except Exception as exc:
            dur = perf_counter() - start_time
            self._total_errors += 1
            err_msg = str(exc)
            call_record.update({
                "duration_seconds": dur,
                "completion_sha256": None,
                "completion_length_chars": 0,
                "success": False,
                "error_type": type(exc).__name__,
                "error_sha256": sha256(err_msg.encode("utf-8")).hexdigest(),
                "error_message_length": len(err_msg),
            })
            self._call_history.append(call_record)
            raise


class V2D3HoldoutBenchmarkEvaluator:
    """Authoritative evaluation harness for frozen candidate V2-D3 over fresh holdout."""

    def __init__(
        self,
        *,
        holdout_packets_path: Path | str,
        holdout_labels_path: Path | str,
        holdout_selection_path: Path | str | None = None,
        label_commitment_path: Path | str | None = None,
        holdout_labels_sha256: str | None = None,
        output_dir: Path | str = "evaluation_outputs/v2_d3_holdout_output",
        package_zip: Path | str | None = None,
        candidate_id: str = CANONICAL_CANDIDATE_ID,
        device: str = CANONICAL_V3_DEVICE,
        torch_dtype: str = CANONICAL_V3_TORCH_DTYPE,
        temperature: float = CANONICAL_V3_TEMPERATURE,
        max_input_tokens: int = CANONICAL_V3_MAX_INPUT_TOKENS,
        max_output_tokens: int = CANONICAL_V3_MAX_OUTPUT_TOKENS,
        max_structured_output_retries: int = CANONICAL_V3_MAX_STRUCTURED_RETRIES,
        timeout_seconds: float = CANONICAL_V3_TIMEOUT_SECONDS,
        repeat_count: int = CANONICAL_REPEAT_COUNT,
        preflight_only: bool = False,
        custom_provider: ChatModelProvider | None = None,
        bypass_source_checksums: bool = False,
    ) -> None:
        self._holdout_packets_path = Path(holdout_packets_path)
        self._holdout_labels_path = Path(holdout_labels_path)
        self._holdout_selection_path = Path(holdout_selection_path) if holdout_selection_path else None
        self._label_commitment_path = Path(label_commitment_path) if label_commitment_path else None
        self._holdout_labels_sha256 = holdout_labels_sha256
        self._output_dir = Path(output_dir)
        self._package_zip = Path(package_zip) if package_zip else None
        self._candidate_id = candidate_id
        self._device = device
        self._torch_dtype = torch_dtype
        self._temperature = temperature
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._max_structured_output_retries = max_structured_output_retries
        self._timeout_seconds = timeout_seconds
        self._repeat_count = repeat_count
        self._preflight_only = preflight_only
        self._custom_provider = custom_provider
        self._bypass_source_checksums = bypass_source_checksums

    def evaluate(self) -> dict[str, Any]:
        """Execute full holdout benchmark evaluation or preflight validation."""
        start_time = perf_counter()

        # 1. Source SHA & Commitment verification (fail-closed)
        sources_info = self._verify_canonical_source_checksums()

        # 2. Execution identity provenance
        exec_identity = self._build_execution_identity(sources_info)

        # 3. Load holdout targets with strict fail-closed validation
        arm_targets, claim_targets = self._load_holdout_targets()
        _LOGGER.info(
            "Loaded %d holdout claim targets across %d arm targets.",
            len(claim_targets),
            len(arm_targets),
        )

        # 4. Model-free V0 verifier replay verification
        v0_arm_results, v0_claim_preds, v0_replay_stats = self._replay_v0_verifier(arm_targets)

        # 5. Preflight Gate Check
        if self._preflight_only:
            _LOGGER.info("Holdout preflight validation requested. Skipping model execution.")
            total_duration = perf_counter() - start_time
            preflight_report = self._build_preflight_report(
                sources_info=sources_info,
                exec_identity=exec_identity,
                claim_targets=claim_targets,
                arm_targets=arm_targets,
                v0_replay_stats=v0_replay_stats,
                total_duration=total_duration,
            )
            self._write_reports(
                report=preflight_report,
                exec_identity=exec_identity,
                v0_claim_preds=v0_claim_preds,
                is_preflight=True,
            )
            return preflight_report

        # 6. Canonical Provenance Validation
        self._validate_canonical_provenance()

        # 7. Initialize provider
        raw_provider = self._init_v3_provider()
        self._validate_runtime_provider_identity(raw_provider)

        obs_provider = ObservationalChatModelProviderWrapper(raw_provider)
        verifier = StructuredSemanticCitationVerifierD3(
            provider=obs_provider,
            max_structured_output_retries=self._max_structured_output_retries,
        )

        # 8. Pass 1: Authoritative Primary Evaluation
        _LOGGER.info("Executing V2-D3 Holdout Pass 1 (Primary Evaluation)...")
        pass1_arm_results, pass1_claim_preds, pass1_telemetry = self._run_inference_pass(
            verifier=verifier,
            provider=obs_provider,
            arm_targets=arm_targets,
            pass_index=1,
        )

        # 9. Pass 2: Two-Pass Stability Evaluation
        _LOGGER.info("Executing V2-D3 Holdout Pass 2 (Stability Evaluation)...")
        pass2_arm_results, pass2_claim_preds, pass2_telemetry = self._run_inference_pass(
            verifier=verifier,
            provider=obs_provider,
            arm_targets=arm_targets,
            pass_index=2,
        )

        # 10. Stability Analysis
        stability_info = self._evaluate_stability(
            claim_targets=claim_targets,
            pass1_preds=pass1_claim_preds,
            pass2_preds=pass2_claim_preds,
        )

        # 11. Compute All Metrics (Pass 1 Authoritative)
        all_metrics = self._compute_all_metrics(
            claim_targets=claim_targets,
            arm_targets=arm_targets,
            v0_claim_preds=v0_claim_preds,
            v2_claim_preds=pass1_claim_preds,
            v0_arm_results=v0_arm_results,
            v2_arm_results=pass1_arm_results,
        )

        # 12. Build Final Reports
        total_duration = perf_counter() - start_time
        final_report, decision_report, stability_report = self._build_reports(
            sources_info=sources_info,
            exec_identity=exec_identity,
            claim_targets=claim_targets,
            arm_targets=arm_targets,
            stability_info=stability_info,
            all_metrics=all_metrics,
            pass1_telemetry=pass1_telemetry,
            pass2_telemetry=pass2_telemetry,
            total_duration=total_duration,
        )

        # 13. Write Output Artifacts
        self._write_reports(
            report=final_report,
            decision_report=decision_report,
            stability_report=stability_report,
            v0_claim_preds=v0_claim_preds,
            pass1_claim_preds=pass1_claim_preds,
            pass2_claim_preds=pass2_claim_preds,
            exec_identity=exec_identity,
            provider=obs_provider,
            is_preflight=False,
        )

        _LOGGER.info("V2-D3 Holdout Benchmark complete. Verdict: %s", final_report["verdict"])
        return final_report

    def _verify_canonical_source_checksums(self) -> dict[str, dict[str, Any]]:
        """Verify holdout input files and frozen commitment against pre-registered SHA signatures."""
        info: dict[str, dict[str, Any]] = {}

        if not self._holdout_packets_path.is_file():
            raise FileNotFoundError(f"Missing required holdout review packets at: {self._holdout_packets_path}")
        if not self._holdout_labels_path.is_file():
            raise FileNotFoundError(f"Missing required holdout labels file at: {self._holdout_labels_path}")

        actual_packets_sha = self._sha256_file(self._holdout_packets_path)
        actual_labels_sha = self._sha256_file(self._holdout_labels_path)
        actual_labels_size = self._holdout_labels_path.stat().st_size

        actual_selection_sha: str | None = None
        if self._holdout_selection_path:
            if not self._holdout_selection_path.is_file():
                raise FileNotFoundError(f"Missing holdout selection file at: {self._holdout_selection_path}")
            actual_selection_sha = self._sha256_file(self._holdout_selection_path)

        if not self._bypass_source_checksums:
            # 1. Check holdout review packets canonical SHA
            if actual_packets_sha != CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256:
                raise DataValidationError(
                    f"SHA-256 mismatch for holdout_review_packets. "
                    f"Expected: {CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256}, Got: {actual_packets_sha}"
                )

            # 2. Check holdout selection canonical SHA (Selection is mandatory for canonical execution)
            if not self._holdout_selection_path or not self._holdout_selection_path.is_file():
                raise DataValidationError(
                    "CANONICAL_HOLDOUT_EXECUTION_BLOCKED: Holdout selection file is mandatory for canonical execution."
                )
            if actual_selection_sha != CANONICAL_HOLDOUT_SELECTION_SHA256:
                raise DataValidationError(
                    f"SHA-256 mismatch for holdout_selection. "
                    f"Expected: {CANONICAL_HOLDOUT_SELECTION_SHA256}, Got: {actual_selection_sha}"
                )

            # 3. Verify Label Commitment (Mandatory for canonical execution)
            if not self._label_commitment_path or not self._label_commitment_path.is_file():
                raise DataValidationError(
                    "CANONICAL_HOLDOUT_EXECUTION_BLOCKED: Canonical holdout execution strictly requires --label-commitment."
                )
            commitment = json.loads(
                self._label_commitment_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )

            if commitment.get("artifact_type") != "verification_v2_holdout_label_commitment":
                raise DataValidationError("Invalid label commitment artifact_type")
            if commitment.get("review_status") != "frozen_human_reviewed":
                raise DataValidationError("Commitment review_status must be 'frozen_human_reviewed'")
            if commitment.get("reviewer_governance_status") != GOVERNANCE_STATUS_EXTERNALLY_REVIEWED:
                raise DataValidationError(
                    f"Commitment governance status mismatch: expected '{GOVERNANCE_STATUS_EXTERNALLY_REVIEWED}', "
                    f"got '{commitment.get('reviewer_governance_status')}'"
                )

            # Check cross-bindings
            if commitment.get("holdout_packets_sha256") != actual_packets_sha:
                raise DataValidationError("Commitment holdout_packets_sha256 does not match actual packets archive")
            if commitment.get("holdout_selection_sha256") != actual_selection_sha:
                raise DataValidationError("Commitment holdout_selection_sha256 does not match actual selection file")

            expected_labels_sha = commitment.get("labels_sha256")
            expected_labels_size = commitment.get("labels_size_bytes")
            if expected_labels_size and actual_labels_size != expected_labels_size:
                raise DataValidationError(
                    f"Holdout labels size mismatch: expected {expected_labels_size} bytes, got {actual_labels_size}"
                )

            if expected_labels_sha and actual_labels_sha != expected_labels_sha:
                raise DataValidationError(
                    f"SHA-256 mismatch for holdout_labels: expected {expected_labels_sha}, got {actual_labels_sha}"
                )

            info["holdout_review_packets"] = {
                "filename": self._holdout_packets_path.name,
                "sha256": actual_packets_sha,
                "size_bytes": self._holdout_packets_path.stat().st_size,
            }
            info["holdout_selection"] = {
                "filename": self._holdout_selection_path.name,
                "sha256": actual_selection_sha,
                "size_bytes": self._holdout_selection_path.stat().st_size,
            }
            info["holdout_labels"] = {
                "filename": self._holdout_labels_path.name,
                "sha256": actual_labels_sha,
                "size_bytes": actual_labels_size,
            }
            if self._label_commitment_path:
                info["label_commitment"] = {
                    "filename": self._label_commitment_path.name,
                    "sha256": self._sha256_file(self._label_commitment_path),
                }
        else:
            info["holdout_review_packets"] = {
                "filename": self._holdout_packets_path.name,
                "sha256": actual_packets_sha,
                "size_bytes": self._holdout_packets_path.stat().st_size,
            }
            if self._holdout_selection_path and self._holdout_selection_path.is_file():
                info["holdout_selection"] = {
                    "filename": self._holdout_selection_path.name,
                    "sha256": actual_selection_sha,
                    "size_bytes": self._holdout_selection_path.stat().st_size,
                }
            info["holdout_labels"] = {
                "filename": self._holdout_labels_path.name,
                "sha256": actual_labels_sha,
                "size_bytes": actual_labels_size,
            }

        return info

    def _build_execution_identity(
        self, sources_info: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Construct full immutable environment and execution provenance identity."""
        source_ver = getattr(legal_agentic_rag, "__version__", "unknown")
        try:
            installed_ver = importlib.metadata.version("legal-agentic-rag")
        except Exception:
            installed_ver = source_ver

        git_commit = self._get_git_commit()
        git_clean = self._is_git_worktree_clean()

        system_instruction_sha = sha256(
            STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest()
        schema_sha = sha256(
            json.dumps(
                D3StructuredClaimAssessmentDraft.model_json_schema(),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        d3_impl_path = Path(legal_agentic_rag.generation.structured_semantic_verifier_d3.__file__)
        d3_impl_sha = self._sha256_file(d3_impl_path) if d3_impl_path.is_file() else "unknown"

        # Verify frozen D3 source identity (fail-closed)
        if not self._bypass_source_checksums:
            if system_instruction_sha != CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256:
                raise DataValidationError(
                    f"Frozen D3 system instruction SHA mismatch. Expected: {CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256}, Got: {system_instruction_sha}"
                )
            if d3_impl_sha != CANONICAL_D3_IMPLEMENTATION_SHA256:
                raise DataValidationError(
                    f"Frozen D3 implementation file SHA mismatch. Expected: {CANONICAL_D3_IMPLEMENTATION_SHA256}, Got: {d3_impl_sha}"
                )
            if schema_sha != CANONICAL_D3_SCHEMA_SHA256:
                raise DataValidationError(
                    f"Frozen D3 schema SHA mismatch. Expected: {CANONICAL_D3_SCHEMA_SHA256}, Got: {schema_sha}"
                )

        provider_name = (
            self._custom_provider.provider_name if self._custom_provider else CANONICAL_V3_BACKEND
        )
        provider_version = (
            self._custom_provider.provider_version
            if self._custom_provider
            else CANONICAL_V3_PROVIDER_VERSION
        )
        model_name = (
            self._custom_provider.model_name if self._custom_provider else CANONICAL_V3_MODEL_NAME
        )
        model_revision = (
            self._custom_provider.model_revision
            if self._custom_provider
            else CANONICAL_V3_MODEL_REVISION
        )

        return {
            "schema_version": "1.0",
            "candidate_id": self._candidate_id,
            "package_version": source_ver,
            "installed_distribution_version": installed_ver,
            "git_commit": git_commit,
            "git_worktree_clean": git_clean,
            "repeat_count": self._repeat_count,
            "timestamp": datetime.now(UTC).isoformat(),
            "frozen_d3_source_identity_verified": True,
            "frozen_d3_canonical_evidence_sha256": CANONICAL_D3_EVIDENCE_ZIP_SHA256,
            "sources": sources_info,
            "provider": {
                "backend": provider_name,
                "provider_version": provider_version,
                "model_name": model_name,
                "model_revision": model_revision,
                "device": self._device,
                "torch_dtype": self._torch_dtype,
                "temperature": self._temperature,
                "max_input_tokens": self._max_input_tokens,
                "max_output_tokens": self._max_output_tokens,
                "max_structured_output_retries": self._max_structured_output_retries,
                "timeout_seconds": self._timeout_seconds,
            },
            "prompt_identities": {
                "d3_base_system_instruction_sha256": system_instruction_sha,
                "d3_schema_sha256": schema_sha,
            },
            "implementation_identities": {
                "structured_semantic_verifier_d3_sha256": d3_impl_sha,
            },
        }

    def _load_holdout_targets(
        self,
    ) -> tuple[list[BenchmarkArmTarget], list[BenchmarkClaimTarget]]:
        """Load holdout review packets and bind to reviewed human labels with strict set equality and metadata validation."""
        raw_text = self._holdout_labels_path.read_text(encoding="utf-8")
        raw_labels = json.loads(raw_text, object_pairs_hook=_reject_duplicate_json_keys)
        labels_by_q_arm_claim: dict[tuple[str, str, str], dict[str, Any]] = {}

        # Validate label artifact top-level metadata
        if isinstance(raw_labels, dict):
            art_type = raw_labels.get("artifact_type")
            if art_type != "verification_v2_holdout_reviewed_labels":
                raise DataValidationError(
                    f"Invalid holdout labels artifact_type: expected 'verification_v2_holdout_reviewed_labels', got '{art_type}'"
                )
            rev_status = raw_labels.get("review_status")
            if rev_status != "frozen_human_reviewed":
                raise DataValidationError(
                    f"Invalid holdout labels review_status: expected 'frozen_human_reviewed', got '{rev_status}'"
                )
            if not self._bypass_source_checksums:
                lbl_pkts_sha = raw_labels.get("holdout_packets_sha256")
                if lbl_pkts_sha != CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256:
                    raise DataValidationError(
                        f"Holdout labels holdout_packets_sha256 mismatch: expected {CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256}, got {lbl_pkts_sha}"
                    )
                lbl_sel_sha = raw_labels.get("holdout_selection_sha256")
                if lbl_sel_sha != CANONICAL_HOLDOUT_SELECTION_SHA256:
                    raise DataValidationError(
                        f"Holdout labels holdout_selection_sha256 mismatch: expected {CANONICAL_HOLDOUT_SELECTION_SHA256}, got {lbl_sel_sha}"
                    )

        if isinstance(raw_labels, dict) and "questions" in raw_labels:
            for qid, q_data in raw_labels["questions"].items():
                for arm_id, arm_data in q_data.get("arms", {}).items():
                    claims_dict = arm_data.get("claims", {})
                    if isinstance(claims_dict, dict):
                        for cid, c_data in claims_dict.items():
                            key = (str(qid), str(arm_id), str(cid))
                            if key in labels_by_q_arm_claim:
                                raise DataValidationError(f"Duplicate claim key {key} in holdout labels file")
                            labels_by_q_arm_claim[key] = c_data
                    elif isinstance(claims_dict, list):
                        for c_data in claims_dict:
                            cid = c_data.get("claim_id")
                            if not cid:
                                raise DataValidationError(f"Missing claim_id in labels for question {qid}")
                            key = (str(qid), str(arm_id), str(cid))
                            if key in labels_by_q_arm_claim:
                                raise DataValidationError(f"Duplicate claim key {key} in holdout labels file")
                            labels_by_q_arm_claim[key] = c_data
        elif isinstance(raw_labels, list):
            for item in raw_labels:
                qid = str(item.get("question_id", ""))
                arm_id = str(item.get("arm_id", ""))
                cid = str(item.get("claim_id", ""))
                if not qid or not arm_id or not cid:
                    raise DataValidationError(f"Invalid label item missing question_id/arm_id/claim_id: {item}")
                key = (qid, arm_id, cid)
                if key in labels_by_q_arm_claim:
                    raise DataValidationError(f"Duplicate claim key {key} in holdout labels file")
                labels_by_q_arm_claim[key] = item
        else:
            raise DataValidationError("Unrecognized labels artifact structure")

        # Validate total claims and class counts if present in metadata
        if isinstance(raw_labels, dict):
            expected_total = raw_labels.get("total_claims")
            if expected_total is not None and expected_total != len(labels_by_q_arm_claim):
                raise DataValidationError(
                    f"Label metadata total_claims mismatch: expected {expected_total}, got {len(labels_by_q_arm_claim)}"
                )
            class_counts_meta = raw_labels.get("class_counts")
            if isinstance(class_counts_meta, dict):
                actual_class_counts: Counter[str] = Counter()
                for c_data in labels_by_q_arm_claim.values():
                    raw_lbl = str(c_data.get("entailment_label", "")).strip().upper()
                    if raw_lbl in HumanEntailment.__members__:
                        actual_class_counts[raw_lbl] += 1
                for cls_name, exp_count in class_counts_meta.items():
                    if actual_class_counts[cls_name] != exp_count:
                        raise DataValidationError(
                            f"Label metadata class_counts mismatch for '{cls_name}': expected {exp_count}, got {actual_class_counts[cls_name]}"
                        )

        arm_targets: list[BenchmarkArmTarget] = []
        claim_targets: list[BenchmarkClaimTarget] = []
        packet_claims_found: set[tuple[str, str, str]] = set()

        with zipfile.ZipFile(self._holdout_packets_path, "r") as zf:
            json_members = [
                m for m in zf.namelist()
                if m.endswith(".json")
                and not m.startswith("__MACOSX")
                and ("holdout_packets/" in m or "packets/" in m)
            ]
            if not json_members:
                json_members = [
                    m for m in zf.namelist()
                    if m.endswith(".json") and not m.startswith("__MACOSX") and "/" not in m
                ]
            for member in sorted(json_members):
                pkt = json.loads(zf.read(member).decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
                qid = str(pkt.get("question_id") or Path(member).stem)
                stratum = pkt.get("stratum") or pkt.get("holdout_metadata", {}).get("stratum", "UNKNOWN")
                q_text = pkt.get("question_text") or pkt.get("question", "")

                pkt_arms = pkt.get("arms") or pkt.get("historical_arms", {})
                if not pkt_arms and "historical_arm" in pkt:
                    pkt_arms = {"PRIMARY": pkt["historical_arm"]}
                for arm_id, arm_data in pkt_arms.items():
                    hist_stop = arm_data.get("historical_stop_reason") or arm_data.get("agent_outcome", {}).get("stop_reason", "unknown")
                    hist_verif = (
                        arm_data.get("historical_verification", {})
                        or arm_data.get("rule_verifier_replay", {}).get("replay_result", {})
                    )

                    raw_resp = (
                        arm_data.get("historical_response", {})
                        or arm_data.get("answer_response", {})
                        or arm_data.get("response", {})
                    )
                    if isinstance(raw_resp, dict):
                        resp_dict = dict(raw_resp)
                        if "question" not in resp_dict:
                            resp_dict["question"] = q_text or "Holdout question"
                        if "retrieval_strategy" not in resp_dict:
                            resp_dict["retrieval_strategy"] = "hybrid"
                        if "warnings" not in resp_dict:
                            resp_dict["warnings"] = []
                        if "trace_id" not in resp_dict:
                            resp_dict["trace_id"] = "trace_holdout"
                        if "insufficient_evidence" not in resp_dict:
                            resp_dict["insufficient_evidence"] = False
                        valid_keys = {
                            "question",
                            "answer",
                            "citations",
                            "insufficient_evidence",
                            "warnings",
                            "retrieval_strategy",
                            "trace_id",
                            "metadata",
                        }
                        clean_resp = {k: v for k, v in resp_dict.items() if k in valid_keys}
                        ans_resp = AnswerResponse.model_validate(clean_resp)
                    else:
                        ans_resp = AnswerResponse.model_validate(raw_resp)

                    raw_ev = (
                        arm_data.get("selected_evidence", [])
                        or arm_data.get("reconstructed_evidence", [])
                        or arm_data.get("evidence_list", [])
                    )
                    ev_list = [Evidence.model_validate(item) for item in raw_ev]

                    arm_claims: list[BenchmarkClaimTarget] = []
                    raw_claims = (
                        arm_data.get("claims")
                        or arm_data.get("historical_verification", {}).get("claim_verifications", [])
                    )
                    for rc in raw_claims:
                        cid = rc.get("claim_id", "")
                        ctext = rc.get("claim_text", "")
                        if not cid:
                            raise DataValidationError(f"Missing claim_id in packet {member} arm {arm_id}")

                        key = (qid, str(arm_id), str(cid))
                        packet_claims_found.add(key)

                        if key not in labels_by_q_arm_claim:
                            raise DataValidationError(
                                f"HOLD_OUT_LABEL_MISSING: No human label found for claim key ({qid}, {arm_id}, {cid})."
                            )

                        lbl_data = labels_by_q_arm_claim[key]
                        lbl_raw = lbl_data.get("entailment_label")
                        if not lbl_raw:
                            raise DataValidationError(
                                f"HOLD_OUT_LABEL_MISSING: Missing entailment_label in label entry for claim ({qid}, {arm_id}, {cid})."
                            )

                        lbl_str = str(lbl_raw).strip().upper()
                        if lbl_str not in HumanEntailment.__members__:
                            raise DataValidationError(
                                f"HOLD_OUT_LABEL_INVALID: Invalid entailment_label '{lbl_raw}' for claim ({qid}, {arm_id}, {cid}). "
                                f"Must be one of {list(HumanEntailment.__members__.keys())}."
                            )
                        human_lbl = HumanEntailment(lbl_str)

                        ctext_sha = sha256(ctext.encode("utf-8")).hexdigest()
                        lbl_claim_sha = lbl_data.get("claim_text_sha256")
                        if not lbl_claim_sha:
                            raise DataValidationError(
                                f"HOLD_OUT_LABEL_MISSING_SHA: Missing claim_text_sha256 in label entry for claim ({qid}, {arm_id}, {cid})."
                            )
                        if lbl_claim_sha != ctext_sha:
                            raise DataValidationError(
                                f"HOLD_OUT_LABEL_MISMATCH: Claim text SHA mismatch for claim ({qid}, {arm_id}, {cid}). "
                                f"Expected packet SHA: {ctext_sha}, Got label SHA: {lbl_claim_sha}"
                            )

                        ctarget = BenchmarkClaimTarget(
                            slice_id="HOLDOUT",
                            question_id=qid,
                            arm_id=str(arm_id),
                            claim_id=cid,
                            claim_text=ctext,
                            claim_text_sha256=ctext_sha,
                            human_label=human_lbl,
                            error_tags=lbl_data.get("error_tags", []),
                            diagnostic_note=lbl_data.get("diagnostic_note"),
                            stratum=stratum,
                        )
                        arm_claims.append(ctarget)
                        claim_targets.append(ctarget)

                    arm_targets.append(
                        BenchmarkArmTarget(
                            slice_id="HOLDOUT",
                            question_id=qid,
                            arm_id=str(arm_id),
                            historical_stop_reason=hist_stop,
                            stratum=stratum,
                            question_text=q_text,
                            answer_response=ans_resp,
                            evidence_list=ev_list,
                            historical_verification=hist_verif,
                            claims=arm_claims,
                        )
                    )

        # Exact Set Equality Check
        label_keys = set(labels_by_q_arm_claim.keys())
        extra_in_labels = label_keys - packet_claims_found
        if extra_in_labels:
            raise DataValidationError(
                f"HOLD_OUT_EXTRA_LABELS: Found {len(extra_in_labels)} labels not present in packets archive: {extra_in_labels}"
            )

        if not claim_targets:
            raise DataValidationError("Holdout packets archive yielded zero claim targets.")

        return arm_targets, claim_targets

    def _replay_v0_verifier(
        self, arm_targets: list[BenchmarkArmTarget]
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Run pure rule-based V0 verifier baseline across holdout arms deterministically."""
        verifier = RuleBasedCitationVerifier()
        arm_results: dict[str, dict[str, Any]] = {}
        claim_preds: list[dict[str, Any]] = []
        total_arms = len(arm_targets)
        match_count = 0

        for arm in arm_targets:
            res = verifier.verify(
                response=arm.answer_response,
                evidence=arm.evidence_list,
            )
            key = f"{arm.question_id}:{arm.arm_id}"
            arm_results[key] = {
                "all_citations_supported": res.is_valid,
                "verified_citations_count": len(res.valid_citations),
                "unsupported_citations_count": len(res.invalid_citations),
                "execution_error": False,
            }

            hist_is_valid = arm.historical_verification.get("all_citations_supported")
            if hist_is_valid is None:
                hist_is_valid = arm.historical_verification.get("is_valid")
            if hist_is_valid is not None and hist_is_valid == res.is_valid:
                match_count += 1

            for claim in arm.claims:
                claim_preds.append({
                    "question_id": arm.question_id,
                    "arm_id": arm.arm_id,
                    "claim_id": claim.claim_id,
                    "human_label": claim.human_label,
                    "v0_binary_prediction": (
                        BinaryPrediction.ACCEPT if res.is_valid else BinaryPrediction.REJECT
                    ),
                    "v0_verified_citations": len(res.valid_citations),
                    "v0_unsupported_citations": len(res.invalid_citations),
                })

        replay_stats = {
            "total_arms_replayed": total_arms,
            "historical_replay_matches": match_count,
            "replay_match_rate": match_count / total_arms if total_arms > 0 else 1.0,
            "deterministic_execution": True,
        }
        return arm_results, claim_preds, replay_stats

    def _validate_canonical_provenance(self) -> None:
        """Validate exact candidate ID, runtime versions, parameter settings, and code signatures."""
        if self._candidate_id != CANONICAL_CANDIDATE_ID:
            raise DataValidationError(
                f"Candidate mismatch: expected '{CANONICAL_CANDIDATE_ID}', got '{self._candidate_id}'"
            )
        if self._repeat_count != CANONICAL_REPEAT_COUNT:
            raise DataValidationError(
                f"Repeat count mismatch: canonical execution requires repeat_count={CANONICAL_REPEAT_COUNT}, got {self._repeat_count}"
            )
        if self._max_input_tokens != CANONICAL_V3_MAX_INPUT_TOKENS:
            raise DataValidationError(
                f"Max input tokens mismatch: expected {CANONICAL_V3_MAX_INPUT_TOKENS}, got {self._max_input_tokens}"
            )
        if self._max_output_tokens != CANONICAL_V3_MAX_OUTPUT_TOKENS:
            raise DataValidationError(
                f"Max output tokens mismatch: expected {CANONICAL_V3_MAX_OUTPUT_TOKENS}, got {self._max_output_tokens}"
            )
        if self._max_structured_output_retries != CANONICAL_V3_MAX_STRUCTURED_RETRIES:
            raise DataValidationError(
                f"Max structured retries mismatch: expected {CANONICAL_V3_MAX_STRUCTURED_RETRIES}, got {self._max_structured_output_retries}"
            )
        if self._timeout_seconds != CANONICAL_V3_TIMEOUT_SECONDS:
            raise DataValidationError(
                f"Timeout seconds mismatch: expected {CANONICAL_V3_TIMEOUT_SECONDS}, got {self._timeout_seconds}"
            )

        if not self._bypass_source_checksums:
            if self._device != CANONICAL_V3_DEVICE:
                raise DataValidationError(
                    f"Device mismatch: canonical execution requires device='{CANONICAL_V3_DEVICE}', got '{self._device}'"
                )
            if self._torch_dtype != CANONICAL_V3_TORCH_DTYPE:
                raise DataValidationError(
                    f"Torch dtype mismatch: canonical execution requires torch_dtype='{CANONICAL_V3_TORCH_DTYPE}', got '{self._torch_dtype}'"
                )
            if self._temperature != CANONICAL_V3_TEMPERATURE:
                raise DataValidationError(
                    f"Temperature mismatch: canonical execution requires temperature={CANONICAL_V3_TEMPERATURE}, got {self._temperature}"
                )

            source_ver = getattr(legal_agentic_rag, "__version__", "unknown")
            if source_ver != CANONICAL_PACKAGE_VERSION:
                raise DataValidationError(
                    f"Package version mismatch: expected '{CANONICAL_PACKAGE_VERSION}', got '{source_ver}'"
                )

            try:
                installed_ver = importlib.metadata.version("legal-agentic-rag")
                if installed_ver != CANONICAL_PACKAGE_VERSION:
                    raise DataValidationError(
                        f"Installed distribution version mismatch: expected '{CANONICAL_PACKAGE_VERSION}', got '{installed_ver}'"
                    )
            except Exception as exc:
                raise DataValidationError(f"Could not verify installed distribution version: {exc}")

            if not self._is_git_worktree_clean():
                raise DataValidationError("Git worktree is dirty. Canonical execution requires a clean git worktree.")

            d3_impl_path = Path(legal_agentic_rag.generation.structured_semantic_verifier_d3.__file__)
            d3_impl_sha = self._sha256_file(d3_impl_path) if d3_impl_path.is_file() else "unknown"
            if d3_impl_sha != CANONICAL_D3_IMPLEMENTATION_SHA256:
                raise DataValidationError(
                    f"Frozen D3 implementation file SHA mismatch. Expected: {CANONICAL_D3_IMPLEMENTATION_SHA256}, Got: {d3_impl_sha}"
                )

            system_instruction_sha = sha256(
                STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION.encode("utf-8")
            ).hexdigest()
            if system_instruction_sha != CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256:
                raise DataValidationError(
                    f"Frozen D3 system instruction SHA mismatch. Expected: {CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256}, Got: {system_instruction_sha}"
                )

            schema_sha = sha256(
                json.dumps(
                    D3StructuredClaimAssessmentDraft.model_json_schema(),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if schema_sha != CANONICAL_D3_SCHEMA_SHA256:
                raise DataValidationError(
                    f"Frozen D3 schema SHA mismatch. Expected: {CANONICAL_D3_SCHEMA_SHA256}, Got: {schema_sha}"
                )

    def _init_v3_provider(self) -> ChatModelProvider:
        """Instantiate canonical or custom ChatModelProvider."""
        if self._custom_provider is not None:
            return self._custom_provider

        _LOGGER.info(
            "Initializing canonical Transformers provider (%s, %s, %s, %s)...",
            CANONICAL_V3_MODEL_NAME,
            CANONICAL_V3_MODEL_REVISION,
            self._device,
            self._torch_dtype,
        )
        return TransformersChatProvider(
            model_name=CANONICAL_V3_MODEL_NAME,
            revision=CANONICAL_V3_MODEL_REVISION,
            device=self._device,
            torch_dtype=self._torch_dtype,
            temperature=self._temperature,
            max_input_tokens=self._max_input_tokens,
            max_output_tokens=self._max_output_tokens,
            timeout_seconds=self._timeout_seconds,
        )

    def _validate_runtime_provider_identity(self, provider: ChatModelProvider) -> None:
        """Assert runtime provider model matches frozen candidate configuration."""
        if not self._bypass_source_checksums:
            if provider.model_name != CANONICAL_V3_MODEL_NAME:
                raise DataValidationError(
                    f"Runtime provider model_name mismatch: expected '{CANONICAL_V3_MODEL_NAME}', got '{provider.model_name}'"
                )
            if provider.model_revision != CANONICAL_V3_MODEL_REVISION:
                raise DataValidationError(
                    f"Runtime provider model_revision mismatch: expected '{CANONICAL_V3_MODEL_REVISION}', got '{provider.model_revision}'"
                )

    def _run_inference_pass(
        self,
        *,
        verifier: StructuredSemanticCitationVerifierD3,
        provider: ObservationalChatModelProviderWrapper,
        arm_targets: list[BenchmarkArmTarget],
        pass_index: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Run single complete inference pass over all arms with telemetry tracking."""
        pass_start_calls = provider.total_calls
        pass_start_errors = provider.total_errors
        structured_retries_total = 0
        semantic_exec_errors_total = 0

        arm_results: dict[str, Any] = {}
        claim_preds: list[dict[str, Any]] = []

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            try:
                cit_res, struct_res = verifier.verify_structured(
                    response=arm.answer_response,
                    evidence=arm.evidence_list,
                )
                arm_results[key] = {
                    "all_citations_supported": cit_res.is_valid,
                    "verified_citations_count": len(cit_res.valid_citations),
                    "unsupported_citations_count": len(cit_res.invalid_citations),
                    "execution_error": False,
                }
                struct_map = {a.claim_id: a for a in struct_res.assessments}
                telemetry_map = struct_res.claim_telemetries

                for claim in arm.claims:
                    assess = struct_map.get(claim.claim_id)
                    telemetry = telemetry_map.get(claim.claim_id)
                    retries_used = telemetry.retry_count if telemetry else 0
                    structured_retries_total += retries_used

                    is_exec_err = (
                        claim.claim_id in struct_res.execution_error_claims
                        or (telemetry is not None and telemetry.semantic_execution_error)
                    )

                    if is_exec_err:
                        semantic_exec_errors_total += 1
                        claim_preds.append({
                            "question_id": arm.question_id,
                            "arm_id": arm.arm_id,
                            "claim_id": claim.claim_id,
                            "human_label": claim.human_label,
                            "v2_d3_binary_prediction": BinaryPrediction.EXECUTION_ERROR,
                            "v2_d3_three_way_prediction": "EXECUTION_ERROR",
                            "evidence_relation": "EXECUTION_ERROR",
                            "diagnostic_flags": {},
                            "rejection_category": "EXECUTION_ERROR",
                            "retries_used": retries_used,
                            "execution_error": True,
                        })
                    elif assess is not None:
                        bin_pred = (
                            BinaryPrediction.ACCEPT
                            if assess.label == SemanticSupportLabel.SUPPORTED
                            else BinaryPrediction.REJECT
                        )
                        three_way = assess.label.value.upper()
                        diag_flags = {
                            "actor_mismatch": assess.actor_mismatch,
                            "condition_exception_mismatch": assess.condition_exception_mismatch,
                            "quantity_temporal_mismatch": assess.quantity_temporal_mismatch,
                            "negation_modality_mismatch": assess.negation_modality_mismatch,
                            "source_scope_mismatch": assess.source_scope_mismatch,
                        }

                        claim_preds.append({
                            "question_id": arm.question_id,
                            "arm_id": arm.arm_id,
                            "claim_id": claim.claim_id,
                            "human_label": claim.human_label,
                            "v2_d3_binary_prediction": bin_pred,
                            "v2_d3_three_way_prediction": three_way,
                            "evidence_relation": assess.relation.value,
                            "diagnostic_flags": diag_flags,
                            "rejection_category": telemetry.draft_rejection_categories[0] if (telemetry and telemetry.draft_rejection_categories) else "NONE",
                            "retries_used": retries_used,
                            "execution_error": False,
                        })
            except Exception as exc:
                err_type = type(exc).__name__
                err_sha = sha256(str(exc).encode("utf-8")).hexdigest()
                err_len = len(str(exc))
                _LOGGER.error(
                    "Execution error during inference pass %d on arm: %s (type=%s, sha=%s)",
                    pass_index,
                    key,
                    err_type,
                    err_sha[:16],
                )
                semantic_exec_errors_total += len(arm.claims)
                arm_results[key] = {
                    "all_citations_supported": False,
                    "verified_citations_count": 0,
                    "unsupported_citations_count": len(arm.claims),
                    "execution_error": True,
                    "error_type": err_type,
                    "error_sha256": err_sha,
                    "error_message_length": err_len,
                }
                for c in arm.claims:
                    claim_preds.append({
                        "question_id": arm.question_id,
                        "arm_id": arm.arm_id,
                        "claim_id": c.claim_id,
                        "human_label": c.human_label,
                        "v2_d3_binary_prediction": BinaryPrediction.EXECUTION_ERROR,
                        "v2_d3_three_way_prediction": "EXECUTION_ERROR",
                        "evidence_relation": "EXECUTION_ERROR",
                        "diagnostic_flags": {},
                        "rejection_category": "EXECUTION_ERROR",
                        "retries_used": 0,
                        "execution_error": True,
                        "error_type": err_type,
                        "error_sha256": err_sha,
                        "error_message_length": err_len,
                    })

        pass_telemetry = {
            "pass_index": pass_index,
            "provider_calls": provider.total_calls - pass_start_calls,
            "provider_invocation_errors": provider.total_errors - pass_start_errors,
            "total_structured_retries": structured_retries_total,
            "semantic_execution_errors": semantic_exec_errors_total,
        }
        return arm_results, claim_preds, pass_telemetry

    def _evaluate_stability(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        pass1_preds: list[dict[str, Any]],
        pass2_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute two-pass stability across semantic claim predictions with exact prediction-set equality."""
        expected_keys = {f"{t.question_id}:{t.arm_id}:{t.claim_id}" for t in claim_targets}

        # 1. Exact prediction-set equality for Pass 1
        p1_keys: list[str] = [f"{r.get('question_id')}:{r.get('arm_id')}:{r.get('claim_id')}" for r in pass1_preds]
        if len(p1_keys) != len(set(p1_keys)):
            raise DataValidationError("Duplicate prediction detected in Pass 1 prediction records")
        if set(p1_keys) != expected_keys:
            raise DataValidationError(
                f"Pass 1 prediction set mismatch: expected {len(expected_keys)} claims, got {len(p1_keys)}"
            )

        # 2. Exact prediction-set equality for Pass 2
        p2_keys: list[str] = [f"{r.get('question_id')}:{r.get('arm_id')}:{r.get('claim_id')}" for r in pass2_preds]
        if len(p2_keys) != len(set(p2_keys)):
            raise DataValidationError("Duplicate prediction detected in Pass 2 prediction records")
        if set(p2_keys) != expected_keys:
            raise DataValidationError(
                f"Pass 2 prediction set mismatch: expected {len(expected_keys)} claims, got {len(p2_keys)}"
            )

        p1_by_key = {f"{r['question_id']}:{r['arm_id']}:{r['claim_id']}": r for r in pass1_preds}
        p2_by_key = {f"{r['question_id']}:{r['arm_id']}:{r['claim_id']}": r for r in pass2_preds}

        total_claims = len(claim_targets)
        claims_with_two_valid = 0
        stable_count = 0
        unstable_count = 0
        p1_error_count = 0
        p2_error_count = 0
        error_any_pass_count = 0
        unstable_details: list[dict[str, Any]] = []

        valid_label_set = {
            HumanEntailment.SUPPORTED.value,
            HumanEntailment.CONTRADICTED.value,
            HumanEntailment.INSUFFICIENT.value,
        }

        for target in claim_targets:
            key = f"{target.question_id}:{target.arm_id}:{target.claim_id}"
            r1 = p1_by_key[key]
            r2 = p2_by_key[key]

            e1 = bool(r1.get("execution_error", False))
            e2 = bool(r2.get("execution_error", False))

            label1 = str(r1.get("v2_d3_three_way_prediction", "")).strip().upper()
            label2 = str(r2.get("v2_d3_three_way_prediction", "")).strip().upper()

            # Enforce valid semantic label for non-error predictions
            if not e1 and label1 not in valid_label_set:
                e1 = True
            if not e2 and label2 not in valid_label_set:
                e2 = True

            if e1:
                p1_error_count += 1
            if e2:
                p2_error_count += 1
            if e1 or e2:
                error_any_pass_count += 1
                continue

            claims_with_two_valid += 1

            if label1 == label2:
                stable_count += 1
            else:
                unstable_count += 1
                unstable_details.append({
                    "question_id": target.question_id,
                    "arm_id": target.arm_id,
                    "claim_id": target.claim_id,
                    "pass1_three_way": label1,
                    "pass2_three_way": label2,
                })

        return {
            "total_claims": total_claims,
            "claims_with_two_valid_semantic_labels": claims_with_two_valid,
            "stable_semantic_claim_count": stable_count,
            "unstable_semantic_claim_count": unstable_count,
            "pass1_execution_error_count": p1_error_count,
            "pass2_execution_error_count": p2_error_count,
            "execution_error_in_any_pass_count": error_any_pass_count,
            "unstable_claim_details": unstable_details,
        }

    def _compute_all_metrics(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        arm_targets: list[BenchmarkArmTarget],
        v0_claim_preds: list[dict[str, Any]],
        v2_claim_preds: list[dict[str, Any]],
        v0_arm_results: dict[str, Any],
        v2_arm_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute full suite of claim-binary, 3-way, and answer-level metrics."""
        binary_metrics = self._compute_claim_binary_metrics(claim_targets, v2_claim_preds)
        three_way_metrics = self._compute_three_way_metrics(claim_targets, v2_claim_preds)
        answer_metrics = self._compute_answer_level_metrics(arm_targets, v2_arm_results)
        v0_binary_metrics = self._compute_claim_binary_metrics(
            claim_targets,
            [
                {
                    "question_id": r["question_id"],
                    "arm_id": r["arm_id"],
                    "claim_id": r["claim_id"],
                    "v2_d3_binary_prediction": r["v0_binary_prediction"],
                    "execution_error": False,
                }
                for r in v0_claim_preds
            ],
        )

        return {
            "v0_claim_binary": v0_binary_metrics,
            "v2_d3_claim_binary": binary_metrics,
            "v2_d3_three_way": three_way_metrics,
            "v2_d3_answer_metrics": answer_metrics,
        }

    def _compute_claim_binary_metrics(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        claim_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute claim binary acceptance/rejection metrics with zero-denominator handling."""
        pred_map = {f"{r['question_id']}:{r['arm_id']}:{r['claim_id']}": r for r in claim_preds}
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        exec_errors = 0

        for target in claim_targets:
            key = f"{target.question_id}:{target.arm_id}:{target.claim_id}"
            pred = pred_map.get(key, {})
            if pred.get("execution_error", False):
                exec_errors += 1
                continue

            bin_pred = pred.get("v2_d3_binary_prediction")
            gold_supp = target.human_label == HumanEntailment.SUPPORTED

            if gold_supp:
                if bin_pred == BinaryPrediction.ACCEPT:
                    tp += 1
                else:
                    fn += 1
            else:
                if bin_pred == BinaryPrediction.REJECT:
                    tn += 1
                else:
                    fp += 1

        total = len(claim_targets)
        eval_total = tp + fp + tn + fn
        eval_supported = tp + fn
        eval_negative = tn + fp
        gold_supported = sum(1 for t in claim_targets if t.human_label == HumanEntailment.SUPPORTED)
        gold_negative = sum(1 for t in claim_targets if t.human_label in (HumanEntailment.CONTRADICTED, HumanEntailment.INSUFFICIENT))

        acc = (tp + tn) / eval_total if eval_total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else (None if eval_total == 0 else 0.0)
        supp_ret = tp / eval_supported if eval_supported > 0 else None
        neg_catch = tn / eval_negative if eval_negative > 0 else None

        f1 = (
            (2 * prec * supp_ret) / (prec + supp_ret)
            if (prec is not None and supp_ret is not None and (prec + supp_ret) > 0)
            else (None if (supp_ret is None or prec is None) else 0.0)
        )
        bal_acc = (
            (supp_ret + neg_catch) / 2.0
            if (supp_ret is not None and neg_catch is not None)
            else None
        )

        return {
            "total_claims": total,
            "evaluated_claims": eval_total,
            "execution_error_claims": exec_errors,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "gold_supported_claims": gold_supported,
            "gold_negative_claims": gold_negative,
            "evaluated_supported_claims": eval_supported,
            "evaluated_negative_claims": eval_negative,
            "accuracy": acc,
            "precision": prec,
            "supported_retention": supp_ret,
            "negative_catch": neg_catch,
            "f1": f1,
            "balanced_accuracy": bal_acc,
        }

    def _compute_three_way_metrics(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        claim_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute 3x3 multi-class confusion matrix and macro metrics."""
        pred_map = {f"{r['question_id']}:{r['arm_id']}:{r['claim_id']}": r for r in claim_preds}
        matrix: dict[str, dict[str, int]] = {
            gold: {pred: 0 for pred in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT")}
            for gold in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT")
        }
        exec_errors = 0

        for target in claim_targets:
            key = f"{target.question_id}:{target.arm_id}:{target.claim_id}"
            pred = pred_map.get(key, {})
            if pred.get("execution_error", False):
                exec_errors += 1
                continue

            gold_lbl = target.human_label.value.upper()
            pred_lbl = str(pred.get("v2_d3_three_way_prediction", "")).strip().upper()
            if gold_lbl in matrix and pred_lbl in matrix[gold_lbl]:
                matrix[gold_lbl][pred_lbl] += 1

        total = len(claim_targets)
        eval_total = total - exec_errors
        correct = sum(matrix[lbl][lbl] for lbl in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"))
        acc = correct / eval_total if eval_total > 0 else 0.0

        return {
            "total_claims": total,
            "evaluated_claims": eval_total,
            "execution_error_claims": exec_errors,
            "confusion_matrix": matrix,
            "accuracy": acc,
        }

    def _compute_answer_level_metrics(
        self,
        arm_targets: list[BenchmarkArmTarget],
        arm_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute answer correctness retention and catch rates with non-vacuous zero-denominator handling."""
        valid_retained = 0
        valid_rejected = 0
        invalid_caught = 0
        invalid_escaped = 0
        exec_errors = 0

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            res = arm_results.get(key, {})
            if res.get("execution_error", False):
                exec_errors += 1
                continue

            # Ground truth valid if 0 contradicted/insufficient claims
            gold_valid = all(c.human_label == HumanEntailment.SUPPORTED for c in arm.claims)
            pred_supported = res.get("all_citations_supported", False)

            if gold_valid:
                if pred_supported:
                    valid_retained += 1
                else:
                    valid_rejected += 1
            else:
                if not pred_supported:
                    invalid_caught += 1
                else:
                    invalid_escaped += 1

        total_arms = len(arm_targets)
        eval_arms = valid_retained + valid_rejected + invalid_caught + invalid_escaped
        gold_valid_count = sum(1 for a in arm_targets if all(c.human_label == HumanEntailment.SUPPORTED for c in a.claims))
        gold_invalid_count = total_arms - gold_valid_count
        eval_valid_count = valid_retained + valid_rejected
        eval_invalid_count = invalid_caught + invalid_escaped

        val_ret_rate = valid_retained / eval_valid_count if eval_valid_count > 0 else None
        inv_catch_rate = invalid_caught / eval_invalid_count if eval_invalid_count > 0 else None
        eval_acc = (valid_retained + invalid_caught) / eval_arms if eval_arms > 0 else 0.0
        full_acc = (valid_retained + invalid_caught) / total_arms if total_arms > 0 else 0.0

        return {
            "total_answers": total_arms,
            "evaluated_answers": eval_arms,
            "execution_error_answers": exec_errors,
            "gold_valid_answers_count": gold_valid_count,
            "gold_invalid_answers_count": gold_invalid_count,
            "evaluated_valid_answers_count": eval_valid_count,
            "evaluated_invalid_answers_count": eval_invalid_count,
            "valid_answers_retained": valid_retained,
            "valid_answers_rejected": valid_rejected,
            "invalid_answers_caught": invalid_caught,
            "invalid_answers_escaped": invalid_escaped,
            "valid_answer_retention_rate": val_ret_rate,
            "invalid_answer_catch_rate": inv_catch_rate,
            "evaluated_answer_accuracy": eval_acc,
            "full_denominator_answer_accuracy": full_acc,
        }

    def _build_reports(
        self,
        sources_info: dict[str, Any],
        exec_identity: dict[str, Any],
        claim_targets: list[BenchmarkClaimTarget],
        arm_targets: list[BenchmarkArmTarget],
        stability_info: dict[str, Any],
        all_metrics: dict[str, Any],
        pass1_telemetry: dict[str, Any],
        pass2_telemetry: dict[str, Any],
        total_duration: float,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Construct authoritative evaluation report, decision report, and stability report."""
        d3_binary = all_metrics["v2_d3_claim_binary"]
        d3_three_way = all_metrics["v2_d3_three_way"]
        d3_answer = all_metrics["v2_d3_answer_metrics"]

        # 1. Non-Vacuous Coverage Denominators Check
        supp_denom_valid = d3_binary["gold_supported_claims"] > 0
        neg_denom_valid = d3_binary["gold_negative_claims"] > 0
        val_ans_denom_valid = d3_answer["gold_valid_answers_count"] > 0
        inv_ans_denom_valid = d3_answer["gold_invalid_answers_count"] > 0
        coverage_sufficient = (
            supp_denom_valid
            and neg_denom_valid
            and val_ans_denom_valid
            and inv_ans_denom_valid
        )

        # 2. Mechanical Gates
        model_errors = pass1_telemetry["provider_invocation_errors"] + pass2_telemetry["provider_invocation_errors"]
        exec_errors = stability_info["execution_error_in_any_pass_count"]
        unstable_claims = stability_info["unstable_semantic_claim_count"]
        two_valid_labels = stability_info["claims_with_two_valid_semantic_labels"] == len(claim_targets)

        # Provider call reconciliation (2 calls per claim + retries across 2 passes)
        total_claims = len(claim_targets)
        total_retries = pass1_telemetry["total_structured_retries"] + pass2_telemetry["total_structured_retries"]
        expected_provider_calls = 2 * total_claims + total_retries
        actual_provider_calls = pass1_telemetry["provider_calls"] + pass2_telemetry["provider_calls"]
        provider_calls_reconciled = (actual_provider_calls == expected_provider_calls)

        mechanical_pass = (
            model_errors == 0
            and exec_errors == 0
            and unstable_claims == 0
            and two_valid_labels
            and provider_calls_reconciled
            and exec_identity.get("frozen_d3_source_identity_verified", False)
        )

        # 3. Quality Rate Gates (Requires coverage to be sufficient)
        supp_ret_val = d3_binary["supported_retention"]
        neg_catch_val = d3_binary["negative_catch"]
        val_ans_ret_val = d3_answer["valid_answer_retention_rate"]
        full_ans_acc_val = d3_answer["full_denominator_answer_accuracy"]
        claim_bin_acc_val = d3_binary["accuracy"]

        supp_ret_pass = supp_ret_val is not None and supp_ret_val >= GATE_MIN_SUPPORTED_RETENTION_RATE
        neg_catch_pass = neg_catch_val is not None and neg_catch_val >= GATE_MIN_NEGATIVE_CATCH_RATE
        val_ans_ret_pass = val_ans_ret_val is not None and val_ans_ret_val >= GATE_MIN_VALID_ANSWER_RETENTION_RATE
        full_ans_acc_pass = full_ans_acc_val is not None and full_ans_acc_val >= GATE_MIN_FULL_ANSWER_ACCURACY_RATE
        claim_bin_acc_pass = claim_bin_acc_val is not None and claim_bin_acc_val >= GATE_MIN_CLAIM_BINARY_ACCURACY_RATE

        quality_gates_pass = (
            coverage_sufficient
            and supp_ret_pass
            and neg_catch_pass
            and val_ans_ret_pass
            and full_ans_acc_pass
            and claim_bin_acc_pass
        )

        # 4. Verdict Determination
        if not coverage_sufficient:
            verdict = "V2_D3_HOLDOUT_COVERAGE_INSUFFICIENT"
            holdout_decision = "REJECT_V2_D3_PROMOTION"
            promotion_recommended = False
        elif not mechanical_pass:
            verdict = "V2_D3_HOLDOUT_EXECUTION_FAILURE"
            holdout_decision = "REJECT_V2_D3_PROMOTION"
            promotion_recommended = False
        elif quality_gates_pass:
            verdict = "V2_D3_HOLDOUT_PROMOTION_RECOMMENDED"
            holdout_decision = "PROMOTE_V2_D3_TO_PRODUCTION"
            promotion_recommended = True
        else:
            verdict = "V2_D3_HOLDOUT_PROMOTION_REJECTED"
            holdout_decision = "REJECT_V2_D3_PROMOTION"
            promotion_recommended = False

        coverage_info = {
            "coverage_sufficient": coverage_sufficient,
            "gold_supported_claims": d3_binary["gold_supported_claims"],
            "gold_negative_claims": d3_binary["gold_negative_claims"],
            "gold_valid_answers": d3_answer["gold_valid_answers_count"],
            "gold_invalid_answers": d3_answer["gold_invalid_answers_count"],
            "supported_claims_denominator_valid": supp_denom_valid,
            "negative_claims_denominator_valid": neg_denom_valid,
            "valid_answers_denominator_valid": val_ans_denom_valid,
            "invalid_answers_denominator_valid": inv_ans_denom_valid,
        }

        final_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_holdout_report",
            "candidate_id": self._candidate_id,
            "verdict": verdict,
            "timestamp": datetime.now(UTC).isoformat(),
            "duration_seconds": total_duration,
            "total_claims": len(claim_targets),
            "total_answers": len(arm_targets),
            "coverage": coverage_info,
            "telemetry": {
                "total_provider_calls": actual_provider_calls,
                "expected_provider_calls": expected_provider_calls,
                "provider_calls_reconciled": provider_calls_reconciled,
                "total_structured_retries": total_retries,
                "model_errors": model_errors,
                "provider_invocation_errors": model_errors,
                "pass1": pass1_telemetry,
                "pass2": pass2_telemetry,
            },
            "stability": stability_info,
            "metrics": all_metrics,
            "execution_identity": exec_identity,
        }

        decision_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_holdout_decision_report",
            "candidate_id": self._candidate_id,
            "holdout_evaluation_decision": holdout_decision,
            "promotion_recommended": promotion_recommended,
            "promotion_authorized": False,  # Fail-closed invariant
            "verdict": verdict,
            "timestamp": datetime.now(UTC).isoformat(),
            "coverage_sufficient": coverage_sufficient,
            "mechanical_pass": mechanical_pass,
            "quality_gates_pass": quality_gates_pass,
            "provider_calls_reconciled": provider_calls_reconciled,
            "gate_evaluations": {
                "supported_retention_pass": supp_ret_pass,
                "negative_catch_pass": neg_catch_pass,
                "valid_answer_retention_pass": val_ans_ret_pass,
                "full_answer_accuracy_pass": full_ans_acc_pass,
                "claim_binary_accuracy_pass": claim_bin_acc_pass,
            },
            "rate_thresholds": {
                "min_supported_retention_rate": GATE_MIN_SUPPORTED_RETENTION_RATE,
                "min_negative_catch_rate": GATE_MIN_NEGATIVE_CATCH_RATE,
                "min_valid_answer_retention_rate": GATE_MIN_VALID_ANSWER_RETENTION_RATE,
                "min_full_answer_accuracy_rate": GATE_MIN_FULL_ANSWER_ACCURACY_RATE,
                "min_claim_binary_accuracy_rate": GATE_MIN_CLAIM_BINARY_ACCURACY_RATE,
            },
            "pass1_actual_rates": {
                "supported_retention": supp_ret_val,
                "negative_catch": neg_catch_val,
                "valid_answer_retention": val_ans_ret_val,
                "full_denominator_answer_accuracy": full_ans_acc_val,
                "claim_binary_accuracy": claim_bin_acc_val,
            },
        }

        stability_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_holdout_stability_report",
            "candidate_id": self._candidate_id,
            "verdict": verdict,
            "timestamp": datetime.now(UTC).isoformat(),
            "stability": stability_info,
        }

        return final_report, decision_report, stability_report

    def _build_preflight_report(
        self,
        sources_info: dict[str, Any],
        exec_identity: dict[str, Any],
        claim_targets: list[BenchmarkClaimTarget],
        arm_targets: list[BenchmarkArmTarget],
        v0_replay_stats: dict[str, Any],
        total_duration: float,
    ) -> dict[str, Any]:
        """Construct model-free preflight gate report without inspecting sensitive content."""
        gold_supported = sum(1 for c in claim_targets if c.human_label == HumanEntailment.SUPPORTED)
        gold_negative = sum(1 for c in claim_targets if c.human_label in (HumanEntailment.CONTRADICTED, HumanEntailment.INSUFFICIENT))
        gold_valid_ans = sum(1 for a in arm_targets if all(c.human_label == HumanEntailment.SUPPORTED for c in a.claims))
        gold_invalid_ans = len(arm_targets) - gold_valid_ans

        coverage_sufficient = (
            gold_supported > 0 and gold_negative > 0 and gold_valid_ans > 0 and gold_invalid_ans > 0
        )

        return {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_holdout_preflight_report",
            "candidate_id": self._candidate_id,
            "verdict": "V2_D3_HOLDOUT_BENCHMARK_READY" if coverage_sufficient else "V2_D3_HOLDOUT_COVERAGE_INSUFFICIENT",
            "timestamp": datetime.now(UTC).isoformat(),
            "duration_seconds": total_duration,
            "total_claims": len(claim_targets),
            "total_answers": len(arm_targets),
            "coverage": {
                "coverage_sufficient": coverage_sufficient,
                "gold_supported_claims": gold_supported,
                "gold_negative_claims": gold_negative,
                "gold_valid_answers": gold_valid_ans,
                "gold_invalid_answers": gold_invalid_ans,
            },
            "v0_replay_stats": v0_replay_stats,
            "execution_identity": exec_identity,
            "preflight_status": {
                "sources_verified": True,
                "schema_validated": True,
                "v0_verifier_replayed": True,
                "model_execution_skipped": True,
                "ready_for_execution": coverage_sufficient,
            },
        }

    def _write_reports(
        self,
        report: dict[str, Any],
        decision_report: dict[str, Any] | None = None,
        stability_report: dict[str, Any] | None = None,
        v0_claim_preds: list[dict[str, Any]] | None = None,
        pass1_claim_preds: list[dict[str, Any]] | None = None,
        pass2_claim_preds: list[dict[str, Any]] | None = None,
        exec_identity: dict[str, Any] | None = None,
        provider: ObservationalChatModelProviderWrapper | None = None,
        is_preflight: bool = False,
    ) -> None:
        """Write evaluation artifacts to disk and package canonical evidence ZIP if requested."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self._output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        exec_dir = self._output_dir / "execution"
        exec_dir.mkdir(parents=True, exist_ok=True)
        telem_dir = self._output_dir / "telemetry"
        telem_dir.mkdir(parents=True, exist_ok=True)

        if is_preflight:
            (results_dir / "v2_d3_holdout_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return

        if exec_identity:
            (exec_dir / "v2_d3_holdout_source_identity.json").write_text(
                json.dumps(exec_identity, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        (results_dir / "v2_d3_holdout_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if decision_report:
            (results_dir / "v2_d3_holdout_decision_report.json").write_text(
                json.dumps(decision_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if stability_report:
            (results_dir / "v2_d3_holdout_stability_report.json").write_text(
                json.dumps(stability_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if v0_claim_preds:
            with (results_dir / "v0_claim_predictions.jsonl").open("w", encoding="utf-8") as f:
                for r in v0_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if pass1_claim_preds:
            with (results_dir / "v2_d3_holdout_claim_predictions_pass1.jsonl").open("w", encoding="utf-8") as f:
                for r in pass1_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if pass2_claim_preds:
            with (results_dir / "v2_d3_holdout_claim_predictions_pass2.jsonl").open("w", encoding="utf-8") as f:
                for r in pass2_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if pass1_claim_preds and pass2_claim_preds:
            p2_map = {f"{r['question_id']}:{r['arm_id']}:{r['claim_id']}": r for r in pass2_claim_preds}
            with (results_dir / "v2_d3_holdout_claim_comparisons.jsonl").open("w", encoding="utf-8") as f:
                for r1 in pass1_claim_preds:
                    key = f"{r1['question_id']}:{r1['arm_id']}:{r1['claim_id']}"
                    r2 = p2_map.get(key, {})
                    comp_record = {
                        "question_id": r1["question_id"],
                        "arm_id": r1["arm_id"],
                        "claim_id": r1["claim_id"],
                        "human_label": r1.get("human_label", "UNKNOWN"),
                        "pass1_binary": r1["v2_d3_binary_prediction"],
                        "pass2_binary": r2.get("v2_d3_binary_prediction"),
                        "pass1_three_way": r1["v2_d3_three_way_prediction"],
                        "pass2_three_way": r2.get("v2_d3_three_way_prediction"),
                        "pass1_relation": r1.get("evidence_relation"),
                        "pass2_relation": r2.get("evidence_relation"),
                        "is_stable": r1["v2_d3_three_way_prediction"] == r2.get("v2_d3_three_way_prediction"),
                    }
                    f.write(json.dumps(comp_record, ensure_ascii=False) + "\n")

        if provider:
            with (telem_dir / "provider_calls.jsonl").open("w", encoding="utf-8") as f:
                for call in provider.call_history:
                    f.write(json.dumps(call, ensure_ascii=False) + "\n")

        if self._package_zip:
            self._package_evidence_archive(self._package_zip)

    def _package_evidence_archive(self, zip_dst: Path) -> None:
        """Create deterministic ZIP package of all generated holdout evaluation artifacts."""
        zip_dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self._output_dir):
                for f in sorted(files):
                    fpath = Path(root) / f
                    arcname = fpath.relative_to(self._output_dir).as_posix()
                    zf.write(fpath, arcname=arcname)
        _LOGGER.info("Packaged holdout evidence archive at: %s", zip_dst)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _get_git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            return "unknown"

    @staticmethod
    def _is_git_worktree_clean() -> bool:
        try:
            status = subprocess.check_output(
                ["git", "status", "--short"], text=True
            ).strip()
            return len(status) == 0
        except Exception:
            return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2-D3 Structured Semantic Verifier Fresh Holdout Benchmark Evaluation Harness"
    )
    parser.add_argument(
        "--holdout-packets",
        type=Path,
        required=True,
        help="Path to verification-v2-holdout-review-packets-v1.zip",
    )
    parser.add_argument(
        "--holdout-labels",
        type=Path,
        required=True,
        help="Path to verification-v2-holdout-reviewed-labels-v1.json",
    )
    parser.add_argument(
        "--holdout-selection",
        type=Path,
        default=None,
        help="Path to verification-v2-holdout-selection-v1.json (Mandatory for canonical execution)",
    )
    parser.add_argument(
        "--label-commitment",
        type=Path,
        default=None,
        help="Path to frozen verification-v2-d3-holdout-label-commitment.json (Mandatory for canonical execution)",
    )
    parser.add_argument(
        "--holdout-labels-sha256",
        type=str,
        default=None,
        help="Explicit SHA-256 digest of holdout labels file (Alternative to commitment file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_outputs/v2_d3_holdout_output"),
        help="Directory to write holdout evaluation reports and predictions",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Optional path to package verification-v2-d3-holdout-evidence.zip",
    )
    parser.add_argument(
        "--candidate-id",
        type=str,
        default=CANONICAL_CANDIDATE_ID,
        help="Candidate identifier (must be V2-D3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CANONICAL_V3_DEVICE,
        help="PyTorch device (default: cuda)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=CANONICAL_V3_TORCH_DTYPE,
        help="PyTorch dtype (default: float16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=CANONICAL_V3_TEMPERATURE,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=CANONICAL_REPEAT_COUNT,
        help="Number of evaluation passes (default: 2)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run model-free preflight validation without LLM inference",
    )
    parser.add_argument(
        "--bypass-source-checksums",
        action="store_true",
        help="Bypass strict canonical source SHA verification (FOR TESTING ONLY)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=args.holdout_packets,
        holdout_labels_path=args.holdout_labels,
        holdout_selection_path=args.holdout_selection,
        label_commitment_path=args.label_commitment,
        holdout_labels_sha256=args.holdout_labels_sha256,
        output_dir=args.output_dir,
        package_zip=args.package_zip,
        candidate_id=args.candidate_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        temperature=args.temperature,
        repeat_count=args.repeat_count,
        preflight_only=args.preflight_only,
        bypass_source_checksums=args.bypass_source_checksums,
    )
    report = evaluator.evaluate()
    print("\n" + "=" * 70)
    print("        V2-D3 FRESH HOLDOUT EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Verdict:   {report.get('verdict')}")
    print(f"Candidate: {report.get('candidate_id')}")
    print(f"Duration:  {report.get('duration_seconds', 0.0):.2f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
