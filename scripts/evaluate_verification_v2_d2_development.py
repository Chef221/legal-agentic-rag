#!/usr/bin/env python3
"""V2-D2 Structured Semantic Verifier Development Benchmark Harness.

This script executes the controlled offline development benchmark for candidate V2-D2
over the frozen composite 38-claim human-annotated dataset:
- Slice A: 11 claims from suspicious forensic cases (B-FORENSIC-1A)
- Slice B: 27 claims from pre-registered positive-control candidates (B-FORENSIC-1C)
- Baseline V1 predictions loaded directly from canonical evidence archive:
  verification-semantic-benchmark-evidence.zip

Key differences from V2-D1:
- Per-Claim Model Invocations: Each claim is evaluated independently with its cited evidence only.
- D2 Vocabulary: ESTABLISHED, CONFLICT, NOT_ESTABLISHED, NOT_MATERIAL.
- Single-claim draft schema (no arrays, no model labels).
- Structured draft rejection telemetry and failure isolation.
- Development freeze criteria evaluated against canonical V1 baseline (23/38).
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
from legal_agentic_rag.configuration.online import (
    ClaimVerificationConfig,
    SemanticVerificationConfig,
)
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
import legal_agentic_rag.generation.structured_semantic_verifier
import legal_agentic_rag.generation.structured_semantic_verifier_d2
from legal_agentic_rag.generation.structured_semantic_verifier_d2 import (
    STRUCTURED_SEMANTIC_D2_SYSTEM_INSTRUCTION,
    D2EvidenceCoverageStatus,
    D2SemanticDimensionStatus,
    D2StructuredClaimAssessmentDraft,
    DraftRejectionCategory,
    StructuredClaimVerificationD2,
    StructuredSemanticCitationVerifierD2,
    StructuredSemanticVerificationResultD2,
    derive_claim_semantic_label_d2,
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

# Canonical Frozen Benchmark Checksums
CANONICAL_FORENSIC_REVIEW_ZIP_SHA256 = (
    "996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a"
)
CANONICAL_FORENSIC_LABELS_SHA256 = (
    "bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733"
)
CANONICAL_CONTROL_REVIEW_ZIP_SHA256 = (
    "cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4"
)
CANONICAL_CONTROL_LABELS_SHA256 = (
    "60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373"
)
CANONICAL_V1_EVIDENCE_ZIP_SHA256 = (
    "bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6"
)

# Canonical Pinned V2-D2 Model & Execution Parameters
CANONICAL_PACKAGE_VERSION = "0.50.7"
CANONICAL_CANDIDATE_ID = "V2-D2"
CANONICAL_V2_BACKEND = "transformers"
CANONICAL_V2_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CANONICAL_V2_MODEL_REVISION = "a1d308dfcc03e09da285d49d912439a655a571e8"
CANONICAL_V2_PROVIDER_VERSION = "4.47.1"
CANONICAL_V2_DEVICE = "cuda"
CANONICAL_V2_TORCH_DTYPE = "float16"
CANONICAL_V2_TEMPERATURE = 0.0
CANONICAL_V2_MAX_INPUT_TOKENS = 8192
CANONICAL_V2_MAX_OUTPUT_TOKENS = 512
CANONICAL_V2_MAX_STRUCTURED_RETRIES = 1
CANONICAL_V2_TIMEOUT_SECONDS = 180.0
CANONICAL_REPEAT_COUNT = 2


def sha256_file(path: Path) -> str:
    """Compute deterministic SHA-256 hex digest for a file."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute deterministic SHA-256 hex digest for UTF-8 text."""
    return sha256(text.encode("utf-8")).hexdigest()


def get_git_commit() -> str:
    """Retrieve immutable 40-character Git commit hash for execution provenance."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        if len(commit) == 40:
            return commit
    except Exception:
        pass
    return "unknown"


def is_git_worktree_clean() -> bool:
    """Check if Git working tree has no uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(res.stdout.strip()) == 0
    except Exception:
        return False


def get_runtime_environment() -> dict[str, Any]:
    """Capture precise Python runtime package versions and hardware state."""
    env: dict[str, Any] = {
        "python_version": sys.version,
    }
    for pkg in ("transformers", "torch", "accelerate", "pydantic"):
        try:
            env[f"{pkg}_version"] = importlib.metadata.version(pkg)
        except Exception:
            env[f"{pkg}_version"] = "unknown"

    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_device_count"] = torch.cuda.device_count()
        else:
            env["cuda_device_name"] = None
            env["cuda_device_count"] = 0
    except Exception:
        env["cuda_available"] = False
        env["cuda_version"] = None
        env["cuda_device_name"] = None
        env["cuda_device_count"] = 0

    return env


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
    stratum: str | None


@dataclass(frozen=True)
class BenchmarkArmTarget:
    slice_id: str
    question_id: str
    arm_id: str
    historical_stop_reason: str
    stratum: str | None
    question_text: str
    answer_response: AnswerResponse
    evidence_list: list[Evidence]
    historical_verification: dict[str, Any]
    claims: list[BenchmarkClaimTarget]


class ObservationalChatModelProviderWrapper(ChatModelProvider):
    """Transparent proxy that records all provider interactions, latencies, and token counts."""

    def __init__(self, provider: ChatModelProvider) -> None:
        self._inner = provider
        self.call_history: list[dict[str, Any]] = []

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
    def model_revision(self) -> str:
        return self._inner.model_revision

    @property
    def total_calls(self) -> int:
        return len(self.call_history)

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        call_idx = len(self.call_history)
        t0 = perf_counter()
        error_msg: str | None = None
        completion: str = ""
        success = False

        sys_sha = sha256_text(system_instruction)
        prompt_sha = sha256_text(user_prompt)

        try:
            completion = self._inner.complete(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            )
            success = True
            return completion
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency = (perf_counter() - t0) * 1000.0
            comp_sha = sha256_text(completion) if success else None
            self.call_history.append(
                {
                    "call_index": call_idx,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "system_instruction_sha256": sys_sha,
                    "user_prompt_sha256": prompt_sha,
                    "user_prompt_length": len(user_prompt),
                    "completion_sha256": comp_sha,
                    "completion_length": len(completion) if success else 0,
                    "latency_ms": round(latency, 2),
                    "call_succeeded": success,
                    "error": error_msg,
                }
            )


class V2D2DevelopmentBenchmarkEvaluator:
    """End-to-end evaluation harness for V2-D2 semantic verifier development benchmark."""

    def __init__(
        self,
        forensic_packets_path: Path,
        forensic_labels_path: Path,
        control_packets_path: Path,
        control_labels_path: Path,
        v1_evidence_path: Path,
        output_dir: Path,
        package_zip_path: Path | None = None,
        preflight_only: bool = False,
        candidate_id: str = CANONICAL_CANDIDATE_ID,
        device: str = CANONICAL_V2_DEVICE,
        repeat_count: int = CANONICAL_REPEAT_COUNT,
        max_structured_output_retries: int = CANONICAL_V2_MAX_STRUCTURED_RETRIES,
        custom_provider: ChatModelProvider | None = None,
    ) -> None:
        self._forensic_packets_path = forensic_packets_path
        self._forensic_labels_path = forensic_labels_path
        self._control_packets_path = control_packets_path
        self._control_labels_path = control_labels_path
        self._v1_evidence_path = v1_evidence_path
        self._output_dir = output_dir
        self._package_zip_path = package_zip_path
        self._preflight_only = preflight_only
        self._candidate_id = candidate_id
        self._device = device
        self._repeat_count = repeat_count
        self._max_structured_output_retries = max_structured_output_retries
        self._custom_provider = custom_provider

    def run(self) -> dict[str, Any]:
        """Execute full D2 benchmark or preflight validation."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "execution").mkdir(parents=True, exist_ok=True)
        (self._output_dir / "results").mkdir(parents=True, exist_ok=True)
        (self._output_dir / "telemetry").mkdir(parents=True, exist_ok=True)

        # 1. Validate All External Sources Fail-Closed
        sources_info = self._validate_sources()

        # Validate canonical candidate ID and package versions if running canonical mode
        if self._custom_provider is None:
            if self._candidate_id != CANONICAL_CANDIDATE_ID:
                raise DataValidationError(
                    f"INVALID_V2_DEVELOPMENT_PROVENANCE: Candidate ID mismatch: expected '{CANONICAL_CANDIDATE_ID}', got '{self._candidate_id}'"
                )
            self._validate_package_provenance()

        # 2. Load Datasets & Build Evaluation Targets
        arm_targets, all_claim_targets = self._load_and_bind_benchmark_targets(sources_info)

        # 3. Execute V0 Rule-Based Citation Verifier Replay
        v0_verifier = RuleBasedCitationVerifier()
        v0_arm_results, v0_claim_preds, v0_replay_stats = self._execute_v0_replay(
            arm_targets, v0_verifier
        )

        # 4. Load and Validate Canonical V1 Baseline Predictions
        v1_claim_preds = self._load_canonical_v1_predictions(all_claim_targets)

        # 5. Handle Preflight Mode
        if self._preflight_only:
            execution_identity = self._build_execution_identity(
                sources_info=sources_info,
                runtime_identity=None,
                v0_replay_stats=v0_replay_stats,
                repeat_count=1,
            )
            v0_binary_metrics = self._compute_binary_metrics(
                all_claim_targets, v0_claim_preds, pred_key="v0_binary_prediction"
            )
            v0_answer_metrics = self._compute_answer_level_metrics(arm_targets, v0_arm_results)
            v1_binary_metrics = self._compute_binary_metrics(
                all_claim_targets, v1_claim_preds, pred_key="v1_binary_prediction"
            )
            v1_three_way = self._compute_three_way_metrics(
                all_claim_targets, v1_claim_preds, label_key="v1_three_way_prediction"
            )
            v1_answer_metrics = self._compute_answer_level_metrics_from_preds(
                arm_targets, v1_claim_preds, pred_key="v1_three_way_prediction", supported_val="SUPPORTED"
            )

            report = {
                "schema_version": "1.0",
                "artifact_type": "v2_d2_development_benchmark_report",
                "candidate_id": self._candidate_id,
                "verdict": "V2_DEVELOPMENT_BENCHMARK_READY",
                "sources_info": sources_info,
                "total_claims": len(all_claim_targets),
                "v0_replay_stats": v0_replay_stats,
                "model_run_executed": False,
                "v0_baseline_metrics": {
                    "v0_claim_binary": v0_binary_metrics,
                    "v0_answer_level": v0_answer_metrics,
                },
                "v1_baseline_metrics": {
                    "v1_claim_binary": v1_binary_metrics,
                    "v1_three_way": v1_three_way,
                    "v1_answer_level": v1_answer_metrics,
                },
                "execution_identity": execution_identity,
            }
            self._write_outputs(
                execution_identity=execution_identity,
                report=report,
                v0_claim_preds=v0_claim_preds,
                v1_claim_preds=v1_claim_preds,
                v2_d2_pass1_preds=[],
                v2_d2_pass2_preds=[],
                comparisons=[],
                dimension_diagnostics={},
                decision_report=report,
                provider_calls=[],
            )
            return report

        # 6. Canonical Execution Config Validation (Fail-Closed)
        self._validate_canonical_config()

        # 7. Initialize V2-D2 Structured Provider & Verifier
        raw_provider = self._init_v2_provider()
        self._validate_runtime_provider_identity(raw_provider)

        obs_provider = ObservationalChatModelProviderWrapper(raw_provider)
        v2_d2_verifier = StructuredSemanticCitationVerifierD2(
            provider=obs_provider,
            max_structured_output_retries=self._max_structured_output_retries,
            rule_verifier=v0_verifier,
        )

        # 8. Execute Multi-Pass V2-D2 Verification (Pass 1 = Metrics, Pass 2 = Stability)
        v2_passes_claim_preds: list[list[dict[str, Any]]] = []
        v2_passes_arm_results: list[dict[str, CitationVerificationResult]] = []
        v2_passes_structured_results: list[dict[str, StructuredSemanticVerificationResultD2]] = []
        arm_observational_traces: list[dict[str, Any]] = []
        model_error_count = 0

        for pass_idx in range(1, self._repeat_count + 1):
            pass_claim_preds, pass_arm_res, pass_struct_res, err_count, pass_traces = (
                self._execute_v2_d2_pass(
                    arm_targets=arm_targets,
                    v2_d2_verifier=v2_d2_verifier,
                    provider=obs_provider,
                    pass_number=pass_idx,
                )
            )
            v2_passes_claim_preds.append(pass_claim_preds)
            v2_passes_arm_results.append(pass_arm_res)
            v2_passes_structured_results.append(pass_struct_res)
            arm_observational_traces.extend(pass_traces)
            model_error_count += err_count

        # 9. Evaluate Two-Pass Stability
        stability_info = self._evaluate_stability(v2_passes_claim_preds)

        # 10. Determine Canonical Verdict Precedence
        if model_error_count > 0:
            verdict = "V2_DEVELOPMENT_EXECUTION_ERROR"
        elif stability_info["unstable_claim_count"] > 0:
            verdict = "V2_DEVELOPMENT_LABEL_INSTABILITY"
        else:
            verdict = "V2_DEVELOPMENT_BENCHMARK_PASS"

        # 11. Compute All Binary, 3-Way, Paired vs V1, and Answer Metrics
        metrics_report = self._compute_all_metrics(
            claim_targets=all_claim_targets,
            arm_targets=arm_targets,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds=v1_claim_preds,
            v2_claim_preds=v2_passes_claim_preds[0],
            v0_arm_results=v0_arm_results,
            v2_arm_results=v2_passes_arm_results[0],
        )

        # 12. Compute Dimension-Level Diagnostics (Exact-count per claim, independent per tag)
        dimension_diagnostics = self._compute_dimension_diagnostics(
            all_claim_targets, v2_passes_claim_preds[0]
        )

        # 13. Build Full Benchmark Report & Decision Report
        runtime_identity = self._collect_runtime_identity(obs_provider)
        execution_identity = self._build_execution_identity(
            sources_info=sources_info,
            runtime_identity=runtime_identity,
            v0_replay_stats=v0_replay_stats,
            repeat_count=self._repeat_count,
        )

        report, decision_report, comparisons = self._build_reports(
            verdict=verdict,
            execution_identity=execution_identity,
            stability_info=stability_info,
            metrics_report=metrics_report,
            dimension_diagnostics=dimension_diagnostics,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds=v1_claim_preds,
            v2_pass1_preds=v2_passes_claim_preds[0],
            all_claim_targets=all_claim_targets,
            model_error_count=model_error_count,
            total_provider_calls=obs_provider.total_calls,
            structured_retry_count=sum(
                t.get("retry_count_for_arm", 0) for t in arm_observational_traces
            ),
        )

        # 14. Write Materialized Outputs
        self._write_outputs(
            execution_identity=execution_identity,
            report=report,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds=v1_claim_preds,
            v2_d2_pass1_preds=v2_passes_claim_preds[0],
            v2_d2_pass2_preds=v2_passes_claim_preds[1] if len(v2_passes_claim_preds) > 1 else [],
            comparisons=comparisons,
            dimension_diagnostics=dimension_diagnostics,
            decision_report=decision_report,
            provider_calls=obs_provider.call_history,
        )

        return report

    def _validate_sources(self) -> dict[str, Any]:
        """Assert existence, file types, and exact SHA-256 checksums of all 5 external sources."""
        sources = [
            (
                self._forensic_packets_path,
                CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
                ".zip",
                "forensic review packets",
            ),
            (
                self._forensic_labels_path,
                CANONICAL_FORENSIC_LABELS_SHA256,
                ".json",
                "forensic labels",
            ),
            (
                self._control_packets_path,
                CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
                ".zip",
                "control review packets",
            ),
            (
                self._control_labels_path,
                CANONICAL_CONTROL_LABELS_SHA256,
                ".json",
                "control labels",
            ),
            (
                self._v1_evidence_path,
                CANONICAL_V1_EVIDENCE_ZIP_SHA256,
                ".zip",
                "canonical V1 baseline evidence archive",
            ),
        ]

        sources_info: dict[str, Any] = {}
        for path, expected_sha, expected_ext, label in sources:
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label} at '{path}'")
            if path.suffix.lower() != expected_ext:
                raise DataValidationError(
                    f"Invalid extension for {label}: expected '{expected_ext}', got '{path.suffix}'"
                )
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise DataValidationError(
                    f"SHA-256 checksum mismatch for {label} at '{path}':\n"
                    f"  Expected: {expected_sha}\n"
                    f"  Observed: {actual_sha}"
                )
            sources_info[label.replace(" ", "_")] = {
                "filename": path.name,
                "sha256": actual_sha,
                "size_bytes": path.stat().st_size,
            }

        return sources_info

    def _unpack_zip(self, zip_path: Path, prefix: str) -> Path:
        """Extract a zip archive into a clean temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
        return temp_dir

    def _cleanup_temp(self, path: Path) -> None:
        """Remove a temporary directory recursively."""
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    def _load_and_bind_benchmark_targets(
        self, sources_info: dict[str, Any]
    ) -> tuple[list[BenchmarkArmTarget], list[BenchmarkClaimTarget]]:
        """Load packets and labels from both slices and bind each claim to exact text."""
        forensic_labels_data = json.loads(self._forensic_labels_path.read_text(encoding="utf-8"))
        control_labels_data = json.loads(self._control_labels_path.read_text(encoding="utf-8"))

        arm_targets: list[BenchmarkArmTarget] = []
        all_claim_targets: list[BenchmarkClaimTarget] = []

        # Process Slice A: Forensic
        f_unpack = self._unpack_zip(self._forensic_packets_path, "forensic_pkts_")
        try:
            f_pkts_dir = (
                f_unpack / "forensic_packets"
                if (f_unpack / "forensic_packets").is_dir()
                else f_unpack
            )
            for qid, q_data in forensic_labels_data.get("questions", {}).items():
                pkt_file = f_pkts_dir / f"{qid}.json"
                if not pkt_file.is_file():
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Forensic packet {qid}.json missing from {f_pkts_dir}"
                    )
                pkt = json.loads(pkt_file.read_text(encoding="utf-8"))

                pkt_arms = pkt.get("arms") or pkt.get("historical_arms", {})
                for arm_id, arm_label_data in q_data.get("arms", {}).items():
                    if not arm_label_data.get("claim_review_applicable"):
                        continue
                    arm_packet_data = pkt_arms.get(arm_id, {})
                    arm_target, claim_targets = self._construct_arm_target(
                        slice_id="suspicious_forensic",
                        qid=qid,
                        arm_id=arm_id,
                        arm_label_data=arm_label_data,
                        arm_packet_data=arm_packet_data,
                        question_text=pkt.get("question", ""),
                        stratum=None,
                    )
                    arm_targets.append(arm_target)
                    all_claim_targets.extend(claim_targets)
        finally:
            self._cleanup_temp(f_unpack)

        # Process Slice B: Control
        c_unpack = self._unpack_zip(self._control_packets_path, "control_pkts_")
        try:
            c_pkts_dir = (
                c_unpack / "positive_control_packets"
                if (c_unpack / "positive_control_packets").is_dir()
                else c_unpack
            )
            for qid, q_data in control_labels_data.get("questions", {}).items():
                pkt_file = c_pkts_dir / f"{qid}.json"
                if not pkt_file.is_file():
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Control packet {qid}.json missing from {c_pkts_dir}"
                    )
                pkt = json.loads(pkt_file.read_text(encoding="utf-8"))
                arm_packet_data = pkt.get("historical_arm", {})
                stratum = q_data.get("stratum")

                arm_target, claim_targets = self._construct_arm_target(
                    slice_id="positive_control",
                    qid=qid,
                    arm_id="PRIMARY",
                    arm_label_data=q_data,
                    arm_packet_data=arm_packet_data,
                    question_text=pkt.get("question", ""),
                    stratum=stratum,
                )
                arm_targets.append(arm_target)
                all_claim_targets.extend(claim_targets)
        finally:
            self._cleanup_temp(c_unpack)

        # Assert composite benchmark totals
        total_claims = len(all_claim_targets)
        label_counts = Counter(c.human_label.value for c in all_claim_targets)

        if total_claims != 38:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Expected 38 composite claims, got {total_claims}"
            )
        if label_counts[HumanEntailment.SUPPORTED.value] != 18:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Expected 18 SUPPORTED claims, got {label_counts[HumanEntailment.SUPPORTED.value]}"
            )
        if label_counts[HumanEntailment.CONTRADICTED.value] != 7:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Expected 7 CONTRADICTED claims, got {label_counts[HumanEntailment.CONTRADICTED.value]}"
            )
        if label_counts[HumanEntailment.INSUFFICIENT.value] != 13:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Expected 13 INSUFFICIENT claims, got {label_counts[HumanEntailment.INSUFFICIENT.value]}"
            )

        return arm_targets, all_claim_targets

    def _construct_arm_target(
        self,
        *,
        slice_id: str,
        qid: str,
        arm_id: str,
        arm_label_data: dict[str, Any],
        arm_packet_data: dict[str, Any],
        question_text: str,
        stratum: str | None,
    ) -> tuple[BenchmarkArmTarget, list[BenchmarkClaimTarget]]:
        """Reconstruct AnswerResponse, Evidence list, and claim targets for one arm."""
        raw_resp = (
            arm_packet_data.get("historical_response", {})
            or arm_packet_data.get("response", {})
        )
        resp_obj = AnswerResponse.model_validate(raw_resp)

        raw_ev = (
            arm_packet_data.get("selected_evidence", [])
            or arm_packet_data.get("reconstructed_evidence", [])
        )
        evidence_list = [Evidence.model_validate(item) for item in raw_ev]

        hist_verification = (
            arm_packet_data.get("historical_verification", {})
            or raw_resp.get("metadata", {}).get("citation_verification", {})
            or arm_packet_data.get("rule_verifier_replay", {}).get("replay_result", {})
        )
        raw_claims_by_id = {
            c["claim_id"]: c for c in hist_verification.get("claim_verifications", [])
        }

        claims_labels = arm_label_data.get("claims", {})
        claim_targets: list[BenchmarkClaimTarget] = []

        for cid in sorted(claims_labels.keys()):
            cl_data = claims_labels[cid]
            if cid not in raw_claims_by_id:
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Claim {cid} missing in raw packet for {qid} ({arm_id})"
                )

            raw_claim_text = raw_claims_by_id[cid].get("claim_text", "")
            raw_sha = sha256_text(raw_claim_text)
            label_sha = cl_data.get("claim_text_sha256")

            if raw_sha != label_sha:
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Claim text SHA mismatch for {qid} {arm_id} {cid}: raw={raw_sha}, label={label_sha}"
                )

            claim_targets.append(
                BenchmarkClaimTarget(
                    slice_id=slice_id,
                    question_id=qid,
                    arm_id=arm_id,
                    claim_id=cid,
                    claim_text=raw_claim_text,
                    claim_text_sha256=raw_sha,
                    human_label=HumanEntailment(cl_data.get("entailment_label")),
                    error_tags=cl_data.get("error_tags", []),
                    diagnostic_note=cl_data.get("diagnostic_note"),
                    stratum=stratum,
                )
            )

        arm_target = BenchmarkArmTarget(
            slice_id=slice_id,
            question_id=qid,
            arm_id=arm_id,
            historical_stop_reason=arm_label_data.get(
                "historical_stop_reason", "answer_verified"
            ),
            stratum=stratum,
            question_text=question_text,
            answer_response=resp_obj,
            evidence_list=evidence_list,
            historical_verification=hist_verification,
            claims=claim_targets,
        )
        return arm_target, claim_targets

    def _execute_v0_replay(
        self,
        arm_targets: list[BenchmarkArmTarget],
        verifier: RuleBasedCitationVerifier,
    ) -> tuple[dict[str, CitationVerificationResult], list[dict[str, Any]], dict[str, Any]]:
        """Replay V0 verifier and assert 100% exact fidelity against historical verification records."""
        arm_results: dict[str, CitationVerificationResult] = {}
        claim_predictions: list[dict[str, Any]] = []
        arm_passes = 0

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            res = verifier.verify(arm.answer_response, arm.evidence_list)
            arm_results[key] = res

            hist = arm.historical_verification

            # 1. Exact is_valid
            if res.is_valid != hist.get("is_valid"):
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay is_valid mismatch on {key}"
                )

            # 2. Exact claim_level_verification_performed
            hist_claim_level = hist.get("claim_level_verification_performed")
            if (
                hist_claim_level is not None
                and res.claim_level_verification_performed != hist_claim_level
            ):
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay claim_level_verification_performed mismatch on {key}"
                )

            # 3. Exact valid and invalid citation IDs and order
            hist_valid_cits = [c["evidence_id"] for c in hist.get("valid_citations", [])]
            res_valid_cits = [c.evidence_id for c in res.valid_citations]
            if res_valid_cits != hist_valid_cits:
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay valid_citations mismatch on {key}: res={res_valid_cits}, hist={hist_valid_cits}"
                )

            hist_invalid_cits = [c["evidence_id"] for c in hist.get("invalid_citations", [])]
            res_invalid_cits = [c.evidence_id for c in res.invalid_citations]
            if res_invalid_cits != hist_invalid_cits:
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay invalid_citations mismatch on {key}"
                )

            # 4. Exact claim_coverage_score within tolerance
            hist_coverage = hist.get("claim_coverage_score", 1.0)
            if not math.isclose(
                res.claim_coverage_score, hist_coverage, rel_tol=1e-5, abs_tol=1e-5
            ):
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay claim_coverage_score mismatch on {key}"
                )

            # 5. Exact claim_verifications count and order
            hist_claims = hist.get("claim_verifications", [])
            if len(res.claim_verifications) != len(hist_claims):
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay claim count mismatch on {key}: res={len(res.claim_verifications)}, hist={len(hist_claims)}"
                )

            for rc, hc in zip(res.claim_verifications, hist_claims, strict=True):
                if rc.claim_id != hc["claim_id"]:
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay claim_id mismatch on {key}"
                    )
                if rc.claim_text != hc["claim_text"]:
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay claim_text mismatch on {key} ({rc.claim_id})"
                    )
                if rc.evidence_ids != hc["evidence_ids"]:
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay evidence_ids mismatch on {key} ({rc.claim_id})"
                    )
                if rc.status.value != hc["status"]:
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay status mismatch on {key} ({rc.claim_id})"
                    )
                if rc.numeric_match != hc["numeric_match"]:
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay numeric_match mismatch on {key} ({rc.claim_id})"
                    )
                if rc.negation_match != hc["negation_match"]:
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay negation_match mismatch on {key} ({rc.claim_id})"
                    )
                if rc.errors != hc.get("errors", []):
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay claim errors mismatch on {key} ({rc.claim_id})"
                    )
                if not math.isclose(
                    rc.lexical_support_score,
                    hc.get("lexical_support_score", 0.0),
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ):
                    raise DataValidationError(
                        f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay lexical score mismatch on {key} ({rc.claim_id})"
                    )

            # 6. Exact top-level errors and warnings
            if res.errors != hist.get("errors", []):
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay top errors mismatch on {key}"
                )
            if res.warnings != hist.get("warnings", []):
                raise DataValidationError(
                    f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay top warnings mismatch on {key}"
                )

            arm_passes += 1

            # Populate V0 prediction records for evaluated claims
            res_claims_by_id = {c.claim_id: c for c in res.claim_verifications}
            for claim_target in arm.claims:
                cid = claim_target.claim_id
                rc = res_claims_by_id[cid]
                pred = (
                    BinaryPrediction.ACCEPT.value
                    if rc.status == ClaimSupportStatus.SUPPORTED
                    else BinaryPrediction.REJECT.value
                )
                is_correct = (
                    pred == BinaryPrediction.ACCEPT.value
                    and claim_target.human_label == HumanEntailment.SUPPORTED
                ) or (
                    pred == BinaryPrediction.REJECT.value
                    and claim_target.human_label != HumanEntailment.SUPPORTED
                )
                claim_predictions.append(
                    {
                        "slice_id": claim_target.slice_id,
                        "question_id": claim_target.question_id,
                        "arm_id": claim_target.arm_id,
                        "claim_id": cid,
                        "claim_text_sha256": claim_target.claim_text_sha256,
                        "stratum": claim_target.stratum,
                        "human_label": claim_target.human_label.value,
                        "v0_rule_status": rc.status.value,
                        "v0_binary_prediction": pred,
                        "is_correct": is_correct,
                        "v0_errors": rc.errors,
                    }
                )

        replay_stats = {
            "v0_replay_arm_passes": arm_passes,
            "v0_replay_arm_total": len(arm_targets),
            "v0_replay_100_percent_fidelity": arm_passes == len(arm_targets),
        }
        return arm_results, claim_predictions, replay_stats

    def _load_canonical_v1_predictions(
        self,
        claim_targets: list[BenchmarkClaimTarget],
    ) -> list[dict[str, Any]]:
        """Load and validate canonical V1 baseline predictions from evidence archive."""
        with zipfile.ZipFile(self._v1_evidence_path, "r") as zf:
            content = zf.read("results/v1_claim_predictions_pass1.jsonl").decode("utf-8")

        preds_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = (item["question_id"], item["arm_id"], item["claim_id"])
            preds_by_key[key] = item

        ordered_preds: list[dict[str, Any]] = []
        for target in claim_targets:
            key = (target.question_id, target.arm_id, target.claim_id)
            pred = preds_by_key.get(key)
            if not pred:
                raise DataValidationError(f"Missing canonical V1 prediction for {key}")
            is_correct = (
                pred.get("v1_binary_prediction") == BinaryPrediction.ACCEPT.value
                and target.human_label == HumanEntailment.SUPPORTED
            ) or (
                pred.get("v1_binary_prediction") == BinaryPrediction.REJECT.value
                and target.human_label != HumanEntailment.SUPPORTED
            )
            pred_copy = dict(pred)
            pred_copy["is_correct"] = is_correct
            ordered_preds.append(pred_copy)

        if len(ordered_preds) != 38:
            raise DataValidationError(f"Expected 38 V1 predictions, got {len(ordered_preds)}")

        return ordered_preds

    def _validate_package_provenance(self) -> None:
        """Assert that source and installed distribution package versions match canonical 0.50.7."""
        source_ver = getattr(legal_agentic_rag, "__version__", "unknown")
        try:
            dist_ver = importlib.metadata.version("legal-agentic-rag")
        except Exception as exc:
            raise DataValidationError(
                f"INVALID_V2_DEVELOPMENT_PROVENANCE: Failed to read installed distribution version: {exc}"
            ) from exc

        if (
            source_ver != CANONICAL_PACKAGE_VERSION
            or dist_ver != CANONICAL_PACKAGE_VERSION
            or source_ver != dist_ver
        ):
            raise DataValidationError(
                f"INVALID_V2_DEVELOPMENT_PROVENANCE: Package version mismatch: source='{source_ver}', installed='{dist_ver}', expected='{CANONICAL_PACKAGE_VERSION}'"
            )

    def _validate_canonical_config(self) -> None:
        """Assert that runtime configuration strictly matches canonical V2 parameters and clean Git state."""
        if self._custom_provider is not None:
            return
        if self._candidate_id != CANONICAL_CANDIDATE_ID:
            raise DataValidationError(
                f"INVALID_V2_DEVELOPMENT_PROVENANCE: Candidate ID mismatch: expected '{CANONICAL_CANDIDATE_ID}', got '{self._candidate_id}'"
            )
        self._validate_package_provenance()
        if not is_git_worktree_clean():
            raise DataValidationError(
                "INVALID_V2_DEVELOPMENT_PROVENANCE: Git worktree must be clean for canonical real execution"
            )
        if self._device != CANONICAL_V2_DEVICE:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Expected device '{CANONICAL_V2_DEVICE}', got '{self._device}'"
            )
        if self._repeat_count != CANONICAL_REPEAT_COUNT:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Expected repeat count {CANONICAL_REPEAT_COUNT}, got {self._repeat_count}"
            )
        if self._max_structured_output_retries != CANONICAL_V2_MAX_STRUCTURED_RETRIES:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Expected max retries {CANONICAL_V2_MAX_STRUCTURED_RETRIES}, got {self._max_structured_output_retries}"
            )

    def _init_v2_provider(self) -> ChatModelProvider:
        """Construct real TransformersChatProvider with canonical parameters."""
        if self._custom_provider is not None:
            return self._custom_provider

        cfg = SemanticVerificationConfig(
            backend=CANONICAL_V2_BACKEND,
            model_name=CANONICAL_V2_MODEL_NAME,
            model_revision=CANONICAL_V2_MODEL_REVISION,
            device=self._device,
            torch_dtype=CANONICAL_V2_TORCH_DTYPE,
            local_files_only=False,
            timeout_seconds=CANONICAL_V2_TIMEOUT_SECONDS,
            max_input_tokens=CANONICAL_V2_MAX_INPUT_TOKENS,
            max_output_tokens=CANONICAL_V2_MAX_OUTPUT_TOKENS,
            max_structured_output_retries=self._max_structured_output_retries,
        )
        generation_cfg = cfg.as_generation_config()

        if (
            generation_cfg.backend != CANONICAL_V2_BACKEND
            or generation_cfg.model_name != CANONICAL_V2_MODEL_NAME
            or generation_cfg.model_revision != CANONICAL_V2_MODEL_REVISION
            or generation_cfg.device != CANONICAL_V2_DEVICE
            or generation_cfg.torch_dtype != CANONICAL_V2_TORCH_DTYPE
            or generation_cfg.local_files_only is not False
            or generation_cfg.timeout_seconds != CANONICAL_V2_TIMEOUT_SECONDS
            or generation_cfg.max_input_tokens != CANONICAL_V2_MAX_INPUT_TOKENS
            or generation_cfg.max_output_tokens != CANONICAL_V2_MAX_OUTPUT_TOKENS
            or generation_cfg.max_structured_output_retries != CANONICAL_V2_MAX_STRUCTURED_RETRIES
            or generation_cfg.temperature != 0.0
        ):
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: GenerationConfig invariants failed: {generation_cfg}"
            )

        return TransformersChatProvider(generation_cfg)

    def _validate_runtime_provider_identity(self, provider: ChatModelProvider) -> None:
        """Validate that provider runtime identity strictly matches canonical constants."""
        if self._custom_provider is not None:
            return
        p_name = getattr(provider, "provider_name", "unknown")
        if p_name != CANONICAL_V2_BACKEND:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Provider name mismatch: expected '{CANONICAL_V2_BACKEND}', got '{p_name}'"
            )
        p_model = getattr(provider, "model_name", "unknown")
        if p_model != CANONICAL_V2_MODEL_NAME:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Model name mismatch: expected '{CANONICAL_V2_MODEL_NAME}', got '{p_model}'"
            )
        p_rev = getattr(provider, "model_revision", "unknown")
        if p_rev != CANONICAL_V2_MODEL_REVISION:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Model revision mismatch: expected '{CANONICAL_V2_MODEL_REVISION}', got '{p_rev}'"
            )
        p_ver = getattr(provider, "provider_version", "unknown")
        if not p_ver or p_ver == "unknown" or p_ver != CANONICAL_V2_PROVIDER_VERSION:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Provider version mismatch: expected '{CANONICAL_V2_PROVIDER_VERSION}', got '{p_ver}'"
            )

    def _execute_v2_d2_pass(
        self,
        arm_targets: list[BenchmarkArmTarget],
        v2_d2_verifier: StructuredSemanticCitationVerifierD2,
        provider: ObservationalChatModelProviderWrapper,
        pass_number: int,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, CitationVerificationResult],
        dict[str, StructuredSemanticVerificationResultD2],
        int,
        list[dict[str, Any]],
    ]:
        """Execute one complete pass of V2-D2 verification across all arms."""
        claim_preds: list[dict[str, Any]] = []
        arm_results_map: dict[str, CitationVerificationResult] = {}
        structured_results_map: dict[str, StructuredSemanticVerificationResultD2] = {}
        arm_traces: list[dict[str, Any]] = []
        model_error_count = 0

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            calls_before = provider.total_calls
            t0 = perf_counter()
            exec_error: str | None = None
            citation_res: CitationVerificationResult | None = None
            structured_res: StructuredSemanticVerificationResultD2 | None = None

            try:
                citation_res, structured_res = v2_d2_verifier.verify_structured(
                    arm.answer_response, arm.evidence_list
                )
            except Exception as exc:
                exec_error = str(exc)
                model_error_count += len(arm.claims)

            latency = (perf_counter() - t0) * 1000.0
            calls_delta = provider.total_calls - calls_before
            max_allowed_calls = len(arm.claims) * (1 + self._max_structured_output_retries)
            if calls_delta > max_allowed_calls:
                exec_error = f"EXCESSIVE_PROVIDER_CALLS: Expected at most {max_allowed_calls} calls, got {calls_delta}"
                model_error_count += 1

            arm_traces.append(
                {
                    "pass_number": pass_number,
                    "question_id": arm.question_id,
                    "arm_id": arm.arm_id,
                    "calls_for_arm": calls_delta,
                    "retry_count_for_arm": max(0, calls_delta - len(arm.claims)),
                    "latency_ms": round(latency, 2),
                    "execution_error": exec_error,
                }
            )

            if citation_res is not None:
                arm_results_map[key] = citation_res
            if structured_res is not None:
                structured_results_map[key] = structured_res
                # Count any per-claim execution errors that were isolated
                model_error_count += len(structured_res.execution_error_claims)

            # Build map of per-claim structured assessment
            struct_by_cid = {}
            if structured_res is not None:
                struct_by_cid = {a.claim_id: a for a in structured_res.assessments}

            for claim in arm.claims:
                has_claim_err = (
                    exec_error is not None
                    or citation_res is None
                    or structured_res is None
                    or (claim.claim_id in getattr(structured_res, "execution_error_claims", []))
                )

                if has_claim_err:
                    claim_preds.append(
                        {
                            "pass_number": pass_number,
                            "slice_id": claim.slice_id,
                            "question_id": claim.question_id,
                            "arm_id": claim.arm_id,
                            "claim_id": claim.claim_id,
                            "claim_text_sha256": claim.claim_text_sha256,
                            "stratum": claim.stratum,
                            "human_label": claim.human_label.value,
                            "error_tags": claim.error_tags,
                            "v2_d2_three_way_prediction": "EXECUTION_ERROR",
                            "v2_d2_binary_prediction": BinaryPrediction.EXECUTION_ERROR.value,
                            "is_correct": False,
                            "structured_assessment": None,
                        }
                    )
                else:
                    assessment = struct_by_cid[claim.claim_id]
                    pred_label = assessment.label.value.upper()
                    pred_binary = (
                        BinaryPrediction.ACCEPT.value
                        if pred_label == "SUPPORTED"
                        else BinaryPrediction.REJECT.value
                    )
                    is_correct = (
                        claim.human_label == HumanEntailment.SUPPORTED
                        and pred_binary == BinaryPrediction.ACCEPT.value
                    ) or (
                        claim.human_label != HumanEntailment.SUPPORTED
                        and pred_binary == BinaryPrediction.REJECT.value
                    )
                    telemetry_dict = (
                        assessment.telemetry.model_dump()
                        if assessment.telemetry is not None
                        else None
                    )
                    claim_preds.append(
                        {
                            "pass_number": pass_number,
                            "slice_id": claim.slice_id,
                            "question_id": claim.question_id,
                            "arm_id": claim.arm_id,
                            "claim_id": claim.claim_id,
                            "claim_text_sha256": claim.claim_text_sha256,
                            "stratum": claim.stratum,
                            "human_label": claim.human_label.value,
                            "error_tags": claim.error_tags,
                            "v2_d2_three_way_prediction": pred_label,
                            "v2_d2_binary_prediction": pred_binary,
                            "is_correct": is_correct,
                            "structured_assessment": {
                                "claim_id": claim.claim_id,
                                "actor_role": assessment.actor_role.value,
                                "action_object": assessment.action_object.value,
                                "condition_exception": assessment.condition_exception.value,
                                "quantity_temporal": assessment.quantity_temporal.value,
                                "negation_modality": assessment.negation_modality.value,
                                "source_article_scope": assessment.source_article_scope.value,
                                "evidence_coverage": assessment.evidence_coverage.value,
                                "telemetry": telemetry_dict,
                            },
                        }
                    )

        return claim_preds, arm_results_map, structured_results_map, model_error_count, arm_traces

    def _evaluate_stability(
        self, passes_preds: list[list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Compute two-pass label agreement and record any unstable claims."""
        if len(passes_preds) < 2:
            return {
                "label_stability_percentage": 100.0,
                "unstable_claim_count": 0,
                "unstable_claims": [],
            }

        pass1_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r["v2_d2_three_way_prediction"]
            for r in passes_preds[0]
        }
        pass2_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r["v2_d2_three_way_prediction"]
            for r in passes_preds[1]
        }

        unstable_claims: list[dict[str, Any]] = []
        matching_count = 0
        total_claims = len(pass1_map)

        for key, p1_val in pass1_map.items():
            p2_val = pass2_map.get(key)
            if p1_val == p2_val:
                matching_count += 1
            else:
                unstable_claims.append(
                    {
                        "question_id": key[0],
                        "arm_id": key[1],
                        "claim_id": key[2],
                        "pass1_prediction": p1_val,
                        "pass2_prediction": p2_val,
                    }
                )

        pct = (matching_count / total_claims * 100.0) if total_claims else 100.0
        return {
            "label_stability_percentage": round(pct, 2),
            "unstable_claim_count": len(unstable_claims),
            "unstable_claims": unstable_claims,
        }

    def _compute_all_metrics(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        arm_targets: list[BenchmarkArmTarget],
        v0_claim_preds: list[dict[str, Any]],
        v1_claim_preds: list[dict[str, Any]],
        v2_claim_preds: list[dict[str, Any]],
        v0_arm_results: dict[str, CitationVerificationResult],
        v2_arm_results: dict[str, CitationVerificationResult],
    ) -> dict[str, Any]:
        """Compute all binary, three-way, paired deltas, and answer-level metrics."""
        v0_binary = self._compute_binary_metrics(
            claim_targets, v0_claim_preds, pred_key="v0_binary_prediction"
        )
        v1_binary = self._compute_binary_metrics(
            claim_targets, v1_claim_preds, pred_key="v1_binary_prediction"
        )
        v2_binary = self._compute_binary_metrics(
            claim_targets, v2_claim_preds, pred_key="v2_d2_binary_prediction"
        )

        v1_three_way = self._compute_three_way_metrics(
            claim_targets, v1_claim_preds, label_key="v1_three_way_prediction"
        )
        v2_three_way = self._compute_three_way_metrics(
            claim_targets, v2_claim_preds, label_key="v2_d2_three_way_prediction"
        )

        paired_v1_vs_v2 = self._compute_paired_metrics(
            claim_targets, v1_claim_preds, v2_claim_preds
        )

        v0_answer = self._compute_answer_level_metrics(arm_targets, v0_arm_results)
        v1_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, v1_claim_preds, pred_key="v1_three_way_prediction", supported_val="SUPPORTED"
        )
        v2_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, v2_claim_preds, pred_key="v2_d2_three_way_prediction", supported_val="SUPPORTED"
        )

        answer_deltas = {
            "v2_vs_v1_valid_retention_delta": round(
                v2_answer["valid_answer_retention_rate"] - v1_answer["valid_answer_retention_rate"], 4
            ),
            "v2_vs_v1_invalid_catch_delta": round(
                v2_answer["invalid_answer_catch_rate"] - v1_answer["invalid_answer_catch_rate"], 4
            ),
            "v2_vs_v1_answer_accuracy_delta": round(
                v2_answer["answer_level_accuracy"] - v1_answer["answer_level_accuracy"], 4
            ),
        }

        return {
            "v0_claim_binary": v0_binary,
            "v1_claim_binary": v1_binary,
            "v2_d2_claim_binary": v2_binary,
            "v1_three_way": v1_three_way,
            "v2_d2_three_way": v2_three_way,
            "paired_v1_vs_v2_d2": paired_v1_vs_v2,
            "v0_answer_metrics": v0_answer,
            "v1_answer_metrics": v1_answer,
            "v2_d2_answer_metrics": v2_answer,
            "v2_d2_vs_v1_answer_deltas": answer_deltas,
        }

    def _compute_binary_metrics(
        self,
        targets: list[BenchmarkClaimTarget],
        preds: list[dict[str, Any]],
        pred_key: str,
    ) -> dict[str, Any]:
        """Compute binary classification metrics with explicit execution error handling."""
        pred_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r[pred_key]
            for r in preds
        }

        tp = fp = tn = fn = exec_errors = 0
        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            pred_val = pred_map.get(k, "REJECT")
            gold_pos = t.human_label == HumanEntailment.SUPPORTED

            if pred_val == BinaryPrediction.EXECUTION_ERROR.value:
                exec_errors += 1
            elif gold_pos:
                if pred_val == BinaryPrediction.ACCEPT.value:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_val == BinaryPrediction.REJECT.value:
                    tn += 1
                else:
                    fp += 1

        total_eval = tp + fp + tn + fn
        accuracy = (tp + tn) / total_eval if total_eval else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        supported_retention = tp / (tp + fn) if (tp + fn) else 0.0
        negative_catch = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (
            2 * (precision * supported_retention) / (precision + supported_retention)
            if (precision + supported_retention)
            else 0.0
        )
        balanced_acc = (supported_retention + negative_catch) / 2.0

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "execution_errors": exec_errors,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "supported_retention": round(supported_retention, 4),
            "negative_catch": round(negative_catch, 4),
            "f1": round(f1, 4),
            "balanced_accuracy": round(balanced_acc, 4),
        }

    def _compute_three_way_metrics(
        self,
        targets: list[BenchmarkClaimTarget],
        preds: list[dict[str, Any]],
        label_key: str,
    ) -> dict[str, Any]:
        """Compute exact multi-class macro and per-class metrics across SUPPORTED, CONTRADICTED, INSUFFICIENT."""
        pred_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): str(r.get(label_key, "INSUFFICIENT")).upper()
            for r in preds
        }

        classes = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
        matrix: dict[str, dict[str, int]] = {
            gold: {pred: 0 for pred in classes} for gold in classes
        }
        exec_errors = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            pred = pred_map.get(k, "INSUFFICIENT")
            gold = t.human_label.value

            if pred in classes and gold in classes:
                matrix[gold][pred] += 1
            else:
                exec_errors += 1

        per_class: dict[str, dict[str, Any]] = {}
        precisions = []
        recalls = []
        f1s = []

        total_correct = sum(matrix[c][c] for c in classes)
        total_eval = sum(sum(matrix[g].values()) for g in classes)

        for c in classes:
            tp = matrix[c][c]
            fp = sum(matrix[g][c] for g in classes if g != c)
            fn = sum(matrix[c][p] for p in classes if p != c)
            support = sum(matrix[c].values())

            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * (p * r) / (p + r) if (p + r) else 0.0

            per_class[c] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4),
                "support": support,
            }
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        acc = total_correct / total_eval if total_eval else 0.0
        macro_p = sum(precisions) / len(precisions)
        macro_r = sum(recalls) / len(recalls)
        macro_f = sum(f1s) / len(f1s)

        return {
            "confusion_matrix": matrix,
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f, 4),
            "per_class": per_class,
            "execution_errors": exec_errors,
        }

    def _compute_paired_metrics(
        self,
        targets: list[BenchmarkClaimTarget],
        v1_preds: list[dict[str, Any]],
        v2_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute paired claim-by-claim deltas between V1 and V2-D2."""
        v1_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): bool(r["is_correct"])
            for r in v1_preds
        }
        v2_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): (
                bool(r["is_correct"]),
                r.get("v2_d2_binary_prediction") == BinaryPrediction.EXECUTION_ERROR.value,
            )
            for r in v2_preds
        }

        both_correct = 0
        v1_only_correct = 0
        v2_only_correct = 0
        both_wrong = 0
        v2_fixes_count = 0
        v2_regressions_count = 0
        v2_exec_error_count = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            v1_c = v1_map.get(k, False)
            v2_c, v2_err = v2_map.get(k, (False, True))

            if v2_err:
                v2_exec_error_count += 1

            if v1_c and v2_c:
                both_correct += 1
            elif v1_c and not v2_c:
                v1_only_correct += 1
                v2_regressions_count += 1
            elif not v1_c and v2_c:
                v2_only_correct += 1
                v2_fixes_count += 1
            else:
                both_wrong += 1

        net_delta = v2_only_correct - v1_only_correct
        return {
            "both_correct": both_correct,
            "v1_only_correct": v1_only_correct,
            "v2_only_correct": v2_only_correct,
            "both_wrong": both_wrong,
            "net_correctness_delta": net_delta,
            "v2_fixes_count": v2_fixes_count,
            "v2_regressions_count": v2_regressions_count,
            "v2_execution_error_count": v2_exec_error_count,
        }

    def _compute_answer_level_metrics(
        self,
        arm_targets: list[BenchmarkArmTarget],
        arm_results: dict[str, CitationVerificationResult],
    ) -> dict[str, Any]:
        """Compute answer-level validity retention and catch rates."""
        valid_retained = 0
        invalid_caught = 0
        gold_valid_count = 0
        gold_invalid_count = 0
        correct_arms = 0
        total_arms = len(arm_targets)

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            gold_valid = all(c.human_label == HumanEntailment.SUPPORTED for c in arm.claims)
            pred_valid = arm_results[key].is_valid if key in arm_results else False

            if gold_valid:
                gold_valid_count += 1
                if pred_valid:
                    valid_retained += 1
                    correct_arms += 1
            else:
                gold_invalid_count += 1
                if not pred_valid:
                    invalid_caught += 1
                    correct_arms += 1

        valid_retention_rate = valid_retained / gold_valid_count if gold_valid_count else 0.0
        invalid_catch_rate = invalid_caught / gold_invalid_count if gold_invalid_count else 0.0
        accuracy = correct_arms / total_arms if total_arms else 0.0

        return {
            "total_answers": total_arms,
            "valid_ground_truth_answers": gold_valid_count,
            "valid_answers_retained": valid_retained,
            "valid_answer_retention_rate": round(valid_retention_rate, 4),
            "invalid_ground_truth_answers": gold_invalid_count,
            "invalid_answers_caught": invalid_caught,
            "invalid_answer_catch_rate": round(invalid_catch_rate, 4),
            "answer_level_accuracy": round(accuracy, 4),
            "execution_errors": 0,
        }

    def _compute_answer_level_metrics_from_preds(
        self,
        arm_targets: list[BenchmarkArmTarget],
        claim_preds: list[dict[str, Any]],
        pred_key: str,
        supported_val: str = "SUPPORTED",
    ) -> dict[str, Any]:
        """Compute answer-level metrics derived directly from individual claim predictions."""
        preds_by_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for p in claim_preds:
            preds_by_arm[(p["question_id"], p["arm_id"])].append(p)

        valid_retained = 0
        invalid_caught = 0
        gold_valid_count = 0
        gold_invalid_count = 0
        correct_arms = 0
        total_arms = len(arm_targets)
        exec_errors = 0

        for arm in arm_targets:
            key = (arm.question_id, arm.arm_id)
            c_preds = preds_by_arm.get(key, [])
            gold_valid = all(c.human_label == HumanEntailment.SUPPORTED for c in arm.claims)

            has_exec_err = any(
                p.get(pred_key) in ("EXECUTION_ERROR", BinaryPrediction.EXECUTION_ERROR.value)
                for p in c_preds
            )
            if has_exec_err:
                exec_errors += 1
                pred_valid = False
            else:
                pred_valid = bool(c_preds) and all(
                    p.get(pred_key) == supported_val for p in c_preds
                )

            if gold_valid:
                gold_valid_count += 1
                if pred_valid:
                    valid_retained += 1
                    correct_arms += 1
            else:
                gold_invalid_count += 1
                if not pred_valid:
                    invalid_caught += 1
                    correct_arms += 1

        valid_retention_rate = valid_retained / gold_valid_count if gold_valid_count else 0.0
        invalid_catch_rate = invalid_caught / gold_invalid_count if gold_invalid_count else 0.0
        accuracy = correct_arms / total_arms if total_arms else 0.0

        return {
            "total_answers": total_arms,
            "valid_ground_truth_answers": gold_valid_count,
            "valid_answers_retained": valid_retained,
            "valid_answer_retention_rate": round(valid_retention_rate, 4),
            "invalid_ground_truth_answers": gold_invalid_count,
            "invalid_answers_caught": invalid_caught,
            "invalid_answer_catch_rate": round(invalid_catch_rate, 4),
            "answer_level_accuracy": round(accuracy, 4),
            "execution_errors": exec_errors,
        }

    def _compute_dimension_diagnostics(
        self,
        targets: list[BenchmarkClaimTarget],
        v2_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute categorical dimension distribution globally and per error tag without double counting."""
        pred_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r
            for r in v2_preds
        }

        global_counts: dict[str, Counter[str]] = {
            "actor_role": Counter(),
            "action_object": Counter(),
            "condition_exception": Counter(),
            "quantity_temporal": Counter(),
            "negation_modality": Counter(),
            "source_article_scope": Counter(),
            "evidence_coverage": Counter(),
        }

        tag_breakdown: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: {
                "actor_role": Counter(),
                "action_object": Counter(),
                "condition_exception": Counter(),
                "quantity_temporal": Counter(),
                "negation_modality": Counter(),
                "source_article_scope": Counter(),
                "evidence_coverage": Counter(),
            }
        )

        rejection_categories = Counter()
        total_retries = 0
        claims_with_exec_error = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            pred = pred_map.get(k)
            if not pred or not pred.get("structured_assessment"):
                claims_with_exec_error += 1
                continue

            sa = pred["structured_assessment"]
            telemetry = sa.get("telemetry") or {}
            total_retries += telemetry.get("retry_count", 0)
            for cat in telemetry.get("draft_rejection_categories", []):
                rejection_categories[cat] += 1
            if telemetry.get("semantic_execution_error"):
                claims_with_exec_error += 1

            for dim in (
                "actor_role",
                "action_object",
                "condition_exception",
                "quantity_temporal",
                "negation_modality",
                "source_article_scope",
                "evidence_coverage",
            ):
                val = str(sa.get(dim, "NONE"))
                global_counts[dim][val] += 1
                for tag in t.error_tags:
                    tag_breakdown[tag][dim][val] += 1

        return {
            "global_dimension_counts": {dim: dict(cnt) for dim, cnt in global_counts.items()},
            "per_error_tag_dimension_breakdown": {
                tag: {dim: dict(cnt) for dim, cnt in dims.items()}
                for tag, dims in tag_breakdown.items()
            },
            "rejection_telemetry_summary": {
                "rejection_category_counts": dict(rejection_categories),
                "total_retries": total_retries,
                "claims_with_execution_error": claims_with_exec_error,
            },
        }

    def _collect_runtime_identity(
        self,
        provider: ObservationalChatModelProviderWrapper,
    ) -> dict[str, Any]:
        """Collect provider identity metadata."""
        return {
            "provider_name": provider.provider_name,
            "provider_version": provider.provider_version,
            "model_name": provider.model_name,
            "model_revision": provider.model_revision,
        }

    def _build_execution_identity(
        self,
        sources_info: dict[str, Any],
        runtime_identity: dict[str, Any] | None,
        v0_replay_stats: dict[str, Any],
        repeat_count: int,
    ) -> dict[str, Any]:
        """Build full candidate execution identity metadata block."""
        source_pkg_ver = getattr(legal_agentic_rag, "__version__", "unknown")
        try:
            installed_pkg_ver = importlib.metadata.version("legal-agentic-rag")
        except Exception:
            installed_pkg_ver = "unknown"

        d2_verifier_path = Path(
            legal_agentic_rag.generation.structured_semantic_verifier_d2.__file__
        )
        d1_verifier_path = Path(
            legal_agentic_rag.generation.structured_semantic_verifier.__file__
        )
        harness_path = Path(__file__)

        json_schema = D2StructuredClaimAssessmentDraft.model_json_schema()
        schema_sha = sha256_text(json.dumps(json_schema, sort_keys=True))

        return {
            "schema_version": "1.0",
            "artifact_type": "v2_d2_development_execution_identity",
            "candidate_id": self._candidate_id,
            "execution_git_commit": get_git_commit(),
            "git_worktree_clean": is_git_worktree_clean(),
            "package_version": source_pkg_ver,
            "source_package_version": source_pkg_ver,
            "installed_distribution_version": installed_pkg_ver,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "sources": sources_info,
            "provider": runtime_identity,
            "model_inference_config": {
                "backend": CANONICAL_V2_BACKEND,
                "device": self._device,
                "torch_dtype": CANONICAL_V2_TORCH_DTYPE,
                "temperature": CANONICAL_V2_TEMPERATURE,
                "local_files_only": False,
                "max_input_tokens": CANONICAL_V2_MAX_INPUT_TOKENS,
                "max_output_tokens": CANONICAL_V2_MAX_OUTPUT_TOKENS,
                "max_structured_output_retries": self._max_structured_output_retries,
                "timeout_seconds": CANONICAL_V2_TIMEOUT_SECONDS,
                "repeat_count": repeat_count,
            },
            "runtime_environment": get_runtime_environment(),
            "implementation_identities": {
                "structured_semantic_verifier_d2_sha256": sha256_file(d2_verifier_path),
                "evaluate_verification_v2_d2_development_sha256": sha256_file(harness_path),
                "historical_d1_verifier_sha256": sha256_file(d1_verifier_path),
            },
            "prompt_identity": {
                "system_instruction_sha256": sha256_text(STRUCTURED_SEMANTIC_D2_SYSTEM_INSTRUCTION),
            },
            "schema_identity": {
                "structured_verification_json_schema_sha256": schema_sha,
            },
            "v0_replay_stats": v0_replay_stats,
            "repeat_count": repeat_count,
        }

    def _build_reports(
        self,
        verdict: str,
        execution_identity: dict[str, Any],
        stability_info: dict[str, Any],
        metrics_report: dict[str, Any],
        dimension_diagnostics: dict[str, Any],
        v0_claim_preds: list[dict[str, Any]],
        v1_claim_preds: list[dict[str, Any]],
        v2_pass1_preds: list[dict[str, Any]],
        all_claim_targets: list[BenchmarkClaimTarget],
        model_error_count: int,
        total_provider_calls: int,
        structured_retry_count: int,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        """Build canonical benchmark report, decision report, and claim-level comparisons."""
        v1_binary = metrics_report["v1_claim_binary"]
        v2_binary = metrics_report["v2_d2_claim_binary"]
        paired = metrics_report["paired_v1_vs_v2_d2"]

        v2_correct = v2_binary["tp"] + v2_binary["tn"]
        v1_correct = v1_binary["tp"] + v1_binary["tn"]

        # Strict Freeze Eligibility Criteria:
        # - model_errors == 0
        # - unstable_claims == 0
        # - v2_correct > 23 (canonical V1)
        # - v2_negative_caught > 7 (canonical V1)
        # - v2_supported_retained >= 16 (canonical V1)
        # - paired net correctness delta > 0
        freeze_eligible = (
            model_error_count == 0
            and stability_info["unstable_claim_count"] == 0
            and v2_correct > 23
            and v2_binary["tn"] > 7
            and v2_binary["tp"] >= 16
            and paired["net_correctness_delta"] > 0
        )

        decision = "CANDIDATE_FREEZE_ELIGIBLE" if freeze_eligible else "KEEP_ITERATING"

        report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d2_development_benchmark_report",
            "candidate_id": self._candidate_id,
            "verdict": verdict,
            "execution_identity": execution_identity,
            "stability": stability_info,
            "telemetry": {
                "total_provider_calls": total_provider_calls,
                "structured_output_retries": structured_retry_count,
                "model_errors": model_error_count,
            },
            "metrics": metrics_report,
            "dimension_diagnostics": dimension_diagnostics,
        }

        decision_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d2_development_decision_report",
            "candidate_id": self._candidate_id,
            "execution_git_commit": execution_identity["execution_git_commit"],
            "verdict": verdict,
            "development_evaluation_decision": decision,
            "promotion_authorized": False,  # INVARIANT: Always False for development evaluations
            "model_error_count": model_error_count,
            "structured_retry_count": structured_retry_count,
            "stability": stability_info,
            "v1_claim_binary": v1_binary,
            "v2_d2_claim_binary": v2_binary,
            "paired_v1_vs_v2_d2": paired,
            "decision_criteria_audit": {
                "zero_model_errors": model_error_count == 0,
                "zero_unstable_claims": stability_info["unstable_claim_count"] == 0,
                "d2_correct_exceeds_v1": v2_correct > 23,
                "d2_negative_catch_exceeds_v1": v2_binary["tn"] > 7,
                "d2_supported_retention_meets_v1": v2_binary["tp"] >= 16,
                "positive_paired_net_correctness": paired["net_correctness_delta"] > 0,
            },
        }

        # Build claim-level comparison records
        v0_map = {(r["question_id"], r["arm_id"], r["claim_id"]): r for r in v0_claim_preds}
        v1_map = {(r["question_id"], r["arm_id"], r["claim_id"]): r for r in v1_claim_preds}
        v2_map = {(r["question_id"], r["arm_id"], r["claim_id"]): r for r in v2_pass1_preds}

        comparisons: list[dict[str, Any]] = []
        for t in all_claim_targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            v0_r = v0_map.get(k, {})
            v1_r = v1_map.get(k, {})
            v2_r = v2_map.get(k, {})

            comparisons.append(
                {
                    "slice_id": t.slice_id,
                    "question_id": t.question_id,
                    "arm_id": t.arm_id,
                    "claim_id": t.claim_id,
                    "claim_text_sha256": t.claim_text_sha256,
                    "stratum": t.stratum,
                    "human_label": t.human_label.value,
                    "error_tags": t.error_tags,
                    "v0_prediction": v0_r.get("v0_binary_prediction"),
                    "v0_is_correct": v0_r.get("is_correct"),
                    "v1_three_way": v1_r.get("v1_three_way_prediction"),
                    "v1_prediction": v1_r.get("v1_binary_prediction"),
                    "v1_is_correct": v1_r.get("is_correct"),
                    "v2_d2_three_way": v2_r.get("v2_d2_three_way_prediction"),
                    "v2_d2_prediction": v2_r.get("v2_d2_binary_prediction"),
                    "v2_d2_is_correct": v2_r.get("is_correct"),
                    "v2_d2_structured_assessment": v2_r.get("structured_assessment"),
                }
            )

        return report, decision_report, comparisons

    def _write_outputs(
        self,
        execution_identity: dict[str, Any],
        report: dict[str, Any],
        v0_claim_preds: list[dict[str, Any]],
        v1_claim_preds: list[dict[str, Any]],
        v2_d2_pass1_preds: list[dict[str, Any]],
        v2_d2_pass2_preds: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        dimension_diagnostics: dict[str, Any],
        decision_report: dict[str, Any],
        provider_calls: list[dict[str, Any]],
    ) -> None:
        """Persist all loose JSON/JSONL artifacts and package optional evidence ZIP."""
        exec_dir = self._output_dir / "execution"
        results_dir = self._output_dir / "results"
        telemetry_dir = self._output_dir / "telemetry"

        exec_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        telemetry_dir.mkdir(parents=True, exist_ok=True)

        (exec_dir / "v2_d2_development_source_identity.json").write_text(
            json.dumps(execution_identity, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (results_dir / "v2_d2_development_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (results_dir / "v2_d2_development_decision_report.json").write_text(
            json.dumps(decision_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (results_dir / "v2_d2_dimension_diagnostics.json").write_text(
            json.dumps(dimension_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        write_jsonl(results_dir / "v0_claim_predictions.jsonl", v0_claim_preds)
        write_jsonl(results_dir / "v1_claim_predictions.jsonl", v1_claim_preds)
        write_jsonl(results_dir / "v2_d2_claim_predictions_pass1.jsonl", v2_d2_pass1_preds)
        write_jsonl(results_dir / "v2_d2_claim_predictions_pass2.jsonl", v2_d2_pass2_preds)
        write_jsonl(results_dir / "v2_d2_claim_comparisons.jsonl", comparisons)
        write_jsonl(telemetry_dir / "provider_calls.jsonl", provider_calls)

        if self._package_zip_path is not None:
            self._package_zip_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(
                self._package_zip_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.write(
                    exec_dir / "v2_d2_development_source_identity.json",
                    "execution/v2_d2_development_source_identity.json",
                )
                zf.write(
                    results_dir / "v2_d2_development_report.json",
                    "results/v2_d2_development_report.json",
                )
                zf.write(
                    results_dir / "v2_d2_development_decision_report.json",
                    "results/v2_d2_development_decision_report.json",
                )
                zf.write(
                    results_dir / "v2_d2_dimension_diagnostics.json",
                    "results/v2_d2_dimension_diagnostics.json",
                )
                zf.write(
                    results_dir / "v0_claim_predictions.jsonl",
                    "results/v0_claim_predictions.jsonl",
                )
                zf.write(
                    results_dir / "v1_claim_predictions.jsonl",
                    "results/v1_claim_predictions.jsonl",
                )
                zf.write(
                    results_dir / "v2_d2_claim_predictions_pass1.jsonl",
                    "results/v2_d2_claim_predictions_pass1.jsonl",
                )
                zf.write(
                    results_dir / "v2_d2_claim_predictions_pass2.jsonl",
                    "results/v2_d2_claim_predictions_pass2.jsonl",
                )
                zf.write(
                    results_dir / "v2_d2_claim_comparisons.jsonl",
                    "results/v2_d2_claim_comparisons.jsonl",
                )
                zf.write(
                    telemetry_dir / "provider_calls.jsonl",
                    "telemetry/provider_calls.jsonl",
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V2-D2 Structured Semantic Verifier Development Benchmark"
    )
    parser.add_argument(
        "--forensic-packets",
        type=Path,
        required=True,
        help="Path to verification-forensic-review-packets.zip",
    )
    parser.add_argument(
        "--forensic-labels",
        type=Path,
        required=True,
        help="Path to verification-human-forensic-labels-v1.json",
    )
    parser.add_argument(
        "--control-packets",
        type=Path,
        required=True,
        help="Path to verification-positive-control-review-packets-v1.zip",
    )
    parser.add_argument(
        "--control-labels",
        type=Path,
        required=True,
        help="Path to verification-positive-control-human-labels-v1.json",
    )
    parser.add_argument(
        "--v1-evidence",
        type=Path,
        required=True,
        help="Path to canonical verification-semantic-benchmark-evidence.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Optional destination path for evidence ZIP package",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only checksum and V0 replay preflight validation without model execution",
    )
    parser.add_argument(
        "--candidate-id",
        type=str,
        default=CANONICAL_CANDIDATE_ID,
        help="Candidate identity label",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CANONICAL_V2_DEVICE,
        help="Inference device (canonical: cuda)",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=CANONICAL_REPEAT_COUNT,
        help="Number of evaluation passes for stability measurement (canonical: 2)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=args.forensic_packets,
        forensic_labels_path=args.forensic_labels,
        control_packets_path=args.control_packets,
        control_labels_path=args.control_labels,
        v1_evidence_path=args.v1_evidence,
        output_dir=args.output_dir,
        package_zip_path=args.package_zip,
        preflight_only=args.preflight_only,
        candidate_id=args.candidate_id,
        device=args.device,
        repeat_count=args.repeat_count,
    )

    report = evaluator.run()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
