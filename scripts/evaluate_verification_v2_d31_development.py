#!/usr/bin/env python3
"""V2-D3.1 Hierarchical Single-Call Two-Gate Semantic Verifier Development Benchmark Harness.

This script executes the controlled offline development benchmark for candidate V2-D3.1
over the frozen composite 38-claim human-annotated dataset:
- Slice A: 11 claims from suspicious forensic cases (B-FORENSIC-1A)
- Slice B: 27 claims from pre-registered positive-control candidates (B-FORENSIC-1C)
- Baseline V1 predictions loaded from: verification-semantic-benchmark-evidence.zip
- Comparison D3 predictions loaded from: verification-v2-d3-development-evidence.zip

Key features in V2-D3.1:
- Hierarchical Single-Call Two-Gate Semantic Classification:
  1. Gate 1: is_contradicted (true ONLY when evidence positively asserts an incompatible proposition).
  2. Gate 2: is_fully_established (true ONLY when evidence establishes 100% of material propositions).
- Deterministic State Machine Mapping:
  - (True, False)  -> CONTRADICTED
  - (False, True)  -> SUPPORTED
  - (False, False) -> INSUFFICIENT
  - (True, True)   -> INVALID (logically inconsistent state; retry/error)
- Primary Comparator: V2-D3 (28/38 binary, 24/38 three-way, 17/18 supported retained, 11/20 negative caught, 14/22 answer).
- Pre-Registered Selection Gate: D31_SUPERSEDES_D3 vs KEEP_D3 based on exact integer thresholds.
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
    SemanticVerificationConfig,
)
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
import legal_agentic_rag.generation.structured_semantic_verifier_d31
from legal_agentic_rag.generation.structured_semantic_verifier_d31 import (
    DraftRejectionCategoryD31,
    STRUCTURED_SEMANTIC_D31_SYSTEM_INSTRUCTION,
    StructuredClaimVerificationD31,
    StructuredSemanticCitationVerifierD31,
    StructuredSemanticVerificationDraftD31,
    StructuredSemanticVerificationResultD31,
    derive_claim_semantic_label_d31,
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
CANONICAL_D3_EVIDENCE_ZIP_SHA256 = (
    "0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff"
)

# Canonical Pinned V2-D3.1 Model & Execution Parameters
CANONICAL_PACKAGE_VERSION = "0.50.7"
CANONICAL_CANDIDATE_ID = "V2-D3.1"
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

# Forensic Error & Gain Target Sets
D3_FIX_CLAIM_KEYS = {
    ("26541", "BASE", "C1"),
    ("95861", "BASE", "C3"),
    ("95861", "CANDIDATE", "C2"),
    ("95861", "CANDIDATE", "C3"),
    ("103983", "PRIMARY", "C2"),
    ("108497", "PRIMARY", "C1"),
    ("155139", "PRIMARY", "C1"),
}

FORENSIC_ERROR_GROUPS = {
    "Group_A_False_Entailment_Of_Contradiction": [
        ("102047", "BASE", "C1"),
        ("102047", "CANDIDATE", "C1"),
        ("103983", "PRIMARY", "C3"),
    ],
    "Group_B_Contradiction_Undercalls": [
        ("147239", "CANDIDATE", "C2"),
        ("95861", "BASE", "C1"),
        ("95861", "CANDIDATE", "C1"),
        ("150131", "PRIMARY", "C1"),
    ],
    "Group_C_False_Entailment_Of_Insufficient": [
        ("30405", "PRIMARY", "C1"),
        ("40489", "PRIMARY", "C1"),
        ("5967", "PRIMARY", "C1"),
        ("5967", "PRIMARY", "C2"),
        ("5967", "PRIMARY", "C3"),
        ("75171", "PRIMARY", "C1"),
    ],
    "Group_D_Supported_Overrejection": [
        ("31883", "PRIMARY", "C1"),
    ],
}


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

    @property
    def failed_call_count(self) -> int:
        return sum(1 for c in self.call_history if not c["call_succeeded"])

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        call_idx = len(self.call_history)
        t0 = perf_counter()
        completion: str = ""
        success = False
        exc_type: str | None = None
        exc_msg_sha: str | None = None

        sys_sha = sha256(system_instruction.encode("utf-8")).hexdigest()
        prompt_sha = sha256(user_prompt.encode("utf-8")).hexdigest()

        try:
            completion = self._inner.complete(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            )
            success = True
            return completion
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_msg_sha = sha256(str(exc).encode("utf-8")).hexdigest()
            raise
        finally:
            latency = (perf_counter() - t0) * 1000.0
            comp_sha = sha256(completion.encode("utf-8")).hexdigest() if success else None
            record: dict[str, Any] = {
                "call_index": call_idx,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "system_instruction_sha256": sys_sha,
                "user_prompt_sha256": prompt_sha,
                "user_prompt_length": len(user_prompt),
                "call_succeeded": success,
                "completion_sha256": comp_sha,
                "completion_length": len(completion) if success else 0,
                "latency_ms": round(latency, 2),
            }
            if not success:
                record["exception_type"] = exc_type
                record["exception_message_sha256"] = exc_msg_sha
            self.call_history.append(record)


class V2D31DevelopmentBenchmarkEvaluator:
    """Evaluates candidate V2-D3.1 against the composite 38-claim development benchmark."""

    def __init__(
        self,
        *,
        forensic_packets_path: Path,
        forensic_labels_path: Path,
        control_packets_path: Path,
        control_labels_path: Path,
        v1_evidence_path: Path,
        d3_evidence_path: Path,
        output_dir: Path,
        package_zip: Path | None = None,
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
    ) -> None:
        self._forensic_packets_path = forensic_packets_path
        self._forensic_labels_path = forensic_labels_path
        self._control_packets_path = control_packets_path
        self._control_labels_path = control_labels_path
        self._v1_evidence_path = v1_evidence_path
        self._d3_evidence_path = d3_evidence_path
        self._output_dir = output_dir
        self._package_zip = package_zip
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

    def evaluate(self) -> dict[str, Any]:
        """Execute full benchmark evaluation workflow with provenance and safety gates."""
        start_time = perf_counter()
        _LOGGER.info("Starting V2-D3.1 Development Benchmark Evaluation...")

        # 1. Verify inputs and canonical hashes (6 inputs)
        sources_info = self._verify_canonical_source_checksums()

        # 2. Package and environment provenance
        exec_identity = self._build_execution_identity(sources_info)

        # 3. Load dataset slices and bind ground truth
        arm_targets, claim_targets = self._load_and_bind_benchmark_targets()
        _LOGGER.info("Bound %d claims across %d arms", len(claim_targets), len(arm_targets))

        # 4. Load canonical V1 and D3 predictions from evidence archives
        v1_claim_preds = self._load_v1_baseline_predictions()
        d3_claim_preds = self._load_d3_comparison_predictions()

        # 5. Execute V0 replay for baseline fidelity verification
        v0_arm_results, v0_claim_preds, v0_fidelity_stats = self._replay_v0_baseline(arm_targets)

        if self._preflight_only:
            _LOGGER.info("Preflight mode active: skipping model inference.")
            preflight_report = self._build_preflight_report(
                sources_info=sources_info,
                exec_identity=exec_identity,
                arm_targets=arm_targets,
                claim_targets=claim_targets,
                v0_fidelity_stats=v0_fidelity_stats,
                v0_arm_results=v0_arm_results,
                v0_claim_preds=v0_claim_preds,
                v1_claim_preds=v1_claim_preds,
                d3_claim_preds=d3_claim_preds,
            )
            self._write_reports(preflight_report, is_preflight=True)
            return preflight_report

        # 6. Canonical Execution Config Validation (Fail-Closed)
        self._validate_canonical_provenance()

        # 7. Initialize V2-D3.1 verifier provider & validate runtime identity
        raw_provider = self._init_v3_provider()
        self._validate_runtime_provider_identity(raw_provider)

        obs_provider = ObservationalChatModelProviderWrapper(raw_provider)
        verifier = StructuredSemanticCitationVerifierD31(
            provider=obs_provider,
            max_structured_output_retries=self._max_structured_output_retries,
        )

        # 8. Execute Pass 1 (Primary Development Evaluation)
        _LOGGER.info("Executing V2-D3.1 Development Pass 1 (Benchmark Evaluation)...")
        pass1_arm_results, pass1_claim_preds, pass1_telemetry = self._run_inference_pass(
            verifier=verifier,
            provider=obs_provider,
            arm_targets=arm_targets,
            pass_index=1,
        )

        # 9. Execute Pass 2 (Stability Evaluation)
        _LOGGER.info("Executing V2-D3.1 Development Pass 2 (Two-Pass Stability)...")
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

        # 11. Compute Metrics (Pass 1 authoritative against D3 comparator and V1 baseline)
        all_metrics = self._compute_all_metrics(
            claim_targets=claim_targets,
            arm_targets=arm_targets,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds=v1_claim_preds,
            d3_claim_preds=d3_claim_preds,
            d31_claim_preds=pass1_claim_preds,
            v0_arm_results=v0_arm_results,
            d31_arm_results=pass1_arm_results,
        )

        # 12. Compute Dimension Diagnostics & Failure Telemetry (Pass 1 Scientific Diagnostics)
        dim_diagnostics = self._compute_dimension_diagnostics(
            targets=claim_targets,
            d31_preds=pass1_claim_preds,
        )

        # 13. Compute Gain Preservation & Forensic Error Group Diagnostics
        gain_preservation_diag = self._compute_gain_preservation_diagnostic(
            claim_targets=claim_targets,
            d31_preds=pass1_claim_preds,
        )
        forensic_groups_diag = self._compute_forensic_groups_diagnostic(
            claim_targets=claim_targets,
            d31_preds=pass1_claim_preds,
        )

        # 14. Build Final Evaluation and Decision Reports
        total_duration = perf_counter() - start_time
        final_report, decision_report = self._build_reports(
            sources_info=sources_info,
            exec_identity=exec_identity,
            claim_targets=claim_targets,
            arm_targets=arm_targets,
            stability_info=stability_info,
            all_metrics=all_metrics,
            dim_diagnostics=dim_diagnostics,
            gain_preservation_diag=gain_preservation_diag,
            forensic_groups_diag=forensic_groups_diag,
            pass1_telemetry=pass1_telemetry,
            pass2_telemetry=pass2_telemetry,
            total_duration=total_duration,
        )

        # 15. Write output artifacts
        self._write_reports(
            final_report,
            decision_report=decision_report,
            dim_diagnostics=dim_diagnostics,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds=v1_claim_preds,
            d3_claim_preds=d3_claim_preds,
            pass1_claim_preds=pass1_claim_preds,
            pass2_claim_preds=pass2_claim_preds,
            exec_identity=exec_identity,
            provider=obs_provider,
            is_preflight=False,
        )

        _LOGGER.info("V2-D3.1 Development Benchmark complete. Verdict: %s", final_report["verdict"])
        return final_report

    def _verify_canonical_source_checksums(self) -> dict[str, dict[str, Any]]:
        """Verify the 6 canonical input datasets against frozen SHA-256 signatures."""
        sources = {
            "forensic_review_packets": (
                self._forensic_packets_path,
                CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
            ),
            "forensic_labels": (
                self._forensic_labels_path,
                CANONICAL_FORENSIC_LABELS_SHA256,
            ),
            "control_review_packets": (
                self._control_packets_path,
                CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
            ),
            "control_labels": (
                self._control_labels_path,
                CANONICAL_CONTROL_LABELS_SHA256,
            ),
            "canonical_V1_baseline_evidence_archive": (
                self._v1_evidence_path,
                CANONICAL_V1_EVIDENCE_ZIP_SHA256,
            ),
            "canonical_D3_comparison_evidence_archive": (
                self._d3_evidence_path,
                CANONICAL_D3_EVIDENCE_ZIP_SHA256,
            ),
        }

        info: dict[str, dict[str, Any]] = {}
        for key, (path, expected_sha) in sources.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing required benchmark source '{key}' at: {path}")

            actual_sha = self._sha256_file(path)
            if actual_sha != expected_sha:
                raise DataValidationError(
                    f"SHA-256 mismatch for benchmark source '{key}'. "
                    f"Expected: {expected_sha}, Got: {actual_sha}"
                )
            info[key] = {
                "filename": path.name,
                "sha256": actual_sha,
                "size_bytes": path.stat().st_size,
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
            STRUCTURED_SEMANTIC_D31_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest()
        schema_sha = sha256(
            json.dumps(
                StructuredSemanticVerificationDraftD31.model_json_schema(),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        return {
            "schema_version": "1.0",
            "artifact_type": "v2_d31_development_execution_identity",
            "candidate_id": self._candidate_id,
            "execution_git_commit": git_commit,
            "git_worktree_clean": git_clean,
            "package_version": source_ver,
            "source_package_version": source_ver,
            "installed_distribution_version": installed_ver,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "sources": sources_info,
            "provider": {
                "backend": CANONICAL_V3_BACKEND,
                "model_name": CANONICAL_V3_MODEL_NAME,
                "model_revision": CANONICAL_V3_MODEL_REVISION,
                "provider_version": CANONICAL_V3_PROVIDER_VERSION,
            },
            "model_inference_config": {
                "backend": CANONICAL_V3_BACKEND,
                "device": self._device,
                "torch_dtype": self._torch_dtype,
                "temperature": self._temperature,
                "local_files_only": False,
                "max_input_tokens": self._max_input_tokens,
                "max_output_tokens": self._max_output_tokens,
                "max_structured_output_retries": self._max_structured_output_retries,
                "timeout_seconds": self._timeout_seconds,
                "repeat_count": self._repeat_count,
            },
            "runtime_environment": {
                "python_version": sys.version,
                "transformers_version": getattr(
                    importlib.import_module("transformers"), "__version__", "unknown"
                ),
                "torch_version": getattr(
                    importlib.import_module("torch"), "__version__", "unknown"
                ),
                "accelerate_version": getattr(
                    importlib.import_module("accelerate"), "__version__", "unknown"
                ),
                "pydantic_version": getattr(
                    importlib.import_module("pydantic"), "__version__", "unknown"
                ),
                "cuda_available": self._is_cuda_available(),
                "cuda_version": self._get_cuda_version(),
                "cuda_device_name": self._get_cuda_device_name(),
                "cuda_device_count": self._get_cuda_device_count(),
            },
            "implementation_identities": {
                "structured_semantic_verifier_d31_sha256": self._sha256_file(
                    Path(legal_agentic_rag.generation.structured_semantic_verifier_d31.__file__)
                ),
                "evaluate_verification_v2_d31_development_sha256": self._sha256_file(
                    Path(__file__)
                ),
            },
            "prompt_identity": {
                "system_instruction_sha256": system_instruction_sha,
            },
            "schema_identity": {
                "structured_verification_json_schema_sha256": schema_sha,
            },
        }

    def _unpack_zip(self, zip_path: Path, prefix: str) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
        return temp_dir

    def _cleanup_temp(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    def _load_and_bind_benchmark_targets(
        self,
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
            raw_sha = sha256(raw_claim_text.encode("utf-8")).hexdigest()
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
                    human_label=HumanEntailment(cl_data.get("entailment_label")),
                    error_tags=cl_data.get("error_tags", []),
                    diagnostic_note=cl_data.get("diagnostic_note"),
                    stratum=stratum or "A_SINGLE_CLAIM_CLEAN",
                )
            )

        arm_target = BenchmarkArmTarget(
            slice_id=slice_id,
            question_id=qid,
            arm_id=arm_id,
            historical_stop_reason=arm_label_data.get(
                "historical_stop_reason", "answer_verified"
            ),
            stratum=stratum or "A_SINGLE_CLAIM_CLEAN",
            question_text=question_text,
            answer_response=resp_obj,
            evidence_list=evidence_list,
            historical_verification=hist_verification,
            claims=claim_targets,
        )
        return arm_target, claim_targets

    def _load_v1_baseline_predictions(self) -> list[dict[str, Any]]:
        with zipfile.ZipFile(self._v1_evidence_path, "r") as zf:
            with zf.open("results/v1_claim_predictions_pass1.jsonl") as f:
                lines = [line.decode("utf-8").strip() for line in f if line.strip()]
        return [json.loads(line) for line in lines]

    def _load_d3_comparison_predictions(self) -> list[dict[str, Any]]:
        with zipfile.ZipFile(self._d3_evidence_path, "r") as zf:
            with zf.open("results/v2_d3_claim_predictions_pass1.jsonl") as f:
                lines = [line.decode("utf-8").strip() for line in f if line.strip()]
        return [json.loads(line) for line in lines]

    def _replay_v0_baseline(
        self, arm_targets: list[BenchmarkArmTarget]
    ) -> tuple[dict[str, CitationVerificationResult], list[dict[str, Any]], dict[str, Any]]:
        rule_verifier = RuleBasedCitationVerifier()
        arm_results: dict[str, CitationVerificationResult] = {}
        claim_preds: list[dict[str, Any]] = []

        total_arms = len(arm_targets)
        exact_passes = 0

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            res = rule_verifier.verify(arm.answer_response, arm.evidence_list)
            arm_results[key] = res

            hist_valid = arm.historical_verification.get("is_valid")
            if hist_valid is not None and res.is_valid == hist_valid:
                exact_passes += 1

            for claim in arm.claims:
                cv = next(
                    (c for c in res.claim_verifications if c.claim_id == claim.claim_id),
                    None,
                )
                bin_pred = (
                    BinaryPrediction.ACCEPT.value
                    if (cv and cv.status == ClaimSupportStatus.SUPPORTED)
                    else BinaryPrediction.REJECT.value
                )
                claim_preds.append({
                    "slice_id": claim.slice_id,
                    "question_id": claim.question_id,
                    "arm_id": claim.arm_id,
                    "claim_id": claim.claim_id,
                    "human_label": claim.human_label.value,
                    "v0_binary_prediction": bin_pred,
                    "v0_rule_status": cv.status.value if cv else "MISSING",
                })

        fidelity_stats = {
            "v0_replay_arm_passes": exact_passes,
            "v0_replay_arm_total": total_arms,
            "v0_replay_100_percent_fidelity": exact_passes == total_arms,
        }
        return arm_results, claim_preds, fidelity_stats

    def _validate_package_provenance(self) -> None:
        source_ver = getattr(legal_agentic_rag, "__version__", "unknown")
        try:
            dist_ver = importlib.metadata.version("legal-agentic-rag")
        except Exception as exc:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Failed to read installed distribution version: {exc}"
            ) from exc

        if (
            source_ver != CANONICAL_PACKAGE_VERSION
            or dist_ver != CANONICAL_PACKAGE_VERSION
            or source_ver != dist_ver
        ):
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Package version mismatch: source='{source_ver}', installed='{dist_ver}', expected='{CANONICAL_PACKAGE_VERSION}'"
            )

    def _validate_canonical_provenance(self) -> None:
        if self._custom_provider is not None:
            return
        if self._candidate_id != CANONICAL_CANDIDATE_ID:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Candidate ID mismatch: expected '{CANONICAL_CANDIDATE_ID}', got '{self._candidate_id}'"
            )
        self._validate_package_provenance()
        if not self._is_git_worktree_clean():
            raise DataValidationError(
                "INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Git worktree must be clean for canonical real execution"
            )
        git_head = self._get_git_commit()
        if len(git_head) != 40 or not all(c in "0123456789abcdef" for c in git_head.lower()):
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Invalid 40-char Git HEAD commit: '{git_head}'"
            )
        if self._device != CANONICAL_V3_DEVICE:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected device '{CANONICAL_V3_DEVICE}', got '{self._device}'"
            )
        if self._torch_dtype != CANONICAL_V3_TORCH_DTYPE:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected torch_dtype '{CANONICAL_V3_TORCH_DTYPE}', got '{self._torch_dtype}'"
            )
        if self._temperature != CANONICAL_V3_TEMPERATURE:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected temperature {CANONICAL_V3_TEMPERATURE}, got {self._temperature}"
            )
        if self._repeat_count != CANONICAL_REPEAT_COUNT:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected repeat count {CANONICAL_REPEAT_COUNT}, got {self._repeat_count}"
            )
        if self._max_structured_output_retries != CANONICAL_V3_MAX_STRUCTURED_RETRIES:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected max retries {CANONICAL_V3_MAX_STRUCTURED_RETRIES}, got {self._max_structured_output_retries}"
            )
        if self._max_input_tokens != CANONICAL_V3_MAX_INPUT_TOKENS:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected max input tokens {CANONICAL_V3_MAX_INPUT_TOKENS}, got {self._max_input_tokens}"
            )
        if self._max_output_tokens != CANONICAL_V3_MAX_OUTPUT_TOKENS:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected max output tokens {CANONICAL_V3_MAX_OUTPUT_TOKENS}, got {self._max_output_tokens}"
            )
        if self._timeout_seconds != CANONICAL_V3_TIMEOUT_SECONDS:
            raise DataValidationError(
                f"INVALID_V2_D31_DEVELOPMENT_PROVENANCE: Expected timeout {CANONICAL_V3_TIMEOUT_SECONDS}, got {self._timeout_seconds}"
            )
        if not self._is_cuda_available():
            raise DataValidationError(
                "INVALID_V2_D31_DEVELOPMENT_PROVENANCE: CUDA device is required for canonical real execution"
            )

    def _init_v3_provider(self) -> ChatModelProvider:
        if self._custom_provider is not None:
            return self._custom_provider

        cfg = SemanticVerificationConfig(
            backend=CANONICAL_V3_BACKEND,
            model_name=CANONICAL_V3_MODEL_NAME,
            model_revision=CANONICAL_V3_MODEL_REVISION,
            device=self._device,
            torch_dtype=self._torch_dtype,
            local_files_only=False,
            timeout_seconds=self._timeout_seconds,
            max_input_tokens=self._max_input_tokens,
            max_output_tokens=self._max_output_tokens,
            max_structured_output_retries=self._max_structured_output_retries,
        )
        generation_cfg = cfg.as_generation_config()

        if (
            generation_cfg.backend != CANONICAL_V3_BACKEND
            or generation_cfg.model_name != CANONICAL_V3_MODEL_NAME
            or generation_cfg.model_revision != CANONICAL_V3_MODEL_REVISION
            or generation_cfg.device != CANONICAL_V3_DEVICE
            or generation_cfg.torch_dtype != CANONICAL_V3_TORCH_DTYPE
            or generation_cfg.local_files_only is not False
            or generation_cfg.timeout_seconds != CANONICAL_V3_TIMEOUT_SECONDS
            or generation_cfg.max_input_tokens != CANONICAL_V3_MAX_INPUT_TOKENS
            or generation_cfg.max_output_tokens != CANONICAL_V3_MAX_OUTPUT_TOKENS
            or generation_cfg.max_structured_output_retries != CANONICAL_V3_MAX_STRUCTURED_RETRIES
            or generation_cfg.temperature != 0.0
        ):
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: GenerationConfig invariants failed: {generation_cfg}"
            )

        return TransformersChatProvider(generation_cfg)

    def _validate_runtime_provider_identity(self, provider: ChatModelProvider) -> None:
        if self._custom_provider is not None:
            return
        p_name = getattr(provider, "provider_name", "unknown")
        if p_name != CANONICAL_V3_BACKEND:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Provider name mismatch: expected '{CANONICAL_V3_BACKEND}', got '{p_name}'"
            )
        p_model = getattr(provider, "model_name", "unknown")
        if p_model != CANONICAL_V3_MODEL_NAME:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Model name mismatch: expected '{CANONICAL_V3_MODEL_NAME}', got '{p_model}'"
            )
        p_rev = getattr(provider, "model_revision", "unknown")
        if p_rev != CANONICAL_V3_MODEL_REVISION:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Model revision mismatch: expected '{CANONICAL_V3_MODEL_REVISION}', got '{p_rev}'"
            )
        p_ver = getattr(provider, "provider_version", "unknown")
        if not p_ver or p_ver == "unknown" or p_ver != CANONICAL_V3_PROVIDER_VERSION:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Provider version mismatch: expected '{CANONICAL_V3_PROVIDER_VERSION}', got '{p_ver}'"
            )

    def _run_inference_pass(
        self,
        verifier: StructuredSemanticCitationVerifierD31,
        provider: ObservationalChatModelProviderWrapper,
        arm_targets: list[BenchmarkArmTarget],
        pass_index: int,
    ) -> tuple[dict[str, CitationVerificationResult], list[dict[str, Any]], dict[str, Any]]:
        start_calls = provider.total_calls
        start_errors = provider.failed_call_count

        arm_results: dict[str, CitationVerificationResult] = {}
        claim_preds: list[dict[str, Any]] = []

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            cit_res, struct_res = verifier.verify_structured(
                arm.answer_response, arm.evidence_list
            )
            arm_results[key] = cit_res

            struct_map = {a.claim_id: a for a in struct_res.assessments}
            telemetry_map = struct_res.claim_telemetries

            for claim in arm.claims:
                assess = struct_map.get(claim.claim_id)
                telemetry = telemetry_map.get(claim.claim_id)
                telemetry_dict = telemetry.model_dump() if telemetry else {}

                is_exec_err = (
                    claim.claim_id in struct_res.execution_error_claims
                    or (telemetry is not None and telemetry.semantic_execution_error)
                )

                if is_exec_err:
                    bin_pred = BinaryPrediction.EXECUTION_ERROR.value
                    three_way_pred = "EXECUTION_ERROR"
                    is_contra = is_estab = False
                    assessment_dict = None
                elif assess is not None:
                    bin_pred = (
                        BinaryPrediction.ACCEPT.value
                        if assess.label == SemanticSupportLabel.SUPPORTED
                        else BinaryPrediction.REJECT.value
                    )
                    three_way_pred = assess.label.value.upper()
                    is_contra = assess.is_contradicted
                    is_estab = assess.is_fully_established
                    assessment_dict = {
                        "claim_id": assess.claim_id,
                        "is_contradicted": assess.is_contradicted,
                        "is_fully_established": assess.is_fully_established,
                        "derived_label": assess.label.value.upper(),
                        "telemetry": telemetry_dict,
                    }
                else:
                    bin_pred = BinaryPrediction.EXECUTION_ERROR.value
                    three_way_pred = "EXECUTION_ERROR"
                    is_contra = is_estab = False
                    assessment_dict = None

                claim_preds.append({
                    "pass_index": pass_index,
                    "slice_id": claim.slice_id,
                    "question_id": claim.question_id,
                    "arm_id": claim.arm_id,
                    "claim_id": claim.claim_id,
                    "human_label": claim.human_label.value,
                    "error_tags": claim.error_tags,
                    "v2_d31_binary_prediction": bin_pred,
                    "v2_d31_three_way_prediction": three_way_pred,
                    "is_contradicted": is_contra,
                    "is_fully_established": is_estab,
                    "telemetry": telemetry_dict,
                    "structured_assessment": assessment_dict,
                })

        end_calls = provider.total_calls
        end_errors = provider.failed_call_count

        provider_calls_in_pass = end_calls - start_calls
        provider_invocation_errors_in_pass = end_errors - start_errors

        claim_provider_call_count_sum = sum(
            p.get("telemetry", {}).get("provider_call_count", 0) for p in claim_preds
        )
        claim_retry_count_sum = sum(
            p.get("telemetry", {}).get("retry_count", 0) for p in claim_preds
        )
        semantic_execution_error_count = sum(
            1 for p in claim_preds if p.get("v2_d31_three_way_prediction") == "EXECUTION_ERROR"
        )
        draft_rejection_categories: dict[str, int] = defaultdict(int)
        for p in claim_preds:
            for cat in p.get("telemetry", {}).get("draft_rejection_categories", []):
                draft_rejection_categories[cat] += 1

        if claim_provider_call_count_sum != provider_calls_in_pass:
            raise DataValidationError(
                f"INVALID_VERIFIER_BENCHMARK_PROVENANCE: Provider call reconciliation mismatch in Pass {pass_index}: "
                f"claim_provider_call_count_sum={claim_provider_call_count_sum} != provider_calls_in_pass={provider_calls_in_pass}"
            )

        pass_telemetry = {
            "pass_index": pass_index,
            "provider_calls": provider_calls_in_pass,
            "provider_invocation_errors": provider_invocation_errors_in_pass,
            "structured_retries": claim_retry_count_sum,
            "semantic_execution_errors": semantic_execution_error_count,
            "draft_rejection_categories": dict(draft_rejection_categories),
        }
        return arm_results, claim_preds, pass_telemetry

    def _evaluate_stability(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        pass1_preds: list[dict[str, Any]],
        pass2_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        p1_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r
            for r in pass1_preds
        }
        p2_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r
            for r in pass2_preds
        }

        total_claims = len(claim_targets)
        claims_with_two_valid_labels = 0
        stable_semantic_count = 0
        unstable_semantic_count = 0

        pass1_exec_error_count = 0
        pass2_exec_error_count = 0
        exec_error_in_any_pass_count = 0
        repeated_exec_error_count = 0

        unstable_claims: list[dict[str, Any]] = []
        execution_error_claims: list[dict[str, Any]] = []

        valid_semantic_labels = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}

        for t in claim_targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            p1 = p1_map.get(k, {})
            p2 = p2_map.get(k, {})

            l1 = p1.get("v2_d31_three_way_prediction", "EXECUTION_ERROR")
            l2 = p2.get("v2_d31_three_way_prediction", "EXECUTION_ERROR")

            p1_err = l1 not in valid_semantic_labels
            p2_err = l2 not in valid_semantic_labels

            if p1_err:
                pass1_exec_error_count += 1
            if p2_err:
                pass2_exec_error_count += 1

            if p1_err or p2_err:
                exec_error_in_any_pass_count += 1
                if p1_err and p2_err:
                    repeated_exec_error_count += 1
                execution_error_claims.append({
                    "question_id": t.question_id,
                    "arm_id": t.arm_id,
                    "claim_id": t.claim_id,
                    "pass1_prediction": l1,
                    "pass2_prediction": l2,
                    "error_tags": t.error_tags,
                })
            else:
                claims_with_two_valid_labels += 1
                if l1 == l2:
                    stable_semantic_count += 1
                else:
                    unstable_semantic_count += 1
                    unstable_claims.append({
                        "question_id": t.question_id,
                        "arm_id": t.arm_id,
                        "claim_id": t.claim_id,
                        "pass1_prediction": l1,
                        "pass2_prediction": l2,
                        "error_tags": t.error_tags,
                    })

        pct = (
            (stable_semantic_count / claims_with_two_valid_labels * 100.0)
            if claims_with_two_valid_labels > 0
            else 0.0
        )

        return {
            "total_claims": total_claims,
            "claims_with_two_valid_semantic_labels": claims_with_two_valid_labels,
            "stable_semantic_claim_count": stable_semantic_count,
            "unstable_semantic_claim_count": unstable_semantic_count,
            "successful_label_stability_percentage": round(pct, 2),
            "label_stability_percentage": round(pct, 2),
            "unstable_claim_count": unstable_semantic_count,
            "pass1_execution_error_count": pass1_exec_error_count,
            "pass2_execution_error_count": pass2_exec_error_count,
            "execution_error_in_any_pass_count": exec_error_in_any_pass_count,
            "repeated_execution_error_claim_count": repeated_exec_error_count,
            "unstable_claims": unstable_claims,
            "execution_error_claims": execution_error_claims,
        }

    def _compute_all_metrics(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        arm_targets: list[BenchmarkArmTarget],
        v0_claim_preds: list[dict[str, Any]],
        v1_claim_preds: list[dict[str, Any]],
        d3_claim_preds: list[dict[str, Any]],
        d31_claim_preds: list[dict[str, Any]],
        v0_arm_results: dict[str, CitationVerificationResult],
        d31_arm_results: dict[str, CitationVerificationResult],
    ) -> dict[str, Any]:
        v0_binary = self._compute_binary_metrics(
            claim_targets, v0_claim_preds, pred_key="v0_binary_prediction"
        )
        v1_binary = self._compute_binary_metrics(
            claim_targets, v1_claim_preds, pred_key="v1_binary_prediction"
        )
        d3_binary = self._compute_binary_metrics(
            claim_targets, d3_claim_preds, pred_key="v2_d3_binary_prediction"
        )
        d31_binary = self._compute_binary_metrics(
            claim_targets, d31_claim_preds, pred_key="v2_d31_binary_prediction"
        )

        v1_three_way = self._compute_three_way_metrics(
            claim_targets, v1_claim_preds, label_key="v1_three_way_prediction"
        )
        d3_three_way = self._compute_three_way_metrics(
            claim_targets, d3_claim_preds, label_key="v2_d3_three_way_prediction"
        )
        d31_three_way = self._compute_three_way_metrics(
            claim_targets, d31_claim_preds, label_key="v2_d31_three_way_prediction"
        )

        # Primary Paired Metrics: D3.1 vs D3
        paired_d31_vs_d3 = self._compute_paired_metrics(
            claim_targets, d3_claim_preds, d31_claim_preds,
            base_key="v2_d3_three_way_prediction", cand_key="v2_d31_three_way_prediction"
        )
        # Secondary Paired Metrics: D3.1 vs V1
        paired_d31_vs_v1 = self._compute_paired_metrics(
            claim_targets, v1_claim_preds, d31_claim_preds,
            base_key="v1_three_way_prediction", cand_key="v2_d31_three_way_prediction"
        )

        # Contradiction-Specific Capabilities
        contra_diag = self._compute_contradiction_diagnostics(
            claim_targets, d3_claim_preds, d31_claim_preds
        )

        v0_answer = self._compute_answer_level_metrics(arm_targets, v0_arm_results)
        v1_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, v1_claim_preds, pred_key="v1_three_way_prediction"
        )
        d3_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, d3_claim_preds, pred_key="v2_d3_three_way_prediction"
        )
        d31_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, d31_claim_preds, pred_key="v2_d31_three_way_prediction"
        )

        answer_deltas = {
            "d31_vs_d3_valid_retention_delta": round(
                d31_answer["valid_answer_retention_rate"] - d3_answer["valid_answer_retention_rate"], 4
            ),
            "d31_vs_d3_invalid_catch_delta": round(
                d31_answer["invalid_answer_catch_rate"] - d3_answer["invalid_answer_catch_rate"], 4
            ),
            "d31_vs_d3_answer_accuracy_delta": round(
                d31_answer["answer_level_accuracy"] - d3_answer["answer_level_accuracy"], 4
            ),
        }

        return {
            "v0_claim_binary": v0_binary,
            "v1_claim_binary": v1_binary,
            "v2_d3_claim_binary": d3_binary,
            "v2_d31_claim_binary": d31_binary,
            "v1_three_way": v1_three_way,
            "v2_d3_three_way": d3_three_way,
            "v2_d31_three_way": d31_three_way,
            "paired_v2_d31_vs_v2_d3": paired_d31_vs_d3,
            "paired_v2_d31_vs_v1": paired_d31_vs_v1,
            "contradiction_capability": contra_diag,
            "v0_answer_metrics": v0_answer,
            "v1_answer_metrics": v1_answer,
            "v2_d3_answer_metrics": d3_answer,
            "v2_d31_answer_metrics": d31_answer,
            "v2_d31_vs_d3_answer_deltas": answer_deltas,
        }

    def _compute_binary_metrics(
        self,
        targets: list[BenchmarkClaimTarget],
        preds: list[dict[str, Any]],
        pred_key: str,
    ) -> dict[str, Any]:
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
                if pred_val == BinaryPrediction.ACCEPT.value:
                    fp += 1
                else:
                    tn += 1

        total = len(targets)
        evaluated = total - exec_errors
        acc = (tp + tn) / evaluated if evaluated else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        retention = tp / (tp + fn) if (tp + fn) else 0.0
        catch = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * prec * retention) / (prec + retention) if (prec + retention) else 0.0
        bal_acc = (retention + catch) / 2.0

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "execution_errors": exec_errors,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "supported_retention": round(retention, 4),
            "negative_catch": round(catch, 4),
            "f1": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
        }

    def _compute_three_way_metrics(
        self,
        targets: list[BenchmarkClaimTarget],
        preds: list[dict[str, Any]],
        label_key: str,
    ) -> dict[str, Any]:
        pred_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r[label_key]
            for r in preds
        }

        classes = [
            HumanEntailment.SUPPORTED.value,
            HumanEntailment.CONTRADICTED.value,
            HumanEntailment.INSUFFICIENT.value,
        ]
        matrix = {g: {p: 0 for p in classes} for g in classes}
        exec_errors = 0
        correct = 0
        total_eval = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            pred_lbl = pred_map.get(k, "INSUFFICIENT")
            gold_lbl = t.human_label.value

            if pred_lbl == "EXECUTION_ERROR":
                exec_errors += 1
            elif pred_lbl in classes:
                matrix[gold_lbl][pred_lbl] += 1
                total_eval += 1
                if pred_lbl == gold_lbl:
                    correct += 1
            else:
                exec_errors += 1

        overall_acc = correct / total_eval if total_eval else 0.0

        per_class: dict[str, dict[str, Any]] = {}
        for c in classes:
            tp = matrix[c][c]
            fp = sum(matrix[other][c] for other in classes if other != c)
            fn = sum(matrix[c][other] for other in classes if other != c)
            support = sum(matrix[c].values())

            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }

        macro_p = sum(v["precision"] for v in per_class.values()) / 3.0
        macro_r = sum(v["recall"] for v in per_class.values()) / 3.0
        macro_f1 = sum(v["f1"] for v in per_class.values()) / 3.0

        return {
            "confusion_matrix": matrix,
            "accuracy": round(overall_acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class,
            "execution_errors": exec_errors,
        }

    def _compute_paired_metrics(
        self,
        targets: list[BenchmarkClaimTarget],
        base_preds: list[dict[str, Any]],
        cand_preds: list[dict[str, Any]],
        base_key: str,
        cand_key: str,
    ) -> dict[str, Any]:
        base_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r[base_key]
            for r in base_preds
        }
        cand_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r[cand_key]
            for r in cand_preds
        }

        both_corr = base_only = cand_only = both_wrg = 0
        cand_exec_err = 0
        semantic_regressions = 0
        exec_error_regressions = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            b_val = base_map.get(k)
            c_val = cand_map.get(k)
            gold = t.human_label.value

            b_c = (b_val == gold)
            c_c = (c_val == gold)

            if c_val == "EXECUTION_ERROR":
                cand_exec_err += 1

            if b_c and c_c:
                both_corr += 1
            elif b_c and not c_c:
                base_only += 1
                if c_val == "EXECUTION_ERROR":
                    exec_error_regressions += 1
                else:
                    semantic_regressions += 1
            elif not b_c and c_c:
                cand_only += 1
            else:
                both_wrg += 1

        net_delta = cand_only - base_only

        return {
            "both_correct": both_corr,
            "base_only_correct": base_only,
            "candidate_only_correct": cand_only,
            "both_wrong": both_wrg,
            "net_correctness_delta": net_delta,
            "candidate_fixes_count": cand_only,
            "candidate_regressions_count": base_only,
            "semantic_regressions_count": semantic_regressions,
            "execution_error_regressions_count": exec_error_regressions,
            "candidate_execution_error_count": cand_exec_err,
        }

    def _compute_contradiction_diagnostics(
        self,
        targets: list[BenchmarkClaimTarget],
        d3_preds: list[dict[str, Any]],
        d31_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        d3_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r.get("v2_d3_three_way_prediction")
            for r in d3_preds
        }
        d31_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r.get("v2_d31_three_way_prediction")
            for r in d31_preds
        }

        gold_contradicted_claims = [
            t for t in targets if t.human_label == HumanEntailment.CONTRADICTED
        ]
        total_gold_contra = len(gold_contradicted_claims)  # 7

        d3_correct_contra = sum(
            1 for t in gold_contradicted_claims
            if d3_map.get((t.question_id, t.arm_id, t.claim_id)) == "CONTRADICTED"
        )
        d31_correct_contra = sum(
            1 for t in gold_contradicted_claims
            if d31_map.get((t.question_id, t.arm_id, t.claim_id)) == "CONTRADICTED"
        )

        contra_claims_breakdown = []
        for t in gold_contradicted_claims:
            k = (t.question_id, t.arm_id, t.claim_id)
            contra_claims_breakdown.append({
                "question_id": t.question_id,
                "arm_id": t.arm_id,
                "claim_id": t.claim_id,
                "d3_prediction": d3_map.get(k),
                "d31_prediction": d31_map.get(k),
                "d31_correctly_identified_as_contradicted": d31_map.get(k) == "CONTRADICTED",
            })

        return {
            "total_gold_contradicted_claims": total_gold_contra,
            "d3_correctly_predicted_contradicted_count": d3_correct_contra,
            "d31_correctly_predicted_contradicted_count": d31_correct_contra,
            "d31_contradicted_recall": round(
                d31_correct_contra / total_gold_contra if total_gold_contra else 0.0, 4
            ),
            "contradiction_discrimination_improved": bool(d31_correct_contra > d3_correct_contra),
            "contradicted_claims_detail": contra_claims_breakdown,
        }

    def _compute_gain_preservation_diagnostic(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        d31_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        d31_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r.get("v2_d31_three_way_prediction")
            for r in d31_preds
        }

        preserved_count = 0
        regressed_claim_ids = []
        details = []

        for t in claim_targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            if k in D3_FIX_CLAIM_KEYS:
                d31_val = d31_map.get(k)
                gold_val = t.human_label.value
                is_preserved = (d31_val == gold_val)
                if is_preserved:
                    preserved_count += 1
                else:
                    regressed_claim_ids.append(f"{t.question_id}:{t.arm_id}:{t.claim_id}")

                details.append({
                    "claim_key": f"{t.question_id}:{t.arm_id}:{t.claim_id}",
                    "human_label": gold_val,
                    "d31_prediction": d31_val,
                    "is_preserved": is_preserved,
                })

        return {
            "total_d3_fixes": len(D3_FIX_CLAIM_KEYS),
            "preserved_gain_count": preserved_count,
            "regressed_gain_count": len(regressed_claim_ids),
            "regressed_gain_claim_ids": regressed_claim_ids,
            "all_7_d3_fixes_preserved": preserved_count == len(D3_FIX_CLAIM_KEYS),
            "d3_fixes_detail": details,
        }

    def _compute_forensic_groups_diagnostic(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        d31_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        d31_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r.get("v2_d31_three_way_prediction")
            for r in d31_preds
        }
        target_map = {
            (t.question_id, t.arm_id, t.claim_id): t for t in claim_targets
        }

        groups_summary = {}
        for group_name, keys in FORENSIC_ERROR_GROUPS.items():
            total = len(keys)
            resolved = 0
            items = []
            for k in keys:
                t = target_map.get(k)
                pred = d31_map.get(k)
                gold = t.human_label.value if t else "UNKNOWN"
                is_fixed = (pred == gold)
                if is_fixed:
                    resolved += 1
                items.append({
                    "claim_key": f"{k[0]}:{k[1]}:{k[2]}",
                    "human_label": gold,
                    "d31_prediction": pred,
                    "is_fixed_in_d31": is_fixed,
                })
            groups_summary[group_name] = {
                "total_claims": total,
                "resolved_in_d31_count": resolved,
                "claims": items,
            }

        return groups_summary

    def _compute_answer_level_metrics(
        self,
        arm_targets: list[BenchmarkArmTarget],
        arm_results: dict[str, CitationVerificationResult],
    ) -> dict[str, Any]:
        total_arms = len(arm_targets)
        valid_retained = 0
        invalid_caught = 0
        total_valid = 0
        total_invalid = 0
        correct_arms = 0

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            res = arm_results.get(key)
            gold_valid = all(c.human_label == HumanEntailment.SUPPORTED for c in arm.claims)

            if gold_valid:
                total_valid += 1
                if res and res.is_valid:
                    valid_retained += 1
                    correct_arms += 1
            else:
                total_invalid += 1
                if res and not res.is_valid:
                    invalid_caught += 1
                    correct_arms += 1

        retention_rate = valid_retained / total_valid if total_valid else 0.0
        catch_rate = invalid_caught / total_invalid if total_invalid else 0.0
        acc = correct_arms / total_arms if total_arms else 0.0

        return {
            "total_answers": total_arms,
            "valid_ground_truth_answers": total_valid,
            "invalid_ground_truth_answers": total_invalid,
            "valid_answers_retained": valid_retained,
            "invalid_answers_caught": invalid_caught,
            "valid_answer_retention_rate": round(retention_rate, 4),
            "invalid_answer_catch_rate": round(catch_rate, 4),
            "answer_level_accuracy": round(acc, 4),
        }

    def _compute_answer_level_metrics_from_preds(
        self,
        arm_targets: list[BenchmarkArmTarget],
        claim_preds: list[dict[str, Any]],
        pred_key: str,
        supported_val: str = "SUPPORTED",
    ) -> dict[str, Any]:
        preds_by_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for p in claim_preds:
            preds_by_arm[(p["question_id"], p["arm_id"])].append(p)

        total_arms = len(arm_targets)
        execution_error_answers = 0
        evaluated_correct_answers = 0
        valid_answers_retained = 0
        invalid_answers_caught = 0
        evaluated_valid_ground_truth = 0
        evaluated_invalid_ground_truth = 0
        total_valid_ground_truth = 0
        total_invalid_ground_truth = 0

        for arm in arm_targets:
            key = (arm.question_id, arm.arm_id)
            c_preds = preds_by_arm.get(key, [])
            gold_valid = all(c.human_label == HumanEntailment.SUPPORTED for c in arm.claims)

            if gold_valid:
                total_valid_ground_truth += 1
            else:
                total_invalid_ground_truth += 1

            has_exec_err = (
                not c_preds
                or any(
                    p.get(pred_key) in ("EXECUTION_ERROR", BinaryPrediction.EXECUTION_ERROR.value)
                    for p in c_preds
                )
            )

            if has_exec_err:
                execution_error_answers += 1
                continue

            # Successfully evaluated arm
            pred_valid = all(p.get(pred_key) == supported_val for p in c_preds)

            if gold_valid:
                evaluated_valid_ground_truth += 1
                if pred_valid:
                    valid_answers_retained += 1
                    evaluated_correct_answers += 1
            else:
                evaluated_invalid_ground_truth += 1
                if not pred_valid:
                    invalid_answers_caught += 1
                    evaluated_correct_answers += 1

        evaluated_answers = total_arms - execution_error_answers
        valid_retention_rate = (
            valid_answers_retained / evaluated_valid_ground_truth
            if evaluated_valid_ground_truth
            else 0.0
        )
        invalid_catch_rate = (
            invalid_answers_caught / evaluated_invalid_ground_truth
            if evaluated_invalid_ground_truth
            else 0.0
        )
        evaluated_accuracy = (
            evaluated_correct_answers / evaluated_answers
            if evaluated_answers
            else 0.0
        )
        full_accuracy = (
            evaluated_correct_answers / total_arms
            if total_arms
            else 0.0
        )

        return {
            "total_answers": total_arms,
            "evaluated_answers": evaluated_answers,
            "execution_error_answers": execution_error_answers,
            "valid_ground_truth_answers": total_valid_ground_truth,
            "invalid_ground_truth_answers": total_invalid_ground_truth,
            "evaluated_valid_ground_truth_answers": evaluated_valid_ground_truth,
            "evaluated_invalid_ground_truth_answers": evaluated_invalid_ground_truth,
            "valid_answers_retained": valid_answers_retained,
            "invalid_answers_caught": invalid_answers_caught,
            "valid_answer_retention_rate": round(valid_retention_rate, 4),
            "evaluated_valid_answer_retention_rate": round(valid_retention_rate, 4),
            "invalid_answer_catch_rate": round(invalid_catch_rate, 4),
            "evaluated_invalid_answer_catch_rate": round(invalid_catch_rate, 4),
            "answer_level_accuracy": round(evaluated_accuracy, 4),
            "evaluated_answer_accuracy": round(evaluated_accuracy, 4),
            "full_denominator_answer_accuracy": round(full_accuracy, 4),
            "execution_errors": execution_error_answers,
        }

    def _compute_dimension_diagnostics(
        self,
        targets: list[BenchmarkClaimTarget],
        d31_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pred_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r
            for r in d31_preds
        }

        two_gate_counts = {
            "is_contradicted_true": 0,
            "is_contradicted_false": 0,
            "is_fully_established_true": 0,
            "is_fully_established_false": 0,
        }
        state_distribution = {
            "State_A_Contradicted (True, False)": 0,
            "State_B_Supported (False, True)": 0,
            "State_C_Insufficient (False, False)": 0,
            "State_Invalid (True, True)": 0,
        }

        rejection_categories = Counter()
        total_retries = 0
        claims_with_exec_error = 0
        successfully_structured_claims = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            pred = pred_map.get(k) or {}

            telemetry = pred.get("telemetry") or {}
            if not telemetry and pred.get("structured_assessment"):
                telemetry = pred["structured_assessment"].get("telemetry") or {}

            total_retries += telemetry.get("retry_count", 0)
            for cat in telemetry.get("draft_rejection_categories", []):
                rejection_categories[cat] += 1

            is_exec_err = (
                telemetry.get("semantic_execution_error")
                or pred.get("v2_d31_binary_prediction") == BinaryPrediction.EXECUTION_ERROR.value
                or not pred.get("structured_assessment")
            )

            if is_exec_err:
                claims_with_exec_error += 1
                continue

            successfully_structured_claims += 1
            sa = pred["structured_assessment"]
            contra = sa.get("is_contradicted", False)
            estab = sa.get("is_fully_established", False)

            if contra:
                two_gate_counts["is_contradicted_true"] += 1
            else:
                two_gate_counts["is_contradicted_false"] += 1

            if estab:
                two_gate_counts["is_fully_established_true"] += 1
            else:
                two_gate_counts["is_fully_established_false"] += 1

            if contra and not estab:
                state_distribution["State_A_Contradicted (True, False)"] += 1
            elif not contra and estab:
                state_distribution["State_B_Supported (False, True)"] += 1
            elif not contra and not estab:
                state_distribution["State_C_Insufficient (False, False)"] += 1
            else:
                state_distribution["State_Invalid (True, True)"] += 1

        return {
            "schema_version": "1.0",
            "artifact_type": "v2_d31_dimension_diagnostics",
            "candidate_id": self._candidate_id,
            "diagnostic_pass": 1,
            "total_claims": len(targets),
            "successfully_structured_claim_count": successfully_structured_claims,
            "execution_error_claim_count": claims_with_exec_error,
            "two_gate_counts": two_gate_counts,
            "state_distribution": state_distribution,
            "rejection_telemetry_summary": {
                "total_retries": total_retries,
                "rejection_categories": dict(rejection_categories),
            },
        }

    def _build_preflight_report(
        self,
        sources_info: dict[str, dict[str, Any]],
        exec_identity: dict[str, Any],
        arm_targets: list[BenchmarkArmTarget],
        claim_targets: list[BenchmarkClaimTarget],
        v0_fidelity_stats: dict[str, Any],
        v0_arm_results: dict[str, CitationVerificationResult],
        v0_claim_preds: list[dict[str, Any]],
        v1_claim_preds: list[dict[str, Any]],
        d3_claim_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        v0_binary = self._compute_binary_metrics(
            claim_targets, v0_claim_preds, pred_key="v0_binary_prediction"
        )
        v0_answer = self._compute_answer_level_metrics(arm_targets, v0_arm_results)
        v1_binary = self._compute_binary_metrics(
            claim_targets, v1_claim_preds, pred_key="v1_binary_prediction"
        )
        d3_binary = self._compute_binary_metrics(
            claim_targets, d3_claim_preds, pred_key="v2_d3_binary_prediction"
        )
        d3_three_way = self._compute_three_way_metrics(
            claim_targets, d3_claim_preds, label_key="v2_d3_three_way_prediction"
        )
        d3_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, d3_claim_preds, pred_key="v2_d3_three_way_prediction"
        )

        return {
            "schema_version": "1.0",
            "artifact_type": "v2_d31_development_benchmark_report",
            "candidate_id": self._candidate_id,
            "verdict": "V2_D31_DEVELOPMENT_BENCHMARK_READY",
            "sources_info": sources_info,
            "total_claims": len(claim_targets),
            "v0_replay_stats": v0_fidelity_stats,
            "model_run_executed": False,
            "v0_baseline_metrics": {
                "v0_claim_binary": v0_binary,
                "v0_answer_level": v0_answer,
            },
            "v1_baseline_metrics": {
                "v1_claim_binary": v1_binary,
            },
            "d3_comparison_metrics": {
                "d3_claim_binary": d3_binary,
                "d3_three_way": d3_three_way,
                "d3_answer_level": d3_answer,
            },
            "execution_identity": exec_identity,
        }

    def _build_reports(
        self,
        sources_info: dict[str, dict[str, Any]],
        exec_identity: dict[str, Any],
        claim_targets: list[BenchmarkClaimTarget],
        arm_targets: list[BenchmarkArmTarget],
        stability_info: dict[str, Any],
        all_metrics: dict[str, Any],
        dim_diagnostics: dict[str, Any],
        gain_preservation_diag: dict[str, Any],
        forensic_groups_diag: dict[str, Any],
        pass1_telemetry: dict[str, Any],
        pass2_telemetry: dict[str, Any],
        total_duration: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        total_provider_calls = (
            pass1_telemetry["provider_calls"] + pass2_telemetry["provider_calls"]
        )
        total_provider_invocation_errors = (
            pass1_telemetry["provider_invocation_errors"]
            + pass2_telemetry["provider_invocation_errors"]
        )
        total_structured_retries = (
            pass1_telemetry["structured_retries"] + pass2_telemetry["structured_retries"]
        )
        total_semantic_execution_errors = (
            pass1_telemetry["semantic_execution_errors"]
            + pass2_telemetry["semantic_execution_errors"]
        )
        model_errors = total_semantic_execution_errors

        d31_claim_binary = all_metrics["v2_d31_claim_binary"]
        d31_three_way = all_metrics["v2_d31_three_way"]
        paired_d31_vs_d3 = all_metrics["paired_v2_d31_vs_v2_d3"]
        contra_diag = all_metrics["contradiction_capability"]
        d31_answer = all_metrics["v2_d31_answer_metrics"]

        # Canonical Verdict Precedence
        if (
            total_semantic_execution_errors > 0
            or stability_info["execution_error_in_any_pass_count"] > 0
            or total_provider_invocation_errors > 0
        ):
            verdict = "V2_D31_DEVELOPMENT_EXECUTION_ERROR"
        elif stability_info["unstable_semantic_claim_count"] > 0:
            verdict = "V2_D31_DEVELOPMENT_LABEL_INSTABILITY"
        else:
            verdict = "V2_D31_DEVELOPMENT_BENCHMARK_PASS"

        # Pre-Registered Selection Gate: D31_SUPERSEDES_D3 vs KEEP_D3
        d31_correct_binary = d31_claim_binary["tp"] + d31_claim_binary["tn"]
        d31_supp_retained = d31_claim_binary["tp"]
        d31_neg_caught = d31_claim_binary["tn"]
        d31_correct_three_way = sum(
            d31_three_way["confusion_matrix"][c][c]
            for c in ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
        )
        d31_correct_answers = d31_answer["valid_answers_retained"] + d31_answer["invalid_answers_caught"]
        net_delta_vs_d3 = paired_d31_vs_d3["net_correctness_delta"]

        mechanical_pass = (
            verdict == "V2_D31_DEVELOPMENT_BENCHMARK_PASS"
            and model_errors == 0
            and total_provider_invocation_errors == 0
            and stability_info["execution_error_in_any_pass_count"] == 0
            and stability_info["unstable_semantic_claim_count"] == 0
            and d31_claim_binary["execution_errors"] == 0
            and d31_three_way["execution_errors"] == 0
            and stability_info["claims_with_two_valid_semantic_labels"] == 38
        )

        quality_pass = (
            d31_correct_binary > 28
            and d31_supp_retained >= 17
            and d31_neg_caught > 11
            and net_delta_vs_d3 > 0
            and d31_correct_three_way > 24
            and contra_diag["d31_correctly_predicted_contradicted_count"] > 0
            and d31_correct_answers >= 14
        )

        d31_supersedes_d3 = mechanical_pass and quality_pass
        dev_decision = "D31_SUPERSEDES_D3" if d31_supersedes_d3 else "KEEP_D3"

        final_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d31_development_benchmark_report",
            "candidate_id": self._candidate_id,
            "verdict": verdict,
            "sources_info": sources_info,
            "total_claims": len(claim_targets),
            "model_run_executed": True,
            "telemetry": {
                "model_errors": model_errors,
                "provider_invocation_errors": total_provider_invocation_errors,
                "structured_output_retries": total_structured_retries,
                "total_provider_calls": total_provider_calls,
                "total_duration_seconds": round(total_duration, 2),
                "pass1": pass1_telemetry,
                "pass2": pass2_telemetry,
                "aggregate": {
                    "total_provider_calls": total_provider_calls,
                    "total_provider_invocation_errors": total_provider_invocation_errors,
                    "total_structured_retries": total_structured_retries,
                    "total_semantic_execution_errors": total_semantic_execution_errors,
                },
            },
            "stability": stability_info,
            "metrics": all_metrics,
            "dimension_diagnostics": dim_diagnostics,
            "gain_preservation_diagnostic": gain_preservation_diag,
            "forensic_groups_diagnostic": forensic_groups_diag,
            "execution_identity": exec_identity,
        }

        decision_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d31_development_decision_report",
            "candidate_id": self._candidate_id,
            "verdict": verdict,
            "development_evaluation_decision": dev_decision,
            "d31_supersedes_d3": d31_supersedes_d3,
            "promotion_authorized": False,  # FAIL-CLOSED: always False during development
            "pre_registered_gate_evaluations": {
                "mechanical_gates_passed": mechanical_pass,
                "verdict_is_pass": verdict == "V2_D31_DEVELOPMENT_BENCHMARK_PASS",
                "model_errors_zero": model_errors == 0,
                "zero_execution_errors_in_stability_passes": stability_info["execution_error_in_any_pass_count"] == 0,
                "zero_unstable_semantic_claims": stability_info["unstable_semantic_claim_count"] == 0,
                "binary_correct_exceeds_d3": bool(d31_correct_binary > 28),
                "supported_retention_preserved": bool(d31_supp_retained >= 17),
                "negative_catch_exceeds_d3": bool(d31_neg_caught > 11),
                "paired_net_delta_vs_d3_positive": bool(net_delta_vs_d3 > 0),
                "three_way_correct_exceeds_d3": bool(d31_correct_three_way > 24),
                "contradiction_caught_gt_zero": bool(contra_diag["d31_correctly_predicted_contradicted_count"] > 0),
                "answer_accuracy_preserved": bool(d31_correct_answers >= 14),
            },
            "metrics_summary": {
                "d31_binary_correct": f"{d31_correct_binary}/38 (D3: 28/38, V1: 23/38)",
                "d31_supported_retained": f"{d31_supp_retained}/18 (D3: 17/18, V1: 16/18)",
                "d31_negative_catch": f"{d31_neg_caught}/20 (D3: 11/20, V1: 7/20)",
                "d31_three_way_correct": f"{d31_correct_three_way}/38 (D3: 24/38, V1: 19/38)",
                "d31_contradicted_caught": f"{contra_diag['d31_correctly_predicted_contradicted_count']}/7 (D3: 0/7, V1: 1/7)",
                "d31_paired_net_delta_vs_d3": net_delta_vs_d3,
                "d31_answer_correct": f"{d31_correct_answers}/22 (D3: 14/22, V1: 14/22)",
                "d3_gains_preserved": f"{gain_preservation_diag['preserved_gain_count']}/7",
            },
        }

        return final_report, decision_report

    def _write_reports(
        self,
        report: dict[str, Any],
        decision_report: dict[str, Any] | None = None,
        dim_diagnostics: dict[str, Any] | None = None,
        v0_claim_preds: list[dict[str, Any]] | None = None,
        v1_claim_preds: list[dict[str, Any]] | None = None,
        d3_claim_preds: list[dict[str, Any]] | None = None,
        pass1_claim_preds: list[dict[str, Any]] | None = None,
        pass2_claim_preds: list[dict[str, Any]] | None = None,
        exec_identity: dict[str, Any] | None = None,
        provider: ObservationalChatModelProviderWrapper | None = None,
        is_preflight: bool = False,
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self._output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        exec_dir = self._output_dir / "execution"
        exec_dir.mkdir(parents=True, exist_ok=True)
        telem_dir = self._output_dir / "telemetry"
        telem_dir.mkdir(parents=True, exist_ok=True)

        if is_preflight:
            (results_dir / "v2_d31_development_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return

        if exec_identity:
            (exec_dir / "v2_d31_development_source_identity.json").write_text(
                json.dumps(exec_identity, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        (results_dir / "v2_d31_development_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if decision_report:
            (results_dir / "v2_d31_development_decision_report.json").write_text(
                json.dumps(decision_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if dim_diagnostics:
            (results_dir / "v2_d31_dimension_diagnostics.json").write_text(
                json.dumps(dim_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if v0_claim_preds:
            with (results_dir / "v0_claim_predictions.jsonl").open("w", encoding="utf-8") as f:
                for r in v0_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if v1_claim_preds:
            with (results_dir / "v1_claim_predictions.jsonl").open("w", encoding="utf-8") as f:
                for r in v1_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if d3_claim_preds:
            with (results_dir / "v2_d3_claim_predictions.jsonl").open("w", encoding="utf-8") as f:
                for r in d3_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if pass1_claim_preds:
            with (results_dir / "v2_d31_claim_predictions_pass1.jsonl").open("w", encoding="utf-8") as f:
                for r in pass1_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if pass2_claim_preds:
            with (results_dir / "v2_d31_claim_predictions_pass2.jsonl").open("w", encoding="utf-8") as f:
                for r in pass2_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if pass1_claim_preds and pass2_claim_preds:
            comparisons = []
            p2_map = {
                (r["question_id"], r["arm_id"], r["claim_id"]): r
                for r in pass2_claim_preds
            }
            for p1 in pass1_claim_preds:
                k = (p1["question_id"], p1["arm_id"], p1["claim_id"])
                p2 = p2_map.get(k, {})
                comparisons.append({
                    "question_id": p1["question_id"],
                    "arm_id": p1["arm_id"],
                    "claim_id": p1["claim_id"],
                    "human_label": p1["human_label"],
                    "pass1_three_way": p1["v2_d31_three_way_prediction"],
                    "pass2_three_way": p2.get("v2_d31_three_way_prediction"),
                    "pass1_is_contradicted": p1.get("is_contradicted"),
                    "pass2_is_contradicted": p2.get("is_contradicted"),
                    "pass1_is_fully_established": p1.get("is_fully_established"),
                    "pass2_is_fully_established": p2.get("is_fully_established"),
                    "stable": p1["v2_d31_three_way_prediction"] == p2.get("v2_d31_three_way_prediction"),
                })
            with (results_dir / "v2_d31_claim_comparisons.jsonl").open("w", encoding="utf-8") as f:
                for r in comparisons:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Telemetry logs
        if provider and hasattr(provider, "call_history"):
            with (telem_dir / "provider_calls.jsonl").open("w", encoding="utf-8") as f:
                for entry in provider.call_history:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Package ZIP if requested
        if self._package_zip:
            self._package_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self._package_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(self._output_dir):
                    for file in files:
                        p = Path(root) / file
                        arcname = p.relative_to(self._output_dir).as_posix()
                        zf.write(p, arcname)
            _LOGGER.info("Wrote benchmark evidence package ZIP: %s", self._package_zip)

    def _sha256_file(self, path: Path) -> str:
        h = sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "unknown"

    def _is_git_worktree_clean(self) -> bool:
        try:
            out = subprocess.check_output(["git", "status", "--short"], text=True).strip()
            return len(out) == 0
        except Exception:
            return False

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _get_cuda_version(self) -> str | None:
        try:
            import torch
            return torch.version.cuda
        except Exception:
            return None

    def _get_cuda_device_name(self) -> str | None:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
            return None
        except Exception:
            return None

    def _get_cuda_device_count(self) -> int:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            return 0
        except Exception:
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2-D3.1 Hierarchical Two-Gate Semantic Verifier Development Benchmark Harness"
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
        help="Path to verification-semantic-benchmark-evidence.zip",
    )
    parser.add_argument(
        "--d3-evidence",
        type=Path,
        required=True,
        help="Path to verification-v2-d3-development-evidence.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write evaluation outputs",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Optional path to write bundled evidence ZIP",
    )
    parser.add_argument(
        "--candidate-id",
        type=str,
        default=CANONICAL_CANDIDATE_ID,
        help="Candidate identity tag",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CANONICAL_V3_DEVICE,
        help="Inference device (cuda/cpu)",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=CANONICAL_REPEAT_COUNT,
        help="Number of evaluation passes for stability",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate source checksums and V0 replay without running model",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=args.forensic_packets,
        forensic_labels_path=args.forensic_labels,
        control_packets_path=args.control_packets,
        control_labels_path=args.control_labels,
        v1_evidence_path=args.v1_evidence,
        d3_evidence_path=args.d3_evidence,
        output_dir=args.output_dir,
        package_zip=args.package_zip,
        candidate_id=args.candidate_id,
        device=args.device,
        repeat_count=args.repeat_count,
        preflight_only=args.preflight_only,
    )

    report = evaluator.evaluate()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
