#!/usr/bin/env python3
"""V2-D3 Structured Semantic Verifier Development Benchmark Harness.

This script executes the controlled offline development benchmark for candidate V2-D3
over the frozen composite 38-claim human-annotated dataset:
- Slice A: 11 claims from suspicious forensic cases (B-FORENSIC-1A)
- Slice B: 27 claims from pre-registered positive-control candidates (B-FORENSIC-1C)
- Baseline V1 predictions loaded directly from canonical evidence archive:
  verification-semantic-benchmark-evidence.zip

Key differences from V2-D2:
- Reduced Semantic Dimensionality: Primary judgment is ONE evidence relation (ENTAILS, CONTRADICTS, DOES_NOT_ESTABLISH).
- 5 Diagnostic Boolean Mismatch Flags (actor, condition/exception, quantity/temporal, negation/modality, source/scope).
- Transparent Deterministic Derivation:
  - ENTAILS -> SUPPORTED
  - CONTRADICTS -> CONTRADICTED
  - DOES_NOT_ESTABLISH -> INSUFFICIENT
- Full Telemetry Preservation: Permanent execution-error claims retain operational telemetry outside structured assessments.
- Unified Aggregate Diagnostics: Rejection categories and retry counts aggregated for both successful and failed claims.
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


class V2D3DevelopmentBenchmarkEvaluator:
    """Evaluates candidate V2-D3 against the composite 38-claim development benchmark."""

    def __init__(
        self,
        *,
        forensic_packets_path: Path,
        forensic_labels_path: Path,
        control_packets_path: Path,
        control_labels_path: Path,
        v1_evidence_path: Path,
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
    ) -> None:
        self._forensic_packets_path = forensic_packets_path
        self._forensic_labels_path = forensic_labels_path
        self._control_packets_path = control_packets_path
        self._control_labels_path = control_labels_path
        self._v1_evidence_path = v1_evidence_path
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

    def evaluate(self) -> dict[str, Any]:
        """Execute full benchmark evaluation workflow with provenance and safety gates."""
        start_time = perf_counter()
        _LOGGER.info("Starting V2-D3 Development Benchmark Evaluation...")

        # 1. Verify inputs and canonical hashes
        sources_info = self._verify_canonical_source_checksums()

        # 2. Package and environment provenance
        exec_identity = self._build_execution_identity(sources_info)

        # 3. Load dataset slices and bind ground truth
        arm_targets, claim_targets = self._load_and_bind_benchmark_targets()
        _LOGGER.info("Bound %d claims across %d arms", len(claim_targets), len(arm_targets))

        # 4. Load canonical V1 predictions from evidence archive
        v1_claim_preds = self._load_v1_baseline_predictions()

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
            )
            self._write_reports(preflight_report, is_preflight=True)
            return preflight_report

        # 6. Initialize V2-D3 verifier provider
        provider = self._init_transformers_provider()
        verifier = StructuredSemanticCitationVerifierD3(
            provider=provider,
            max_structured_output_retries=self._max_structured_output_retries,
        )

        # 7. Execute Pass 1 (Primary Development Evaluation)
        _LOGGER.info("Executing V2-D3 Development Pass 1 (Benchmark Evaluation)...")
        pass1_arm_results, pass1_claim_preds, pass1_telemetry = self._run_inference_pass(
            verifier=verifier,
            provider=provider,
            arm_targets=arm_targets,
            pass_index=1,
        )

        # 8. Execute Pass 2 (Stability Evaluation)
        _LOGGER.info("Executing V2-D3 Development Pass 2 (Two-Pass Stability)...")
        pass2_arm_results, pass2_claim_preds, pass2_telemetry = self._run_inference_pass(
            verifier=verifier,
            provider=provider,
            arm_targets=arm_targets,
            pass_index=2,
        )

        # 9. Stability Analysis
        stability_info = self._evaluate_stability(
            claim_targets=claim_targets,
            pass1_preds=pass1_claim_preds,
            pass2_preds=pass2_claim_preds,
        )

        # 10. Compute Metrics (Pass 1 as authoritative primary evaluation)
        all_metrics = self._compute_all_metrics(
            claim_targets=claim_targets,
            arm_targets=arm_targets,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds=v1_claim_preds,
            v2_claim_preds=pass1_claim_preds,
            v0_arm_results=v0_arm_results,
            v2_arm_results=pass1_arm_results,
        )

        # 11. Compute Dimension Diagnostics & Failure Telemetry
        dim_diagnostics = self._compute_dimension_diagnostics(
            targets=claim_targets,
            v2_preds=pass1_claim_preds,
        )

        # 12. Build Final Evaluation and Decision Reports
        total_duration = perf_counter() - start_time
        final_report, decision_report = self._build_reports(
            sources_info=sources_info,
            exec_identity=exec_identity,
            claim_targets=claim_targets,
            arm_targets=arm_targets,
            stability_info=stability_info,
            all_metrics=all_metrics,
            dim_diagnostics=dim_diagnostics,
            pass1_telemetry=pass1_telemetry,
            pass2_telemetry=pass2_telemetry,
            total_duration=total_duration,
        )

        # 13. Write output artifacts
        self._write_reports(
            final_report,
            decision_report=decision_report,
            dim_diagnostics=dim_diagnostics,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds=v1_claim_preds,
            pass1_claim_preds=pass1_claim_preds,
            pass2_claim_preds=pass2_claim_preds,
            exec_identity=exec_identity,
            provider=provider,
            is_preflight=False,
        )

        _LOGGER.info("V2-D3 Development Benchmark complete. Verdict: %s", final_report["verdict"])
        return final_report

    def _verify_canonical_source_checksums(self) -> dict[str, dict[str, Any]]:
        """Verify the 5 canonical input datasets against frozen SHA-256 signatures."""
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
            STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest()
        schema_sha = sha256(
            json.dumps(
                D3StructuredClaimAssessmentDraft.model_json_schema(),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        return {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_development_execution_identity",
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
                "structured_semantic_verifier_d3_sha256": self._sha256_file(
                    Path(legal_agentic_rag.generation.structured_semantic_verifier_d3.__file__)
                ),
                "evaluate_verification_v2_d3_development_sha256": self._sha256_file(
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
        """Extract canonical V1 baseline predictions directly from the canonical evidence archive."""
        with zipfile.ZipFile(self._v1_evidence_path, "r") as zf:
            if "results/v1_claim_predictions_pass1.jsonl" not in zf.namelist():
                raise DataValidationError("Missing 'results/v1_claim_predictions_pass1.jsonl' in canonical V1 evidence archive")
            content = zf.read("results/v1_claim_predictions_pass1.jsonl").decode("utf-8")

        preds: list[dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                preds.append(json.loads(line))

        if len(preds) != 38:
            raise DataValidationError(f"Canonical V1 archive must contain 38 predictions, got {len(preds)}")
        return preds

    def _replay_v0_baseline(
        self, arm_targets: list[BenchmarkArmTarget]
    ) -> tuple[dict[str, CitationVerificationResult], list[dict[str, Any]], dict[str, Any]]:
        """Replay RuleBasedCitationVerifier (V0) on all benchmark arms."""
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

    def _init_transformers_provider(self) -> TransformersChatProvider:
        """Initialize pinned HuggingFace Transformers provider for Qwen2.5-3B-Instruct."""
        inference_config = SemanticVerificationConfig(
            provider="transformers",
            model=CANONICAL_V3_MODEL_NAME,
            model_revision=CANONICAL_V3_MODEL_REVISION,
            device=self._device,
            torch_dtype=self._torch_dtype,
            temperature=self._temperature,
            max_input_tokens=self._max_input_tokens,
            max_output_tokens=self._max_output_tokens,
            timeout_seconds=self._timeout_seconds,
            local_files_only=False,
        ).as_generation_config()

        claim_config = ClaimVerificationConfig(
            max_structured_output_retries=self._max_structured_output_retries,
        )

        return TransformersChatProvider(
            inference_config=inference_config,
            claim_config=claim_config,
        )

    def _run_inference_pass(
        self,
        verifier: StructuredSemanticCitationVerifierD3,
        provider: TransformersChatProvider,
        arm_targets: list[BenchmarkArmTarget],
        pass_index: int,
    ) -> tuple[dict[str, CitationVerificationResult], list[dict[str, Any]], dict[str, Any]]:
        """Run complete verifier evaluation pass over all benchmark arms."""
        start_calls = getattr(provider, "call_count", 0)
        start_errors = getattr(provider, "error_count", 0)

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
                    rel_val = None
                    actor_mis = cond_mis = qty_mis = neg_mis = src_mis = False
                    assessment_dict = None
                elif assess is not None:
                    bin_pred = (
                        BinaryPrediction.ACCEPT.value
                        if assess.label == SemanticSupportLabel.SUPPORTED
                        else BinaryPrediction.REJECT.value
                    )
                    three_way_pred = assess.label.value.upper()
                    rel_val = assess.relation.value
                    actor_mis = assess.actor_mismatch
                    cond_mis = assess.condition_exception_mismatch
                    qty_mis = assess.quantity_temporal_mismatch
                    neg_mis = assess.negation_modality_mismatch
                    src_mis = assess.source_scope_mismatch
                    assessment_dict = {
                        "claim_id": assess.claim_id,
                        "relation": assess.relation.value,
                        "actor_mismatch": assess.actor_mismatch,
                        "condition_exception_mismatch": assess.condition_exception_mismatch,
                        "quantity_temporal_mismatch": assess.quantity_temporal_mismatch,
                        "negation_modality_mismatch": assess.negation_modality_mismatch,
                        "source_scope_mismatch": assess.source_scope_mismatch,
                        "derived_label": assess.label.value.upper(),
                        "telemetry": telemetry_dict,
                    }
                else:
                    bin_pred = BinaryPrediction.EXECUTION_ERROR.value
                    three_way_pred = "EXECUTION_ERROR"
                    rel_val = None
                    actor_mis = cond_mis = qty_mis = neg_mis = src_mis = False
                    assessment_dict = None

                claim_preds.append({
                    "pass_index": pass_index,
                    "slice_id": claim.slice_id,
                    "question_id": claim.question_id,
                    "arm_id": claim.arm_id,
                    "claim_id": claim.claim_id,
                    "human_label": claim.human_label.value,
                    "error_tags": claim.error_tags,
                    "v2_d3_binary_prediction": bin_pred,
                    "v2_d3_three_way_prediction": three_way_pred,
                    "relation": rel_val,
                    "actor_mismatch": actor_mis,
                    "condition_exception_mismatch": cond_mis,
                    "quantity_temporal_mismatch": qty_mis,
                    "negation_modality_mismatch": neg_mis,
                    "source_scope_mismatch": src_mis,
                    "telemetry": telemetry_dict,
                    "structured_assessment": assessment_dict,
                })

        end_calls = getattr(provider, "call_count", 0)
        end_errors = getattr(provider, "error_count", 0)

        pass_telemetry = {
            "pass_index": pass_index,
            "provider_calls_in_pass": end_calls - start_calls,
            "provider_errors_in_pass": end_errors - start_errors,
        }
        return arm_results, claim_preds, pass_telemetry

    def _evaluate_stability(
        self,
        claim_targets: list[BenchmarkClaimTarget],
        pass1_preds: list[dict[str, Any]],
        pass2_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute two-pass deterministic reproducibility metrics with strict error partitioning."""
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

            l1 = p1.get("v2_d3_three_way_prediction", "EXECUTION_ERROR")
            l2 = p2.get("v2_d3_three_way_prediction", "EXECUTION_ERROR")

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
            claim_targets, v2_claim_preds, pred_key="v2_d3_binary_prediction"
        )

        v1_three_way = self._compute_three_way_metrics(
            claim_targets, v1_claim_preds, label_key="v1_three_way_prediction"
        )
        v2_three_way = self._compute_three_way_metrics(
            claim_targets, v2_claim_preds, label_key="v2_d3_three_way_prediction"
        )

        paired_v1_vs_v2 = self._compute_paired_metrics(
            claim_targets, v1_claim_preds, v2_claim_preds
        )

        v0_answer = self._compute_answer_level_metrics(arm_targets, v0_arm_results)
        v1_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, v1_claim_preds, pred_key="v1_three_way_prediction", supported_val="SUPPORTED"
        )
        v2_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, v2_claim_preds, pred_key="v2_d3_three_way_prediction", supported_val="SUPPORTED"
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
            "v2_d3_claim_binary": v2_binary,
            "v1_three_way": v1_three_way,
            "v2_d3_three_way": v2_three_way,
            "paired_v1_vs_v2_d3": paired_v1_vs_v2,
            "v0_answer_metrics": v0_answer,
            "v1_answer_metrics": v1_answer,
            "v2_d3_answer_metrics": v2_answer,
            "v2_d3_vs_v1_answer_deltas": answer_deltas,
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
        """Compute multi-class three-way entailment metrics (SUPPORTED/CONTRADICTED/INSUFFICIENT)."""
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
        v1_preds: list[dict[str, Any]],
        v2_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute paired claim-level transitions between V1 baseline and candidate."""
        v1_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r["v1_three_way_prediction"]
            for r in v1_preds
        }
        v2_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r["v2_d3_three_way_prediction"]
            for r in v2_preds
        }

        both_corr = v1_only = v2_only = both_wrg = 0
        v2_exec_err = 0
        semantic_regressions = 0
        exec_error_regressions = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            v1_val = v1_map.get(k)
            v2_val = v2_map.get(k)
            gold = t.human_label.value

            v1_c = (v1_val == gold)
            v2_c = (v2_val == gold)

            if v2_val == "EXECUTION_ERROR":
                v2_exec_err += 1

            if v1_c and v2_c:
                both_corr += 1
            elif v1_c and not v2_c:
                v1_only += 1
                if v2_val == "EXECUTION_ERROR":
                    exec_error_regressions += 1
                else:
                    semantic_regressions += 1
            elif not v1_c and v2_c:
                v2_only += 1
            else:
                both_wrg += 1

        net_delta = v2_only - v1_only

        return {
            "both_correct": both_corr,
            "v1_only_correct": v1_only,
            "v2_only_correct": v2_only,
            "both_wrong": both_wrg,
            "net_correctness_delta": net_delta,
            "v2_fixes_count": v2_only,
            "v2_regressions_count": v1_only,
            "semantic_regressions_count": semantic_regressions,
            "execution_error_regressions_count": exec_error_regressions,
            "v2_execution_error_count": v2_exec_err,
        }

    def _compute_answer_level_metrics(
        self,
        arm_targets: list[BenchmarkArmTarget],
        arm_results: dict[str, CitationVerificationResult],
    ) -> dict[str, Any]:
        """Compute answer-level validity retention and invalid catch rates from verified arm results."""
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
        """Compute answer-level metrics derived directly from individual claim predictions."""
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
        v2_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute D3 relation distributions, diagnostic mismatch flags, and unified failure telemetry."""
        pred_map = {
            (r["question_id"], r["arm_id"], r["claim_id"]): r
            for r in v2_preds
        }

        relation_counts = Counter()
        diagnostic_flag_counts = {
            "actor_mismatch": 0,
            "condition_exception_mismatch": 0,
            "quantity_temporal_mismatch": 0,
            "negation_modality_mismatch": 0,
            "source_scope_mismatch": 0,
        }

        rejection_categories = Counter()
        total_retries = 0
        claims_with_exec_error = 0
        successfully_structured_claims = 0

        for t in targets:
            k = (t.question_id, t.arm_id, t.claim_id)
            pred = pred_map.get(k) or {}

            # Read operational telemetry across ALL claims (both success and permanent failure)
            telemetry = pred.get("telemetry") or {}
            if not telemetry and pred.get("structured_assessment"):
                telemetry = pred["structured_assessment"].get("telemetry") or {}

            total_retries += telemetry.get("retry_count", 0)
            for cat in telemetry.get("draft_rejection_categories", []):
                rejection_categories[cat] += 1

            is_exec_err = (
                telemetry.get("semantic_execution_error")
                or pred.get("v2_d3_binary_prediction") == BinaryPrediction.EXECUTION_ERROR.value
                or not pred.get("structured_assessment")
            )

            if is_exec_err:
                claims_with_exec_error += 1
                continue

            successfully_structured_claims += 1
            sa = pred["structured_assessment"]
            rel = sa.get("relation", "UNKNOWN")
            relation_counts[rel] += 1

            for flag in (
                "actor_mismatch",
                "condition_exception_mismatch",
                "quantity_temporal_mismatch",
                "negation_modality_mismatch",
                "source_scope_mismatch",
            ):
                if sa.get(flag) is True:
                    diagnostic_flag_counts[flag] += 1

        return {
            "total_claims": len(targets),
            "successfully_structured_claim_count": successfully_structured_claims,
            "execution_error_claim_count": claims_with_exec_error,
            "relation_distribution": dict(relation_counts),
            "diagnostic_mismatch_flag_counts": diagnostic_flag_counts,
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
    ) -> dict[str, Any]:
        """Build lightweight report for preflight verification mode."""
        v0_binary = self._compute_binary_metrics(
            claim_targets, v0_claim_preds, pred_key="v0_binary_prediction"
        )
        v0_answer = self._compute_answer_level_metrics(arm_targets, v0_arm_results)
        v1_binary = self._compute_binary_metrics(
            claim_targets, v1_claim_preds, pred_key="v1_binary_prediction"
        )
        v1_three_way = self._compute_three_way_metrics(
            claim_targets, v1_claim_preds, label_key="v1_three_way_prediction"
        )
        v1_answer = self._compute_answer_level_metrics_from_preds(
            arm_targets, v1_claim_preds, pred_key="v1_three_way_prediction"
        )

        return {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_development_benchmark_report",
            "candidate_id": self._candidate_id,
            "verdict": "V2_DEVELOPMENT_BENCHMARK_READY",
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
                "v1_three_way": v1_three_way,
                "v1_answer_level": v1_answer,
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
        pass1_telemetry: dict[str, Any],
        pass2_telemetry: dict[str, Any],
        total_duration: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Construct final comprehensive benchmark evaluation report and decision report."""
        total_calls = (
            pass1_telemetry["provider_calls_in_pass"] + pass2_telemetry["provider_calls_in_pass"]
        )
        model_errors = (
            pass1_telemetry["provider_errors_in_pass"] + pass2_telemetry["provider_errors_in_pass"]
        )

        v2_claim_binary = all_metrics["v2_d3_claim_binary"]
        v2_three_way = all_metrics["v2_d3_three_way"]
        paired = all_metrics["paired_v1_vs_v2_d3"]
        v2_answer = all_metrics["v2_d3_answer_metrics"]

        # Canonical Verdict Precedence
        if model_errors > 0 or stability_info["execution_error_in_any_pass_count"] > 0:
            verdict = "V2_DEVELOPMENT_EXECUTION_ERROR"
        elif stability_info["unstable_semantic_claim_count"] > 0:
            verdict = "V2_DEVELOPMENT_LABEL_INSTABILITY"
        else:
            verdict = "V2_DEVELOPMENT_BENCHMARK_PASS"

        # Freeze Gate Evaluation
        v2_correct = v2_claim_binary["tp"] + v2_claim_binary["tn"]
        v2_neg_caught = v2_claim_binary["tn"]
        v2_supp_retained = v2_claim_binary["tp"]
        net_delta = paired["net_correctness_delta"]

        is_freeze_eligible = (
            verdict == "V2_DEVELOPMENT_BENCHMARK_PASS"
            and model_errors == 0
            and stability_info["execution_error_in_any_pass_count"] == 0
            and stability_info["unstable_semantic_claim_count"] == 0
            and v2_claim_binary["execution_errors"] == 0
            and v2_three_way["execution_errors"] == 0
            and v2_correct > 23
            and v2_neg_caught > 7
            and v2_supp_retained >= 16
            and net_delta > 0
        )

        dev_decision = "CANDIDATE_FREEZE_ELIGIBLE" if is_freeze_eligible else "KEEP_ITERATING"

        final_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_development_benchmark_report",
            "candidate_id": self._candidate_id,
            "verdict": verdict,
            "sources_info": sources_info,
            "total_claims": len(claim_targets),
            "model_run_executed": True,
            "telemetry": {
                "provider_calls": total_calls,
                "model_errors": model_errors,
                "structured_output_retries": dim_diagnostics["rejection_telemetry_summary"]["total_retries"],
                "total_duration_seconds": round(total_duration, 2),
                "pass1_telemetry": pass1_telemetry,
                "pass2_telemetry": pass2_telemetry,
            },
            "stability": stability_info,
            "metrics": all_metrics,
            "dimension_diagnostics": dim_diagnostics,
            "execution_identity": exec_identity,
        }

        decision_report = {
            "schema_version": "1.0",
            "artifact_type": "v2_d3_development_decision_report",
            "candidate_id": self._candidate_id,
            "verdict": verdict,
            "development_evaluation_decision": dev_decision,
            "freeze_eligible": is_freeze_eligible,
            "promotion_authorized": False,  # FAIL-CLOSED: always False during development
            "decision_reasons": {
                "verdict_is_pass": verdict == "V2_DEVELOPMENT_BENCHMARK_PASS",
                "model_errors_zero": model_errors == 0,
                "zero_execution_errors_in_stability_passes": stability_info["execution_error_in_any_pass_count"] == 0,
                "zero_unstable_semantic_claims": stability_info["unstable_semantic_claim_count"] == 0,
                "zero_binary_execution_errors": v2_claim_binary["execution_errors"] == 0,
                "zero_three_way_execution_errors": v2_three_way["execution_errors"] == 0,
                "total_correct_exceeds_v1": bool(v2_correct > 23),
                "negative_catch_exceeds_v1": bool(v2_neg_caught > 7),
                "supported_retention_retained": bool(v2_supp_retained >= 16),
                "paired_net_delta_positive": bool(net_delta > 0),
            },
            "metrics_summary": {
                "v2_correct_claims": f"{v2_correct}/38 (V1: 23/38)",
                "v2_negative_catch": f"{v2_neg_caught}/20 (V1: 7/20)",
                "v2_supported_retention": f"{v2_supp_retained}/18 (V1: 16/18)",
                "paired_net_delta": net_delta,
                "answer_accuracy": f"{v2_answer['answer_level_accuracy']*100:.2f}% (V1: 63.64%)",
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
        pass1_claim_preds: list[dict[str, Any]] | None = None,
        pass2_claim_preds: list[dict[str, Any]] | None = None,
        exec_identity: dict[str, Any] | None = None,
        provider: TransformersChatProvider | None = None,
        is_preflight: bool = False,
    ) -> None:
        """Write all JSON reports, prediction JSONLs, and evidence package ZIP."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self._output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        exec_dir = self._output_dir / "execution"
        exec_dir.mkdir(parents=True, exist_ok=True)
        telem_dir = self._output_dir / "telemetry"
        telem_dir.mkdir(parents=True, exist_ok=True)

        if is_preflight:
            (results_dir / "v2_d3_development_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return

        if exec_identity:
            (exec_dir / "v2_d3_development_source_identity.json").write_text(
                json.dumps(exec_identity, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        (results_dir / "v2_d3_development_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if decision_report:
            (results_dir / "v2_d3_development_decision_report.json").write_text(
                json.dumps(decision_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if dim_diagnostics:
            (results_dir / "v2_d3_dimension_diagnostics.json").write_text(
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

        if pass1_claim_preds:
            with (results_dir / "v2_d3_claim_predictions_pass1.jsonl").open("w", encoding="utf-8") as f:
                for r in pass1_claim_preds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if pass2_claim_preds:
            with (results_dir / "v2_d3_claim_predictions_pass2.jsonl").open("w", encoding="utf-8") as f:
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
                    "pass1_three_way": p1["v2_d3_three_way_prediction"],
                    "pass2_three_way": p2.get("v2_d3_three_way_prediction"),
                    "pass1_relation": p1.get("relation"),
                    "pass2_relation": p2.get("relation"),
                    "stable": p1["v2_d3_three_way_prediction"] == p2.get("v2_d3_three_way_prediction"),
                })
            with (results_dir / "v2_d3_claim_comparisons.jsonl").open("w", encoding="utf-8") as f:
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
        description="V2-D3 Structured Semantic Verifier Development Benchmark Harness"
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

    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=args.forensic_packets,
        forensic_labels_path=args.forensic_labels,
        control_packets_path=args.control_packets,
        control_labels_path=args.control_labels,
        v1_evidence_path=args.v1_evidence,
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
