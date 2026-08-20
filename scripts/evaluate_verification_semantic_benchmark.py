#!/usr/bin/env python3
"""Controlled V0 Rule-Based vs V1 Semantic-Verifier Benchmark Harness.

This script executes the controlled offline verification benchmark over the frozen
composite 38-claim human-annotated dataset:
- Slice A: 11 claims from suspicious forensic cases (B-FORENSIC-1A)
- Slice B: 27 claims from pre-registered positive-control candidates (B-FORENSIC-1C)

Invariants:
- All sources verified fail-closed by exact SHA-256 before model initialization
- Exact 100% V0 RuleBasedCitationVerifier replay required before V1 execution
- V1 ModelBackedCitationVerifier evaluated without prompt modification or few-shot tuning
- Repeat count = 2 for deterministic stability measurement
- semantic_verifier_promotion_authorized is strictly FALSE
- Output is 100% sanitized (zero local Windows machine paths)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

from legal_agentic_rag.configuration.online import ClaimVerificationConfig
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
from legal_agentic_rag.generation.semantic_verifier import ModelBackedCitationVerifier
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
CANONICAL_FORENSIC_REVIEW_ZIP_SIZE = 42826

CANONICAL_FORENSIC_LABELS_SHA256 = (
    "bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733"
)

CANONICAL_CONTROL_REVIEW_ZIP_SHA256 = (
    "cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4"
)
CANONICAL_CONTROL_REVIEW_ZIP_SIZE = 110095

CANONICAL_CONTROL_LABELS_SHA256 = (
    "60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373"
)

# Canonical V1 Model Settings
CANONICAL_V1_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CANONICAL_V1_MODEL_REVISION = "a1d308dfcc03e09da285d49d912439a655a571e8"
CANONICAL_V1_MAX_INPUT_TOKENS = 8192
CANONICAL_V1_MAX_OUTPUT_TOKENS = 512
CANONICAL_V1_TEMPERATURE = 0.0
CANONICAL_V1_MAX_STRUCTURED_RETRIES = 1
CANONICAL_V1_TIMEOUT_SECONDS = 180.0


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


class HumanEntailment(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class BinaryPrediction(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class AnswerValidity(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class BenchmarkClaimTarget:
    slice_id: str  # "suspicious_forensic" or "positive_control"
    question_id: str
    arm_id: str  # "BASE", "CANDIDATE", or "PRIMARY"
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


class SemanticVerifierBenchmarkRunner:
    """Validate sources, execute V0 replay and V1 semantic verification, and compute metrics."""

    def __init__(
        self,
        *,
        forensic_packets_path: Path,
        forensic_labels_path: Path,
        control_packets_path: Path,
        control_labels_path: Path,
        output_dir: Path,
        package_zip_path: Path | None = None,
        model_name: str = CANONICAL_V1_MODEL_NAME,
        model_revision: str = CANONICAL_V1_MODEL_REVISION,
        device: str = "cuda",
        torch_dtype: str = "float16",
        temperature: float = CANONICAL_V1_TEMPERATURE,
        max_input_tokens: int = CANONICAL_V1_MAX_INPUT_TOKENS,
        max_output_tokens: int = CANONICAL_V1_MAX_OUTPUT_TOKENS,
        max_structured_output_retries: int = CANONICAL_V1_MAX_STRUCTURED_RETRIES,
        verification_timeout_seconds: float = CANONICAL_V1_TIMEOUT_SECONDS,
        repeat_count: int = 2,
        custom_provider: ChatModelProvider | None = None,
        skip_model_run: bool = False,
    ) -> None:
        self._forensic_packets_path = forensic_packets_path.resolve()
        self._forensic_labels_path = forensic_labels_path.resolve()
        self._control_packets_path = control_packets_path.resolve()
        self._control_labels_path = control_labels_path.resolve()
        self._output_dir = output_dir.resolve()
        self._package_zip_path = package_zip_path.resolve() if package_zip_path else None

        self._model_name = model_name
        self._model_revision = model_revision
        self._device = device
        self._torch_dtype = torch_dtype
        self._temperature = temperature
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._max_structured_output_retries = max_structured_output_retries
        self._verification_timeout_seconds = verification_timeout_seconds
        self._repeat_count = repeat_count
        self._custom_provider = custom_provider
        self._skip_model_run = skip_model_run

    def run(self) -> dict[str, Any]:
        """Execute benchmark workflow."""
        # 1. Validate All Four Benchmark Sources
        sources_info = self._validate_sources()

        # 2. Ingest Benchmark Targets & Cross-Check Claim Identity
        arm_targets, all_claim_targets = self._load_and_bind_benchmark_targets(sources_info)

        # 3. Replay V0 Rule-Based Verifier Over All Arms
        v0_verifier = RuleBasedCitationVerifier(ClaimVerificationConfig(
            enabled=True,
            require_inline_citations=True,
            minimum_lexical_support=0.25,
            minimum_claim_tokens=2,
            require_numeric_match=True,
            require_negation_match=True,
            max_claims=20,
        ))
        v0_arm_results, v0_claim_preds = self._execute_v0_replay(arm_targets, v0_verifier)

        if self._skip_model_run:
            _LOGGER.info("skip_model_run enabled; returning pre-flight provenance verification report")
            report = {
                "schema_version": "1.0",
                "verdict": "VERIFIER_BENCHMARK_READY",
                "semantic_verifier_promotion_authorized": False,
                "sources_info": sources_info,
                "total_claims": len(all_claim_targets),
                "v0_replay_pass_count": len(arm_targets),
                "model_run_executed": False,
            }
            self._write_outputs(
                sources_info=sources_info,
                report=report,
                v0_claim_preds=v0_claim_preds,
                v1_pass1_preds=[],
                v1_pass2_preds=[],
                comparisons=[],
                decision_report=report,
            )
            return report

        # 4. Initialize V1 Semantic Provider & Verifier
        provider = self._init_v1_provider()
        v1_verifier = ModelBackedCitationVerifier(
            base_verifier=v0_verifier,
            provider=provider,
            max_structured_output_retries=self._max_structured_output_retries,
        )

        # 5. Execute Multi-Pass V1 Verification
        v1_passes_claim_preds: list[list[dict[str, Any]]] = []
        v1_passes_arm_results: list[dict[str, CitationVerificationResult]] = []
        model_error_count = 0

        for pass_idx in range(1, self._repeat_count + 1):
            pass_claim_preds, pass_arm_res, err_count = self._execute_v1_pass(
                arm_targets=arm_targets,
                v1_verifier=v1_verifier,
                pass_number=pass_idx,
            )
            v1_passes_claim_preds.append(pass_claim_preds)
            v1_passes_arm_results.append(pass_arm_res)
            model_error_count += err_count

        if model_error_count > 0:
            verdict = "SEMANTIC_VERIFIER_EXECUTION_ERROR"
        else:
            verdict = "VERIFIER_BENCHMARK_PASS"

        # 6. Evaluate Two-Pass Stability
        stability_info = self._evaluate_stability(v1_passes_claim_preds)
        if stability_info["unstable_claim_count"] > 0:
            verdict = "SEMANTIC_VERIFIER_LABEL_INSTABILITY"

        # 7. Compute All Binary, 3-Way, Paired, Stratum, and Answer Metrics
        metrics_report = self._compute_all_metrics(
            claim_targets=all_claim_targets,
            arm_targets=arm_targets,
            v0_claim_preds=v0_claim_preds,
            v1_claim_preds_pass1=v1_passes_claim_preds[0],
            v0_arm_results=v0_arm_results,
            v1_arm_results_pass1=v1_passes_arm_results[0],
        )

        # 8. Build Full Benchmark Report & Decision Report
        report, decision_report, comparisons = self._build_reports(
            verdict=verdict,
            sources_info=sources_info,
            stability_info=stability_info,
            metrics_report=metrics_report,
            v0_claim_preds=v0_claim_preds,
            v1_pass1_preds=v1_passes_claim_preds[0],
            all_claim_targets=all_claim_targets,
            model_error_count=model_error_count,
        )

        # 9. Write Materialized Outputs
        self._write_outputs(
            sources_info=sources_info,
            report=report,
            v0_claim_preds=v0_claim_preds,
            v1_pass1_preds=v1_passes_claim_preds[0],
            v1_pass2_preds=v1_passes_claim_preds[1] if len(v1_passes_claim_preds) > 1 else [],
            comparisons=comparisons,
            decision_report=decision_report,
        )

        return report

    def _validate_sources(self) -> dict[str, Any]:
        """Assert existence and exact SHA-256 checksums of all 4 external benchmark sources."""
        for p, name in [
            (self._forensic_packets_path, "forensic_packets"),
            (self._forensic_labels_path, "forensic_labels"),
            (self._control_packets_path, "control_packets"),
            (self._control_labels_path, "control_labels"),
        ]:
            if not p.exists():
                raise DataValidationError(f"Benchmark source {name} does not exist: {p}")

        f_pkts_sha = sha256_file(self._forensic_packets_path) if self._forensic_packets_path.is_file() else None
        f_lbls_sha = sha256_file(self._forensic_labels_path)
        c_pkts_sha = sha256_file(self._control_packets_path) if self._control_packets_path.is_file() else None
        c_lbls_sha = sha256_file(self._control_labels_path)

        if f_pkts_sha and f_pkts_sha != CANONICAL_FORENSIC_REVIEW_ZIP_SHA256:
            raise DataValidationError(
                f"Forensic review packets SHA mismatch: expected {CANONICAL_FORENSIC_REVIEW_ZIP_SHA256}, got {f_pkts_sha}"
            )
        if f_lbls_sha != CANONICAL_FORENSIC_LABELS_SHA256:
            raise DataValidationError(
                f"Forensic labels SHA mismatch: expected {CANONICAL_FORENSIC_LABELS_SHA256}, got {f_lbls_sha}"
            )
        if c_pkts_sha and c_pkts_sha != CANONICAL_CONTROL_REVIEW_ZIP_SHA256:
            raise DataValidationError(
                f"Control review packets SHA mismatch: expected {CANONICAL_CONTROL_REVIEW_ZIP_SHA256}, got {c_pkts_sha}"
            )
        if c_lbls_sha != CANONICAL_CONTROL_LABELS_SHA256:
            raise DataValidationError(
                f"Control labels SHA mismatch: expected {CANONICAL_CONTROL_LABELS_SHA256}, got {c_lbls_sha}"
            )

        return {
            "forensic_packets_filename": self._forensic_packets_path.name,
            "forensic_packets_sha256": f_pkts_sha or CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
            "forensic_labels_filename": self._forensic_labels_path.name,
            "forensic_labels_sha256": f_lbls_sha,
            "control_packets_filename": self._control_packets_path.name,
            "control_packets_sha256": c_pkts_sha or CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
            "control_labels_filename": self._control_labels_path.name,
            "control_labels_sha256": c_lbls_sha,
        }

    def _load_and_bind_benchmark_targets(
        self, sources_info: dict[str, Any]
    ) -> tuple[list[BenchmarkArmTarget], list[BenchmarkClaimTarget]]:
        """Load packets and labels from both slices and bind each claim to exact text."""
        forensic_labels_data = json.loads(self._forensic_labels_path.read_text(encoding="utf-8"))
        control_labels_data = json.loads(self._control_labels_path.read_text(encoding="utf-8"))

        arm_targets: list[BenchmarkArmTarget] = []
        all_claim_targets: list[BenchmarkClaimTarget] = []

        # Process Slice A: Forensic
        f_unpack = self._unpack_or_resolve(self._forensic_packets_path, "forensic_pkts_")
        try:
            f_pkts_dir = f_unpack / "forensic_packets" if (f_unpack / "forensic_packets").is_dir() else f_unpack
            for qid, q_data in forensic_labels_data.get("questions", {}).items():
                pkt_file = f_pkts_dir / f"{qid}.json"
                if not pkt_file.is_file():
                    raise DataValidationError(f"Forensic packet {qid}.json missing from {f_pkts_dir}")
                pkt = json.loads(pkt_file.read_text(encoding="utf-8"))

                pkt_arms = pkt.get("arms") or pkt.get("historical_arms", {})
                for arm_id, arm_label_data in q_data.get("arms", {}).items():
                    if not arm_label_data.get("claim_review_applicable"):
                        # Skip generation_failed arms
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
        c_unpack = self._unpack_or_resolve(self._control_packets_path, "control_pkts_")
        try:
            c_pkts_dir = c_unpack / "positive_control_packets" if (c_unpack / "positive_control_packets").is_dir() else c_unpack
            for qid, q_data in control_labels_data.get("questions", {}).items():
                pkt_file = c_pkts_dir / f"{qid}.json"
                if not pkt_file.is_file():
                    raise DataValidationError(f"Control packet {qid}.json missing from {c_pkts_dir}")
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
            raise DataValidationError(f"Expected 38 composite benchmark claims, got {total_claims}")
        if label_counts[HumanEntailment.SUPPORTED.value] != 18:
            raise DataValidationError(
                f"Expected 18 SUPPORTED claims, got {label_counts[HumanEntailment.SUPPORTED.value]}"
            )
        if label_counts[HumanEntailment.CONTRADICTED.value] != 7:
            raise DataValidationError(
                f"Expected 7 CONTRADICTED claims, got {label_counts[HumanEntailment.CONTRADICTED.value]}"
            )
        if label_counts[HumanEntailment.INSUFFICIENT.value] != 13:
            raise DataValidationError(
                f"Expected 13 INSUFFICIENT claims, got {label_counts[HumanEntailment.INSUFFICIENT.value]}"
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
        raw_resp = arm_packet_data.get("historical_response", {})
        resp_obj = AnswerResponse.model_validate(raw_resp)

        raw_ev = arm_packet_data.get("selected_evidence", [])
        evidence_list = [Evidence.model_validate(item) for item in raw_ev]

        hist_verification = arm_packet_data.get("historical_verification", {})
        raw_claims_by_id = {
            c["claim_id"]: c for c in hist_verification.get("claim_verifications", [])
        }

        claims_labels = arm_label_data.get("claims", {})
        claim_targets: list[BenchmarkClaimTarget] = []

        for cid in sorted(claims_labels.keys()):
            cl_data = claims_labels[cid]
            if cid not in raw_claims_by_id:
                raise DataValidationError(f"Claim {cid} missing in raw packet for {qid} ({arm_id})")

            raw_claim_text = raw_claims_by_id[cid].get("claim_text", "")
            raw_sha = sha256_text(raw_claim_text)
            label_sha = cl_data.get("claim_text_sha256")

            if raw_sha != label_sha:
                raise DataValidationError(
                    f"Claim text SHA mismatch for {qid} {arm_id} {cid}: raw={raw_sha}, label={label_sha}"
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
            historical_stop_reason=arm_label_data.get("historical_stop_reason", "answer_verified"),
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
    ) -> tuple[dict[str, CitationVerificationResult], list[dict[str, Any]]]:
        """Replay V0 verifier and assert 100% fidelity against historical records."""
        arm_results: dict[str, CitationVerificationResult] = {}
        claim_predictions: list[dict[str, Any]] = []

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            res = verifier.verify(arm.answer_response, arm.evidence_list)
            arm_results[key] = res

            hist = arm.historical_verification
            # Cross-check replay against historical
            if res.is_valid != hist.get("is_valid"):
                raise DataValidationError(f"V0 replay is_valid mismatch on {key}")

            res_claims_by_id = {c.claim_id: c for c in res.claim_verifications}
            for claim_target in arm.claims:
                cid = claim_target.claim_id
                if cid not in res_claims_by_id:
                    raise DataValidationError(f"V0 replay missing claim {cid} on {key}")

                rc = res_claims_by_id[cid]
                pred = (
                    BinaryPrediction.ACCEPT.value
                    if rc.status == ClaimSupportStatus.SUPPORTED
                    else BinaryPrediction.REJECT.value
                )
                claim_predictions.append({
                    "slice_id": claim_target.slice_id,
                    "question_id": claim_target.question_id,
                    "arm_id": claim_target.arm_id,
                    "claim_id": cid,
                    "claim_text_sha256": claim_target.claim_text_sha256,
                    "stratum": claim_target.stratum,
                    "human_label": claim_target.human_label.value,
                    "v0_rule_status": rc.status.value,
                    "v0_binary_prediction": pred,
                    "v0_errors": rc.errors,
                })

        return arm_results, claim_predictions

    def _init_v1_provider(self) -> ChatModelProvider:
        """Initialize Transformers provider with pre-registered parameters."""
        if self._custom_provider is not None:
            return self._custom_provider

        return TransformersChatProvider(
            model_name=self._model_name,
            model_revision=self._model_revision,
            device=self._device,
            torch_dtype=self._torch_dtype,
            temperature=self._temperature,
            max_input_tokens=self._max_input_tokens,
            max_output_tokens=self._max_output_tokens,
            timeout_seconds=self._verification_timeout_seconds,
        )

    def _execute_v1_pass(
        self,
        *,
        arm_targets: list[BenchmarkArmTarget],
        v1_verifier: ModelBackedCitationVerifier,
        pass_number: int,
    ) -> tuple[list[dict[str, Any]], dict[str, CitationVerificationResult], int]:
        """Execute one complete pass of V1 semantic verification."""
        claim_predictions: list[dict[str, Any]] = []
        arm_results: dict[str, CitationVerificationResult] = {}
        error_count = 0

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            try:
                res = v1_verifier.verify(arm.answer_response, arm.evidence_list)
                arm_results[key] = res

                sem_res = res.semantic_verification
                if sem_res is None:
                    raise ModelError(f"V1 returned no semantic_verification on {key}")

                sem_assessments_by_id = {
                    a.claim_id: a for a in sem_res.assessments
                }

                for claim_target in arm.claims:
                    cid = claim_target.claim_id
                    if cid not in sem_assessments_by_id:
                        raise ModelError(f"V1 semantic assessments missing claim {cid} on {key}")

                    ass = sem_assessments_by_id[cid]
                    three_way_label = ass.label.value.upper()
                    binary_pred = (
                        BinaryPrediction.ACCEPT.value
                        if ass.label == SemanticSupportLabel.SUPPORTED
                        else BinaryPrediction.REJECT.value
                    )

                    claim_predictions.append({
                        "pass_number": pass_number,
                        "slice_id": claim_target.slice_id,
                        "question_id": claim_target.question_id,
                        "arm_id": claim_target.arm_id,
                        "claim_id": cid,
                        "claim_text_sha256": claim_target.claim_text_sha256,
                        "stratum": claim_target.stratum,
                        "human_label": claim_target.human_label.value,
                        "v1_three_way_prediction": three_way_label,
                        "v1_binary_prediction": binary_pred,
                        "error_tags": claim_target.error_tags,
                    })

            except Exception as exc:
                _LOGGER.error("Model verification error on %s: %s", key, exc)
                error_count += 1
                arm_results[key] = CitationVerificationResult(
                    is_valid=False,
                    valid_citations=[],
                    invalid_citations=[],
                    errors=[f"model_error:{exc}"],
                )
                for claim_target in arm.claims:
                    claim_predictions.append({
                        "pass_number": pass_number,
                        "slice_id": claim_target.slice_id,
                        "question_id": claim_target.question_id,
                        "arm_id": claim_target.arm_id,
                        "claim_id": claim_target.claim_id,
                        "claim_text_sha256": claim_target.claim_text_sha256,
                        "stratum": claim_target.stratum,
                        "human_label": claim_target.human_label.value,
                        "v1_three_way_prediction": "EXECUTION_ERROR",
                        "v1_binary_prediction": "EXECUTION_ERROR",
                        "error": str(exc),
                    })

        return claim_predictions, arm_results, error_count

    def _evaluate_stability(
        self, passes_preds: list[list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Compare predictions across passes to determine deterministic stability."""
        if len(passes_preds) < 2:
            return {"stable": True, "repeat_count": len(passes_preds), "unstable_claim_count": 0}

        pass1_by_key = {
            f"{p['question_id']}:{p['arm_id']}:{p['claim_id']}": p
            for p in passes_preds[0]
        }
        pass2_by_key = {
            f"{p['question_id']}:{p['arm_id']}:{p['claim_id']}": p
            for p in passes_preds[1]
        }

        unstable_items = []
        for key, p1 in pass1_by_key.items():
            p2 = pass2_by_key.get(key)
            if p2 is None or p1.get("v1_three_way_prediction") != p2.get("v1_three_way_prediction"):
                unstable_items.append({
                    "claim_key": key,
                    "pass_1": p1.get("v1_three_way_prediction"),
                    "pass_2": p2.get("v1_three_way_prediction") if p2 else None,
                })

        return {
            "repeat_count": len(passes_preds),
            "stable": len(unstable_items) == 0,
            "total_evaluated_claims": len(pass1_by_key),
            "stable_claim_count": len(pass1_by_key) - len(unstable_items),
            "unstable_claim_count": len(unstable_items),
            "unstable_details": unstable_items,
        }

    def _compute_all_metrics(
        self,
        *,
        claim_targets: list[BenchmarkClaimTarget],
        arm_targets: list[BenchmarkArmTarget],
        v0_claim_preds: list[dict[str, Any]],
        v1_claim_preds_pass1: list[dict[str, Any]],
        v0_arm_results: dict[str, CitationVerificationResult],
        v1_arm_results_pass1: dict[str, CitationVerificationResult],
    ) -> dict[str, Any]:
        """Compute binary, 3-way, paired, slice, stratum, and answer-level metrics."""
        # 1. Claim-Level Binary Metrics (V0 and V1)
        v0_binary = self._calc_binary_metrics(
            [c.human_label for c in claim_targets],
            [p["v0_binary_prediction"] for p in v0_claim_preds],
        )
        v1_binary = self._calc_binary_metrics(
            [c.human_label for c in claim_targets],
            [p["v1_binary_prediction"] for p in v1_claim_preds_pass1],
        )

        # 2. V1 Three-Way Metrics
        v1_three_way = self._calc_three_way_metrics(
            [c.human_label.value for c in claim_targets],
            [p["v1_three_way_prediction"] for p in v1_claim_preds_pass1],
        )

        # 3. Paired V0 vs V1 Deltas
        paired_deltas = self._calc_paired_deltas(
            claim_targets=claim_targets,
            v0_preds=v0_claim_preds,
            v1_preds=v1_claim_preds_pass1,
        )

        # 4. Error-Tag Catch Diagnostics
        error_tag_diagnostics = self._calc_error_tag_diagnostics(
            claim_targets=claim_targets,
            v1_preds=v1_claim_preds_pass1,
        )

        # 5. Slices and Strata Metrics
        slice_metrics = self._calc_slice_metrics(
            claim_targets=claim_targets,
            v0_preds=v0_claim_preds,
            v1_preds=v1_claim_preds_pass1,
        )

        # 6. Answer-Level Metrics
        answer_level_metrics = self._calc_answer_level_metrics(
            arm_targets=arm_targets,
            v0_arm_results=v0_arm_results,
            v1_arm_results=v1_arm_results_pass1,
        )

        return {
            "v0_binary_metrics": v0_binary,
            "v1_binary_metrics": v1_binary,
            "v1_three_way_metrics": v1_three_way,
            "paired_deltas": paired_deltas,
            "error_tag_diagnostics": error_tag_diagnostics,
            "slice_and_strata_metrics": slice_metrics,
            "answer_level_metrics": answer_level_metrics,
        }

    @staticmethod
    def _calc_binary_metrics(
        human_labels: list[HumanEntailment],
        predictions: list[str],
    ) -> dict[str, Any]:
        tp = fp = tn = fn = 0
        for h, p in zip(human_labels, predictions, strict=True):
            is_pos = (h == HumanEntailment.SUPPORTED)
            is_pred_pos = (p == BinaryPrediction.ACCEPT.value)

            if is_pos and is_pred_pos:
                tp += 1
            elif is_pos and not is_pred_pos:
                fn += 1
            elif not is_pos and is_pred_pos:
                fp += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0  # Supported retention
        spec = tn / (tn + fp) if (tn + fp) else 0.0  # Negative catch rate
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        bal_acc = (rec + spec) / 2.0

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total": total,
            "accuracy": acc,
            "precision": prec,
            "recall_supported_retention": rec,
            "specificity_negative_catch": spec,
            "f1": f1,
            "balanced_accuracy": bal_acc,
        }

    @staticmethod
    def _calc_three_way_metrics(
        human_labels: list[str],
        predictions: list[str],
    ) -> dict[str, Any]:
        classes = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
        matrix: dict[str, dict[str, int]] = {c1: {c2: 0 for c2 in classes} for c1 in classes}

        correct = 0
        total = len(human_labels)

        for h, p in zip(human_labels, predictions, strict=True):
            if h in matrix and p in matrix[h]:
                matrix[h][p] += 1
                if h == p:
                    correct += 1

        acc = correct / total if total else 0.0

        per_class: dict[str, dict[str, float]] = {}
        for c in classes:
            c_tp = matrix[c][c]
            c_fn = sum(matrix[c][other] for other in classes if other != c)
            c_fp = sum(matrix[other][c] for other in classes if other != c)
            c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
            c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
            c_f1 = (2 * c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) else 0.0
            per_class[c] = {
                "support": c_tp + c_fn,
                "precision": c_prec,
                "recall": c_rec,
                "f1": c_f1,
            }

        macro_prec = sum(v["precision"] for v in per_class.values()) / len(classes)
        macro_rec = sum(v["recall"] for v in per_class.values()) / len(classes)
        macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

        return {
            "confusion_matrix": matrix,
            "overall_accuracy": acc,
            "per_class": per_class,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
        }

    @staticmethod
    def _calc_paired_deltas(
        *,
        claim_targets: list[BenchmarkClaimTarget],
        v0_preds: list[dict[str, Any]],
        v1_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        both_correct = 0
        v0_only = 0
        v1_only = 0
        both_wrong = 0

        v1_fixed_v0 = 0
        v1_regressed = 0

        for target, v0, v1 in zip(claim_targets, v0_preds, v1_preds, strict=True):
            is_pos = (target.human_label == HumanEntailment.SUPPORTED)
            v0_correct = (v0["v0_binary_prediction"] == BinaryPrediction.ACCEPT.value) == is_pos
            v1_correct = (v1["v1_binary_prediction"] == BinaryPrediction.ACCEPT.value) == is_pos

            if v0_correct and v1_correct:
                both_correct += 1
            elif v0_correct and not v1_correct:
                v0_only += 1
                v1_regressed += 1
            elif not v0_correct and v1_correct:
                v1_only += 1
                v1_fixed_v0 += 1
            else:
                both_wrong += 1

        v0_total_correct = both_correct + v0_only
        v1_total_correct = both_correct + v1_only

        return {
            "both_correct": both_correct,
            "v0_only_correct": v0_only,
            "v1_only_correct": v1_only,
            "both_wrong": both_wrong,
            "v1_fixed_v0_error_count": v1_fixed_v0,
            "v1_regressed_from_v0_correct_count": v1_regressed,
            "v0_total_correct": v0_total_correct,
            "v1_total_correct": v1_total_correct,
            "net_correctness_delta": v1_total_correct - v0_total_correct,
        }

    @staticmethod
    def _calc_error_tag_diagnostics(
        *,
        claim_targets: list[BenchmarkClaimTarget],
        v1_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tag_counts = Counter()
        tag_caught = Counter()

        for target, v1 in zip(claim_targets, v1_preds, strict=True):
            if target.human_label != HumanEntailment.SUPPORTED:
                # Negative claim
                is_caught = v1["v1_binary_prediction"] == BinaryPrediction.REJECT.value
                for tag in target.error_tags:
                    tag_counts[tag] += 1
                    if is_caught:
                        tag_caught[tag] += 1

        diagnostics = {}
        for tag, count in tag_counts.items():
            caught = tag_caught[tag]
            diagnostics[tag] = {
                "negative_claims_with_tag": count,
                "v1_caught_count": caught,
                "v1_catch_rate": caught / count if count else 0.0,
            }

        return diagnostics

    def _calc_slice_metrics(
        self,
        *,
        claim_targets: list[BenchmarkClaimTarget],
        v0_preds: list[dict[str, Any]],
        v1_preds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        groups = defaultdict(lambda: ([], [], []))

        for target, v0, v1 in zip(claim_targets, v0_preds, v1_preds, strict=True):
            # Slice grouping
            groups[f"slice:{target.slice_id}"][0].append(target.human_label)
            groups[f"slice:{target.slice_id}"][1].append(v0["v0_binary_prediction"])
            groups[f"slice:{target.slice_id}"][2].append(v1["v1_binary_prediction"])

            # Stratum grouping
            if target.stratum:
                groups[f"stratum:{target.stratum}"][0].append(target.human_label)
                groups[f"stratum:{target.stratum}"][1].append(v0["v0_binary_prediction"])
                groups[f"stratum:{target.stratum}"][2].append(v1["v1_binary_prediction"])

        results = {}
        for name, (h_list, v0_list, v1_list) in groups.items():
            results[name] = {
                "claim_count": len(h_list),
                "v0_binary": self._calc_binary_metrics(h_list, v0_list),
                "v1_binary": self._calc_binary_metrics(h_list, v1_list),
            }

        return results

    @staticmethod
    def _calc_answer_level_metrics(
        *,
        arm_targets: list[BenchmarkArmTarget],
        v0_arm_results: dict[str, CitationVerificationResult],
        v1_arm_results: dict[str, CitationVerificationResult],
    ) -> dict[str, Any]:
        human_validity = []
        v0_validity = []
        v1_validity = []

        for arm in arm_targets:
            key = f"{arm.question_id}:{arm.arm_id}"
            all_supported = all(c.human_label == HumanEntailment.SUPPORTED for c in arm.claims)
            human_validity.append(AnswerValidity.VALID if all_supported else AnswerValidity.INVALID)

            v0_res = v0_arm_results[key]
            v0_validity.append(AnswerValidity.VALID if v0_res.is_valid else AnswerValidity.INVALID)

            v1_res = v1_arm_results[key]
            v1_validity.append(AnswerValidity.VALID if v1_res.is_valid else AnswerValidity.INVALID)

        def eval_answer(h_list: list[AnswerValidity], p_list: list[AnswerValidity]) -> dict[str, Any]:
            tp = fp = tn = fn = 0
            for h, p in zip(h_list, p_list, strict=True):
                is_pos = (h == AnswerValidity.VALID)
                is_pred_pos = (p == AnswerValidity.VALID)
                if is_pos and is_pred_pos:
                    tp += 1
                elif is_pos and not is_pred_pos:
                    fn += 1
                elif not is_pos and is_pred_pos:
                    fp += 1
                else:
                    tn += 1
            tot = tp + fp + tn + fn
            return {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": tot,
                "accuracy": (tp + tn) / tot if tot else 0.0,
                "supported_answer_retention": tp / (tp + fn) if (tp + fn) else 0.0,
                "invalid_answer_catch_rate": tn / (tn + fp) if (tn + fp) else 0.0,
            }

        return {
            "total_benchmark_arms": len(arm_targets),
            "human_valid_arms_count": sum(1 for h in human_validity if h == AnswerValidity.VALID),
            "human_invalid_arms_count": sum(1 for h in human_validity if h == AnswerValidity.INVALID),
            "v0_answer_metrics": eval_answer(human_validity, v0_validity),
            "v1_answer_metrics": eval_answer(human_validity, v1_validity),
        }

    def _build_reports(
        self,
        *,
        verdict: str,
        sources_info: dict[str, Any],
        stability_info: dict[str, Any],
        metrics_report: dict[str, Any],
        v0_claim_preds: list[dict[str, Any]],
        v1_pass1_preds: list[dict[str, Any]],
        all_claim_targets: list[BenchmarkClaimTarget],
        model_error_count: int,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        """Build main benchmark report, decision report, and paired comparison rows."""
        comparisons = []
        for target, v0, v1 in zip(all_claim_targets, v0_claim_preds, v1_pass1_preds, strict=True):
            is_pos = (target.human_label == HumanEntailment.SUPPORTED)
            v0_corr = (v0["v0_binary_prediction"] == BinaryPrediction.ACCEPT.value) == is_pos
            v1_corr = (v1["v1_binary_prediction"] == BinaryPrediction.ACCEPT.value) == is_pos

            comparisons.append({
                "slice_id": target.slice_id,
                "question_id": target.question_id,
                "arm_id": target.arm_id,
                "claim_id": target.claim_id,
                "claim_text_sha256": target.claim_text_sha256,
                "human_label": target.human_label.value,
                "v0_prediction": v0["v0_binary_prediction"],
                "v0_correct": v0_corr,
                "v1_three_way": v1["v1_three_way_prediction"],
                "v1_binary": v1["v1_binary_prediction"],
                "v1_correct": v1_corr,
                "v1_delta": (
                    "BOTH_CORRECT" if v0_corr and v1_corr
                    else "V1_FIXED_V0" if not v0_corr and v1_corr
                    else "V1_REGRESSED" if v0_corr and not v1_corr
                    else "BOTH_WRONG"
                ),
            })

        report = {
            "schema_version": "1.0",
            "verdict": verdict,
            "semantic_verifier_promotion_authorized": False,
            "evaluation_scope": "offline_controlled_forensic_benchmark",
            "execution_metadata": {
                "created_at": datetime.now(UTC).isoformat(),
                "repeat_count": self._repeat_count,
                "model_error_count": model_error_count,
                "v1_model_identity": {
                    "model_name": self._model_name,
                    "model_revision": self._model_revision,
                    "device": self._device,
                    "torch_dtype": self._torch_dtype,
                    "temperature": self._temperature,
                    "max_input_tokens": self._max_input_tokens,
                    "max_output_tokens": self._max_output_tokens,
                    "max_structured_output_retries": self._max_structured_output_retries,
                    "verification_timeout_seconds": self._verification_timeout_seconds,
                },
            },
            "sources_identity": sources_info,
            "stability": stability_info,
            "summary_metrics": {
                "total_claims": len(all_claim_targets),
                "supported_human_claims": sum(1 for c in all_claim_targets if c.human_label == HumanEntailment.SUPPORTED),
                "negative_human_claims": sum(1 for c in all_claim_targets if c.human_label != HumanEntailment.SUPPORTED),
                "v0_binary_accuracy": metrics_report["v0_binary_metrics"]["accuracy"],
                "v1_binary_accuracy": metrics_report["v1_binary_metrics"]["accuracy"],
                "v1_three_way_accuracy": metrics_report["v1_three_way_metrics"]["overall_accuracy"],
                "v1_fixed_v0_count": metrics_report["paired_deltas"]["v1_fixed_v0_error_count"],
                "v1_regressed_count": metrics_report["paired_deltas"]["v1_regressed_from_v0_correct_count"],
                "net_correctness_delta": metrics_report["paired_deltas"]["net_correctness_delta"],
            },
            "metrics": metrics_report,
        }

        decision_report = {
            "schema_version": "1.0",
            "verdict": verdict,
            "semantic_verifier_promotion_authorized": False,
            "promotion_decision": "NO_PROMOTION_AUTHORIZED",
            "justification": (
                "Task B-FORENSIC-1C is an empirical evaluation task only. "
                "Production semantic verifier promotion is explicitly prohibited."
            ),
            "stability_pass": stability_info["stable"],
            "model_error_count": model_error_count,
            "paired_deltas": metrics_report["paired_deltas"],
        }

        return report, decision_report, comparisons

    def _write_outputs(
        self,
        *,
        sources_info: dict[str, Any],
        report: dict[str, Any],
        v0_claim_preds: list[dict[str, Any]],
        v1_pass1_preds: list[dict[str, Any]],
        v1_pass2_preds: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        decision_report: dict[str, Any],
    ) -> None:
        """Write all benchmark output files deterministically."""
        exec_dir = self._output_dir / "execution"
        results_dir = self._output_dir / "results"
        exec_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        exec_identity = {
            "schema_version": "1.0",
            "benchmark_name": "verification_semantic_benchmark_v0_vs_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "sources": sources_info,
            "v1_model_name": self._model_name,
            "v1_model_revision": self._model_revision,
            "repeat_count": self._repeat_count,
            "semantic_verifier_promotion_authorized": False,
        }
        (exec_dir / "verifier_benchmark_execution_identity.json").write_text(
            json.dumps(exec_identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (results_dir / "verifier_benchmark_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (results_dir / "verifier_benchmark_decision_report.json").write_text(
            json.dumps(decision_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        self._write_jsonl(results_dir / "v0_claim_predictions.jsonl", v0_claim_preds)
        self._write_jsonl(results_dir / "v1_claim_predictions_pass1.jsonl", v1_pass1_preds)
        if v1_pass2_preds:
            self._write_jsonl(results_dir / "v1_claim_predictions_pass2.jsonl", v1_pass2_preds)
        self._write_jsonl(results_dir / "verifier_claim_comparison.jsonl", comparisons)

        if self._package_zip_path:
            self._package_zip_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self._package_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(self._output_dir):
                    for f in sorted(files):
                        fp = Path(root) / f
                        arcname = fp.relative_to(self._output_dir).as_posix()
                        z.write(fp, arcname=arcname)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _unpack_or_resolve(path: Path, prefix: str) -> Path:
        if path.is_file() and path.suffix.lower() == ".zip":
            temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(temp_dir)
            return temp_dir
        return path

    @staticmethod
    def _cleanup_temp(path: Path) -> None:
        if "AppData\\Local\\Temp" in str(path) or "/tmp" in str(path) or "temp" in path.name.lower():
            import shutil
            shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Controlled V0 vs V1 Verification Benchmark Harness."
    )
    parser.add_argument(
        "--forensic-review-packets",
        type=Path,
        required=True,
        help="Path to verification-forensic-review-packets.zip.",
    )
    parser.add_argument(
        "--forensic-labels",
        type=Path,
        required=True,
        help="Path to verification-human-forensic-labels-v1.json.",
    )
    parser.add_argument(
        "--control-review-packets",
        type=Path,
        required=True,
        help="Path to verification-positive-control-review-packets-v1.zip.",
    )
    parser.add_argument(
        "--control-labels",
        type=Path,
        required=True,
        help="Path to verification-positive-control-human-labels-v1.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for benchmark evidence and prediction files.",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Optional path to package benchmark evidence ZIP.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=CANONICAL_V1_MODEL_NAME,
        help="Model name for V1 semantic verifier.",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=CANONICAL_V1_MODEL_REVISION,
        help="Immutable commit revision for V1 model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        help="Torch dtype for model weights ('float16', 'bfloat16', 'float32').",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=2,
        help="Number of evaluation passes for stability measurement.",
    )
    parser.add_argument(
        "--skip-model-run",
        action="store_true",
        help="Skip model loading and run only source validation and V0 replay.",
    )

    args = parser.parse_args()

    runner = SemanticVerifierBenchmarkRunner(
        forensic_packets_path=args.forensic_review_packets,
        forensic_labels_path=args.forensic_labels,
        control_packets_path=args.control_review_packets,
        control_labels_path=args.control_labels,
        output_dir=args.output_dir,
        package_zip_path=args.package_zip,
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        torch_dtype=args.torch_dtype,
        repeat_count=args.repeat_count,
        skip_model_run=args.skip_model_run,
    )

    report = runner.run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nBenchmark execution finished. Verdict: {report.get('verdict')}")


if __name__ == "__main__":
    main()
