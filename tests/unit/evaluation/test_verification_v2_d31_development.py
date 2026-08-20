"""Unit tests for V2-D3.1 Development Benchmark Evaluation Harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.structured_semantic_verifier_d31 import (
    DraftRejectionCategoryD31,
    StructuredClaimVerificationD31,
    StructuredSemanticVerificationDraftD31,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Evidence,
    SemanticSupportLabel,
)
from scripts.evaluate_verification_v2_d31_development import (
    CANONICAL_CANDIDATE_ID,
    CANONICAL_CONTROL_LABELS_SHA256,
    CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
    CANONICAL_D3_EVIDENCE_ZIP_SHA256,
    CANONICAL_FORENSIC_LABELS_SHA256,
    CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
    CANONICAL_PACKAGE_VERSION,
    CANONICAL_V1_EVIDENCE_ZIP_SHA256,
    CANONICAL_V3_BACKEND,
    CANONICAL_V3_MODEL_NAME,
    CANONICAL_V3_MODEL_REVISION,
    CANONICAL_V3_PROVIDER_VERSION,
    D3_FIX_CLAIM_KEYS,
    FORENSIC_ERROR_GROUPS,
    BenchmarkArmTarget,
    BenchmarkClaimTarget,
    HumanEntailment,
    ObservationalChatModelProviderWrapper,
    V2D31DevelopmentBenchmarkEvaluator,
)


class MockCanonicalD31Provider(ChatModelProvider):
    """Synthetic provider simulating Qwen responses for 38 claims."""

    def __init__(self, mode: str = "perfect") -> None:
        self._mode = mode
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return CANONICAL_V3_BACKEND

    @property
    def provider_version(self) -> str:
        return CANONICAL_V3_PROVIDER_VERSION

    @property
    def model_name(self) -> str:
        return CANONICAL_V3_MODEL_NAME

    @property
    def model_revision(self) -> str:
        return CANONICAL_V3_MODEL_REVISION

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        self.call_count += 1
        # Extract claim ID from prompt
        cid = "C1"
        for line in user_prompt.splitlines():
            if "Claim ID:" in line:
                cid = line.split("Claim ID:")[-1].strip()
                break

        if self._mode == "perfect":
            # State B (Supported)
            return json.dumps({"claim_id": cid, "is_contradicted": False, "is_fully_established": True})
        elif self._mode == "contradicted":
            # State A (Contradicted)
            return json.dumps({"claim_id": cid, "is_contradicted": True, "is_fully_established": False})
        elif self._mode == "insufficient":
            # State C (Insufficient)
            return json.dumps({"claim_id": cid, "is_contradicted": False, "is_fully_established": False})
        elif self._mode == "invalid_state":
            # Invalid State
            return json.dumps({"claim_id": cid, "is_contradicted": True, "is_fully_established": True})
        return json.dumps({"claim_id": cid, "is_contradicted": False, "is_fully_established": True})


@pytest.fixture
def mock_sources(tmp_path: Path) -> dict[str, Path]:
    """Create minimal synthetic archives and label files matching SHA verification if mocked."""
    f_zip = tmp_path / "forensic_pkts.zip"
    f_lbl = tmp_path / "forensic_labels.json"
    c_zip = tmp_path / "control_pkts.zip"
    c_lbl = tmp_path / "control_labels.json"
    v1_zip = tmp_path / "v1_evidence.zip"
    d3_zip = tmp_path / "d3_evidence.zip"

    for p in [f_zip, f_lbl, c_zip, c_lbl, v1_zip, d3_zip]:
        p.write_bytes(b"dummy")

    return {
        "forensic_packets": f_zip,
        "forensic_labels": f_lbl,
        "control_packets": c_zip,
        "control_labels": c_lbl,
        "v1_evidence": v1_zip,
        "d3_evidence": d3_zip,
    }


# Test 1: SHA-256 Fail-Closed Check on all 6 datasets
def test_sha256_fail_closed_validation(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        evaluator._verify_canonical_source_checksums()


# Test 2: Package version gate
def test_package_version_gate(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )
    with patch("legal_agentic_rag.__version__", "0.99.0"):
        with pytest.raises(DataValidationError, match="Package version mismatch"):
            evaluator._validate_package_provenance()


# Test 3: Candidate ID gate
def test_candidate_id_gate(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
        candidate_id="INVALID-CANDIDATE",
    )
    with pytest.raises(DataValidationError, match="Candidate ID mismatch"):
        evaluator._validate_canonical_provenance()


# Test 4: Provider identity validation
def test_runtime_provider_identity_validation(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )
    mock_prov = MagicMock(spec=ChatModelProvider)
    mock_prov.provider_name = "wrong_provider"
    mock_prov.model_name = CANONICAL_V3_MODEL_NAME
    mock_prov.model_revision = CANONICAL_V3_MODEL_REVISION
    mock_prov.provider_version = CANONICAL_V3_PROVIDER_VERSION

    with pytest.raises(DataValidationError, match="Provider name mismatch"):
        evaluator._validate_runtime_provider_identity(mock_prov)


# Test 5: Binary & Three-Way Metrics Computation
def test_binary_and_three_way_metrics_calculation(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s1", "Q1", "A1", "C1", "Text 1", HumanEntailment.SUPPORTED, [], None, "stratum"),
        BenchmarkClaimTarget("s1", "Q1", "A1", "C2", "Text 2", HumanEntailment.CONTRADICTED, [], None, "stratum"),
        BenchmarkClaimTarget("s1", "Q1", "A1", "C3", "Text 3", HumanEntailment.INSUFFICIENT, [], None, "stratum"),
    ]

    preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d31_binary_prediction": "ACCEPT", "v2_d31_three_way_prediction": "SUPPORTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d31_binary_prediction": "REJECT", "v2_d31_three_way_prediction": "CONTRADICTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C3", "v2_d31_binary_prediction": "REJECT", "v2_d31_three_way_prediction": "INSUFFICIENT"},
    ]

    bin_res = evaluator._compute_binary_metrics(targets, preds, "v2_d31_binary_prediction")
    assert bin_res["tp"] == 1
    assert bin_res["tn"] == 2
    assert bin_res["fp"] == 0
    assert bin_res["fn"] == 0
    assert bin_res["accuracy"] == 1.0

    three_way_res = evaluator._compute_three_way_metrics(targets, preds, "v2_d31_three_way_prediction")
    assert three_way_res["accuracy"] == 1.0
    assert three_way_res["macro_f1"] == 1.0


# Test 6: Paired D3.1 vs D3 Metrics & Net Delta
def test_paired_d31_vs_d3_metrics(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s1", "Q1", "A1", "C1", "T1", HumanEntailment.SUPPORTED, [], None, "s"),
        BenchmarkClaimTarget("s1", "Q1", "A1", "C2", "T2", HumanEntailment.CONTRADICTED, [], None, "s"),
    ]
    # D3: C1 correct (SUPPORTED), C2 wrong (INSUFFICIENT)
    d3_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d3_three_way_prediction": "SUPPORTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d3_three_way_prediction": "INSUFFICIENT"},
    ]
    # D3.1: C1 correct (SUPPORTED), C2 correct (CONTRADICTED) -> +1 fix!
    d31_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d31_three_way_prediction": "SUPPORTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d31_three_way_prediction": "CONTRADICTED"},
    ]

    paired = evaluator._compute_paired_metrics(
        targets, d3_preds, d31_preds,
        base_key="v2_d3_three_way_prediction", cand_key="v2_d31_three_way_prediction",
    )
    assert paired["both_correct"] == 1
    assert paired["candidate_only_correct"] == 1
    assert paired["base_only_correct"] == 0
    assert paired["net_correctness_delta"] == 1


# Test 7: Contradiction Capability Diagnostic
def test_contradiction_capability_diagnostic(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s1", "Q1", "A1", "C1", "T1", HumanEntailment.CONTRADICTED, [], None, "s"),
        BenchmarkClaimTarget("s1", "Q1", "A1", "C2", "T2", HumanEntailment.CONTRADICTED, [], None, "s"),
    ]
    d3_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d3_three_way_prediction": "INSUFFICIENT"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d3_three_way_prediction": "INSUFFICIENT"},
    ]
    d31_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d31_three_way_prediction": "CONTRADICTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d31_three_way_prediction": "INSUFFICIENT"},
    ]

    contra_diag = evaluator._compute_contradiction_diagnostics(targets, d3_preds, d31_preds)
    assert contra_diag["total_gold_contradicted_claims"] == 2
    assert contra_diag["d3_correctly_predicted_contradicted_count"] == 0
    assert contra_diag["d31_correctly_predicted_contradicted_count"] == 1
    assert contra_diag["d31_contradicted_recall"] == 0.5
    assert contra_diag["contradiction_discrimination_improved"] is True


# Test 8: 7 D3 Gains Preservation Diagnostic
def test_d3_gain_preservation_diagnostic(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s1", "26541", "BASE", "C1", "T", HumanEntailment.INSUFFICIENT, [], None, "s"),
        BenchmarkClaimTarget("s1", "95861", "BASE", "C3", "T", HumanEntailment.INSUFFICIENT, [], None, "s"),
    ]
    d31_preds = [
        {"question_id": "26541", "arm_id": "BASE", "claim_id": "C1", "v2_d31_three_way_prediction": "INSUFFICIENT"},
        {"question_id": "95861", "arm_id": "BASE", "claim_id": "C3", "v2_d31_three_way_prediction": "SUPPORTED"},  # regressed
    ]

    diag = evaluator._compute_gain_preservation_diagnostic(targets, d31_preds)
    assert diag["preserved_gain_count"] == 1
    assert diag["regressed_gain_count"] == 1
    assert diag["regressed_gain_claim_ids"] == ["95861:BASE:C3"]
    assert diag["all_7_d3_fixes_preserved"] is False


# Test 9: Two-Pass Stability Evaluation
def test_two_pass_stability_evaluation(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s1", "Q1", "A1", "C1", "T1", HumanEntailment.SUPPORTED, [], None, "s"),
        BenchmarkClaimTarget("s1", "Q1", "A1", "C2", "T2", HumanEntailment.CONTRADICTED, [], None, "s"),
    ]
    p1 = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d31_three_way_prediction": "SUPPORTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d31_three_way_prediction": "CONTRADICTED"},
    ]
    p2 = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d31_three_way_prediction": "SUPPORTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d31_three_way_prediction": "INSUFFICIENT"},  # unstable
    ]

    stab = evaluator._evaluate_stability(targets, p1, p2)
    assert stab["total_claims"] == 2
    assert stab["stable_semantic_claim_count"] == 1
    assert stab["unstable_semantic_claim_count"] == 1
    assert stab["label_stability_percentage"] == 50.0
    assert len(stab["unstable_claims"]) == 1


# Test 10: Pre-Registered Selection Gate Logic (D31_SUPERSEDES_D3 vs KEEP_D3)
def test_selection_gate_logic(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )

    # Case A: Meets all quality & mechanical targets
    all_metrics = {
        "v2_d31_claim_binary": {"tp": 17, "tn": 13, "fp": 7, "fn": 1, "execution_errors": 0},
        "v2_d31_three_way": {
            "confusion_matrix": {
                "SUPPORTED": {"SUPPORTED": 17, "CONTRADICTED": 0, "INSUFFICIENT": 1},
                "CONTRADICTED": {"SUPPORTED": 0, "CONTRADICTED": 4, "INSUFFICIENT": 3},
                "INSUFFICIENT": {"SUPPORTED": 3, "CONTRADICTED": 0, "INSUFFICIENT": 9},
            },
            "execution_errors": 0,
        },
        "paired_v2_d31_vs_v2_d3": {"net_correctness_delta": 4},
        "contradiction_capability": {"d31_correctly_predicted_contradicted_count": 4},
        "v2_d31_answer_metrics": {"valid_answers_retained": 6, "invalid_answers_caught": 9},
    }
    stability_info = {
        "execution_error_in_any_pass_count": 0,
        "unstable_semantic_claim_count": 0,
        "claims_with_two_valid_semantic_labels": 38,
    }
    pass_telem = {
        "provider_calls": 38,
        "provider_invocation_errors": 0,
        "structured_retries": 0,
        "semantic_execution_errors": 0,
    }

    report, decision_report = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=[MagicMock()] * 38,
        arm_targets=[],
        stability_info=stability_info,
        all_metrics=all_metrics,
        dim_diagnostics={},
        gain_preservation_diag={"preserved_gain_count": 7},
        forensic_groups_diag={},
        pass1_telemetry=pass_telem,
        pass2_telemetry=pass_telem,
        total_duration=1.0,
    )
    assert decision_report["development_evaluation_decision"] == "D31_SUPERSEDES_D3"
    assert decision_report["d31_supersedes_d3"] is True
    assert decision_report["promotion_authorized"] is False  # Fail-closed invariant


# Test 11: Selection Gate Rejection (KEEP_D3 if binary accuracy does not exceed D3)
def test_selection_gate_keep_d3_if_no_gain(mock_sources, tmp_path):
    evaluator = V2D31DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        output_dir=tmp_path / "out",
    )

    all_metrics = {
        "v2_d31_claim_binary": {"tp": 16, "tn": 10, "fp": 10, "fn": 2, "execution_errors": 0},  # 26/38 <= 28
        "v2_d31_three_way": {
            "confusion_matrix": {
                "SUPPORTED": {"SUPPORTED": 16, "CONTRADICTED": 0, "INSUFFICIENT": 2},
                "CONTRADICTED": {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT": 7},
                "INSUFFICIENT": {"SUPPORTED": 5, "CONTRADICTED": 0, "INSUFFICIENT": 8},
            },
            "execution_errors": 0,
        },
        "paired_v2_d31_vs_v2_d3": {"net_correctness_delta": -2},
        "contradiction_capability": {"d31_correctly_predicted_contradicted_count": 0},
        "v2_d31_answer_metrics": {"valid_answers_retained": 5, "invalid_answers_caught": 7},
    }
    stability_info = {
        "execution_error_in_any_pass_count": 0,
        "unstable_semantic_claim_count": 0,
        "claims_with_two_valid_semantic_labels": 38,
    }
    pass_telem = {
        "provider_calls": 38,
        "provider_invocation_errors": 0,
        "structured_retries": 0,
        "semantic_execution_errors": 0,
    }

    report, decision_report = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=[MagicMock()] * 38,
        arm_targets=[],
        stability_info=stability_info,
        all_metrics=all_metrics,
        dim_diagnostics={},
        gain_preservation_diag={"preserved_gain_count": 5},
        forensic_groups_diag={},
        pass1_telemetry=pass_telem,
        pass2_telemetry=pass_telem,
        total_duration=1.0,
    )
    assert decision_report["development_evaluation_decision"] == "KEEP_D3"
    assert decision_report["d31_supersedes_d3"] is False
