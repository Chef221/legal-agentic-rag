"""Unit tests for V2-D3.2 Development Benchmark Evaluation Harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.structured_semantic_verifier_d32 import (
    D32ClaimVerificationTelemetry,
    StructuredClaimVerificationD32,
    StructuredSemanticCitationVerifierD32,
)
from legal_agentic_rag.generation.structured_semantic_verifier_d32_conflict import (
    StrictConflictStatus,
    StructuredClaimConflictAssessmentD32,
    StructuredSemanticConflictDraftD32,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    Evidence,
    SemanticSupportLabel,
)
from scripts.evaluate_verification_v2_d32_development import (
    CANONICAL_CANDIDATE_ID,
    CANONICAL_CONTROL_LABELS_SHA256,
    CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
    CANONICAL_D3_EVIDENCE_ZIP_SHA256,
    CANONICAL_D31_EVIDENCE_ZIP_SHA256,
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
    V2D32DevelopmentBenchmarkEvaluator,
)


class MockCanonicalD32Provider(ChatModelProvider):
    """Synthetic provider simulating Qwen responses for D3 base + D3.2 conflict calls."""

    def __init__(self, mode: str = "perfect") -> None:
        self._mode = mode
        self.call_count = 0
        self.d3_calls = 0
        self.conflict_calls = 0

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
        cid = "C1"
        for line in user_prompt.splitlines():
            if "Claim ID:" in line:
                cid = line.split("Claim ID:")[-1].strip()
                break

        if "conflict verifier" in system_instruction.lower() or "cannot_both_be_true" in system_instruction:
            self.conflict_calls += 1
            if self._mode == "contradicted":
                return json.dumps({"claim_id": cid, "same_material_proposition": True, "cannot_both_be_true": True})
            return json.dumps({"claim_id": cid, "same_material_proposition": False, "cannot_both_be_true": False})
        else:
            self.d3_calls += 1
            if self._mode == "perfect":
                return json.dumps({
                    "claim_id": cid, "relation": "ENTAILS",
                    "actor_mismatch": False, "condition_exception_mismatch": False,
                    "quantity_temporal_mismatch": False, "negation_modality_mismatch": False,
                    "source_scope_mismatch": False,
                })
            elif self._mode == "contradicted":
                return json.dumps({
                    "claim_id": cid, "relation": "CONTRADICTS",
                    "actor_mismatch": False, "condition_exception_mismatch": False,
                    "quantity_temporal_mismatch": False, "negation_modality_mismatch": False,
                    "source_scope_mismatch": False,
                })
            return json.dumps({
                "claim_id": cid, "relation": "DOES_NOT_ESTABLISH",
                "actor_mismatch": False, "condition_exception_mismatch": False,
                "quantity_temporal_mismatch": False, "negation_modality_mismatch": False,
                "source_scope_mismatch": False,
            })


@pytest.fixture
def mock_sources(tmp_path: Path) -> dict[str, Path]:
    """Create minimal synthetic archives and label files matching SHA verification if mocked."""
    f_zip = tmp_path / "forensic_pkts.zip"
    f_lbl = tmp_path / "forensic_labels.json"
    c_zip = tmp_path / "control_pkts.zip"
    c_lbl = tmp_path / "control_labels.json"
    v1_zip = tmp_path / "v1_evidence.zip"
    d3_zip = tmp_path / "d3_evidence.zip"
    d31_zip = tmp_path / "d31_evidence.zip"

    # Create dummy zip with sample prediction lines
    for p in [f_zip, c_zip]:
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("sample.txt", "content")

    with zipfile.ZipFile(v1_zip, "w") as zf:
        zf.writestr("results/v1_claim_predictions_pass1.jsonl", '{"question_id": "1", "arm_id": "BASE", "claim_id": "C1", "v1_binary_prediction": "ACCEPT", "v1_three_way_prediction": "SUPPORTED"}\n')

    with zipfile.ZipFile(d3_zip, "w") as zf:
        zf.writestr("results/v2_d3_claim_predictions_pass1.jsonl", '{"question_id": "1", "arm_id": "BASE", "claim_id": "C1", "v2_d3_binary_prediction": "ACCEPT", "v2_d3_three_way_prediction": "SUPPORTED"}\n')

    with zipfile.ZipFile(d31_zip, "w") as zf:
        zf.writestr("results/v2_d31_claim_predictions_pass1.jsonl", '{"question_id": "1", "arm_id": "BASE", "claim_id": "C1", "v2_d31_binary_prediction": "ACCEPT", "v2_d31_three_way_prediction": "SUPPORTED"}\n')

    f_lbl.write_text('{"questions": {}}', encoding="utf-8")
    c_lbl.write_text('{"questions": {}}', encoding="utf-8")

    return {
        "forensic_packets": f_zip,
        "forensic_labels": f_lbl,
        "control_packets": c_zip,
        "control_labels": c_lbl,
        "v1_evidence": v1_zip,
        "d3_evidence": d3_zip,
        "d31_evidence": d31_zip,
    }


# Test 1: Canonical Source SHA Verification Fail-Closed
def test_source_checksum_failure(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        evaluator._verify_canonical_source_checksums()


# Test 2: Canonical Package Version & Candidate ID
def test_package_version_and_candidate_id(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
        candidate_id="WRONG_ID",
    )
    with pytest.raises(DataValidationError, match="Candidate ID mismatch"):
        evaluator._validate_canonical_provenance()


# Test 3: Provider Identity Verification
def test_provider_identity_verification(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )
    bad_provider = MagicMock()
    bad_provider.provider_name = "wrong_backend"
    with pytest.raises(DataValidationError, match="Provider name mismatch"):
        evaluator._validate_runtime_provider_identity(bad_provider)


# Test 4: Two Calls Per Claim Call Accounting Reconciliation
def test_call_accounting_reconciliation(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
        custom_provider=MockCanonicalD32Provider(mode="perfect"),
    )

    arm_target = BenchmarkArmTarget(
        slice_id="s1",
        question_id="Q1",
        arm_id="A1",
        historical_stop_reason="r",
        stratum="s",
        question_text="Q",
        answer_response=AnswerResponse(
            question="Q",
            answer="Ans [E1].",
            insufficient_evidence=False,
            retrieval_strategy="hybrid_rerank",
            trace_id="t1",
            citations=[Citation(evidence_id="E1", chunk_id="chk1", document_id="doc1", source_url="u")],
        ),
        evidence_list=[Evidence(evidence_id="E1", chunk_id="chk1", document_id="doc1", text="T")],
        historical_verification={},
        claims=[
            BenchmarkClaimTarget("s1", "Q1", "A1", "C1", "C", HumanEntailment.SUPPORTED, [], None, "s"),
        ],
    )

    obs_provider = ObservationalChatModelProviderWrapper(MockCanonicalD32Provider(mode="perfect"))
    verifier = StructuredSemanticCitationVerifierD32(obs_provider)

    arm_res, claim_preds, pass_telem = evaluator._run_inference_pass(
        verifier=verifier,
        provider=obs_provider,
        arm_targets=[arm_target],
        pass_index=1,
    )

    # 1 claim -> 1 D3 call + 1 conflict call = 2 provider calls
    assert pass_telem["provider_calls"] == 2
    assert pass_telem["d3_base_calls"] == 1
    assert pass_telem["conflict_calls"] == 1
    assert len(claim_preds) == 1
    assert claim_preds[0]["telemetry"]["total_provider_calls"] == 2


# Test 5: Paired Net Delta vs D3 Calculation
def test_paired_metrics_calculation(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s", "Q1", "A1", "C1", "T", HumanEntailment.SUPPORTED, [], None, "s"),
        BenchmarkClaimTarget("s", "Q1", "A1", "C2", "T", HumanEntailment.CONTRADICTED, [], None, "s"),
    ]
    # D3: C1 correct (SUPPORTED), C2 wrong (INSUFFICIENT)
    d3_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d3_three_way_prediction": "SUPPORTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d3_three_way_prediction": "INSUFFICIENT"},
    ]
    # D3.2: C1 correct (SUPPORTED), C2 fixed (CONTRADICTED)
    d32_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d32_three_way_prediction": "SUPPORTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d32_three_way_prediction": "CONTRADICTED"},
    ]

    paired = evaluator._compute_paired_metrics(
        targets, d3_preds, d32_preds,
        base_key="v2_d3_three_way_prediction", cand_key="v2_d32_three_way_prediction"
    )

    assert paired["both_correct"] == 1
    assert paired["candidate_only_correct"] == 1
    assert paired["base_only_correct"] == 0
    assert paired["net_correctness_delta"] == 1


# Test 6: Strict Conflict Precision and Recall Diagnostics
def test_strict_conflict_diagnostics(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s", "Q1", "A1", "C1", "T", HumanEntailment.CONTRADICTED, [], None, "s"),
        BenchmarkClaimTarget("s", "Q1", "A1", "C2", "T", HumanEntailment.SUPPORTED, [], None, "s"),
    ]

    # C1: Confirmed conflict (True Positive), C2: No conflict
    d32_preds = [
        {
            "question_id": "Q1", "arm_id": "A1", "claim_id": "C1",
            "v2_d32_three_way_prediction": "CONTRADICTED",
            "base_d3_label": "INSUFFICIENT",
            "override_applied": True,
            "structured_assessment": {
                "conflict_assessment": {
                    "status": "STRICT_CONTRADICTION_CONFIRMED",
                    "same_material_proposition": True,
                    "cannot_both_be_true": True,
                }
            }
        },
        {
            "question_id": "Q1", "arm_id": "A1", "claim_id": "C2",
            "v2_d32_three_way_prediction": "SUPPORTED",
            "base_d3_label": "SUPPORTED",
            "override_applied": False,
            "structured_assessment": {
                "conflict_assessment": {
                    "status": "NO_STRICT_CONTRADICTION",
                    "same_material_proposition": False,
                    "cannot_both_be_true": False,
                }
            }
        },
    ]

    diag = evaluator._compute_strict_conflict_diagnostics(targets, d32_preds)
    assert diag["strict_conflict_positives_count"] == 1
    assert diag["true_human_contradicted_among_positives"] == 1
    assert diag["false_strict_conflict_positives"] == 0
    assert diag["strict_conflict_precision"] == 1.0


# Test 7: D3.1 Learning Diagnostic
def test_d31_learning_diagnostic(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s", "Q1", "A1", "C1", "T", HumanEntailment.CONTRADICTED, [], None, "s"),
        BenchmarkClaimTarget("s", "Q1", "A1", "C2", "T", HumanEntailment.SUPPORTED, [], None, "s"),
    ]
    # D3.1 flagged both as contradiction positives
    d31_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d31_three_way_prediction": "CONTRADICTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d31_three_way_prediction": "CONTRADICTED"},
    ]
    # D3.2 strictly confirmed C1, filtered out C2
    d32_preds = [
        {
            "question_id": "Q1", "arm_id": "A1", "claim_id": "C1",
            "v2_d32_three_way_prediction": "CONTRADICTED",
            "structured_assessment": {
                "conflict_assessment": {
                    "status": "STRICT_CONTRADICTION_CONFIRMED",
                    "same_material_proposition": True,
                    "cannot_both_be_true": True,
                }
            }
        },
        {
            "question_id": "Q1", "arm_id": "A1", "claim_id": "C2",
            "v2_d32_three_way_prediction": "SUPPORTED",
            "structured_assessment": {
                "conflict_assessment": {
                    "status": "NO_STRICT_CONTRADICTION",
                    "same_material_proposition": False,
                    "cannot_both_be_true": False,
                }
            }
        },
    ]

    learning = evaluator._compute_d31_learning_diagnostic(targets, d31_preds, d32_preds)
    assert learning["d31_contradiction_positives_total"] == 2
    assert learning["retained_by_d32_strict_conflict_count"] == 1
    assert learning["filtered_out_by_d32_count"] == 1


# Test 8: 7 D3 Gains Preservation Diagnostic
def test_d3_gains_preservation(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s", "26541", "BASE", "C1", "T", HumanEntailment.INSUFFICIENT, [], None, "s"),
        BenchmarkClaimTarget("s", "95861", "BASE", "C3", "T", HumanEntailment.INSUFFICIENT, [], None, "s"),
    ]
    # D3.2 correctly predicted both
    d32_preds = [
        {"question_id": "26541", "arm_id": "BASE", "claim_id": "C1", "v2_d32_three_way_prediction": "INSUFFICIENT"},
        {"question_id": "95861", "arm_id": "BASE", "claim_id": "C3", "v2_d32_three_way_prediction": "INSUFFICIENT"},
    ]

    gain_diag = evaluator._compute_gain_preservation_diagnostic(targets, d32_preds)
    assert gain_diag["preserved_gain_count"] == 2
    assert gain_diag["regressed_gain_count"] == 0


# Test 9: Two-Pass Stability Evaluation
def test_two_pass_stability_evaluation(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget("s", "Q1", "A1", "C1", "T", HumanEntailment.SUPPORTED, [], None, "s"),
    ]
    p1 = [{"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d32_three_way_prediction": "SUPPORTED"}]
    p2 = [{"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d32_three_way_prediction": "SUPPORTED"}]

    stab = evaluator._evaluate_stability(targets, p1, p2)
    assert stab["stable_semantic_claim_count"] == 1
    assert stab["unstable_semantic_claim_count"] == 0
    assert stab["label_stability_percentage"] == 100.0


# Test 10: Selection Gate Approval (D32_SUPERSEDES_D3)
def test_selection_gate_supersedes_d3(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    all_metrics = {
        "v2_d32_claim_binary": {"tp": 17, "tn": 13, "fp": 7, "fn": 1, "execution_errors": 0},  # 30/38 > 28
        "v2_d32_three_way": {
            "confusion_matrix": {
                "SUPPORTED": {"SUPPORTED": 17, "CONTRADICTED": 0, "INSUFFICIENT": 1},
                "CONTRADICTED": {"SUPPORTED": 0, "CONTRADICTED": 4, "INSUFFICIENT": 3},
                "INSUFFICIENT": {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT": 13},
            },
            "execution_errors": 0,
        },
        "paired_binary_v2_d32_vs_v2_d3": {"net_correctness_delta": 4},
        "paired_three_way_v2_d32_vs_v2_d3": {"net_correctness_delta": 4},
        "paired_v2_d32_vs_v2_d3": {"net_correctness_delta": 4},
        "contradiction_capability": {"d32_correctly_predicted_contradicted_count": 4},
        "v2_d32_answer_metrics": {"valid_answers_retained": 7, "invalid_answers_caught": 10},  # 17/22 >= 14
    }
    stability_info = {
        "execution_error_in_any_pass_count": 0,
        "unstable_semantic_claim_count": 0,
        "claims_with_two_valid_semantic_labels": 38,
    }
    pass_telem = {
        "provider_calls": 76,
        "provider_invocation_errors": 0,
        "total_structured_retries": 0,
        "semantic_execution_errors": 0,
    }
    pass_fidelity = {
        "total_claims": 38,
        "match_count": 38,
        "mismatch_count": 0,
        "mismatches": [],
        "exact_fidelity_pass": True,
    }

    report, decision_report = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=[MagicMock()] * 38,
        arm_targets=[],
        stability_info=stability_info,
        all_metrics=all_metrics,
        strict_conflict_diag={"strict_conflict_precision": 1.0, "true_human_contradicted_among_positives": 4, "strict_conflict_positives_count": 4},
        d31_learning_diag={},
        gain_preservation_diag={"preserved_gain_count": 7},
        forensic_groups_diag={},
        pass1_telemetry=pass_telem,
        pass2_telemetry=pass_telem,
        pass1_base_d3_fidelity=pass_fidelity,
        pass2_base_d3_fidelity=pass_fidelity,
        total_duration=1.0,
    )
    assert decision_report["development_evaluation_decision"] == "D32_SUPERSEDES_D3"
    assert decision_report["d32_supersedes_d3"] is True
    assert decision_report["promotion_authorized"] is False  # Fail-closed invariant
    assert decision_report["pre_registered_gate_evaluations"]["paired_binary_net_delta_vs_d3_positive"] is True
    assert decision_report["pre_registered_gate_evaluations"]["pass1_base_d3_fidelity_passed"] is True
    assert decision_report["pre_registered_gate_evaluations"]["pass2_base_d3_fidelity_passed"] is True


# Test 11: Selection Gate Rejection (KEEP_D3 if binary accuracy does not exceed D3)
def test_selection_gate_keep_d3_if_no_gain(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    all_metrics = {
        "v2_d32_claim_binary": {"tp": 16, "tn": 10, "fp": 10, "fn": 2, "execution_errors": 0},  # 26/38 <= 28
        "v2_d32_three_way": {
            "confusion_matrix": {
                "SUPPORTED": {"SUPPORTED": 16, "CONTRADICTED": 0, "INSUFFICIENT": 2},
                "CONTRADICTED": {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT": 7},
                "INSUFFICIENT": {"SUPPORTED": 5, "CONTRADICTED": 0, "INSUFFICIENT": 8},
            },
            "execution_errors": 0,
        },
        "paired_binary_v2_d32_vs_v2_d3": {"net_correctness_delta": -2},
        "paired_three_way_v2_d32_vs_v2_d3": {"net_correctness_delta": -2},
        "paired_v2_d32_vs_v2_d3": {"net_correctness_delta": -2},
        "contradiction_capability": {"d32_correctly_predicted_contradicted_count": 0},
        "v2_d32_answer_metrics": {"valid_answers_retained": 5, "invalid_answers_caught": 7},
    }
    stability_info = {
        "execution_error_in_any_pass_count": 0,
        "unstable_semantic_claim_count": 0,
        "claims_with_two_valid_semantic_labels": 38,
    }
    pass_telem = {
        "provider_calls": 76,
        "provider_invocation_errors": 0,
        "total_structured_retries": 0,
        "semantic_execution_errors": 0,
    }
    pass_fidelity = {
        "total_claims": 38,
        "match_count": 38,
        "mismatch_count": 0,
        "mismatches": [],
        "exact_fidelity_pass": True,
    }

    report, decision_report = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=[MagicMock()] * 38,
        arm_targets=[],
        stability_info=stability_info,
        all_metrics=all_metrics,
        strict_conflict_diag={"strict_conflict_precision": 0.0, "true_human_contradicted_among_positives": 0, "strict_conflict_positives_count": 0},
        d31_learning_diag={},
        gain_preservation_diag={"preserved_gain_count": 5},
        forensic_groups_diag={},
        pass1_telemetry=pass_telem,
        pass2_telemetry=pass_telem,
        pass1_base_d3_fidelity=pass_fidelity,
        pass2_base_d3_fidelity=pass_fidelity,
        total_duration=1.0,
    )
    assert decision_report["development_evaluation_decision"] == "KEEP_D3"
    assert decision_report["d32_supersedes_d3"] is False


# Test 12: Base D3 Fidelity Verification and Drift Fail-Closed
def test_base_d3_fidelity_drift_verdict(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    # Simulate drift in pass 1
    pass1_drift_fidelity = {
        "total_claims": 38,
        "match_count": 37,
        "mismatch_count": 1,
        "mismatches": [{"claim_id": "C1", "live_base_d3_label": "CONTRADICTED", "canonical_d3_label": "SUPPORTED"}],
        "exact_fidelity_pass": False,
    }
    pass2_fidelity = {
        "total_claims": 38,
        "match_count": 38,
        "mismatch_count": 0,
        "mismatches": [],
        "exact_fidelity_pass": True,
    }

    all_metrics = {
        "v2_d32_claim_binary": {"tp": 17, "tn": 13, "fp": 7, "fn": 1, "execution_errors": 0},
        "v2_d32_three_way": {
            "confusion_matrix": {
                "SUPPORTED": {"SUPPORTED": 17, "CONTRADICTED": 0, "INSUFFICIENT": 1},
                "CONTRADICTED": {"SUPPORTED": 0, "CONTRADICTED": 4, "INSUFFICIENT": 3},
                "INSUFFICIENT": {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT": 13},
            },
            "execution_errors": 0,
        },
        "paired_binary_v2_d32_vs_v2_d3": {"net_correctness_delta": 4},
        "paired_three_way_v2_d32_vs_v2_d3": {"net_correctness_delta": 4},
        "paired_v2_d32_vs_v2_d3": {"net_correctness_delta": 4},
        "contradiction_capability": {"d32_correctly_predicted_contradicted_count": 4},
        "v2_d32_answer_metrics": {"valid_answers_retained": 7, "invalid_answers_caught": 10},
    }
    stability_info = {
        "execution_error_in_any_pass_count": 0,
        "unstable_semantic_claim_count": 0,
        "claims_with_two_valid_semantic_labels": 38,
    }
    pass_telem = {
        "provider_calls": 76,
        "provider_invocation_errors": 0,
        "total_structured_retries": 0,
        "semantic_execution_errors": 0,
    }

    report, decision_report = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=[MagicMock()] * 38,
        arm_targets=[],
        stability_info=stability_info,
        all_metrics=all_metrics,
        strict_conflict_diag={"strict_conflict_precision": 1.0, "true_human_contradicted_among_positives": 4, "strict_conflict_positives_count": 4},
        d31_learning_diag={},
        gain_preservation_diag={"preserved_gain_count": 7},
        forensic_groups_diag={},
        pass1_telemetry=pass_telem,
        pass2_telemetry=pass_telem,
        pass1_base_d3_fidelity=pass1_drift_fidelity,
        pass2_base_d3_fidelity=pass2_fidelity,
        total_duration=1.0,
    )
    assert report["verdict"] == "V2_D32_DEVELOPMENT_BASE_DRIFT"
    assert decision_report["development_evaluation_decision"] == "KEEP_D3"
    assert decision_report["d32_supersedes_d3"] is False
    assert decision_report["promotion_authorized"] is False
    assert decision_report["pre_registered_gate_evaluations"]["base_drift_detected"] is True
    assert decision_report["pre_registered_gate_evaluations"]["mechanical_gates_passed"] is False


# Test 13: Paired Binary vs Three-Way Metrics Separation Contract
def test_paired_binary_vs_three_way_metrics_accounting(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    targets = [
        BenchmarkClaimTarget(
            slice_id="s", question_id="Q1", arm_id="A1", claim_id="C1",
            claim_text="t", human_label=HumanEntailment.CONTRADICTED,
            error_tags=[], diagnostic_note=None, stratum="s",
        ),
        BenchmarkClaimTarget(
            slice_id="s", question_id="Q1", arm_id="A1", claim_id="C2",
            claim_text="t", human_label=HumanEntailment.SUPPORTED,
            error_tags=[], diagnostic_note=None, stratum="s",
        ),
    ]

    # In D3: C1 was predicted INSUFFICIENT (binary REJECT -> correct, 3-way INSUFFICIENT -> wrong)
    # In D3.2: C1 is predicted CONTRADICTED (binary REJECT -> correct, 3-way CONTRADICTED -> correct)
    d3_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d3_binary_prediction": "REJECT", "v2_d3_three_way_prediction": "INSUFFICIENT"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d3_binary_prediction": "ACCEPT", "v2_d3_three_way_prediction": "SUPPORTED"},
    ]
    d32_preds = [
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "v2_d32_binary_prediction": "REJECT", "v2_d32_three_way_prediction": "CONTRADICTED"},
        {"question_id": "Q1", "arm_id": "A1", "claim_id": "C2", "v2_d32_binary_prediction": "ACCEPT", "v2_d32_three_way_prediction": "SUPPORTED"},
    ]

    paired_binary = evaluator._compute_paired_binary_metrics(targets, d3_preds, d32_preds)
    paired_three_way = evaluator._compute_paired_three_way_metrics(targets, d3_preds, d32_preds)

    # For binary: both D3 and D3.2 correctly reject C1 and accept C2 -> both_correct=2, net_delta=0
    assert paired_binary["both_correct"] == 2
    assert paired_binary["candidate_only_correct"] == 0
    assert paired_binary["net_correctness_delta"] == 0

    # For three-way: D3 is wrong on C1 (INSUFFICIENT vs CONTRADICTED), D3.2 is correct on C1 -> candidate_only_correct=1, net_delta=+1
    assert paired_three_way["both_correct"] == 1
    assert paired_three_way["candidate_only_correct"] == 1
    assert paired_three_way["net_correctness_delta"] == 1


# Test 14: Evidence Package Canonical Member Inventory Contract
def test_evidence_package_canonical_member_inventory(mock_sources, tmp_path):
    out_dir = tmp_path / "out"
    pkg_zip = tmp_path / "evidence_pkg.zip"

    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=out_dir,
        package_zip=pkg_zip,
    )

    report = {"schema_version": "1.0", "verdict": "V2_D32_DEVELOPMENT_BENCHMARK_PASS"}
    decision_report = {"schema_version": "1.0", "development_evaluation_decision": "KEEP_D3"}
    strict_diag = {"schema_version": "1.0", "artifact_type": "v2_d32_strict_conflict_diagnostics"}
    d31_diag = {"schema_version": "1.0", "artifact_type": "v2_d32_d31_learning_diagnostic"}
    exec_id = {"schema_version": "1.0", "candidate_id": CANONICAL_CANDIDATE_ID}

    evaluator._write_reports(
        report=report,
        decision_report=decision_report,
        strict_conflict_diag=strict_diag,
        d31_learning_diag=d31_diag,
        v0_claim_preds=[{"claim_id": "C1"}],
        v1_claim_preds=[{"claim_id": "C1"}],
        d3_claim_preds=[{"claim_id": "C1"}],
        d31_claim_preds=[{"claim_id": "C1"}],
        pass1_claim_preds=[{"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "human_label": "SUPPORTED", "v2_d32_three_way_prediction": "SUPPORTED"}],
        pass2_claim_preds=[{"question_id": "Q1", "arm_id": "A1", "claim_id": "C1", "human_label": "SUPPORTED", "v2_d32_three_way_prediction": "SUPPORTED"}],
        exec_identity=exec_id,
        provider=None,
        is_preflight=False,
    )

    assert pkg_zip.is_file()
    with zipfile.ZipFile(pkg_zip, "r") as zf:
        members = set(zf.namelist())

    expected_members = {
        "execution/v2_d32_development_source_identity.json",
        "results/v2_d32_development_report.json",
        "results/v2_d32_development_decision_report.json",
        "results/v2_d32_strict_conflict_diagnostics.json",
        "results/v2_d32_d31_learning_diagnostic.json",
        "results/v0_claim_predictions.jsonl",
        "results/v1_claim_predictions.jsonl",
        "results/v2_d3_claim_predictions.jsonl",
        "results/v2_d31_claim_predictions.jsonl",
        "results/v2_d32_claim_predictions_pass1.jsonl",
        "results/v2_d32_claim_predictions_pass2.jsonl",
        "results/v2_d32_claim_comparisons.jsonl",
    }
    for m in expected_members:
        assert m in members, f"Expected canonical archive member '{m}' missing"


# Test 15: Frozen D3 Source Identity Verification & Tamper Detection
def test_frozen_d3_source_identity_verification(mock_sources, tmp_path):
    evaluator = V2D32DevelopmentBenchmarkEvaluator(
        forensic_packets_path=mock_sources["forensic_packets"],
        forensic_labels_path=mock_sources["forensic_labels"],
        control_packets_path=mock_sources["control_packets"],
        control_labels_path=mock_sources["control_labels"],
        v1_evidence_path=mock_sources["v1_evidence"],
        d3_evidence_path=mock_sources["d3_evidence"],
        d31_evidence_path=mock_sources["d31_evidence"],
        output_dir=tmp_path / "out",
    )

    exec_id = evaluator._build_execution_identity(sources_info={})
    assert exec_id["frozen_d3_source_identity_verified"] is True
    assert (
        exec_id["prompt_identities"]["d3_base_system_instruction_sha256"]
        == "546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b"
    )
    assert (
        exec_id["prompt_identities"]["d32_conflict_system_instruction_sha256"]
        == "de032266a5700a5459c21e65c5cb383e97f17bf92aabb43e6153b64faccf0312"
    )
    assert "structured_semantic_verifier_d3_sha256" in exec_id["implementation_identities"]


# Test 16: Observational Provider Call Component Reconciliation
def test_observational_provider_call_reconciliation():
    mock_inner = MockCanonicalD32Provider(mode="perfect")
    obs = ObservationalChatModelProviderWrapper(mock_inner)

    d3_instruction = "System Instruction with D3 verifier rules..."
    conflict_instruction = "System Instruction with conflict verifier and cannot_both_be_true rules..."

    # 2 D3 calls and 2 conflict calls
    obs.complete(system_instruction=d3_instruction, user_prompt="Claim ID: C1\nClaim: text 1")
    obs.complete(system_instruction=conflict_instruction, user_prompt="Claim ID: C1\nClaim: text 1")
    obs.complete(system_instruction=d3_instruction, user_prompt="Claim ID: C2\nClaim: text 2")
    obs.complete(system_instruction=conflict_instruction, user_prompt="Claim ID: C2\nClaim: text 2")

    assert obs.total_calls == 4
    assert len(obs.call_history) == 4

    from hashlib import sha256
    d3_sha = sha256(d3_instruction.encode("utf-8")).hexdigest()
    conflict_sha = sha256(conflict_instruction.encode("utf-8")).hexdigest()

    d3_count = sum(1 for c in obs.call_history if c["system_instruction_sha256"] == d3_sha)
    conflict_count = sum(1 for c in obs.call_history if c["system_instruction_sha256"] == conflict_sha)

    assert d3_count == 2
    assert conflict_count == 2
    assert d3_count + conflict_count == obs.total_calls
