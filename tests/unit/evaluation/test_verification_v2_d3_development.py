"""Unit tests for V2-D3 Development Benchmark Harness (evaluate_verification_v2_d3_development.py)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import zipfile

from pydantic import ValidationError
import pytest

from legal_agentic_rag.configuration.online import GenerationConfig, SemanticVerificationConfig
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
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
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    Evidence,
    SemanticSupportLabel,
)
from scripts.evaluate_verification_v2_d3_development import (
    CANONICAL_CANDIDATE_ID,
    CANONICAL_CONTROL_LABELS_SHA256,
    CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
    CANONICAL_FORENSIC_LABELS_SHA256,
    CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
    CANONICAL_PACKAGE_VERSION,
    CANONICAL_REPEAT_COUNT,
    CANONICAL_V1_EVIDENCE_ZIP_SHA256,
    CANONICAL_V3_BACKEND,
    CANONICAL_V3_DEVICE,
    CANONICAL_V3_MODEL_NAME,
    CANONICAL_V3_MODEL_REVISION,
    CANONICAL_V3_PROVIDER_VERSION,
    CANONICAL_V3_TEMPERATURE,
    CANONICAL_V3_TORCH_DTYPE,
    BenchmarkArmTarget,
    BenchmarkClaimTarget,
    BinaryPrediction,
    HumanEntailment,
    V2D3DevelopmentBenchmarkEvaluator,
    parse_args,
)


def _make_dummy_claim_target(
    qid: str,
    arm_id: str,
    cid: str,
    label: HumanEntailment,
    tags: list[str] | None = None,
) -> BenchmarkClaimTarget:
    return BenchmarkClaimTarget(
        slice_id="test_slice",
        question_id=qid,
        arm_id=arm_id,
        claim_id=cid,
        claim_text=f"Claim text for {cid}",
        human_label=label,
        error_tags=tags or [],
        diagnostic_note=None,
        stratum="A_SINGLE_CLAIM_CLEAN",
    )


def _make_dummy_arm_target(
    qid: str,
    arm_id: str,
    claims: list[BenchmarkClaimTarget],
) -> BenchmarkArmTarget:
    return BenchmarkArmTarget(
        slice_id="test_slice",
        question_id=qid,
        arm_id=arm_id,
        historical_stop_reason="answer_verified",
        stratum="A_SINGLE_CLAIM_CLEAN",
        question_text="Test question",
        answer_response=AnswerResponse(
            question="Test question",
            answer="Test answer",
            insufficient_evidence=False,
            retrieval_strategy="hybrid",
            trace_id="test_trace",
        ),
        evidence_list=[],
        historical_verification={},
        claims=claims,
    )


def test_canonical_parameters():
    """Verify pinned canonical model, device, and candidate parameters."""
    assert CANONICAL_CANDIDATE_ID == "V2-D3"
    assert CANONICAL_PACKAGE_VERSION == "0.50.7"
    assert CANONICAL_V3_BACKEND == "transformers"
    assert CANONICAL_V3_MODEL_NAME == "Qwen/Qwen2.5-3B-Instruct"
    assert CANONICAL_V3_MODEL_REVISION == "a1d308dfcc03e09da285d49d912439a655a571e8"
    assert CANONICAL_V3_PROVIDER_VERSION == "4.47.1"
    assert CANONICAL_V3_DEVICE == "cuda"
    assert CANONICAL_V3_TORCH_DTYPE == "float16"
    assert CANONICAL_V3_TEMPERATURE == 0.0
    assert CANONICAL_REPEAT_COUNT == 2


def test_zero_holdout_or_phase_a_cli_args():
    """Verify CLI args contain zero holdout or Phase-A parameters."""
    with patch("sys.argv", [
        "evaluate_verification_v2_d3_development.py",
        "--forensic-packets", "f.zip",
        "--forensic-labels", "f.json",
        "--control-packets", "c.zip",
        "--control-labels", "c.json",
        "--v1-evidence", "v1.zip",
        "--output-dir", "out_dir",
    ]):
        args = parse_args()
        for k in vars(args):
            assert "holdout" not in k.lower()
            assert "phase_a" not in k.lower()


def test_source_checksum_verification_fail_closed(tmp_path: Path):
    """Corrupting any of the 5 canonical input sources raises DataValidationError."""
    f_zip = tmp_path / "f.zip"
    f_json = tmp_path / "f.json"
    c_zip = tmp_path / "c.zip"
    c_json = tmp_path / "c.json"
    v1_zip = tmp_path / "v1.zip"

    for p in (f_zip, f_json, c_zip, c_json, v1_zip):
        p.write_bytes(b"corrupt")

    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=f_zip,
        forensic_labels_path=f_json,
        control_packets_path=c_zip,
        control_labels_path=c_json,
        v1_evidence_path=v1_zip,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        evaluator._verify_canonical_source_checksums()


def test_binary_metrics_computation():
    """Binary metrics compute TP, FP, TN, FN, accuracy, precision, retention, catch."""
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
        _make_dummy_claim_target("q4", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
    ]
    preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_binary_prediction": "ACCEPT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_binary_prediction": "REJECT"},
        {"question_id": "q3", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_binary_prediction": "REJECT"},
        {"question_id": "q4", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_binary_prediction": "REJECT"},
    ]
    res = evaluator._compute_binary_metrics(targets, preds, pred_key="v2_d3_binary_prediction")
    assert res["tp"] == 1
    assert res["fp"] == 0
    assert res["tn"] == 2
    assert res["fn"] == 1
    assert res["accuracy"] == 0.75
    assert res["supported_retention"] == 0.5
    assert res["negative_catch"] == 1.0


def test_three_way_metrics_computation():
    """Three-way metrics compute 3x3 confusion matrix and macro averages."""
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
    ]
    preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "CONTRADICTED"},
        {"question_id": "q3", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "INSUFFICIENT"},
    ]
    res = evaluator._compute_three_way_metrics(targets, preds, label_key="v2_d3_three_way_prediction")
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0


def test_paired_metrics_tracks_semantic_vs_exec_regressions():
    """Paired metrics distinguish semantic regression from execution error regression."""
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
    ]
    v1_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v1_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v1_three_way_prediction": "SUPPORTED"},
        {"question_id": "q3", "arm_id": "PRIMARY", "claim_id": "C1", "v1_three_way_prediction": "SUPPORTED"},
    ]
    v2_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "INSUFFICIENT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "EXECUTION_ERROR"},
        {"question_id": "q3", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "CONTRADICTED"},
    ]
    res = evaluator._compute_paired_metrics(targets, v1_preds, v2_preds)
    assert res["v2_fixes_count"] == 1
    assert res["v2_regressions_count"] == 2
    assert res["semantic_regressions_count"] == 1
    assert res["execution_error_regressions_count"] == 1
    assert res["v2_execution_error_count"] == 1
    assert res["net_correctness_delta"] == -1


def test_answer_level_error_aware_metrics():
    """Answer level execution error does not count as caught invalid answer."""
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    c1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.CONTRADICTED)
    arm = _make_dummy_arm_target("q1", "PRIMARY", [c1])
    preds = [{"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "EXECUTION_ERROR"}]

    res = evaluator._compute_answer_level_metrics_from_preds([arm], preds, pred_key="v2_d3_three_way_prediction")
    assert res["execution_error_answers"] == 1
    assert res["evaluated_answers"] == 0
    assert res["invalid_answers_caught"] == 0
    assert res["valid_answers_retained"] == 0


def test_stability_evaluation_partitions_labels_vs_errors():
    """Repeated execution errors do not count as stable semantic labels."""
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    t1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)
    t2 = _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED)

    p1 = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "EXECUTION_ERROR"},
    ]
    p2 = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d3_three_way_prediction": "EXECUTION_ERROR"},
    ]

    res = evaluator._evaluate_stability([t1, t2], p1, p2)
    assert res["claims_with_two_valid_semantic_labels"] == 1
    assert res["stable_semantic_claim_count"] == 1
    assert res["repeated_execution_error_claim_count"] == 1
    assert res["execution_error_in_any_pass_count"] == 1
    assert res["successful_label_stability_percentage"] == 100.0


def test_dimension_diagnostics_preserves_permanent_error_telemetry():
    """Dimension diagnostics aggregates retry counts and rejection categories across permanent errors."""
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    t1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)
    t2 = _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED)

    preds = [
        {
            "question_id": "q1",
            "arm_id": "PRIMARY",
            "claim_id": "C1",
            "v2_d3_binary_prediction": "ACCEPT",
            "structured_assessment": {
                "relation": "ENTAILS",
                "actor_mismatch": False,
                "condition_exception_mismatch": False,
                "quantity_temporal_mismatch": False,
                "negation_modality_mismatch": False,
                "source_scope_mismatch": False,
                "telemetry": {
                    "retry_count": 1,
                    "draft_rejection_categories": ["CLAIM_ID_MISMATCH"],
                },
            },
        },
        {
            "question_id": "q2",
            "arm_id": "PRIMARY",
            "claim_id": "C1",
            "v2_d3_binary_prediction": "EXECUTION_ERROR",
            "telemetry": {
                "retry_count": 1,
                "semantic_execution_error": True,
                "draft_rejection_categories": ["JSON_PARSE_ERROR", "JSON_PARSE_ERROR"],
            },
            "structured_assessment": None,
        },
    ]

    res = evaluator._compute_dimension_diagnostics([t1, t2], preds)
    assert res["total_claims"] == 2
    assert res["successfully_structured_claim_count"] == 1
    assert res["execution_error_claim_count"] == 1
    assert res["relation_distribution"] == {"ENTAILS": 1}
    assert res["rejection_telemetry_summary"]["total_retries"] == 2
    assert res["rejection_telemetry_summary"]["rejection_categories"] == {
        "CLAIM_ID_MISMATCH": 1,
        "JSON_PARSE_ERROR": 2,
    }


def test_freeze_gate_criteria_all_10_conditions():
    """Freeze gate requires all 10 canonical criteria."""
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    targets = [_make_dummy_claim_target(f"q_{i}", "PRIMARY", "C1", HumanEntailment.SUPPORTED) for i in range(38)]
    arms = [_make_dummy_arm_target(f"q_{i}", "PRIMARY", [targets[i]]) for i in range(22)]

    stability_pass = {
        "total_claims": 38,
        "claims_with_two_valid_semantic_labels": 38,
        "stable_semantic_claim_count": 38,
        "unstable_semantic_claim_count": 0,
        "successful_label_stability_percentage": 100.0,
        "label_stability_percentage": 100.0,
        "unstable_claim_count": 0,
        "pass1_execution_error_count": 0,
        "pass2_execution_error_count": 0,
        "execution_error_in_any_pass_count": 0,
        "repeated_execution_error_claim_count": 0,
        "unstable_claims": [],
        "execution_error_claims": [],
    }

    all_metrics = {
        "v0_claim_binary": {},
        "v1_claim_binary": {},
        "v2_d3_claim_binary": {
            "tp": 17,  # >= 16
            "fp": 5,
            "tn": 10,  # > 7 (total correct: 27 > 23)
            "fn": 6,
            "execution_errors": 0,
        },
        "v1_three_way": {},
        "v2_d3_three_way": {"execution_errors": 0},
        "paired_v1_vs_v2_d3": {
            "both_correct": 20,
            "v1_only_correct": 3,
            "v2_only_correct": 7,
            "both_wrong": 8,
            "net_correctness_delta": 4,  # > 0
            "v2_fixes_count": 7,
            "v2_regressions_count": 3,
            "semantic_regressions_count": 3,
            "execution_error_regressions_count": 0,
            "v2_execution_error_count": 0,
        },
        "v0_answer_metrics": {},
        "v1_answer_metrics": {},
        "v2_d3_answer_metrics": {"answer_level_accuracy": 0.75},
        "v2_d3_vs_v1_answer_deltas": {},
    }

    dim_diag = {
        "total_claims": 38,
        "successfully_structured_claim_count": 38,
        "execution_error_claim_count": 0,
        "relation_distribution": {"ENTAILS": 17, "CONTRADICTS": 10, "DOES_NOT_ESTABLISH": 11},
        "diagnostic_mismatch_flag_counts": {},
        "rejection_telemetry_summary": {"total_retries": 0, "rejection_categories": {}},
    }

    pass1_telem = {"provider_calls_in_pass": 38, "provider_errors_in_pass": 0}
    pass2_telem = {"provider_calls_in_pass": 38, "provider_errors_in_pass": 0}

    report, decision = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=targets,
        arm_targets=arms,
        stability_info=stability_pass,
        all_metrics=all_metrics,
        dim_diagnostics=dim_diag,
        pass1_telemetry=pass1_telem,
        pass2_telemetry=pass2_telem,
        total_duration=10.0,
    )

    assert report["verdict"] == "V2_DEVELOPMENT_BENCHMARK_PASS"
    assert decision["development_evaluation_decision"] == "CANDIDATE_FREEZE_ELIGIBLE"
    assert decision["freeze_eligible"] is True
    assert decision["promotion_authorized"] is False
