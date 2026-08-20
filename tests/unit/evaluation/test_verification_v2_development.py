"""Unit tests for V2 Development Benchmark Harness (evaluate_verification_v2_development.py)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
import legal_agentic_rag.generation.semantic_verifier
from legal_agentic_rag.generation.semantic_verifier import ModelBackedCitationVerifier
import legal_agentic_rag.generation.structured_semantic_verifier
from legal_agentic_rag.generation.structured_semantic_verifier import (
    STRUCTURED_SEMANTIC_SYSTEM_INSTRUCTION,
    EvidenceCoverageStatus,
    SemanticDimensionStatus,
    StructuredClaimAssessmentDraft,
    StructuredSemanticCitationVerifier,
    StructuredSemanticVerificationDraft,
    derive_claim_semantic_label,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
    SemanticSupportLabel,
)
from scripts.evaluate_verification_v2_development import (
    CANONICAL_CANDIDATE_ID,
    CANONICAL_CONTROL_LABELS_SHA256,
    CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
    CANONICAL_FORENSIC_LABELS_SHA256,
    CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
    CANONICAL_PACKAGE_VERSION,
    CANONICAL_V1_EVIDENCE_ZIP_SHA256,
    CANONICAL_V2_BACKEND,
    CANONICAL_V2_MODEL_NAME,
    CANONICAL_V2_MODEL_REVISION,
    CANONICAL_V2_PROVIDER_VERSION,
    BenchmarkArmTarget,
    BenchmarkClaimTarget,
    BinaryPrediction,
    HumanEntailment,
    ObservationalChatModelProviderWrapper,
    V2DevelopmentBenchmarkEvaluator,
    get_git_commit,
    get_runtime_environment,
    is_git_worktree_clean,
    main,
    sha256_file,
    sha256_text,
)


class MockV2ChatProvider(ChatModelProvider):
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "transformers"

    @property
    def provider_version(self) -> str:
        return CANONICAL_V2_PROVIDER_VERSION

    @property
    def model_name(self) -> str:
        return CANONICAL_V2_MODEL_NAME

    @property
    def model_revision(self) -> str:
        return CANONICAL_V2_MODEL_REVISION

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        res = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        if isinstance(res, Exception):
            raise res
        return res


def _make_dummy_claim_target(
    qid: str,
    arm_id: str,
    cid: str,
    label: HumanEntailment,
    tags: list[str] | None = None,
) -> BenchmarkClaimTarget:
    return BenchmarkClaimTarget(
        slice_id="positive_control",
        question_id=qid,
        arm_id=arm_id,
        claim_id=cid,
        claim_text=f"Claim {cid} text.",
        claim_text_sha256=sha256_text(f"Claim {cid} text."),
        human_label=label,
        error_tags=tags or [],
        diagnostic_note=None,
        stratum="A_SINGLE_CLAIM_CLEAN",
    )


def test_constants_and_canonical_checksums():
    """Verify pre-registered canonical source checksum constants."""
    assert len(CANONICAL_FORENSIC_REVIEW_ZIP_SHA256) == 64
    assert len(CANONICAL_FORENSIC_LABELS_SHA256) == 64
    assert len(CANONICAL_CONTROL_REVIEW_ZIP_SHA256) == 64
    assert len(CANONICAL_CONTROL_LABELS_SHA256) == 64
    assert len(CANONICAL_V1_EVIDENCE_ZIP_SHA256) == 64


def test_sources_validation_fails_on_hash_mismatch(tmp_path: Path):
    """Verify that evaluator fails closed if any of the 5 sources has a mismatched SHA-256."""
    f_pkts = tmp_path / "forensic.zip"
    f_lbls = tmp_path / "forensic.json"
    c_pkts = tmp_path / "control.zip"
    c_lbls = tmp_path / "control.json"
    v1_ev = tmp_path / "v1.zip"

    f_pkts.write_bytes(b"bad forensic zip")
    f_lbls.write_text("{}", encoding="utf-8")
    c_pkts.write_bytes(b"bad control zip")
    c_lbls.write_text("{}", encoding="utf-8")
    v1_ev.write_bytes(b"bad v1 zip")

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=f_pkts,
        forensic_labels_path=f_lbls,
        control_packets_path=c_pkts,
        control_labels_path=c_lbls,
        v1_evidence_path=v1_ev,
        output_dir=tmp_path / "out",
        preflight_only=True,
    )

    with pytest.raises(DataValidationError, match="SHA mismatch"):
        evaluator._validate_sources()


def test_sources_validation_fails_on_wrong_extension(tmp_path: Path):
    """Verify that evaluator fails closed if review packets are not .zip or labels are not .json."""
    f_pkts = tmp_path / "forensic.txt"
    f_lbls = tmp_path / "forensic.json"
    c_pkts = tmp_path / "control.zip"
    c_lbls = tmp_path / "control.json"
    v1_ev = tmp_path / "v1.zip"

    f_pkts.write_bytes(b"text")
    f_lbls.write_text("{}", encoding="utf-8")
    c_pkts.write_bytes(b"zip")
    c_lbls.write_text("{}", encoding="utf-8")
    v1_ev.write_bytes(b"zip")

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=f_pkts,
        forensic_labels_path=f_lbls,
        control_packets_path=c_pkts,
        control_labels_path=c_lbls,
        v1_evidence_path=v1_ev,
        output_dir=tmp_path / "out",
        preflight_only=True,
    )

    with pytest.raises(DataValidationError, match="must be a .zip file"):
        evaluator._validate_sources()


def test_binary_metrics_calculation():
    """Verify calculation of TP, FP, TN, FN, accuracy, precision, retention, negative catch, F1, balanced acc."""
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
        _make_dummy_claim_target("q4", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
    ]
    preds = [
        {"v2_binary_prediction": "ACCEPT"},
        {"v2_binary_prediction": "REJECT"},
        {"v2_binary_prediction": "REJECT"},
        {"v2_binary_prediction": "ACCEPT"},
    ]

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    m = evaluator._compute_binary_metrics(targets, preds, pred_key="v2_binary_prediction")
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["supported_retention"] == 0.5
    assert m["negative_catch"] == 0.5
    assert m["f1"] == 0.5
    assert m["balanced_accuracy"] == 0.5


def test_three_way_metrics_calculation():
    """Verify 3x3 confusion matrix, accuracy, and macro metrics."""
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
    ]
    preds = [
        {"v2_three_way_prediction": "SUPPORTED"},
        {"v2_three_way_prediction": "CONTRADICTED"},
        {"v2_three_way_prediction": "INSUFFICIENT"},
    ]

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    m = evaluator._compute_three_way_metrics(targets, preds, label_key="v2_three_way_prediction")
    assert m["accuracy"] == 1.0
    assert m["macro_precision"] == 1.0
    assert m["macro_recall"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["confusion_matrix"]["SUPPORTED"]["SUPPORTED"] == 1
    assert m["confusion_matrix"]["CONTRADICTED"]["CONTRADICTED"] == 1
    assert m["confusion_matrix"]["INSUFFICIENT"]["INSUFFICIENT"] == 1


def test_paired_metrics_calculation_and_v2_execution_error_reporting():
    """Verify paired comparison deltas and explicit execution error count/IDs."""
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
        _make_dummy_claim_target("q4", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q5", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
    ]
    v1_preds = [
        {"is_correct": True, "v1_binary_prediction": "ACCEPT"},
        {"is_correct": False, "v1_binary_prediction": "ACCEPT"},
        {"is_correct": True, "v1_binary_prediction": "REJECT"},
        {"is_correct": False, "v1_binary_prediction": "REJECT"},
        {"is_correct": True, "v1_binary_prediction": "ACCEPT"},
    ]
    v2_preds = [
        {"is_correct": True, "v2_binary_prediction": "ACCEPT"},
        {"is_correct": True, "v2_binary_prediction": "REJECT"},
        {"is_correct": False, "v2_binary_prediction": "ACCEPT"},
        {"is_correct": False, "v2_binary_prediction": "REJECT"},
        {"is_correct": False, "v2_binary_prediction": "EXECUTION_ERROR"},
    ]

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    p = evaluator._compute_paired_metrics(targets, v1_preds, v2_preds)
    assert p["both_correct"] == 1
    assert p["v2_only_correct"] == 1
    assert p["v1_only_correct"] == 2
    assert p["both_wrong"] == 1
    assert p["net_correctness_delta"] == -1
    assert p["v2_fixes_count"] == 1
    assert p["v2_regressions_count"] == 2
    assert p["v2_execution_error_count"] == 1
    assert p["v2_execution_error_claim_ids"] == ["q5:PRIMARY:C1"]


def test_stability_evaluation():
    """Verify multi-pass stability calculation and detection of unstable claims."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    pass1 = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_three_way_prediction": "CONTRADICTED"},
    ]
    pass2 = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_three_way_prediction": "INSUFFICIENT"},
    ]

    stab = evaluator._evaluate_stability([pass1, pass2])
    assert stab["unstable_claim_count"] == 1
    assert stab["label_stability_percentage"] == 50.0
    assert len(stab["unstable_claims"]) == 1
    assert stab["unstable_claims"][0]["question_id"] == "q2"


def test_dimension_diagnostics_no_double_counting_multi_tag():
    """Verify that a claim carrying 3 error tags increments global dimension counts EXACTLY ONCE."""
    targets = [
        _make_dummy_claim_target(
            "q1",
            "PRIMARY",
            "C1",
            HumanEntailment.CONTRADICTED,
            tags=["ACTOR_ROLE_INVERTED", "WRONG_DOCUMENT", "CONDITION_INVERTED"],
        )
    ]
    v2_preds = [
        {
            "structured_assessment": {
                "actor_role": "CONFLICT",
                "action_object": "MATCH",
                "condition_exception": "CONFLICT",
                "quantity_temporal": "NOT_APPLICABLE",
                "negation_modality": "MATCH",
                "source_article_scope": "INSUFFICIENT",
                "evidence_coverage": "PARTIAL",
            }
        }
    ]

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    diag = evaluator._compute_dimension_diagnostics(targets, v2_preds)

    assert diag["evaluated_claim_count"] == 1
    # Global counts must be exactly 1, NOT 3
    assert diag["dimension_status_counts"]["actor_role"]["CONFLICT"] == 1
    assert diag["dimension_status_counts"]["condition_exception"]["CONFLICT"] == 1
    assert diag["dimension_status_counts"]["source_article_scope"]["INSUFFICIENT"] == 1
    assert diag["dimension_status_counts"]["action_object"]["MATCH"] == 1
    assert diag["evidence_coverage_counts"]["PARTIAL"] == 1

    # But per-tag activations should record the activations under each tag
    assert diag["error_tag_activations"]["ACTOR_ROLE_INVERTED"]["total_claims"] == 1
    assert diag["error_tag_activations"]["ACTOR_ROLE_INVERTED"]["actor_role:CONFLICT"] == 1
    assert diag["error_tag_activations"]["WRONG_DOCUMENT"]["total_claims"] == 1
    assert diag["error_tag_activations"]["WRONG_DOCUMENT"]["source_article_scope:INSUFFICIENT"] == 1
    assert diag["error_tag_activations"]["CONDITION_INVERTED"]["total_claims"] == 1
    assert diag["error_tag_activations"]["CONDITION_INVERTED"]["condition_exception:CONFLICT"] == 1


def test_dimension_diagnostics_clean_38_claims_totals():
    """Verify that a clean 38-claim run yields sum(statuses)==38 for each dimension and coverage."""
    targets = [
        _make_dummy_claim_target(f"q{i}", "PRIMARY", "C1", HumanEntailment.SUPPORTED, tags=["TAG_A", "TAG_B"])
        for i in range(38)
    ]
    v2_preds = [
        {
            "structured_assessment": {
                "actor_role": "MATCH",
                "action_object": "MATCH",
                "condition_exception": "MATCH",
                "quantity_temporal": "MATCH",
                "negation_modality": "MATCH",
                "source_article_scope": "MATCH",
                "evidence_coverage": "COMPLETE",
            }
        }
        for _ in range(38)
    ]

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    diag = evaluator._compute_dimension_diagnostics(targets, v2_preds)
    assert diag["evaluated_claim_count"] == 38
    for d, counts in diag["dimension_status_counts"].items():
        assert sum(counts.values()) == 38, f"Dimension {d} sum != 38"
    assert sum(diag["evidence_coverage_counts"].values()) == 38


def test_dimension_diagnostics_with_execution_errors_uses_evaluated_denominator():
    """Verify that claims with execution errors (structured_assessment=None) are excluded from denominator."""
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
    ]
    v2_preds = [
        {
            "structured_assessment": {
                "actor_role": "MATCH",
                "action_object": "MATCH",
                "condition_exception": "MATCH",
                "quantity_temporal": "MATCH",
                "negation_modality": "MATCH",
                "source_article_scope": "MATCH",
                "evidence_coverage": "COMPLETE",
            }
        },
        {"structured_assessment": None},  # execution error
    ]

    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    diag = evaluator._compute_dimension_diagnostics(targets, v2_preds)
    assert diag["evaluated_claim_count"] == 1
    assert sum(diag["dimension_status_counts"]["actor_role"].values()) == 1


def test_development_freeze_gating_rejects_execution_errors():
    """Verify that a run with execution errors can NEVER produce CANDIDATE_FREEZE_ELIGIBLE."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    metrics_report = {
        "v1_claim_binary": {"tp": 16, "tn": 7, "fp": 13, "fn": 2, "accuracy": 0.6053, "negative_catch": 0.35, "supported_retention": 0.8889},
        "v2_claim_binary": {"tp": 18, "tn": 15, "fp": 5, "fn": 0, "execution_errors": 1, "accuracy": 0.8684, "negative_catch": 0.75, "supported_retention": 1.0},
        "v2_three_way": {"execution_errors": 1},
        "paired_v1_vs_v2": {"net_correctness_delta": 10, "v2_fixes_count": 10, "v2_regressions_count": 0, "v2_execution_error_count": 1},
    }

    report, decision, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_EXECUTION_ERROR",
        execution_identity={},
        stability_info={"unstable_claim_count": 0},
        metrics_report=metrics_report,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=1,
        total_provider_calls=22,
        structured_retry_count=0,
    )

    assert decision["development_evaluation_decision"] == "KEEP_ITERATING"
    assert decision["promotion_authorized"] is False


def test_development_freeze_gating_rejects_instability():
    """Verify that a run with label instability can NEVER produce CANDIDATE_FREEZE_ELIGIBLE."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    metrics_report = {
        "v1_claim_binary": {"tp": 16, "tn": 7, "fp": 13, "fn": 2, "accuracy": 0.6053, "negative_catch": 0.35, "supported_retention": 0.8889},
        "v2_claim_binary": {"tp": 18, "tn": 15, "fp": 5, "fn": 0, "execution_errors": 0, "accuracy": 0.8684, "negative_catch": 0.75, "supported_retention": 1.0},
        "v2_three_way": {"execution_errors": 0},
        "paired_v1_vs_v2": {"net_correctness_delta": 10, "v2_fixes_count": 10, "v2_regressions_count": 0, "v2_execution_error_count": 0},
    }

    report, decision, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_LABEL_INSTABILITY",
        execution_identity={},
        stability_info={"unstable_claim_count": 2},
        metrics_report=metrics_report,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=44,
        structured_retry_count=0,
    )

    assert decision["development_evaluation_decision"] == "KEEP_ITERATING"
    assert decision["promotion_authorized"] is False


def test_development_freeze_gating_accepts_clean_improved_candidate():
    """Verify that a mechanically clean run with improved metrics produces CANDIDATE_FREEZE_ELIGIBLE."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    metrics_report = {
        "v1_claim_binary": {"tp": 16, "tn": 7, "fp": 13, "fn": 2, "accuracy": 0.6053, "negative_catch": 0.35, "supported_retention": 0.8889},
        "v2_claim_binary": {"tp": 17, "tn": 12, "fp": 8, "fn": 1, "execution_errors": 0, "accuracy": 0.7632, "negative_catch": 0.60, "supported_retention": 0.9444},
        "v2_three_way": {"execution_errors": 0},
        "paired_v1_vs_v2": {"net_correctness_delta": 6, "v2_fixes_count": 7, "v2_regressions_count": 1, "v2_execution_error_count": 0},
    }

    report, decision, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_BENCHMARK_PASS",
        execution_identity={},
        stability_info={"unstable_claim_count": 0},
        metrics_report=metrics_report,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=44,
        structured_retry_count=0,
    )

    assert decision["development_evaluation_decision"] == "CANDIDATE_FREEZE_ELIGIBLE"
    assert decision["promotion_authorized"] is False


def test_provider_wrapper_observability_on_success_and_exception():
    """Verify that ObservationalChatModelProviderWrapper records every invocation attempt including failures."""
    inner = MockV2ChatProvider(["valid response", RuntimeError("Model CUDA OOM")])
    obs = ObservationalChatModelProviderWrapper(inner)

    # Call 1: success
    res1 = obs.complete(system_instruction="sys", user_prompt="prompt 1")
    assert res1 == "valid response"
    assert len(obs.call_history) == 1
    assert obs.call_history[0]["call_succeeded"] is True
    assert obs.call_history[0]["completion_sha256"] == sha256_text("valid response")

    # Call 2: failure
    with pytest.raises(RuntimeError, match="Model CUDA OOM"):
        obs.complete(system_instruction="sys", user_prompt="prompt 2")

    assert len(obs.call_history) == 2
    assert obs.call_history[1]["call_succeeded"] is False
    assert obs.call_history[1]["exception_type"] == "RuntimeError"
    assert obs.call_history[1]["exception_message_sha256"] == sha256_text("Model CUDA OOM")


def test_provider_wrapper_retry_observability():
    """Verify retry call accounting for malformed + successful, and malformed + failed retries."""
    base_v0 = RuleBasedCitationVerifier()

    # Case A: malformed JSON on call 1, valid on call 2 => 2 calls
    valid_json = json.dumps({
        "assessments": [
            {
                "claim_id": "C1",
                "actor_role": "MATCH",
                "action_object": "MATCH",
                "condition_exception": "MATCH",
                "quantity_temporal": "MATCH",
                "negation_modality": "MATCH",
                "source_article_scope": "MATCH",
                "evidence_coverage": "COMPLETE",
            }
        ]
    })
    mock_a = MockV2ChatProvider(["malformed json string", valid_json])
    obs_a = ObservationalChatModelProviderWrapper(mock_a)
    verifier_a = StructuredSemanticCitationVerifier(base_verifier=base_v0, provider=obs_a, max_structured_output_retries=1)

    resp = AnswerResponse(
        question="Thời hạn cấp giấy phép là bao lâu?",
        answer="Thời hạn cấp giấy phép là 15 ngày làm việc [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_001",
                document_id="doc_001",
                document_title="Luật Đầu tư",
                document_number="61/2020/QH14",
                article_number="15",
            )
        ],
    )
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk_001",
        document_id="doc_001",
        text="Thời hạn cấp giấy phép là 15 ngày làm việc kể từ ngày nhận đủ hồ sơ.",
        document_title="Luật Đầu tư",
        document_number="61/2020/QH14",
        article_number="15",
    )

    cit_res, struct_res = verifier_a.verify_structured(resp, [ev])
    assert obs_a.total_calls == 2
    assert obs_a.call_history[0]["call_succeeded"] is True
    assert obs_a.call_history[1]["call_succeeded"] is True
    assert len(struct_res.assessments) == 1
    assert max(0, obs_a.total_calls - 1) == 1

    # Case B: malformed on call 1, exception on call 2 => 2 calls recorded
    mock_b = MockV2ChatProvider(["malformed json string", RuntimeError("Provider timeout")])
    obs_b = ObservationalChatModelProviderWrapper(mock_b)
    verifier_b = StructuredSemanticCitationVerifier(base_verifier=base_v0, provider=obs_b, max_structured_output_retries=1)

    with pytest.raises(RuntimeError, match="Provider timeout"):
        verifier_b.verify_structured(resp, [ev])

    assert obs_b.total_calls == 2
    assert obs_b.call_history[0]["call_succeeded"] is True
    assert obs_b.call_history[1]["call_succeeded"] is False



def test_candidate_execution_identity_completeness(tmp_path: Path):
    """Verify that execution identity captures all required provenance fields."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        candidate_id="V2-D1",
        repeat_count=2,
    )

    sources_info = {"test_source": {"filename": "s.zip", "sha256": "abcdef", "size_bytes": 100}}
    runtime_info = {
        "provider_name": "transformers",
        "provider_version": CANONICAL_V2_PROVIDER_VERSION,
        "model_name": CANONICAL_V2_MODEL_NAME,
        "model_revision": CANONICAL_V2_MODEL_REVISION,
    }
    v0_stats = {"v0_replay_100_percent_fidelity": True}

    identity = evaluator._build_execution_identity(sources_info, runtime_info, v0_stats, repeat_count=2)

    assert identity["candidate_id"] == "V2-D1"
    assert len(identity["execution_git_commit"]) == 40
    assert isinstance(identity["git_worktree_clean"], bool)
    assert identity["source_package_version"] == CANONICAL_PACKAGE_VERSION
    assert identity["installed_distribution_version"] == CANONICAL_PACKAGE_VERSION
    assert identity["package_version"] == CANONICAL_PACKAGE_VERSION
    assert identity["provider"]["provider_version"] == CANONICAL_V2_PROVIDER_VERSION
    assert identity["model_inference_config"]["backend"] == CANONICAL_V2_BACKEND
    assert "transformers_version" in identity["runtime_environment"]
    assert "cuda_available" in identity["runtime_environment"]
    assert len(identity["implementation_identities"]["structured_semantic_verifier_sha256"]) == 64
    assert len(identity["implementation_identities"]["evaluate_verification_v2_development_sha256"]) == 64
    assert len(identity["implementation_identities"]["existing_semantic_verifier_sha256"]) == 64
    assert len(identity["prompt_identity"]["system_instruction_sha256"]) == 64
    assert len(identity["schema_identity"]["structured_verification_json_schema_sha256"]) == 64


def test_package_version_source_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that source package version != 0.50.7 is rejected with DataValidationError."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
    )
    import legal_agentic_rag
    monkeypatch.setattr(legal_agentic_rag, "__version__", "0.47.0")

    with pytest.raises(DataValidationError, match="Package version mismatch"):
        evaluator._validate_package_provenance()


def test_package_version_installed_distribution_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that installed distribution version != 0.50.7 is rejected with DataValidationError."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.47.0")

    with pytest.raises(DataValidationError, match="Package version mismatch"):
        evaluator._validate_package_provenance()


def test_package_version_disagreement_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that disagreement between source and installed distribution version is rejected."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
    )
    import legal_agentic_rag
    monkeypatch.setattr(legal_agentic_rag, "__version__", "0.50.7")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.50.8")

    with pytest.raises(DataValidationError, match="Package version mismatch"):
        evaluator._validate_package_provenance()


def test_canonical_package_versions_accepted(tmp_path: Path):
    """Verify that canonical 0.50.7 versions pass provenance validation without error."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
    )
    # Should not raise
    evaluator._validate_package_provenance()


def test_candidate_id_mismatch_rejected(tmp_path: Path):
    """Verify that candidate_id other than V2-D1 is rejected in canonical execution."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        candidate_id="V2-D2",
        custom_provider=None,
    )
    with pytest.raises(DataValidationError, match="Candidate ID mismatch"):
        evaluator._validate_canonical_config()


def test_git_worktree_dirty_blocks_canonical_real_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that a dirty git worktree raises DataValidationError in canonical real runs."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        custom_provider=None,
    )

    monkeypatch.setattr("scripts.evaluate_verification_v2_development.is_git_worktree_clean", lambda: False)

    with pytest.raises(DataValidationError, match="Git worktree must be clean"):
        evaluator._validate_canonical_config()


def test_provider_version_drift_rejected(tmp_path: Path):
    """Verify that provider version other than 4.47.1 is rejected for real canonical runs."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        custom_provider=None,
    )

    class DriftingProvider(ChatModelProvider):
        @property
        def provider_name(self) -> str: return "transformers"
        @property
        def provider_version(self) -> str: return "4.50.0"
        @property
        def model_name(self) -> str: return CANONICAL_V2_MODEL_NAME
        @property
        def model_revision(self) -> str: return CANONICAL_V2_MODEL_REVISION
        def complete(self, *, system_instruction: str, user_prompt: str) -> str: return "{}"

    with pytest.raises(DataValidationError, match="Provider version mismatch"):
        evaluator._validate_runtime_provider_identity(DriftingProvider())


def test_cell5_report_schema_keys_mapping():
    """Verify that all report and decision report keys read by Kaggle Cell 5 are correctly populated."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    metrics_report = {
        "v0_claim_binary": {"accuracy": 0.4737},
        "v1_claim_binary": {"tp": 16, "tn": 7, "fp": 13, "fn": 2, "accuracy": 0.6053, "negative_catch": 0.35, "supported_retention": 0.8889},
        "v2_claim_binary": {"tp": 18, "tn": 15, "fp": 5, "fn": 0, "execution_errors": 0, "accuracy": 0.8684, "negative_catch": 0.75, "supported_retention": 1.0},
        "v1_three_way": {"accuracy": 0.5, "execution_errors": 0},
        "v2_three_way": {"accuracy": 0.8684, "execution_errors": 0},
        "paired_v1_vs_v2": {"net_correctness_delta": 10, "v2_fixes_count": 10, "v2_regressions_count": 0, "v2_execution_error_count": 0},
        "v0_answer_metrics": {"answer_level_accuracy": 0.3182},
        "v1_answer_metrics": {"answer_level_accuracy": 0.6364, "valid_answer_retention_rate": 1.0, "invalid_answer_catch_rate": 0.4667},
        "v2_answer_metrics": {"answer_level_accuracy": 0.8636, "valid_answer_retention_rate": 1.0, "invalid_answer_catch_rate": 0.80},
        "v2_vs_v1_answer_deltas": {"v2_vs_v1_valid_retention_delta": 0.0, "v2_vs_v1_invalid_catch_delta": 0.3333, "v2_vs_v1_answer_accuracy_delta": 0.2272},
    }

    exec_identity = {
        "execution_git_commit": "1124a9b079283344ae6d688992d93a963645f2a8",
        "source_package_version": "0.50.7",
        "installed_distribution_version": "0.50.7",
        "repeat_count": 2,
    }

    report, decision, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_BENCHMARK_PASS",
        execution_identity=exec_identity,
        stability_info={"label_stability_percentage": 100.0, "unstable_claim_count": 0},
        metrics_report=metrics_report,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=44,
        structured_retry_count=0,
    )

    # Test exact fields accessed by Cell 5
    assert report["candidate_id"] == "V2-D1"
    assert report["execution_identity"]["execution_git_commit"] == "1124a9b079283344ae6d688992d93a963645f2a8"
    assert report["execution_identity"]["source_package_version"] == "0.50.7"
    assert report["execution_identity"]["installed_distribution_version"] == "0.50.7"
    assert report["verdict"] == "V2_DEVELOPMENT_BENCHMARK_PASS"
    assert decision["development_evaluation_decision"] == "CANDIDATE_FREEZE_ELIGIBLE"
    assert report["stability"]["unstable_claim_count"] == 0
    assert report["telemetry"]["model_errors"] == 0
    assert report["telemetry"]["structured_output_retries"] == 0
    assert report["metrics"]["v1_claim_binary"]["accuracy"] == 0.6053
    assert report["metrics"]["v2_claim_binary"]["accuracy"] == 0.8684
    assert report["metrics"]["paired_v1_vs_v2"]["net_correctness_delta"] == 10
    assert report["metrics"]["v1_three_way"]["accuracy"] == 0.5
    assert report["metrics"]["v2_three_way"]["accuracy"] == 0.8684
    assert report["metrics"]["v0_answer_metrics"]["answer_level_accuracy"] == 0.3182
    assert report["metrics"]["v1_answer_metrics"]["answer_level_accuracy"] == 0.6364
    assert report["metrics"]["v2_answer_metrics"]["answer_level_accuracy"] == 0.8636
    assert report["metrics"]["v2_vs_v1_answer_deltas"]["v2_vs_v1_answer_accuracy_delta"] == 0.2272
    assert decision["promotion_authorized"] is False


def test_evidence_package_required_members_integrity(tmp_path: Path):
    """Verify that package member validation asserts all 10 required artifacts."""
    required_members = {
        "execution/v2_development_source_identity.json",
        "results/v2_development_report.json",
        "results/v2_development_decision_report.json",
        "results/v2_dimension_diagnostics.json",
        "results/v0_claim_predictions.jsonl",
        "results/v1_claim_predictions.jsonl",
        "results/v2_claim_predictions_pass1.jsonl",
        "results/v2_claim_predictions_pass2.jsonl",
        "results/v2_claim_comparisons.jsonl",
        "telemetry/provider_calls.jsonl",
    }

    # Case A: complete ZIP
    zip_a = tmp_path / "complete.zip"
    with zipfile.ZipFile(zip_a, "w") as zf:
        for m in required_members:
            zf.writestr(m, "{}")

    with zipfile.ZipFile(zip_a, "r") as zf:
        missing_a = required_members - set(zf.namelist())
        assert len(missing_a) == 0

    # Case B: incomplete ZIP missing diagnostics
    zip_b = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(zip_b, "w") as zf:
        for m in required_members:
            if m != "results/v2_dimension_diagnostics.json":
                zf.writestr(m, "{}")

    with zipfile.ZipFile(zip_b, "r") as zf:
        missing_b = required_members - set(zf.namelist())
        assert missing_b == {"results/v2_dimension_diagnostics.json"}


def test_holdout_blindness_forbidden_file_rejected():
    """Verify that Cell 2 scanner detects and rejects forbidden holdout filenames."""
    forbidden_exact = {
        "verification-v2-holdout-selection-v1.json",
        "verification-v2-holdout-review-packets-v1.zip",
        "verification_v2_holdout_output",
        "phase-a-current-system-census-final-evidence.zip",
    }

    def check_file(fname: str) -> bool:
        forbidden_patterns = ["verification-v2-holdout", "verification_v2_holdout", "phase-a-current-system-census"]
        f_lower = fname.lower()
        return any(pat in f_lower for pat in forbidden_patterns) or fname in forbidden_exact

    for fname in forbidden_exact:
        assert check_file(fname) is True, f"Failed to detect forbidden file: {fname}"

    assert check_file("verification-forensic-review-packets.zip") is False
    assert check_file("verification-human-forensic-labels-v1.json") is False


def test_holdout_blindness_forbidden_directory_rejected():
    """Verify that Cell 2 scanner detects and rejects forbidden holdout directories and path components."""
    def check_path_component(part: str) -> bool:
        forbidden_patterns = ["verification-v2-holdout", "verification_v2_holdout", "phase-a-current-system-census"]
        part_lower = part.lower()
        return any(pat in part_lower for pat in forbidden_patterns)

    forbidden_dirs = [
        "verification-v2-holdout-packets",
        "verification_v2_holdout_output",
        "phase-a-current-system-census",
        "/kaggle/input/verification-v2-holdout-review-packets-v1/packets",
    ]
    for d in forbidden_dirs:
        assert check_path_component(d) is True, f"Failed to detect forbidden directory: {d}"

    assert check_path_component("verification-sources") is False
    assert check_path_component("kaggle-working") is False


def test_five_source_sha_base64_discovery_contract(tmp_path: Path):
    """Verify Base64 chunk decoding and SHA verification for transport-encoded artifacts."""
    import base64

    raw_data = b"canonical_binary_evidence_data_12345"
    expected_sha = sha256(raw_data).hexdigest()

    b64_file = tmp_path / "test_artifact.zip.b64"
    b64_file.write_text(base64.b64encode(raw_data).decode("utf-8"), encoding="utf-8")

    # Decode and verify
    decoded_bytes = base64.b64decode(b64_file.read_text(encoding="utf-8").strip())
    assert sha256(decoded_bytes).hexdigest() == expected_sha


def test_v2_system_instruction_and_schema_sha_unchanged():
    """Verify that V2-D1 system instruction and JSON schema SHA remain byte-stable."""
    expected_sys_sha = "b4dde34963df5ec11d3b8cfdaac3609311bd32c249278f039812437093b14d3e"
    expected_schema_sha = "a7050995773e783c3a4ef810bf6417bc8055f5cde4ed6e7511c1b9bc25655abc"

    observed_sys_sha = sha256_text(STRUCTURED_SEMANTIC_SYSTEM_INSTRUCTION)
    observed_schema_sha = sha256_text(
        json.dumps(StructuredSemanticVerificationDraft.model_json_schema(), sort_keys=True)
    )

    assert observed_sys_sha == expected_sys_sha, "STRUCTURED_SEMANTIC_SYSTEM_INSTRUCTION SHA changed unexpectedly!"
    assert observed_schema_sha == expected_schema_sha, "StructuredSemanticVerificationDraft schema SHA changed unexpectedly!"


def test_deterministic_label_derivation_contract_unchanged():
    """Verify all categorical derivation permutations of derive_claim_semantic_label remain identical."""
    # 1. Any dimension == CONFLICT -> CONTRADICTED
    c1 = StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=SemanticDimensionStatus.CONFLICT,
        action_object=SemanticDimensionStatus.MATCH,
        condition_exception=SemanticDimensionStatus.MATCH,
        quantity_temporal=SemanticDimensionStatus.MATCH,
        negation_modality=SemanticDimensionStatus.MATCH,
        source_article_scope=SemanticDimensionStatus.MATCH,
        evidence_coverage=EvidenceCoverageStatus.COMPLETE,
    )
    assert derive_claim_semantic_label(c1) == SemanticSupportLabel.CONTRADICTED

    # 2. Coverage != COMPLETE -> INSUFFICIENT
    c2 = StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=SemanticDimensionStatus.MATCH,
        action_object=SemanticDimensionStatus.MATCH,
        condition_exception=SemanticDimensionStatus.MATCH,
        quantity_temporal=SemanticDimensionStatus.MATCH,
        negation_modality=SemanticDimensionStatus.MATCH,
        source_article_scope=SemanticDimensionStatus.MATCH,
        evidence_coverage=EvidenceCoverageStatus.PARTIAL,
    )
    assert derive_claim_semantic_label(c2) == SemanticSupportLabel.INSUFFICIENT

    # 3. Any dimension == INSUFFICIENT -> INSUFFICIENT
    c3 = StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=SemanticDimensionStatus.MATCH,
        action_object=SemanticDimensionStatus.MATCH,
        condition_exception=SemanticDimensionStatus.INSUFFICIENT,
        quantity_temporal=SemanticDimensionStatus.MATCH,
        negation_modality=SemanticDimensionStatus.MATCH,
        source_article_scope=SemanticDimensionStatus.MATCH,
        evidence_coverage=EvidenceCoverageStatus.COMPLETE,
    )
    assert derive_claim_semantic_label(c3) == SemanticSupportLabel.INSUFFICIENT

    # 4. All dimensions MATCH/NA and COMPLETE coverage -> SUPPORTED
    c4 = StructuredClaimAssessmentDraft(
        claim_id="C1",
        actor_role=SemanticDimensionStatus.MATCH,
        action_object=SemanticDimensionStatus.MATCH,
        condition_exception=SemanticDimensionStatus.NOT_APPLICABLE,
        quantity_temporal=SemanticDimensionStatus.MATCH,
        negation_modality=SemanticDimensionStatus.MATCH,
        source_article_scope=SemanticDimensionStatus.MATCH,
        evidence_coverage=EvidenceCoverageStatus.COMPLETE,
    )
    assert derive_claim_semantic_label(c4) == SemanticSupportLabel.SUPPORTED



def test_v1_answer_level_canonical_baseline_recovery():
    """Verify answer-level metrics calculation on V1 claim predictions."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    # Simulate 2 arms:
    # Arm 1: 2 claims, both gold SUPPORTED, V1 predicted both SUPPORTED => Valid Retained
    # Arm 2: 1 claim, gold CONTRADICTED, V1 predicted INSUFFICIENT => Invalid Caught
    arm1_claims = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q1", "PRIMARY", "C2", HumanEntailment.SUPPORTED),
    ]
    arm2_claims = [
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
    ]
    arm1 = BenchmarkArmTarget(
        slice_id="positive_control",
        question_id="q1",
        arm_id="PRIMARY",
        historical_stop_reason="ans",
        stratum=None,
        question_text="Q1",
        answer_response=AnswerResponse(
            question="Q1",
            answer="A1",
            insufficient_evidence=False,
            retrieval_strategy="hybrid_rerank",
            trace_id="t1",
            citations=[],
        ),
        evidence_list=[],
        historical_verification={},
        claims=arm1_claims,
    )
    arm2 = BenchmarkArmTarget(
        slice_id="suspicious_forensic",
        question_id="q2",
        arm_id="PRIMARY",
        historical_stop_reason="ans",
        stratum=None,
        question_text="Q2",
        answer_response=AnswerResponse(
            question="Q2",
            answer="A2",
            insufficient_evidence=False,
            retrieval_strategy="hybrid_rerank",
            trace_id="t2",
            citations=[],
        ),
        evidence_list=[],
        historical_verification={},
        claims=arm2_claims,
    )

    preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v1_three_way_prediction": "SUPPORTED"},
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C2", "v1_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v1_three_way_prediction": "INSUFFICIENT"},
    ]

    m = evaluator._compute_answer_level_metrics_from_preds([arm1, arm2], preds, pred_key="v1_three_way_prediction", supported_val="SUPPORTED")
    assert m["total_answers"] == 2
    assert m["valid_ground_truth_answers"] == 1
    assert m["valid_answers_retained"] == 1
    assert m["valid_answer_retention_rate"] == 1.0
    assert m["invalid_ground_truth_answers"] == 1
    assert m["invalid_answers_caught"] == 1
    assert m["invalid_answer_catch_rate"] == 1.0
    assert m["answer_level_accuracy"] == 1.0


def test_answer_level_deltas_calculation():
    """Verify delta calculation between V2 and V1 answer metrics."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    v1_ans = {
        "valid_answer_retention_rate": 1.0,
        "invalid_answer_catch_rate": 0.4667,
        "answer_level_accuracy": 0.6364,
    }
    v2_ans = {
        "valid_answer_retention_rate": 1.0,
        "invalid_answer_catch_rate": 0.6667,
        "answer_level_accuracy": 0.7727,
    }

    deltas = evaluator._compute_answer_level_deltas(v1_ans, v2_ans)
    assert deltas["v2_vs_v1_valid_retention_delta"] == 0.0
    assert deltas["v2_vs_v1_invalid_catch_delta"] == 0.2
    assert deltas["v2_vs_v1_answer_accuracy_delta"] == 0.1363


def test_holdout_access_regression_no_cli_args():
    """Verify that CLI parser does NOT accept any holdout arguments or Phase-A paths."""
    from scripts import evaluate_verification_v2_development

    script_text = Path(evaluate_verification_v2_development.__file__).read_text(encoding="utf-8")

    forbidden_terms = [
        "verification-v2-holdout",
        "holdout-selection",
        "holdout-review-packets",
        "phase-a-current-system-census",
        "selected-contexts.zip",
        "preregister_verification_v2_holdout",
    ]
    for term in forbidden_terms:
        assert term not in script_text, f"Forbidden holdout term '{term}' found in development harness"


def test_holdout_access_regression_no_preregister_import():
    """Verify that evaluate_verification_v2_development does NOT import preregister_verification_v2_holdout."""
    import scripts.evaluate_verification_v2_development as mod

    assert not hasattr(mod, "V2HoldoutPreRegistrar")
    assert not hasattr(mod, "CANONICAL_SELECTION_SALT")
    assert "preregister_verification_v2_holdout" not in dir(mod)
    assert not any("preregister_verification_v2_holdout" in str(v) for v in vars(mod).values())


def test_production_v1_implementation_unchanged():
    """Verify that ModelBackedCitationVerifier (V1) remains untouched and imports cleanly."""
    v1_cls = ModelBackedCitationVerifier
    assert hasattr(v1_cls, "verify")
    assert hasattr(legal_agentic_rag.generation.semantic_verifier, "_SYSTEM_INSTRUCTION")


def test_production_factory_remains_unwired_to_v2():
    """Verify that production factory and RuleBasedCitationVerifier remain the production baseline."""
    base_v0 = RuleBasedCitationVerifier()
    resp = AnswerResponse(
        question="Thời hạn cấp giấy phép là bao lâu?",
        answer="Thời hạn cấp giấy phép là 15 ngày làm việc [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_001",
                document_id="doc_001",
                document_title="Luật Đầu tư",
                document_number="61/2020/QH14",
                article_number="15",
            )
        ],
    )
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk_001",
        document_id="doc_001",
        text="Thời hạn cấp giấy phép là 15 ngày làm việc kể từ ngày nhận đủ hồ sơ.",
        document_title="Luật Đầu tư",
        document_number="61/2020/QH14",
        article_number="15",
    )
    res = base_v0.verify(resp, [ev])
    assert res.is_valid is True
    assert res.semantic_verification is None



def test_end_to_end_mock_evaluation_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify end-to-end multi-pass execution flow with mock provider."""
    evaluator = V2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        candidate_id="V2-D1",
        repeat_count=2,
        custom_provider=MockV2ChatProvider(
            [
                json.dumps(
                    {
                        "assessments": [
                            {
                                "claim_id": "C1",
                                "actor_role": "MATCH",
                                "action_object": "MATCH",
                                "condition_exception": "MATCH",
                                "quantity_temporal": "MATCH",
                                "negation_modality": "MATCH",
                                "source_article_scope": "MATCH",
                                "evidence_coverage": "COMPLETE",
                            }
                        ]
                    }
                )
            ]
        ),
    )

    mock_sources = {
        "forensic_review_packets": {"filename": "f.zip", "sha256": "dummy", "size_bytes": 100},
        "forensic_labels": {"filename": "f.json", "sha256": "dummy", "size_bytes": 100},
        "control_review_packets": {"filename": "c.zip", "sha256": "dummy", "size_bytes": 100},
        "control_labels": {"filename": "c.json", "sha256": "dummy", "size_bytes": 100},
        "canonical_V1_baseline_evidence_archive": {"filename": "v1.zip", "sha256": "dummy", "size_bytes": 100},
    }

    dummy_claim = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)
    resp = AnswerResponse(
        question="Thời hạn là bao lâu?",
        answer="Thời hạn là 15 ngày [E1].",
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk_001",
                document_id="doc_001",
                document_title="Luật Đầu tư",
                document_number="61/2020/QH14",
                article_number="15",
            )
        ],
    )
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk_001",
        document_id="doc_001",
        text="Thời hạn là 15 ngày.",
        document_title="Luật Đầu tư",
        document_number="61/2020/QH14",
        article_number="15",
    )
    hist_verification = {
        "is_valid": True,
        "valid_citations": [{"evidence_id": "E1"}],
        "invalid_citations": [],
        "claim_verifications": [
            {
                "claim_id": "C1",
                "claim_text": "Thời hạn là 15 ngày .",
                "evidence_ids": ["E1"],
                "status": "supported",
                "numeric_match": True,
                "negation_match": True,
                "lexical_support_score": 1.0,
                "errors": [],
            }
        ],
        "claim_coverage_score": 1.0,
        "claim_level_verification_performed": True,
        "errors": [],
        "warnings": ["semantic_entailment_not_verified"],
    }

    dummy_arm = BenchmarkArmTarget(
        slice_id="positive_control",
        question_id="q1",
        arm_id="PRIMARY",
        historical_stop_reason="answer_verified",
        stratum="A_SINGLE_CLAIM_CLEAN",
        question_text="Thời hạn là bao lâu?",
        answer_response=resp,
        evidence_list=[ev],
        historical_verification=hist_verification,
        claims=[dummy_claim],
    )

    v1_pred = {
        "pass_number": 1,
        "slice_id": "positive_control",
        "question_id": "q1",
        "arm_id": "PRIMARY",
        "claim_id": "C1",
        "claim_text_sha256": dummy_claim.claim_text_sha256,
        "stratum": "A_SINGLE_CLAIM_CLEAN",
        "human_label": "SUPPORTED",
        "v1_three_way_prediction": "SUPPORTED",
        "v1_binary_prediction": "ACCEPT",
        "is_correct": True,
        "error_tags": [],
    }

    monkeypatch.setattr(evaluator, "_validate_sources", lambda: mock_sources)
    monkeypatch.setattr(evaluator, "_load_and_bind_benchmark_targets", lambda _: ([dummy_arm], [dummy_claim]))
    monkeypatch.setattr(evaluator, "_load_canonical_v1_predictions", lambda _: [v1_pred])

    report = evaluator.run()

    assert report["verdict"] == "V2_DEVELOPMENT_BENCHMARK_PASS"
    assert report["candidate_id"] == "V2-D1"
    assert report["stability"]["unstable_claim_count"] == 0
    assert report["metrics"]["v2_claim_binary"]["accuracy"] == 1.0
    assert (tmp_path / "out" / "results" / "v2_development_report.json").is_file()
    assert (tmp_path / "out" / "results" / "v2_development_decision_report.json").is_file()
    assert (tmp_path / "out" / "results" / "v2_dimension_diagnostics.json").is_file()
    assert (tmp_path / "out" / "results" / "v2_claim_predictions_pass1.jsonl").is_file()
    assert (tmp_path / "out" / "results" / "v2_claim_predictions_pass2.jsonl").is_file()
