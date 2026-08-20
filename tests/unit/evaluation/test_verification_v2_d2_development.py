"""Unit tests for V2-D2 Development Benchmark Harness (evaluate_verification_v2_d2_development.py)."""

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
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
    SemanticSupportLabel,
)
from scripts.evaluate_verification_v2_d2_development import (
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
    V2D2DevelopmentBenchmarkEvaluator,
    get_git_commit,
    get_runtime_environment,
    is_git_worktree_clean,
    main,
    sha256_file,
    sha256_text,
)


class MockD2ChatProvider(ChatModelProvider):
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.call_count = 0
        self.call_history: list[dict[str, str]] = []

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
        self.call_history.append({"system_instruction": system_instruction, "user_prompt": user_prompt})
        idx = self.call_count
        self.call_count += 1
        res = self.responses[idx % len(self.responses)]
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
        slice_id="test_slice",
        question_id=qid,
        arm_id=arm_id,
        claim_id=cid,
        claim_text=f"Claim text for {cid}",
        claim_text_sha256=sha256_text(f"Claim text for {cid}"),
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


def test_candidate_id_is_v2_d2():
    """Verify that canonical candidate ID is V2-D2."""
    assert CANONICAL_CANDIDATE_ID == "V2-D2"


def test_canonical_package_version_provenance_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify package version 0.50.7 validation gates."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
    )
    # 1. Canonical passes
    evaluator._validate_package_provenance()

    # 2. Source mismatch rejected
    import legal_agentic_rag
    monkeypatch.setattr(legal_agentic_rag, "__version__", "0.47.0")
    with pytest.raises(DataValidationError, match="Package version mismatch"):
        evaluator._validate_package_provenance()

    # 3. Installed mismatch rejected
    monkeypatch.setattr(legal_agentic_rag, "__version__", "0.50.7")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.47.0")
    with pytest.raises(DataValidationError, match="Package version mismatch"):
        evaluator._validate_package_provenance()


def test_candidate_id_mismatch_rejected_in_canonical_mode(tmp_path: Path):
    """Verify that candidate ID other than V2-D2 is rejected in canonical mode."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        candidate_id="V2-D1",
        custom_provider=None,
    )
    with pytest.raises(DataValidationError, match="Candidate ID mismatch"):
        evaluator._validate_canonical_config()


def test_git_dirty_rejected_in_canonical_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that a dirty git worktree raises DataValidationError in canonical mode."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        custom_provider=None,
    )
    monkeypatch.setattr("scripts.evaluate_verification_v2_d2_development.is_git_worktree_clean", lambda: False)
    with pytest.raises(DataValidationError, match="Git worktree must be clean"):
        evaluator._validate_canonical_config()


def test_init_v2_provider_generation_config_invariants(tmp_path: Path):
    """Verify that _init_v2_provider executes real config construction and passes valid GenerationConfig."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        device="cuda",
        repeat_count=2,
        custom_provider=None,
    )

    with patch(
        "scripts.evaluate_verification_v2_d2_development.TransformersChatProvider",
        autospec=True,
    ) as mock_provider_cls:
        evaluator._init_v2_provider()

        mock_provider_cls.assert_called_once()
        call_args = mock_provider_cls.call_args
        assert len(call_args.args) == 1
        gen_cfg = call_args.args[0]
        assert isinstance(gen_cfg, GenerationConfig)
        assert gen_cfg.backend == "transformers"
        assert gen_cfg.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert gen_cfg.model_revision == "a1d308dfcc03e09da285d49d912439a655a571e8"
        assert gen_cfg.device == "cuda"
        assert gen_cfg.torch_dtype == "float16"
        assert gen_cfg.local_files_only is False
        assert gen_cfg.temperature == 0.0
        assert gen_cfg.max_input_tokens == 8192
        assert gen_cfg.max_output_tokens == 512
        assert gen_cfg.timeout_seconds == 180.0
        assert gen_cfg.max_structured_output_retries == 1


def test_execution_identity_structure(tmp_path: Path):
    """Verify that execution identity captures all required provenance fields and implementation hashes."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=tmp_path / "f.zip",
        forensic_labels_path=tmp_path / "f.json",
        control_packets_path=tmp_path / "c.zip",
        control_labels_path=tmp_path / "c.json",
        v1_evidence_path=tmp_path / "v1.zip",
        output_dir=tmp_path / "out",
        candidate_id="V2-D2",
        repeat_count=2,
    )

    sources_info = {"test": {"filename": "t.zip", "sha256": "abcdef", "size_bytes": 10}}
    runtime_info = {
        "provider_name": "transformers",
        "provider_version": CANONICAL_V2_PROVIDER_VERSION,
        "model_name": CANONICAL_V2_MODEL_NAME,
        "model_revision": CANONICAL_V2_MODEL_REVISION,
    }
    v0_stats = {"v0_replay_100_percent_fidelity": True}

    identity = evaluator._build_execution_identity(sources_info, runtime_info, v0_stats, repeat_count=2)

    assert identity["candidate_id"] == "V2-D2"
    assert identity["package_version"] == CANONICAL_PACKAGE_VERSION
    assert identity["source_package_version"] == CANONICAL_PACKAGE_VERSION
    assert identity["installed_distribution_version"] == CANONICAL_PACKAGE_VERSION
    assert len(identity["implementation_identities"]["structured_semantic_verifier_d2_sha256"]) == 64
    assert len(identity["implementation_identities"]["evaluate_verification_v2_d2_development_sha256"]) == 64
    assert len(identity["implementation_identities"]["historical_d1_verifier_sha256"]) == 64
    assert len(identity["prompt_identity"]["system_instruction_sha256"]) == 64
    assert len(identity["schema_identity"]["structured_verification_json_schema_sha256"]) == 64


def test_paired_metrics_and_freeze_criteria():
    """Verify paired comparison calculations including semantic and error regressions."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q1", "PRIMARY", "C2", HumanEntailment.CONTRADICTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
        _make_dummy_claim_target("q2", "PRIMARY", "C2", HumanEntailment.SUPPORTED),
    ]

    v1_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v1_binary_prediction": "ACCEPT"},
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C2", "is_correct": False, "v1_binary_prediction": "ACCEPT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v1_binary_prediction": "REJECT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C2", "is_correct": True, "v1_binary_prediction": "ACCEPT"},
    ]
    # D2 fixes C2 on q1, errors on C2 on q2
    v2_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v2_d2_binary_prediction": "ACCEPT"},
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C2", "is_correct": True, "v2_d2_binary_prediction": "REJECT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v2_d2_binary_prediction": "REJECT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C2", "is_correct": False, "v2_d2_binary_prediction": "EXECUTION_ERROR"},
    ]

    paired = evaluator._compute_paired_metrics(targets, v1_preds, v2_preds)
    assert paired["both_correct"] == 2
    assert paired["v2_only_correct"] == 1
    assert paired["v1_only_correct"] == 1
    assert paired["net_correctness_delta"] == 0
    assert paired["v2_fixes_count"] == 1
    assert paired["v2_regressions_count"] == 1
    assert paired["semantic_regressions_count"] == 0
    assert paired["execution_error_regressions_count"] == 1
    assert paired["v2_execution_error_count"] == 1


def test_answer_level_execution_error_on_invalid_answer_not_counted_as_invalid_caught():
    """1. human INVALID answer + execution error: MUST NOT increment invalid_answers_caught."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    c1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.CONTRADICTED)
    arm = _make_dummy_arm_target("q1", "PRIMARY", [c1])
    claim_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "EXECUTION_ERROR"}
    ]
    res = evaluator._compute_answer_level_metrics_from_preds([arm], claim_preds, pred_key="v2_d2_three_way_prediction")
    assert res["total_answers"] == 1
    assert res["execution_error_answers"] == 1
    assert res["evaluated_answers"] == 0
    assert res["invalid_answers_caught"] == 0
    assert res["valid_answers_retained"] == 0
    assert res["evaluated_invalid_ground_truth_answers"] == 0
    assert res["evaluated_answer_accuracy"] == 0.0


def test_answer_level_execution_error_on_valid_answer_not_counted_as_valid_retained():
    """2. human VALID answer + execution error: MUST NOT increment valid_answers_retained."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    c1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)
    arm = _make_dummy_arm_target("q1", "PRIMARY", [c1])
    claim_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "EXECUTION_ERROR"}
    ]
    res = evaluator._compute_answer_level_metrics_from_preds([arm], claim_preds, pred_key="v2_d2_three_way_prediction")
    assert res["total_answers"] == 1
    assert res["execution_error_answers"] == 1
    assert res["evaluated_answers"] == 0
    assert res["valid_answers_retained"] == 0
    assert res["invalid_answers_caught"] == 0
    assert res["evaluated_valid_ground_truth_answers"] == 0


def test_answer_level_execution_error_increments_error_count_and_excludes_from_evaluated():
    """3 & 4. answer execution error increments execution_error_answers and is excluded from evaluated denominator."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    arm1 = _make_dummy_arm_target("q1", "PRIMARY", [_make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)])
    arm2 = _make_dummy_arm_target("q2", "PRIMARY", [_make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED)])
    arm3 = _make_dummy_arm_target("q3", "PRIMARY", [_make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.SUPPORTED)])

    claim_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "SUPPORTED"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "EXECUTION_ERROR"},
        {"question_id": "q3", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "INSUFFICIENT"},
    ]
    res = evaluator._compute_answer_level_metrics_from_preds([arm1, arm2, arm3], claim_preds, pred_key="v2_d2_three_way_prediction")
    assert res["total_answers"] == 3
    assert res["execution_error_answers"] == 1
    assert res["evaluated_answers"] == 2
    assert res["evaluated_valid_ground_truth_answers"] == 2
    assert res["evaluated_invalid_ground_truth_answers"] == 0
    assert res["valid_answers_retained"] == 1
    assert res["invalid_answers_caught"] == 0
    assert res["evaluated_answer_accuracy"] == 0.5
    assert res["full_denominator_answer_accuracy"] == round(1 / 3, 4)


def test_clean_answer_metrics_reproduce_canonical_semantics():
    """5. Clean answer metrics reproduce ordinary 22-answer semantics."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    # Build 7 valid arms and 15 invalid arms
    arms = []
    preds = []
    for i in range(7):
        qid = f"v_{i}"
        c = _make_dummy_claim_target(qid, "PRIMARY", "C1", HumanEntailment.SUPPORTED)
        arms.append(_make_dummy_arm_target(qid, "PRIMARY", [c]))
        preds.append({"question_id": qid, "arm_id": "PRIMARY", "claim_id": "C1", "v1_three_way_prediction": "SUPPORTED"})
    for i in range(15):
        qid = f"inv_{i}"
        c = _make_dummy_claim_target(qid, "PRIMARY", "C1", HumanEntailment.CONTRADICTED)
        arms.append(_make_dummy_arm_target(qid, "PRIMARY", [c]))
        # 7 caught, 8 missed
        pred_val = "CONTRADICTED" if i < 7 else "SUPPORTED"
        preds.append({"question_id": qid, "arm_id": "PRIMARY", "claim_id": "C1", "v1_three_way_prediction": pred_val})

    res = evaluator._compute_answer_level_metrics_from_preds(arms, preds, pred_key="v1_three_way_prediction")
    assert res["total_answers"] == 22
    assert res["evaluated_answers"] == 22
    assert res["execution_error_answers"] == 0
    assert res["valid_ground_truth_answers"] == 7
    assert res["invalid_ground_truth_answers"] == 15
    assert res["valid_answers_retained"] == 7
    assert res["invalid_answers_caught"] == 7
    assert res["valid_answer_retention_rate"] == 1.0
    assert res["invalid_answer_catch_rate"] == round(7 / 15, 4)
    assert res["answer_level_accuracy"] == round(14 / 22, 4)


def test_stability_two_pass_distinguishes_errors_from_stable_labels():
    """6, 7, 8, 9. Two-pass stability classifies errors vs stable vs unstable correctly."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    pass1 = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "EXECUTION_ERROR"},  # 6: err+err
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "SUPPORTED"},        # 7: sem+err
        {"question_id": "q3", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "SUPPORTED"},        # 8: supp+supp
        {"question_id": "q4", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "SUPPORTED"},        # 9: supp+insuff
    ]
    pass2 = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "EXECUTION_ERROR"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "EXECUTION_ERROR"},
        {"question_id": "q3", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "SUPPORTED"},
        {"question_id": "q4", "arm_id": "PRIMARY", "claim_id": "C1", "v2_d2_three_way_prediction": "INSUFFICIENT"},
    ]

    res = evaluator._evaluate_stability([pass1, pass2])
    assert res["total_claims"] == 4
    assert res["claims_with_two_valid_semantic_labels"] == 2  # q3 and q4 only
    assert res["stable_semantic_claim_count"] == 1           # q3 only
    assert res["unstable_semantic_claim_count"] == 1         # q4 only
    assert res["successful_label_stability_percentage"] == 50.0
    assert res["pass1_execution_error_count"] == 1           # q1
    assert res["pass2_execution_error_count"] == 2           # q1, q2
    assert res["execution_error_in_any_pass_count"] == 2     # q1, q2
    assert res["repeated_execution_error_claim_count"] == 1  # q1


def test_freeze_gate_execution_errors_and_verdict_hardening():
    """10, 11, 12, 13, 14. Freeze gate hardening across errors, verdict, binary, three-way."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    exec_identity = {"execution_git_commit": "abcdef1234567890abcdef1234567890abcdef12"}
    stability_clean = {
        "unstable_claim_count": 0,
        "unstable_semantic_claim_count": 0,
        "execution_error_in_any_pass_count": 0,
        "label_stability_percentage": 100.0,
    }

    base_metrics = {
        "v1_claim_binary": {"tp": 16, "tn": 7, "fp": 13, "fn": 2},
        "v2_d2_claim_binary": {"tp": 18, "tn": 15, "fp": 5, "fn": 0, "execution_errors": 0},
        "v2_d2_three_way": {"execution_errors": 0},
        "paired_v1_vs_v2_d2": {"net_correctness_delta": 10},
    }

    # 10. Execution error in any pass -> verdict is EXECUTION_ERROR and NEVER freezes
    stability_with_err = dict(stability_clean, execution_error_in_any_pass_count=1)
    _, d10, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_EXECUTION_ERROR",
        execution_identity=exec_identity,
        stability_info=stability_with_err,
        metrics_report=base_metrics,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=38,
        structured_retry_count=0,
    )
    assert d10["development_evaluation_decision"] == "KEEP_ITERATING"

    # 11. Verdict != V2_DEVELOPMENT_BENCHMARK_PASS can NEVER freeze
    _, d11, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_LABEL_INSTABILITY",
        execution_identity=exec_identity,
        stability_info=stability_clean,
        metrics_report=base_metrics,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=38,
        structured_retry_count=0,
    )
    assert d11["development_evaluation_decision"] == "KEEP_ITERATING"

    # 12. Binary execution_errors > 0 can NEVER freeze
    m12 = dict(base_metrics)
    m12["v2_d2_claim_binary"] = {"tp": 18, "tn": 15, "fp": 5, "fn": 0, "execution_errors": 1}
    _, d12, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_BENCHMARK_PASS",
        execution_identity=exec_identity,
        stability_info=stability_clean,
        metrics_report=m12,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=38,
        structured_retry_count=0,
    )
    assert d12["development_evaluation_decision"] == "KEEP_ITERATING"

    # 13. Three-way execution_errors > 0 can NEVER freeze
    m13 = dict(base_metrics)
    m13["v2_d2_three_way"] = {"execution_errors": 1}
    _, d13, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_BENCHMARK_PASS",
        execution_identity=exec_identity,
        stability_info=stability_clean,
        metrics_report=m13,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=38,
        structured_retry_count=0,
    )
    assert d13["development_evaluation_decision"] == "KEEP_ITERATING"

    # 14. Clean improved stable candidate CAN freeze
    _, d14, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_BENCHMARK_PASS",
        execution_identity=exec_identity,
        stability_info=stability_clean,
        metrics_report=base_metrics,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=38,
        structured_retry_count=0,
    )
    assert d14["development_evaluation_decision"] == "CANDIDATE_FREEZE_ELIGIBLE"
    assert d14["promotion_authorized"] is False


def test_dimension_diagnostics_denominator_contract():
    """15. Dimension totals use successfully structured denominator only."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED, tags=["FACTUAL_INCORRECT"]),
        _make_dummy_claim_target("q1", "PRIMARY", "C2", HumanEntailment.CONTRADICTED, tags=["SOURCE_MISATTRIBUTION"]),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
    ]

    preds = [
        {
            "question_id": "q1",
            "arm_id": "PRIMARY",
            "claim_id": "C1",
            "v2_d2_binary_prediction": "ACCEPT",
            "structured_assessment": {
                "claim_id": "C1",
                "actor_role": "ESTABLISHED",
                "action_object": "ESTABLISHED",
                "condition_exception": "ESTABLISHED",
                "quantity_temporal": "ESTABLISHED",
                "negation_modality": "ESTABLISHED",
                "source_article_scope": "ESTABLISHED",
                "evidence_coverage": "COMPLETE",
                "telemetry": {"retry_count": 0, "draft_rejection_categories": []},
            },
        },
        {
            "question_id": "q1",
            "arm_id": "PRIMARY",
            "claim_id": "C2",
            "v2_d2_binary_prediction": "REJECT",
            "structured_assessment": {
                "claim_id": "C2",
                "actor_role": "CONFLICT",
                "action_object": "ESTABLISHED",
                "condition_exception": "ESTABLISHED",
                "quantity_temporal": "ESTABLISHED",
                "negation_modality": "ESTABLISHED",
                "source_article_scope": "CONFLICT",
                "evidence_coverage": "PARTIAL",
                "telemetry": {"retry_count": 1, "draft_rejection_categories": ["MISSING_FIELD"]},
            },
        },
        {
            "question_id": "q2",
            "arm_id": "PRIMARY",
            "claim_id": "C1",
            "v2_d2_binary_prediction": "EXECUTION_ERROR",
            "structured_assessment": None,
        },
    ]

    diag = evaluator._compute_dimension_diagnostics(targets, preds)
    assert diag["total_claims"] == 3
    assert diag["successfully_structured_claim_count"] == 2
    assert diag["execution_error_claim_count"] == 1
    # Check that each dimension sum equals successfully_structured_claim_count (2)
    for dim, counts in diag["global_dimension_counts"].items():
        assert sum(counts.values()) == 2


def test_package_zip_archive_members_inventory(tmp_path: Path):
    """Verify that evidence package ZIP contains all 10 required archive members."""
    out_dir = tmp_path / "out"
    pkg_zip = tmp_path / "pkg.zip"

    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=out_dir,
        package_zip_path=pkg_zip,
    )

    evaluator._write_outputs(
        execution_identity={"test": 1},
        report={"test": 2},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_d2_pass1_preds=[],
        v2_d2_pass2_preds=[],
        comparisons=[],
        dimension_diagnostics={},
        decision_report={"test": 3},
        provider_calls=[],
    )

    assert pkg_zip.is_file()
    required_members = {
        "execution/v2_d2_development_source_identity.json",
        "results/v2_d2_development_report.json",
        "results/v2_d2_development_decision_report.json",
        "results/v2_d2_dimension_diagnostics.json",
        "results/v0_claim_predictions.jsonl",
        "results/v1_claim_predictions.jsonl",
        "results/v2_d2_claim_predictions_pass1.jsonl",
        "results/v2_d2_claim_predictions_pass2.jsonl",
        "results/v2_d2_claim_comparisons.jsonl",
        "telemetry/provider_calls.jsonl",
    }
    with zipfile.ZipFile(pkg_zip, "r") as zf:
        members = set(zf.namelist())
        assert required_members.issubset(members)
