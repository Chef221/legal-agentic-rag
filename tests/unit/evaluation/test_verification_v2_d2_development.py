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
    """Verify paired comparison calculations and freeze criteria."""
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
    ]

    v1_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v1_binary_prediction": "ACCEPT"},
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C2", "is_correct": False, "v1_binary_prediction": "ACCEPT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v1_binary_prediction": "REJECT"},
    ]
    # D2 fixes C2
    v2_preds = [
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v2_d2_binary_prediction": "ACCEPT"},
        {"question_id": "q1", "arm_id": "PRIMARY", "claim_id": "C2", "is_correct": True, "v2_d2_binary_prediction": "REJECT"},
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "is_correct": True, "v2_d2_binary_prediction": "REJECT"},
    ]

    paired = evaluator._compute_paired_metrics(targets, v1_preds, v2_preds)
    assert paired["both_correct"] == 2
    assert paired["v2_only_correct"] == 1
    assert paired["v1_only_correct"] == 0
    assert paired["net_correctness_delta"] == 1
    assert paired["v2_fixes_count"] == 1
    assert paired["v2_regressions_count"] == 0


def test_freeze_eligibility_gating():
    """Verify strict freeze gating logic."""
    evaluator = V2D2DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    exec_identity = {"execution_git_commit": "abcdef1234567890abcdef1234567890abcdef12"}
    stability_clean = {"unstable_claim_count": 0, "label_stability_percentage": 100.0}
    stability_dirty = {"unstable_claim_count": 1, "label_stability_percentage": 97.37}

    # Case 1: Model errors > 0 -> KEEP_ITERATING
    metrics_report = {
        "v1_claim_binary": {"tp": 16, "tn": 7, "fp": 13, "fn": 2},
        "v2_d2_claim_binary": {"tp": 18, "tn": 15, "fp": 5, "fn": 0},
        "paired_v1_vs_v2_d2": {"net_correctness_delta": 10},
    }
    _, dec1, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_EXECUTION_ERROR",
        execution_identity=exec_identity,
        stability_info=stability_clean,
        metrics_report=metrics_report,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=2,
        total_provider_calls=40,
        structured_retry_count=2,
    )
    assert dec1["development_evaluation_decision"] == "KEEP_ITERATING"
    assert dec1["promotion_authorized"] is False

    # Case 2: Clean pass with correct > 23, tn > 7, tp >= 16, delta > 0 -> CANDIDATE_FREEZE_ELIGIBLE
    _, dec2, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_BENCHMARK_PASS",
        execution_identity=exec_identity,
        stability_info=stability_clean,
        metrics_report=metrics_report,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=38,
        structured_retry_count=0,
    )
    assert dec2["development_evaluation_decision"] == "CANDIDATE_FREEZE_ELIGIBLE"
    assert dec2["promotion_authorized"] is False

    # Case 3: Label instability -> KEEP_ITERATING
    _, dec3, _ = evaluator._build_reports(
        verdict="V2_DEVELOPMENT_LABEL_INSTABILITY",
        execution_identity=exec_identity,
        stability_info=stability_dirty,
        metrics_report=metrics_report,
        dimension_diagnostics={},
        v0_claim_preds=[],
        v1_claim_preds=[],
        v2_pass1_preds=[],
        all_claim_targets=[],
        model_error_count=0,
        total_provider_calls=38,
        structured_retry_count=0,
    )
    assert dec3["development_evaluation_decision"] == "KEEP_ITERATING"


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
