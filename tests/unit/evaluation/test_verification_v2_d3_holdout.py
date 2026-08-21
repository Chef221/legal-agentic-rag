"""Unit tests for V2-D3 Fresh Holdout Benchmark Evaluation Harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.structured_semantic_verifier_d3 import (
    STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION,
    StructuredClaimVerificationD3,
    StructuredSemanticCitationVerifierD3,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    Evidence,
    SemanticSupportLabel,
)
from scripts.evaluate_verification_v2_d3_holdout import (
    CANONICAL_CANDIDATE_ID,
    CANONICAL_D3_EVIDENCE_ZIP_SHA256,
    CANONICAL_D3_IMPLEMENTATION_SHA256,
    CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256,
    CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256,
    CANONICAL_HOLDOUT_SELECTION_SHA256,
    CANONICAL_PACKAGE_VERSION,
    CANONICAL_V3_BACKEND,
    CANONICAL_V3_MODEL_NAME,
    CANONICAL_V3_MODEL_REVISION,
    CANONICAL_V3_PROVIDER_VERSION,
    GATE_MIN_CLAIM_BINARY_ACCURACY_RATE,
    GATE_MIN_FULL_ANSWER_ACCURACY_RATE,
    GATE_MIN_NEGATIVE_CATCH_RATE,
    GATE_MIN_SUPPORTED_RETENTION_RATE,
    GATE_MIN_VALID_ANSWER_RETENTION_RATE,
    BenchmarkArmTarget,
    BenchmarkClaimTarget,
    BinaryPrediction,
    HumanEntailment,
    ObservationalChatModelProviderWrapper,
    V2D3HoldoutBenchmarkEvaluator,
)


class MockCanonicalD3Provider(ChatModelProvider):
    """Synthetic provider simulating Qwen responses for frozen V2-D3."""

    def __init__(self, mode: str = "perfect", model_name: str = CANONICAL_V3_MODEL_NAME) -> None:
        self._mode = mode
        self._model_name = model_name
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return CANONICAL_V3_BACKEND

    @property
    def provider_version(self) -> str:
        return CANONICAL_V3_PROVIDER_VERSION

    @property
    def model_name(self) -> str:
        return self._model_name

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

        if self._mode == "perfect":
            return json.dumps({
                "claim_id": cid,
                "relation": "ENTAILS",
                "actor_mismatch": False,
                "condition_exception_mismatch": False,
                "quantity_temporal_mismatch": False,
                "negation_modality_mismatch": False,
                "source_scope_mismatch": False,
            })
        elif self._mode == "contradicted":
            return json.dumps({
                "claim_id": cid,
                "relation": "CONTRADICTS",
                "actor_mismatch": False,
                "condition_exception_mismatch": False,
                "quantity_temporal_mismatch": False,
                "negation_modality_mismatch": False,
                "source_scope_mismatch": False,
            })
        return json.dumps({
            "claim_id": cid,
            "relation": "DOES_NOT_ESTABLISH",
            "actor_mismatch": False,
            "condition_exception_mismatch": False,
            "quantity_temporal_mismatch": False,
            "negation_modality_mismatch": False,
            "source_scope_mismatch": False,
        })


@pytest.fixture
def mock_holdout_sources(tmp_path: Path) -> dict[str, Path]:
    """Create minimal synthetic holdout packet archive, selection file, and label file."""
    h_zip = tmp_path / "mock_holdout_packets.zip"
    h_sel = tmp_path / "mock_holdout_selection.json"
    h_lbl = tmp_path / "mock_holdout_labels.json"

    # Synthetic review packet with schema-compliant AnswerResponse and Evidence
    synthetic_packet = {
        "question_id": "SYNTH_Q1",
        "stratum": "A_SINGLE_CLAIM_CLEAN",
        "question_text": "Sample synthetic legal question?",
        "arms": {
            "BASE": {
                "historical_stop_reason": "answer_verified",
                "answer_response": {
                    "question": "Sample synthetic legal question?",
                    "answer": "Synthetic legal answer text [E1].",
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "CHK_001",
                            "document_id": "DOC_001",
                        }
                    ],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": "synth_trace",
                },
                "evidence_list": [
                    {
                        "evidence_id": "E1",
                        "chunk_id": "CHK_001",
                        "document_id": "DOC_001",
                        "text": "Synthetic legal answer text passage establishing claim 1.",
                        "metadata": {},
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C1",
                        "claim_text": "Synthetic legal answer text.",
                        "entailment_label": "SUPPORTED",
                        "error_tags": [],
                    }
                ],
                "historical_verification": {
                    "is_valid": True,
                    "v0_rule_verifier": {
                        "all_citations_supported": True,
                        "verified_citations_count": 1,
                        "unsupported_citations_count": 0,
                    },
                },
            }
        },
    }

    with zipfile.ZipFile(h_zip, "w") as zf:
        zf.writestr("packets/SYNTH_Q1.json", json.dumps(synthetic_packet))

    h_sel.write_text(json.dumps({"schema_version": "1.0", "holdout_count": 1}), encoding="utf-8")
    h_lbl.write_text(
        json.dumps({
            "schema_version": "1.0",
            "questions": {
                "SYNTH_Q1": {
                    "arms": {
                        "BASE": {
                            "claims": {
                                "C1": {
                                    "entailment_label": "SUPPORTED",
                                    "error_tags": [],
                                }
                            }
                        }
                    }
                }
            },
        }),
        encoding="utf-8",
    )

    return {
        "holdout_packets": h_zip,
        "holdout_selection": h_sel,
        "holdout_labels": h_lbl,
    }


# Test 1: Canonical Source SHA Verification Fail-Closed
def test_holdout_source_checksum_failure(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=False,
    )
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        evaluator._verify_canonical_source_checksums()


# Test 2: Candidate ID and Package Version Validation
def test_candidate_id_and_package_version_validation(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        candidate_id="WRONG_CANDIDATE",
        bypass_source_checksums=True,
    )
    with pytest.raises(DataValidationError, match="Candidate ID mismatch"):
        evaluator._validate_canonical_provenance()


# Test 3: Frozen D3 Source and Prompt Identity Verification
def test_frozen_d3_source_and_prompt_identity_verification(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    exec_id = evaluator._build_execution_identity(sources_info={})
    assert exec_id["frozen_d3_source_identity_verified"] is True
    assert exec_id["prompt_identities"]["d3_base_system_instruction_sha256"] == CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256
    assert exec_id["implementation_identities"]["structured_semantic_verifier_d3_sha256"] == CANONICAL_D3_IMPLEMENTATION_SHA256


# Test 4: Provider Identity Validation
def test_provider_identity_validation(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    bad_provider = MockCanonicalD3Provider(model_name="Wrong/Model")

    with pytest.raises(DataValidationError, match="Model name mismatch"):
        evaluator._validate_runtime_provider_identity(bad_provider)


# Test 5: Target Loading and V0 Replay Verification
def test_holdout_target_loading_and_v0_replay(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    arm_targets, claim_targets = evaluator._load_holdout_targets()
    assert len(arm_targets) == 1
    assert len(claim_targets) == 1
    assert claim_targets[0].human_label == HumanEntailment.SUPPORTED

    v0_arms, v0_claims, stats = evaluator._replay_v0_verifier(arm_targets)
    assert stats["total_arms"] == 1
    assert stats["v0_replay_arm_passes"] == 1
    assert stats["v0_historical_fidelity_matches"] == 1
    assert len(v0_claims) == 1
    assert v0_claims[0]["v0_binary_prediction"] == BinaryPrediction.ACCEPT


# Test 6: Model-Free Preflight Mode
def test_model_free_preflight_mode(mock_holdout_sources, tmp_path):
    out_dir = tmp_path / "preflight_out"
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=out_dir,
        preflight_only=True,
        bypass_source_checksums=True,
    )
    rep = evaluator.evaluate()
    assert rep["verdict"] == "V2_D3_HOLDOUT_BENCHMARK_READY"
    assert rep["total_claims"] == 1
    assert rep["total_answers"] == 1
    assert rep["preflight_status"]["model_execution_skipped"] is True
    assert (out_dir / "results/v2_d3_holdout_report.json").is_file()


# Test 7: Observational Provider Content Safety & Call Telemetry
def test_observational_provider_telemetry():
    inner = MockCanonicalD3Provider(mode="perfect")
    obs = ObservationalChatModelProviderWrapper(inner)

    res = obs.complete(
        system_instruction=STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION,
        user_prompt="Claim ID: C1\nClaim: Some secret text\nPassage: Some secret passage",
    )
    assert obs.total_calls == 1
    assert obs.total_errors == 0

    call_entry = obs.call_history[0]
    assert call_entry["call_index"] == 1
    assert call_entry["success"] is True
    assert "user_prompt_sha256" in call_entry
    assert "completion_sha256" in call_entry
    # Invariant: No raw prompt or completion text stored in call record
    assert "Some secret text" not in json.dumps(call_entry)
    assert "Some secret passage" not in json.dumps(call_entry)


# Test 8: End-to-End Evaluation with Synthetic Provider (Pass Scenario)
def test_end_to_end_holdout_evaluation_pass(mock_holdout_sources, tmp_path):
    out_dir = tmp_path / "eval_out"
    pkg_zip = tmp_path / "evidence.zip"
    provider = MockCanonicalD3Provider(mode="perfect")

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=out_dir,
        package_zip=pkg_zip,
        custom_provider=provider,
        bypass_source_checksums=True,
    )

    report = evaluator.evaluate()
    assert report["verdict"] == "V2_D3_HOLDOUT_PROMOTION_RECOMMENDED"
    assert report["telemetry"]["total_provider_calls"] == 2  # 1 claim * 2 passes
    assert report["stability"]["unstable_semantic_claim_count"] == 0

    dec_path = out_dir / "results/v2_d3_holdout_decision_report.json"
    assert dec_path.is_file()
    dec = json.loads(dec_path.read_text(encoding="utf-8"))
    assert dec["holdout_evaluation_decision"] == "PROMOTE_V2_D3_TO_PRODUCTION"
    assert dec["promotion_recommended"] is True
    assert dec["promotion_authorized"] is False  # Fail-closed invariant
    assert dec["production_action_required"] == "PENDING_HUMAN_GOVERNANCE_SIGN_OFF"
    assert dec["pre_registered_gate_evaluations"]["mechanical_gates_passed"] is True
    assert dec["pre_registered_gate_evaluations"]["supported_retention_passed"] is True

    assert pkg_zip.is_file()


# Test 9: Promotion Rejection on Rate Failure
def test_promotion_rejection_on_rate_failure(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )

    # Synthetic metrics failing supported retention rate
    all_metrics = {
        "v0_claim_binary": {},
        "v2_d3_claim_binary": {
            "tp": 5, "fp": 10, "tn": 10, "fn": 15, "execution_errors": 0,
            "evaluated_claims": 40, "total_claims": 40,
            "accuracy": 0.375, "precision": 0.333,
            "supported_retention": 0.25,  # Fails < 0.88
            "negative_catch": 0.50,
            "f1": 0.285, "balanced_accuracy": 0.375,
        },
        "v2_d3_three_way": {"accuracy": 0.375},
        "v2_d3_answer_metrics": {
            "valid_answer_retention_rate": 0.25,  # Fails < 0.80
            "invalid_answer_catch_rate": 0.50,
            "full_denominator_answer_accuracy": 0.35,  # Fails < 0.60
            "valid_answers_retained": 5, "gold_valid_answers_count": 20,
            "invalid_answers_caught": 10, "gold_invalid_answers_count": 20,
            "total_answers": 40, "evaluated_answers": 40, "execution_error_answers": 0,
        },
    }
    stability_info = {
        "total_claims": 40,
        "claims_with_two_valid_semantic_labels": 40,
        "stable_semantic_claim_count": 40,
        "unstable_semantic_claim_count": 0,
        "execution_error_in_any_pass_count": 0,
    }
    pass_telem = {"provider_calls": 40, "provider_invocation_errors": 0}

    final_rep, dec_rep, stab_rep = evaluator._build_reports(
        sources_info={},
        exec_identity={"frozen_d3_source_identity_verified": True},
        claim_targets=[MagicMock()] * 40,
        arm_targets=[MagicMock()] * 40,
        stability_info=stability_info,
        all_metrics=all_metrics,
        pass1_telemetry=pass_telem,
        pass2_telemetry=pass_telem,
        total_duration=1.0,
    )

    assert final_rep["verdict"] == "V2_D3_HOLDOUT_PROMOTION_REJECTED"
    assert dec_rep["holdout_evaluation_decision"] == "REJECT_V2_D3_PROMOTION"
    assert dec_rep["promotion_recommended"] is False
    assert dec_rep["promotion_authorized"] is False
    assert dec_rep["pre_registered_gate_evaluations"]["mechanical_gates_passed"] is True
    assert dec_rep["pre_registered_gate_evaluations"]["supported_retention_passed"] is False


# Test 10: Mechanical Execution Failure Handling
def test_mechanical_execution_failure_handling(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )

    all_metrics = {
        "v0_claim_binary": {},
        "v2_d3_claim_binary": {"accuracy": 0.95, "supported_retention": 0.95, "negative_catch": 0.95, "evaluated_claims": 38, "total_claims": 38, "tp": 18, "tn": 18, "fp": 1, "fn": 1},
        "v2_d3_three_way": {"accuracy": 0.95},
        "v2_d3_answer_metrics": {"valid_answer_retention_rate": 0.95, "full_denominator_answer_accuracy": 0.90, "valid_answers_retained": 10, "gold_valid_answers_count": 10, "invalid_answers_caught": 10, "gold_invalid_answers_count": 10, "total_answers": 20},
    }
    # 2 unstable claims
    stability_info = {
        "total_claims": 38,
        "claims_with_two_valid_semantic_labels": 38,
        "stable_semantic_claim_count": 36,
        "unstable_semantic_claim_count": 2,
        "execution_error_in_any_pass_count": 0,
    }
    pass_telem = {"provider_calls": 38, "provider_invocation_errors": 0}

    final_rep, dec_rep, stab_rep = evaluator._build_reports(
        sources_info={},
        exec_identity={"frozen_d3_source_identity_verified": True},
        claim_targets=[MagicMock()] * 38,
        arm_targets=[MagicMock()] * 20,
        stability_info=stability_info,
        all_metrics=all_metrics,
        pass1_telemetry=pass_telem,
        pass2_telemetry=pass_telem,
        total_duration=1.0,
    )

    assert final_rep["verdict"] == "V2_D3_HOLDOUT_EXECUTION_FAILURE"
    assert dec_rep["holdout_evaluation_decision"] == "REJECT_V2_D3_PROMOTION"
    assert dec_rep["promotion_recommended"] is False
    assert dec_rep["pre_registered_gate_evaluations"]["mechanical_gates_passed"] is False
    assert dec_rep["pre_registered_gate_evaluations"]["zero_unstable_claims"] is False


# Test 11: Holdout Evidence Package Canonical Inventory
def test_holdout_evidence_package_canonical_inventory(mock_holdout_sources, tmp_path):
    out_dir = tmp_path / "out"
    pkg_zip = tmp_path / "holdout_evidence.zip"

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=out_dir,
        package_zip=pkg_zip,
        bypass_source_checksums=True,
    )

    report = {"schema_version": "1.0", "verdict": "V2_D3_HOLDOUT_PROMOTION_RECOMMENDED"}
    dec_report = {"schema_version": "1.0", "holdout_evaluation_decision": "PROMOTE_V2_D3_TO_PRODUCTION"}
    stab_report = {"schema_version": "1.0", "stability_summary": {}}
    exec_id = {"schema_version": "1.0", "candidate_id": CANONICAL_CANDIDATE_ID}

    evaluator._write_reports(
        report=report,
        decision_report=dec_report,
        stability_report=stab_report,
        v0_claim_preds=[{"claim_id": "C1"}],
        pass1_claim_preds=[{"question_id": "Q1", "arm_id": "BASE", "claim_id": "C1", "v2_d3_binary_prediction": "ACCEPT", "v2_d3_three_way_prediction": "SUPPORTED"}],
        pass2_claim_preds=[{"question_id": "Q1", "arm_id": "BASE", "claim_id": "C1", "v2_d3_binary_prediction": "ACCEPT", "v2_d3_three_way_prediction": "SUPPORTED"}],
        exec_identity=exec_id,
        provider=None,
        is_preflight=False,
    )

    assert pkg_zip.is_file()
    with zipfile.ZipFile(pkg_zip, "r") as zf:
        members = set(zf.namelist())

    expected_members = {
        "execution/v2_d3_holdout_source_identity.json",
        "results/v2_d3_holdout_report.json",
        "results/v2_d3_holdout_decision_report.json",
        "results/v2_d3_holdout_stability_report.json",
        "results/v0_claim_predictions.jsonl",
        "results/v2_d3_holdout_claim_predictions_pass1.jsonl",
        "results/v2_d3_holdout_claim_predictions_pass2.jsonl",
        "results/v2_d3_holdout_claim_comparisons.jsonl",
    }
    for m in expected_members:
        assert m in members, f"Expected canonical archive member '{m}' missing"


# Test 12: Prompt Invariants (Zero Leak of Human Labels / Dev Predictions)
def test_holdout_prompt_invariants_no_label_or_reference_leak(mock_holdout_sources, tmp_path):
    observed_prompts: list[str] = []

    class InspectingProvider(ChatModelProvider):
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
            observed_prompts.append(user_prompt)
            return json.dumps({
                "claim_id": "C1",
                "relation": "ENTAILS",
                "actor_mismatch": False,
                "condition_exception_mismatch": False,
                "quantity_temporal_mismatch": False,
                "negation_modality_mismatch": False,
                "source_scope_mismatch": False,
            })

    provider = InspectingProvider()
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "eval_out",
        custom_provider=provider,
        bypass_source_checksums=True,
    )
    evaluator.evaluate()

    assert len(observed_prompts) == 2  # Pass 1 + Pass 2
    for prompt in observed_prompts:
        assert "Human Label" not in prompt
        assert "entailment_label" not in prompt
        assert "gold" not in prompt.lower()
        assert "ground_truth" not in prompt.lower()
        assert "v1_baseline" not in prompt.lower()
        assert "d3.1" not in prompt.lower()
        assert "d3.2" not in prompt.lower()


# Test 13: Model Revision Validation Fail-Closed
def test_provider_model_revision_validation(mock_holdout_sources, tmp_path):
    class BadRevisionProvider(MockCanonicalD3Provider):
        @property
        def model_revision(self) -> str:
            return "wrong_revision_sha_123"

    bad_rev_provider = BadRevisionProvider()
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    with pytest.raises(DataValidationError, match="Model revision mismatch"):
        evaluator._validate_runtime_provider_identity(bad_rev_provider)

