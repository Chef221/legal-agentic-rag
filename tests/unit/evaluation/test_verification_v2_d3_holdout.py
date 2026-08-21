"""Unit tests for V2-D3 Fresh Holdout Benchmark Evaluation Harness."""

from __future__ import annotations

from hashlib import sha256
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

        if self._mode == "gold":
            if "CHK_001" in user_prompt or "question 1" in user_prompt.lower():
                return json.dumps({
                    "claim_id": cid,
                    "relation": "ENTAILS",
                    "actor_mismatch": False,
                    "condition_exception_mismatch": False,
                    "quantity_temporal_mismatch": False,
                    "negation_modality_mismatch": False,
                    "source_scope_mismatch": False,
                })
            else:
                return json.dumps({
                    "claim_id": cid,
                    "relation": "CONTRADICTS",
                    "actor_mismatch": False,
                    "condition_exception_mismatch": False,
                    "quantity_temporal_mismatch": False,
                    "negation_modality_mismatch": False,
                    "source_scope_mismatch": False,
                })
        elif self._mode == "perfect":
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
    h_cmt = tmp_path / "mock_holdout_commitment.json"

    # Synthetic review packet 1: Valid Answer (SUPPORTED claim)
    packet_1 = {
        "question_id": "SYNTH_Q1",
        "stratum": "A_SINGLE_CLAIM_CLEAN",
        "question_text": "Sample synthetic legal question 1?",
        "arms": {
            "BASE": {
                "historical_stop_reason": "answer_verified",
                "answer_response": {
                    "question": "Sample synthetic legal question 1?",
                    "answer": "Synthetic legal answer text 1 [E1].",
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "CHK_001",
                            "document_id": "DOC_001",
                        }
                    ],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": "synth_trace_1",
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
                        "claim_text": "Synthetic supported claim text.",
                        "entailment_label": "SUPPORTED",
                        "error_tags": [],
                    }
                ],
                "historical_verification": {
                    "is_valid": True,
                    "all_citations_supported": True,
                },
            }
        },
    }

    # Synthetic review packet 2: Invalid Answer (CONTRADICTED claim)
    packet_2 = {
        "question_id": "SYNTH_Q2",
        "stratum": "D_NEGATION_MODALITY",
        "question_text": "Sample synthetic legal question 2?",
        "arms": {
            "PRIMARY": {
                "historical_stop_reason": "answer_verified",
                "answer_response": {
                    "question": "Sample synthetic legal question 2?",
                    "answer": "Synthetic legal answer text 2 [E1].",
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "CHK_002",
                            "document_id": "DOC_002",
                        }
                    ],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": "synth_trace_2",
                },
                "evidence_list": [
                    {
                        "evidence_id": "E1",
                        "chunk_id": "CHK_002",
                        "document_id": "DOC_002",
                        "text": "Synthetic legal answer text passage contradicting claim 2.",
                        "metadata": {},
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C1",
                        "claim_text": "Synthetic contradicted claim text.",
                        "entailment_label": "CONTRADICTED",
                        "error_tags": ["NEGATION_INVERTED"],
                    }
                ],
                "historical_verification": {
                    "is_valid": True,
                    "all_citations_supported": True,
                },
            }
        },
    }

    with zipfile.ZipFile(h_zip, "w") as zf:
        zf.writestr("packets/SYNTH_Q1.json", json.dumps(packet_1))
        zf.writestr("packets/SYNTH_Q2.json", json.dumps(packet_2))

    h_sel.write_text(json.dumps({"schema_version": "1.0", "holdout_count": 2}), encoding="utf-8")

    labels_content = {
        "schema_version": "1.0",
        "artifact_type": "verification_v2_holdout_reviewed_labels",
        "review_status": "frozen_human_reviewed",
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": {
                            "C1": {
                                "entailment_label": "SUPPORTED",
                                "claim_text_sha256": sha256("Synthetic supported claim text.".encode("utf-8")).hexdigest(),
                                "error_tags": [],
                            }
                        }
                    }
                }
            },
            "SYNTH_Q2": {
                "arms": {
                    "PRIMARY": {
                        "claims": {
                            "C1": {
                                "entailment_label": "CONTRADICTED",
                                "claim_text_sha256": sha256("Synthetic contradicted claim text.".encode("utf-8")).hexdigest(),
                                "error_tags": ["NEGATION_INVERTED"],
                            }
                        }
                    }
                }
            },
        },
    }
    h_lbl.write_text(json.dumps(labels_content, indent=2), encoding="utf-8")

    packets_sha = sha256(h_zip.read_bytes()).hexdigest()
    selection_sha = sha256(h_sel.read_bytes()).hexdigest()
    labels_sha = sha256(h_lbl.read_bytes()).hexdigest()
    labels_size = h_lbl.stat().st_size

    commitment_content = {
        "schema_version": "1.0",
        "artifact_type": "verification_v2_holdout_label_commitment",
        "artifact_filename": h_lbl.name,
        "labels_sha256": labels_sha,
        "labels_size_bytes": labels_size,
        "total_questions": 2,
        "total_arms": 2,
        "total_claims": 2,
        "class_counts": {
            "SUPPORTED": 1,
            "CONTRADICTED": 1,
            "INSUFFICIENT": 0,
        },
        "review_status": "frozen_human_reviewed",
        "holdout_packets_sha256": packets_sha,
        "holdout_selection_sha256": selection_sha,
        "review_timestamp": "2026-08-21T00:00:00Z",
        "reviewer_governance_status": "GOVERNANCE_REVIEWED_AND_COMMITTED",
    }
    h_cmt.write_text(json.dumps(commitment_content, indent=2), encoding="utf-8")

    return {
        "holdout_packets": h_zip,
        "holdout_selection": h_sel,
        "holdout_labels": h_lbl,
        "label_commitment": h_cmt,
    }


# Test 1: Canonical Source SHA Verification Fail-Closed
def test_holdout_source_checksum_failure(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        label_commitment_path=mock_holdout_sources["label_commitment"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=False,
    )
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        evaluator._verify_canonical_source_checksums()


# Test 2: Canonical execution blocked if no label commitment or SHA provided
def test_canonical_execution_blocked_without_label_commitment(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        label_commitment_path=None,
        output_dir=tmp_path / "out",
        bypass_source_checksums=False,
    )
    # Patched canonical checksums so packets & selection pass
    with patch(
        "scripts.evaluate_verification_v2_d3_holdout.CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256",
        sha256(mock_holdout_sources["holdout_packets"].read_bytes()).hexdigest(),
    ), patch(
        "scripts.evaluate_verification_v2_d3_holdout.CANONICAL_HOLDOUT_SELECTION_SHA256",
        sha256(mock_holdout_sources["holdout_selection"].read_bytes()).hexdigest(),
    ):
        with pytest.raises(DataValidationError, match="CANONICAL_HOLDOUT_EXECUTION_BLOCKED"):
            evaluator._verify_canonical_source_checksums()


# Test 3: Candidate ID and Package Version Validation
def test_candidate_id_validation(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        candidate_id="WRONG_CANDIDATE",
        bypass_source_checksums=True,
    )
    with pytest.raises(DataValidationError, match="Candidate mismatch"):
        evaluator._validate_canonical_provenance()


# Test 4: Frozen D3 Source and Prompt Identity Verification
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


# Test 5: Target Loading and Exact Set Equality (No Missing Labels)
def test_holdout_target_loading_exact_set_equality(mock_holdout_sources, tmp_path):
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    arm_targets, claim_targets = evaluator._load_holdout_targets()
    assert len(arm_targets) == 2
    assert len(claim_targets) == 2
    assert claim_targets[0].human_label == HumanEntailment.SUPPORTED
    assert claim_targets[1].human_label == HumanEntailment.CONTRADICTED


# Test 6: Target Loading Missing Label Fails Closed (No Fallbacks!)
def test_holdout_target_loading_missing_label_fails(mock_holdout_sources, tmp_path):
    incomplete_lbl = tmp_path / "incomplete_labels.json"
    incomplete_lbl.write_text(
        json.dumps({
            "questions": {
                "SYNTH_Q1": {
                    "arms": {
                        "BASE": {
                            "claims": {
                                "C1": {"entailment_label": "SUPPORTED"}
                            }
                        }
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=incomplete_lbl,
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    with pytest.raises(DataValidationError, match="HOLD_OUT_LABEL_MISSING"):
        evaluator._load_holdout_targets()


# Test 7: Target Loading Extra Label Fails Closed
def test_holdout_target_loading_extra_label_fails(mock_holdout_sources, tmp_path):
    extra_lbl = tmp_path / "extra_labels.json"
    extra_lbl.write_text(
        json.dumps({
            "questions": {
                "SYNTH_Q1": {
                    "arms": {
                        "BASE": {
                            "claims": {
                                "C1": {"entailment_label": "SUPPORTED"}
                            }
                        }
                    }
                },
                "SYNTH_Q2": {
                    "arms": {
                        "PRIMARY": {
                            "claims": {
                                "C1": {"entailment_label": "CONTRADICTED"}
                            }
                        }
                    }
                },
                "SYNTH_Q3_EXTRA": {
                    "arms": {
                        "BASE": {
                            "claims": {
                                "C1": {"entailment_label": "SUPPORTED"}
                            }
                        }
                    }
                },
            }
        }),
        encoding="utf-8",
    )
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=extra_lbl,
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    with pytest.raises(DataValidationError, match="HOLD_OUT_LABEL_EXTRA"):
        evaluator._load_holdout_targets()


# Test 8: Target Loading Invalid Label String Fails Closed
def test_holdout_target_loading_invalid_label_fails(mock_holdout_sources, tmp_path):
    invalid_lbl = tmp_path / "invalid_labels.json"
    invalid_lbl.write_text(
        json.dumps({
            "questions": {
                "SYNTH_Q1": {
                    "arms": {
                        "BASE": {
                            "claims": {
                                "C1": {"entailment_label": "INVALID_LABEL_VALUE"}
                            }
                        }
                    }
                },
                "SYNTH_Q2": {
                    "arms": {
                        "PRIMARY": {
                            "claims": {
                                "C1": {"entailment_label": "CONTRADICTED"}
                            }
                        }
                    }
                },
            }
        }),
        encoding="utf-8",
    )
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=invalid_lbl,
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    with pytest.raises(DataValidationError, match="HOLD_OUT_LABEL_INVALID"):
        evaluator._load_holdout_targets()


# Test 9: Model-Free Preflight Mode
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
    assert rep["total_claims"] == 2
    assert rep["total_answers"] == 2
    assert rep["coverage"]["coverage_sufficient"] is True
    assert rep["preflight_status"]["model_execution_skipped"] is True
    assert (out_dir / "results/v2_d3_holdout_report.json").is_file()


# Test 10: Non-Vacuous Coverage Gate - Zero Negative Denominator
def test_zero_negative_denominator_coverage_insufficient(mock_holdout_sources, tmp_path):
    # Only 1 supported claim -> negative claims == 0
    single_lbl = tmp_path / "single_supported_labels.json"
    single_lbl.write_text(
        json.dumps({
            "questions": {
                "SYNTH_Q1": {
                    "arms": {
                        "BASE": {
                            "claims": {
                                "C1": {"entailment_label": "SUPPORTED"}
                            }
                        }
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    # Create single-arm packet zip
    single_zip = tmp_path / "single_packet.zip"
    with zipfile.ZipFile(mock_holdout_sources["holdout_packets"], "r") as zf_in:
        with zipfile.ZipFile(single_zip, "w") as zf_out:
            zf_out.writestr("packets/SYNTH_Q1.json", zf_in.read("packets/SYNTH_Q1.json"))

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=single_zip,
        holdout_labels_path=single_lbl,
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        custom_provider=MockCanonicalD3Provider(mode="perfect"),
        bypass_source_checksums=True,
    )
    rep = evaluator.evaluate()
    assert rep["verdict"] == "V2_D3_HOLDOUT_COVERAGE_INSUFFICIENT"
    assert rep["coverage"]["coverage_sufficient"] is False
    assert rep["coverage"]["negative_claims_denominator_valid"] is False
    assert rep["metrics"]["v2_d3_claim_binary"]["negative_catch"] is None


# Test 11: Observational Telemetry Wrapper Content Safety
def test_observational_telemetry_wrapper_safety():
    mock_inner = MockCanonicalD3Provider(mode="perfect")
    wrapper = ObservationalChatModelProviderWrapper(mock_inner)

    res = wrapper.complete(
        system_instruction=STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION,
        user_prompt="[RAW LEGAL PASSAGE SENSITIVE TEXT]",
    )
    assert len(wrapper.call_history) == 1
    call = wrapper.call_history[0]
    assert call["system_instruction_sha256"] == CANONICAL_D3_SYSTEM_INSTRUCTION_SHA256
    assert "user_prompt_sha256" in call
    assert "completion_sha256" in call
    assert "RAW LEGAL PASSAGE" not in json.dumps(call)


# Test 12: End-to-End Evaluation Pass (Authoritative Pass 1 + Stability Pass 2)
def test_end_to_end_holdout_evaluation_pass(mock_holdout_sources, tmp_path):
    out_dir = tmp_path / "eval_out"
    pkg_zip = tmp_path / "evidence.zip"
    mock_provider = MockCanonicalD3Provider(mode="gold")

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=out_dir,
        package_zip=pkg_zip,
        custom_provider=mock_provider,
        bypass_source_checksums=True,
    )
    report = evaluator.evaluate()
    dec_path = out_dir / "results/v2_d3_holdout_decision_report.json"
    if dec_path.is_file():
        print("DECISION REPORT:", dec_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "V2_D3_HOLDOUT_PROMOTION_RECOMMENDED"
    assert report["telemetry"]["total_provider_calls"] == 4  # 2 claims * 2 passes
    assert pkg_zip.is_file()
    assert (out_dir / "results/v2_d3_holdout_decision_report.json").is_file()


# Test 13: Mechanical Failure Handling
def test_mechanical_failure_handling(mock_holdout_sources, tmp_path):
    class ErrorProvider(ChatModelProvider):
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
            raise ModelError("Simulated CUDA out of memory")

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "err_out",
        custom_provider=ErrorProvider(),
        bypass_source_checksums=True,
    )
    rep = evaluator.evaluate()
    assert rep["verdict"] == "V2_D3_HOLDOUT_EXECUTION_FAILURE"


# Test 14: Packet embedded label cannot act as gold fallback
def test_packet_embedded_label_not_used_as_fallback(mock_holdout_sources, tmp_path):
    # Packet has entailment_label: "SUPPORTED", but label file is missing the claim
    empty_lbl = tmp_path / "empty_labels.json"
    empty_lbl.write_text(json.dumps({"questions": {}}), encoding="utf-8")

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=empty_lbl,
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        bypass_source_checksums=True,
    )
    with pytest.raises(DataValidationError, match="HOLD_OUT_LABEL_MISSING"):
        evaluator._load_holdout_targets()


# Test 15: Wrong labels SHA in commitment fails closed
def test_wrong_labels_sha_in_commitment_fails(mock_holdout_sources, tmp_path):
    bad_cmt = tmp_path / "bad_commitment.json"
    bad_cmt.write_text(
        json.dumps({
            "schema_version": "1.0",
            "artifact_type": "verification_v2_holdout_label_commitment",
            "labels_sha256": "wrong_sha_digest_00000000000000000000000000000000000000000000",
            "labels_size_bytes": mock_holdout_sources["holdout_labels"].stat().st_size,
            "holdout_packets_sha256": sha256(mock_holdout_sources["holdout_packets"].read_bytes()).hexdigest(),
            "holdout_selection_sha256": sha256(mock_holdout_sources["holdout_selection"].read_bytes()).hexdigest(),
            "review_status": "frozen_human_reviewed",
        }),
        encoding="utf-8",
    )

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        label_commitment_path=bad_cmt,
        output_dir=tmp_path / "out",
        bypass_source_checksums=False,
    )
    with patch(
        "scripts.evaluate_verification_v2_d3_holdout.CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256",
        sha256(mock_holdout_sources["holdout_packets"].read_bytes()).hexdigest(),
    ), patch(
        "scripts.evaluate_verification_v2_d3_holdout.CANONICAL_HOLDOUT_SELECTION_SHA256",
        sha256(mock_holdout_sources["holdout_selection"].read_bytes()).hexdigest(),
    ):
        with pytest.raises(DataValidationError, match="SHA-256 mismatch for holdout_labels"):
            evaluator._verify_canonical_source_checksums()


# Test 16: Zero supported denominator yields COVERAGE_INSUFFICIENT
def test_zero_supported_denominator_coverage_insufficient(mock_holdout_sources, tmp_path):
    # Only 1 contradicted claim -> supported claims == 0
    single_lbl = tmp_path / "single_contradicted_labels.json"
    single_lbl.write_text(
        json.dumps({
            "questions": {
                "SYNTH_Q2": {
                    "arms": {
                        "PRIMARY": {
                            "claims": {
                                "C1": {"entailment_label": "CONTRADICTED"}
                            }
                        }
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    single_zip = tmp_path / "single_contradicted_packet.zip"
    with zipfile.ZipFile(mock_holdout_sources["holdout_packets"], "r") as zf_in:
        with zipfile.ZipFile(single_zip, "w") as zf_out:
            zf_out.writestr("packets/SYNTH_Q2.json", zf_in.read("packets/SYNTH_Q2.json"))

    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=single_zip,
        holdout_labels_path=single_lbl,
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out",
        custom_provider=MockCanonicalD3Provider(mode="contradicted"),
        bypass_source_checksums=True,
    )
    rep = evaluator.evaluate()
    assert rep["verdict"] == "V2_D3_HOLDOUT_COVERAGE_INSUFFICIENT"
    assert rep["coverage"]["coverage_sufficient"] is False
    assert rep["coverage"]["supported_claims_denominator_valid"] is False
    assert rep["metrics"]["v2_d3_claim_binary"]["supported_retention"] is None


# Test 17: Quality rate gate failure yields PROMOTION_REJECTED
def test_quality_rate_gate_failure_verdict(mock_holdout_sources, tmp_path):
    # Mode contradicted predicts REJECT on all claims, failing supported retention on SYNTH_Q1
    evaluator = V2D3HoldoutBenchmarkEvaluator(
        holdout_packets_path=mock_holdout_sources["holdout_packets"],
        holdout_labels_path=mock_holdout_sources["holdout_labels"],
        holdout_selection_path=mock_holdout_sources["holdout_selection"],
        output_dir=tmp_path / "out_fail",
        custom_provider=MockCanonicalD3Provider(mode="contradicted"),
        bypass_source_checksums=True,
    )
    rep = evaluator.evaluate()
    assert rep["verdict"] == "V2_D3_HOLDOUT_PROMOTION_REJECTED"
    assert rep["coverage"]["coverage_sufficient"] is True
    assert rep["stability"]["unstable_semantic_claim_count"] == 0

