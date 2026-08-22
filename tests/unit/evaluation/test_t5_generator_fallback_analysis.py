"""Unit tests for T5-6A Generator Contract & Fallback Efficiency Analysis Tooling (Fix 3 Comprehensive Suite)."""

import json
from pathlib import Path
import zipfile
from pydantic import ValidationError
import pytest
from scripts.t5_generator_fallback_analysis import (
    CANONICAL_TUNE20_ORDERED_QIDS,
    EXPECTED_FAST30_ARCHIVE_SHA256,
    FallbackReconstructionStatus,
    GeneratorForensicPacket,
    GeneratorPathClassification,
    GeneratorRejectionDetail,
    PREREGISTERED_PRODUCTION_GENERATOR_BLOB_SHA,
    T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256,
    T5_6B_COMPACT_GENERATION_CONFIG_SHA256,
    T5_6B_CONTROL_GENERATION_CONFIG_SHA256,
    T5_6B_FROZEN_GENERATOR_INPUT_SHA256,
    T5_6B_JSON_GENERATION_CONFIG_SHA256,
    T5_6B_TUNE20_ORDERED_QIDS_SHA256,
    TUNE20_HISTORICAL_METEOR,
    TUNE20_HISTORICAL_ROUGE_L,
    build_generator_forensic_packet,
    classify_generator_path_from_telemetry,
    compute_frozen_generator_input_sha256,
    compute_model_config_canonical_sha256,
    compute_tune20_ordered_qids_sha256,
    derive_t5_6b_generator_input_authority,
    evaluate_tune_fallback_counterfactuals,
    extract_frozen_generator_inputs,
    get_preregistered_claim_verification_config,
    get_preregistered_generation_config,
    load_fast30_generator_forensics,
    verify_fallback_identity_reconstruction,
)
from legal_agentic_rag.configuration.online import ClaimVerificationConfig, GenerationConfig


# --- Forensic Census & Classification Regression Tests ---

def test_actual_byte_artifact_authority_loader(tmp_path: Path):
    fake_zip = tmp_path / "fake.zip"
    with zipfile.ZipFile(fake_zip, "w") as z:
        z.writestr("diagnostics.jsonl", "")
    with pytest.raises(ValueError, match="Archive SHA mismatch"):
        load_fast30_generator_forensics(fake_zip)


def test_structured_output_schema_classification():
    record = {
        "question_id": "89271",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": True,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": ["generator_model_error_fallback"],
        "generator_draft_rejections": [
            {"error_type": "structured_output_schema", "structured_output_attempt": 1},
            {"error_type": "structured_output_schema", "structured_output_attempt": 2},
        ],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.STRUCTURED_OUTPUT_REJECTION_MODEL_FALLBACK
    assert packet.is_model_error_fallback is True
    assert packet.fallback_reconstruction_status == FallbackReconstructionStatus.EXACT_IDENTITY_MATCH
    assert packet.matches_e1_verbatim is True
    assert len(packet.rejections) == 2


def test_two_rejection_attempts_counted():
    rejections = [
        {"error_type": "structured_output_schema", "structured_output_attempt": 1},
        {"error_type": "structured_output_schema", "structured_output_attempt": 2},
    ]
    record = {
        "question_id": "q1",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": True,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": ["generator_model_error_fallback"],
        "generator_draft_rejections": rejections,
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert len(packet.rejections) == 2
    assert packet.rejections[0].structured_output_attempt == 1
    assert packet.rejections[1].structured_output_attempt == 2


def test_other_model_error_fallback_classification():
    record = {
        "question_id": "q2",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": True,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": ["generator_model_error_fallback"],
        "generator_draft_rejections": [{"error_type": "unknown_evidence_id", "structured_output_attempt": 1}],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.OTHER_MODEL_ERROR_FALLBACK


def test_grounding_extractive_fallback_classification():
    record = {
        "question_id": "q3",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": False,
        "is_grounding_extractive_fallback": True,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": ["extractive_fallback_applied"],
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.GROUNDING_EXTRACTIVE_FALLBACK
    assert packet.is_grounding_fallback is True


def test_fabricated_grounding_warning_fails_closed():
    record = {
        "question_id": "q-fab",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": False,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": False,
        "is_insufficient_evidence": False,
        "warnings": ["grounding_extractive_fallback"],
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1"},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.AMBIGUOUS


def test_persisted_boolean_warning_mismatch_fails_closed():
    record = {
        "question_id": "q-mismatch",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": True,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": [],  # missing generator_model_error_fallback
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1"},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.AMBIGUOUS


def test_persisted_is_any_fallback_inconsistency_fails_closed():
    record = {
        "question_id": "q-inconsistent",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": False,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,  # inconsistent
        "is_insufficient_evidence": False,
        "warnings": [],
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1"},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.AMBIGUOUS


def test_supported_claim_salvage_classification():
    record = {
        "question_id": "83501",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": False,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": False,
        "is_insufficient_evidence": False,
        "warnings": ["supported_claim_salvage_applied"],
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "Chủ tịch nước có nhiệm vụ..."},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.SUPPORTED_CLAIM_SALVAGE


def test_insufficient_evidence_classification():
    record = {
        "question_id": "54485",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": False,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": False,
        "is_insufficient_evidence": True,
        "warnings": ["insufficient_context"],
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "Hệ thống chưa tìm thấy..."},
    }
    packet = build_generator_forensic_packet(record, "Holdout10")
    assert packet.path_classification == GeneratorPathClassification.INSUFFICIENT_EVIDENCE
    assert packet.is_insufficient_evidence is True


def test_semantic_synthesis_success_classification():
    record = {
        "question_id": "q-clean",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": False,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": False,
        "is_insufficient_evidence": False,
        "warnings": ["effect_status_unknown:E1"],
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "Câu trả lời tổng hợp [E1]."},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.SEMANTIC_SYNTHESIS_SUCCESS


def test_conflicting_fallback_and_salvage_warnings_fail_closed():
    record = {
        "question_id": "q-conflict",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": True,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": ["generator_model_error_fallback", "supported_claim_salvage_applied"],
        "generator_draft_rejections": [],
        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
        "public_response": {"answer": "[E1] t1"},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.AMBIGUOUS


# --- Fallback Reconstruction Tests ---

def test_fallback_reconstruction_non_e1_marker_succeeds():
    pub_ans = "[E7] Nội dung điều luật 7"
    cits = [{"evidence_id": "E7", "chunk_id": "chunk_777"}]
    sel_ev = [{"evidence_id": "E7", "chunk_id": "chunk_777", "text": "Nội dung điều luật 7"}]
    
    status, count, is_n1_e1 = verify_fallback_identity_reconstruction(pub_ans, cits, sel_ev, is_fallback=True)
    assert status == FallbackReconstructionStatus.EXACT_IDENTITY_MATCH
    assert count == 1
    assert is_n1_e1 is False


def test_fallback_reconstruction_wrong_citation_evidence_id_fails():
    pub_ans = "[E1] Nội dung"
    cits = [{"evidence_id": "E99", "chunk_id": "chunk_1"}]
    sel_ev = [{"evidence_id": "E1", "chunk_id": "chunk_1", "text": "Nội dung"}]
    
    status, _, _ = verify_fallback_identity_reconstruction(pub_ans, cits, sel_ev, is_fallback=True)
    assert status == FallbackReconstructionStatus.IDENTITY_MISMATCH


def test_fallback_reconstruction_wrong_citation_chunk_id_fails():
    pub_ans = "[E1] Nội dung"
    cits = [{"evidence_id": "E1", "chunk_id": "chunk_wrong"}]
    sel_ev = [{"evidence_id": "E1", "chunk_id": "chunk_1", "text": "Nội dung"}]
    
    status, _, _ = verify_fallback_identity_reconstruction(pub_ans, cits, sel_ev, is_fallback=True)
    assert status == FallbackReconstructionStatus.IDENTITY_MISMATCH


def test_fallback_reconstruction_duplicate_citations_fail_closed():
    pub_ans = "[E1] t1\n\n[E1] t1"
    cits = [{"evidence_id": "E1", "chunk_id": "c1"}, {"evidence_id": "E1", "chunk_id": "c1"}]
    sel_ev = [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}]
    
    status, _, _ = verify_fallback_identity_reconstruction(pub_ans, cits, sel_ev, is_fallback=True)
    assert status == FallbackReconstructionStatus.IDENTITY_MISMATCH


def test_fallback_reconstruction_citation_ordering_mismatch_fails():
    pub_ans = "[E1] t1\n\n[E2] t2"
    cits = [{"evidence_id": "E2", "chunk_id": "c2"}, {"evidence_id": "E1", "chunk_id": "c1"}]
    sel_ev = [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}, {"evidence_id": "E2", "chunk_id": "c2", "text": "t2"}]
    
    status, _, _ = verify_fallback_identity_reconstruction(pub_ans, cits, sel_ev, is_fallback=True)
    assert status == FallbackReconstructionStatus.IDENTITY_MISMATCH


def test_duplicate_selected_evidence_ids_fail_closed():
    record = {
        "question_id": "q-dup",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": True,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": ["generator_model_error_fallback"],
        "generator_draft_rejections": [],
        "selected_evidence": [
            {"evidence_id": "E1", "chunk_id": "c1", "text": "t1"},
            {"evidence_id": "E1", "chunk_id": "c2", "text": "t2"},
        ],
        "public_response": {"answer": "[E1] t1"},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.AMBIGUOUS


def test_duplicate_selected_chunk_ids_fail_closed():
    record = {
        "question_id": "q-dup2",
        "question": "Q",
        "reference_answer": "A",
        "is_generator_model_error_fallback": True,
        "is_grounding_extractive_fallback": False,
        "is_any_extractive_fallback": True,
        "is_insufficient_evidence": False,
        "warnings": ["generator_model_error_fallback"],
        "generator_draft_rejections": [],
        "selected_evidence": [
            {"evidence_id": "E1", "chunk_id": "c1", "text": "t1"},
            {"evidence_id": "E2", "chunk_id": "c1", "text": "t2"},
        ],
        "public_response": {"answer": "[E1] t1"},
    }
    packet = build_generator_forensic_packet(record, "Tune20")
    assert packet.path_classification == GeneratorPathClassification.AMBIGUOUS


# --- Partition Safety & Oracle Isolation Tests ---

def test_tune_discovery_accepts_tune20():
    packets = [
        build_generator_forensic_packet(
            {
                "question_id": f"q-{i}", "question": "Q", "reference_answer": "A",
                "is_generator_model_error_fallback": True, "is_grounding_extractive_fallback": False,
                "is_any_extractive_fallback": True, "is_insufficient_evidence": False,
                "warnings": ["generator_model_error_fallback"],
                "generator_draft_rejections": [{"error_type": "structured_output_schema", "structured_output_attempt": 1}],
                "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
                "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
            },
            "Tune20",
        )
        for i in range(20)
    ]
    res = evaluate_tune_fallback_counterfactuals(packets)
    assert res["tune_count"] == 20
    assert res["tune_fallbacks"] == 20
    assert res["decision"] == "NEW_CONTROLLED_GENERATOR_MEASUREMENT_REQUIRED"


def test_tune_discovery_rejects_holdout():
    p = build_generator_forensic_packet(
        {"question_id": "q-h", "question": "Q", "reference_answer": "A"},
        "Holdout10",
    )
    with pytest.raises(ValueError, match="Holdout10 is contaminated"):
        evaluate_tune_fallback_counterfactuals([p])


def test_tune_discovery_rejects_mixed():
    p1 = build_generator_forensic_packet(
        {"question_id": "q-1", "question": "Q", "reference_answer": "A"},
        "Tune20",
    )
    p2 = build_generator_forensic_packet(
        {"question_id": "q-2", "question": "Q", "reference_answer": "A"},
        "Holdout10",
    )
    with pytest.raises(ValueError, match="exclusively Tune20"):
        evaluate_tune_fallback_counterfactuals([p1, p2])


def test_oracle_deployable_boundary_isolation():
    packets_high = [
        build_generator_forensic_packet(
            {
                "question_id": f"q-{i}", "question": "Q", "reference_answer": "A",
                "is_generator_model_error_fallback": True, "is_grounding_extractive_fallback": False,
                "is_any_extractive_fallback": True, "is_insufficient_evidence": False,
                "warnings": ["generator_model_error_fallback"],
                "generator_draft_rejections": [{"error_type": "structured_output_schema", "structured_output_attempt": 1}],
                "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
                "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
                "rouge_l_score": 0.99, "meteor_score": 0.99,
            },
            "Tune20",
        )
        for i in range(20)
    ]
    packets_low = [
        build_generator_forensic_packet(
            {
                "question_id": f"q-{i}", "question": "Q", "reference_answer": "A" * 5000,
                "is_generator_model_error_fallback": True, "is_grounding_extractive_fallback": False,
                "is_any_extractive_fallback": True, "is_insufficient_evidence": False,
                "warnings": ["generator_model_error_fallback"],
                "generator_draft_rejections": [{"error_type": "structured_output_schema", "structured_output_attempt": 1}],
                "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
                "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
                "rouge_l_score": 0.0, "meteor_score": 0.0,
            },
            "Tune20",
        )
        for i in range(20)
    ]
    assert evaluate_tune_fallback_counterfactuals(packets_high)["decision"] == evaluate_tune_fallback_counterfactuals(packets_low)["decision"]


# --- Preregistration Schema & Configuration Authority Tests ---

def test_preregistered_generation_config_validates():
    config = get_preregistered_generation_config(prompt_schema_mode="plain_text_markers")
    assert isinstance(config, GenerationConfig)
    assert config.backend == "transformers"
    assert config.model_loader == "image_text_to_text"
    assert config.device == "cuda"
    assert config.torch_dtype == "float16"
    assert config.max_input_tokens == 8192
    assert config.max_output_tokens == 1536
    assert config.repetition_penalty == 1.08
    assert config.no_repeat_ngram_size == 8
    assert config.salvage_rendering == "standalone"
    assert config.prompt_schema_mode == "plain_text_markers"
    assert config.answer_style == "competition_reference"


def test_preregistered_claim_verification_config_validates():
    config = get_preregistered_claim_verification_config()
    assert isinstance(config, ClaimVerificationConfig)
    assert config.enabled is True
    assert config.require_inline_citations is False
    assert config.minimum_lexical_support == 0.2
    assert config.minimum_claim_tokens == 2
    assert config.require_numeric_match is True
    assert config.require_negation_match is True
    assert config.max_claims == 60


def test_invalid_salvage_rendering_rejected():
    with pytest.raises(ValidationError):
        GenerationConfig(salvage_rendering="clean_prose")  # type: ignore[arg-type]


def test_exact_one_dtype_pinned():
    config = get_preregistered_generation_config()
    assert config.torch_dtype == "float16"


def test_candidate_generation_configs_differ_only_in_prompt_schema_mode():
    ctrl = get_preregistered_generation_config(prompt_schema_mode="plain_text_markers")
    compact = get_preregistered_generation_config(prompt_schema_mode="compact_example")
    json_mode = get_preregistered_generation_config(prompt_schema_mode="json_schema")

    diff_compact = {k: (v, compact.model_dump()[k]) for k, v in ctrl.model_dump().items() if v != compact.model_dump()[k]}
    diff_json = {k: (v, json_mode.model_dump()[k]) for k, v in ctrl.model_dump().items() if v != json_mode.model_dump()[k]}

    assert list(diff_compact.keys()) == ["prompt_schema_mode"]
    assert diff_compact["prompt_schema_mode"] == ("plain_text_markers", "compact_example")

    assert list(diff_json.keys()) == ["prompt_schema_mode"]
    assert diff_json["prompt_schema_mode"] == ("plain_text_markers", "json_schema")


def test_candidate_config_hashes_deterministic():
    ctrl = get_preregistered_generation_config(prompt_schema_mode="plain_text_markers")
    compact = get_preregistered_generation_config(prompt_schema_mode="compact_example")
    json_mode = get_preregistered_generation_config(prompt_schema_mode="json_schema")
    claim_cfg = get_preregistered_claim_verification_config()

    assert compute_model_config_canonical_sha256(ctrl) == T5_6B_CONTROL_GENERATION_CONFIG_SHA256
    assert compute_model_config_canonical_sha256(compact) == T5_6B_COMPACT_GENERATION_CONFIG_SHA256
    assert compute_model_config_canonical_sha256(json_mode) == T5_6B_JSON_GENERATION_CONFIG_SHA256
    assert compute_model_config_canonical_sha256(claim_cfg) == T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256


def test_tune20_ordered_qids_hash_deterministic():
    h = compute_tune20_ordered_qids_sha256(CANONICAL_TUNE20_ORDERED_QIDS)
    assert h == T5_6B_TUNE20_ORDERED_QIDS_SHA256
    assert h == "9cb88a00c2bcf9fbc0f24411de2f427d6a30f5da0f57feaaafb629f9fcd60b28"


def test_frozen_generator_input_hash_deterministic():
    dummy_records = [
        {
            "question_id": qid,
            "question": f"Question text {qid}",
            "selected_evidence": [{"evidence_id": "E1", "chunk_id": f"c_{qid}", "text": f"t_{qid}"}],
        }
        for qid in CANONICAL_TUNE20_ORDERED_QIDS
    ]
    extracted = extract_frozen_generator_inputs(dummy_records)
    h = compute_frozen_generator_input_sha256(extracted)
    assert isinstance(h, str)
    assert len(h) == 64


def test_archive_sha_differs_from_generator_input_sha():
    assert EXPECTED_FAST30_ARCHIVE_SHA256 != T5_6B_FROZEN_GENERATOR_INPUT_SHA256
    assert EXPECTED_FAST30_ARCHIVE_SHA256 == "be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f"
    assert T5_6B_FROZEN_GENERATOR_INPUT_SHA256 == "2fefbb03125f9927edf67c8bc8c165bdd856e1dd2eef0c737aefc7387a2cbbf2"


def test_derive_t5_6b_generator_input_authority_synthetic(tmp_path: Path):
    dummy_records = [
        {
            "question_id": qid,
            "question": f"Question {qid}",
            "reference_answer": f"Answer {qid}",
            "selected_evidence": [{"evidence_id": "E1", "chunk_id": f"c_{qid}", "text": f"t_{qid}"}],
        }
        for qid in CANONICAL_TUNE20_ORDERED_QIDS
    ]
    fake_zip = tmp_path / "fake_archive.zip"
    with zipfile.ZipFile(fake_zip, "w") as z:
        z.writestr("diagnostics.jsonl", "\n".join(json.dumps(r) for r in dummy_records))
        
    with pytest.raises(ValueError, match="Archive SHA mismatch"):
        derive_t5_6b_generator_input_authority(fake_zip)


def test_generic_helper_historical_19_failures_triggers_measurement():
    packets = [
        build_generator_forensic_packet(
            {
                "question_id": f"q-{i}", "question": "Q", "reference_answer": "A",
                "is_generator_model_error_fallback": (i < 19), "is_grounding_extractive_fallback": False,
                "is_any_extractive_fallback": (i < 19), "is_insufficient_evidence": False,
                "warnings": ["generator_model_error_fallback"] if i < 19 else [],
                "generator_draft_rejections": [{"error_type": "structured_output_schema", "structured_output_attempt": 1}] if i < 19 else [],
                "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
                "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
            },
            "Tune20",
        )
        for i in range(20)
    ]
    res = evaluate_tune_fallback_counterfactuals(packets)
    assert res["tune_count"] == 20
    assert res["tune_contract_rejection_fallbacks"] == 19
    assert res["decision"] == "NEW_CONTROLLED_GENERATOR_MEASUREMENT_REQUIRED"
    assert "19/20 contract rejection fallbacks" in res["rationale"]


def test_generic_helper_zero_failures_no_trigger():
    packets = [
        build_generator_forensic_packet(
            {
                "question_id": f"q-{i}", "question": "Q", "reference_answer": "A",
                "is_generator_model_error_fallback": False, "is_grounding_extractive_fallback": False,
                "is_any_extractive_fallback": False, "is_insufficient_evidence": False,
                "warnings": [], "generator_draft_rejections": [],
                "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
                "public_response": {"answer": "Tổng hợp [E1].", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
            },
            "Tune20",
        )
        for i in range(20)
    ]
    res = evaluate_tune_fallback_counterfactuals(packets)
    assert res["tune_count"] == 20
    assert res["tune_contract_rejection_fallbacks"] == 0
    assert res["decision"] == "NO_CONTRACT_MEASUREMENT_TRIGGER_FROM_THIS_POPULATION"
    assert "0/20 contract rejection fallbacks" in res["rationale"]


def test_generic_helper_rationale_dynamic_counts():
    packets = [
        build_generator_forensic_packet(
            {
                "question_id": f"q-{i}", "question": "Q", "reference_answer": "A",
                "is_generator_model_error_fallback": (i < 5), "is_grounding_extractive_fallback": False,
                "is_any_extractive_fallback": (i < 5), "is_insufficient_evidence": False,
                "warnings": ["generator_model_error_fallback"] if i < 5 else [],
                "generator_draft_rejections": [{"error_type": "structured_output_schema", "structured_output_attempt": 1}] if i < 5 else [],
                "selected_evidence": [{"evidence_id": "E1", "chunk_id": "c1", "text": "t1"}],
                "public_response": {"answer": "[E1] t1", "citations": [{"evidence_id": "E1", "chunk_id": "c1"}]},
            },
            "Tune20",
        )
        for i in range(20)
    ]
    res = evaluate_tune_fallback_counterfactuals(packets)
    assert res["tune_contract_rejection_fallbacks"] == 5
    assert "5/20 contract rejection fallbacks" in res["rationale"]
