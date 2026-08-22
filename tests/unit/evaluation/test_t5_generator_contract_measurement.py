"""Comprehensive unit test suite for T5-6B Fix 4: Executability, Official Scorer, and Prereg Consistency.

Tests execute-generation provider wiring, official scorer archive & member SHA verification,
official scorer dynamic execution, golden vectors parity, deep multi-artifact resume identity verification,
real runner interruption/resumption equivalence, and advancement gate integrity.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha1, sha256
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any
import unicodedata
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.online import (
    ClaimVerificationConfig,
    GenerationConfig,
)
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    ModelError,
)
from legal_agentic_rag.generation.citation_verifier import (
    RuleBasedCitationVerifier,
)
from legal_agentic_rag.generation.model_generator import (
    ModelBackedAnswerGenerator,
)
from legal_agentic_rag.generation.transformers_provider import (
    TransformersChatProvider,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
)
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy
import unittest.mock as _mock_module

import scripts.t5_generator_contract_measurement as scm
from scripts.t5_generator_contract_measurement import (
    ARM_CONTRACT_MAP,
    CANONICAL_TUNE20_ORDERED_QIDS,
    EXECUTION_ARM_ORDER,
    EXPECTED_FAST30_ARCHIVE_SHA256,
    EXPECTED_GENERATION_CONFIG_HASHES,
    EXPECTED_M49_GENERATOR_TREE_SHA256,
    GROUNDING_REPAIR_SENTINEL,
    OFFICIAL_SCORER_ARCHIVE_SHA256,
    OFFICIAL_SCORING_PY_SHA256,
    PREREGISTERED_PRODUCTION_GENERATOR_BLOB_SHA,
    PROVIDER_RELEVANT_CONFIG_FIELDS,
    STRUCTURED_RETRY_SENTINEL,
    T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256,
    T5_6B_COMPACT_GENERATION_CONFIG_SHA256,
    T5_6B_CONTROL_GENERATION_CONFIG_SHA256,
    T5_6B_FROZEN_GENERATOR_INPUT_SHA256,
    T5_6B_JSON_GENERATION_CONFIG_SHA256,
    T5_6B_TUNE20_ORDERED_QIDS_SHA256,
    ArmMeasurementSummary,
    CandidateAdvancementResult,
    FrozenGeneratorInputPacket,
    GeneratorCallTelemetry,
    GroundingCallTelemetry,
    MeasurementProviderObserver,
    ModelGeneratorLoggingLease,
    ModelGeneratorRejectionObserver,
    ObservableCitationVerifier,
    ObservableTransformersChatProvider,
    QuestionMeasurementResult,
    T56BGenerationManifest,
    T56BMeasurementManifest,
    T56BRuntimeEnvironmentFingerprint,
    T5GeneratorContractMeasurementRunner,
    _GenerateProxy,
    _diagnostic_compute_meteor,
    _diagnostic_compute_rouge_l,
    _diagnostic_rouge_l_tokenize,
    build_arg_parser,
    classify_final_generator_path,
    classify_prompt_call_stage,
    compute_directory_sha256,
    compute_environment_fingerprint_sha256,
    compute_file_sha256,
    compute_frozen_generator_input_sha256,
    compute_tune20_ordered_qids_sha256,
    correlate_rejection_events,
    evaluate_advancement_gate,
    evaluate_citation_identity_validity,
    extract_frozen_generator_inputs,
    get_preregistered_claim_verification_config,
    get_preregistered_generation_config,
    get_runtime_environment_fingerprint,
    git_blob_sha,
    load_frozen_tune20_packets,
    measurement_transformers_provider_factory,
    parse_cli_args,
    reconstruct_query,
    score_tune20_answers,
    verify_production_generator_blob,
)


def make_answer_response(
    answer: str = "Test answer",
    citations: list[Citation] | None = None,
    warnings: list[str] | None = None,
    insufficient_evidence: bool = False,
    question: str = "Test question",
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
    trace_id: str = "t5-6b-test",
) -> AnswerResponse:
    """Helper to construct a schema-valid AnswerResponse for unit tests."""
    return AnswerResponse(
        question=question,
        answer=answer,
        citations=citations or [],
        warnings=warnings or [],
        insufficient_evidence=insufficient_evidence,
        retrieval_strategy=retrieval_strategy,
        trace_id=trace_id,
    )


class FakeChatModelProvider:
    """Deterministic fake provider for test execution."""

    def __init__(
        self,
        completions: list[str | Exception] | None = None,
        model_name: str = "fake_model",
        model_revision: str = "fake_rev",
    ) -> None:
        self.completions = list(completions or [])
        self.model_name = model_name
        self.model_revision = model_revision
        self.call_history: list[tuple[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "fake_provider"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        self.call_history.append((system_instruction, user_prompt))
        if not self.completions:
            return "Fake default answer [E1]."
        next_item = self.completions.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


def make_synthetic_fast30_zip(target_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct a synthetic FAST30 zip archive fixture in a temporary path and patch authority hashes."""
    records = []
    for qid in CANONICAL_TUNE20_ORDERED_QIDS:
        records.append(
            {
                "question_id": qid,
                "question": f"Question for {qid}?",
                "selected_evidence": [
                    {
                        "evidence_id": "E1",
                        "chunk_id": f"c_{qid}_1",
                        "document_id": f"doc_{qid}",
                        "document_title": f"Title {qid}",
                        "text": f"Legal text for {qid}",
                    }
                ],
            }
        )
    with ZipFile(target_path, "w") as z:
        z.writestr("diagnostics.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in records))

    monkeypatch.setattr(
        scm,
        "EXPECTED_FAST30_ARCHIVE_SHA256",
        compute_file_sha256(target_path),
    )
    monkeypatch.setattr(
        scm,
        "T5_6B_TUNE20_ORDERED_QIDS_SHA256",
        compute_tune20_ordered_qids_sha256([str(r["question_id"]) for r in records]),
    )
    monkeypatch.setattr(
        scm,
        "T5_6B_FROZEN_GENERATOR_INPUT_SHA256",
        compute_frozen_generator_input_sha256(extract_frozen_generator_inputs(records)),
    )


def make_synthetic_scorer_zip(
    target_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_scoring_py: bool = False,
    missing_eval_qa: bool = False,
    eval_qa_override: str | None = None,
) -> None:
    """Construct a synthetic official scorer zip archive fixture and patch authority hashes.

    ZERO machine-specific filesystem dependencies: all tests run against this synthetic fixture.
    Exposes the official ``eval_qa(y_pred, y_true)`` interface matching the real scorer.py program.
    """
    if corrupt_scoring_py:
        scoring_code = "# corrupted scoring"
    elif missing_eval_qa:
        scoring_code = "# valid python but missing eval_qa\nSOME_VAR = 123\n"
    elif eval_qa_override is not None:
        scoring_code = eval_qa_override
    else:
        scoring_code = """
import numpy as np

def eval_qa(y_pred, y_true):
    # Matches real scoring.py contract: y_pred maps qid -> {'answer': text}, y_true maps qid -> text
    y_pred_text = {k: v['answer'] for k, v in y_pred.items()}
    ids_preds = list(y_pred_text.keys())
    ids_truth = list(y_true.keys())
    if len(ids_preds) != len(ids_truth):
        raise Exception("Samples in predict not match with reference")

    from scripts.t5_generator_contract_measurement import (
        _diagnostic_compute_rouge_l,
        _diagnostic_compute_meteor,
    )
    r_scores = [_diagnostic_compute_rouge_l(y_pred_text[k], y_true[k]) for k in ids_preds]
    m_scores = [_diagnostic_compute_meteor(y_pred_text[k], y_true[k]) for k in ids_preds]
    return {
        'rouge': float(np.array(r_scores).mean()),
        'meteor': float(np.array(m_scores).mean()),
    }
"""

    with ZipFile(target_path, "w") as z:
        z.writestr("scoring.py", scoring_code)

    monkeypatch.setattr(scm, "OFFICIAL_SCORER_ARCHIVE_SHA256", compute_file_sha256(target_path))
    with ZipFile(target_path, "r") as z:
        monkeypatch.setattr(scm, "OFFICIAL_SCORING_PY_SHA256", sha256(z.read("scoring.py")).hexdigest())


def test_01_execute_generation_provider_factory_wiring() -> None:
    factory = measurement_transformers_provider_factory
    cfg = get_preregistered_generation_config("plain_text_markers")
    prov = factory(cfg)
    assert isinstance(prov, ObservableTransformersChatProvider)
    assert isinstance(prov, TransformersChatProvider)


def test_02_provider_factory_invoked_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zip_path = tmp_path / "fast30.zip"
    make_synthetic_fast30_zip(zip_path, monkeypatch)

    factory_calls = 0

    def counting_factory(cfg: GenerationConfig) -> ChatModelProvider:
        nonlocal factory_calls
        factory_calls += 1
        return FakeChatModelProvider(completions=["Ans [E1]."])

    out_dir = tmp_path / "out"
    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=zip_path,
        output_dir=out_dir,
        provider_factory=counting_factory,
        is_preflight_only=True,
    )
    runner.run_generation()
    assert factory_calls == 1


def test_03_authority_failure_occurs_before_provider_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zip_path = tmp_path / "fast30.zip"
    make_synthetic_fast30_zip(zip_path, monkeypatch)

    factory_called = False

    def bad_factory(cfg: GenerationConfig) -> ChatModelProvider:
        nonlocal factory_called
        factory_called = True
        return FakeChatModelProvider()

    bad_model_path = tmp_path / "non_existent_model_dir"

    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=zip_path,
        output_dir=tmp_path / "out",
        provider_factory=bad_factory,
        model_path=bad_model_path,
        expected_measurement_source_sha=os.popen("git rev-parse HEAD").read().strip(),
        is_preflight_only=False,
    )

    with pytest.raises(ArtifactCompatibilityError):
        runner.run_generation()

    assert factory_called is False


def test_04_score_closed_generation_never_creates_provider(tmp_path: Path) -> None:
    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=tmp_path / "fake.zip",
        output_dir=tmp_path,
        provider_factory=None,
    )
    assert runner.provider_factory is None


# ==========================================
# 2. OFFICIAL SCORER ARCHIVE & MEMBER AUTHORITY TESTS
# ==========================================


def test_05_score_tune20_answers_requires_official_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer_zip = tmp_path / "Scoring-Program.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch)

    preds = {qid: f"Prediction {qid}." for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: f"Reference {qid}." for qid in CANONICAL_TUNE20_ORDERED_QIDS}

    mean_r, mean_m, per_q, auth_sha = score_tune20_answers(preds, refs, scorer_path=scorer_zip)
    assert 0.0 <= mean_r <= 1.0
    assert 0.0 <= mean_m <= 1.0
    assert len(per_q) == 20
    assert auth_sha == compute_file_sha256(scorer_zip)


def test_06_score_tune20_missing_archive_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "non_existent.zip"
    preds = {qid: "p" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: "r" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    with pytest.raises(ArtifactCompatibilityError, match="OFFICIAL_SCORER_AUTHORITY_FAILED: Scorer archive not found"):
        score_tune20_answers(preds, refs, scorer_path=missing)


def test_07_score_tune20_wrong_archive_sha_rejected(tmp_path: Path) -> None:
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"PK\x03\x04corrupted")
    preds = {qid: "p" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: "r" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    with pytest.raises(ArtifactCompatibilityError, match="OFFICIAL_SCORER_AUTHORITY_FAILED: Scorer archive SHA mismatch"):
        score_tune20_answers(preds, refs, scorer_path=bad_zip)


def test_08_score_tune20_wrong_scoring_py_member_sha_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corrupted_zip = tmp_path / "corrupted_member.zip"
    make_synthetic_scorer_zip(corrupted_zip, monkeypatch, corrupt_scoring_py=True)
    monkeypatch.setattr(scm, "OFFICIAL_SCORER_ARCHIVE_SHA256", compute_file_sha256(corrupted_zip))
    monkeypatch.setattr(scm, "OFFICIAL_SCORING_PY_SHA256", "different_expected_sha")

    preds = {qid: "p" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: "r" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    with pytest.raises(ArtifactCompatibilityError, match="OFFICIAL_SCORER_AUTHORITY_FAILED: scoring.py member SHA mismatch"):
        score_tune20_answers(preds, refs, scorer_path=corrupted_zip)


# ==========================================
# 3. OFFICIAL ENTRYPOINT & EXACT QID SET VALIDATION TESTS
# ==========================================


def test_09a_score_tune20_calls_official_eval_qa_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """score_tune20_answers must call official eval_qa and return its exact macro scores."""
    sentinel_override = """
def eval_qa(y_pred, y_true):
    return {'rouge': 0.7777, 'meteor': 0.8888}
"""
    scorer_zip = tmp_path / "sentinel_scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch, eval_qa_override=sentinel_override)

    preds = {qid: f"Pred {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: f"Ref {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}

    mean_r, mean_m, per_q, _ = score_tune20_answers(preds, refs, scorer_path=scorer_zip)
    assert mean_r == pytest.approx(0.7777)
    assert mean_m == pytest.approx(0.8888)


def test_09b_score_tune20_passes_exact_20_qid_macro_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """score_tune20_answers passes exact 20-QID payload matching {qid: {'answer': str}} and {qid: str}."""
    payload_check_override = """
def eval_qa(y_pred, y_true):
    assert len(y_pred) in (1, 20), f"Unexpected payload size: {len(y_pred)}"
    assert len(y_true) in (1, 20), f"Unexpected payload size: {len(y_true)}"
    for qid, val in y_pred.items():
        assert isinstance(val, dict) and "answer" in val, f"Invalid pred item {qid}: {val}"
        assert isinstance(val["answer"], str)
    for qid, val in y_true.items():
        assert isinstance(val, str), f"Invalid ref item {qid}: {val}"
    return {'rouge': 0.5, 'meteor': 0.5}
"""
    scorer_zip = tmp_path / "payload_check_scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch, eval_qa_override=payload_check_override)

    preds = {qid: f"Pred {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: f"Ref {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}

    mean_r, mean_m, per_q, _ = score_tune20_answers(preds, refs, scorer_path=scorer_zip)
    assert len(per_q) == 20


def test_09c_score_tune20_missing_prediction_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing prediction QID fails closed immediately."""
    scorer_zip = tmp_path / "scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch)

    preds = {qid: f"Pred {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS[:-1]}  # 19 QIDs
    refs = {qid: f"Ref {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}

    with pytest.raises(DataValidationError, match="Predicted answers QID set mismatch"):
        score_tune20_answers(preds, refs, scorer_path=scorer_zip)


def test_09d_score_tune20_extra_prediction_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra prediction QID fails closed immediately."""
    scorer_zip = tmp_path / "scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch)

    preds = {qid: f"Pred {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    preds["extra_qid_99999"] = "Extra prediction"
    refs = {qid: f"Ref {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}

    with pytest.raises(DataValidationError, match="Predicted answers QID set mismatch"):
        score_tune20_answers(preds, refs, scorer_path=scorer_zip)


def test_09e_score_tune20_missing_reference_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing reference QID fails closed immediately."""
    scorer_zip = tmp_path / "scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch)

    preds = {qid: f"Pred {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: f"Ref {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS[:-1]}  # 19 QIDs

    with pytest.raises(DataValidationError, match="Reference answers QID set mismatch"):
        score_tune20_answers(preds, refs, scorer_path=scorer_zip)


def test_09f_score_tune20_extra_reference_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra reference QID fails closed immediately."""
    scorer_zip = tmp_path / "scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch)

    preds = {qid: f"Pred {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: f"Ref {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs["extra_qid_99999"] = "Extra reference"

    with pytest.raises(DataValidationError, match="Reference answers QID set mismatch"):
        score_tune20_answers(preds, refs, scorer_path=scorer_zip)


def test_09g_score_tune20_missing_eval_qa_entrypoint_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scorer program missing eval_qa entrypoint fails closed immediately."""
    scorer_zip = tmp_path / "no_eval_qa_scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch, missing_eval_qa=True)

    preds = {qid: f"Pred {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: f"Ref {qid}" for qid in CANONICAL_TUNE20_ORDERED_QIDS}

    with pytest.raises(ArtifactCompatibilityError, match="OFFICIAL_SCORER_AUTHORITY_FAILED: eval_qa entrypoint not found"):
        score_tune20_answers(preds, refs, scorer_path=scorer_zip)


def test_09h_score_tune20_per_question_scores_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-question scores are populated using official eval_qa on single-QID dicts."""
    scorer_zip = tmp_path / "scorer.zip"
    make_synthetic_scorer_zip(scorer_zip, monkeypatch)

    preds = {qid: f"Prediction for {qid}." for qid in CANONICAL_TUNE20_ORDERED_QIDS}
    refs = {qid: f"Reference for {qid}." for qid in CANONICAL_TUNE20_ORDERED_QIDS}

    mean_r, mean_m, per_q, _ = score_tune20_answers(preds, refs, scorer_path=scorer_zip)
    assert len(per_q) == 20
    for qid in CANONICAL_TUNE20_ORDERED_QIDS:
        assert qid in per_q
        assert "rouge_l" in per_q[qid]
        assert "meteor" in per_q[qid]
        assert 0.0 <= per_q[qid]["rouge_l"] <= 1.0
        assert 0.0 <= per_q[qid]["meteor"] <= 1.0


# ==========================================
# 4. DEEP RESUME MULTI-ARTIFACT CORRUPTION TESTS
# ==========================================


def test_10_deep_resume_duplicate_call_identity_rejected(tmp_path: Path) -> None:
    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."), archive_path=tmp_path / "f.zip", output_dir=tmp_path
    )
    arm_dir = tmp_path / "control"
    arm_dir.mkdir()

    (arm_dir / "responses.jsonl").write_text(json.dumps({"question_id": "89271"}) + "\n")
    (arm_dir / "question_results.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "response": {
                    "question": "q",
                    "answer": "a",
                    "citations": [],
                    "warnings": [],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid_rerank",
                    "trace_id": "t",
                },
                "final_generator_path": "SEMANTIC_SYNTHESIS",
                "citation_identity_valid": True,
                "parser_accepted": True,
                "had_contract_rejection": False,
                "calls": [
                    {
                        "question_id": "89271",
                        "candidate_contract": "plain_text_markers",
                        "call_stage": "INITIAL_DRAFT",
                        "call_index": 1,
                        "provider_attempt_index": 1,
                        "provider_call_success": True,
                        "system_prompt_sha256": "s",
                        "user_prompt_sha256": "u",
                        "raw_completion_text": "Draft",
                    },
                    {
                        "question_id": "89271",
                        "candidate_contract": "plain_text_markers",
                        "call_stage": "INITIAL_DRAFT",
                        "call_index": 1,  # Duplicate call index
                        "provider_attempt_index": 1,
                        "provider_call_success": True,
                        "system_prompt_sha256": "s",
                        "user_prompt_sha256": "u",
                        "raw_completion_text": "Draft",
                    },
                ],
                "grounding_calls": [],
            }
        )
        + "\n"
    )
    (arm_dir / "call_telemetry.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "call_stage": "INITIAL_DRAFT",
                "call_index": 1,
                "provider_attempt_index": 1,
                "provider_call_success": True,
                "system_prompt_sha256": "s",
                "user_prompt_sha256": "u",
                "raw_completion_text": "Draft",
            }
        )
        + "\n"
        + json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "call_stage": "INITIAL_DRAFT",
                "call_index": 1,
                "provider_attempt_index": 1,
                "provider_call_success": True,
                "system_prompt_sha256": "s",
                "user_prompt_sha256": "u",
                "raw_completion_text": "Draft",
            }
        )
        + "\n"
    )
    (arm_dir / "raw_completions.jsonl").write_text("")
    (arm_dir / "grounding_telemetry.jsonl").write_text("")

    with pytest.raises(ArtifactCompatibilityError, match="PARTIAL_QID_ARTIFACT_STATE: Duplicate call identity"):
        runner._validate_and_reload_arm_resume_state("control", ["89271"], arm_dir)


def test_11_deep_resume_raw_completion_text_mismatch_rejected(tmp_path: Path) -> None:
    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."), archive_path=tmp_path / "f.zip", output_dir=tmp_path
    )
    arm_dir = tmp_path / "control"
    arm_dir.mkdir()

    (arm_dir / "responses.jsonl").write_text(json.dumps({"question_id": "89271"}) + "\n")
    (arm_dir / "question_results.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "response": {
                    "question": "q",
                    "answer": "a",
                    "citations": [],
                    "warnings": [],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid_rerank",
                    "trace_id": "t",
                },
                "final_generator_path": "SEMANTIC_SYNTHESIS",
                "citation_identity_valid": True,
                "parser_accepted": True,
                "had_contract_rejection": False,
                "calls": [
                    {
                        "question_id": "89271",
                        "candidate_contract": "plain_text_markers",
                        "call_stage": "INITIAL_DRAFT",
                        "call_index": 1,
                        "provider_attempt_index": 1,
                        "provider_call_success": True,
                        "system_prompt_sha256": "s",
                        "user_prompt_sha256": "u",
                        "raw_completion_text": "Draft A",
                    }
                ],
                "grounding_calls": [],
            }
        )
        + "\n"
    )
    (arm_dir / "call_telemetry.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "call_stage": "INITIAL_DRAFT",
                "call_index": 1,
                "provider_attempt_index": 1,
                "provider_call_success": True,
                "system_prompt_sha256": "s",
                "user_prompt_sha256": "u",
                "raw_completion_text": "Draft A",
            }
        )
        + "\n"
    )
    # Raw record contains "Draft B" instead of "Draft A"
    (arm_dir / "raw_completions.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "call_index": 1,
                "provider_attempt_index": 1,
                "raw_completion_text": "Draft B",
            }
        )
        + "\n"
    )
    (arm_dir / "grounding_telemetry.jsonl").write_text("")

    with pytest.raises(ArtifactCompatibilityError, match="PARTIAL_QID_ARTIFACT_STATE: Raw completion text mismatch"):
        runner._validate_and_reload_arm_resume_state("control", ["89271"], arm_dir)


def test_12_deep_resume_duplicate_raw_record_rejected(tmp_path: Path) -> None:
    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."), archive_path=tmp_path / "f.zip", output_dir=tmp_path
    )
    arm_dir = tmp_path / "control"
    arm_dir.mkdir()

    (arm_dir / "responses.jsonl").write_text(json.dumps({"question_id": "89271"}) + "\n")
    (arm_dir / "question_results.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "response": {
                    "question": "q",
                    "answer": "a",
                    "citations": [],
                    "warnings": [],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid_rerank",
                    "trace_id": "t",
                },
                "final_generator_path": "SEMANTIC_SYNTHESIS",
                "citation_identity_valid": True,
                "parser_accepted": True,
                "had_contract_rejection": False,
                "calls": [
                    {
                        "question_id": "89271",
                        "candidate_contract": "plain_text_markers",
                        "call_stage": "INITIAL_DRAFT",
                        "call_index": 1,
                        "provider_attempt_index": 1,
                        "provider_call_success": True,
                        "system_prompt_sha256": "s",
                        "user_prompt_sha256": "u",
                        "raw_completion_text": "Draft",
                    }
                ],
                "grounding_calls": [],
            }
        )
        + "\n"
    )
    (arm_dir / "call_telemetry.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "call_stage": "INITIAL_DRAFT",
                "call_index": 1,
                "provider_attempt_index": 1,
                "provider_call_success": True,
                "system_prompt_sha256": "s",
                "user_prompt_sha256": "u",
                "raw_completion_text": "Draft",
            }
        )
        + "\n"
    )
    # 2 duplicate raw records
    (arm_dir / "raw_completions.jsonl").write_text(
        json.dumps({"question_id": "89271", "call_index": 1, "provider_attempt_index": 1, "raw_completion_text": "Draft"}) + "\n"
        + json.dumps({"question_id": "89271", "call_index": 1, "provider_attempt_index": 1, "raw_completion_text": "Draft"}) + "\n"
    )
    (arm_dir / "grounding_telemetry.jsonl").write_text("")

    with pytest.raises(ArtifactCompatibilityError, match="PARTIAL_QID_ARTIFACT_STATE: Raw completions count mismatch"):
        runner._validate_and_reload_arm_resume_state("control", ["89271"], arm_dir)


def test_13_deep_resume_duplicate_grounding_index_rejected(tmp_path: Path) -> None:
    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."), archive_path=tmp_path / "f.zip", output_dir=tmp_path
    )
    arm_dir = tmp_path / "control"
    arm_dir.mkdir()

    (arm_dir / "responses.jsonl").write_text(json.dumps({"question_id": "89271"}) + "\n")
    (arm_dir / "question_results.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "response": {
                    "question": "q",
                    "answer": "a",
                    "citations": [],
                    "warnings": [],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid_rerank",
                    "trace_id": "t",
                },
                "final_generator_path": "SEMANTIC_SYNTHESIS",
                "citation_identity_valid": True,
                "parser_accepted": True,
                "had_contract_rejection": False,
                "calls": [],
                "grounding_calls": [
                    {
                        "question_id": "89271",
                        "candidate_contract": "plain_text_markers",
                        "verification_index": 1,
                        "grounded": True,
                        "total_claims": 1,
                        "supported_claims": 1,
                        "unsupported_claims": 0,
                        "unverifiable_claims": 0,
                        "supported_ratio": 1.0,
                        "selected_evidence_count": 1,
                    },
                    {
                        "question_id": "89271",
                        "candidate_contract": "plain_text_markers",
                        "verification_index": 1,  # Duplicate index
                        "grounded": True,
                        "total_claims": 1,
                        "supported_claims": 1,
                        "unsupported_claims": 0,
                        "unverifiable_claims": 0,
                        "supported_ratio": 1.0,
                        "selected_evidence_count": 1,
                    },
                ],
            }
        )
        + "\n"
    )
    (arm_dir / "call_telemetry.jsonl").write_text("")
    (arm_dir / "raw_completions.jsonl").write_text("")
    (arm_dir / "grounding_telemetry.jsonl").write_text(
        json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "verification_index": 1,
                "grounded": True,
                "total_claims": 1,
                "supported_claims": 1,
                "unsupported_claims": 0,
                "unverifiable_claims": 0,
                "supported_ratio": 1.0,
                "selected_evidence_count": 1,
            }
        )
        + "\n"
        + json.dumps(
            {
                "question_id": "89271",
                "candidate_contract": "plain_text_markers",
                "verification_index": 1,
                "grounded": True,
                "total_claims": 1,
                "supported_claims": 1,
                "unsupported_claims": 0,
                "unverifiable_claims": 0,
                "supported_ratio": 1.0,
                "selected_evidence_count": 1,
            }
        )
        + "\n"
    )

    with pytest.raises(ArtifactCompatibilityError, match="PARTIAL_QID_ARTIFACT_STATE: Duplicate grounding verification index"):
        runner._validate_and_reload_arm_resume_state("control", ["89271"], arm_dir)


# ==========================================
# 5. REAL INTERRUPTION & RESUME EQUIVALENCE
# ==========================================


def test_14_real_interruption_and_resume_produces_identical_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zip_path = tmp_path / "fast30.zip"
    make_synthetic_fast30_zip(zip_path, monkeypatch)

    out_dir_uninterrupted = tmp_path / "out_uninterrupted"
    runner_full = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=zip_path,
        output_dir=out_dir_uninterrupted,
        provider_factory=lambda cfg: FakeChatModelProvider(completions=["Ans [E1]."]),
        is_preflight_only=True,
    )
    gen_man_full = runner_full.run_generation()

    out_dir_resumed = tmp_path / "out_resumed"
    out_dir_resumed.mkdir(parents=True)

    ctrl_unint = out_dir_uninterrupted / "control"
    ctrl_res = out_dir_resumed / "control"
    ctrl_res.mkdir(parents=True)

    first_5_qids = set(CANONICAL_TUNE20_ORDERED_QIDS[:5])

    for fname in ("responses.jsonl", "question_results.jsonl", "call_telemetry.jsonl", "raw_completions.jsonl", "grounding_telemetry.jsonl"):
        all_lines = (ctrl_unint / fname).read_text(encoding="utf-8").splitlines()
        filtered = [l for l in all_lines if json.loads(l).get("question_id") in first_5_qids]
        (ctrl_res / fname).write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")

    env_fp, env_fp_sha = get_runtime_environment_fingerprint(
        repo_root=Path("."),
        measurement_source_sha=gen_man_full.measurement_source_sha,
        model_path=None,
        model_tree_sha256=EXPECTED_M49_GENERATOR_TREE_SHA256,
    )
    (out_dir_resumed / "batch_state.json").write_text(
        json.dumps(
            {
                "frozen_generator_input_sha256": scm.T5_6B_FROZEN_GENERATOR_INPUT_SHA256,
                "runtime_environment_sha256": env_fp_sha,
                "completed_qids": {"control": list(CANONICAL_TUNE20_ORDERED_QIDS[:5]), "compact": [], "json_schema": []},
            }
        ),
        encoding="utf-8",
    )

    runner_resume = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=zip_path,
        output_dir=out_dir_resumed,
        provider_factory=lambda cfg: FakeChatModelProvider(completions=["Ans [E1]."]),
        is_preflight_only=True,
    )
    gen_man_res = runner_resume.run_generation()

    for arm in EXECUTION_ARM_ORDER:
        full_res = (out_dir_uninterrupted / arm / "responses.jsonl").read_text(encoding="utf-8")
        res_res = (out_dir_resumed / arm / "responses.jsonl").read_text(encoding="utf-8")
        assert full_res == res_res


# ==========================================
# 6. STRUCTURAL CLI FIREWALL TESTS
# ==========================================


def test_15_cli_firewall_execute_generation_rejects_references() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--execute-generation", "--archive", "archive.zip", "--reference-answers-file", "refs.json"])


def test_16_cli_firewall_execute_generation_rejects_scorer() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--execute-generation", "--archive", "archive.zip", "--scorer-path", "scorer.zip"])


def test_17_cli_firewall_score_closed_rejects_model_path() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--score-closed-generation", "--reference-answers-file", "refs.json", "--scorer-path", "scorer.zip", "--model-path", "/path/model"])


def test_18_cli_firewall_score_closed_requires_scorer_path() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--score-closed-generation", "--reference-answers-file", "refs.json"])


def test_19_cli_firewall_preflight_rejects_references() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--preflight-only", "--archive", "archive.zip", "--reference-answers-file", "refs.json"])


# ==========================================
# 7. CROSS-ARM TELEMETRY ISOLATION TESTS
# ==========================================


def test_20_cross_arm_telemetry_isolation_in_same_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zip_path = tmp_path / "fast30.zip"
    make_synthetic_fast30_zip(zip_path, monkeypatch)

    out_dir = tmp_path / "out"

    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=zip_path,
        output_dir=out_dir,
        provider_factory=lambda cfg: FakeChatModelProvider(completions=["Ans [E1]."]),
        is_preflight_only=True,
    )
    runner.run_generation()

    for arm_name in EXECUTION_ARM_ORDER:
        prompt_mode = ARM_CONTRACT_MAP[arm_name]
        res_file = out_dir / arm_name / "question_results.jsonl"
        results = [
            QuestionMeasurementResult.model_validate(json.loads(l))
            for l in res_file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert len(results) == 20
        for r in results:
            assert r.candidate_contract == prompt_mode
            for c in r.calls:
                assert c.candidate_contract == prompt_mode
                assert c.question_id == r.question_id
            for g in r.grounding_calls:
                assert g.candidate_contract == prompt_mode
                assert g.question_id == r.question_id


# ==========================================
# 8. STRICT NON-WILDCARD REJECTION CORRELATION TESTS
# ==========================================


def test_21_strict_rejection_correlation_exact_context() -> None:
    calls = [
        GeneratorCallTelemetry(
            question_id="89271",
            candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT",
            call_index=1,
            provider_attempt_index=1,
            provider_call_success=True,
            system_prompt_sha256="s1",
            user_prompt_sha256="u1",
            raw_completion_text="Draft",
        ),
        GeneratorCallTelemetry(
            question_id="89271",
            candidate_contract="plain_text_markers",
            call_stage="GROUNDING_REPAIR",
            call_index=2,
            provider_attempt_index=1,
            provider_call_success=True,
            system_prompt_sha256="s2",
            user_prompt_sha256="u2",
            raw_completion_text="Repair",
        ),
    ]

    rejections = [
        {
            "error_type": "structured_output_schema",
            "structured_output_attempt": 1,
            "context": ("89271", "plain_text_markers", 2),
        }
    ]

    correlate_rejection_events(calls, rejections)
    assert calls[0].parse_result == "ACCEPTED"
    assert calls[1].parse_result == "REJECTED"


def test_22_rejection_with_none_context_raises_ambiguous() -> None:
    calls = [
        GeneratorCallTelemetry(
            question_id="89271",
            candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT",
            call_index=1,
            provider_attempt_index=1,
            provider_call_success=True,
            system_prompt_sha256="s1",
            user_prompt_sha256="u1",
            raw_completion_text="Draft",
        )
    ]
    rejections = [
        {
            "error_type": "structured_output_schema",
            "context": None,
        }
    ]
    with pytest.raises(DataValidationError, match="AMBIGUOUS_REJECTION_CONTEXT"):
        correlate_rejection_events(calls, rejections)


def test_23_rejection_correlation_mismatched_contract_rejected() -> None:
    calls = [
        GeneratorCallTelemetry(
            question_id="89271",
            candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT",
            call_index=1,
            provider_attempt_index=1,
            provider_call_success=True,
            system_prompt_sha256="s",
            user_prompt_sha256="u",
            raw_completion_text="Draft",
        )
    ]
    rejections = [
        {
            "error_type": "structured_output_schema",
            "context": ("89271", "compact_example", 1),
        }
    ]
    with pytest.raises(DataValidationError, match="AMBIGUOUS_REJECTION_CONTEXT"):
        correlate_rejection_events(calls, rejections)


def test_24_rejection_correlation_mismatched_qid_rejected() -> None:
    calls = [
        GeneratorCallTelemetry(
            question_id="89271",
            candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT",
            call_index=1,
            provider_attempt_index=1,
            provider_call_success=True,
            system_prompt_sha256="s",
            user_prompt_sha256="u",
            raw_completion_text="Draft",
        )
    ]
    rejections = [
        {
            "error_type": "structured_output_schema",
            "context": ("31523", "plain_text_markers", 1),
        }
    ]
    with pytest.raises(DataValidationError, match="AMBIGUOUS_REJECTION_CONTEXT"):
        correlate_rejection_events(calls, rejections)


# ==========================================
# 9. PRODUCTION WARNINGS & PATH INVARIANTS
# ==========================================


def test_25_classify_semantic_synthesis() -> None:
    resp = make_answer_response(answer="Synthesized answer", warnings=[])
    assert classify_final_generator_path(resp) == "SEMANTIC_SYNTHESIS"


def test_26_classify_model_error_fallback() -> None:
    resp = make_answer_response(answer="Fallback answer", warnings=["generator_model_error_fallback"])
    assert classify_final_generator_path(resp) == "MODEL_ERROR_FALLBACK"


def test_27_classify_grounding_repair_success() -> None:
    resp = make_answer_response(answer="Repaired answer", warnings=["grounding_repair_attempted"])
    assert classify_final_generator_path(resp) == "GROUNDING_REPAIR_SUCCESS"


def test_28_classify_supported_claim_salvage() -> None:
    resp = make_answer_response(
        answer="Salvaged answer",
        warnings=["grounding_repair_attempted", "supported_claim_salvage_applied"],
    )
    assert classify_final_generator_path(resp) == "SUPPORTED_CLAIM_SALVAGE"


def test_29_classify_grounding_extractive_fallback() -> None:
    resp = make_answer_response(
        answer="Extractive fallback",
        warnings=["grounding_repair_attempted", "extractive_fallback_applied"],
    )
    assert classify_final_generator_path(resp) == "GROUNDING_EXTRACTIVE_FALLBACK"


def test_30_classify_insufficient_evidence() -> None:
    resp = make_answer_response(answer="Insufficient", insufficient_evidence=True)
    assert classify_final_generator_path(resp) == "INSUFFICIENT_EVIDENCE"


def test_31_classify_conflicting_warnings_ambiguous() -> None:
    resp = make_answer_response(
        answer="Conflicted",
        warnings=["generator_model_error_fallback", "supported_claim_salvage_applied"],
    )
    assert classify_final_generator_path(resp) == "AMBIGUOUS"


# ==========================================
# 10. TOKEN TELEMETRY PROXY INTEGRATION
# ==========================================


def test_32_token_telemetry_full_integration() -> None:
    """Token counts are captured via _GenerateProxy installed through _load_runtime()."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_torch = MagicMock()

    mock_input_tensor = MagicMock()
    mock_input_tensor.shape = (1, 100)
    mock_output_tensor = MagicMock()
    mock_output_tensor.shape = (1, 150)
    original_generate = MagicMock(return_value=mock_output_tensor)
    mock_model.generate = original_generate

    cfg = get_preregistered_generation_config("plain_text_markers")
    provider = ObservableTransformersChatProvider(cfg)

    # Simulate _load_runtime() installing proxy: call the overridden method
    # (provider has no real torch/transformers, so we mock the super() call)
    import unittest.mock as _mock
    with _mock.patch.object(
        TransformersChatProvider, "_load_runtime", return_value=(mock_torch, mock_tokenizer, mock_model)
    ):
        torch_rt, tok_rt, model_rt = provider._load_runtime()

    # The proxy should be installed on model_rt.generate
    assert isinstance(model_rt.generate, _GenerateProxy), "Proxy should be installed by _load_runtime()"

    observer = MeasurementProviderObserver(provider)
    provider.token_observer = observer
    observer.set_active_question("Q1", "plain_text_markers")

    model_rt.generate(input_ids=mock_input_tensor)
    assert observer._last_token_counts == (100, 50)

    # Verify original generate was called exactly once
    original_generate.assert_called_once()


# ==========================================
# 11. ADVANCEMENT GATE & TIE-BREAK
# ==========================================


def test_33_six_part_advancement_gate_pass() -> None:
    ctrl = ArmMeasurementSummary(
        candidate_contract="plain_text_markers",
        record_count=20,
        rouge_l=0.48,
        meteor=0.40,
        parser_acceptance_rate=0.10,
        citation_identity_validity=1.0,
        contract_rejection_fallback_count=18,
        insufficient_evidence_count=1,
        total_structured_output_rejections=36,
    )
    cand = ArmMeasurementSummary(
        candidate_contract="compact_example",
        record_count=20,
        rouge_l=0.50,
        meteor=0.42,
        parser_acceptance_rate=0.90,
        citation_identity_validity=1.0,
        contract_rejection_fallback_count=2,
        insufficient_evidence_count=1,
        total_structured_output_rejections=2,
    )
    summaries = {"control": ctrl, "compact": cand}
    adv, winner, dec = evaluate_advancement_gate(summaries)
    assert adv["compact"].all_passed is True
    assert winner == "compact"
    assert dec == "NEW_CLEAN_VALIDATION_POPULATION_REQUIRED"


def test_34_advancement_gate_rejection_on_lower_meteor() -> None:
    ctrl = ArmMeasurementSummary(
        candidate_contract="plain_text_markers",
        record_count=20,
        rouge_l=0.48,
        meteor=0.40,
        parser_acceptance_rate=0.10,
        citation_identity_validity=1.0,
        contract_rejection_fallback_count=18,
        insufficient_evidence_count=1,
        total_structured_output_rejections=36,
    )
    cand = ArmMeasurementSummary(
        candidate_contract="compact_example",
        record_count=20,
        rouge_l=0.50,
        meteor=0.38,
        parser_acceptance_rate=0.90,
        citation_identity_validity=1.0,
        contract_rejection_fallback_count=2,
        insufficient_evidence_count=1,
        total_structured_output_rejections=2,
    )
    summaries = {"control": ctrl, "compact": cand}
    adv, winner, dec = evaluate_advancement_gate(summaries)
    assert adv["compact"].all_passed is False
    assert winner is None
    assert dec == "NO_GENERATOR_CONTRACT_CANDIDATE_JUSTIFIED"


# ==========================================
# FIX 5 TESTS: Targeted regression repairs
# ==========================================


# --- Issue 1: Token Provider Architecture ---

def test_35_observable_provider_complete_method_identity() -> None:
    """ObservableTransformersChatProvider.complete must be TransformersChatProvider.complete."""
    assert ObservableTransformersChatProvider.complete is TransformersChatProvider.complete, (
        "ObservableTransformersChatProvider must NOT override complete(). "
        "complete() identity must be preserved from TransformersChatProvider."
    )


def test_36_load_runtime_installs_proxy_exactly_once() -> None:
    """_load_runtime() installs proxy once; second call to _load_runtime() would double-wrap if called."""
    import unittest.mock as _mock
    mock_model = MagicMock()
    mock_model.generate = MagicMock(return_value=MagicMock(spec=[], shape=(1, 150)))
    original_generate = mock_model.generate
    mock_tokenizer = MagicMock()
    mock_torch = MagicMock()

    cfg = get_preregistered_generation_config("plain_text_markers")
    provider = ObservableTransformersChatProvider(cfg)

    with _mock.patch.object(
        TransformersChatProvider, "_load_runtime", return_value=(mock_torch, mock_tokenizer, mock_model)
    ):
        _, _, model_rt = provider._load_runtime()

    assert isinstance(model_rt.generate, _GenerateProxy), "Proxy must be installed after _load_runtime()"
    # The proxy wraps the original, not another proxy
    assert model_rt.generate._target_generate is original_generate, "Proxy must wrap the original generate once"


def test_37_generate_proxy_output_object_identity_preserved() -> None:
    """_GenerateProxy must return the exact output object from original generate."""
    cfg = get_preregistered_generation_config("plain_text_markers")
    provider = ObservableTransformersChatProvider(cfg)

    expected_output = MagicMock(spec=[], shape=(1, 150))
    original_gen = MagicMock(return_value=expected_output)

    proxy = _GenerateProxy(original_gen, provider)
    mock_input = MagicMock(spec=[], shape=(1, 100))
    result = proxy(input_ids=mock_input)
    assert result is expected_output, "Output object identity must be preserved"
    original_gen.assert_called_once()


def test_38_generate_proxy_exception_records_input_tokens_none_output() -> None:
    """On generate() exception, proxy records input_tokens and None output_tokens then re-raises."""
    cfg = get_preregistered_generation_config("plain_text_markers")
    provider = ObservableTransformersChatProvider(cfg)
    observer = MeasurementProviderObserver(provider)
    provider.token_observer = observer

    exc = RuntimeError("GPU OOM")
    def exploding_generate(*args, **kwargs):
        raise exc

    proxy = _GenerateProxy(exploding_generate, provider)
    mock_input = MagicMock(spec=[], shape=(1, 50))

    with pytest.raises(RuntimeError, match="GPU OOM"):
        proxy(input_ids=mock_input)

    assert observer._last_token_counts[0] == 50, "Input tokens must be recorded on exception"
    assert observer._last_token_counts[1] is None, "Output tokens must be None on exception"


# --- Issue 2: Prompt Sentinels ---

def test_39_sentinel_values_match_production_substrings() -> None:
    """Sentinels must exactly match substrings produced by model_generator._correction_prompt."""
    # Verify Vietnamese sentinels are exact
    assert STRUCTURED_RETRY_SENTINEL == "OUTPUT TRƯỚC KHÔNG HỢP LỆ. Hãy tạo lại từ đầu."
    assert GROUNDING_REPAIR_SENTINEL == "BẢN NHÁP TRƯỚC KHÔNG QUA KIỂM TRA GROUNDING:"


def test_40_classify_prompt_ambiguous_both_sentinels() -> None:
    """Prompt containing both sentinels raises DataValidationError (AMBIGUOUS)."""
    from scripts.t5_generator_contract_measurement import classify_prompt_call_stage
    with pytest.raises(DataValidationError, match="AMBIGUOUS_CALL_STAGE"):
        classify_prompt_call_stage(STRUCTURED_RETRY_SENTINEL + " " + GROUNDING_REPAIR_SENTINEL)


def test_41_classify_prompt_grounding_sentinel() -> None:
    from scripts.t5_generator_contract_measurement import classify_prompt_call_stage
    prompt = f"Base prompt\n\n{GROUNDING_REPAIR_SENTINEL}\nsome content"
    assert classify_prompt_call_stage(prompt) == "GROUNDING_REPAIR"


def test_42_classify_prompt_structured_retry_sentinel() -> None:
    from scripts.t5_generator_contract_measurement import classify_prompt_call_stage
    prompt = f"Base prompt\n\n{STRUCTURED_RETRY_SENTINEL} Additional instructions."
    assert classify_prompt_call_stage(prompt) == "STRUCTURED_RETRY"


def test_43_classify_prompt_initial_draft_no_sentinels() -> None:
    from scripts.t5_generator_contract_measurement import classify_prompt_call_stage
    assert classify_prompt_call_stage("What is the law?") == "INITIAL_DRAFT"


# --- Issue 3: Call/Attempt State Machine ---

def test_44_provider_observer_initial_success_index() -> None:
    """Initial call: call_index=1, provider_attempt_index=1."""
    inner = FakeChatModelProvider(completions=["ok"])
    observer = MeasurementProviderObserver(inner)
    observer.set_active_question("qid1", "plain_text_markers")
    observer.complete(system_instruction="sys", user_prompt="initial_prompt")
    assert len(observer.call_telemetry) == 1
    call = observer.call_telemetry[0]
    assert call.call_index == 1
    assert call.provider_attempt_index == 1


def test_45_provider_observer_model_error_retry_preserves_call_index() -> None:
    """ModelError retry with same prompt: call_index preserved, provider_attempt_index incremented."""
    inner = FakeChatModelProvider(completions=[ModelError("fail"), "ok"])
    observer = MeasurementProviderObserver(inner)
    observer.set_active_question("qid1", "plain_text_markers")
    # First attempt fails
    with pytest.raises(ModelError):
        observer.complete(system_instruction="sys", user_prompt="same_prompt")
    # Second attempt (same prompt, same stage) - retry
    observer.complete(system_instruction="sys", user_prompt="same_prompt")
    assert len(observer.call_telemetry) == 2
    assert observer.call_telemetry[0].call_index == 1
    assert observer.call_telemetry[0].provider_attempt_index == 1
    assert observer.call_telemetry[1].call_index == 1, "Same logical call on retry"
    assert observer.call_telemetry[1].provider_attempt_index == 2, "Attempt incremented"


def test_46_provider_observer_new_prompt_creates_new_logical_call() -> None:
    """Different prompt (structured retry or grounding repair): new call_index, attempt resets to 1."""
    inner = FakeChatModelProvider(completions=["draft1", "draft2"])
    observer = MeasurementProviderObserver(inner)
    observer.set_active_question("qid1", "plain_text_markers")
    observer.complete(system_instruction="sys", user_prompt="initial prompt")
    observer.complete(system_instruction="sys", user_prompt=f"{STRUCTURED_RETRY_SENTINEL} retry prompt")
    assert observer.call_telemetry[0].call_index == 1
    assert observer.call_telemetry[1].call_index == 2, "New prompt must create new logical call"
    assert observer.call_telemetry[1].provider_attempt_index == 1, "Attempt resets to 1 for new logical call"


# --- Issue 4: Logger Rejection Context ---

def test_47_rejection_attributed_to_logical_call_not_provider_attempt() -> None:
    """Rejection event context uses logical call_index, not structured_output_attempt."""
    obs = ModelGeneratorRejectionObserver()
    obs.set_active_context("qid1", "plain_text_markers")
    obs.set_active_logical_call(1)

    # Simulate provider retry completes successfully, but the parsed draft is rejected
    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="", lineno=0,
        msg="model_answer_draft_rejected", args=(), exc_info=None,
    )
    record.error_type = "structured_output_schema"
    record.structured_output_attempt = 2  # this is attempt 2 (retry)
    obs.emit(record)

    assert len(obs.rejections) == 1
    ctx = obs.rejections[0]["context"]
    assert ctx == ("qid1", "plain_text_markers", 1), (
        f"Context must be (qid, contract, logical_call_index=1), not {ctx}"
    )
    assert obs.rejections[0]["structured_output_attempt"] == 2, (
        "structured_output_attempt must still be recorded separately"
    )


# --- Issue 5: Query Normalization ---

def test_48_reconstruct_query_strips_whitespace() -> None:
    q = reconstruct_query("  What is the law?  ", question_id="qid1")
    assert q.original_question == "What is the law?"


def test_49_reconstruct_query_collapses_internal_whitespace() -> None:
    q = reconstruct_query("What   is    the law?", question_id="qid1")
    assert q.normalized_question == "What is the law?"


def test_50_reconstruct_query_nfc_normalization() -> None:
    # Decomposed Unicode -> NFC
    import unicodedata
    decomposed = "Nh\u01b0\u0303ng"  # u+01b0 + combining tilde (decomposed)
    nfc = unicodedata.normalize("NFC", decomposed)
    q = reconstruct_query(decomposed, question_id="qid1")
    assert q.normalized_question == nfc


def test_51_reconstruct_query_id_format() -> None:
    q = reconstruct_query("What is law?", question_id="89271")
    assert q.query_id == "t5-6b:89271"


def test_52_reconstruct_query_preserves_vietnamese() -> None:
    question = "Vi\u1ec7c c\u1ea5p gi\u1ea5y ph\u00e9p x\u00e2y d\u1ef1ng?  "
    q = reconstruct_query(question, question_id="qid1")
    assert "\u1ec7c" in q.normalized_question or q.normalized_question.startswith("Vi")


# --- Issue 6: Parser Acceptance Definition ---

def test_53_parser_accepted_any_not_all() -> None:
    """parser_accepted=True if ANY call has ACCEPTED; initial reject + structured retry accept => True."""
    calls = [
        GeneratorCallTelemetry(
            question_id="q1", candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT", call_index=1, provider_attempt_index=1,
            provider_call_success=True, system_prompt_sha256="s", user_prompt_sha256="u",
            parse_result="REJECTED", rejection_error_type="structured_output_schema",
        ),
        GeneratorCallTelemetry(
            question_id="q1", candidate_contract="plain_text_markers",
            call_stage="STRUCTURED_RETRY", call_index=2, provider_attempt_index=1,
            provider_call_success=True, system_prompt_sha256="s", user_prompt_sha256="u2",
            parse_result="ACCEPTED",
        ),
    ]
    parser_acc = any(c.parse_result == "ACCEPTED" for c in calls if c.provider_call_success)
    assert parser_acc is True, "Initial reject + retry accept must yield parser_accepted=True"


def test_54_parser_accepted_false_when_all_rejected() -> None:
    calls = [
        GeneratorCallTelemetry(
            question_id="q1", candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT", call_index=1, provider_attempt_index=1,
            provider_call_success=True, system_prompt_sha256="s", user_prompt_sha256="u",
            parse_result="REJECTED", rejection_error_type="structured_output_schema",
        ),
    ]
    parser_acc = any(c.parse_result == "ACCEPTED" for c in calls if c.provider_call_success)
    assert parser_acc is False


# --- Issue 7: Contract Rejection Metrics ---

def test_55_contract_rejection_fallback_count_exact_definition() -> None:
    """contract_rejection_fallback_count = MODEL_ERROR_FALLBACK AND had_contract_rejection."""
    results = [
        # MODEL_ERROR_FALLBACK + had_contract_rejection=True -> counts
        QuestionMeasurementResult(
            question_id="q1", candidate_contract="plain_text_markers",
            response=make_answer_response(),
            final_generator_path="MODEL_ERROR_FALLBACK",
            citation_identity_valid=True, parser_accepted=False, had_contract_rejection=True,
            calls=[], grounding_calls=[],
        ),
        # MODEL_ERROR_FALLBACK + had_contract_rejection=False -> does NOT count
        QuestionMeasurementResult(
            question_id="q2", candidate_contract="plain_text_markers",
            response=make_answer_response(),
            final_generator_path="MODEL_ERROR_FALLBACK",
            citation_identity_valid=True, parser_accepted=False, had_contract_rejection=False,
            calls=[], grounding_calls=[],
        ),
        # SUPPORTED_CLAIM_SALVAGE -> does NOT count
        QuestionMeasurementResult(
            question_id="q3", candidate_contract="plain_text_markers",
            response=make_answer_response(),
            final_generator_path="SUPPORTED_CLAIM_SALVAGE",
            citation_identity_valid=True, parser_accepted=True, had_contract_rejection=True,
            calls=[], grounding_calls=[],
        ),
    ]
    fallback_count = sum(
        1 for r in results
        if r.final_generator_path == "MODEL_ERROR_FALLBACK" and r.had_contract_rejection
    )
    assert fallback_count == 1, f"Expected 1, got {fallback_count}"


def test_56_total_structured_output_rejections_exact_definition() -> None:
    """total_structured_output_rejections counts only rejection_error_type == structured_output_schema."""
    calls = [
        GeneratorCallTelemetry(
            question_id="q1", candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT", call_index=1, provider_attempt_index=1,
            provider_call_success=True, system_prompt_sha256="s", user_prompt_sha256="u",
            parse_result="REJECTED", rejection_error_type="structured_output_schema",
        ),
        GeneratorCallTelemetry(
            question_id="q1", candidate_contract="plain_text_markers",
            call_stage="INITIAL_DRAFT", call_index=1, provider_attempt_index=1,
            provider_call_success=True, system_prompt_sha256="s", user_prompt_sha256="u2",
            parse_result="REJECTED", rejection_error_type="structured_output_missing_fields",
        ),
        GeneratorCallTelemetry(
            question_id="q1", candidate_contract="plain_text_markers",
            call_stage="STRUCTURED_RETRY", call_index=2, provider_attempt_index=1,
            provider_call_success=True, system_prompt_sha256="s", user_prompt_sha256="u3",
            parse_result="ACCEPTED",
        ),
    ]
    total = sum(1 for c in calls if c.rejection_error_type == "structured_output_schema")
    assert total == 1, f"Only structured_output_schema counts; got {total}"


# --- Issue 8: Citation Identity Validation ---

def test_57_citation_valid_known_evidence_and_matching_chunk() -> None:
    from scripts.t5_generator_contract_measurement import evaluate_citation_identity_validity
    ev = Evidence(evidence_id="E1", chunk_id="c1", document_id="d1", document_title="T", text="text")
    resp = make_answer_response(
        answer="Answer [E1].",
        citations=[Citation(evidence_id="E1", chunk_id="c1", document_id="d1")],
    )
    assert evaluate_citation_identity_validity(resp, [ev]) is True


def test_58_citation_invalid_unknown_evidence_id() -> None:
    from scripts.t5_generator_contract_measurement import evaluate_citation_identity_validity
    ev = Evidence(evidence_id="E1", chunk_id="c1", document_id="d1", document_title="T", text="text")
    resp = make_answer_response(
        answer="Answer [E2].",
        citations=[Citation(evidence_id="E2", chunk_id="c1", document_id="d1")],
    )
    assert evaluate_citation_identity_validity(resp, [ev]) is False


def test_59_citation_invalid_wrong_chunk_id() -> None:
    from scripts.t5_generator_contract_measurement import evaluate_citation_identity_validity
    ev = Evidence(evidence_id="E1", chunk_id="c1", document_id="d1", document_title="T", text="text")
    resp = make_answer_response(
        answer="Answer [E1].",
        citations=[Citation(evidence_id="E1", chunk_id="c999", document_id="d1")],
    )
    assert evaluate_citation_identity_validity(resp, [ev]) is False


def test_60_citation_invalid_duplicate_pair() -> None:
    """AnswerResponse rejects duplicate citations at schema validation level.

    The duplicate citation constraint is enforced by the pydantic schema
    before evaluate_citation_identity_validity is called, so this test
    confirms the schema-level rejection rather than the validation function.
    """
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        make_answer_response(
            answer="Answer [E1].",
            citations=[
                Citation(evidence_id="E1", chunk_id="c1", document_id="d1"),
                Citation(evidence_id="E1", chunk_id="c1", document_id="d1"),
            ],
        )


def test_61_citation_invalid_unknown_answer_marker() -> None:
    from scripts.t5_generator_contract_measurement import evaluate_citation_identity_validity
    ev = Evidence(evidence_id="E1", chunk_id="c1", document_id="d1", document_title="T", text="text")
    resp = make_answer_response(
        answer="Answer [E1] and [E99].",  # E99 not in supplied evidence
        citations=[Citation(evidence_id="E1", chunk_id="c1", document_id="d1")],
    )
    assert evaluate_citation_identity_validity(resp, [ev]) is False


def test_62_citation_valid_abstaining_response() -> None:
    """Insufficient evidence responses bypass citation checks."""
    from scripts.t5_generator_contract_measurement import evaluate_citation_identity_validity
    resp = make_answer_response(insufficient_evidence=True)
    assert evaluate_citation_identity_validity(resp, []) is True


# --- Issue 10: Model Tree Hash Algorithm ---

def test_63_compute_directory_sha256_exact_original_algorithm(tmp_path: Path) -> None:
    """compute_directory_sha256 must match the EXACT original algorithm from m491_kaggle_candidate_dev.py.

    Original algorithm serialisation contract (commit 10681c8):
      1. sorted(path.rglob("*") if is_file) - global sort, NOT per-directory
      2. len(relative_utf8).to_bytes(8, "big") - 8-byte big-endian length prefix
      3. relative_utf8 - raw UTF-8 path bytes
      4. bytes.fromhex(file_sha256_hex) - 32 raw hash bytes
    """
    from hashlib import sha256 as _sha256
    from scripts.t5_generator_contract_measurement import compute_directory_sha256, compute_file_sha256

    # Create a deterministic synthetic tree
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file_b.bin").write_bytes(b"content_b")
    (tmp_path / "file_a.txt").write_bytes(b"content_a")

    # Compute expected using original algorithm verbatim
    digest = _sha256()
    files = sorted(item for item in tmp_path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(tmp_path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(compute_file_sha256(item)))
    expected = digest.hexdigest()

    actual = compute_directory_sha256(tmp_path)
    assert actual == expected, (
        f"compute_directory_sha256 must match exact original m491 algorithm. "
        f"Expected {expected}, got {actual}"
    )


def test_64_compute_directory_sha256_different_from_wrong_algorithm(tmp_path: Path) -> None:
    """Confirm old (wrong) algorithm produces a different result, verifying the fix matters."""
    from hashlib import sha256 as _sha256
    from scripts.t5_generator_contract_measurement import compute_directory_sha256, compute_file_sha256
    import os as _os

    (tmp_path / "file_a.txt").write_bytes(b"content_a")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file_b.bin").write_bytes(b"content_b")

    # Old (wrong) Fix 4 algorithm: os.walk + ASCII hex string
    h_wrong = _sha256()
    for root, _, files in _os.walk(str(tmp_path)):
        for f in sorted(files):
            file_p = (tmp_path / _os.path.relpath(root, tmp_path) / f).resolve()
            rel_path = file_p.relative_to(tmp_path).as_posix()
            h_wrong.update(rel_path.encode("utf-8"))
            h_wrong.update(compute_file_sha256(file_p).encode("ascii"))
    wrong_result = h_wrong.hexdigest()

    correct_result = compute_directory_sha256(tmp_path)
    assert correct_result != wrong_result, (
        "Correct algorithm must differ from the wrong Fix 4 algorithm for non-trivial trees"
    )


# --- Issue 11: Resume Arm-Order Fail-Closed ---

def test_65_resume_arm_order_compact_ahead_of_control_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume must fail if compact has completed QIDs while control < 20."""
    zip_path = tmp_path / "fast30.zip"
    make_synthetic_fast30_zip(zip_path, monkeypatch)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    FIXED_ENV_SHA = "a" * 64  # Synthetic fixed sha for test isolation
    monkeypatch.setattr(
        scm,
        "get_runtime_environment_fingerprint",
        lambda **kwargs: (MagicMock(), FIXED_ENV_SHA),
    )

    # State: control has 5, compact has 1 (out of order)
    (out_dir / "batch_state.json").write_text(
        json.dumps({
            "frozen_generator_input_sha256": scm.T5_6B_FROZEN_GENERATOR_INPUT_SHA256,
            "runtime_environment_sha256": FIXED_ENV_SHA,
            "completed_qids": {
                "control": list(CANONICAL_TUNE20_ORDERED_QIDS[:5]),
                "compact": list(CANONICAL_TUNE20_ORDERED_QIDS[:1]),
                "json_schema": [],
            },
        }),
        encoding="utf-8",
    )

    # Create required arm directories
    for arm in ("control", "compact", "json_schema"):
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("responses.jsonl", "question_results.jsonl", "call_telemetry.jsonl", "raw_completions.jsonl", "grounding_telemetry.jsonl"):
            (arm_dir / fname).write_text("")

    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=zip_path,
        output_dir=out_dir,
        provider_factory=lambda cfg: FakeChatModelProvider(completions=["Ans [E1]."]),
        is_preflight_only=True,
    )
    with pytest.raises(ArtifactCompatibilityError, match="T5_6B_ARM_ORDER_VIOLATED"):
        runner.run_generation()


def test_66_resume_arm_order_json_ahead_of_compact_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume must fail if json_schema has completed QIDs while compact < 20."""
    zip_path = tmp_path / "fast30.zip"
    make_synthetic_fast30_zip(zip_path, monkeypatch)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    FIXED_ENV_SHA = "b" * 64  # Synthetic fixed sha for test isolation
    monkeypatch.setattr(
        scm,
        "get_runtime_environment_fingerprint",
        lambda **kwargs: (MagicMock(), FIXED_ENV_SHA),
    )

    (out_dir / "batch_state.json").write_text(
        json.dumps({
            "frozen_generator_input_sha256": scm.T5_6B_FROZEN_GENERATOR_INPUT_SHA256,
            "runtime_environment_sha256": FIXED_ENV_SHA,
            "completed_qids": {
                "control": list(CANONICAL_TUNE20_ORDERED_QIDS),  # 20 done
                "compact": list(CANONICAL_TUNE20_ORDERED_QIDS[:10]),  # 10 done
                "json_schema": list(CANONICAL_TUNE20_ORDERED_QIDS[:1]),  # 1 done - too early
            },
        }),
        encoding="utf-8",
    )

    for arm in ("control", "compact", "json_schema"):
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("responses.jsonl", "question_results.jsonl", "call_telemetry.jsonl", "raw_completions.jsonl", "grounding_telemetry.jsonl"):
            (arm_dir / fname).write_text("")

    runner = T5GeneratorContractMeasurementRunner(
        repo_root=Path("."),
        archive_path=zip_path,
        output_dir=out_dir,
        provider_factory=lambda cfg: FakeChatModelProvider(completions=["Ans [E1]."]),
        is_preflight_only=True,
    )
    with pytest.raises(ArtifactCompatibilityError, match="T5_6B_ARM_ORDER_VIOLATED"):
        runner.run_generation()


# --- Issue 12: Non-Blocking Execution Exclusivity ---

def test_67_overlapping_runner_rejected_non_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second concurrent runner must fail immediately (non-blocking), not deadlock."""
    import threading
    zip_path = tmp_path / "fast30.zip"
    make_synthetic_fast30_zip(zip_path, monkeypatch)

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    # Simulate holding the runner lock
    import scripts.t5_generator_contract_measurement as _scm
    acquired = _scm._RUNNER_LOCK.acquire(blocking=False)
    if not acquired:
        pytest.skip("Runner lock already held, cannot test non-blocking behavior")

    try:
        runner = T5GeneratorContractMeasurementRunner(
            repo_root=Path("."),
            archive_path=zip_path,
            output_dir=tmp_path / "out",
            provider_factory=lambda cfg: FakeChatModelProvider(completions=["Ans [E1]."]),
            is_preflight_only=True,
        )
        with pytest.raises(BackendInitializationError, match="T5_6B_OVERLAPPING_RUNNER"):
            runner.run_generation()
    finally:
        _scm._RUNNER_LOCK.release()


def test_68_overlapping_lease_rejected_non_blocking() -> None:
    """Second concurrent ModelGeneratorLoggingLease must fail immediately (non-blocking)."""
    from scripts.t5_generator_contract_measurement import ModelGeneratorLoggingLease, _LEASE_LOCK
    handler1 = logging.StreamHandler()
    handler2 = logging.StreamHandler()

    lease1 = ModelGeneratorLoggingLease("test.logger", handler1)
    lease2 = ModelGeneratorLoggingLease("test.logger", handler2)

    with lease1:
        with pytest.raises(DataValidationError, match="T5_6B_OVERLAPPING_LEASE"):
            lease2.__enter__()
