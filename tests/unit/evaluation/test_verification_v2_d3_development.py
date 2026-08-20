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
    CANONICAL_V3_MAX_INPUT_TOKENS,
    CANONICAL_V3_MAX_OUTPUT_TOKENS,
    CANONICAL_V3_MAX_STRUCTURED_RETRIES,
    CANONICAL_V3_MODEL_NAME,
    CANONICAL_V3_MODEL_REVISION,
    CANONICAL_V3_PROVIDER_VERSION,
    CANONICAL_V3_TEMPERATURE,
    CANONICAL_V3_TIMEOUT_SECONDS,
    CANONICAL_V3_TORCH_DTYPE,
    BenchmarkArmTarget,
    BenchmarkClaimTarget,
    BinaryPrediction,
    HumanEntailment,
    ObservationalChatModelProviderWrapper,
    V2D3DevelopmentBenchmarkEvaluator,
    parse_args,
)


class MockD3ChatProvider(ChatModelProvider):
    """Mock ChatModelProvider for D3 testing."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.call_count = 0
        self.call_history: list[dict[str, str]] = []

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
        human_label=label,
        error_tags=tags or [],
        diagnostic_note=None,
        stratum="A_SINGLE_CLAIM_CLEAN",
    )


def _make_dummy_evidence(eid: str = "E1") -> Evidence:
    return Evidence(
        evidence_id=eid,
        chunk_id=f"chunk_{eid}",
        document_id="doc_001",
        document_title="Luật Doanh nghiệp 2020",
        document_number="59/2020/QH14",
        article_number="10",
        article_title="Tiêu chí doanh nghiệp",
        effect_status="active",
        text="Doanh nghiệp nhà nước bao gồm doanh nghiệp do Nhà nước nắm giữ 100% vốn điều lệ.",
    )


def _make_dummy_arm_target(
    qid: str,
    arm_id: str,
    claims: list[BenchmarkClaimTarget],
) -> BenchmarkArmTarget:
    ev = _make_dummy_evidence("E1")
    citation = Citation(
        evidence_id="E1",
        chunk_id="chunk_E1",
        document_id="doc_001",
        document_title="Luật Doanh nghiệp 2020",
        document_number="59/2020/QH14",
        article_number="10",
    )
    return BenchmarkArmTarget(
        slice_id="test_slice",
        question_id=qid,
        arm_id=arm_id,
        historical_stop_reason="answer_verified",
        stratum="A_SINGLE_CLAIM_CLEAN",
        question_text="Test question",
        answer_response=AnswerResponse(
            question="Test question",
            answer="Doanh nghiệp nhà nước do Nhà nước nắm giữ 100% vốn điều lệ [E1].",
            citations=[citation],
            insufficient_evidence=False,
            retrieval_strategy="hybrid",
            trace_id="test_trace",
        ),
        evidence_list=[ev],
        historical_verification={},
        claims=claims,
    )


# 1. Real SemanticVerificationConfig path succeeds
def test_real_semantic_verification_config_path_succeeds():
    cfg = SemanticVerificationConfig(
        backend=CANONICAL_V3_BACKEND,
        model_name=CANONICAL_V3_MODEL_NAME,
        model_revision=CANONICAL_V3_MODEL_REVISION,
        device="cuda",
        torch_dtype="float16",
        local_files_only=False,
        timeout_seconds=180.0,
        max_input_tokens=8192,
        max_output_tokens=512,
        max_structured_output_retries=1,
    )
    gen_cfg = cfg.as_generation_config()
    assert isinstance(gen_cfg, GenerationConfig)
    assert gen_cfg.backend == "transformers"
    assert gen_cfg.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert gen_cfg.model_revision == "a1d308dfcc03e09da285d49d912439a655a571e8"
    assert gen_cfg.device == "cuda"
    assert gen_cfg.torch_dtype == "float16"
    assert gen_cfg.temperature == 0.0


# 2. Unsupported constructor pattern raises ValidationError (extra='forbid')
def test_unsupported_provider_model_temperature_constructor_raises():
    with pytest.raises(ValidationError):
        SemanticVerificationConfig(
            provider="transformers",  # invalid extra field
            model="Qwen/Qwen2.5-3B-Instruct",  # invalid extra field
            temperature=0.0,  # invalid extra field
        )


# 3 & 4. TransformersChatProvider signature regression test & exact invariants
def test_transformers_chat_provider_signature_with_mock():
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
        device="cuda",
    )

    with patch(
        "scripts.evaluate_verification_v2_d3_development.TransformersChatProvider",
        autospec=True,
    ) as mock_tcp:
        evaluator._init_v3_provider()
        mock_tcp.assert_called_once()
        assert len(mock_tcp.call_args.args) == 1
        gen_cfg = mock_tcp.call_args.args[0]
        assert isinstance(gen_cfg, GenerationConfig)
        assert gen_cfg.backend == CANONICAL_V3_BACKEND
        assert gen_cfg.model_name == CANONICAL_V3_MODEL_NAME
        assert gen_cfg.model_revision == CANONICAL_V3_MODEL_REVISION
        assert gen_cfg.device == "cuda"
        assert gen_cfg.torch_dtype == "float16"
        assert gen_cfg.local_files_only is False
        assert gen_cfg.timeout_seconds == 180.0
        assert gen_cfg.max_input_tokens == 8192
        assert gen_cfg.max_output_tokens == 512
        assert gen_cfg.max_structured_output_retries == 1
        assert gen_cfg.temperature == 0.0


# 5. Observational wrapper counts successful call
def test_observational_wrapper_counts_successful_call():
    inner = MockD3ChatProvider([json.dumps({"claim_id": "C1", "relation": "ENTAILS"})])
    obs = ObservationalChatModelProviderWrapper(inner)
    assert obs.total_calls == 0
    assert obs.failed_call_count == 0

    res = obs.complete(system_instruction="sys", user_prompt="user prompt text")
    assert "ENTAILS" in res
    assert obs.total_calls == 1
    assert obs.failed_call_count == 0
    assert len(obs.call_history) == 1
    rec = obs.call_history[0]
    assert rec["call_succeeded"] is True
    assert rec["user_prompt_length"] == len("user prompt text")
    assert rec["completion_length"] > 0
    assert rec["system_instruction_sha256"] == sha256(b"sys").hexdigest()
    assert rec["user_prompt_sha256"] == sha256(b"user prompt text").hexdigest()


# 6 & 7. Observational wrapper counts exception call & persists content-safe error identity
def test_observational_wrapper_counts_exception_call():
    inner = MockD3ChatProvider([ModelError("CUDA out of memory error details with raw prompt")])
    obs = ObservationalChatModelProviderWrapper(inner)

    with pytest.raises(ModelError):
        obs.complete(system_instruction="sys", user_prompt="user prompt text")

    assert obs.total_calls == 1
    assert obs.failed_call_count == 1
    rec = obs.call_history[0]
    assert rec["call_succeeded"] is False
    assert rec["exception_type"] == "ModelError"
    # Raw exception message not persisted as raw string
    assert "exception_message_sha256" in rec
    assert "CUDA out of memory" not in json.dumps(rec)


# 8 & 9 & 10. Calls per pass, retries, and reconciliation
def test_claim_provider_call_sum_reconciles_with_observational_calls():
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    t1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)
    arm = _make_dummy_arm_target("q1", "PRIMARY", [t1])

    # 1 retry then success -> 2 calls
    p1_bad = "NOT JSON"
    p1_good = json.dumps({
        "claim_id": "C1",
        "relation": "ENTAILS",
        "actor_mismatch": False,
        "condition_exception_mismatch": False,
        "quantity_temporal_mismatch": False,
        "negation_modality_mismatch": False,
        "source_scope_mismatch": False,
    })

    inner = MockD3ChatProvider([p1_bad, p1_good])
    obs = ObservationalChatModelProviderWrapper(inner)
    verifier = StructuredSemanticCitationVerifierD3(obs, max_structured_output_retries=1)

    arm_res, claim_preds, pass_telem = evaluator._run_inference_pass(verifier, obs, [arm], pass_index=1)
    assert pass_telem["provider_calls"] == 2
    assert pass_telem["structured_retries"] == 1
    assert pass_telem["semantic_execution_errors"] == 0
    assert pass_telem["draft_rejection_categories"] == {"JSON_PARSE_ERROR": 1}


# 11 & 12. Two-pass retry independence and aggregate totals
def test_two_pass_retry_independence_and_aggregate_totals():
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    t1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)
    arm = _make_dummy_arm_target("q1", "PRIMARY", [t1])

    pass1_telem = {
        "pass_index": 1,
        "provider_calls": 2,
        "provider_invocation_errors": 0,
        "structured_retries": 1,
        "semantic_execution_errors": 0,
        "draft_rejection_categories": {"JSON_PARSE_ERROR": 1},
    }
    pass2_telem = {
        "pass_index": 2,
        "provider_calls": 3,
        "provider_invocation_errors": 0,
        "structured_retries": 2,
        "semantic_execution_errors": 0,
        "draft_rejection_categories": {"CLAIM_ID_MISMATCH": 2},
    }

    stability_info = {
        "total_claims": 1,
        "claims_with_two_valid_semantic_labels": 1,
        "stable_semantic_claim_count": 1,
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
        "v2_d3_claim_binary": {"tp": 1, "fp": 0, "tn": 0, "fn": 0, "execution_errors": 0},
        "v1_three_way": {},
        "v2_d3_three_way": {"execution_errors": 0},
        "paired_v1_vs_v2_d3": {"net_correctness_delta": 1},
        "v0_answer_metrics": {},
        "v1_answer_metrics": {},
        "v2_d3_answer_metrics": {"answer_level_accuracy": 1.0},
        "v2_d3_vs_v1_answer_deltas": {},
    }

    dim_diag = {
        "schema_version": "1.0",
        "artifact_type": "v2_d3_dimension_diagnostics",
        "diagnostic_pass": 1,
        "total_claims": 1,
        "successfully_structured_claim_count": 1,
        "execution_error_claim_count": 0,
        "relation_distribution": {"ENTAILS": 1},
        "diagnostic_mismatch_flag_counts": {},
        "rejection_telemetry_summary": {"total_retries": 1, "rejection_categories": {"JSON_PARSE_ERROR": 1}},
    }

    report, decision = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=[t1],
        arm_targets=[arm],
        stability_info=stability_info,
        all_metrics=all_metrics,
        dim_diagnostics=dim_diag,
        pass1_telemetry=pass1_telem,
        pass2_telemetry=pass2_telem,
        total_duration=1.0,
    )

    assert report["telemetry"]["total_provider_calls"] == 5
    assert report["telemetry"]["structured_output_retries"] == 3
    assert report["telemetry"]["aggregate"]["total_structured_retries"] == 3
    assert report["telemetry"]["pass1"]["structured_retries"] == 1
    assert report["telemetry"]["pass2"]["structured_retries"] == 2


# 13 & 14. Permanent failure vs provider exceptions
def test_permanent_failure_and_provider_exception_separation():
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=Path("out"),
    )

    t1 = _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED)
    arm = _make_dummy_arm_target("q1", "PRIMARY", [t1])

    pass1_telem = {
        "pass_index": 1,
        "provider_calls": 2,
        "provider_invocation_errors": 1,
        "structured_retries": 1,
        "semantic_execution_errors": 1,
        "draft_rejection_categories": {},
    }
    pass2_telem = {
        "pass_index": 2,
        "provider_calls": 2,
        "provider_invocation_errors": 0,
        "structured_retries": 1,
        "semantic_execution_errors": 1,
        "draft_rejection_categories": {},
    }

    stability_info = {
        "total_claims": 1,
        "claims_with_two_valid_semantic_labels": 0,
        "stable_semantic_claim_count": 0,
        "unstable_semantic_claim_count": 0,
        "successful_label_stability_percentage": 0.0,
        "label_stability_percentage": 0.0,
        "unstable_claim_count": 0,
        "pass1_execution_error_count": 1,
        "pass2_execution_error_count": 1,
        "execution_error_in_any_pass_count": 1,
        "repeated_execution_error_claim_count": 1,
        "unstable_claims": [],
        "execution_error_claims": [],
    }

    all_metrics = {
        "v0_claim_binary": {},
        "v1_claim_binary": {},
        "v2_d3_claim_binary": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "execution_errors": 1},
        "v1_three_way": {},
        "v2_d3_three_way": {"execution_errors": 1},
        "paired_v1_vs_v2_d3": {"net_correctness_delta": 0},
        "v0_answer_metrics": {},
        "v1_answer_metrics": {},
        "v2_d3_answer_metrics": {"answer_level_accuracy": 0.0},
        "v2_d3_vs_v1_answer_deltas": {},
    }

    dim_diag = {
        "schema_version": "1.0",
        "artifact_type": "v2_d3_dimension_diagnostics",
        "diagnostic_pass": 1,
        "total_claims": 1,
        "successfully_structured_claim_count": 0,
        "execution_error_claim_count": 1,
        "relation_distribution": {},
        "diagnostic_mismatch_flag_counts": {},
        "rejection_telemetry_summary": {"total_retries": 2, "rejection_categories": {}},
    }

    report, decision = evaluator._build_reports(
        sources_info={},
        exec_identity={},
        claim_targets=[t1],
        arm_targets=[arm],
        stability_info=stability_info,
        all_metrics=all_metrics,
        dim_diagnostics=dim_diag,
        pass1_telemetry=pass1_telem,
        pass2_telemetry=pass2_telem,
        total_duration=1.0,
    )

    assert report["verdict"] == "V2_DEVELOPMENT_EXECUTION_ERROR"
    assert report["telemetry"]["model_errors"] == 2
    assert report["telemetry"]["provider_invocation_errors"] == 1


# 15. Provider calls jsonl row count equals total calls
def test_provider_calls_jsonl_output(tmp_path: Path):
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=tmp_path / "out",
    )

    inner = MockD3ChatProvider(["res1", "res2", "res3"])
    obs = ObservationalChatModelProviderWrapper(inner)
    obs.complete(system_instruction="s", user_prompt="p1")
    obs.complete(system_instruction="s", user_prompt="p2")
    obs.complete(system_instruction="s", user_prompt="p3")

    report = {
        "candidate_id": "V2-D3",
        "verdict": "V2_DEVELOPMENT_BENCHMARK_PASS",
        "telemetry": {"total_provider_calls": 3},
    }

    evaluator._write_reports(report, provider=obs, is_preflight=False)
    p_calls_file = tmp_path / "out" / "telemetry" / "provider_calls.jsonl"
    assert p_calls_file.is_file()
    lines = [json.loads(l) for l in p_calls_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3


# 16. Dirty Git blocks canonical real execution
def test_dirty_git_blocks_canonical_real_execution(tmp_path: Path):
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=tmp_path / "out",
    )
    with patch.object(evaluator, "_is_git_worktree_clean", return_value=False):
        with pytest.raises(DataValidationError, match="Git worktree must be clean"):
            evaluator._validate_canonical_provenance()


# 17. Wrong package version blocks execution
def test_wrong_package_version_blocks_execution(tmp_path: Path):
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=tmp_path / "out",
    )
    with patch("importlib.metadata.version", return_value="0.99.0"):
        with pytest.raises(DataValidationError, match="Package version mismatch"):
            evaluator._validate_package_provenance()


# 18. Wrong candidate ID blocks execution
def test_wrong_candidate_id_blocks_execution(tmp_path: Path):
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=tmp_path / "out",
        candidate_id="INVALID-ID",
    )
    with pytest.raises(DataValidationError, match="Candidate ID mismatch"):
        evaluator._validate_canonical_provenance()


# 19. Provider version drift rejected
def test_provider_version_drift_rejected(tmp_path: Path):
    evaluator = V2D3DevelopmentBenchmarkEvaluator(
        forensic_packets_path=Path("f.zip"),
        forensic_labels_path=Path("f.json"),
        control_packets_path=Path("c.zip"),
        control_labels_path=Path("c.json"),
        v1_evidence_path=Path("v1.zip"),
        output_dir=tmp_path / "out",
    )
    mock_prov = MagicMock()
    mock_prov.provider_name = CANONICAL_V3_BACKEND
    mock_prov.model_name = CANONICAL_V3_MODEL_NAME
    mock_prov.model_revision = CANONICAL_V3_MODEL_REVISION
    mock_prov.provider_version = "4.99.9"  # drifted version

    with pytest.raises(DataValidationError, match="Provider version mismatch"):
        evaluator._validate_runtime_provider_identity(mock_prov)


# 20. D3 semantic prompt SHA unchanged
def test_d3_semantic_prompt_sha_unchanged():
    sys_sha = sha256(STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest()
    assert sys_sha == "546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b"


# 21. D3 schema SHA unchanged
def test_d3_schema_sha_unchanged():
    schema_sha = sha256(
        json.dumps(D3StructuredClaimAssessmentDraft.model_json_schema(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert schema_sha == "3591144a40b0519d5da9dd262e8edf8814531d798b69deea94fd81fae39f5f61"


# 22. Deterministic relation -> final label mappings
def test_deterministic_relation_mappings():
    d_entails = D3StructuredClaimAssessmentDraft(
        claim_id="C1", relation=D3EvidenceRelation.ENTAILS,
        actor_mismatch=False, condition_exception_mismatch=False,
        quantity_temporal_mismatch=False, negation_modality_mismatch=False, source_scope_mismatch=False,
    )
    d_contra = D3StructuredClaimAssessmentDraft(
        claim_id="C1", relation=D3EvidenceRelation.CONTRADICTS,
        actor_mismatch=True, condition_exception_mismatch=False,
        quantity_temporal_mismatch=False, negation_modality_mismatch=False, source_scope_mismatch=False,
    )
    d_does_not = D3StructuredClaimAssessmentDraft(
        claim_id="C1", relation=D3EvidenceRelation.DOES_NOT_ESTABLISH,
        actor_mismatch=False, condition_exception_mismatch=False,
        quantity_temporal_mismatch=False, negation_modality_mismatch=False, source_scope_mismatch=False,
    )

    assert derive_claim_semantic_label_d3(d_entails) == SemanticSupportLabel.SUPPORTED
    assert derive_claim_semantic_label_d3(d_contra) == SemanticSupportLabel.CONTRADICTED
    assert derive_claim_semantic_label_d3(d_does_not) == SemanticSupportLabel.INSUFFICIENT
