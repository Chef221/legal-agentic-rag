"""Unit tests for controlled V0 vs V1 verification benchmark harness."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from legal_agentic_rag.configuration.online import GenerationConfig
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation.transformers_provider import TransformersChatProvider
from scripts.evaluate_verification_semantic_benchmark import (
    CANONICAL_CONTROL_LABELS_SHA256,
    CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
    CANONICAL_FORENSIC_LABELS_SHA256,
    CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
    CANONICAL_REPEAT_COUNT,
    CANONICAL_V1_BACKEND,
    CANONICAL_V1_DEVICE,
    CANONICAL_V1_MAX_INPUT_TOKENS,
    CANONICAL_V1_MAX_OUTPUT_TOKENS,
    CANONICAL_V1_MAX_STRUCTURED_RETRIES,
    CANONICAL_V1_MODEL_NAME,
    CANONICAL_V1_MODEL_REVISION,
    CANONICAL_V1_TIMEOUT_SECONDS,
    CANONICAL_V1_TORCH_DTYPE,
    BinaryPrediction,
    HumanEntailment,
    ObservationalChatModelProviderWrapper,
    SemanticVerifierBenchmarkRunner,
    sha256_file,
    sha256_text,
)


class FakeChatModelProvider(ChatModelProvider):
    """Deterministic fake ChatModelProvider for unit testing verifier evaluations."""

    def __init__(
        self,
        *,
        default_label: str = "supported",
        labels_by_claim_id: dict[str, str] | None = None,
        fail_on_attempt: int = 0,
        unstable_on_pass: int = 0,
    ) -> None:
        self._default_label = default_label
        self._labels_by_claim_id = labels_by_claim_id or {}
        self._fail_on_attempt = fail_on_attempt
        self._unstable_on_pass = unstable_on_pass
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake_test_provider"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def model_name(self) -> str:
        return CANONICAL_V1_MODEL_NAME

    @property
    def model_revision(self) -> str:
        return CANONICAL_V1_MODEL_REVISION

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        self._call_count += 1
        if self._fail_on_attempt and self._call_count == self._fail_on_attempt:
            return "INVALID JSON NOT MATCHING SCHEMA"

        # Parse user_prompt to find claim IDs
        claim_ids = []
        if "CLAIMS_AND_CITED_EVIDENCE_JSON:\n" in user_prompt:
            parts = user_prompt.split("CLAIMS_AND_CITED_EVIDENCE_JSON:\n")[1].split("\n\nOUTPUT_JSON_SCHEMA:")[0]
            try:
                claims_data = json.loads(parts)
                claim_ids = [c["claim_id"] for c in claims_data]
            except Exception:
                claim_ids = ["C1"]
        else:
            claim_ids = ["C1"]

        assessments = []
        for cid in claim_ids:
            lbl = self._labels_by_claim_id.get(cid, self._default_label)
            assessments.append({"claim_id": cid, "label": lbl})

        return json.dumps({"assessments": assessments})


def _create_mock_benchmark_environment(tmp_path: Path) -> dict[str, Path]:
    """Create complete mock benchmark environment (4 sources with 38 composite claims)."""
    # 1. Slice A: Forensic (4 questions: 102047, 147239, 26541, 95861 -> 11 claims)
    # Counts: 2 SUPPORTED, 5 CONTRADICTED, 4 INSUFFICIENT
    forensic_bundle = tmp_path / "forensic_bundle"
    f_pkts_dir = forensic_bundle / "forensic_packets"
    f_pkts_dir.mkdir(parents=True)

    forensic_spec = {
        "102047": {
            "BASE": [("C1", "CONTRADICTED", ["CONDITION_INVERTED"])],
            "CANDIDATE": [("C1", "CONTRADICTED", ["CONDITION_OMITTED"])],
        },
        "147239": {
            "CANDIDATE": [("C1", "SUPPORTED", ["NONE"]), ("C2", "CONTRADICTED", ["ACTOR_ROLE_INVERTED"])],
        },
        "26541": {
            "BASE": [("C1", "INSUFFICIENT", ["WRONG_DOCUMENT"])],
        },
        "95861": {
            "BASE": [("C1", "CONTRADICTED", ["WRONG_DOCUMENT"]), ("C2", "INSUFFICIENT", ["WRONG_DOCUMENT"]), ("C3", "INSUFFICIENT", ["WRONG_DOCUMENT"])],
            "CANDIDATE": [("C1", "CONTRADICTED", ["WRONG_DOCUMENT"]), ("C2", "INSUFFICIENT", ["ACTOR_ROLE_INVERTED"]), ("C3", "SUPPORTED", ["NONE"])],
        },
    }

    mock_evidence_text = "Cơ quan có thẩm quyền tiếp nhận và giải quyết hồ sơ theo đúng quy định pháp luật."

    f_labels_data: dict[str, Any] = {"questions": {}}
    for qid, arms in forensic_spec.items():
        pkt = {
            "schema_version": "1.0",
            "question_id": qid,
            "question": f"Question text for forensic {qid}",
            "arms": {},
        }
        f_labels_data["questions"][qid] = {"question_id": qid, "arms": {}}

        for arm_id, claims_def in arms.items():
            claims_list = []
            claims_labels = {}
            for cid, h_label, error_tags in claims_def:
                ctext = "Cơ quan có thẩm quyền giải quyết theo quy định pháp luật ."
                claims_list.append({
                    "claim_id": cid,
                    "claim_text": ctext,
                    "evidence_ids": ["E1"],
                    "status": "supported",
                    "lexical_support_score": 1.0,
                    "numeric_match": True,
                    "negation_match": True,
                    "errors": [],
                })
                claims_labels[cid] = {
                    "claim_id": cid,
                    "claim_text_sha256": sha256_text(ctext),
                    "claim_text": ctext,
                    "entailment_label": h_label,
                    "error_tags": error_tags,
                }

            pkt["arms"][arm_id] = {
                "historical_response": {
                    "question": f"Question text for forensic {qid}",
                    "answer": " ".join(f"Cơ quan có thẩm quyền giải quyết theo quy định pháp luật [E1]." for _ in claims_list),
                    "citations": [{
                        "evidence_id": "E1", "chunk_id": "c1", "document_id": "d1",
                        "document_title": "Doc", "document_number": "1", "article_number": "1",
                        "source_url": "url",
                    }],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": "t",
                    "warnings": [],
                },
                "agent_outcome": {"stop_reason": "answer_verified", "attempt": 1},
                "selected_evidence": [{
                    "evidence_id": "E1", "chunk_id": "c1", "document_id": "d1",
                    "text": mock_evidence_text, "document_title": "Doc",
                    "document_number": "1", "article_number": "1", "source_url": "url",
                    "document_type": "Luat", "effective_date": "2023-01-01", "expiry_date": None,
                    "effect_status": "Con hieu luc", "metadata": {},
                }],
                "historical_verification": {
                    "is_valid": True,
                    "valid_citations": [{"evidence_id": "E1"}],
                    "invalid_citations": [],
                    "claim_coverage_score": 1.0,
                    "claim_verifications": claims_list,
                    "errors": [],
                    "warnings": ["semantic_entailment_not_verified"],
                },
            }
            f_labels_data["questions"][qid]["arms"][arm_id] = {
                "historical_stop_reason": "answer_verified",
                "claim_review_applicable": True,
                "claims": claims_labels,
            }

        (f_pkts_dir / f"{qid}.json").write_text(json.dumps(pkt, indent=2), encoding="utf-8")

    f_pkts_zip = tmp_path / "forensic_packets.zip"
    with zipfile.ZipFile(f_pkts_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(forensic_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(forensic_bundle).as_posix())

    f_labels_json = tmp_path / "forensic_labels.json"
    f_labels_json.write_text(json.dumps(f_labels_data, indent=2), encoding="utf-8")

    # 2. Slice B: Controls (16 questions -> 27 claims)
    # Counts: 16 SUPPORTED, 2 CONTRADICTED, 9 INSUFFICIENT
    control_bundle = tmp_path / "control_bundle"
    c_pkts_dir = control_bundle / "positive_control_packets"
    c_pkts_dir.mkdir(parents=True)

    # 16 primary questions matching B-FORENSIC-1C
    control_spec = {
        "75171": [("C1", "INSUFFICIENT", ["SCOPE_OVERGENERALIZED"])],
        "150131": [("C1", "CONTRADICTED", ["ACTOR_ROLE_INVERTED"])],
        "30405": [("C1", "INSUFFICIENT", ["SCOPE_OVERGENERALIZED"])],
        "36801": [("C1", "SUPPORTED", ["NONE"])],
        "116877": [("C1", "SUPPORTED", ["NONE"]), ("C2", "SUPPORTED", ["NONE"]), ("C3", "SUPPORTED", ["NONE"])],
        "15181": [("C1", "SUPPORTED", ["NONE"]), ("C2", "SUPPORTED", ["NONE"]), ("C3", "SUPPORTED", ["NONE"])],
        "5967": [("C1", "INSUFFICIENT", ["SCOPE_OVERGENERALIZED"]), ("C2", "INSUFFICIENT", ["SCOPE_OVERGENERALIZED"]), ("C3", "INSUFFICIENT", ["SCOPE_OVERGENERALIZED"])],
        "139413": [("C1", "SUPPORTED", ["NONE"]), ("C2", "SUPPORTED", ["NONE"]), ("C3", "SUPPORTED", ["NONE"])],
        "34351": [("C1", "SUPPORTED", ["NONE"]), ("C2", "INSUFFICIENT", ["OTHER"])],
        "31883": [("C1", "SUPPORTED", ["NONE"])],
        "40489": [("C1", "INSUFFICIENT", ["OTHER"])],
        "155139": [("C1", "INSUFFICIENT", ["QUANTITY_ERROR"])],
        "108497": [("C1", "INSUFFICIENT", ["SCOPE_OVERGENERALIZED"])],
        "4031": [("C1", "SUPPORTED", ["NONE"])],
        "103983": [("C1", "SUPPORTED", ["NONE"]), ("C2", "SUPPORTED", ["NONE"]), ("C3", "CONTRADICTED", ["CONDITION_OMITTED"])],
        "140693": [("C1", "SUPPORTED", ["NONE"])],
    }

    c_labels_data: dict[str, Any] = {"questions": {}}
    for qid, claims_def in control_spec.items():
        claims_list = []
        claims_labels = {}
        for cid, h_label, error_tags in claims_def:
            ctext = "Cơ quan có thẩm quyền giải quyết theo quy định pháp luật ."
            claims_list.append({
                "claim_id": cid,
                "claim_text": ctext,
                "evidence_ids": ["E1"],
                "status": "supported",
                "lexical_support_score": 1.0,
                "numeric_match": True,
                "negation_match": True,
                "errors": [],
            })
            claims_labels[cid] = {
                "claim_id": cid,
                "claim_text_sha256": sha256_text(ctext),
                "claim_text": ctext,
                "entailment_label": h_label,
                "error_tags": error_tags,
            }

        pkt = {
            "schema_version": "1.0",
            "question_id": qid,
            "question": f"Question text for control {qid}",
            "control_metadata": {
                "stratum": "A_SINGLE_CLAIM_CLEAN",
                "selection_key": sha256_text(f"verification-positive-control-v1|{qid}"),
                "pool_type": "primary",
            },
            "historical_arm": {
                "historical_response": {
                    "question": f"Question text for control {qid}",
                    "answer": " ".join(f"Cơ quan có thẩm quyền giải quyết theo quy định pháp luật [E1]." for _ in claims_list),
                    "citations": [{
                        "evidence_id": "E1", "chunk_id": "c1", "document_id": "d1",
                        "document_title": "Doc", "document_number": "1", "article_number": "1",
                        "source_url": "url",
                    }],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": "t",
                    "warnings": [],
                },
                "agent_outcome": {"stop_reason": "answer_verified", "attempt": 1},
                "selected_evidence": [{
                    "evidence_id": "E1", "chunk_id": "c1", "document_id": "d1",
                    "text": mock_evidence_text, "document_title": "Doc",
                    "document_number": "1", "article_number": "1", "source_url": "url",
                    "document_type": "Luat", "effective_date": "2023-01-01", "expiry_date": None,
                    "effect_status": "Con hieu luc", "metadata": {},
                }],
                "historical_verification": {
                    "is_valid": True,
                    "valid_citations": [{"evidence_id": "E1"}],
                    "invalid_citations": [],
                    "claim_coverage_score": 1.0,
                    "claim_verifications": claims_list,
                    "errors": [],
                    "warnings": ["semantic_entailment_not_verified"],
                },
            },
        }
        c_labels_data["questions"][qid] = {
            "question_id": qid,
            "stratum": "A_SINGLE_CLAIM_CLEAN",
            "selection_key": sha256_text(f"verification-positive-control-v1|{qid}"),
            "pool_type": "primary",
            "historical_stop_reason": "answer_verified",
            "claim_review_applicable": True,
            "claims": claims_labels,
        }
        (c_pkts_dir / f"{qid}.json").write_text(json.dumps(pkt, indent=2), encoding="utf-8")

    c_pkts_zip = tmp_path / "control_packets.zip"
    with zipfile.ZipFile(c_pkts_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(control_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(control_bundle).as_posix())

    c_labels_json = tmp_path / "control_labels.json"
    c_labels_json.write_text(json.dumps(c_labels_data, indent=2), encoding="utf-8")

    return {
        "forensic_packets": f_pkts_zip,
        "forensic_labels": f_labels_json,
        "control_packets": c_pkts_zip,
        "control_labels": c_labels_json,
        "output_dir": tmp_path / "out",
        "package_zip": tmp_path / "out.zip",
    }


def _run_runner_with_patched_env(
    env: dict[str, Path],
    provider: ChatModelProvider | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    f_p_sha = sha256_file(env["forensic_packets"])
    f_l_sha = sha256_file(env["forensic_labels"])
    c_p_sha = sha256_file(env["control_packets"])
    c_l_sha = sha256_file(env["control_labels"])

    with (
        patch("scripts.evaluate_verification_semantic_benchmark.CANONICAL_FORENSIC_REVIEW_ZIP_SHA256", f_p_sha),
        patch("scripts.evaluate_verification_semantic_benchmark.CANONICAL_FORENSIC_LABELS_SHA256", f_l_sha),
        patch("scripts.evaluate_verification_semantic_benchmark.CANONICAL_CONTROL_REVIEW_ZIP_SHA256", c_p_sha),
        patch("scripts.evaluate_verification_semantic_benchmark.CANONICAL_CONTROL_LABELS_SHA256", c_l_sha),
    ):
        runner = SemanticVerifierBenchmarkRunner(
            forensic_packets_path=env["forensic_packets"],
            forensic_labels_path=env["forensic_labels"],
            control_packets_path=env["control_packets"],
            control_labels_path=env["control_labels"],
            output_dir=env["output_dir"],
            package_zip_path=env.get("package_zip"),
            custom_provider=provider or FakeChatModelProvider(),
            **kwargs,
        )
        return runner.run()


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------


def test_01_real_transformers_chat_provider_signature_regression(tmp_path: Path) -> None:
    """Regression test: Verify _init_v1_provider constructs TransformersChatProvider with real GenerationConfig."""
    env = _create_mock_benchmark_environment(tmp_path)
    runner = SemanticVerifierBenchmarkRunner(
        forensic_packets_path=env["forensic_packets"],
        forensic_labels_path=env["forensic_labels"],
        control_packets_path=env["control_packets"],
        control_labels_path=env["control_labels"],
        output_dir=env["output_dir"],
        custom_provider=None,  # Exercise real provider init path
    )

    with patch("scripts.evaluate_verification_semantic_benchmark.TransformersChatProvider", autospec=True) as mock_class:
        provider = runner._init_v1_provider()

        # Assert exactly one positional argument passed to constructor
        assert mock_class.call_count == 1
        args, kwargs = mock_class.call_args
        assert len(args) == 1
        config = args[0]

        assert isinstance(config, GenerationConfig)
        assert config.backend == "transformers"
        assert config.model_name == CANONICAL_V1_MODEL_NAME
        assert config.model_revision == CANONICAL_V1_MODEL_REVISION
        assert config.device == "cuda"
        assert config.torch_dtype == "float16"
        assert config.temperature == 0.0
        assert config.max_input_tokens == 8192
        assert config.max_output_tokens == 512
        assert config.timeout_seconds == 180.0
        assert config.local_files_only is False


def test_02_canonical_config_drift_rejected(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)

    # Model name drift
    with pytest.raises(DataValidationError, match="INVALID_VERIFIER_BENCHMARK_CONFIG: model_name must be"):
        runner = SemanticVerifierBenchmarkRunner(
            forensic_packets_path=env["forensic_packets"],
            forensic_labels_path=env["forensic_labels"],
            control_packets_path=env["control_packets"],
            control_labels_path=env["control_labels"],
            output_dir=env["output_dir"],
            model_name="unauthorized-model",
            custom_provider=None,
        )
        runner._validate_canonical_config()

    # Repeat count != 2 drift
    with pytest.raises(DataValidationError, match="INVALID_VERIFIER_BENCHMARK_CONFIG: repeat_count must be 2"):
        runner = SemanticVerifierBenchmarkRunner(
            forensic_packets_path=env["forensic_packets"],
            forensic_labels_path=env["forensic_labels"],
            control_packets_path=env["control_packets"],
            control_labels_path=env["control_labels"],
            output_dir=env["output_dir"],
            repeat_count=1,
            custom_provider=None,
        )
        runner._validate_canonical_config()


def test_03_directory_packet_sources_rejected(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    # Pass directory instead of ZIP
    dir_path = tmp_path / "forensic_bundle"
    runner = SemanticVerifierBenchmarkRunner(
        forensic_packets_path=dir_path,
        forensic_labels_path=env["forensic_labels"],
        control_packets_path=env["control_packets"],
        control_labels_path=env["control_labels"],
        output_dir=env["output_dir"],
    )
    with pytest.raises(DataValidationError, match="INVALID_VERIFIER_BENCHMARK_PROVENANCE: forensic review source must be a ZIP file"):
        runner._validate_sources()


def test_04_exact_v0_replay_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)

    # Corrupt historical verification claim status
    pkt_data = json.loads((tmp_path / "control_bundle" / "positive_control_packets" / "75171.json").read_text(encoding="utf-8"))
    pkt_data["historical_arm"]["historical_verification"]["claim_verifications"][0]["status"] = "unsupported"
    (tmp_path / "control_bundle" / "positive_control_packets" / "75171.json").write_text(json.dumps(pkt_data), encoding="utf-8")

    # Re-zip
    with zipfile.ZipFile(env["control_packets"], "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_path / "control_bundle"):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(tmp_path / "control_bundle").as_posix())

    with pytest.raises(DataValidationError, match="INVALID_VERIFIER_BENCHMARK_PROVENANCE: V0 replay status mismatch on 75171:PRIMARY"):
        _run_runner_with_patched_env(env)


def test_05_exact_composite_benchmark_counts(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    report = _run_runner_with_patched_env(env)

    summary = report["summary_metrics"]
    assert summary["total_claims"] == 38
    assert summary["supported_human_claims"] == 18
    assert summary["negative_human_claims"] == 20


def test_06_claim_text_sha_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    data = json.loads(env["control_labels"].read_text())
    data["questions"]["75171"]["claims"]["C1"]["claim_text_sha256"] = "0" * 64
    env["control_labels"].write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DataValidationError, match="INVALID_VERIFIER_BENCHMARK_PROVENANCE: Claim text SHA mismatch for 75171"):
        _run_runner_with_patched_env(env)


def test_07_skip_model_run_preflight_reports_v0_baseline(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    report = _run_runner_with_patched_env(env, skip_model_run=True)
    assert report["verdict"] == "VERIFIER_BENCHMARK_READY"
    assert report["semantic_verifier_promotion_authorized"] is False
    assert report["model_run_executed"] is False
    assert "v0_baseline_metrics" in report
    assert report["v0_baseline_metrics"]["v0_claim_binary"]["total_evaluated_claims"] == 38


def test_08_deterministic_two_pass_stability_pass(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    provider = FakeChatModelProvider(default_label="supported")
    report = _run_runner_with_patched_env(env, provider=provider, repeat_count=2)

    assert report["verdict"] == "VERIFIER_BENCHMARK_PASS"
    assert report["stability"]["stable"] is True
    assert report["stability"]["unstable_claim_count"] == 0
    assert report["stability"]["total_evaluated_claims"] == 38


def test_09_semantic_verifier_promotion_authorized_is_strictly_false(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    report = _run_runner_with_patched_env(env)
    assert report["semantic_verifier_promotion_authorized"] is False

    dec_file = env["output_dir"] / "results" / "verifier_benchmark_decision_report.json"
    dec_data = json.loads(dec_file.read_text(encoding="utf-8"))
    assert dec_data["semantic_verifier_promotion_authorized"] is False


def test_10_paired_deltas_correctness(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    provider = FakeChatModelProvider(default_label="supported")
    report = _run_runner_with_patched_env(env, provider=provider)

    paired = report["metrics"]["paired_deltas"]
    assert paired["both_correct"] == 18
    assert paired["both_wrong"] == 20
    assert paired["v0_only_correct"] == 0
    assert paired["v1_only_correct"] == 0
    assert paired["net_correctness_delta"] == 0


def test_11_error_tag_diagnostics_calculation(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    provider = FakeChatModelProvider(default_label="contradicted")
    report = _run_runner_with_patched_env(env, provider=provider)

    diag = report["metrics"]["error_tag_diagnostics"]
    assert "SCOPE_OVERGENERALIZED" in diag
    assert diag["SCOPE_OVERGENERALIZED"]["v1_catch_rate"] == 1.0


def test_12_answer_level_metrics(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    report = _run_runner_with_patched_env(env)

    ans_m = report["metrics"]["answer_level_metrics"]
    assert ans_m["total_benchmark_arms"] == 22  # 6 forensic arms + 16 control arms
    assert ans_m["v0_answer_metrics"]["total_evaluated"] == 22
    assert ans_m["v1_answer_metrics"]["total_evaluated"] == 22


def test_13_no_semantic_prompt_reimplementation(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    provider = FakeChatModelProvider()
    report = _run_runner_with_patched_env(env, provider=provider)

    assert provider._call_count >= 22
    assert report["verdict"] == "VERIFIER_BENCHMARK_PASS"


def test_14_no_retrieval_or_generation_invoked(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    with (
        patch("legal_agentic_rag.retrieval.dense.DenseRetriever") as mock_dense,
        patch("legal_agentic_rag.retrieval.fixed.FixedRetriever") as mock_fixed,
        patch("legal_agentic_rag.generation.model_generator.ModelBackedAnswerGenerator") as mock_gen,
    ):
        _run_runner_with_patched_env(env)
        mock_dense.assert_not_called()
        mock_fixed.assert_not_called()
        mock_gen.assert_not_called()


def test_15_no_absolute_windows_paths_in_output(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    _run_runner_with_patched_env(env)

    report_text = (env["output_dir"] / "results" / "verifier_benchmark_report.json").read_text(encoding="utf-8")
    assert "C:\\" not in report_text
    assert "c:\\" not in report_text
    assert "Users" not in report_text


class UnstableFakeChatModelProvider(FakeChatModelProvider):
    """Fake provider that returns 'supported' on pass 1 calls and 'contradicted' on pass 2 calls."""

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        self._call_count += 1
        # 22 calls per pass
        lbl = "supported" if self._call_count <= 22 else "contradicted"
        claim_ids = ["C1"]
        if "CLAIMS_AND_CITED_EVIDENCE_JSON:\n" in user_prompt:
            parts = user_prompt.split("CLAIMS_AND_CITED_EVIDENCE_JSON:\n")[1].split("\n\nOUTPUT_JSON_SCHEMA:")[0]
            try:
                claims_data = json.loads(parts)
                claim_ids = [c["claim_id"] for c in claims_data]
            except Exception:
                pass
        assessments = [{"claim_id": cid, "label": lbl} for cid in claim_ids]
        return json.dumps({"assessments": assessments})


def test_16_two_pass_instability_detected(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    provider = UnstableFakeChatModelProvider()
    report = _run_runner_with_patched_env(env, provider=provider, repeat_count=2)

    assert report["verdict"] == "SEMANTIC_VERIFIER_LABEL_INSTABILITY"
    assert report["stability"]["stable"] is False
    assert report["stability"]["unstable_claim_count"] == 38


class BrokenFakeChatModelProvider(FakeChatModelProvider):
    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        raise ModelError("Permanent model failure on GPU")


def test_17_permanent_model_failure_does_not_count_as_tn(tmp_path: Path) -> None:
    """Regression test: Execution errors on human negatives must NOT increase TN."""
    env = _create_mock_benchmark_environment(tmp_path)
    provider = BrokenFakeChatModelProvider()
    report = _run_runner_with_patched_env(env, provider=provider, repeat_count=2)

    assert report["verdict"] == "SEMANTIC_VERIFIER_EXECUTION_ERROR"
    assert report["execution_metadata"]["model_error_count"] > 0

    v1_binary = report["metrics"]["v1_binary_metrics"]
    assert v1_binary["tn"] == 0
    assert v1_binary["tp"] == 0
    assert v1_binary["execution_error_count"] == 38
    assert v1_binary["total_evaluated_claims"] == 0


def test_18_slice_and_stratum_metrics(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    report = _run_runner_with_patched_env(env)

    slices = report["metrics"]["slice_and_strata_metrics"]
    assert "slice:suspicious_forensic" in slices
    assert slices["slice:suspicious_forensic"]["claim_count"] == 11
    assert "slice:positive_control" in slices
    assert slices["slice:positive_control"]["claim_count"] == 27
    assert "stratum:A_SINGLE_CLAIM_CLEAN" in slices


def test_19_reference_answers_never_supplied_to_verifier(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    prompts_seen: list[str] = []

    class PromptSpyProvider(FakeChatModelProvider):
        def complete(self, *, system_instruction: str, user_prompt: str) -> str:
            prompts_seen.append(user_prompt)
            return super().complete(system_instruction=system_instruction, user_prompt=user_prompt)

    _run_runner_with_patched_env(env, provider=PromptSpyProvider())

    for prompt in prompts_seen:
        assert "reference_answer" not in prompt.lower()
        assert "ground_truth_answer" not in prompt.lower()
        assert "gold_answer" not in prompt.lower()


def test_20_observational_wrapper_call_history(tmp_path: Path) -> None:
    env = _create_mock_benchmark_environment(tmp_path)
    report = _run_runner_with_patched_env(env)

    calls_file = env["output_dir"] / "execution" / "observational_provider_calls.jsonl"
    assert calls_file.is_file()
    lines = calls_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 44  # 22 arms * 2 passes
