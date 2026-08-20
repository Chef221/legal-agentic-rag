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
from legal_agentic_rag.generation.structured_semantic_verifier import (
    EvidenceCoverageStatus,
    SemanticDimensionStatus,
    StructuredClaimAssessmentDraft,
    StructuredSemanticCitationVerifier,
    derive_claim_semantic_label,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
    SemanticSupportLabel,
)
from scripts.evaluate_verification_v2_development import (
    CANONICAL_CONTROL_LABELS_SHA256,
    CANONICAL_CONTROL_REVIEW_ZIP_SHA256,
    CANONICAL_FORENSIC_LABELS_SHA256,
    CANONICAL_FORENSIC_REVIEW_ZIP_SHA256,
    CANONICAL_V1_EVIDENCE_ZIP_SHA256,
    BenchmarkArmTarget,
    BenchmarkClaimTarget,
    BinaryPrediction,
    HumanEntailment,
    V2DevelopmentBenchmarkEvaluator,
    main,
    sha256_file,
    sha256_text,
)


def _compute_sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        return sha256(data.encode("utf-8")).hexdigest()
    return sha256(data).hexdigest()


class MockV2ChatProvider(ChatModelProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "transformers"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def model_name(self) -> str:
        return "Qwen/Qwen2.5-3B-Instruct"

    @property
    def model_revision(self) -> str:
        return "a1d308dfcc03e09da285d49d912439a655a571e8"

    def complete(self, *, system_instruction: str, user_prompt: str) -> str:
        res = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
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
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),   # TP
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.SUPPORTED),   # FN
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.CONTRADICTED), # TN
        _make_dummy_claim_target("q4", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT), # FP
    ]
    preds = [
        {"v2_binary_prediction": "ACCEPT"},  # TP
        {"v2_binary_prediction": "REJECT"},  # FN
        {"v2_binary_prediction": "REJECT"},  # TN
        {"v2_binary_prediction": "ACCEPT"},  # FP
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


def test_paired_metrics_calculation():
    """Verify paired comparison deltas between V1 and V2."""
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.CONTRADICTED),
        _make_dummy_claim_target("q3", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT),
        _make_dummy_claim_target("q4", "PRIMARY", "C1", HumanEntailment.SUPPORTED),
    ]
    v1_preds = [
        {"is_correct": True},   # both correct
        {"is_correct": False},  # V2 fix
        {"is_correct": True},   # V2 regression
        {"is_correct": False},  # both wrong
    ]
    v2_preds = [
        {"is_correct": True},   # both correct
        {"is_correct": True},   # V2 fix
        {"is_correct": False},  # V2 regression
        {"is_correct": False},  # both wrong
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
    assert p["v1_only_correct"] == 1
    assert p["both_wrong"] == 1
    assert p["net_correctness_delta"] == 0
    assert p["v2_fixes_count"] == 1
    assert p["v2_regressions_count"] == 1


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
        {"question_id": "q2", "arm_id": "PRIMARY", "claim_id": "C1", "v2_three_way_prediction": "INSUFFICIENT"},  # unstable
    ]

    stab = evaluator._evaluate_stability([pass1, pass2])
    assert stab["unstable_claim_count"] == 1
    assert stab["label_stability_percentage"] == 50.0
    assert len(stab["unstable_claims"]) == 1
    assert stab["unstable_claims"][0]["question_id"] == "q2"


def test_dimension_diagnostics():
    """Verify aggregation of structured dimension statuses and error tag activations."""
    targets = [
        _make_dummy_claim_target("q1", "PRIMARY", "C1", HumanEntailment.CONTRADICTED, tags=["CONDITION_INVERTED"]),
        _make_dummy_claim_target("q2", "PRIMARY", "C1", HumanEntailment.INSUFFICIENT, tags=["SCOPE_OVERGENERALIZED"]),
    ]
    v2_preds = [
        {
            "structured_assessment": {
                "actor_role": "MATCH",
                "action_object": "MATCH",
                "condition_exception": "CONFLICT",
                "quantity_temporal": "MATCH",
                "negation_modality": "MATCH",
                "source_article_scope": "MATCH",
                "evidence_coverage": "COMPLETE",
            }
        },
        {
            "structured_assessment": {
                "actor_role": "MATCH",
                "action_object": "MATCH",
                "condition_exception": "MATCH",
                "quantity_temporal": "MATCH",
                "negation_modality": "MATCH",
                "source_article_scope": "INSUFFICIENT",
                "evidence_coverage": "PARTIAL",
            }
        },
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
    assert diag["dimension_status_counts"]["condition_exception"]["CONFLICT"] == 1
    assert diag["dimension_status_counts"]["source_article_scope"]["INSUFFICIENT"] == 1
    assert diag["evidence_coverage_counts"]["PARTIAL"] == 1
    assert diag["error_tag_activations"]["CONDITION_INVERTED"]["condition_exception:CONFLICT"] == 1
    assert diag["error_tag_activations"]["SCOPE_OVERGENERALIZED"]["source_article_scope:INSUFFICIENT"] == 1


def test_holdout_access_regression_no_cli_args():
    """Verify that CLI parser does NOT accept any holdout arguments or Phase-A paths."""
    from scripts import evaluate_verification_v2_development

    parser = argparse.ArgumentParser()
    # Read script content to verify no holdout strings in CLI flags
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

