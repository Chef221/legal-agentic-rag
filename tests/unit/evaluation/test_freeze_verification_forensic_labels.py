"""Comprehensive unit tests for freezing human-approved verification forensic labels (B-FORENSIC-1A)."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch
import zipfile

import pytest

from legal_agentic_rag.exceptions import DataValidationError
from scripts.freeze_verification_forensic_labels import (
    APPROVED_HUMAN_SPEC,
    CANONICAL_REVIEW_ZIP_SHA256,
    CANONICAL_TARGET_IDS,
    ApprovedClaimSpec,
    ClaimEntailmentLabel,
    ForensicErrorTag,
    ForensicLabelFreezer,
    sha256_file,
)


def _create_mock_review_zip(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Create a mock verification-forensic-review-packets.zip for testing."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True)
    (staging_dir / "execution").mkdir()
    (staging_dir / "results").mkdir()
    (staging_dir / "forensic_packets").mkdir()

    (staging_dir / "execution" / "forensic_source_identity.json").write_text("{}", encoding="utf-8")
    (staging_dir / "results" / "forensic_source_report.json").write_text("{}", encoding="utf-8")

    packets: dict[str, Any] = {}

    # Case 102047: BASE (1 claim), CANDIDATE (1 claim)
    packets["102047"] = {
        "question_id": "102047",
        "arms": {
            "BASE": {
                "agent_outcome": {"stop_reason": "answer_verified"},
                "rule_verifier_replay": {
                    "replay_result": {
                        "claim_verifications": [
                            {"claim_id": "C1", "claim_text": "Theo quy dinh phap luat 102047 BASE."}
                        ]
                    }
                },
            },
            "CANDIDATE": {
                "agent_outcome": {"stop_reason": "answer_verified"},
                "rule_verifier_replay": {
                    "replay_result": {
                        "claim_verifications": [
                            {"claim_id": "C1", "claim_text": "Theo quy dinh phap luat 102047 CAND."}
                        ]
                    }
                },
            },
        },
    }

    # Case 147239: BASE (gen_failed), CANDIDATE (2 claims)
    packets["147239"] = {
        "question_id": "147239",
        "arms": {
            "BASE": {
                "agent_outcome": {"stop_reason": "generation_failed"},
                "rule_verifier_replay": {"replay_applicable": False},
            },
            "CANDIDATE": {
                "agent_outcome": {"stop_reason": "answer_verified"},
                "rule_verifier_replay": {
                    "replay_result": {
                        "claim_verifications": [
                            {"claim_id": "C1", "claim_text": "Theo quy dinh phap luat 147239 C1."},
                            {"claim_id": "C2", "claim_text": "Theo quy dinh phap luat 147239 C2."},
                        ]
                    }
                },
            },
        },
    }

    # Case 26541: BASE (1 claim), CANDIDATE (gen_failed)
    packets["26541"] = {
        "question_id": "26541",
        "arms": {
            "BASE": {
                "agent_outcome": {"stop_reason": "answer_verified"},
                "rule_verifier_replay": {
                    "replay_result": {
                        "claim_verifications": [
                            {"claim_id": "C1", "claim_text": "Theo quy dinh phap luat 26541 BASE."}
                        ]
                    }
                },
            },
            "CANDIDATE": {
                "agent_outcome": {"stop_reason": "generation_failed"},
                "rule_verifier_replay": {"replay_applicable": False},
            },
        },
    }

    # Case 95861: BASE (3 claims), CANDIDATE (3 claims)
    packets["95861"] = {
        "question_id": "95861",
        "arms": {
            "BASE": {
                "agent_outcome": {"stop_reason": "answer_verified"},
                "rule_verifier_replay": {
                    "replay_result": {
                        "claim_verifications": [
                            {"claim_id": "C1", "claim_text": "Theo quy dinh phap luat 95861 BASE C1."},
                            {"claim_id": "C2", "claim_text": "Theo quy dinh phap luat 95861 BASE C2."},
                            {"claim_id": "C3", "claim_text": "Theo quy dinh phap luat 95861 BASE C3."},
                        ]
                    }
                },
            },
            "CANDIDATE": {
                "agent_outcome": {"stop_reason": "answer_verified"},
                "rule_verifier_replay": {
                    "replay_result": {
                        "claim_verifications": [
                            {"claim_id": "C1", "claim_text": "Theo quy dinh phap luat 95861 CAND C1."},
                            {"claim_id": "C2", "claim_text": "Theo quy dinh phap luat 95861 CAND C2."},
                            {"claim_id": "C3", "claim_text": "Theo quy dinh phap luat 95861 CAND C3."},
                        ]
                    }
                },
            },
        },
    }

    for qid, pkt in packets.items():
        (staging_dir / "forensic_packets" / f"{qid}.json").write_text(
            json.dumps(pkt, indent=2), encoding="utf-8"
        )

    zip_path = tmp_path / "verification-forensic-review-packets.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(staging_dir):
            for file in files:
                fp = Path(root) / file
                arcname = fp.relative_to(staging_dir).as_posix()
                z.write(fp, arcname=arcname)

    return zip_path, packets


def _run_with_patched_zip(zip_path: Path, output_path: Path, **kwargs: Any) -> dict[str, Any]:
    """Helper to run label freezer with patched canonical zip hash."""
    actual_sha = sha256_file(zip_path)
    with patch("scripts.freeze_verification_forensic_labels.CANONICAL_REVIEW_ZIP_SHA256", actual_sha):
        freezer = ForensicLabelFreezer(
            review_packets_path=zip_path,
            output_path=output_path,
            **kwargs,
        )
        return freezer.run()


# ----------------------------------------------------------------------
# 20 REQUIRED TESTS
# ----------------------------------------------------------------------


def test_01_wrong_review_zip_sha_rejected(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "out.json"
    freezer = ForensicLabelFreezer(review_packets_path=zip_path, output_path=out_path)
    with pytest.raises(DataValidationError, match="Review packets ZIP SHA mismatch"):
        freezer.run()


def test_02_missing_forensic_packet_rejected(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging_bad"
    staging_dir.mkdir()
    (staging_dir / "forensic_packets").mkdir()
    (staging_dir / "forensic_packets" / "102047.json").write_text('{"question_id": "102047"}')

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.write(staging_dir / "forensic_packets" / "102047.json", arcname="forensic_packets/102047.json")

    out_path = tmp_path / "out.json"
    with (
        patch("scripts.freeze_verification_forensic_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        pytest.raises(DataValidationError, match="Missing forensic packet 'forensic_packets/147239.json'"),
    ):
        freezer = ForensicLabelFreezer(review_packets_path=bad_zip, output_path=out_path)
        freezer.run()


def test_03_wrong_question_id_in_packet_rejected(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging_bad2"
    staging_dir.mkdir()
    (staging_dir / "forensic_packets").mkdir()
    for qid in CANONICAL_TARGET_IDS:
        # Write wrong question_id
        (staging_dir / "forensic_packets" / f"{qid}.json").write_text(
            json.dumps({"question_id": "wrong_id"}), encoding="utf-8"
        )

    bad_zip = tmp_path / "bad2.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for f in (staging_dir / "forensic_packets").glob("*.json"):
            z.write(f, arcname=f"forensic_packets/{f.name}")

    out_path = tmp_path / "out.json"
    with (
        patch("scripts.freeze_verification_forensic_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        pytest.raises(DataValidationError, match="Packet question_id mismatch"),
    ):
        freezer = ForensicLabelFreezer(review_packets_path=bad_zip, output_path=out_path)
        freezer.run()


def test_04_missing_historical_arm_rejected(tmp_path: Path) -> None:
    zip_path, packets = _create_mock_review_zip(tmp_path)
    # Remove CANDIDATE arm from 102047
    bad_staging = tmp_path / "staging_arm_bad"
    bad_staging.mkdir()
    (bad_staging / "forensic_packets").mkdir()
    for qid, pkt in packets.items():
        pkt_copy = copy.deepcopy(pkt)
        if qid == "102047":
            del pkt_copy["arms"]["CANDIDATE"]
        (bad_staging / "forensic_packets" / f"{qid}.json").write_text(json.dumps(pkt_copy), encoding="utf-8")

    bad_zip = tmp_path / "bad_arm.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for f in (bad_staging / "forensic_packets").glob("*.json"):
            z.write(f, arcname=f"forensic_packets/{f.name}")

    out_path = tmp_path / "out.json"
    with (
        patch("scripts.freeze_verification_forensic_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        pytest.raises(DataValidationError, match="Arm 'CANDIDATE' missing from packet for question '102047'"),
    ):
        freezer = ForensicLabelFreezer(review_packets_path=bad_zip, output_path=out_path)
        freezer.run()


def test_05_historical_stop_reason_mismatch_rejected(tmp_path: Path) -> None:
    zip_path, packets = _create_mock_review_zip(tmp_path)
    bad_staging = tmp_path / "staging_stop_bad"
    bad_staging.mkdir()
    (bad_staging / "forensic_packets").mkdir()
    for qid, pkt in packets.items():
        pkt_copy = copy.deepcopy(pkt)
        if qid == "102047":
            pkt_copy["arms"]["BASE"]["agent_outcome"]["stop_reason"] = "generation_failed"
        (bad_staging / "forensic_packets" / f"{qid}.json").write_text(json.dumps(pkt_copy), encoding="utf-8")

    bad_zip = tmp_path / "bad_stop.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for f in (bad_staging / "forensic_packets").glob("*.json"):
            z.write(f, arcname=f"forensic_packets/{f.name}")

    out_path = tmp_path / "out.json"
    with (
        patch("scripts.freeze_verification_forensic_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        pytest.raises(DataValidationError, match="Stop reason mismatch for 102047 BASE"),
    ):
        freezer = ForensicLabelFreezer(review_packets_path=bad_zip, output_path=out_path)
        freezer.run()


def test_06_missing_claim_id_rejected(tmp_path: Path) -> None:
    zip_path, packets = _create_mock_review_zip(tmp_path)
    bad_staging = tmp_path / "staging_cid_bad"
    bad_staging.mkdir()
    (bad_staging / "forensic_packets").mkdir()
    for qid, pkt in packets.items():
        pkt_copy = copy.deepcopy(pkt)
        if qid == "95861":
            # Remove C3
            pkt_copy["arms"]["BASE"]["rule_verifier_replay"]["replay_result"]["claim_verifications"] = [
                {"claim_id": "C1", "claim_text": "Text 1"},
                {"claim_id": "C2", "claim_text": "Text 2"},
            ]
        (bad_staging / "forensic_packets" / f"{qid}.json").write_text(json.dumps(pkt_copy), encoding="utf-8")

    bad_zip = tmp_path / "bad_cid.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for f in (bad_staging / "forensic_packets").glob("*.json"):
            z.write(f, arcname=f"forensic_packets/{f.name}")

    out_path = tmp_path / "out.json"
    with (
        patch("scripts.freeze_verification_forensic_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        pytest.raises(DataValidationError, match="Approved claim 'C3' for 95861 BASE not found"),
    ):
        freezer = ForensicLabelFreezer(review_packets_path=bad_zip, output_path=out_path)
        freezer.run()


def test_07_claim_text_empty_rejected(tmp_path: Path) -> None:
    zip_path, packets = _create_mock_review_zip(tmp_path)
    bad_staging = tmp_path / "staging_empty_bad"
    bad_staging.mkdir()
    (bad_staging / "forensic_packets").mkdir()
    for qid, pkt in packets.items():
        pkt_copy = copy.deepcopy(pkt)
        if qid == "102047":
            pkt_copy["arms"]["BASE"]["rule_verifier_replay"]["replay_result"]["claim_verifications"] = [
                {"claim_id": "C1", "claim_text": ""}
            ]
        (bad_staging / "forensic_packets" / f"{qid}.json").write_text(json.dumps(pkt_copy), encoding="utf-8")

    bad_zip = tmp_path / "bad_empty.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for f in (bad_staging / "forensic_packets").glob("*.json"):
            z.write(f, arcname=f"forensic_packets/{f.name}")

    out_path = tmp_path / "out.json"
    with (
        patch("scripts.freeze_verification_forensic_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        pytest.raises(DataValidationError, match="has empty claim_text"),
    ):
        freezer = ForensicLabelFreezer(review_packets_path=bad_zip, output_path=out_path)
        freezer.run()


def test_08_approved_label_mismatch_rejected(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    bad_spec = copy.deepcopy(APPROVED_HUMAN_SPEC)
    # Modify 102047 BASE from CONTRADICTED to SUPPORTED
    bad_spec["102047"]["BASE"]["claims"]["C1"] = ApprovedClaimSpec(
        label=ClaimEntailmentLabel.SUPPORTED,
        error_tags=(ForensicErrorTag.NONE,),
    )

    out_path = tmp_path / "out.json"
    with pytest.raises(DataValidationError, match="Aggregate label counts mismatch"):
        _run_with_patched_zip(zip_path, out_path, approved_spec=bad_spec)


def test_09_unsupported_extra_label_rejected(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    bad_spec = copy.deepcopy(APPROVED_HUMAN_SPEC)
    # Add extra claim C2 to 102047 BASE
    bad_spec["102047"]["BASE"]["claims"]["C2"] = ApprovedClaimSpec(
        label=ClaimEntailmentLabel.SUPPORTED,
        error_tags=(ForensicErrorTag.NONE,),
    )

    out_path = tmp_path / "out.json"
    with pytest.raises(DataValidationError, match="Approved claim 'C2' for 102047 BASE not found"):
        _run_with_patched_zip(zip_path, out_path, approved_spec=bad_spec)


def test_10_wrong_error_tag_rejected(tmp_path: Path) -> None:
    # Error tag enum must be valid
    with pytest.raises(ValueError):
        ForensicErrorTag("INVALID_TAG_NAME")


def test_11_generation_failed_arm_cannot_receive_fabricated_claim(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    bad_spec = copy.deepcopy(APPROVED_HUMAN_SPEC)
    # Attempt to assign claim C1 to 147239 BASE (which is generation_failed)
    bad_spec["147239"]["BASE"]["claims"]["C1"] = ApprovedClaimSpec(
        label=ClaimEntailmentLabel.SUPPORTED,
        error_tags=(ForensicErrorTag.NONE,),
    )

    out_path = tmp_path / "out.json"
    with pytest.raises(DataValidationError, match="Generation-failed arm 147239 BASE cannot receive approved claims"):
        _run_with_patched_zip(zip_path, out_path, approved_spec=bad_spec)


def test_12_exact_aggregate_counts_required(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "out.json"
    artifact = _run_with_patched_zip(zip_path, out_path)

    agg = artifact["aggregate"]
    assert agg["question_count"] == 4
    assert agg["historical_arm_count"] == 8
    assert agg["labeled_claim_count"] == 11
    assert agg["supported"] == 2
    assert agg["contradicted"] == 5
    assert agg["insufficient"] == 4
    assert agg["generation_failed_unlabeled_arms"] == 2


def test_13_original_source_zip_not_modified(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    sha_before = sha256_file(zip_path)
    out_path = tmp_path / "out.json"
    _run_with_patched_zip(zip_path, out_path)
    sha_after = sha256_file(zip_path)
    assert sha_before == sha_after


def test_14_real_label_artifact_output_path_must_be_external_untracked(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "verification-human-forensic-labels-v1.json"
    _run_with_patched_zip(zip_path, out_path)
    assert out_path.exists()
    assert not Path("src/verification-human-forensic-labels-v1.json").exists()
    assert not Path("docs/verification-human-forensic-labels-v1.json").exists()


def test_15_no_semantic_verifier_invoked(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "out.json"
    with patch("legal_agentic_rag.generation.semantic_verifier.ModelBackedCitationVerifier") as mock_sem:
        _run_with_patched_zip(zip_path, out_path)
        mock_sem.assert_not_called()


def test_16_no_retrieval_invoked(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "out.json"
    with (
        patch("legal_agentic_rag.retrieval.dense.DenseRetriever") as mock_dense,
        patch("legal_agentic_rag.retrieval.fixed.FixedRetriever") as mock_fixed,
    ):
        _run_with_patched_zip(zip_path, out_path)
        mock_dense.assert_not_called()
        mock_fixed.assert_not_called()


def test_17_no_generation_invoked(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "out.json"
    with (
        patch("legal_agentic_rag.generation.model_generator.ModelBackedAnswerGenerator") as mock_gen,
        patch("legal_agentic_rag.generation.transformers_provider.TransformersChatProvider") as mock_trans,
    ):
        _run_with_patched_zip(zip_path, out_path)
        mock_gen.assert_not_called()
        mock_trans.assert_not_called()


def test_18_usage_policy_prohibits_training_fine_tuning_retrieval_supervision(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "out.json"
    artifact = _run_with_patched_zip(zip_path, out_path)
    prohibited = artifact["usage_policy"]["prohibited_initial_uses"]
    assert "training" in prohibited
    assert "fine_tuning" in prohibited
    assert "retrieval_relevance_supervision" in prohibited
    assert "public_test_annotation" in prohibited
    assert "private_test_annotation" in prohibited


def test_19_artifact_contains_review_source_sha(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path = tmp_path / "out.json"
    artifact = _run_with_patched_zip(zip_path, out_path)
    assert artifact["source_review_package"]["sha256"] == sha256_file(zip_path)
    assert artifact["source_review_package"]["filename"] == "verification-forensic-review-packets.zip"


def test_20_deterministic_output_bytes_for_identical_inputs(tmp_path: Path) -> None:
    zip_path, _ = _create_mock_review_zip(tmp_path)
    out_path_1 = tmp_path / "out1.json"
    out_path_2 = tmp_path / "out2.json"

    _run_with_patched_zip(zip_path, out_path_1)
    _run_with_patched_zip(zip_path, out_path_2)

    assert out_path_1.read_bytes() == out_path_2.read_bytes()
    assert sha256_file(out_path_1) == sha256_file(out_path_2)
