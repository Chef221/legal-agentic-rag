"""Unit tests for freezing positive-control human labels (B-FORENSIC-1C)."""

from __future__ import annotations

from collections import Counter
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
from scripts.freeze_verification_positive_control_labels import (
    APPROVED_POSITIVE_CONTROLS_MATRIX,
    CANONICAL_APPROVAL_DATE,
    CANONICAL_APPROVAL_KIND,
    CANONICAL_APPROVAL_STATEMENT,
    CANONICAL_REVIEW_ZIP_SHA256,
    CANONICAL_REVIEWER_ID,
    ApprovedClaimReview,
    HumanEntailmentLabel,
    LegalErrorTag,
    PositiveControlLabelFreezer,
    sha256_file,
    sha256_text,
)


def _create_mock_review_bundle(tmp_path: Path) -> dict[str, Path]:
    """Create a valid mock review bundle matching the 16 PRIMARY positive controls."""
    bundle_dir = tmp_path / "mock_review_bundle"
    exec_dir = bundle_dir / "execution"
    results_dir = bundle_dir / "results"
    packets_dir = bundle_dir / "positive_control_packets"

    exec_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    packets_dir.mkdir(parents=True)

    exec_id = {
        "source_kind": "canonical_zip",
        "archive_filename": "phase-a-current-system-census-final-evidence.zip",
        "primary_candidate_count": 16,
        "reserve_candidate_count": 8,
    }
    (exec_dir / "control_source_identity.json").write_text(json.dumps(exec_id, indent=2), encoding="utf-8")

    # Construct mock packets for all 16 approved questions
    for qid, approved_claims in APPROVED_POSITIVE_CONTROLS_MATRIX.items():
        claims_list = []
        for cid in sorted(approved_claims.keys()):
            claims_list.append({
                "claim_id": cid,
                "claim_text": f"Mock claim text for {qid} {cid} with legal terms.",
                "evidence_ids": ["E1"],
                "status": "supported",
                "lexical_support_score": 1.0,
                "numeric_match": True,
                "negation_match": True,
                "errors": [],
            })

        packet = {
            "schema_version": "1.0",
            "question_id": qid,
            "control_metadata": {
                "stratum": "A_SINGLE_CLAIM_CLEAN",
                "selection_key": sha256_text(f"verification-positive-control-v1|{qid}"),
                "pool_type": "primary",
                "sampling_algorithm": "deterministic_sha256_stratified_v1",
            },
            "historical_arm": {
                "agent_outcome": {"stop_reason": "answer_verified", "attempt": 1},
                "historical_verification": {
                    "is_valid": True,
                    "valid_citations": [{"evidence_id": "E1"}],
                    "invalid_citations": [],
                    "claim_verifications": claims_list,
                },
            },
            "human_forensic_review": {
                "review_status": "unreviewed",
                "claim_labels": None,
            },
        }
        (packets_dir / f"{qid}.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")

    zip_path = tmp_path / "mock_review_packets.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                fp = Path(root) / file
                arcname = fp.relative_to(bundle_dir).as_posix()
                z.write(fp, arcname=arcname)

    return {
        "bundle_dir": bundle_dir,
        "zip_path": zip_path,
        "out_json": tmp_path / "out_labels.json",
        "out_zip": tmp_path / "out_labels.zip",
    }


def _run_freezer_with_mock(env: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    """Helper to run PositiveControlLabelFreezer patched for mock ZIP SHA."""
    zip_sha = sha256_file(env["zip_path"])
    zip_size = env["zip_path"].stat().st_size

    with (
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SHA256", zip_sha),
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SIZE", zip_size),
    ):
        freezer = PositiveControlLabelFreezer(
            review_packets_path=env["zip_path"],
            output_json_path=env["out_json"],
            output_zip_path=env.get("out_zip"),
            **kwargs,
        )
        return freezer.run()


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------


def test_01_wrong_source_zip_sha_rejected(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    with pytest.raises(DataValidationError, match="Review packets ZIP SHA mismatch"):
        freezer = PositiveControlLabelFreezer(
            review_packets_path=env["zip_path"],
            output_json_path=env["out_json"],
        )
        freezer.run()


def test_02_missing_primary_packet_rejected(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    bad_bundle = tmp_path / "bad_bundle"
    import shutil
    shutil.copytree(env["bundle_dir"], bad_bundle)
    # Remove one packet
    (bad_bundle / "positive_control_packets" / "75171.json").unlink()

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for root, _, files in os.walk(bad_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(bad_bundle).as_posix())

    with (
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SIZE", bad_zip.stat().st_size),
        pytest.raises(DataValidationError, match="Expected exactly 16 primary packet JSON files, got 15"),
    ):
        freezer = PositiveControlLabelFreezer(
            review_packets_path=bad_zip,
            output_json_path=env["out_json"],
        )
        freezer.run()


def test_03_unexpected_primary_qid_rejected(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    bad_bundle = tmp_path / "bad_bundle_2"
    import shutil
    shutil.copytree(env["bundle_dir"], bad_bundle)
    # Replace 75171 with unknown 999999
    (bad_bundle / "positive_control_packets" / "75171.json").unlink()
    pkt_data = {
        "schema_version": "1.0",
        "question_id": "999999",
        "control_metadata": {"stratum": "A_SINGLE_CLAIM_CLEAN", "selection_key": "k", "pool_type": "primary"},
        "historical_arm": {"agent_outcome": {"stop_reason": "answer_verified"}},
    }
    (bad_bundle / "positive_control_packets" / "999999.json").write_text(json.dumps(pkt_data), encoding="utf-8")

    bad_zip = tmp_path / "bad2.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for root, _, files in os.walk(bad_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(bad_bundle).as_posix())

    with (
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SIZE", bad_zip.stat().st_size),
        pytest.raises(DataValidationError, match="Unexpected packet question_id '999999'"),
    ):
        freezer = PositiveControlLabelFreezer(
            review_packets_path=bad_zip,
            output_json_path=env["out_json"],
        )
        freezer.run()


def test_04_reserve_substitution_rejected(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    bad_bundle = tmp_path / "bad_bundle_reserve"
    import shutil
    shutil.copytree(env["bundle_dir"], bad_bundle)
    # Change pool_type to reserve
    pkt_file = bad_bundle / "positive_control_packets" / "75171.json"
    data = json.loads(pkt_file.read_text(encoding="utf-8"))
    data["control_metadata"]["pool_type"] = "reserve"
    pkt_file.write_text(json.dumps(data), encoding="utf-8")

    bad_zip = tmp_path / "bad_reserve.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for root, _, files in os.walk(bad_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(bad_bundle).as_posix())

    with (
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SIZE", bad_zip.stat().st_size),
        pytest.raises(DataValidationError, match="has pool_type 'reserve', expected 'primary'"),
    ):
        freezer = PositiveControlLabelFreezer(
            review_packets_path=bad_zip,
            output_json_path=env["out_json"],
        )
        freezer.run()


def test_05_missing_claim_rejected(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    bad_bundle = tmp_path / "bad_bundle_claim"
    import shutil
    shutil.copytree(env["bundle_dir"], bad_bundle)
    # Remove C3 from 116877
    pkt_file = bad_bundle / "positive_control_packets" / "116877.json"
    data = json.loads(pkt_file.read_text(encoding="utf-8"))
    data["historical_arm"]["historical_verification"]["claim_verifications"] = [
        c for c in data["historical_arm"]["historical_verification"]["claim_verifications"] if c["claim_id"] != "C3"
    ]
    pkt_file.write_text(json.dumps(data), encoding="utf-8")

    bad_zip = tmp_path / "bad_claim.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for root, _, files in os.walk(bad_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(bad_bundle).as_posix())

    with (
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SHA256", sha256_file(bad_zip)),
        patch("scripts.freeze_verification_positive_control_labels.CANONICAL_REVIEW_ZIP_SIZE", bad_zip.stat().st_size),
        pytest.raises(DataValidationError, match="Claim IDs mismatch for QID 116877"),
    ):
        freezer = PositiveControlLabelFreezer(
            review_packets_path=bad_zip,
            output_json_path=env["out_json"],
        )
        freezer.run()


def test_06_claim_text_sha256_bound_exact(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    overlay = _run_freezer_with_mock(env)

    for qid, q_data in overlay["questions"].items():
        for cid, c_data in q_data["claims"].items():
            expected_sha = sha256_text(c_data["claim_text"])
            assert c_data["claim_text_sha256"] == expected_sha


def test_07_exact_aggregate_counts(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    overlay = _run_freezer_with_mock(env)

    agg = overlay["aggregate"]
    assert agg["question_count"] == 16
    assert agg["historical_arm_count"] == 16
    assert agg["labeled_claim_count"] == 27
    assert agg["supported"] == 16
    assert agg["contradicted"] == 2
    assert agg["insufficient"] == 9


def test_08_combined_benchmark_counts(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    overlay = _run_freezer_with_mock(env)

    comb = overlay["combined_benchmark_summary"]
    assert comb["suspicious_forensic_claims"] == 11
    assert comb["suspicious_forensic_supported"] == 2
    assert comb["suspicious_forensic_contradicted"] == 5
    assert comb["suspicious_forensic_insufficient"] == 4

    assert comb["positive_control_claims"] == 27
    assert comb["positive_control_supported"] == 16
    assert comb["positive_control_contradicted"] == 2
    assert comb["positive_control_insufficient"] == 9

    assert comb["total_combined_benchmark_claims"] == 38
    assert comb["total_combined_supported"] == 18
    assert comb["total_combined_contradicted"] == 7
    assert comb["total_combined_insufficient"] == 13


def test_09_source_zip_unchanged(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    before_sha = sha256_file(env["zip_path"])
    _run_freezer_with_mock(env)
    after_sha = sha256_file(env["zip_path"])
    assert before_sha == after_sha


def test_10_json_output_deterministic(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    _run_freezer_with_mock(env)
    sha1 = sha256_file(env["out_json"])

    out_json2 = tmp_path / "out_labels_2.json"
    env["out_json"] = out_json2
    _run_freezer_with_mock(env)
    sha2 = sha256_file(out_json2)

    assert sha1 == sha2


def test_11_usage_policy_forbids_training_and_retrieval(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    overlay = _run_freezer_with_mock(env)

    prohibited = set(overlay["usage_policy"]["prohibited_initial_uses"])
    assert "training" in prohibited
    assert "fine_tuning" in prohibited
    assert "retrieval_relevance_supervision" in prohibited


def test_12_no_semantic_verifier_invoked(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    with patch("legal_agentic_rag.generation.semantic_verifier.ModelBackedCitationVerifier") as mock_sem:
        _run_freezer_with_mock(env)
        mock_sem.assert_not_called()


def test_13_no_retrieval_invoked(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    with (
        patch("legal_agentic_rag.retrieval.dense.DenseRetriever") as mock_dense,
        patch("legal_agentic_rag.retrieval.fixed.FixedRetriever") as mock_fixed,
    ):
        _run_freezer_with_mock(env)
        mock_dense.assert_not_called()
        mock_fixed.assert_not_called()


def test_14_no_generation_invoked(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    with (
        patch("legal_agentic_rag.generation.model_generator.ModelBackedAnswerGenerator") as mock_gen,
        patch("legal_agentic_rag.generation.transformers_provider.TransformersChatProvider") as mock_trans,
    ):
        _run_freezer_with_mock(env)
        mock_gen.assert_not_called()
        mock_trans.assert_not_called()


def test_15_transport_zip_packaged(tmp_path: Path) -> None:
    env = _create_mock_review_bundle(tmp_path)
    _run_freezer_with_mock(env)

    assert env["out_zip"].exists()
    with zipfile.ZipFile(env["out_zip"], "r") as z:
        names = z.namelist()
        assert env["out_json"].name in names
        assert "control_human_label_identity.json" in names
