"""Comprehensive unit tests for positive-control discovery and deterministic candidate selection (B-FORENSIC-1B)."""

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

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from scripts.select_verification_positive_controls import (
    CANONICAL_DEVELOPMENT_SHA256,
    CANONICAL_EXCLUDED_RELATIONSHIP_IDS,
    CANONICAL_PHASE_A_RESULTS_SHA256,
    CANONICAL_PHASE_A_ZIP_SHA256,
    CANONICAL_SERVING_DATASET_NAME,
    CANONICAL_SERVING_DATASET_REVISION,
    CANONICAL_SERVING_RECORD_COUNT,
    ControlStratum,
    PositiveControlSelector,
    sha256_file,
    sha256_text,
)


def _make_dummy_manifest(
    artifact_type: str,
    record_count: int,
    records_file: str = "records.jsonl",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": artifact_type,
        "artifact_version": "1.0",
        "dataset_name": CANONICAL_SERVING_DATASET_NAME,
        "dataset_revision": CANONICAL_SERVING_DATASET_REVISION,
        "created_at": datetime.now(UTC).isoformat(),
        "record_count": record_count,
        "processing_config_hash": "4cd125739ca9b4046654d00c9c5c468ccc4bcfabe8312ca50638c0559d42b843",
        "code_version": "0.40.0",
        "backend": None,
        "model_name": None,
        "model_revision": None,
        "warnings": [],
        "metadata": {
            "payload_file": records_file,
            "payload_sha256": "0" * 64,
        },
    }


def _create_mock_environment(tmp_path: Path) -> dict[str, Path]:
    """Create complete mock environment for testing positive control selection."""
    # 1. Mock development.json with 991 records
    dev_path = tmp_path / "development.json"
    dev_records = {}
    for i in range(1, 992):
        qid = f"dev_q_{i}"
        dev_records[qid] = {
            "question": f"Question text for {qid}",
            "answer": f"Reference answer for {qid}",
        }
    dev_path.write_text(json.dumps(dev_records, indent=2), encoding="utf-8")

    # 2. Mock serving root with legal_chunks
    serving_root = tmp_path / "serving_root"
    chunks_dir = serving_root / "legal_chunks"
    chunks_dir.mkdir(parents=True)

    chunks_data = []
    chunk_full_text = (
        "Người sử dụng không được phép thực hiện hành vi bị cấm. "
        "Thời hạn giải quyết là 15 ngày kể từ ngày nhận hồ sơ. "
        "Cơ quan có thẩm quyền giải quyết theo quy định pháp luật. "
        "Hồ sơ hợp lệ sẽ được tiếp nhận."
    )
    for i in range(1, 100):
        cid = f"chunk_{i}"
        chunks_data.append({
            "chunk_id": cid,
            "document_id": f"doc_{i}",
            "text": chunk_full_text,
            "source_dataset": "uit-dsc-2026-task2-selected-contexts",
            "source_url": f"https://example.com/{cid}",
            "structure": {"article_number": "1", "article_title": "Title"},
            "metadata": {
                "document_title": f"Doc {i}",
                "document_number": "10/2023",
                "document_type": "Luat",
                "effective_date": "2023-01-01",
                "expiry_date": None,
                "effect_status": "Con hieu luc",
                "source_url": f"https://example.com/{cid}",
            },
        })

    records_file = chunks_dir / "records.jsonl"
    with records_file.open("w", encoding="utf-8") as f:
        for c in chunks_data:
            f.write(json.dumps(c) + "\n")

    manifest_dict = _make_dummy_manifest("legal_chunks", CANONICAL_SERVING_RECORD_COUNT)
    manifest_dict["metadata"]["payload_sha256"] = sha256_file(records_file)
    (chunks_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")

    # 3. Mock Phase-A evidence
    bundle_dir = tmp_path / "phase_a_bundle"
    batch_dir = bundle_dir / "phase-a-current-system-census-batch"
    batch_dir.mkdir(parents=True)

    # 991 records with exact stop reason counts:
    # answer_verified: 806
    # generation_failed: 177
    # citation_verification_failed: 7
    # max_retry_reached: 1
    phase_a_records = []
    dev_qids = list(dev_records.keys())

    for idx, qid in enumerate(dev_qids):
        if idx < 806:
            # answer_verified
            # Distribute into strata:
            # 0..199: D_NEGATION
            # 200..399: C_NUMERIC
            # 400..599: B_MULTI
            # 600..805: A_SINGLE
            if idx < 200:
                claim_text = "Người sử dụng không được phép thực hiện hành vi bị cấm ."
                claims = [
                    {
                        "claim_id": "C1",
                        "claim_text": claim_text,
                        "evidence_ids": ["E1"],
                        "status": "supported",
                        "lexical_support_score": 1.0,
                        "numeric_match": True,
                        "negation_match": True,
                        "errors": [],
                    }
                ]
                answer_text = "Người sử dụng không được phép thực hiện hành vi bị cấm [E1]."
            elif idx < 400:
                claim_text = "Thời hạn giải quyết là 15 ngày kể từ ngày nhận hồ sơ ."
                claims = [
                    {
                        "claim_id": "C1",
                        "claim_text": claim_text,
                        "evidence_ids": ["E1"],
                        "status": "supported",
                        "lexical_support_score": 1.0,
                        "numeric_match": True,
                        "negation_match": True,
                        "errors": [],
                    }
                ]
                answer_text = "Thời hạn giải quyết là 15 ngày kể từ ngày nhận hồ sơ [E1]."
            elif idx < 600:
                claim_text_1 = "Cơ quan có thẩm quyền giải quyết theo quy định ."
                claim_text_2 = "Hồ sơ hợp lệ sẽ được tiếp nhận ."
                claims = [
                    {
                        "claim_id": "C1",
                        "claim_text": claim_text_1,
                        "evidence_ids": ["E1"],
                        "status": "supported",
                        "lexical_support_score": 1.0,
                        "numeric_match": True,
                        "negation_match": True,
                        "errors": [],
                    },
                    {
                        "claim_id": "C2",
                        "claim_text": claim_text_2,
                        "evidence_ids": ["E1"],
                        "status": "supported",
                        "lexical_support_score": 1.0,
                        "numeric_match": True,
                        "negation_match": True,
                        "errors": [],
                    },
                ]
                answer_text = "Cơ quan có thẩm quyền giải quyết theo quy định [E1]. Hồ sơ hợp lệ sẽ được tiếp nhận [E1]."
            else:
                claim_text = "Cơ quan có thẩm quyền giải quyết theo quy định pháp luật ."
                claims = [
                    {
                        "claim_id": "C1",
                        "claim_text": claim_text,
                        "evidence_ids": ["E1"],
                        "status": "supported",
                        "lexical_support_score": 1.0,
                        "numeric_match": True,
                        "negation_match": True,
                        "errors": [],
                    }
                ]
                answer_text = "Cơ quan có thẩm quyền giải quyết theo quy định pháp luật [E1]."
            rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": answer_text,
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "chunk_1",
                            "document_id": "doc_1",
                            "document_title": "Doc 1",
                            "document_number": "10/2023",
                            "article_number": "1",
                            "source_url": "https://example.com/chunk_1",
                        }
                    ],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": qid,
                    "warnings": [],
                    "metadata": {
                        "agent": {
                            "stop_reason": "answer_verified",
                            "attempt": 1,
                        },
                        "context": {
                            "selection_trace": [
                                {
                                    "chunk_id": "chunk_1",
                                    "source_rank": 1,
                                    "selection_rank": 1,
                                    "selected": True,
                                    "reason": "selected",
                                }
                            ]
                        },
                        "selected_evidence": [{"evidence_id": "E1", "chunk_id": "chunk_1"}],
                        "citation_verification": {
                            "is_valid": True,
                            "valid_citations": [
                                {
                                    "evidence_id": "E1",
                                    "chunk_id": "chunk_1",
                                    "document_id": "doc_1",
                                    "document_title": "Doc 1",
                                    "document_number": "10/2023",
                                    "article_number": "1",
                                    "source_url": "https://example.com/chunk_1",
                                }
                            ],
                            "invalid_citations": [],
                            "claim_verifications": claims,
                            "claim_coverage_score": 1.0,
                            "claim_level_verification_performed": True,
                            "errors": [],
                            "warnings": ["semantic_entailment_not_verified"],
                        },
                    },
                },
            }
        elif idx < 806 + 177:
            # generation_failed
            rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": "He thong chua the tra loi...",
                    "citations": [],
                    "insufficient_evidence": True,
                    "retrieval_strategy": "hybrid",
                    "trace_id": qid,
                    "warnings": ["generator:generation_failed"],
                    "metadata": {
                        "agent": {"stop_reason": "generation_failed", "attempt": 1},
                        "context": {"selection_trace": []},
                        "selected_evidence": [],
                    },
                },
            }
        elif idx < 806 + 177 + 7:
            # citation_verification_failed
            rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": "Answer without valid citations...",
                    "citations": [],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": qid,
                    "warnings": [],
                    "metadata": {
                        "agent": {"stop_reason": "citation_verification_failed", "attempt": 2},
                        "context": {"selection_trace": []},
                        "selected_evidence": [],
                    },
                },
            }
        else:
            # max_retry_reached
            rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": "Max retry reached...",
                    "citations": [],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid",
                    "trace_id": qid,
                    "warnings": [],
                    "metadata": {
                        "agent": {"stop_reason": "max_retry_reached", "attempt": 2},
                        "context": {"selection_trace": []},
                        "selected_evidence": [],
                    },
                },
            }
        phase_a_records.append(rec)

    results_file = batch_dir / "results.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for r in phase_a_records:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "schema_version": "1.0",
        "code_version": "0.50.5",
        "record_count": 991,
        "records_sha256": sha256_file(results_file),
        "question_source_sha256": sha256_file(dev_path),
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = tmp_path / "phase-a-evidence.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                fp = Path(root) / file
                arcname = fp.relative_to(bundle_dir).as_posix()
                z.write(fp, arcname=arcname)

    return {
        "development": dev_path,
        "serving_root": serving_root,
        "bundle_dir": bundle_dir,
        "zip_path": zip_path,
        "output_dir": tmp_path / "out",
        "results_file": results_file,
    }


def _run_with_patched_env(env: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    """Helper to run selector with mock environment patched for canonical constants."""
    zip_sha = sha256_file(env["zip_path"])
    dev_sha = sha256_file(env["development"])
    results_sha = sha256_file(env["results_file"])

    with (
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_ZIP_SHA256", zip_sha),
        patch("scripts.select_verification_positive_controls.CANONICAL_DEVELOPMENT_SHA256", dev_sha),
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_RESULTS_SHA256", results_sha),
    ):
        selector = PositiveControlSelector(
            phase_a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
            **kwargs,
        )
        return selector.run()


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------


def test_01_wrong_phase_a_zip_sha_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with (
        patch("scripts.select_verification_positive_controls.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        pytest.raises(DataValidationError, match="Phase-A ZIP SHA mismatch"),
    ):
        selector = PositiveControlSelector(
            phase_a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        selector.run()


def test_02_archive_missing_raw_result_source_blocked(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    bad_bundle = tmp_path / "bad_bundle"
    import shutil
    shutil.copytree(env["bundle_dir"], bad_bundle)
    (bad_bundle / "phase-a-current-system-census-batch" / "results.jsonl").unlink()

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for root, _, files in os.walk(bad_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(bad_bundle).as_posix())

    with (
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_ZIP_SHA256", sha256_file(bad_zip)),
        patch("scripts.select_verification_positive_controls.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        pytest.raises(DataValidationError, match="missing required batch results or manifest"),
    ):
        selector = PositiveControlSelector(
            phase_a_evidence_path=bad_zip,
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        selector.run()


def test_03_wrong_991_record_count_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    # Remove 1 line from results.jsonl inside bundle
    lines = env["results_file"].read_text(encoding="utf-8").splitlines()[:-1]
    env["results_file"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    mf = json.loads((env["bundle_dir"] / "phase-a-current-system-census-batch" / "manifest.json").read_text())
    mf["records_sha256"] = sha256_file(env["results_file"])
    mf["record_count"] = 990
    (env["bundle_dir"] / "phase-a-current-system-census-batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    # Repack zip
    with zipfile.ZipFile(env["zip_path"], "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(env["bundle_dir"]):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(env["bundle_dir"]).as_posix())

    with (
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_ZIP_SHA256", sha256_file(env["zip_path"])),
        patch("scripts.select_verification_positive_controls.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_RESULTS_SHA256", sha256_file(env["results_file"])),
        pytest.raises(DataValidationError, match="Phase-A manifest record_count mismatch: expected 991, got 990"),
    ):
        selector = PositiveControlSelector(
            phase_a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        selector.run()


def test_04_wrong_development_sha_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with pytest.raises(DataValidationError, match="development.json SHA mismatch"):
        selector = PositiveControlSelector(
            phase_a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        selector.run()


def test_05_stop_reason_count_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    lines = [json.loads(line) for line in env["results_file"].read_text().splitlines() if line.strip()]
    # Change one answer_verified to generation_failed
    lines[0]["response"]["metadata"]["agent"]["stop_reason"] = "generation_failed"
    env["results_file"].write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    mf = json.loads((env["bundle_dir"] / "phase-a-current-system-census-batch" / "manifest.json").read_text())
    mf["records_sha256"] = sha256_file(env["results_file"])
    (env["bundle_dir"] / "phase-a-current-system-census-batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    # Repack zip
    with zipfile.ZipFile(env["zip_path"], "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(env["bundle_dir"]):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(env["bundle_dir"]).as_posix())

    with (
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_ZIP_SHA256", sha256_file(env["zip_path"])),
        patch("scripts.select_verification_positive_controls.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_RESULTS_SHA256", sha256_file(env["results_file"])),
        pytest.raises(DataValidationError, match="stop reason 'answer_verified' count mismatch"),
    ):
        selector = PositiveControlSelector(
            phase_a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        selector.run()


def test_06_relationship_ids_excluded_from_primary_and_reserve(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    excluded = {"dev_q_1", "dev_q_2", "dev_q_3"}
    report = _run_with_patched_env(env, excluded_qids=excluded)

    primary_qids = {c["question_id"] for c in report["primary_candidates"]}
    reserve_qids = {c["question_id"] for c in report["reserve_candidates"]}

    assert primary_qids.isdisjoint(excluded)
    assert reserve_qids.isdisjoint(excluded)


def test_07_deterministic_strata_precedence(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_env(env)
    dist = report["candidate_sampling"]["strata_distribution_before_sampling"]
    assert dist["D_NEGATION_MODALITY"] > 0
    assert dist["C_NUMERIC"] > 0
    assert dist["B_MULTI_CLAIM_CLEAN"] > 0
    assert dist["A_SINGLE_CLAIM_CLEAN"] > 0


def test_08_exact_primary_and_reserve_quotas(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_env(env)

    primary = report["primary_candidates"]
    reserve = report["reserve_candidates"]

    assert len(primary) == 16
    assert len(reserve) == 8

    primary_counts = Counter(c["stratum"] for c in primary)
    reserve_counts = Counter(c["stratum"] for c in reserve)

    for stratum in [s.value for s in ControlStratum]:
        assert primary_counts[stratum] == 4
        assert reserve_counts[stratum] == 2


def test_09_primary_and_reserve_disjoint(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_env(env)

    primary_qids = {c["question_id"] for c in report["primary_candidates"]}
    reserve_qids = {c["question_id"] for c in report["reserve_candidates"]}

    assert len(primary_qids) == 16
    assert len(reserve_qids) == 8
    assert primary_qids.isdisjoint(reserve_qids)


def test_10_repeat_run_produces_identical_ids(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report1 = _run_with_patched_env(env)

    out_dir2 = tmp_path / "out2"
    selector = PositiveControlSelector(
        phase_a_evidence_path=env["zip_path"],
        serving_root=env["serving_root"],
        development_path=env["development"],
        output_dir=out_dir2,
    )
    with (
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_ZIP_SHA256", sha256_file(env["zip_path"])),
        patch("scripts.select_verification_positive_controls.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.select_verification_positive_controls.CANONICAL_PHASE_A_RESULTS_SHA256", sha256_file(env["results_file"])),
    ):
        report2 = selector.run()

    p1 = [c["question_id"] for c in report1["primary_candidates"]]
    p2 = [c["question_id"] for c in report2["primary_candidates"]]
    assert p1 == p2


def test_11_selection_key_matches_sha256_formula(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_env(env)

    for c in report["primary_candidates"] + report["reserve_candidates"]:
        expected_key = sha256_text(f"verification-positive-control-v1|{c['question_id']}")
        assert c["selection_key"] == expected_key


def test_12_all_primary_evidence_and_traces_reconstructed(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_env(env)

    val = report["primary_validation"]
    assert val["total_primary_materialized"] == 16
    assert val["selected_chunk_lookup_pass_count"] == 16
    assert val["source_mapping_pass_count"] == 16
    assert val["metadata_crosscheck_pass_count"] == 16
    assert val["rule_verifier_replay_pass_count"] == 16


def test_13_human_labels_remain_unreviewed(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_env(env)

    for p_file in (env["output_dir"] / "positive_control_packets").glob("*.json"):
        pkt = json.loads(p_file.read_text(encoding="utf-8"))
        review = pkt["human_forensic_review"]
        assert review["review_status"] == "unreviewed"
        assert review["claim_labels"] is None
        assert review["reviewer_notes"] is None


def test_14_no_semantic_verifier_invoked(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with patch("legal_agentic_rag.generation.semantic_verifier.ModelBackedCitationVerifier") as mock_sem:
        _run_with_patched_env(env)
        mock_sem.assert_not_called()


def test_15_no_retrieval_invoked(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with (
        patch("legal_agentic_rag.retrieval.dense.DenseRetriever") as mock_dense,
        patch("legal_agentic_rag.retrieval.fixed.FixedRetriever") as mock_fixed,
    ):
        _run_with_patched_env(env)
        mock_dense.assert_not_called()
        mock_fixed.assert_not_called()


def test_16_no_generation_invoked(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with (
        patch("legal_agentic_rag.generation.model_generator.ModelBackedAnswerGenerator") as mock_gen,
        patch("legal_agentic_rag.generation.transformers_provider.TransformersChatProvider") as mock_trans,
    ):
        _run_with_patched_env(env)
        mock_gen.assert_not_called()
        mock_trans.assert_not_called()


def test_17_real_packet_outputs_not_in_tracked_dirs(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_env(env)

    assert (env["output_dir"] / "positive_control_packets").exists()
    assert len(list((env["output_dir"] / "positive_control_packets").glob("*.json"))) == 16
    assert not Path("src/positive_control_packets").exists()
    assert not Path("docs/positive_control_packets").exists()


def test_18_no_absolute_windows_paths_in_output(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_env(env)

    report_text = (env["output_dir"] / "results" / "control_selection_report.json").read_text(encoding="utf-8")
    assert "C:\\" not in report_text
    assert "c:\\" not in report_text
    assert "Users" not in report_text

    for p_file in (env["output_dir"] / "positive_control_packets").glob("*.json"):
        pkt_text = p_file.read_text(encoding="utf-8")
        assert "C:\\" not in pkt_text
        assert "c:\\" not in pkt_text
        assert "Users" not in pkt_text
