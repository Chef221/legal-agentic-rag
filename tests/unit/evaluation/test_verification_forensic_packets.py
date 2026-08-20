"""Comprehensive unit test suite for verification forensic source materializer (Priority B - B-FORENSIC-0)."""

from __future__ import annotations

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
from scripts.materialize_verification_forensic_packets import (
    CANONICAL_B1A_MEMBER_HASHES,
    CANONICAL_B1A_ZIP_SHA256,
    CANONICAL_BASE_RESULTS_SHA256,
    CANONICAL_CANDIDATE_RESULTS_SHA256,
    CANONICAL_DEVELOPMENT_SHA256,
    CANONICAL_MATERIALIZED_QUESTIONS_SHA256,
    CANONICAL_SERVING_DATASET_NAME,
    CANONICAL_SERVING_DATASET_REVISION,
    CANONICAL_SERVING_RECORD_COUNT,
    CANONICAL_TARGET_IDS,
    ForensicSourceMaterializer,
    sha256_bytes,
    sha256_file,
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
    """Create complete mock environment for testing forensic materialization."""
    # 1. Mock development.json
    dev_path = tmp_path / "development.json"
    dev_records = {}
    for target_qid in CANONICAL_TARGET_IDS:
        dev_records[target_qid] = {
            "question": f"Question text for {target_qid}",
            "answer": f"Reference answer for {target_qid}",
        }
    counter = 1
    while len(dev_records) < 991:
        dev_records[f"dev_q_{counter}"] = {
            "question": f"Question text for dev_q_{counter}",
            "answer": f"Reference answer for dev_q_{counter}",
        }
        counter += 1
    dev_path.write_text(json.dumps(dev_records, indent=2), encoding="utf-8")

    # 2. Mock serving root with legal_chunks
    serving_root = tmp_path / "serving_root"
    chunks_dir = serving_root / "legal_chunks"
    chunks_dir.mkdir(parents=True)

    chunks_data = []
    chunk_ids = [
        "chunk_102047_1", "chunk_102047_2",
        "chunk_147239_1", "chunk_147239_2",
        "chunk_26541_1", "chunk_26541_2",
        "chunk_95861_1", "chunk_95861_2",
    ]
    for cid in chunk_ids:
        chunk_rec = {
            "chunk_id": cid,
            "document_id": f"doc_{cid}",
            "text": f"Legal content text for {cid} including numbers 15 and 2023 and words quy dinh phap luat.",
            "source_dataset": "uit-dsc-2026-task2-selected-contexts",
            "source_url": f"https://example.com/{cid}",
            "structure": {
                "article_number": "10",
                "article_title": "Tieu de dieu",
            },
            "metadata": {
                "document_title": f"Van ban {cid}",
                "document_number": "10/2023/TT-BCT",
                "document_type": "Thong tu",
                "effective_date": "2023-06-15",
                "expiry_date": None,
                "effect_status": "Con hieu luc",
                "source_url": f"https://example.com/{cid}",
            },
        }
        chunks_data.append(chunk_rec)

    records_file = chunks_dir / "records.jsonl"
    with records_file.open("w", encoding="utf-8") as f:
        for c in chunks_data:
            f.write(json.dumps(c) + "\n")

    manifest_dict = _make_dummy_manifest("legal_chunks", CANONICAL_SERVING_RECORD_COUNT)
    manifest_dict["metadata"]["payload_sha256"] = sha256_file(records_file)
    (chunks_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")

    # 3. Mock B1A bundle
    bundle_dir = tmp_path / "b1a_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "configs").mkdir()
    (bundle_dir / "evidence").mkdir()
    (bundle_dir / "results").mkdir()
    (bundle_dir / "base_batch").mkdir()
    (bundle_dir / "candidate_batch").mkdir()

    (bundle_dir / "configs" / "phase-b1a-graph-routing-cases.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "configs" / "base_runtime_config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "configs" / "candidate_runtime_config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "results" / "phase_b1a_paired_report.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "results" / "phase_b1a_decision_report.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "base_batch" / "batch_state.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "candidate_batch" / "batch_state.json").write_text("{}", encoding="utf-8")

    mat_ident = {
        "candidate": "PHASE-B1A",
        "source_question_count": 991,
        "source_question_sha256": sha256_file(dev_path),
        "materialized_case_count": 22,
        "materialized_case_sha256": CANONICAL_MATERIALIZED_QUESTIONS_SHA256,
        "materialized_question_ids": CANONICAL_TARGET_IDS + [f"other_{i}" for i in range(18)],
    }
    (bundle_dir / "evidence" / "materialized_questions_identity.json").write_text(
        json.dumps(mat_ident, indent=2), encoding="utf-8"
    )

    # Construct mock 22 records for base and candidate
    base_records = []
    cand_records = []
    all_22_ids = CANONICAL_TARGET_IDS + [f"other_{i}" for i in range(18)]

    for qid in all_22_ids:
        # Base record
        cid = f"chunk_{qid}_1" if qid in CANONICAL_TARGET_IDS else "chunk_102047_1"
        is_gen_failed_base = qid == "147239"

        if is_gen_failed_base:
            base_rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": "He thong chua tim thay...",
                    "citations": [],
                    "insufficient_evidence": True,
                    "retrieval_strategy": "graph",
                    "trace_id": qid,
                    "warnings": ["generator:generation_failed"],
                    "metadata": {
                        "agent": {
                            "stop_reason": "generation_failed",
                            "attempt": 1,
                        },
                        "context": {
                            "selection_trace": [
                                {
                                    "chunk_id": cid,
                                    "source_rank": 1,
                                    "selection_rank": 1,
                                    "selected": True,
                                    "reason": "selected",
                                }
                            ]
                        },
                        "selected_evidence": [{"evidence_id": "E1", "chunk_id": cid}],
                    },
                },
            }
        else:
            base_rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": "Theo quy dinh phap luat 15 va 2023 [E1].",
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": cid,
                            "document_id": f"doc_{cid}",
                            "document_title": f"Van ban {cid}",
                            "document_number": "10/2023/TT-BCT",
                            "article_number": "10",
                            "source_url": f"https://example.com/{cid}",
                        }
                    ],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "graph",
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
                                    "chunk_id": cid,
                                    "source_rank": 1,
                                    "selection_rank": 1,
                                    "selected": True,
                                    "reason": "selected",
                                }
                            ]
                        },
                        "selected_evidence": [{"evidence_id": "E1", "chunk_id": cid}],
                        "citation_verification": {
                            "is_valid": True,
                            "valid_citations": [
                                {
                                    "evidence_id": "E1",
                                    "chunk_id": cid,
                                    "document_id": f"doc_{cid}",
                                    "document_title": f"Van ban {cid}",
                                    "document_number": "10/2023/TT-BCT",
                                    "article_number": "10",
                                    "source_url": f"https://example.com/{cid}",
                                }
                            ],
                            "invalid_citations": [],
                            "claim_verifications": [
                                {
                                    "claim_id": "C1",
                                    "claim_text": "Theo quy dinh phap luat 15 va 2023 .",
                                    "evidence_ids": ["E1"],
                                    "status": "supported",
                                    "lexical_support_score": 0.8571428571428571,
                                    "numeric_match": True,
                                    "negation_match": True,
                                    "errors": [],
                                }
                            ],
                            "claim_coverage_score": 1.0,
                            "claim_level_verification_performed": True,
                            "errors": [],
                            "warnings": ["semantic_entailment_not_verified"],
                        },
                    },
                },
            }
        base_records.append(base_rec)

        # Candidate record
        is_gen_failed_cand = qid == "26541"
        if is_gen_failed_cand:
            cand_rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": "He thong chua tim thay...",
                    "citations": [],
                    "insufficient_evidence": True,
                    "retrieval_strategy": "hybrid_rerank",
                    "trace_id": qid,
                    "warnings": ["generator:generation_failed"],
                    "metadata": {
                        "agent": {
                            "stop_reason": "generation_failed",
                            "attempt": 1,
                        },
                        "context": {
                            "selection_trace": [
                                {
                                    "chunk_id": cid,
                                    "source_rank": 1,
                                    "selection_rank": 1,
                                    "selected": True,
                                    "reason": "selected",
                                }
                            ]
                        },
                        "selected_evidence": [{"evidence_id": "E1", "chunk_id": cid}],
                    },
                },
            }
        else:
            cand_rec = {
                "question_id": qid,
                "response": {
                    "question": f"Question text for {qid}",
                    "answer": "Theo quy dinh phap luat 15 va 2023 [E1].",
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": cid,
                            "document_id": f"doc_{cid}",
                            "document_title": f"Van ban {cid}",
                            "document_number": "10/2023/TT-BCT",
                            "article_number": "10",
                            "source_url": f"https://example.com/{cid}",
                        }
                    ],
                    "insufficient_evidence": False,
                    "retrieval_strategy": "hybrid_rerank",
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
                                    "chunk_id": cid,
                                    "source_rank": 1,
                                    "selection_rank": 1,
                                    "selected": True,
                                    "reason": "selected",
                                }
                            ]
                        },
                        "selected_evidence": [{"evidence_id": "E1", "chunk_id": cid}],
                        "citation_verification": {
                            "is_valid": True,
                            "valid_citations": [
                                {
                                    "evidence_id": "E1",
                                    "chunk_id": cid,
                                    "document_id": f"doc_{cid}",
                                    "document_title": f"Van ban {cid}",
                                    "document_number": "10/2023/TT-BCT",
                                    "article_number": "10",
                                    "source_url": f"https://example.com/{cid}",
                                }
                            ],
                            "invalid_citations": [],
                            "claim_verifications": [
                                {
                                    "claim_id": "C1",
                                    "claim_text": "Theo quy dinh phap luat 15 va 2023 .",
                                    "evidence_ids": ["E1"],
                                    "status": "supported",
                                    "lexical_support_score": 0.8571428571428571,
                                    "numeric_match": True,
                                    "negation_match": True,
                                    "errors": [],
                                }
                            ],
                            "claim_coverage_score": 1.0,
                            "claim_level_verification_performed": True,
                            "errors": [],
                            "warnings": ["semantic_entailment_not_verified"],
                        },
                    },
                },
            }
        cand_records.append(cand_rec)

    base_results_file = bundle_dir / "base_batch" / "results.jsonl"
    with base_results_file.open("w", encoding="utf-8") as f:
        for r in base_records:
            f.write(json.dumps(r) + "\n")

    cand_results_file = bundle_dir / "candidate_batch" / "results.jsonl"
    with cand_results_file.open("w", encoding="utf-8") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    base_manifest = {
        "schema_version": "1.0",
        "code_version": "0.50.6",
        "record_count": 22,
        "records_sha256": sha256_file(base_results_file),
        "question_source_sha256": CANONICAL_MATERIALIZED_QUESTIONS_SHA256,
    }
    (bundle_dir / "base_batch" / "manifest.json").write_text(
        json.dumps(base_manifest, indent=2), encoding="utf-8"
    )

    cand_manifest = {
        "schema_version": "1.0",
        "code_version": "0.50.6",
        "record_count": 22,
        "records_sha256": sha256_file(cand_results_file),
        "question_source_sha256": CANONICAL_MATERIALIZED_QUESTIONS_SHA256,
    }
    (bundle_dir / "candidate_batch" / "manifest.json").write_text(
        json.dumps(cand_manifest, indent=2), encoding="utf-8"
    )

    # Pack into zip
    zip_path = tmp_path / "b1a_evidence.zip"
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
        "base_results_file": base_results_file,
        "cand_results_file": cand_results_file,
    }


def _run_with_patched_hashes(env: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    """Helper to run materializer with mock environment patched for canonical constants."""
    target_ids = kwargs.pop("target_ids", CANONICAL_TARGET_IDS)
    zip_sha = sha256_file(env["zip_path"])
    dev_sha = sha256_file(env["development"])
    base_sha = sha256_file(env["base_results_file"])
    cand_sha = sha256_file(env["cand_results_file"])

    # Compute mock member hashes
    mock_member_hashes = {}
    for req in CANONICAL_B1A_MEMBER_HASHES:
        member_file = env["bundle_dir"] / req
        mock_member_hashes[req] = sha256_file(member_file)

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_ZIP_SHA256", zip_sha),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", dev_sha),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", base_sha),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_CANDIDATE_RESULTS_SHA256", cand_sha),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
            target_ids=target_ids,
            **kwargs,
        )
        return materializer.run()


# ----------------------------------------------------------------------
# TESTS (ORIGINAL 24 + NEW PROVENANCE HARDENING TESTS)
# ----------------------------------------------------------------------


def test_01_wrong_b1a_zip_sha_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        pytest.raises(DataValidationError, match="B1A ZIP SHA mismatch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_02_missing_required_b1a_member_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    bad_bundle = tmp_path / "bad_bundle"
    import shutil
    shutil.copytree(env["bundle_dir"], bad_bundle)
    (bad_bundle / "base_batch" / "manifest.json").unlink()

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        for root, _, files in os.walk(bad_bundle):
            for file in files:
                fp = Path(root) / file
                z.write(fp, arcname=fp.relative_to(bad_bundle).as_posix())

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_ZIP_SHA256", sha256_file(bad_zip)),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        pytest.raises(DataValidationError, match="missing required member 'base_batch/manifest.json'"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=bad_zip,
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_03_base_results_sha_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }
    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_ZIP_SHA256", sha256_file(env["zip_path"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", "0" * 64),
        pytest.raises(DataValidationError, match="base_batch results.jsonl SHA mismatch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_04_candidate_results_sha_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }
    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_ZIP_SHA256", sha256_file(env["zip_path"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", sha256_file(env["base_results_file"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_CANDIDATE_RESULTS_SHA256", "0" * 64),
        pytest.raises(DataValidationError, match="candidate_batch results.jsonl SHA mismatch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_05_manifest_results_sha_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    # Corrupt manifest hash inside bundle
    manifest_file = env["bundle_dir"] / "base_batch" / "manifest.json"
    mf = json.loads(manifest_file.read_text(encoding="utf-8"))
    mf["records_sha256"] = "f" * 64
    manifest_file.write_text(json.dumps(mf), encoding="utf-8")

    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", sha256_file(env["base_results_file"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        pytest.raises(DataValidationError, match="base_batch manifest records_sha256 mismatch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["bundle_dir"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_06_record_count_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    manifest_file = env["bundle_dir"] / "base_batch" / "manifest.json"
    mf = json.loads(manifest_file.read_text(encoding="utf-8"))
    mf["record_count"] = 21
    manifest_file.write_text(json.dumps(mf), encoding="utf-8")

    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", sha256_file(env["base_results_file"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        pytest.raises(DataValidationError, match="base_batch manifest record_count mismatch: expected 22, got 21"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["bundle_dir"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_07_target_missing_from_base_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with pytest.raises(DataValidationError, match="Target question ID 'nonexistent_id' missing from BASE batch"):
        _run_with_patched_hashes(env, target_ids=["102047", "nonexistent_id"])


def test_08_target_missing_from_candidate_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    lines = (env["bundle_dir"] / "candidate_batch" / "results.jsonl").read_text().splitlines()
    recs = [json.loads(line) for line in lines if line.strip()]
    recs[0]["question_id"] = "changed_id"
    cand_file = env["bundle_dir"] / "candidate_batch" / "results.jsonl"
    cand_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    mf = json.loads((env["bundle_dir"] / "candidate_batch" / "manifest.json").read_text())
    mf["records_sha256"] = sha256_file(cand_file)
    (env["bundle_dir"] / "candidate_batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", sha256_file(env["base_results_file"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_CANDIDATE_RESULTS_SHA256", sha256_file(cand_file)),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        pytest.raises(DataValidationError, match="missing from CANDIDATE batch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["bundle_dir"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_09_duplicate_target_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with pytest.raises(DataValidationError, match="Target IDs contain duplicates"):
        ForensicSourceMaterializer(
            b1a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
            target_ids=["102047", "102047"],
        )


def test_10_wrong_development_sha_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with pytest.raises(DataValidationError, match="development.json SHA mismatch"):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["zip_path"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_11_selected_chunk_missing_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    # Clear records.jsonl in legal_chunks and update manifest hash
    records_file = env["serving_root"] / "legal_chunks" / "records.jsonl"
    records_file.write_text("", encoding="utf-8")
    mf = json.loads((env["serving_root"] / "legal_chunks" / "manifest.json").read_text())
    mf["metadata"]["payload_sha256"] = sha256_file(records_file)
    (env["serving_root"] / "legal_chunks" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    with pytest.raises(DataValidationError, match="legal_chunks artifact missing .* required chunk IDs"):
        _run_with_patched_hashes(env)


def test_12_selected_evidence_eid_chunk_id_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    base_file = env["bundle_dir"] / "base_batch" / "results.jsonl"
    lines = base_file.read_text().splitlines()
    recs = [json.loads(line) for line in lines if line.strip()]
    # Change chunk_id to unknown chunk
    recs[0]["response"]["metadata"]["selected_evidence"] = [{"evidence_id": "E1", "chunk_id": "unknown_chunk_xyz"}]
    base_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    mf = json.loads((env["bundle_dir"] / "base_batch" / "manifest.json").read_text())
    mf["records_sha256"] = sha256_file(base_file)
    (env["bundle_dir"] / "base_batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    # Repack zip
    with zipfile.ZipFile(env["zip_path"], "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(env["bundle_dir"]):
            for file in files:
                fp = Path(root) / file
                arcname = fp.relative_to(env["bundle_dir"]).as_posix()
                z.write(fp, arcname=arcname)

    with pytest.raises(DataValidationError, match="legal_chunks artifact missing 1 required chunk IDs: \\['unknown_chunk_xyz'\\]"):
        _run_with_patched_hashes(env)


def test_13_citation_metadata_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    base_file = env["bundle_dir"] / "base_batch" / "results.jsonl"
    lines = base_file.read_text().splitlines()
    recs = [json.loads(line) for line in lines if line.strip()]
    # Mismatch document_id in citation
    recs[0]["response"]["citations"][0]["document_id"] = "wrong_doc_id"
    base_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    mf = json.loads((env["bundle_dir"] / "base_batch" / "manifest.json").read_text())
    mf["records_sha256"] = sha256_file(base_file)
    (env["bundle_dir"] / "base_batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    # Repack zip
    with zipfile.ZipFile(env["zip_path"], "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(env["bundle_dir"]):
            for file in files:
                fp = Path(root) / file
                arcname = fp.relative_to(env["bundle_dir"]).as_posix()
                z.write(fp, arcname=arcname)

    report = _run_with_patched_hashes(env)
    assert report["verdict"] == "INVALID_FORENSIC_PROVENANCE"
    assert report["aggregate"]["metadata_crosscheck_pass_count"] == 7  # 1 failed out of 8


def test_14_selection_trace_loaded_from_response_metadata_context(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_hashes(env)
    assert report["verdict"] == "FORENSIC_SOURCE_READY"
    packet_102047 = json.loads((env["output_dir"] / "forensic_packets" / "102047.json").read_text())
    base_trace = packet_102047["arms"]["BASE"]["context_selection_trace"]
    assert len(base_trace) > 0
    assert base_trace[0]["chunk_id"] == "chunk_102047_1"


def test_15_answer_verified_arm_verifier_replay_exact(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_hashes(env)
    packet_102047 = json.loads((env["output_dir"] / "forensic_packets" / "102047.json").read_text())
    base_replay = packet_102047["arms"]["BASE"]["rule_verifier_replay"]
    assert base_replay["replay_applicable"] is True
    assert base_replay["replay_matches_historical"] is True
    assert base_replay["replay_result"]["is_valid"] is True


def test_16_generation_failed_arm_replay_marked_not_applicable(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_hashes(env)
    packet_147239 = json.loads((env["output_dir"] / "forensic_packets" / "147239.json").read_text())
    base_arm = packet_147239["arms"]["BASE"]
    assert base_arm["agent_outcome"]["stop_reason"] == "generation_failed"
    assert base_arm["rule_verifier_replay"]["replay_applicable"] is False
    assert base_arm["rule_verifier_replay"]["reason"] == "historical_verifier_not_reached"


def test_17_no_semantic_verifier_invoked(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with patch("legal_agentic_rag.generation.semantic_verifier.ModelBackedCitationVerifier") as mock_sem:
        _run_with_patched_hashes(env)
        mock_sem.assert_not_called()


def test_18_no_retrieval_functions_invoked(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with (
        patch("legal_agentic_rag.retrieval.dense.DenseRetriever") as mock_dense,
        patch("legal_agentic_rag.retrieval.fixed.FixedRetriever") as mock_fixed,
        patch("legal_agentic_rag.retrieval.rrf.reciprocal_rank_fusion") as mock_rrf,
        patch("legal_agentic_rag.retrieval.rerank.RerankingRetriever") as mock_ce,
    ):
        _run_with_patched_hashes(env)
        mock_dense.assert_not_called()
        mock_fixed.assert_not_called()
        mock_rrf.assert_not_called()
        mock_ce.assert_not_called()


def test_19_no_generation_functions_invoked(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    with (
        patch("legal_agentic_rag.generation.model_generator.ModelBackedAnswerGenerator") as mock_gen,
        patch("legal_agentic_rag.generation.transformers_provider.TransformersChatProvider") as mock_trans,
    ):
        _run_with_patched_hashes(env)
        mock_gen.assert_not_called()
        mock_trans.assert_not_called()


def test_20_no_human_labels_auto_populated(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_hashes(env)
    for qid in CANONICAL_TARGET_IDS:
        packet = json.loads((env["output_dir"] / "forensic_packets" / f"{qid}.json").read_text())
        review = packet["human_forensic_review"]
        assert review["review_status"] == "unreviewed"
        assert review["base_claim_labels"] is None
        assert review["candidate_claim_labels"] is None
        assert review["root_cause_classification"] is None


def test_21_paired_base_candidate_packet_structure(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_hashes(env)
    for qid in CANONICAL_TARGET_IDS:
        packet = json.loads((env["output_dir"] / "forensic_packets" / f"{qid}.json").read_text())
        assert packet["schema_version"] == "1.0"
        assert packet["question_id"] == qid
        assert "BASE" in packet["arms"]
        assert "CANDIDATE" in packet["arms"]
        assert "reference_answer_context" in packet
        assert "ground_truth_status" in packet["reference_answer_context"]


def test_22_output_packet_content_not_written_into_tracked_repo_paths(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_hashes(env)
    assert (env["output_dir"] / "forensic_packets").exists()
    assert len(list((env["output_dir"] / "forensic_packets").glob("*.json"))) == 4
    assert not Path("src/forensic_packets").exists()
    assert not Path("docs/forensic_packets").exists()


def test_23_exact_target_count_equals_4(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_hashes(env)
    assert report["target_question_count"] == 4
    assert len(report["per_arm"]) == 8


def test_24_historical_arm_count_equals_8(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    report = _run_with_patched_hashes(env)
    assert report["historical_arm_count"] == 8
    arm_names = [item["arm"] for item in report["per_arm"]]
    assert arm_names.count("BASE") == 4
    assert arm_names.count("CANDIDATE") == 4


# ----------------------------------------------------------------------
# NEW PROVENANCE HARDENING TESTS (FIX 1 - FIX 5)
# ----------------------------------------------------------------------


def test_25_legal_chunks_payload_checksum_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    # Corrupt legal_chunks payload checksum in manifest
    mf = json.loads((env["serving_root"] / "legal_chunks" / "manifest.json").read_text())
    mf["metadata"]["payload_sha256"] = "f" * 64
    (env["serving_root"] / "legal_chunks" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Artifact payload checksum is incompatible"):
        _run_with_patched_hashes(env)


def test_26_wrong_serving_dataset_name_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mf = json.loads((env["serving_root"] / "legal_chunks" / "manifest.json").read_text())
    mf["dataset_name"] = "wrong-dataset-name"
    (env["serving_root"] / "legal_chunks" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Serving dataset_name mismatch"):
        _run_with_patched_hashes(env)


def test_27_wrong_serving_dataset_revision_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mf = json.loads((env["serving_root"] / "legal_chunks" / "manifest.json").read_text())
    mf["dataset_revision"] = "sha256:wrong_revision"
    (env["serving_root"] / "legal_chunks" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Serving dataset_revision mismatch"):
        _run_with_patched_hashes(env)


def test_28_wrong_serving_record_count_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mf = json.loads((env["serving_root"] / "legal_chunks" / "manifest.json").read_text())
    mf["record_count"] = 12345
    (env["serving_root"] / "legal_chunks" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Serving record_count mismatch"):
        _run_with_patched_hashes(env)


def test_29_wrong_b1a_arm_code_version_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mf = json.loads((env["bundle_dir"] / "base_batch" / "manifest.json").read_text())
    mf["code_version"] = "0.99.0"
    (env["bundle_dir"] / "base_batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", sha256_file(env["base_results_file"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        pytest.raises(DataValidationError, match="base_batch manifest code_version mismatch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["bundle_dir"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_30_wrong_b1a_arm_question_source_sha256_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mf = json.loads((env["bundle_dir"] / "base_batch" / "manifest.json").read_text())
    mf["question_source_sha256"] = "f" * 64
    (env["bundle_dir"] / "base_batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", sha256_file(env["base_results_file"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        pytest.raises(DataValidationError, match="base_batch manifest question_source_sha256 mismatch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["bundle_dir"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_31_extracted_bundle_wrong_member_sha_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    mock_member_hashes = {
        req: sha256_file(env["bundle_dir"] / req) for req in CANONICAL_B1A_MEMBER_HASHES
    }
    # Corrupt one member file in directory mode
    (env["bundle_dir"] / "configs" / "base_runtime_config.json").write_text('{"corrupted": true}', encoding="utf-8")

    with (
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_DEVELOPMENT_SHA256", sha256_file(env["development"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_BASE_RESULTS_SHA256", sha256_file(env["base_results_file"])),
        patch("scripts.materialize_verification_forensic_packets.CANONICAL_B1A_MEMBER_HASHES", mock_member_hashes),
        pytest.raises(DataValidationError, match="member 'configs/base_runtime_config.json' SHA mismatch"),
    ):
        materializer = ForensicSourceMaterializer(
            b1a_evidence_path=env["bundle_dir"],
            serving_root=env["serving_root"],
            development_path=env["development"],
            output_dir=env["output_dir"],
        )
        materializer.run()


def test_32_existing_chunk_selected_evidence_trace_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    base_file = env["bundle_dir"] / "base_batch" / "results.jsonl"
    lines = base_file.read_text().splitlines()
    recs = [json.loads(line) for line in lines if line.strip()]
    # Point selected_evidence E1 to chunk_102047_2 while selection_trace says chunk_102047_1
    recs[0]["response"]["metadata"]["selected_evidence"] = [
        {"evidence_id": "E1", "chunk_id": "chunk_102047_2"}
    ]
    # Keep citation matching the new chunk so citation cross-check passes but trace check fails
    recs[0]["response"]["citations"][0]["chunk_id"] = "chunk_102047_2"
    recs[0]["response"]["citations"][0]["document_id"] = "doc_chunk_102047_2"
    recs[0]["response"]["citations"][0]["document_title"] = "Van ban chunk_102047_2"
    base_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    mf = json.loads((env["bundle_dir"] / "base_batch" / "manifest.json").read_text())
    mf["records_sha256"] = sha256_file(base_file)
    (env["bundle_dir"] / "base_batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    # Repack zip
    with zipfile.ZipFile(env["zip_path"], "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(env["bundle_dir"]):
            for file in files:
                fp = Path(root) / file
                arcname = fp.relative_to(env["bundle_dir"]).as_posix()
                z.write(fp, arcname=arcname)

    report = _run_with_patched_hashes(env)
    assert report["verdict"] == "INVALID_FORENSIC_PROVENANCE"
    assert report["aggregate"]["source_mapping_pass_count"] == 7  # 1 failed out of 8


def test_33_non_cited_generation_failed_arm_trace_mismatch_rejected(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    base_file = env["bundle_dir"] / "base_batch" / "results.jsonl"
    lines = base_file.read_text().splitlines()
    recs = [json.loads(line) for line in lines if line.strip()]
    # Case 147239 is generation_failed in BASE arm (0 citations). Modify selected_evidence vs trace:
    target_idx = next(i for i, r in enumerate(recs) if r["question_id"] == "147239")
    recs[target_idx]["response"]["metadata"]["selected_evidence"] = [
        {"evidence_id": "E1", "chunk_id": "chunk_147239_2"}
    ]
    # trace has chunk_147239_1
    base_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    mf = json.loads((env["bundle_dir"] / "base_batch" / "manifest.json").read_text())
    mf["records_sha256"] = sha256_file(base_file)
    (env["bundle_dir"] / "base_batch" / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")

    # Repack zip
    with zipfile.ZipFile(env["zip_path"], "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(env["bundle_dir"]):
            for file in files:
                fp = Path(root) / file
                arcname = fp.relative_to(env["bundle_dir"]).as_posix()
                z.write(fp, arcname=arcname)

    report = _run_with_patched_hashes(env)
    assert report["verdict"] == "INVALID_FORENSIC_PROVENANCE"
    assert report["aggregate"]["source_mapping_pass_count"] == 7  # 1 failed out of 8


def test_34_canonical_packet_contains_no_absolute_windows_path(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_hashes(env)
    for qid in CANONICAL_TARGET_IDS:
        raw_text = (env["output_dir"] / "forensic_packets" / f"{qid}.json").read_text(encoding="utf-8")
        assert "C:\\" not in raw_text
        assert "c:\\" not in raw_text
        assert "C:/" not in raw_text
        assert "c:/" not in raw_text
        assert "Users" not in raw_text


def test_35_canonical_source_report_contains_no_local_scratch_path(tmp_path: Path) -> None:
    env = _create_mock_environment(tmp_path)
    _run_with_patched_hashes(env)
    raw_text = (env["output_dir"] / "results" / "forensic_source_report.json").read_text(encoding="utf-8")
    assert "C:\\" not in raw_text
    assert "c:\\" not in raw_text
    assert "C:/" not in raw_text
    assert "c:/" not in raw_text
    assert "scratch" not in raw_text
    assert "antigravity-ide" not in raw_text
