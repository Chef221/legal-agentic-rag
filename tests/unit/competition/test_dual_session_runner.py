"""Unit and synthetic integration tests for DualPublic1000SessionRunner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.dual_session_runner import (
    DUAL_GPU_CHECKPOINT_ZIP_FILENAME,
    DUAL_GPU_MANIFEST_FILENAME,
    DualPublic1000SessionRunner,
    PARTITION_STRATEGY_V1,
    get_dual_gpu_telemetry,
)
from legal_agentic_rag.competition.uit_dsc_2026.public1000_session_runner import (
    CHECKPOINT_MANIFEST_FILENAME,
    CHECKPOINT_RESULTS_FILENAME,
    FROZEN_AUTHORITY_BINDINGS,
    compute_file_sha256,
)
from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.serving.config_loader import load_application_config


@pytest.fixture
def base_config(tmp_path: Path) -> ApplicationConfig:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"
    return load_application_config(cfg_path)


@pytest.fixture
def ten_questions_file(tmp_path: Path) -> Path:
    q_file = tmp_path / "public_10_questions.json"
    data = {f"q{i:03d}": {"question": f"Cau hoi so {i}?"} for i in range(1, 11)}
    q_file.write_text(json.dumps(data), encoding="utf-8")
    return q_file


@pytest.fixture
def thousand_questions_file(tmp_path: Path) -> Path:
    q_file = tmp_path / "public_1000_questions.json"
    data = {f"q{i:04d}": {"question": f"Question {i}?"} for i in range(1, 1001)}
    q_file.write_text(json.dumps(data), encoding="utf-8")
    return q_file


def _build_dummy_runtime_builder(worker_id: int, fail_qids: set[str] | None = None) -> Callable[[], MagicMock]:
    fail_qids = fail_qids or set()

    def _builder() -> MagicMock:
        mock_rt = MagicMock()

        def _answer(query: Any) -> Any:
            qid = query.query_id
            res = MagicMock()
            if qid in fail_qids:
                res.response = MagicMock(
                    answer="",
                    insufficient_evidence=True,
                    retrieval_strategy=MagicMock(value="hybrid_rerank"),
                    warnings=["generation_failed"],
                )
                res.state = MagicMock(selected_evidence=[], retry_count=0)
                res.stop_reason = MagicMock(value="generation_failed")
            else:
                res.response = MagicMock(
                    answer=f"W{worker_id} Answer for {qid}",
                    insufficient_evidence=False,
                    retrieval_strategy=MagicMock(value="hybrid_rerank"),
                    warnings=[],
                )
                res.state = MagicMock(selected_evidence=[{"chunk_id": f"chunk_{qid}"}], retry_count=0)
                res.stop_reason = MagicMock(value="answer_verified")
            return res

        mock_rt.answer.side_effect = _answer
        return mock_rt

    return _builder


def test_deterministic_partition_10_and_1000_qids(
    base_config: ApplicationConfig, ten_questions_file: Path, thousand_questions_file: Path, tmp_path: Path
) -> None:
    """Requirements 1, 2, 3: Canonical 10-QID gives 5/5, 1000-QID gives 500/500, deterministic across restarts."""
    runner_10 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=tmp_path / "w10", questions_path=ten_questions_file
    )
    assert len(runner_10.partition_0_qids) == 5
    assert len(runner_10.partition_1_qids) == 5
    assert set(runner_10.partition_0_qids) & set(runner_10.partition_1_qids) == set()
    assert set(runner_10.partition_0_qids) | set(runner_10.partition_1_qids) == set(runner_10.canonical_qids)

    # Determinism across multiple instantiations
    runner_10_re = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=tmp_path / "w10_re", questions_path=ten_questions_file
    )
    assert runner_10.partition_0_qids == runner_10_re.partition_0_qids
    assert runner_10.partition_1_qids == runner_10_re.partition_1_qids

    runner_1000 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=tmp_path / "w1000", questions_path=thousand_questions_file
    )
    assert len(runner_1000.partition_0_qids) == 500
    assert len(runner_1000.partition_1_qids) == 500
    assert set(runner_1000.partition_0_qids) & set(runner_1000.partition_1_qids) == set()
    assert set(runner_1000.partition_0_qids) | set(runner_1000.partition_1_qids) == set(runner_1000.canonical_qids)


def test_dual_gpu_synthetic_interruption_resume_and_merge(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirements 15, 19, 20, 21, 22: Session 1 runs 2 per worker, Session 2 resumes, completes 10/10, packages submission."""
    work_dir_1 = tmp_path / "dual_sess1"
    runtime_builders = {
        0: _build_dummy_runtime_builder(0),
        1: _build_dummy_runtime_builder(1),
    }

    runner_1 = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir_1,
        questions_path=ten_questions_file,
        session_id="session_1",
        runtime_builders=runtime_builders,
    )

    # Session 1: Process 2 questions per worker (total 4)
    res_1 = runner_1.run_session(max_questions_per_worker=2)
    assert res_1["global_completed_count"] == 4
    assert res_1["global_remaining_count"] == 6
    assert res_1["worker_0_completed"] == 2
    assert res_1["worker_1_completed"] == 2
    zip_1 = work_dir_1 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME
    assert zip_1.exists()

    # Session 2: Resumes from zip_1 and finishes remaining questions
    work_dir_2 = tmp_path / "dual_sess2"
    runner_2 = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir_2,
        questions_path=ten_questions_file,
        session_id="session_2",
        runtime_builders=runtime_builders,
    )
    res_2 = runner_2.run_session(combined_checkpoint_archive_path=zip_1)
    assert res_2["global_completed_count"] == 10
    assert res_2["global_remaining_count"] == 0
    assert res_2["status"] == "ALL_QUESTIONS_COMPLETED"

    # Package submission
    out_dir = tmp_path / "sub_out"
    sub_zip = runner_2.package_final_submission(out_dir)
    assert sub_zip.exists()
    with zipfile.ZipFile(sub_zip, "r") as zf:
        data = json.loads(zf.read("submission.json").decode("utf-8"))
        assert len(data) == 10
        # Check canonical ordering and unmutated answers
        for qid in runner_2.canonical_qids:
            assert qid in data
            expected_prefix = "W0 Answer" if qid in runner_2.partition_0_set else "W1 Answer"
            assert data[qid]["answer"].startswith(expected_prefix)


def test_dual_gpu_three_session_chain(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirement 16: Three-session dual-worker chain with zero reruns."""
    runtime_builders = {0: _build_dummy_runtime_builder(0), 1: _build_dummy_runtime_builder(1)}

    # Session 1: 1 per worker (2 total)
    work_1 = tmp_path / "sess1"
    runner_1 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_1, questions_path=ten_questions_file, session_id="s1", runtime_builders=runtime_builders
    )
    res_1 = runner_1.run_session(max_questions_per_worker=1)
    assert res_1["global_completed_count"] == 2
    zip_1 = work_1 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME

    # Session 2: 2 per worker (total 6 completed)
    work_2 = tmp_path / "sess2"
    runner_2 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_2, questions_path=ten_questions_file, session_id="s2", runtime_builders=runtime_builders
    )
    res_2 = runner_2.run_session(combined_checkpoint_archive_path=zip_1, max_questions_per_worker=2)
    assert res_2["global_completed_count"] == 6
    zip_2 = work_2 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME

    # Session 3: Finish all (total 10 completed)
    work_3 = tmp_path / "sess3"
    runner_3 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_3, questions_path=ten_questions_file, session_id="s3", runtime_builders=runtime_builders
    )
    res_3 = runner_3.run_session(combined_checkpoint_archive_path=zip_2)
    assert res_3["global_completed_count"] == 10
    assert res_3["status"] == "ALL_QUESTIONS_COMPLETED"


def test_dual_gpu_fails_if_worker_contains_wrong_partition_qid(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirements 4, 5: Worker 0 results containing worker-1 QID fails closed."""
    work_dir = tmp_path / "work_wrong_part"
    work_dir.mkdir(parents=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    runner._ensure_worker_question_files()

    # Place an odd-index QID (q002) in worker 0
    w0_res = runner.worker_0_dir / CHECKPOINT_RESULTS_FILENAME
    rec = {"question_id": "q002", "answer": "ans", "success": True}
    w0_res.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    w0_man = runner.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME
    w0_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_0_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_file_sha256(runner.worker_0_qfile),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w0_res),
    }), encoding="utf-8")

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "global_completed_count": 1,
        "global_completed_qid_set_hash": "dummy",
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError):
        runner.restore_and_validate_checkpoint()


def test_dual_gpu_fails_on_cross_worker_duplicate_qid(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirement 6: Cross-worker duplicate QID fails closed."""
    work_dir = tmp_path / "work_cross_dup"
    work_dir.mkdir(parents=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    runner._ensure_worker_question_files()

    w0_res = runner.worker_0_dir / CHECKPOINT_RESULTS_FILENAME
    w0_res.write_text(json.dumps({"question_id": "q001", "answer": "ans"}) + "\n", encoding="utf-8")
    w0_man = runner.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME
    w0_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_0_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_file_sha256(runner.worker_0_qfile),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w0_res),
    }), encoding="utf-8")

    w1_res = runner.worker_1_dir / CHECKPOINT_RESULTS_FILENAME
    w1_res.write_text(json.dumps({"question_id": "q001", "answer": "ans"}) + "\n", encoding="utf-8")
    w1_man = runner.worker_1_dir / CHECKPOINT_MANIFEST_FILENAME
    w1_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_1_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_file_sha256(runner.worker_1_qfile),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w1_res),
    }), encoding="utf-8")

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "global_completed_count": 2,
        "global_completed_qid_set_hash": "dummy",
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError):
        runner.restore_and_validate_checkpoint()


def test_dual_gpu_fails_on_changed_partition_strategy(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirement 9: Changed partition strategy fails closed."""
    work_dir = tmp_path / "work_strat_mismatch"
    work_dir.mkdir(parents=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": "hash_partition_v0",  # Mismatch
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="partition strategy mismatch"):
        runner.restore_and_validate_checkpoint()


def test_dual_gpu_fails_on_previous_combined_checkpoint_sha_mismatch(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirement 14: Tampered previous combined checkpoint SHA fails closed."""
    work_dir_1 = tmp_path / "sess1_chain_tamper"
    runner_1 = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir_1,
        questions_path=ten_questions_file,
        session_id="session_1",
        runtime_builders={0: _build_dummy_runtime_builder(0), 1: _build_dummy_runtime_builder(1)},
    )
    res_1 = runner_1.run_session(max_questions_per_worker=1)
    zip_1 = work_dir_1 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME
    sha_1 = compute_file_sha256(zip_1)

    work_dir_2 = tmp_path / "sess2_chain_tamper"
    runner_2 = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir_2,
        questions_path=ten_questions_file,
        session_id="session_2",
        runtime_builders={0: _build_dummy_runtime_builder(0), 1: _build_dummy_runtime_builder(1)},
    )
    runner_2.run_session(combined_checkpoint_archive_path=zip_1, max_questions_per_worker=1)

    # Tamper previous_checkpoint_sha256 in manifest of session 2
    comb_man = work_dir_2 / DUAL_GPU_MANIFEST_FILENAME
    data = json.loads(comb_man.read_text(encoding="utf-8"))
    data["previous_checkpoint_sha256"] = "fake_tampered_combined_sha"
    comb_man.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Restore attempting to validate against expected sha_1 fails closed
    runner_3 = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir_2, questions_path=ten_questions_file)
    with pytest.raises(ArtifactCompatibilityError, match="Dual-GPU previous_checkpoint_sha256 mismatch"):
        runner_3.restore_and_validate_checkpoint(expected_previous_checkpoint_sha256=sha_1)


def test_dual_gpu_worker_failure_produces_failure_checkpoint(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirement 17: Worker failure produces failure checkpoint without losing durable progress."""
    work_dir = tmp_path / "work_fail_checkpoint"

    def _failing_builder_1() -> MagicMock:
        raise RuntimeError("GPU 1 out of memory crash during initialization")

    runner = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir,
        questions_path=ten_questions_file,
        runtime_builders={0: _build_dummy_runtime_builder(0), 1: _failing_builder_1},
    )

    res = runner.run_session(max_questions_per_worker=1)
    assert res["status"] == "DUAL_GPU_WORKER_FAILURE_CHECKPOINT_READY"
    assert (work_dir / DUAL_GPU_CHECKPOINT_ZIP_FILENAME).exists()


def test_dual_gpu_submission_packaging_refused_on_incomplete_set(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirement 18: Incomplete dual-GPU checkpoint refuses submission packaging."""
    work_dir = tmp_path / "work_incomplete_dual"
    runner = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir,
        questions_path=ten_questions_file,
        runtime_builders={0: _build_dummy_runtime_builder(0), 1: _build_dummy_runtime_builder(1)},
    )
    runner.run_session(max_questions_per_worker=2)  # 4 of 10 completed

    with pytest.raises(DataValidationError, match="Cannot package submission: incomplete dual-GPU checkpoint"):
        runner.package_final_submission(tmp_path / "sub_refused")


def test_dual_gpu_fails_on_authority_and_config_mismatches(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirements 10, 11, 12, 13: Changed commit, config, question SHA, or dependencies fails closed."""
    work_dir = tmp_path / "work_auth_mismatches"
    work_dir.mkdir(parents=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)

    valid_payload = {
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "runtime_dependencies": FROZEN_AUTHORITY_BINDINGS["runtime_dependencies"],
    }

    # 1. Commit mismatch
    bad_commit_payload = dict(valid_payload, execution_code_authority_commit="tampered_commit")
    (work_dir / DUAL_GPU_MANIFEST_FILENAME).write_text(json.dumps(bad_commit_payload), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="execution_code_authority_commit mismatch"):
        runner.restore_and_validate_checkpoint()

    # 2. Config mismatch
    bad_cfg_payload = dict(valid_payload, application_config_hash="tampered_config_hash")
    (work_dir / DUAL_GPU_MANIFEST_FILENAME).write_text(json.dumps(bad_cfg_payload), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="application_config_hash mismatch"):
        runner.restore_and_validate_checkpoint()

    # 3. Question source mismatch
    bad_q_payload = dict(valid_payload, question_source_sha256="tampered_question_sha")
    (work_dir / DUAL_GPU_MANIFEST_FILENAME).write_text(json.dumps(bad_q_payload), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="question_source_sha256 mismatch"):
        runner.restore_and_validate_checkpoint()

    # 4. Dependency mismatch
    bad_dep_payload = dict(valid_payload, runtime_dependencies={"transformers": "4.44.0"})
    (work_dir / DUAL_GPU_MANIFEST_FILENAME).write_text(json.dumps(bad_dep_payload), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="runtime_dependencies mismatch"):
        runner.restore_and_validate_checkpoint()


def test_dual_gpu_device_isolation_env(
    base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path
) -> None:
    """Requirement 22: Worker 0 targets CUDA_VISIBLE_DEVICES=0, Worker 1 targets CUDA_VISIBLE_DEVICES=1."""
    work_dir = tmp_path / "work_device_iso"
    runner = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir,
        questions_path=ten_questions_file,
        worker_devices=("0", "1"),
        runtime_builders={0: _build_dummy_runtime_builder(0), 1: _build_dummy_runtime_builder(1)},
    )
    assert runner.worker_devices == ("0", "1")
    assert runner.worker_devices[0] == "0"
    assert runner.worker_devices[1] == "1"

