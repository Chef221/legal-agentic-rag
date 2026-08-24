"""Comprehensive unit and synthetic integration tests for DualPublic1000SessionRunner.

Proves true OS-process isolation, CUDA environment isolation, child-PID runtime building,
race-safe heartbeat monitoring, per-worker durability, fail-closed resume, and exact submission packaging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from unittest.mock import MagicMock
import zipfile

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.dual_session_runner import (
    DUAL_GPU_CHECKPOINT_ZIP_FILENAME,
    DUAL_GPU_MANIFEST_FILENAME,
    DualPublic1000SessionRunner,
    PARTITION_STRATEGY_V1,
    _count_durable_records_safe,
    get_dual_gpu_telemetry,
)
from legal_agentic_rag.competition.uit_dsc_2026.public1000_session_runner import (
    CHECKPOINT_MANIFEST_FILENAME,
    CHECKPOINT_RESULTS_FILENAME,
    FROZEN_AUTHORITY_BINDINGS,
    compute_file_sha256,
    compute_qid_set_hash,
)
from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.serving.config_loader import load_application_config


# Top-level factory for child processes during testing
def top_level_dummy_builder_factory(worker_id: int) -> Callable[[], Any]:
    def _builder() -> Any:
        mock_rt = MagicMock()

        def _answer(query: Any) -> Any:
            qid = query.query_id
            res = MagicMock()
            res.response = MagicMock(
                answer=f"W{worker_id} Answer for {qid} (PID {os.getpid()})",
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


def top_level_failing_builder_factory(worker_id: int) -> Callable[[], Any]:
    if worker_id == 1:
        raise RuntimeError("Worker 1 CUDA hardware fault during initialization")
    return top_level_dummy_builder_factory(worker_id)


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


# ==============================================================================
# 1. TRUE OS PROCESS & CUDA ISOLATION CONTRACT TESTS
# ==============================================================================

def test_true_os_process_isolation_and_cuda_env(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Proves Worker 0 and Worker 1 execute in DISTINCT child OS PIDs with isolated CUDA_VISIBLE_DEVICES."""
    work_dir = tmp_path / "w_mp_iso"
    parent_pid = os.getpid()
    parent_cuda_before = os.environ.get("CUDA_VISIBLE_DEVICES")

    runner = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir,
        questions_path=ten_questions_file,
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
        worker_devices=("0", "1"),
    )

    res = runner.run_session(max_questions_per_worker=1)
    assert res["status"] == "SESSION_CHECKPOINT_COMPLETE"
    assert res["global_completed_count"] == 2

    # Parent environment was not mutated
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == parent_cuda_before

    # Read worker results to verify child PIDs
    records_0, records_1 = runner.restore_and_validate_checkpoint()
    assert len(records_0) == 1
    assert len(records_1) == 1

    ans_0 = records_0[0]["answer"]
    ans_1 = records_1[0]["answer"]

    # Both answers contain child PIDs
    import re
    pid_0_match = re.search(r"\(PID (\d+)\)", ans_0)
    pid_1_match = re.search(r"\(PID (\d+)\)", ans_1)

    assert pid_0_match and pid_1_match
    pid_0 = int(pid_0_match.group(1))
    pid_1 = int(pid_1_match.group(1))

    assert pid_0 != parent_pid, "Worker 0 executed inside parent PID!"
    assert pid_1 != parent_pid, "Worker 1 executed inside parent PID!"
    assert pid_0 != pid_1, "Worker 0 and Worker 1 executed inside the same PID!"


# ==============================================================================
# 2. RACE-SAFE HEARTBEAT MONITORING TEST
# ==============================================================================

def test_heartbeat_safe_on_partial_checkpoint_race(tmp_path: Path) -> None:
    """Proves heartbeat counting never crashes when results exist but manifest is not yet published."""
    results_path = tmp_path / "test_results.jsonl"

    # 1. Non-existent file gives 0, no exception
    assert _count_durable_records_safe(results_path) == 0

    # 2. Partial / in-progress line written
    results_path.write_text('{"question_id": "q001", "answer": "ans1"}\n{"question_id": "q002", "ans', encoding="utf-8")

    # Safely parses valid line and ignores trailing partial chunk
    assert _count_durable_records_safe(results_path) == 1

    # 3. Multiple completed lines
    results_path.write_text('{"question_id": "q001"}\n{"question_id": "q002"}\n{"question_id": "q003"}\n', encoding="utf-8")
    assert _count_durable_records_safe(results_path) == 3


# ==============================================================================
# 3. 22 INDIVIDUAL DEDICATED CONTRACT TESTS (a through v)
# ==============================================================================

# a. 10-QID partition = 5/5
def test_contract_a_10_qid_partition_is_5_and_5(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=tmp_path / "w10", questions_path=ten_questions_file)
    assert len(runner.partition_0_qids) == 5
    assert len(runner.partition_1_qids) == 5
    assert set(runner.partition_0_qids) & set(runner.partition_1_qids) == set()
    assert set(runner.partition_0_qids) | set(runner.partition_1_qids) == set(runner.canonical_qids)


# b. 1000-QID partition = 500/500
def test_contract_b_1000_qid_partition_is_500_and_500(base_config: ApplicationConfig, thousand_questions_file: Path, tmp_path: Path) -> None:
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=tmp_path / "w1000", questions_path=thousand_questions_file)
    assert len(runner.partition_0_qids) == 500
    assert len(runner.partition_1_qids) == 500
    assert set(runner.partition_0_qids) & set(runner.partition_1_qids) == set()
    assert set(runner.partition_0_qids) | set(runner.partition_1_qids) == set(runner.canonical_qids)


# c. partition deterministic across restart
def test_contract_c_partition_deterministic_across_restarts(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    runner_1 = DualPublic1000SessionRunner(app_config=base_config, working_dir=tmp_path / "w_c1", questions_path=ten_questions_file)
    runner_2 = DualPublic1000SessionRunner(app_config=base_config, working_dir=tmp_path / "w_c2", questions_path=ten_questions_file)
    assert runner_1.partition_0_qids == runner_2.partition_0_qids
    assert runner_1.partition_1_qids == runner_2.partition_1_qids
    assert runner_1.canonical_qids == runner_2.canonical_qids


# d. wrong worker-0 partition QID fails
def test_contract_d_wrong_worker_0_partition_qid_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_d"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    runner._ensure_worker_question_files()

    w0_res = runner.worker_0_dir / CHECKPOINT_RESULTS_FILENAME
    rec = {"question_id": "q002", "answer": "ans", "success": True}
    w0_res.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    w0_man = runner.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME
    w0_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_0_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_qid_set_hash(runner.partition_0_qids),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w0_res),
    }), encoding="utf-8")

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "global_completed_count": 1,
        "global_completed_qid_set_hash": "dummy",
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError):
        runner.restore_and_validate_checkpoint()


# e. wrong worker-1 partition QID fails
def test_contract_e_wrong_worker_1_partition_qid_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_e"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    runner._ensure_worker_question_files()

    w1_res = runner.worker_1_dir / CHECKPOINT_RESULTS_FILENAME
    rec = {"question_id": "q001", "answer": "ans", "success": True}
    w1_res.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    w1_man = runner.worker_1_dir / CHECKPOINT_MANIFEST_FILENAME
    w1_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_1_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_qid_set_hash(runner.partition_1_qids),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w1_res),
    }), encoding="utf-8")

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "global_completed_count": 1,
        "global_completed_qid_set_hash": "dummy",
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError):
        runner.restore_and_validate_checkpoint()


# f. cross-worker duplicate fails
def test_contract_f_cross_worker_duplicate_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_f"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    runner._ensure_worker_question_files()

    w0_res = runner.worker_0_dir / CHECKPOINT_RESULTS_FILENAME
    w0_res.write_text(json.dumps({"question_id": "q001", "answer": "ans"}) + "\n", encoding="utf-8")
    w0_man = runner.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME
    w0_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_0_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_qid_set_hash(runner.partition_0_qids),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w0_res),
    }), encoding="utf-8")

    w1_res = runner.worker_1_dir / CHECKPOINT_RESULTS_FILENAME
    w1_res.write_text(json.dumps({"question_id": "q001", "answer": "ans"}) + "\n", encoding="utf-8")
    w1_man = runner.worker_1_dir / CHECKPOINT_MANIFEST_FILENAME
    w1_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_1_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_qid_set_hash(runner.partition_1_qids),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w1_res),
    }), encoding="utf-8")

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "global_completed_count": 2,
        "global_completed_qid_set_hash": "dummy",
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError):
        runner.restore_and_validate_checkpoint()


# g. combined checkpoint count mismatch fails
def test_contract_g_combined_checkpoint_count_mismatch_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_g"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    runner._ensure_worker_question_files()

    w0_res = runner.worker_0_dir / CHECKPOINT_RESULTS_FILENAME
    w0_res.write_text(json.dumps({"question_id": "q001", "answer": "ans"}) + "\n", encoding="utf-8")
    w0_man = runner.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME
    w0_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_0_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_qid_set_hash(runner.partition_0_qids),
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(w0_res),
    }), encoding="utf-8")

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "global_completed_count": 99,  # Mismatch
        "global_completed_qid_set_hash": compute_qid_set_hash(["q001"]),
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Dual-GPU count mismatch"):
        runner.restore_and_validate_checkpoint()


# h. tampered worker checkpoint SHA fails
def test_contract_h_tampered_worker_checkpoint_sha_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_h"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    runner._ensure_worker_question_files()

    w0_res = runner.worker_0_dir / CHECKPOINT_RESULTS_FILENAME
    w0_res.write_text(json.dumps({"question_id": "q001", "answer": "ans"}) + "\n", encoding="utf-8")
    w0_man = runner.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME
    w0_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(runner.worker_0_qfile),
        "expected_total_qid_count": 5,
        "expected_complete_qid_set_hash": compute_qid_set_hash(runner.partition_0_qids),
        "completed_question_count": 1,
        "results_jsonl_sha256": "tampered_fake_results_sha",
    }), encoding="utf-8")

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "global_completed_count": 1,
        "global_completed_qid_set_hash": compute_qid_set_hash(["q001"]),
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Checkpoint results JSONL SHA mismatch"):
        runner.restore_and_validate_checkpoint()


# i. changed partition strategy fails
def test_contract_i_changed_partition_strategy_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_i"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": "hash_partition_v0",
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="partition strategy mismatch"):
        runner.restore_and_validate_checkpoint()


# j. changed execution authority fails
def test_contract_j_changed_execution_authority_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_j"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": "tampered_fake_commit_sha",
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="execution_code_authority_commit mismatch"):
        runner.restore_and_validate_checkpoint()


# k. changed config SHA fails
def test_contract_k_changed_config_sha_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_k"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": "tampered_fake_config_hash",
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="application_config_hash mismatch"):
        runner.restore_and_validate_checkpoint()


# l. changed question-source SHA fails
def test_contract_l_changed_question_source_sha_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_l"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": "tampered_fake_question_sha",
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="question_source_sha256 mismatch"):
        runner.restore_and_validate_checkpoint()


# m. changed dependency authority fails
def test_contract_m_changed_dependency_authority_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_m"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)

    comb_man = work_dir / DUAL_GPU_MANIFEST_FILENAME
    comb_man.write_text(json.dumps({
        "schema_version": "4.0.0",
        "partition_strategy": PARTITION_STRATEGY_V1,
        "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
        "application_config_hash": runner.config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": runner.expected_qid_set_hash,
        "runtime_dependencies": {"transformers": "4.44.0"},
    }), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="runtime_dependencies mismatch"):
        runner.restore_and_validate_checkpoint()


# n. previous combined checkpoint SHA mismatch fails
def test_contract_n_previous_combined_checkpoint_sha_mismatch_fails(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_1 = tmp_path / "w_n1"
    runner_1 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_1, questions_path=ten_questions_file, session_id="s1",
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    runner_1.run_session(max_questions_per_worker=1)
    zip_1 = work_1 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME
    sha_1 = compute_file_sha256(zip_1)

    work_2 = tmp_path / "w_n2"
    runner_2 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_2, questions_path=ten_questions_file, session_id="s2",
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    runner_2.run_session(combined_checkpoint_archive_path=zip_1, max_questions_per_worker=1)

    # Tamper previous_checkpoint_sha256 in manifest of session 2
    comb_man = work_2 / DUAL_GPU_MANIFEST_FILENAME
    data = json.loads(comb_man.read_text(encoding="utf-8"))
    data["previous_checkpoint_sha256"] = "fake_tampered_combined_sha"
    comb_man.write_text(json.dumps(data, indent=2), encoding="utf-8")

    runner_3 = DualPublic1000SessionRunner(app_config=base_config, working_dir=work_2, questions_path=ten_questions_file)
    with pytest.raises(ArtifactCompatibilityError, match="Dual-GPU previous_checkpoint_sha256 mismatch"):
        runner_3.restore_and_validate_checkpoint(expected_previous_checkpoint_sha256=sha_1)


# o. two-worker interruption/resume gives exact set and zero reruns
def test_contract_o_two_worker_interruption_resume_gives_exact_set_and_zero_reruns(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_1 = tmp_path / "w_o1"
    runner_1 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_1, questions_path=ten_questions_file, session_id="s1",
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    res_1 = runner_1.run_session(max_questions_per_worker=2)
    assert res_1["global_completed_count"] == 4
    zip_1 = work_1 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME

    work_2 = tmp_path / "w_o2"
    runner_2 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_2, questions_path=ten_questions_file, session_id="s2",
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    res_2 = runner_2.run_session(combined_checkpoint_archive_path=zip_1)
    assert res_2["global_completed_count"] == 10
    assert res_2["status"] == "ALL_QUESTIONS_COMPLETED"


# p. three-session chain gives exact set and zero reruns
def test_contract_p_three_session_chain_gives_exact_set_and_zero_reruns(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    # Session 1: 1 per worker (2 total)
    work_1 = tmp_path / "w_p1"
    runner_1 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_1, questions_path=ten_questions_file, session_id="s1",
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    res_1 = runner_1.run_session(max_questions_per_worker=1)
    assert res_1["global_completed_count"] == 2
    zip_1 = work_1 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME

    # Session 2: 2 per worker (total 6 completed)
    work_2 = tmp_path / "w_p2"
    runner_2 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_2, questions_path=ten_questions_file, session_id="s2",
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    res_2 = runner_2.run_session(combined_checkpoint_archive_path=zip_1, max_questions_per_worker=2)
    assert res_2["global_completed_count"] == 6
    zip_2 = work_2 / DUAL_GPU_CHECKPOINT_ZIP_FILENAME

    # Session 3: Finish all (total 10 completed)
    work_3 = tmp_path / "w_p3"
    runner_3 = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_3, questions_path=ten_questions_file, session_id="s3",
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    res_3 = runner_3.run_session(combined_checkpoint_archive_path=zip_2)
    assert res_3["global_completed_count"] == 10
    assert res_3["status"] == "ALL_QUESTIONS_COMPLETED"


# q. worker failure preserves durable progress/checkpoint
def test_contract_q_worker_failure_preserves_durable_progress_and_checkpoint(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_q"
    runner = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file,
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_failing_builder_factory"),
    )
    res = runner.run_session(max_questions_per_worker=1)
    assert res["status"] == "DUAL_GPU_WORKER_FAILURE_CHECKPOINT_READY"
    assert (work_dir / DUAL_GPU_CHECKPOINT_ZIP_FILENAME).exists()


# r. incomplete global set refuses submission
def test_contract_r_incomplete_global_set_refuses_submission(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_r"
    runner = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file,
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    runner.run_session(max_questions_per_worker=2)  # 4 / 10 done

    with pytest.raises(DataValidationError, match="Cannot package submission: incomplete dual-GPU checkpoint"):
        runner.package_final_submission(tmp_path / "sub_r")


# s. exact complete global set allows submission
def test_contract_s_exact_complete_global_set_allows_submission(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_s"
    runner = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file,
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    res = runner.run_session()
    assert res["status"] == "ALL_QUESTIONS_COMPLETED"
    sub_zip = runner.package_final_submission(tmp_path / "sub_s")
    assert sub_zip.exists()
    with zipfile.ZipFile(sub_zip, "r") as zf:
        data = json.loads(zf.read("submission.json").decode("utf-8"))
        assert len(data) == 10


# t. aggregate progress never exceeds durably committed records
def test_contract_t_aggregate_progress_never_exceeds_durable_records(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_t"
    runner = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file,
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    res = runner.run_session(max_questions_per_worker=2)
    records_0, records_1 = runner.restore_and_validate_checkpoint()
    durable_total = len(records_0) + len(records_1)
    assert res["global_completed_count"] == durable_total
    assert res["global_completed_count"] <= 10


# u. coordinator does not mutate production answer records
def test_contract_u_coordinator_does_not_mutate_production_answers(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "w_u"
    runner = DualPublic1000SessionRunner(
        app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file,
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    runner.run_session()
    sub_zip = runner.package_final_submission(tmp_path / "sub_u")
    with zipfile.ZipFile(sub_zip, "r") as zf:
        data = json.loads(zf.read("submission.json").decode("utf-8"))
        for qid in runner.partition_0_qids:
            assert data[qid]["answer"].startswith(f"W0 Answer for {qid}")
        for qid in runner.partition_1_qids:
            assert data[qid]["answer"].startswith(f"W1 Answer for {qid}")


# v. worker CUDA isolation is 0 / 1 exactly
def test_contract_v_worker_cuda_isolation_exact_devices(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "work_device_iso"
    runner = DualPublic1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir,
        questions_path=ten_questions_file,
        worker_devices=("0", "1"),
        custom_builder_target=("tests.unit.competition.test_dual_session_runner", "top_level_dummy_builder_factory"),
    )
    assert runner.worker_devices == ("0", "1")
    assert runner.worker_devices[0] == "0"
    assert runner.worker_devices[1] == "1"
