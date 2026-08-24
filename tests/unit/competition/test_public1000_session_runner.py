"""Comprehensive unit and integration tests for Public1000SessionRunner multi-session crash-safe execution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import zipfile

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.public1000_session_runner import (
    CHECKPOINT_LATEST_ZIP_FILENAME,
    CHECKPOINT_MANIFEST_FILENAME,
    CHECKPOINT_RESULTS_FILENAME,
    Public1000SessionRunner,
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


def _build_dummy_runtime(fail_qids: set[str] | None = None) -> MagicMock:
    fail_qids = fail_qids or set()
    mock_rt = MagicMock()

    def _answer(query: Any) -> Any:
        qid = query.query_id
        res = MagicMock()
        if qid in fail_qids:
            res.response = MagicMock(answer="", insufficient_evidence=True, retrieval_strategy=MagicMock(value="hybrid_rerank"), warnings=["generation_failed"])
            res.state = MagicMock(selected_evidence=[], retry_count=0)
            res.stop_reason = MagicMock(value="generation_failed")
        else:
            res.response = MagicMock(answer=f"Answer for {qid}", insufficient_evidence=False, retrieval_strategy=MagicMock(value="hybrid_rerank"), warnings=[])
            res.state = MagicMock(selected_evidence=[{"chunk_id": f"chunk_{qid}"}], retry_count=0)
            res.stop_reason = MagicMock(value="answer_verified")
        return res

    mock_rt.answer.side_effect = _answer
    return mock_rt


def test_public1000_session_interrupted_and_resumed(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Tests 1, 2, 11, 12: Session 1 runs 3 QIDs, Session 2 resumes from checkpoint and finishes remaining QIDs."""
    work_dir_1 = tmp_path / "session_1_work"
    runner_1 = Public1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir_1,
        questions_path=ten_questions_file,
        session_id="session_1",
    )
    runner_1.runtime = _build_dummy_runtime()

    # Run max 3 questions in Session 1
    res_1 = runner_1.run_session(max_questions_in_session=3)
    assert res_1["completed_count"] == 3
    assert res_1["remaining_count"] == 7
    assert (work_dir_1 / CHECKPOINT_RESULTS_FILENAME).exists()
    assert (work_dir_1 / CHECKPOINT_LATEST_ZIP_FILENAME).exists()

    checkpoint_zip_1 = work_dir_1 / CHECKPOINT_LATEST_ZIP_FILENAME

    # Session 2 resumes from checkpoint zip
    work_dir_2 = tmp_path / "session_2_work"
    runner_2 = Public1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir_2,
        questions_path=ten_questions_file,
        session_id="session_2",
    )
    mock_rt_2 = _build_dummy_runtime()
    runner_2.runtime = mock_rt_2

    res_2 = runner_2.run_session(checkpoint_archive_path=checkpoint_zip_1)
    assert res_2["completed_count"] == 10
    assert res_2["remaining_count"] == 0
    assert res_2["status"] == "ALL_QUESTIONS_COMPLETED"

    # Verify runtime 2 only answered the remaining 7 questions (q004 to q010)
    answered_queries = [call[0][0].query_id for call in mock_rt_2.answer.call_args_list]
    assert len(answered_queries) == 7
    assert answered_queries == [f"q{i:03d}" for i in range(4, 11)]

    # Packaging submission succeeds
    sub_zip = runner_2.package_final_submission(tmp_path / "submission_out")
    assert sub_zip.exists()
    with zipfile.ZipFile(sub_zip, "r") as zf:
        assert "submission.json" in zf.namelist()
        sub_data = json.loads(zf.read("submission.json").decode("utf-8"))
        assert len(sub_data) == 10
        assert set(sub_data.keys()) == {f"q{i:03d}" for i in range(1, 11)}


def test_public1000_fails_on_duplicate_checkpoint_qid(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Test 3: Duplicate QID in checkpoint results fails closed."""
    work_dir = tmp_path / "work_dup"
    work_dir.mkdir(parents=True)

    results_file = work_dir / CHECKPOINT_RESULTS_FILENAME
    rec1 = {"question_id": "q001", "answer": "ans1", "success": True}
    rec2 = {"question_id": "q001", "answer": "ans2", "success": True}  # duplicate
    results_file.write_text(f"{json.dumps(rec1)}\n{json.dumps(rec2)}\n", encoding="utf-8")

    manifest_file = work_dir / CHECKPOINT_MANIFEST_FILENAME
    manifest = {
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).expected_qid_set_hash,
        "completed_question_count": 2,
        "results_jsonl_sha256": compute_file_sha256(results_file),
    }
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    runner = Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    with pytest.raises(ArtifactCompatibilityError, match="Duplicate question ID"):
        runner.restore_and_validate_checkpoint()


def test_public1000_fails_on_corrupt_or_truncated_jsonl(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Test 4: Malformed/truncated JSONL line fails closed."""
    work_dir = tmp_path / "work_corrupt"
    work_dir.mkdir(parents=True)

    results_file = work_dir / CHECKPOINT_RESULTS_FILENAME
    results_file.write_text('{"question_id": "q001", "answer": "ans1"}\n{"question_id": "q002", "ans', encoding="utf-8")

    manifest_file = work_dir / CHECKPOINT_MANIFEST_FILENAME
    manifest = {
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).expected_qid_set_hash,
        "completed_question_count": 2,
        "results_jsonl_sha256": compute_file_sha256(results_file),
    }
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    runner = Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    with pytest.raises(ArtifactCompatibilityError, match="Corrupt JSONL in checkpoint"):
        runner.restore_and_validate_checkpoint()


def test_public1000_fails_on_config_sha_mismatch(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Test 5: Changed configuration hash fails closed."""
    work_dir = tmp_path / "work_cfg_mismatch"
    work_dir.mkdir(parents=True)

    results_file = work_dir / CHECKPOINT_RESULTS_FILENAME
    rec1 = {"question_id": "q001", "answer": "ans1"}
    results_file.write_text(f"{json.dumps(rec1)}\n", encoding="utf-8")

    manifest_file = work_dir / CHECKPOINT_MANIFEST_FILENAME
    manifest = {
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": "different_cfg_sha_12345",
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).expected_qid_set_hash,
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(results_file),
    }
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    runner = Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    with pytest.raises(ArtifactCompatibilityError, match="Checkpoint application_config_hash does not match"):
        runner.restore_and_validate_checkpoint()


def test_public1000_fails_on_question_input_sha_mismatch(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Test 7: Changed question input SHA fails closed."""
    work_dir = tmp_path / "work_q_mismatch"
    work_dir.mkdir(parents=True)

    results_file = work_dir / CHECKPOINT_RESULTS_FILENAME
    rec1 = {"question_id": "q001", "answer": "ans1"}
    results_file.write_text(f"{json.dumps(rec1)}\n", encoding="utf-8")

    manifest_file = work_dir / CHECKPOINT_MANIFEST_FILENAME
    manifest = {
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).config_hash,
        "question_source_sha256": "different_questions_sha_67890",
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).expected_qid_set_hash,
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(results_file),
    }
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    runner = Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    with pytest.raises(ArtifactCompatibilityError, match="Checkpoint question_source_sha256 does not match"):
        runner.restore_and_validate_checkpoint()


def test_public1000_fails_on_unexpected_qid(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Test 9: Unexpected QID outside canonical input fails closed."""
    work_dir = tmp_path / "work_unexp_qid"
    work_dir.mkdir(parents=True)

    results_file = work_dir / CHECKPOINT_RESULTS_FILENAME
    rec1 = {"question_id": "unrecognized_qid_999", "answer": "ans"}
    results_file.write_text(f"{json.dumps(rec1)}\n", encoding="utf-8")

    manifest_file = work_dir / CHECKPOINT_MANIFEST_FILENAME
    manifest = {
        "schema_version": "4.0.0",
        "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
        "application_config_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).config_hash,
        "question_source_sha256": compute_file_sha256(ten_questions_file),
        "expected_total_qid_count": 10,
        "expected_complete_qid_set_hash": Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file).expected_qid_set_hash,
        "completed_question_count": 1,
        "results_jsonl_sha256": compute_file_sha256(results_file),
    }
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    runner = Public1000SessionRunner(app_config=base_config, working_dir=work_dir, questions_path=ten_questions_file)
    with pytest.raises(ArtifactCompatibilityError, match="Unexpected question ID"):
        runner.restore_and_validate_checkpoint()


def test_public1000_clean_session_budget_stop(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Test 10: Clean session-budget deadline stop produces export zip and returns SESSION_CHECKPOINT_COMPLETE."""
    work_dir = tmp_path / "work_budget"
    # Zero session budget force-triggers immediate budget stop after 0 or 1 question
    runner = Public1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir,
        questions_path=ten_questions_file,
        session_budget_hours=0.00001,  # ~0.036 seconds
    )
    runner.runtime = _build_dummy_runtime()

    res = runner.run_session()
    assert res["status"] == "SESSION_CHECKPOINT_COMPLETE"
    assert (work_dir / CHECKPOINT_LATEST_ZIP_FILENAME).exists()


def test_public1000_submission_packaging_refused_on_incomplete_checkpoint(base_config: ApplicationConfig, ten_questions_file: Path, tmp_path: Path) -> None:
    """Test 13: Submission packaging is strictly refused if completed records < expected (e.g. 9/10)."""
    work_dir = tmp_path / "work_incomplete"
    runner = Public1000SessionRunner(
        app_config=base_config,
        working_dir=work_dir,
        questions_path=ten_questions_file,
    )
    runner.runtime = _build_dummy_runtime()

    # Process only 9 of 10 questions
    runner.run_session(max_questions_in_session=9)

    with pytest.raises(DataValidationError, match="Cannot package submission: incomplete checkpoint"):
        runner.package_final_submission(tmp_path / "sub_refused")
