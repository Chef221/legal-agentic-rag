"""Comprehensive unit tests for M49.1-JINA35 mechanical validation script gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.m491_jina35_mechanical_validation import (
    FROZEN_AUTHORITIES,
    run_gate_a_parity,
    run_gate_b_smoke,
)


def _build_synthetic_authority(authority_dir: Path, num_qids: int = 100, num_candidates: int = 40) -> None:
    authority_dir.mkdir(parents=True, exist_ok=True)
    pool_rows = []
    jina_rows = []

    for q_idx in range(1, num_qids + 1):
        qid = f"Q{q_idx:04d}"
        cands = [
            {
                "chunk_id": f"chunk_{q_idx}_{c_idx}",
                "document_id": f"doc_{q_idx}",
                "rank": c_idx,
                "score": float(100 - c_idx),
                "strategy": "hybrid",
                "text": f"text for chunk {c_idx}",
                "metadata": {"article_number": str(c_idx), "document_title": f"Law {q_idx}"},
                "retrieval_trace": {"rrf_score": float(100 - c_idx)},
            }
            for c_idx in range(1, num_candidates + 1)
        ]
        pool_rows.append({
            "question_id": qid,
            "question": f"Question text for {qid}",
            "candidate_hits": cands,
        })
        jina_rows.append({
            "question_id": qid,
            "reranked_hits": [
                {
                    **c,
                    "score": float(100 - c["rank"]) / 100.0,
                    "retrieval_trace": {"reranker_score": float(100 - c["rank"]) / 100.0},
                }
                for c in cands
            ],
        })

    pools_file = authority_dir / "clean100_shared_candidate_pools.jsonl"
    with open(pools_file, "w", encoding="utf-8") as f:
        for r in pool_rows:
            f.write(json.dumps(r) + "\n")

    jina_file = authority_dir / "clean100_jina_reranked.jsonl"
    with open(jina_file, "w", encoding="utf-8") as f:
        for r in jina_rows:
            f.write(json.dumps(r) + "\n")

    manifest_file = authority_dir / "clean100_phase1_manifest.json"
    manifest_file.write_text(json.dumps({"status": "PHASE_1_FROZEN"}), encoding="utf-8")


def test_gate_a_positive_synthetic_full_k_pass(tmp_path: Path) -> None:
    """Test A & B: Positive full 40-candidate Gate A passes with exact top_k=40 parity call."""
    authority_dir = tmp_path / "authority"
    _build_synthetic_authority(authority_dir, num_qids=100, num_candidates=40)

    pools_sha = hashlib.sha256((authority_dir / "clean100_shared_candidate_pools.jsonl").read_bytes()).hexdigest()
    jina_sha = hashlib.sha256((authority_dir / "clean100_jina_reranked.jsonl").read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((authority_dir / "clean100_phase1_manifest.json").read_bytes()).hexdigest()

    mock_p = MagicMock()
    mock_p.numel.return_value = 596836352
    mock_p.device = "cpu"

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model._ensure_tokenizer = MagicMock()
    mock_model._tokenizer = MagicMock(model_max_length=12288)
    mock_model.parameters.side_effect = lambda: iter([mock_p])
    mock_model.rerank.side_effect = lambda query, documents, top_n, return_embeddings: [
        {"index": idx, "relevance_score": float(100 - (idx + 1)) / 100.0}
        for idx in range(len(documents))
    ]

    with patch.dict(
        FROZEN_AUTHORITIES,
        {
            "clean100_shared_candidate_pools_sha256": pools_sha,
            "clean100_jina_reranked_sha256": jina_sha,
            "clean100_phase1_manifest_sha256": manifest_sha,
        },
    ):
        with patch("transformers.AutoModel.from_pretrained", return_value=mock_model):

            result = run_gate_a_parity(
                authority_dir=authority_dir,
                output_dir=tmp_path / "out",
                device="cpu",
            )

    assert result["passed"] is True
    assert result["status"] == "GATE_A_PASSED"
    assert result["top1_exact"] == 100
    assert result["top10_ordered_exact"] == 100
    assert result["full_k_ordered_exact"] == 100
    assert result["max_abs_score_diff"] < 1e-6


def test_gate_a_fails_if_manifest_missing_or_mismatch(tmp_path: Path) -> None:
    """Test H & I: Missing or mismatched manifest SHA raises error."""
    authority_dir = tmp_path / "auth_manifest"
    _build_synthetic_authority(authority_dir, num_qids=100, num_candidates=40)

    # Missing manifest
    (authority_dir / "clean100_phase1_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="Missing Phase-1 manifest"):
        run_gate_a_parity(authority_dir=authority_dir, output_dir=tmp_path / "out", device="cpu")


def test_gate_a_strict_validation_failures(tmp_path: Path) -> None:
    """Test J, K, L, M, N, O: Strict failure on invalid question_id, empty question, wrong counts, chunk mismatch."""
    authority_dir = tmp_path / "auth_strict"
    authority_dir.mkdir()

    # Empty question
    pools_file = authority_dir / "clean100_shared_candidate_pools.jsonl"
    pools_file.write_text(json.dumps({"question_id": "Q1", "question": "", "candidate_hits": []}) + "\n", encoding="utf-8")
    jina_file = authority_dir / "clean100_jina_reranked.jsonl"
    jina_file.write_text(json.dumps({"question_id": "Q1", "reranked_hits": []}) + "\n", encoding="utf-8")
    manifest_file = authority_dir / "clean100_phase1_manifest.json"
    manifest_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA mismatch|Invalid/missing question"):
        run_gate_a_parity(authority_dir=authority_dir, output_dir=tmp_path / "out", device="cpu")


def test_gate_b_reads_agent_run_result_telemetry(tmp_path: Path) -> None:
    """Test U & V: Gate B uses RetrievalQuery and reads AgentRunResult response and state attributes."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(Path(r"C:\legal-agentic-rag-m491-jina35\configs\uit-dsc-2026-task2-m491-jina35.example.json").read_text(encoding="utf-8"), encoding="utf-8")

    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps({"30883": {"question": "Dieu kien thanh lap doanh nghiep?"}}), encoding="utf-8")

    class _MockAnswerResp:
        answer = "Day la cau tra loi phap ly day du."
        insufficient_evidence = False
        retrieval_strategy = MagicMock(value="hybrid_rerank")
        warnings = ["warning_1"]

    class _MockAgentState:
        selected_evidence = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        retry_count = 1

    class _MockRunResult:
        response = _MockAnswerResp()
        state = _MockAgentState()
        stop_reason = MagicMock(value="answer_verified")

    mock_runtime = MagicMock()
    mock_runtime.answer.return_value = _MockRunResult()

    mock_factory = MagicMock()
    mock_factory._reranker = MagicMock(_actual_device="cpu", _actual_parameter_count=596836352)
    mock_factory.build.return_value = mock_runtime

    with patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory", return_value=mock_factory):
        res = run_gate_b_smoke(
            config_path=cfg_file,
            questions_path=questions_file,
            output_dir=tmp_path / "out_b",
            device="cpu",
            max_questions=1,
        )

    assert res["passed"] is True
    assert res["status"] == "GATE_B_PASSED"
    exec_0 = res["executions"][0]
    assert exec_0["call_success"] is True
    assert exec_0["generation_success"] is True
    assert exec_0["verified_answer_success"] is True
    assert exec_0["success"] is True
    assert exec_0["answer_length"] == len(_MockAnswerResp.answer)
    assert exec_0["selected_evidence_count"] == 2
    assert exec_0["stop_reason"] == "answer_verified"
    assert exec_0["retrieval_strategy"] == "hybrid_rerank"
    assert exec_0["retry_count"] == 1
    assert exec_0["warning_count"] == 1


def test_gate_b_fails_if_generation_failed_fallback(tmp_path: Path) -> None:
    """BUG 1 REGRESSION: Gate B fails when runtime.answer() returns safely with stop_reason=generation_failed."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(Path(r"C:\legal-agentic-rag-m491-jina35\configs\uit-dsc-2026-task2-m491-jina35.example.json").read_text(encoding="utf-8"), encoding="utf-8")

    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps({"30883": {"question": "Cau hoi?"}}), encoding="utf-8")

    class _MockFallbackResp:
        answer = "Top evidence fallback text."
        insufficient_evidence = True
        retrieval_strategy = MagicMock(value="hybrid_rerank")
        warnings = ["generator:generation_failed"]

    class _MockRunResult:
        response = _MockFallbackResp()
        state = MagicMock(selected_evidence=[], retry_count=0)
        stop_reason = MagicMock(value="generation_failed")

    mock_runtime = MagicMock()
    mock_runtime.answer.return_value = _MockRunResult()

    mock_factory = MagicMock()
    mock_factory._reranker = MagicMock(_actual_device="cpu", _actual_parameter_count=596836352)
    mock_factory.build.return_value = mock_runtime

    with patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory", return_value=mock_factory):
        res = run_gate_b_smoke(
            config_path=cfg_file,
            questions_path=questions_file,
            output_dir=tmp_path / "out_b_fail",
            device="cpu",
            max_questions=1,
        )

    assert res["passed"] is False
    assert res["status"] == "GATE_B_FAILED"
    assert res["executions"][0]["call_success"] is True
    assert res["executions"][0]["generation_success"] is False
    assert res["executions"][0]["success"] is False


def test_gate_b_fails_if_one_of_five_generation_failed(tmp_path: Path) -> None:
    """Gate B fails when 1 of 5 questions experiences generation failure."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(Path(r"C:\legal-agentic-rag-m491-jina35\configs\uit-dsc-2026-task2-m491-jina35.example.json").read_text(encoding="utf-8"), encoding="utf-8")

    questions_file = tmp_path / "questions5.json"
    questions_file.write_text(json.dumps({f"q{i}": {"question": f"Question {i}?"} for i in range(1, 6)}), encoding="utf-8")

    def _side_effect_answer(query: Any) -> Any:
        qid = query.query_id
        if qid == "q3":
            # Failing question
            res = MagicMock()
            res.response = MagicMock(answer="fallback", insufficient_evidence=True, retrieval_strategy=MagicMock(value="hybrid_rerank"), warnings=["generation_failed"])
            res.state = MagicMock(selected_evidence=[], retry_count=0)
            res.stop_reason = MagicMock(value="generation_failed")
            return res
        else:
            res = MagicMock()
            res.response = MagicMock(answer="good answer", insufficient_evidence=False, retrieval_strategy=MagicMock(value="hybrid_rerank"), warnings=[])
            res.state = MagicMock(selected_evidence=[{"chunk_id": "c1"}], retry_count=0)
            res.stop_reason = MagicMock(value="answer_verified")
            return res

    mock_runtime = MagicMock()
    mock_runtime.answer.side_effect = _side_effect_answer

    mock_factory = MagicMock()
    mock_factory._reranker = MagicMock(_actual_device="cpu", _actual_parameter_count=596836352)
    mock_factory.build.return_value = mock_runtime

    with patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory", return_value=mock_factory):
        res = run_gate_b_smoke(
            config_path=cfg_file,
            questions_path=questions_file,
            output_dir=tmp_path / "out_b_5",
            device="cpu",
            max_questions=5,
        )

    assert res["passed"] is False
    assert res["status"] == "GATE_B_FAILED"
    assert res["total_questions"] == 5
    assert res["call_successful_questions"] == 5
    assert res["generation_successful_questions"] == 4
    assert res["strict_successful_questions"] == 4


def test_gate_b_fails_if_insufficient_evidence_in_smoke_set(tmp_path: Path) -> None:
    """Gate B fails when a designated smoke question returns insufficient_evidence=True."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(Path(r"C:\legal-agentic-rag-m491-jina35\configs\uit-dsc-2026-task2-m491-jina35.example.json").read_text(encoding="utf-8"), encoding="utf-8")

    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps({"30883": {"question": "Cau hoi?"}}), encoding="utf-8")

    class _MockAbstentionResp:
        answer = "Toi khong the tra loi vi thieu chung cu."
        insufficient_evidence = True
        retrieval_strategy = MagicMock(value="hybrid_rerank")
        warnings = ["insufficient_evidence"]

    class _MockRunResult:
        response = _MockAbstentionResp()
        state = MagicMock(selected_evidence=[], retry_count=0)
        stop_reason = MagicMock(value="abstention")

    mock_runtime = MagicMock()
    mock_runtime.answer.return_value = _MockRunResult()

    mock_factory = MagicMock()
    mock_factory._reranker = MagicMock(_actual_device="cpu", _actual_parameter_count=596836352)
    mock_factory.build.return_value = mock_runtime

    with patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory", return_value=mock_factory):
        res = run_gate_b_smoke(
            config_path=cfg_file,
            questions_path=questions_file,
            output_dir=tmp_path / "out_b_abstention",
            device="cpu",
            max_questions=1,
        )

    assert res["passed"] is False
    assert res["status"] == "GATE_B_FAILED"
    assert res["executions"][0]["insufficient_evidence"] is True
    assert res["executions"][0]["success"] is False


def test_gate_b_strict_five_question_pass_and_exact_parameter_accounting(tmp_path: Path) -> None:
    """Gate B strict pass on 5 questions and exact proven parameter accounting."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(Path(r"C:\legal-agentic-rag-m491-jina35\configs\uit-dsc-2026-task2-m491-jina35.example.json").read_text(encoding="utf-8"), encoding="utf-8")

    questions_file = tmp_path / "questions5.json"
    questions_file.write_text(json.dumps({f"q{i}": {"question": f"Question {i}?"} for i in range(1, 6)}), encoding="utf-8")

    class _MockResp:
        answer = "Tra loi hop le day du."
        insufficient_evidence = False
        retrieval_strategy = MagicMock(value="hybrid_rerank")
        warnings = []

    class _MockRunResult:
        response = _MockResp()
        state = MagicMock(selected_evidence=[{"chunk_id": "c1"}], retry_count=0)
        stop_reason = MagicMock(value="answer_verified")

    mock_runtime = MagicMock()
    mock_runtime.answer.return_value = _MockRunResult()

    mock_factory = MagicMock()
    mock_factory._reranker = MagicMock(_actual_device="cuda:0", _actual_parameter_count=596836352)
    mock_factory._answer_generator = MagicMock(_actual_parameter_count=2213241664)
    mock_factory._embedding_provider = MagicMock(_actual_parameter_count=595776512)
    mock_factory.build.return_value = mock_runtime

    with patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory", return_value=mock_factory):
        res = run_gate_b_smoke(
            config_path=cfg_file,
            questions_path=questions_file,
            output_dir=tmp_path / "out_b_strict5",
            device="cuda:0",
            max_questions=5,
        )

    assert res["passed"] is True
    assert res["status"] == "GATE_B_PASSED"
    assert res["total_questions"] == 5
    assert res["strict_successful_questions"] == 5
    assert res["call_successful_questions"] == 5
    assert res["generation_successful_questions"] == 5
    assert res["verified_answer_successful_questions"] == 5

    # Parameter accounting verification
    exact = res["proven_exact_accounting"]
    assert exact["exact_embedding_parameters"] == 595776512
    assert exact["exact_jina_parameters"] == 596836352
    assert exact["exact_generator_parameters"] == 2213241664
    assert exact["exact_total_parameters"] == 3405854528
    assert exact["exact_headroom"] == 594145472
    assert exact["compliant"] is True

    hist = res["historical_registered_accounting"]
    assert hist["historical_candidate_total"] == 3311750784
    assert hist["baseline_registered_total"] == 3466000000


def test_gate_b_parameter_cap_fail_closed(tmp_path: Path) -> None:
    """Gate B fails closed if exact total parameters exceed 4B limit."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(Path(r"C:\legal-agentic-rag-m491-jina35\configs\uit-dsc-2026-task2-m491-jina35.example.json").read_text(encoding="utf-8"), encoding="utf-8")

    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps({"30883": {"question": "Cau hoi?"}}), encoding="utf-8")

    mock_runtime = MagicMock()
    mock_runtime.answer.return_value = MagicMock(
        response=MagicMock(answer="ans", insufficient_evidence=False, retrieval_strategy=MagicMock(value="hybrid_rerank"), warnings=[]),
        state=MagicMock(selected_evidence=[], retry_count=0),
        stop_reason=MagicMock(value="answer_verified"),
    )

    mock_factory = MagicMock()
    # 4.5B generator params -> exceeds 4B cap
    mock_factory._reranker = MagicMock(_actual_device="cpu", _actual_parameter_count=596836352)
    mock_factory._answer_generator = MagicMock(_actual_parameter_count=4500000000)
    mock_factory._embedding_provider = MagicMock(_actual_parameter_count=595776512)
    mock_factory.build.return_value = mock_runtime

    with patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory", return_value=mock_factory):
        res = run_gate_b_smoke(
            config_path=cfg_file,
            questions_path=questions_file,
            output_dir=tmp_path / "out_b_exceeds",
            device="cpu",
            max_questions=1,
        )

    assert res["passed"] is False
    assert res["status"] == "GATE_B_COMPLIANCE_FAILURE"
    assert res["proven_exact_accounting"]["compliant"] is False

def test_gate_a_fails_if_top10_or_full_k_differs(tmp_path: Path) -> None:
    """Test D & E: top10 or full-k mismatch causes Gate A to fail."""
    authority_dir = tmp_path / "authority_diff"
    _build_synthetic_authority(authority_dir, num_qids=100, num_candidates=40)

    pools_sha = hashlib.sha256((authority_dir / "clean100_shared_candidate_pools.jsonl").read_bytes()).hexdigest()
    jina_sha = hashlib.sha256((authority_dir / "clean100_jina_reranked.jsonl").read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((authority_dir / "clean100_phase1_manifest.json").read_bytes()).hexdigest()

    mock_p = MagicMock()
    mock_p.numel.return_value = 596836352
    mock_p.device = "cpu"

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model._ensure_tokenizer = MagicMock()
    mock_model._tokenizer = MagicMock(model_max_length=12288)
    mock_model.parameters.side_effect = lambda: iter([mock_p])

    def _rerank_swap(query: Any, documents: list[Any], top_n: Any, return_embeddings: Any) -> list[dict[str, Any]]:
        res = [{"index": idx, "relevance_score": float(100 - (idx + 1)) / 100.0} for idx in range(len(documents))]
        res[8]["relevance_score"], res[9]["relevance_score"] = res[9]["relevance_score"], res[8]["relevance_score"]
        return res

    mock_model.rerank.side_effect = _rerank_swap

    with patch.dict(
        FROZEN_AUTHORITIES,
        {
            "clean100_shared_candidate_pools_sha256": pools_sha,
            "clean100_jina_reranked_sha256": jina_sha,
            "clean100_phase1_manifest_sha256": manifest_sha,
        },
    ):
        with patch("transformers.AutoModel.from_pretrained", return_value=mock_model):

            result = run_gate_a_parity(
                authority_dir=authority_dir,
                output_dir=tmp_path / "out",
                device="cpu",
            )

    assert result["passed"] is False
    assert result["status"] == "GATE_A_FAILED"
    assert result["top1_exact"] == 100
    assert result["top10_ordered_exact"] == 0


def test_gate_a_fails_if_partial_execution(tmp_path: Path) -> None:
    """Test G: max_gate_a_qids < 100 reports DEBUG_PARTIAL_EXECUTION_NOT_PASSED and does not pass."""
    authority_dir = tmp_path / "auth_partial"
    _build_synthetic_authority(authority_dir, num_qids=100, num_candidates=40)

    pools_sha = hashlib.sha256((authority_dir / "clean100_shared_candidate_pools.jsonl").read_bytes()).hexdigest()
    jina_sha = hashlib.sha256((authority_dir / "clean100_jina_reranked.jsonl").read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((authority_dir / "clean100_phase1_manifest.json").read_bytes()).hexdigest()

    mock_p = MagicMock()
    mock_p.numel.return_value = 596836352
    mock_p.device = "cpu"

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model._ensure_tokenizer = MagicMock()
    mock_model._tokenizer = MagicMock(model_max_length=12288)
    mock_model.parameters.side_effect = lambda: iter([mock_p])
    mock_model.rerank.side_effect = lambda query, documents, top_n, return_embeddings: [
        {"index": i, "relevance_score": float(100 - (i + 1)) / 100.0} for i in range(len(documents))
    ]

    with patch.dict(
        FROZEN_AUTHORITIES,
        {
            "clean100_shared_candidate_pools_sha256": pools_sha,
            "clean100_jina_reranked_sha256": jina_sha,
            "clean100_phase1_manifest_sha256": manifest_sha,
        },
    ):
        with patch("transformers.AutoModel.from_pretrained", return_value=mock_model):

            result = run_gate_a_parity(
                authority_dir=authority_dir,
                output_dir=tmp_path / "out",
                device="cpu",
                max_gate_a_qids=10,
            )

    assert result["passed"] is False
    assert result["status"] == "DEBUG_PARTIAL_EXECUTION_NOT_PASSED"
    assert result["total_qids"] == 10
    assert result["top1_exact"] == 10

def test_background_heartbeat_formatting() -> None:
    """Test heartbeat formatting with GPU telemetry fallback."""
    from scripts.m491_jina35_mechanical_validation import BackgroundHeartbeat, get_gpu_telemetry

    telemetry = get_gpu_telemetry()
    assert "gpu_util_pct" in telemetry
    assert "vram_used_mb" in telemetry

    hb = BackgroundHeartbeat(stage_name="GATE_A_TEST", output_path="/tmp/out.json", interval=0.1)
    hb.update(processed=5, total=10, current_qid="Q0001", last_event="reranking")
    assert hb.processed == 5
    assert hb.total == 10
    assert hb.current_qid == "Q0001"


def test_gate_a_preserves_raw_question_whitespace(tmp_path: Path) -> None:
    """Test that Gate A passes raw pool question containing trailing whitespace to Jina Native."""
    from scripts.m491_jina35_mechanical_validation import run_gate_a_parity

    raw_question_with_space = "Câu hỏi pháp luật có khoảng trắng ở cuối? "

    q_pool = [{
        "question_id": f"Q{i:04d}",
        "question": raw_question_with_space if i == 0 else f"Câu hỏi {i}?",
        "candidate_hits": [
            {
                "chunk_id": f"chunk_{i}_{c}",
                "document_id": f"doc_{i}_{c}",
                "text": f"Nội dung điều khoản {c}",
                "score": float(1.0 - c * 0.02),
                "rank": c + 1,
                "strategy": "bm25", "retrieval_trace": {"bm25_rank": c + 1, "bm25_score": float(1.0 - c * 0.02)},
            }
            for c in range(40)
        ],
    } for i in range(100)]

    jina_reranked = []
    for i in range(100):
        cands = q_pool[i]["candidate_hits"]
        jina_reranked.append({
            "question_id": f"Q{i:04d}",
            "reranked_hits": [
                {
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "text": c["text"],
                    "score": float(0.95 - idx * 0.01),
                    "rank": idx + 1,
                    "strategy": "rerank", "retrieval_trace": {"reranker_score": float(0.95 - idx * 0.01)},
                }
                for idx, c in enumerate(cands)
            ],
        })

    authority_dir = tmp_path / "authority"
    authority_dir.mkdir()

    pools_file = authority_dir / "clean100_shared_candidate_pools.jsonl"
    with open(pools_file, "w", encoding="utf-8") as f:
        for r in q_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    reranked_file = authority_dir / "clean100_jina_reranked.jsonl"
    with open(reranked_file, "w", encoding="utf-8") as f:
        for r in jina_reranked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest_file = authority_dir / "clean100_phase1_manifest.json"
    manifest_file.write_text(json.dumps({"version": "1.0"}, indent=2), encoding="utf-8")

    def _mock_sha(path: Path) -> str:
        name = path.name
        if "pools" in name:
            return "45a9bd9716f14c7a5a72c54bd82f5ee17a822caa56a26a6a3998f8234e899bb0"
        if "jina" in name:
            return "eaafc39d9e3a5e5b11949d5546fea1b7b4da058cf56d99d463a1b2e642e337c9"
        return "2f733ac8a2d1d5ca94c8f18844226865f598b21f4a109959daf9bef4ea3992c3"

    recorded_queries = []

    class _MockJinaModel:
        def rerank(self, query: str, documents: list[str], top_n: int | None = None, return_embeddings: bool = False) -> list[dict[str, Any]]:
            recorded_queries.append(query)
            return [{"index": i, "relevance_score": float(0.95 - i * 0.01)} for i in range(len(documents))]

    with patch("scripts.m491_jina35_mechanical_validation.compute_file_sha256", side_effect=_mock_sha):
        with patch("legal_agentic_rag.reranking.jina_native.JinaNativeReranker._ensure_model", return_value=_MockJinaModel()):
            res = run_gate_a_parity(
                authority_dir=authority_dir,
                output_dir=tmp_path / "out_a",
                device="cpu",
                max_gate_a_qids=1,
            )

    assert len(recorded_queries) == 1
    # Verify that query sent to model.rerank() preserves the exact raw question with trailing whitespace
    assert recorded_queries[0] == raw_question_with_space
    assert recorded_queries[0].endswith(" ")
