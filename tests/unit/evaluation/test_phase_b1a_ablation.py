import sys
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.exceptions import DataValidationError
from scripts.phase_b1a_graph_routing_ablation import (
    CANONICAL_SOURCE_QUESTION_COUNT,
    CANONICAL_SOURCE_QUESTION_SHA256,
    EXPECTED_CASE_COUNT,
    _extract_routing_info,
    analyze_b1a_ablation,
    evaluate_b1a_decision_gate,
    prepare_b1a_dataset,
    verify_b1a_configs,
)

MANIFEST_PATH = ROOT_DIR / "configs" / "phase-b1a-graph-routing-cases.json"
BASE_CONFIG_PATH = ROOT_DIR / "configs" / "phase-a-current-system-census-kaggle.example.json"
CAND_CONFIG_PATH = ROOT_DIR / "configs" / "phase-b1a-graph-routing-ablation-kaggle.example.json"

EXPECTED_22_IDS = [
    "102047", "107487", "110287", "111905", "113537", "122659", "125393", "133075",
    "134605", "147239", "147869", "150051", "26541", "29491", "29877", "39671",
    "45219", "47537", "48905", "64035", "95861", "99639",
]


def _build_realistic_record(
    question_id: str,
    *,
    strategy: str = "graph",
    tools: list[str] | None = None,
    stop_reason: str = "answer_verified",
    latency_ms: float = 1200.0,
    warnings: list[str] | None = None,
    answer: str = "Test answer.",
) -> dict[str, object]:
    """Build a realistic batch record matching CompetitionBatchRunner persisted contract."""
    tool_names = tools or (["graph_search"] if strategy == "graph" else ["rerank_search"])
    invocations = [{"tool_name": t, "success": True} for t in tool_names]
    return {
        "question_id": question_id,
        "response": {
            "answer": answer,
            "citations": [],
            "warnings": warnings or [],
            "insufficient_evidence": False,
            "retrieval_strategy": strategy,
            "metadata": {
                "agent": {
                    "tool_invocations": invocations,
                    "stop_reason": stop_reason,
                    "total_latency_ms": latency_ms,
                    "retry_count": 0,
                },
                "context": {
                    "selected_count": 2,
                    "estimated_token_count": 500,
                },
            },
        },
    }


def test_1_canonical_22_case_manifest() -> None:
    """TEST 1: The committed manifest contains exactly 22 unique IDs in canonical order with matching source SHA."""
    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["candidate"] == "PHASE-B1A"
    assert manifest["case_count"] == EXPECTED_CASE_COUNT
    assert manifest["source_question_count"] == CANONICAL_SOURCE_QUESTION_COUNT
    assert manifest["source_question_sha256"] == CANONICAL_SOURCE_QUESTION_SHA256
    assert manifest["question_ids"] == EXPECTED_22_IDS
    assert len(set(manifest["question_ids"])) == 22
    assert len(manifest["historical_ordinals"]) == 22
    assert manifest["historical_ordinals"] == sorted(manifest["historical_ordinals"])


def test_2_case_materialization(tmp_path: Path) -> None:
    """TEST 2: prepare_b1a_dataset extracts only requested IDs preserving canonical order and validates SHAs."""
    synthetic_dev: dict[str, dict[str, str]] = {}
    filler_count = CANONICAL_SOURCE_QUESTION_COUNT - len(EXPECTED_22_IDS)
    for i in range(filler_count):
        qid = str(100000 + i)
        synthetic_dev[qid] = {"question": f"Question {qid}?", "answer": f"Answer {qid}."}

    for qid in EXPECTED_22_IDS:
        synthetic_dev[qid] = {"question": f"Question {qid}?", "answer": f"Answer {qid}."}

    assert len(synthetic_dev) == CANONICAL_SOURCE_QUESTION_COUNT

    dev_path = tmp_path / "development.json"
    dev_bytes = json.dumps(synthetic_dev, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    dev_path.write_bytes(dev_bytes)
    actual_dev_sha = sha256(dev_bytes).hexdigest()

    mock_manifest = tmp_path / "mock_manifest.json"
    mock_manifest.write_text(json.dumps({
        "question_ids": EXPECTED_22_IDS,
    }), encoding="utf-8")

    out_path = tmp_path / "materialized_22.json"
    ident_path = tmp_path / "identity.json"

    with pytest.raises(DataValidationError, match="Source development.json SHA mismatch"):
        prepare_b1a_dataset(dev_path, mock_manifest, out_path, ident_path)

    from unittest import mock
    with mock.patch("scripts.phase_b1a_graph_routing_ablation.CANONICAL_SOURCE_QUESTION_SHA256", actual_dev_sha):
        ident = prepare_b1a_dataset(dev_path, mock_manifest, out_path, ident_path)
        assert ident["materialized_case_count"] == 22
        assert out_path.exists()
        mat_data = json.loads(out_path.read_text(encoding="utf-8"))
        assert list(mat_data.keys()) == EXPECTED_22_IDS

        bad_manifest = tmp_path / "bad_manifest.json"
        bad_manifest.write_text(json.dumps({
            "question_ids": EXPECTED_22_IDS[:-1] + ["nonexistent_id"],
        }), encoding="utf-8")
        with pytest.raises(DataValidationError, match="not found in development.json"):
            prepare_b1a_dataset(dev_path, bad_manifest, out_path)


def test_3_candidate_config_diff() -> None:
    """TEST 3: Base and candidate example configs differ ONLY in adaptive_routing_enabled."""
    res = verify_b1a_configs(BASE_CONFIG_PATH, CAND_CONFIG_PATH)
    assert res["valid"] is True
    assert res["semantic_diff"]["path"] == "online.query_understanding.adaptive_routing_enabled"
    assert res["semantic_diff"]["base_value"] is True
    assert res["semantic_diff"]["candidate_value"] is False


def test_4_config_parses_through_application_config() -> None:
    """TEST 4: Both base and candidate example configs parse strictly through ApplicationConfig."""
    base = ApplicationConfig.model_validate(json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8")))
    cand = ApplicationConfig.model_validate(json.loads(CAND_CONFIG_PATH.read_text(encoding="utf-8")))

    assert base.online.query_understanding.adaptive_routing_enabled is True
    assert cand.online.query_understanding.adaptive_routing_enabled is False
    assert base.online.retrieval.candidate_k == 40
    assert cand.online.retrieval.candidate_k == 40
    assert base.online.agent.strategy_order == cand.online.agent.strategy_order


def test_5_real_batch_record_routing_extraction() -> None:
    """TEST 5: _extract_routing_info parses the real CompetitionBatchRunner response metadata shape."""
    # BASE record with graph route
    base_rec = _build_realistic_record(
        "102047",
        strategy="graph",
        tools=["graph_search"],
        stop_reason="answer_verified",
        latency_ms=1550.5,
        warnings=["retrieval:model_error"],
    )
    b_info = _extract_routing_info(base_rec)
    assert b_info["graph_attempts"] == 1
    assert b_info["rerank_attempts"] == 0
    assert b_info["first_retrieval_tool"] == "graph_search"
    assert b_info["final_retrieval_strategy"] == "graph"
    assert b_info["stop_reason"] == "answer_verified"
    assert b_info["latency_ms"] == 1550.5
    assert b_info["retrieval_model_errors"] == 1

    # CANDIDATE record with hybrid_rerank route
    cand_rec = _build_realistic_record(
        "102047",
        strategy="hybrid_rerank",
        tools=["rerank_search"],
        stop_reason="answer_verified",
        latency_ms=920.0,
        warnings=[],
    )
    c_info = _extract_routing_info(cand_rec)
    assert c_info["graph_attempts"] == 0
    assert c_info["rerank_attempts"] == 1
    assert c_info["first_retrieval_tool"] == "rerank_search"
    assert c_info["final_retrieval_strategy"] == "hybrid_rerank"
    assert c_info["stop_reason"] == "answer_verified"
    assert c_info["latency_ms"] == 920.0
    assert c_info["retrieval_model_errors"] == 0


def test_6_batch_alignment_duplicate_and_count_rejection(tmp_path: Path) -> None:
    """TEST 6: _load_batch_records and analyze_b1a_ablation reject duplicate IDs and wrong counts."""
    q_data = {qid: {"question": f"Q {qid}", "answer": f"A {qid}"} for qid in EXPECTED_22_IDS}
    q_file = tmp_path / "questions.json"
    q_file.write_text(json.dumps(q_data), encoding="utf-8")

    base_dir = tmp_path / "base_batch"
    cand_dir = tmp_path / "cand_batch"
    base_dir.mkdir()
    cand_dir.mkdir()

    # 1. 21 records instead of 22
    b_records_21 = [_build_realistic_record(qid) for qid in EXPECTED_22_IDS[:-1]]
    b_bytes_21 = ("\n".join(json.dumps(r) for r in b_records_21) + "\n").encode("utf-8")
    (base_dir / "results.jsonl").write_bytes(b_bytes_21)
    (base_dir / "manifest.json").write_text(json.dumps({"record_count": len(b_records_21), "records_sha256": sha256(b_bytes_21).hexdigest()}), encoding="utf-8")

    c_records_22 = [_build_realistic_record(qid, strategy="hybrid_rerank") for qid in EXPECTED_22_IDS]
    c_bytes_22 = ("\n".join(json.dumps(r) for r in c_records_22) + "\n").encode("utf-8")
    (cand_dir / "results.jsonl").write_bytes(c_bytes_22)
    (cand_dir / "manifest.json").write_text(json.dumps({"record_count": len(c_records_22), "records_sha256": sha256(c_bytes_22).hexdigest()}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="record count mismatch: expected 22, got 21"):
        analyze_b1a_ablation(
            questions_path=q_file,
            base_batch_dir=base_dir,
            candidate_batch_dir=cand_dir,
            output_report_path=tmp_path / "rep.json",
            output_decision_path=tmp_path / "dec.json",
        )

    # 2. 23 records instead of 22
    b_records_23 = [_build_realistic_record(qid) for qid in EXPECTED_22_IDS] + [_build_realistic_record("999999")]
    b_bytes_23 = ("\n".join(json.dumps(r) for r in b_records_23) + "\n").encode("utf-8")
    (base_dir / "results.jsonl").write_bytes(b_bytes_23)
    (base_dir / "manifest.json").write_text(json.dumps({"record_count": len(b_records_23), "records_sha256": sha256(b_bytes_23).hexdigest()}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="record count mismatch: expected 22, got 23"):
        analyze_b1a_ablation(
            questions_path=q_file,
            base_batch_dir=base_dir,
            candidate_batch_dir=cand_dir,
            output_report_path=tmp_path / "rep.json",
            output_decision_path=tmp_path / "dec.json",
        )

    # 3. Duplicate ID (22 records, but one duplicated)
    b_records_dup = [_build_realistic_record(qid) for qid in EXPECTED_22_IDS[:-1]] + [_build_realistic_record(EXPECTED_22_IDS[0])]
    b_bytes_dup = ("\n".join(json.dumps(r) for r in b_records_dup) + "\n").encode("utf-8")
    (base_dir / "results.jsonl").write_bytes(b_bytes_dup)
    (base_dir / "manifest.json").write_text(json.dumps({"record_count": len(b_records_dup), "records_sha256": sha256(b_bytes_dup).hexdigest()}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="contains duplicate question IDs"):
        analyze_b1a_ablation(
            questions_path=q_file,
            base_batch_dir=base_dir,
            candidate_batch_dir=cand_dir,
            output_report_path=tmp_path / "rep.json",
            output_decision_path=tmp_path / "dec.json",
        )

    # 4. Wrong order
    b_records_rev = [_build_realistic_record(qid) for qid in reversed(EXPECTED_22_IDS)]
    b_bytes_rev = ("\n".join(json.dumps(r) for r in b_records_rev) + "\n").encode("utf-8")
    (base_dir / "results.jsonl").write_bytes(b_bytes_rev)
    (base_dir / "manifest.json").write_text(json.dumps({"record_count": len(b_records_rev), "records_sha256": sha256(b_bytes_rev).hexdigest()}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="BASE batch question IDs do not match canonical order"):
        analyze_b1a_ablation(
            questions_path=q_file,
            base_batch_dir=base_dir,
            candidate_batch_dir=cand_dir,
            output_report_path=tmp_path / "rep.json",
            output_decision_path=tmp_path / "dec.json",
        )


def test_7_retrieval_model_error_triggers_invalid_experiment() -> None:
    """TEST 7: retrieval:model_error warnings cause INVALID_EXPERIMENT while generation_failed does not."""
    base_routing = {"graph_search_attempt_count": 22, "graph_terminal_count": 22}
    cand_routing = {"graph_search_attempt_count": 0, "rerank_search_primary_count": 22}
    base_stop = Counter({"answer_verified": 18, "generation_failed": 4})
    cand_stop = Counter({"answer_verified": 18, "generation_failed": 4})

    # BASE contains retrieval model error
    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.01,
        mean_rouge_delta=0.01,
        case_count=22,
        base_retrieval_model_errors=1,
        candidate_retrieval_model_errors=0,
    )
    assert verdict == "INVALID_EXPERIMENT"
    assert any("BASE contains 1 retrieval:model_error warnings" in r for r in reasons)

    # CANDIDATE contains retrieval model error
    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.01,
        mean_rouge_delta=0.01,
        case_count=22,
        base_retrieval_model_errors=0,
        candidate_retrieval_model_errors=2,
    )
    assert verdict == "INVALID_EXPERIMENT"
    assert any("CANDIDATE contains 2 retrieval:model_error warnings" in r for r in reasons)

    # Clean run with ordinary generation_failed is NOT invalid
    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.01,
        mean_rouge_delta=0.01,
        case_count=22,
        base_retrieval_model_errors=0,
        candidate_retrieval_model_errors=0,
    )
    assert verdict == "PASS_TO_B1B"


def test_8_config_verification_rejects_device_and_root_mismatches(tmp_path: Path) -> None:
    """TEST 8: verify_b1a_configs fails closed on any device, artifact path, or parameter diff besides routing."""
    base_cfg = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    cand_cfg = json.loads(CAND_CONFIG_PATH.read_text(encoding="utf-8"))

    # 1. Generation device mismatch (e.g. cuda:0 vs cuda:1)
    bad_cand_1 = json.loads(json.dumps(cand_cfg))
    bad_cand_1["online"]["generation"]["device"] = "cuda:1"
    p_base = tmp_path / "base1.json"
    p_cand1 = tmp_path / "cand1.json"
    p_base.write_text(json.dumps(base_cfg), encoding="utf-8")
    p_cand1.write_text(json.dumps(bad_cand_1), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unexpected config differences"):
        verify_b1a_configs(p_base, p_cand1)

    # 2. Vector runtime search device mismatch
    bad_cand_2 = json.loads(json.dumps(cand_cfg))
    bad_cand_2["online"]["vector_runtime"]["search_device"] = "cpu"
    p_cand2 = tmp_path / "cand2.json"
    p_cand2.write_text(json.dumps(bad_cand_2), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unexpected config differences"):
        verify_b1a_configs(p_base, p_cand2)

    # 3. Artifact root mismatch
    bad_cand_3 = json.loads(json.dumps(cand_cfg))
    bad_cand_3["artifacts"]["root_path"] = "/different/root"
    p_cand3 = tmp_path / "cand3.json"
    p_cand3.write_text(json.dumps(bad_cand_3), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unexpected config differences"):
        verify_b1a_configs(p_base, p_cand3)

    # 4. Candidate_k mismatch
    bad_cand_4 = json.loads(json.dumps(cand_cfg))
    bad_cand_4["online"]["retrieval"]["candidate_k"] = 50
    p_cand4 = tmp_path / "cand4.json"
    p_cand4.write_text(json.dumps(bad_cand_4), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Unexpected config differences"):
        verify_b1a_configs(p_base, p_cand4)

    # 5. Exact adaptive_routing_enabled-only diff passes
    res = verify_b1a_configs(BASE_CONFIG_PATH, CAND_CONFIG_PATH)
    assert res["valid"] is True


def test_9_decision_gate_pass() -> None:
    """TEST 9: Both mean deltas nonnegative + reliability non-regression yields PASS_TO_B1B."""
    base_routing = {"graph_search_attempt_count": 22, "graph_terminal_count": 22}
    cand_routing = {"graph_search_attempt_count": 0, "rerank_search_primary_count": 22}
    base_stop = Counter({"answer_verified": 18, "generation_failed": 4})
    cand_stop = Counter({"answer_verified": 19, "generation_failed": 3})

    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.005,
        mean_rouge_delta=0.002,
        case_count=22,
    )
    assert verdict == "PASS_TO_B1B"
    assert any("Semantic strong pass" in r for r in reasons)


def test_10_decision_gate_inconclusive() -> None:
    """TEST 10: Small negative delta > -0.005 with passed reliability yields INCONCLUSIVE."""
    base_routing = {"graph_search_attempt_count": 22, "graph_terminal_count": 22}
    cand_routing = {"graph_search_attempt_count": 0, "rerank_search_primary_count": 22}
    base_stop = Counter({"answer_verified": 18, "generation_failed": 4})
    cand_stop = Counter({"answer_verified": 18, "generation_failed": 4})

    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=-0.001,
        mean_rouge_delta=0.002,
        case_count=22,
    )
    assert verdict == "INCONCLUSIVE"
    assert any("Inconclusive band" in r for r in reasons)


def test_11_decision_gate_fail_on_drop() -> None:
    """TEST 11: Mean delta <= -0.005 on either metric yields FAIL_RETAIN_CURRENT_GRAPH_PATH."""
    base_routing = {"graph_search_attempt_count": 22, "graph_terminal_count": 22}
    cand_routing = {"graph_search_attempt_count": 0, "rerank_search_primary_count": 22}
    base_stop = Counter({"answer_verified": 18, "generation_failed": 4})
    cand_stop = Counter({"answer_verified": 18, "generation_failed": 4})

    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=-0.006,
        mean_rouge_delta=0.002,
        case_count=22,
    )
    assert verdict == "FAIL_RETAIN_CURRENT_GRAPH_PATH"
    assert any("Semantic clear failure" in r for r in reasons)


def test_12_decision_gate_fail_on_reliability_regression() -> None:
    """TEST 12: Candidate generation failures > base yields FAIL_RETAIN_CURRENT_GRAPH_PATH."""
    base_routing = {"graph_search_attempt_count": 22, "graph_terminal_count": 22}
    cand_routing = {"graph_search_attempt_count": 0, "rerank_search_primary_count": 22}
    base_stop = Counter({"answer_verified": 18, "generation_failed": 4})
    cand_stop = Counter({"answer_verified": 16, "generation_failed": 6})

    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.010,
        mean_rouge_delta=0.010,
        case_count=22,
    )
    assert verdict == "FAIL_RETAIN_CURRENT_GRAPH_PATH"
    assert any("generation_failed increased" in r for r in reasons)


def test_13_bootstrap_determinism() -> None:
    """TEST 13: Paired bootstrap with same seed yields identical 95% confidence intervals."""
    from legal_agentic_rag.fine_tuning.paired_metrics import compute_paired_bootstrap_ci

    deltas = [0.01 * (i % 5 - 2) for i in range(22)]
    ci1 = compute_paired_bootstrap_ci(deltas, "METEOR", resamples=1000, seed=20260819)
    ci2 = compute_paired_bootstrap_ci(deltas, "METEOR", resamples=1000, seed=20260819)

    assert ci1.ci_lower_95 == ci2.ci_lower_95
    assert ci1.ci_upper_95 == ci2.ci_upper_95
    assert ci1.mean_delta == ci2.mean_delta


def test_14_no_raw_benchmark_leakage() -> None:
    """TEST 14: Manifest contains ONLY content-free metadata and IDs, with no question/answer text."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden_keys = {"question", "questions", "answer", "answers", "reference", "prompt"}
    assert not (set(manifest.keys()) & forbidden_keys)
    for qid in manifest["question_ids"]:
        assert isinstance(qid, str)
        assert qid.isdigit()
