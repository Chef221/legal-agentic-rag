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
    analyze_b1a_ablation,
    evaluate_b1a_decision_gate,
    prepare_b1a_dataset,
    verify_b1a_configs,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT_DIR / "configs" / "phase-b1a-graph-routing-cases.json"
BASE_CONFIG_PATH = ROOT_DIR / "configs" / "phase-a-current-system-census-kaggle.example.json"
CAND_CONFIG_PATH = ROOT_DIR / "configs" / "phase-b1a-graph-routing-ablation-kaggle.example.json"

EXPECTED_22_IDS = [
    "102047", "107487", "110287", "111905", "113537", "122659", "125393", "133075",
    "134605", "147239", "147869", "150051", "26541", "29491", "29877", "39671",
    "45219", "47537", "48905", "64035", "95861", "99639",
]


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
    # Build synthetic 991 development dataset containing exactly the 22 IDs
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

    # Create mock manifest with actual SHA
    mock_manifest = tmp_path / "mock_manifest.json"
    mock_manifest.write_text(json.dumps({
        "question_ids": EXPECTED_22_IDS,
    }), encoding="utf-8")

    out_path = tmp_path / "materialized_22.json"
    ident_path = tmp_path / "identity.json"

    # Mismatch SHA fails closed
    with pytest.raises(DataValidationError, match="Source development.json SHA mismatch"):
        prepare_b1a_dataset(dev_path, mock_manifest, out_path, ident_path)

    # Patch expected SHA to test success
    from unittest import mock
    with mock.patch("scripts.phase_b1a_graph_routing_ablation.CANONICAL_SOURCE_QUESTION_SHA256", actual_dev_sha):
        ident = prepare_b1a_dataset(dev_path, mock_manifest, out_path, ident_path)
        assert ident["materialized_case_count"] == 22
        assert out_path.exists()
        mat_data = json.loads(out_path.read_text(encoding="utf-8"))
        assert list(mat_data.keys()) == EXPECTED_22_IDS

        # Missing ID fails closed
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


def test_5_baseline_routing_evidence_validator() -> None:
    """TEST 5: Baseline routing evidence requires exactly 22 graph attempts and 22 graph terminals."""
    base_routing_pass = {
        "graph_search_attempt_count": 22,
        "graph_terminal_count": 22,
    }
    cand_routing_pass = {
        "graph_search_attempt_count": 0,
        "rerank_search_primary_count": 22,
    }
    base_stop = Counter({"answer_verified": 18, "generation_failed": 4})
    cand_stop = Counter({"answer_verified": 18, "generation_failed": 4})

    # Pass case
    verdict, _ = evaluate_b1a_decision_gate(
        base_routing=base_routing_pass,
        candidate_routing=cand_routing_pass,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.01,
        mean_rouge_delta=0.01,
        case_count=22,
    )
    assert verdict == "PASS_TO_B1B"

    # 21/22 graph attempts fails hard gate
    base_routing_fail = {
        "graph_search_attempt_count": 21,
        "graph_terminal_count": 21,
    }
    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing_fail,
        candidate_routing=cand_routing_pass,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.01,
        mean_rouge_delta=0.01,
        case_count=22,
    )
    assert verdict == "INVALID_EXPERIMENT"
    assert any("BASE graph attempts (21) != 22" in r for r in reasons)


def test_6_candidate_routing_evidence_validator() -> None:
    """TEST 6: Candidate routing evidence requires 0 graph attempts and 22 primary rerank routes."""
    base_routing = {
        "graph_search_attempt_count": 22,
        "graph_terminal_count": 22,
    }
    cand_routing_with_graph = {
        "graph_search_attempt_count": 1,
        "rerank_search_primary_count": 21,
    }
    base_stop = Counter({"answer_verified": 18, "generation_failed": 4})
    cand_stop = Counter({"answer_verified": 18, "generation_failed": 4})

    verdict, reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing,
        candidate_routing=cand_routing_with_graph,
        base_stop_reasons=base_stop,
        candidate_stop_reasons=cand_stop,
        mean_meteor_delta=0.01,
        mean_rouge_delta=0.01,
        case_count=22,
    )
    assert verdict == "INVALID_EXPERIMENT"
    assert any("CANDIDATE graph attempts (1) != 0" in r for r in reasons)


def test_7_paired_alignment_and_missing_case_rejection(tmp_path: Path) -> None:
    """TEST 7: Analysis fails closed if cases are missing, duplicated, or misaligned."""
    # Write mock 22 questions
    q_data = {qid: {"question": f"Q {qid}", "answer": f"A {qid}"} for qid in EXPECTED_22_IDS}
    q_file = tmp_path / "questions.json"
    q_file.write_text(json.dumps(q_data), encoding="utf-8")

    # Mock batch dirs
    base_dir = tmp_path / "base_batch"
    cand_dir = tmp_path / "cand_batch"
    base_dir.mkdir()
    cand_dir.mkdir()

    # BASE has 21 records instead of 22
    b_records = [{"question_id": qid, "response": {"answer": "ans", "metadata": {"tool_invocations": [{"tool_name": "graph_search"}], "retrieval_strategy": "graph", "stop_reason": "answer_verified"}}} for qid in EXPECTED_22_IDS[:-1]]
    b_bytes = ("\n".join(json.dumps(r) for r in b_records) + "\n").encode("utf-8")
    (base_dir / "results.jsonl").write_bytes(b_bytes)
    (base_dir / "manifest.json").write_text(json.dumps({"record_count": len(b_records), "records_sha256": sha256(b_bytes).hexdigest()}), encoding="utf-8")

    c_records = [{"question_id": qid, "response": {"answer": "ans", "metadata": {"tool_invocations": [{"tool_name": "rerank_search"}], "retrieval_strategy": "hybrid_rerank", "stop_reason": "answer_verified"}}} for qid in EXPECTED_22_IDS]
    c_bytes = ("\n".join(json.dumps(r) for r in c_records) + "\n").encode("utf-8")
    (cand_dir / "results.jsonl").write_bytes(c_bytes)
    (cand_dir / "manifest.json").write_text(json.dumps({"record_count": len(c_records), "records_sha256": sha256(c_bytes).hexdigest()}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="BASE batch question IDs do not match canonical order"):
        analyze_b1a_ablation(
            questions_path=q_file,
            base_batch_dir=base_dir,
            candidate_batch_dir=cand_dir,
            output_report_path=tmp_path / "rep.json",
            output_decision_path=tmp_path / "dec.json",
        )


def test_8_decision_gate_pass() -> None:
    """TEST 8: Both mean deltas nonnegative + reliability non-regression yields PASS_TO_B1B."""
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


def test_9_decision_gate_inconclusive() -> None:
    """TEST 9: Small negative delta > -0.005 with passed reliability yields INCONCLUSIVE."""
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


def test_10_decision_gate_fail_on_drop() -> None:
    """TEST 10: Mean delta <= -0.005 on either metric yields FAIL_RETAIN_CURRENT_GRAPH_PATH."""
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


def test_11_decision_gate_fail_on_reliability_regression() -> None:
    """TEST 11: Candidate generation failures > base yields FAIL_RETAIN_CURRENT_GRAPH_PATH."""
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


def test_12_bootstrap_determinism() -> None:
    """TEST 12: Paired bootstrap with same seed yields identical 95% confidence intervals."""
    from legal_agentic_rag.fine_tuning.paired_metrics import compute_paired_bootstrap_ci

    deltas = [0.01 * (i % 5 - 2) for i in range(22)]
    ci1 = compute_paired_bootstrap_ci(deltas, "METEOR", resamples=1000, seed=20260819)
    ci2 = compute_paired_bootstrap_ci(deltas, "METEOR", resamples=1000, seed=20260819)

    assert ci1.ci_lower_95 == ci2.ci_lower_95
    assert ci1.ci_upper_95 == ci2.ci_upper_95
    assert ci1.mean_delta == ci2.mean_delta


def test_13_no_raw_benchmark_leakage() -> None:
    """TEST 13: Manifest contains ONLY content-free metadata and IDs, with no question/answer text."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden_keys = {"question", "questions", "answer", "answers", "reference", "prompt"}
    assert not (set(manifest.keys()) & forbidden_keys)
    for qid in manifest["question_ids"]:
        assert isinstance(qid, str)
        assert qid.isdigit()
