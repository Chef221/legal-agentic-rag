#!/usr/bin/env python3
"""Phase B1A: Paired Graph-Routing Behavioral Ablation protocol tooling."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable
import zipfile

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.official_scoring import (
    score_official_compatible_answer,
)
from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.fine_tuning.paired_metrics import compute_paired_bootstrap_ci

FLOAT_TOLERANCE = 1e-12
EXPECTED_CASE_COUNT = 22
CANONICAL_SOURCE_QUESTION_COUNT = 991
CANONICAL_SOURCE_QUESTION_SHA256 = (
    "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"
)
_RETRIEVAL_TOOLS = {
    "bm25_search",
    "dense_search",
    "hybrid_search",
    "rerank_search",
    "graph_search",
}


def sha256_file(path: Path) -> str:
    """Compute deterministic SHA-256 hex digest for a file."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest for bytes."""
    return sha256(data).hexdigest()


# ----------------------------------------------------------------------
# 1. PREPARE SUBCOMMAND
# ----------------------------------------------------------------------


def prepare_b1a_dataset(
    development_path: Path,
    manifest_path: Path,
    output_path: Path,
    identity_output_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize the exact 22-question subset preserving canonical order."""
    dev_sha = sha256_file(development_path)
    if dev_sha != CANONICAL_SOURCE_QUESTION_SHA256:
        raise DataValidationError(
            f"Source development.json SHA mismatch: expected {CANONICAL_SOURCE_QUESTION_SHA256}, got {dev_sha}"
        )

    raw_dev_content = json.loads(development_path.read_text(encoding="utf-8"))
    if not isinstance(raw_dev_content, Mapping):
        raise DataValidationError("development.json root must be a mapping")
    if len(raw_dev_content) != CANONICAL_SOURCE_QUESTION_COUNT:
        raise DataValidationError(
            f"Source development.json question count mismatch: expected {CANONICAL_SOURCE_QUESTION_COUNT}, got {len(raw_dev_content)}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_ids: list[str] = manifest.get("question_ids", [])
    if len(target_ids) != EXPECTED_CASE_COUNT:
        raise DataValidationError(
            f"Manifest question_ids count mismatch: expected {EXPECTED_CASE_COUNT}, got {len(target_ids)}"
        )
    if len(set(target_ids)) != len(target_ids):
        raise DataValidationError("Manifest question_ids contain duplicates")

    # Verify all IDs exist in development.json and determine their order
    dev_key_order = list(raw_dev_content.keys())
    for qid in target_ids:
        if qid not in raw_dev_content:
            raise DataValidationError(f"Question ID {qid} not found in development.json")

    # Filter while strictly preserving the canonical development.json order
    ordered_subset = {
        qid: raw_dev_content[qid]
        for qid in dev_key_order
        if qid in set(target_ids)
    }

    if list(ordered_subset.keys()) != target_ids:
        raise DataValidationError(
            "Target IDs order in manifest does not match canonical development.json order"
        )

    output_bytes = json.dumps(ordered_subset, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    output_path.write_bytes(output_bytes)
    output_sha = sha256_bytes(output_bytes)

    identity = {
        "candidate": "PHASE-B1A",
        "created_at": datetime.now(UTC).isoformat(),
        "source_question_count": len(raw_dev_content),
        "source_question_sha256": dev_sha,
        "materialized_case_count": len(ordered_subset),
        "materialized_case_sha256": output_sha,
        "materialized_question_ids": list(ordered_subset.keys()),
        "output_path": str(output_path),
    }

    if identity_output_path is not None:
        identity_output_path.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return identity


# ----------------------------------------------------------------------
# 2. VERIFY-CONFIGS SUBCOMMAND
# ----------------------------------------------------------------------


def _find_dict_diff(
    dict_a: dict[str, Any],
    dict_b: dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, Any, Any]]:
    """Recursively find differences between two dictionaries."""
    diffs: list[tuple[str, Any, Any]] = []
    all_keys = sorted(set(dict_a.keys()) | set(dict_b.keys()))
    for key in all_keys:
        path = f"{prefix}.{key}" if prefix else key
        if key not in dict_a:
            diffs.append((path, None, dict_b[key]))
        elif key not in dict_b:
            diffs.append((path, dict_a[key], None))
        else:
            val_a = dict_a[key]
            val_b = dict_b[key]
            if isinstance(val_a, dict) and isinstance(val_b, dict):
                diffs.extend(_find_dict_diff(val_a, val_b, path))
            elif val_a != val_b:
                diffs.append((path, val_a, val_b))
    return diffs


def verify_b1a_configs(
    base_config_path: Path,
    candidate_config_path: Path,
) -> dict[str, Any]:
    """Verify that candidate config differs from base config ONLY in adaptive_routing_enabled."""
    base_raw = json.loads(base_config_path.read_text(encoding="utf-8"))
    cand_raw = json.loads(candidate_config_path.read_text(encoding="utf-8"))

    # Ensure both configs parse via ApplicationConfig
    base_app_config = ApplicationConfig.model_validate(base_raw)
    cand_app_config = ApplicationConfig.model_validate(cand_raw)

    diffs = _find_dict_diff(base_raw, cand_raw)

    expected_diff_path = "online.query_understanding.adaptive_routing_enabled"

    if len(diffs) != 1 or diffs[0][0] != expected_diff_path:
        raise DataValidationError(
            f"Unexpected config differences: {diffs}. "
            f"Expected ONLY {expected_diff_path}: True -> False"
        )

    path, val_base, val_cand = diffs[0]
    if val_base is not True or val_cand is not False:
        raise DataValidationError(
            f"Config diff value mismatch at {path}: expected True -> False, got {val_base} -> {val_cand}"
        )

    # Invariant assertions
    assert base_app_config.online.retrieval.candidate_k == cand_app_config.online.retrieval.candidate_k
    assert base_app_config.online.retrieval.top_k == cand_app_config.online.retrieval.top_k
    assert base_app_config.online.agent.strategy_order == cand_app_config.online.agent.strategy_order
    assert base_app_config.online.generation.model_name == cand_app_config.online.generation.model_name
    assert base_app_config.online.generation.model_revision == cand_app_config.online.generation.model_revision
    assert base_app_config.online.generation.device == cand_app_config.online.generation.device
    assert base_app_config.online.vector_runtime.search_device == cand_app_config.online.vector_runtime.search_device
    assert base_app_config.offline.embedding.device == cand_app_config.offline.embedding.device
    assert base_app_config.online.reranker.device == cand_app_config.online.reranker.device
    assert base_app_config.online.reranker.model_name == cand_app_config.online.reranker.model_name
    assert base_app_config.artifacts.root_path == cand_app_config.artifacts.root_path

    return {
        "valid": True,
        "base_config_sha256": sha256_file(base_config_path),
        "candidate_config_sha256": sha256_file(candidate_config_path),
        "semantic_diff": {
            "path": expected_diff_path,
            "base_value": True,
            "candidate_value": False,
        },
    }


# ----------------------------------------------------------------------
# 3. ANALYZE SUBCOMMAND & DECISION GATE
# ----------------------------------------------------------------------


def _load_batch_records(batch_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = batch_dir / "manifest.json"
    results_path = batch_dir / "results.jsonl"
    if not manifest_path.exists() or not results_path.exists():
        raise DataValidationError(f"Batch directory {batch_dir} missing manifest.json or results.jsonl")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_sha = sha256_file(results_path)
    if manifest.get("records_sha256") and manifest["records_sha256"] != records_sha:
        raise DataValidationError(
            f"Batch {batch_dir} results.jsonl SHA mismatch: expected {manifest['records_sha256']}, got {records_sha}"
        )

    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_CASE_COUNT:
        raise DataValidationError(
            f"Batch {batch_dir} record count mismatch: expected {EXPECTED_CASE_COUNT}, got {len(records)}"
        )
    if manifest.get("record_count") and manifest["record_count"] != len(records):
        raise DataValidationError(
            f"Batch {batch_dir} record count mismatch: expected {manifest['record_count']}, got {len(records)}"
        )

    # Fail closed on empty or duplicate question IDs
    raw_ids = [str(r.get("question_id", "")).strip() for r in records]
    if any(not qid for qid in raw_ids):
        raise DataValidationError(f"Batch {batch_dir} contains empty or whitespace question IDs")
    if len(set(raw_ids)) != len(raw_ids):
        raise DataValidationError(f"Batch {batch_dir} contains duplicate question IDs: {raw_ids}")

    return manifest, records


def _extract_routing_info(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response")
    if not isinstance(response, dict):
        raise DataValidationError("Batch record response is missing or not a dict")

    metadata = response.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    agent = metadata.get("agent")
    agent = agent if isinstance(agent, dict) else {}
    invocations = agent.get("tool_invocations")
    invocations = invocations if isinstance(invocations, list) else []

    retrieval_tools = [
        inv.get("tool_name") for inv in invocations
        if isinstance(inv, dict) and inv.get("tool_name") in _RETRIEVAL_TOOLS
    ]
    graph_attempts = sum(1 for inv in invocations if isinstance(inv, dict) and inv.get("tool_name") == "graph_search")
    rerank_attempts = sum(1 for inv in invocations if isinstance(inv, dict) and inv.get("tool_name") == "rerank_search")
    hybrid_attempts = sum(1 for inv in invocations if isinstance(inv, dict) and inv.get("tool_name") == "hybrid_search")

    first_tool = retrieval_tools[0] if retrieval_tools else None
    final_strategy = response.get("retrieval_strategy")

    warnings = response.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []
    retrieval_model_errors = sum(1 for w in warnings if w == "retrieval:model_error")

    latency = agent.get("total_latency_ms", 0.0)
    latency_ms = float(latency) if isinstance(latency, (int, float)) and not isinstance(latency, bool) else 0.0

    return {
        "graph_attempts": graph_attempts,
        "rerank_attempts": rerank_attempts,
        "hybrid_attempts": hybrid_attempts,
        "first_retrieval_tool": first_tool,
        "final_retrieval_strategy": final_strategy,
        "stop_reason": agent.get("stop_reason"),
        "latency_ms": latency_ms,
        "retrieval_model_errors": retrieval_model_errors,
    }


def evaluate_b1a_decision_gate(
    base_routing: dict[str, Any],
    candidate_routing: dict[str, Any],
    base_stop_reasons: Counter[str],
    candidate_stop_reasons: Counter[str],
    mean_meteor_delta: float,
    mean_rouge_delta: float,
    case_count: int,
    base_retrieval_model_errors: int = 0,
    candidate_retrieval_model_errors: int = 0,
) -> tuple[str, list[str]]:
    """Mechanically evaluate the pre-registered B1A decision gate."""
    reasons: list[str] = []

    # 1. Hard protocol checks
    hard_fail = False
    if case_count != EXPECTED_CASE_COUNT:
        reasons.append(f"Hard protocol failure: expected {EXPECTED_CASE_COUNT} cases, got {case_count}")
        hard_fail = True
    if base_routing.get("graph_search_attempt_count") != EXPECTED_CASE_COUNT:
        reasons.append(
            f"Hard protocol failure: BASE graph attempts ({base_routing.get('graph_search_attempt_count')}) != {EXPECTED_CASE_COUNT}"
        )
        hard_fail = True
    if base_routing.get("graph_terminal_count") != EXPECTED_CASE_COUNT:
        reasons.append(
            f"Hard protocol failure: BASE graph terminal count ({base_routing.get('graph_terminal_count')}) != {EXPECTED_CASE_COUNT}"
        )
        hard_fail = True
    if candidate_routing.get("graph_search_attempt_count") != 0:
        reasons.append(
            f"Hard protocol failure: CANDIDATE graph attempts ({candidate_routing.get('graph_search_attempt_count')}) != 0"
        )
        hard_fail = True
    if candidate_routing.get("rerank_search_primary_count") != EXPECTED_CASE_COUNT:
        reasons.append(
            f"Hard protocol failure: CANDIDATE primary rerank_search ({candidate_routing.get('rerank_search_primary_count')}) != {EXPECTED_CASE_COUNT}"
        )
        hard_fail = True
    if base_retrieval_model_errors > 0:
        reasons.append(
            f"Hard protocol failure: BASE contains {base_retrieval_model_errors} retrieval:model_error warnings"
        )
        hard_fail = True
    if candidate_retrieval_model_errors > 0:
        reasons.append(
            f"Hard protocol failure: CANDIDATE contains {candidate_retrieval_model_errors} retrieval:model_error warnings"
        )
        hard_fail = True

    if hard_fail:
        return "INVALID_EXPERIMENT", reasons

    # 2. Reliability non-regression checks
    base_gen_fail = base_stop_reasons.get("generation_failed", 0)
    cand_gen_fail = candidate_stop_reasons.get("generation_failed", 0)
    base_cit_fail = base_stop_reasons.get("citation_verification_failed", 0)
    cand_cit_fail = candidate_stop_reasons.get("citation_verification_failed", 0)
    base_verified = base_stop_reasons.get("answer_verified", 0)
    cand_verified = candidate_stop_reasons.get("answer_verified", 0)

    rel_fail = False
    if cand_gen_fail > base_gen_fail:
        reasons.append(f"Reliability regression: generation_failed increased ({base_gen_fail} -> {cand_gen_fail})")
        rel_fail = True
    if cand_cit_fail > base_cit_fail:
        reasons.append(f"Reliability regression: citation_verification_failed increased ({base_cit_fail} -> {cand_cit_fail})")
        rel_fail = True
    if cand_verified < base_verified:
        reasons.append(f"Reliability regression: answer_verified decreased ({base_verified} -> {cand_verified})")
        rel_fail = True

    if rel_fail:
        return "FAIL_RETAIN_CURRENT_GRAPH_PATH", reasons

    # 3. Semantic gates
    if mean_meteor_delta <= -0.005 or mean_rouge_delta <= -0.005:
        reasons.append(
            f"Semantic clear failure: METEOR delta ({mean_meteor_delta:+.6f}) or ROUGE-L delta ({mean_rouge_delta:+.6f}) <= -0.005"
        )
        return "FAIL_RETAIN_CURRENT_GRAPH_PATH", reasons

    if mean_meteor_delta >= 0.0 and mean_rouge_delta >= 0.0:
        reasons.append(
            f"Semantic strong pass: METEOR delta ({mean_meteor_delta:+.6f}) >= 0.0 and ROUGE-L delta ({mean_rouge_delta:+.6f}) >= 0.0 with reliability non-regression"
        )
        return "PASS_TO_B1B", reasons

    reasons.append(
        f"Inconclusive band: Reliability passed and no delta <= -0.005, but one or both mean deltas are negative (METEOR: {mean_meteor_delta:+.6f}, ROUGE-L: {mean_rouge_delta:+.6f})"
    )
    return "INCONCLUSIVE", reasons


def analyze_b1a_ablation(
    questions_path: Path,
    base_batch_dir: Path,
    candidate_batch_dir: Path,
    output_report_path: Path,
    output_decision_path: Path,
    seed: int = 20260819,
    resamples: int = 10000,
    meteor_scorer: Callable[[list[list[str]], list[str]], float] | None = None,
) -> dict[str, Any]:
    """Run full paired evaluation, bootstrap CI, and pre-registered decision gating."""
    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_questions, Mapping):
        raise DataValidationError("22-question source must be an object mapping")

    ordered_qids = list(raw_questions.keys())
    if len(ordered_qids) != EXPECTED_CASE_COUNT:
        raise DataValidationError(
            f"Questions file count mismatch: expected {EXPECTED_CASE_COUNT}, got {len(ordered_qids)}"
        )

    base_manifest, base_records = _load_batch_records(base_batch_dir)
    cand_manifest, cand_records = _load_batch_records(candidate_batch_dir)

    base_ids = [str(r.get("question_id", "")).strip() for r in base_records]
    cand_ids = [str(r.get("question_id", "")).strip() for r in cand_records]

    # Verify ID alignment in exact canonical order
    if base_ids != ordered_qids:
        raise DataValidationError(
            f"BASE batch question IDs do not match canonical order: expected {ordered_qids}, got {base_ids}"
        )
    if cand_ids != ordered_qids:
        raise DataValidationError(
            f"CANDIDATE batch question IDs do not match canonical order: expected {ordered_qids}, got {cand_ids}"
        )

    base_by_id = {str(r.get("question_id")): r for r in base_records}
    cand_by_id = {str(r.get("question_id")): r for r in cand_records}

    # Routing audits
    base_graph_attempts = 0
    base_graph_terminals = 0
    base_rerank_attempts = 0
    base_hybrid_attempts = 0
    base_retrieval_model_errors = 0

    cand_graph_attempts = 0
    cand_rerank_attempts = 0
    cand_hybrid_attempts = 0
    cand_rerank_primary_count = 0
    cand_retrieval_model_errors = 0

    base_stop_reasons: Counter[str] = Counter()
    cand_stop_reasons: Counter[str] = Counter()

    paired_cases: list[dict[str, Any]] = []
    meteor_base_list: list[float] = []
    meteor_cand_list: list[float] = []
    meteor_deltas: list[float] = []

    rouge_base_list: list[float] = []
    rouge_cand_list: list[float] = []
    rouge_deltas: list[float] = []

    base_latencies: list[float] = []
    cand_latencies: list[float] = []

    for qid in ordered_qids:
        q_record = raw_questions[qid]
        reference_answer = q_record.get("answer") or ""

        b_rec = base_by_id[qid]
        c_rec = cand_by_id[qid]

        b_resp = b_rec.get("response") or {}
        c_resp = c_rec.get("response") or {}

        b_pred = b_resp.get("answer") or ""
        c_pred = c_resp.get("answer") or ""

        b_route = _extract_routing_info(b_rec)
        c_route = _extract_routing_info(c_rec)

        base_graph_attempts += b_route["graph_attempts"]
        base_rerank_attempts += b_route["rerank_attempts"]
        base_hybrid_attempts += b_route["hybrid_attempts"]
        base_retrieval_model_errors += b_route["retrieval_model_errors"]
        if b_route["final_retrieval_strategy"] == "graph":
            base_graph_terminals += 1
        base_stop_reasons[str(b_route["stop_reason"])] += 1
        base_latencies.append(b_route["latency_ms"])

        cand_graph_attempts += c_route["graph_attempts"]
        cand_rerank_attempts += c_route["rerank_attempts"]
        cand_hybrid_attempts += c_route["hybrid_attempts"]
        cand_retrieval_model_errors += c_route["retrieval_model_errors"]
        if c_route["first_retrieval_tool"] == "rerank_search":
            cand_rerank_primary_count += 1
        cand_stop_reasons[str(c_route["stop_reason"])] += 1
        cand_latencies.append(c_route["latency_ms"])

        # Score BASE and CANDIDATE predictions
        b_metrics = score_official_compatible_answer(
            b_pred, reference_answer, meteor_scorer=meteor_scorer
        )
        c_metrics = score_official_compatible_answer(
            c_pred, reference_answer, meteor_scorer=meteor_scorer
        )

        d_meteor = c_metrics.meteor - b_metrics.meteor
        d_rouge = c_metrics.rouge_l - b_metrics.rouge_l

        meteor_base_list.append(b_metrics.meteor)
        meteor_cand_list.append(c_metrics.meteor)
        meteor_deltas.append(d_meteor)

        rouge_base_list.append(b_metrics.rouge_l)
        rouge_cand_list.append(c_metrics.rouge_l)
        rouge_deltas.append(d_rouge)

        paired_cases.append({
            "question_id": qid,
            "base_metrics": {
                "meteor": b_metrics.meteor,
                "rouge_l": b_metrics.rouge_l,
                "exact_match": b_metrics.exact_match,
            },
            "candidate_metrics": {
                "meteor": c_metrics.meteor,
                "rouge_l": c_metrics.rouge_l,
                "exact_match": c_metrics.exact_match,
            },
            "deltas": {
                "meteor_delta": d_meteor,
                "rouge_l_delta": d_rouge,
            },
            "base_routing": b_route,
            "candidate_routing": c_route,
        })

    # W/T/L calculation
    def _wtl(deltas: list[float]) -> dict[str, int]:
        w = sum(1 for d in deltas if d > FLOAT_TOLERANCE)
        t = sum(1 for d in deltas if abs(d) <= FLOAT_TOLERANCE)
        l = sum(1 for d in deltas if d < -FLOAT_TOLERANCE)
        return {"win": w, "tie": t, "loss": l}

    meteor_wtl = _wtl(meteor_deltas)
    rouge_wtl = _wtl(rouge_deltas)

    # Bootstrap CIs
    meteor_bootstrap = compute_paired_bootstrap_ci(
        deltas=meteor_deltas,
        metric_name="METEOR",
        resamples=resamples,
        seed=seed,
    )
    rouge_bootstrap = compute_paired_bootstrap_ci(
        deltas=rouge_deltas,
        metric_name="ROUGE-L",
        resamples=resamples,
        seed=seed,
    )

    base_routing_summary = {
        "graph_search_attempt_count": base_graph_attempts,
        "graph_terminal_count": base_graph_terminals,
        "rerank_search_attempt_count": base_rerank_attempts,
        "hybrid_search_attempt_count": base_hybrid_attempts,
        "retrieval_model_error_count": base_retrieval_model_errors,
    }
    cand_routing_summary = {
        "graph_search_attempt_count": cand_graph_attempts,
        "rerank_search_primary_count": cand_rerank_primary_count,
        "rerank_search_attempt_count": cand_rerank_attempts,
        "hybrid_search_attempt_count": cand_hybrid_attempts,
        "retrieval_model_error_count": cand_retrieval_model_errors,
    }

    mean_d_meteor = fmean(meteor_deltas)
    mean_d_rouge = fmean(rouge_deltas)

    decision, decision_reasons = evaluate_b1a_decision_gate(
        base_routing=base_routing_summary,
        candidate_routing=cand_routing_summary,
        base_stop_reasons=base_stop_reasons,
        candidate_stop_reasons=cand_stop_reasons,
        mean_meteor_delta=mean_d_meteor,
        mean_rouge_delta=mean_d_rouge,
        case_count=len(ordered_qids),
        base_retrieval_model_errors=base_retrieval_model_errors,
        candidate_retrieval_model_errors=cand_retrieval_model_errors,
    )

    paired_report = {
        "experiment_id": "PHASE-B1A",
        "code_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "source_question_count": len(ordered_qids),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "summary": {
            "meteor": {
                "base_mean": fmean(meteor_base_list),
                "candidate_mean": fmean(meteor_cand_list),
                "mean_delta": mean_d_meteor,
                "median_delta": median(meteor_deltas),
                "wtl": meteor_wtl,
                "ci_95": {
                    "lower": meteor_bootstrap.ci_lower_95,
                    "upper": meteor_bootstrap.ci_upper_95,
                },
            },
            "rouge_l": {
                "base_mean": fmean(rouge_base_list),
                "candidate_mean": fmean(rouge_cand_list),
                "mean_delta": mean_d_rouge,
                "median_delta": median(rouge_deltas),
                "wtl": rouge_wtl,
                "ci_95": {
                    "lower": rouge_bootstrap.ci_lower_95,
                    "upper": rouge_bootstrap.ci_upper_95,
                },
            },
            "reliability": {
                "base_stop_reasons": dict(base_stop_reasons),
                "candidate_stop_reasons": dict(cand_stop_reasons),
                "base_answer_verified_count": base_stop_reasons.get("answer_verified", 0),
                "candidate_answer_verified_count": cand_stop_reasons.get("answer_verified", 0),
                "base_generation_failed_count": base_stop_reasons.get("generation_failed", 0),
                "candidate_generation_failed_count": cand_stop_reasons.get("generation_failed", 0),
                "base_citation_verification_failed_count": base_stop_reasons.get("citation_verification_failed", 0),
                "candidate_citation_verification_failed_count": cand_stop_reasons.get("citation_verification_failed", 0),
                "base_retrieval_model_error_count": base_retrieval_model_errors,
                "candidate_retrieval_model_error_count": cand_retrieval_model_errors,
            },
            "routing": {
                "base": base_routing_summary,
                "candidate": cand_routing_summary,
            },
            "latency_exploratory_ms": {
                "base_mean": fmean(base_latencies) if base_latencies else 0.0,
                "candidate_mean": fmean(cand_latencies) if cand_latencies else 0.0,
            },
        },
        "decision": {
            "verdict": decision,
            "reasons": decision_reasons,
        },
        "cases": paired_cases,
    }

    decision_report = {
        "experiment_id": "PHASE-B1A",
        "verdict": decision,
        "reasons": decision_reasons,
        "meteor_mean_delta": mean_d_meteor,
        "rouge_l_mean_delta": mean_d_rouge,
        "meteor_wtl": meteor_wtl,
        "rouge_l_wtl": rouge_wtl,
        "meteor_ci_95": [meteor_bootstrap.ci_lower_95, meteor_bootstrap.ci_upper_95],
        "rouge_l_ci_95": [rouge_bootstrap.ci_lower_95, rouge_bootstrap.ci_upper_95],
    }

    output_report_path.write_text(
        json.dumps(paired_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_decision_path.write_text(
        json.dumps(decision_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return paired_report


# ----------------------------------------------------------------------
# 4. PACKAGE SUBCOMMAND
# ----------------------------------------------------------------------


def package_b1a_evidence(
    output_zip_path: Path,
    manifest_path: Path,
    questions_identity_path: Path,
    base_config_path: Path,
    candidate_config_path: Path,
    base_batch_dir: Path,
    candidate_batch_dir: Path,
    paired_report_path: Path,
    decision_report_path: Path,
) -> dict[str, Any]:
    """Bundle all B1A evidence artifacts into a single ZIP archive."""
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname="configs/phase-b1a-graph-routing-cases.json")
        zf.write(questions_identity_path, arcname="evidence/materialized_questions_identity.json")
        zf.write(base_config_path, arcname="configs/base_runtime_config.json")
        zf.write(candidate_config_path, arcname="configs/candidate_runtime_config.json")
        zf.write(paired_report_path, arcname="results/phase_b1a_paired_report.json")
        zf.write(decision_report_path, arcname="results/phase_b1a_decision_report.json")

        for fname in ["manifest.json", "results.jsonl", "batch_state.json"]:
            b_file = base_batch_dir / fname
            if b_file.exists():
                zf.write(b_file, arcname=f"base_batch/{fname}")
            c_file = candidate_batch_dir / fname
            if c_file.exists():
                zf.write(c_file, arcname=f"candidate_batch/{fname}")

    zip_sha = sha256_file(output_zip_path)
    zip_size = output_zip_path.stat().st_size

    return {
        "zip_path": str(output_zip_path),
        "zip_sha256": zip_sha,
        "zip_size_bytes": zip_size,
    }


# ----------------------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B1A: Paired Graph-Routing Behavioral Ablation Tooling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Materialize the 22-case benchmark subset")
    p_prep.add_argument("--development", type=Path, required=True, help="Path to development.json")
    p_prep.add_argument("--manifest", type=Path, required=True, help="Path to B1A cases manifest")
    p_prep.add_argument("--output", type=Path, required=True, help="Path to output 22-case JSON")
    p_prep.add_argument("--identity-output", type=Path, default=None, help="Path to output identity JSON")

    # verify-configs
    p_vc = subparsers.add_parser("verify-configs", help="Verify baseline vs candidate config diff")
    p_vc.add_argument("--base-config", type=Path, required=True, help="Path to base config")
    p_vc.add_argument("--candidate-config", type=Path, required=True, help="Path to candidate config")

    # analyze
    p_an = subparsers.add_parser("analyze", help="Run paired analysis on completed batches")
    p_an.add_argument("--questions", type=Path, required=True, help="Path to 22-question source JSON")
    p_an.add_argument("--base-batch", type=Path, required=True, help="Path to completed base batch directory")
    p_an.add_argument("--candidate-batch", type=Path, required=True, help="Path to completed candidate batch directory")
    p_an.add_argument("--output-report", type=Path, required=True, help="Path to output paired report JSON")
    p_an.add_argument("--output-decision", type=Path, required=True, help="Path to output decision report JSON")
    p_an.add_argument("--seed", type=int, default=20260819, help="Bootstrap seed")
    p_an.add_argument("--resamples", type=int, default=10000, help="Bootstrap resample count")

    # package
    p_pk = subparsers.add_parser("package", help="Package B1A evidence ZIP")
    p_pk.add_argument("--output-zip", type=Path, required=True, help="Path to output evidence ZIP")
    p_pk.add_argument("--manifest", type=Path, required=True, help="Path to B1A cases manifest")
    p_pk.add_argument("--questions-identity", type=Path, required=True, help="Path to materialized questions identity JSON")
    p_pk.add_argument("--base-config", type=Path, required=True, help="Path to base config")
    p_pk.add_argument("--candidate-config", type=Path, required=True, help="Path to candidate config")
    p_pk.add_argument("--base-batch", type=Path, required=True, help="Path to completed base batch directory")
    p_pk.add_argument("--candidate-batch", type=Path, required=True, help="Path to completed candidate batch directory")
    p_pk.add_argument("--paired-report", type=Path, required=True, help="Path to paired report JSON")
    p_pk.add_argument("--decision-report", type=Path, required=True, help="Path to decision report JSON")

    args = parser.parse_args()

    if args.command == "prepare":
        res = prepare_b1a_dataset(
            development_path=args.development.resolve(),
            manifest_path=args.manifest.resolve(),
            output_path=args.output.resolve(),
            identity_output_path=args.identity_output.resolve() if args.identity_output else None,
        )
        print(json.dumps(res, indent=2))

    elif args.command == "verify-configs":
        res = verify_b1a_configs(
            base_config_path=args.base_config.resolve(),
            candidate_config_path=args.candidate_config.resolve(),
        )
        print(json.dumps(res, indent=2))

    elif args.command == "analyze":
        res = analyze_b1a_ablation(
            questions_path=args.questions.resolve(),
            base_batch_dir=args.base_batch.resolve(),
            candidate_batch_dir=args.candidate_batch.resolve(),
            output_report_path=args.output_report.resolve(),
            output_decision_path=args.output_decision.resolve(),
            seed=args.seed,
            resamples=args.resamples,
        )
        print(json.dumps(res["decision"], indent=2))

    elif args.command == "package":
        res = package_b1a_evidence(
            output_zip_path=args.output_zip.resolve(),
            manifest_path=args.manifest.resolve(),
            questions_identity_path=args.questions_identity.resolve(),
            base_config_path=args.base_config.resolve(),
            candidate_config_path=args.candidate_config.resolve(),
            base_batch_dir=args.base_batch.resolve(),
            candidate_batch_dir=args.candidate_batch.resolve(),
            paired_report_path=args.paired_report.resolve(),
            decision_report_path=args.decision_report.resolve(),
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
