#!/usr/bin/env python3
"""Summarize one completed competition batch for Phase A without persisting content."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

_RETRIEVAL_TOOLS = {
    "bm25_search",
    "dense_search",
    "hybrid_search",
    "rerank_search",
    "graph_search",
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _load(batch: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = batch / "manifest.json"
    records_path = batch / "results.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest.get("records_sha256")
    actual_sha = _sha256_file(records_path)
    if expected_sha != actual_sha:
        raise SystemExit("results.jsonl SHA-256 does not match manifest")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest.get("record_count") != len(records):
        raise SystemExit("record count does not match manifest")
    ids = [str(record.get("question_id", "")) for record in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise SystemExit("question IDs are blank or duplicated")
    return manifest, records


def _metadata(response: dict[str, object]) -> dict[str, object]:
    value = response.get("metadata")
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a content-free Phase A census")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest, records = _load(args.batch.resolve())
    stop_reasons: Counter[str] = Counter()
    final_strategies: Counter[str] = Counter()
    retrieval_tool_attempts: Counter[str] = Counter()
    retrieval_tool_successes: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    selection_reasons: Counter[str] = Counter()
    selected_counts: list[float] = []
    context_tokens: list[float] = []
    agent_latencies: list[float] = []
    answer_characters: list[float] = []
    answer_words: list[float] = []
    insufficient = 0
    graph_touched_records = 0
    graph_final_records = 0
    trace_present = 0

    for record in records:
        response = record.get("response")
        if not isinstance(response, dict):
            raise SystemExit("batch record response is invalid")
        metadata = _metadata(response)
        agent = metadata.get("agent")
        agent = agent if isinstance(agent, dict) else {}
        context = metadata.get("context")
        context = context if isinstance(context, dict) else {}

        stop = agent.get("stop_reason")
        stop_reasons[str(stop) if isinstance(stop, str) else "missing"] += 1
        strategy = response.get("retrieval_strategy")
        strategy_value = str(strategy) if isinstance(strategy, str) else "missing"
        final_strategies[strategy_value] += 1
        graph_final_records += int(strategy_value == "graph")
        insufficient += int(response.get("insufficient_evidence") is True)

        warnings = response.get("warnings")
        if isinstance(warnings, list):
            warning_counts.update(str(value) for value in warnings if isinstance(value, str))

        invocations = agent.get("tool_invocations")
        graph_touched = False
        if isinstance(invocations, list):
            for invocation in invocations:
                if not isinstance(invocation, dict):
                    continue
                tool = invocation.get("tool_name")
                if not isinstance(tool, str) or tool not in _RETRIEVAL_TOOLS:
                    continue
                retrieval_tool_attempts[tool] += 1
                if invocation.get("success") is True:
                    retrieval_tool_successes[tool] += 1
                graph_touched = graph_touched or tool == "graph_search"
        graph_touched_records += int(graph_touched)

        latency = agent.get("total_latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            agent_latencies.append(float(latency))

        trace = context.get("selection_trace")
        if isinstance(trace, list):
            trace_present += 1
            for item in trace:
                if isinstance(item, dict) and isinstance(item.get("reason"), str):
                    selection_reasons[str(item["reason"])] += 1
        selected = context.get("selected_count")
        if isinstance(selected, int) and not isinstance(selected, bool):
            selected_counts.append(float(selected))
        tokens = context.get("estimated_token_count")
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            context_tokens.append(float(tokens))

        answer = response.get("answer")
        if isinstance(answer, str):
            answer_characters.append(float(len(answer)))
            answer_words.append(float(len(answer.split())))

    report = {
        "schema_version": "1.0",
        "purpose": "phase_a_current_system_census",
        "content_free": True,
        "batch_identity": {
            "question_source_sha256": manifest.get("question_source_sha256"),
            "application_config_hash": manifest.get("application_config_hash"),
            "code_version": manifest.get("code_version"),
            "records_sha256": manifest.get("records_sha256"),
            "record_count": len(records),
        },
        "outcomes": {
            "insufficient_evidence_count": insufficient,
            "insufficient_evidence_rate": insufficient / len(records),
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
        },
        "routing": {
            "final_strategy_counts": dict(sorted(final_strategies.items())),
            "retrieval_tool_attempt_counts": dict(sorted(retrieval_tool_attempts.items())),
            "retrieval_tool_success_counts": dict(sorted(retrieval_tool_successes.items())),
            "graph_touched_record_count": graph_touched_records,
            "graph_final_strategy_record_count": graph_final_records,
        },
        "context": {
            "selection_trace_present_count": trace_present,
            "selection_reason_counts": dict(sorted(selection_reasons.items())),
            "selected_evidence_count_summary": _summary(selected_counts),
            "estimated_context_token_summary": _summary(context_tokens),
        },
        "answer_shape": {
            "character_count_summary": _summary(answer_characters),
            "whitespace_word_count_summary": _summary(answer_words),
        },
        "latency_ms": _summary(agent_latencies),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit("output already exists; census reports are immutable")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
