"""Benchmark identity and immutable report persistence tests."""

import json
from pathlib import Path

import pytest

from legal_agentic_rag.evaluation import load_benchmark, persist_report
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas import (
    EvaluationCaseResult,
    EvaluationResourceUsage,
    EvaluationRunResult,
    EvaluationSummary,
    LatencySummary,
    RetrievalCaseMetrics,
    RetrievalStrategy,
)
from datetime import UTC, datetime


def test_benchmark_loader_validates_jsonl_and_hashes_exact_bytes(
    tmp_path: Path,
) -> None:
    """Benchmark provenance changes whenever its exact payload changes."""
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "question": "Câu hỏi",
                "target_granularity": "chunk",
                "relevance_grades": {"chunk-a": 1},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases, digest = load_benchmark(path)

    assert [case.case_id for case in cases] == ["case-1"]
    assert len(digest) == 64


def test_benchmark_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    """A benchmark cannot silently merge two results under one identity."""
    path = tmp_path / "duplicate.jsonl"
    line = json.dumps(
        {
            "case_id": "same",
            "question": "Câu hỏi",
            "target_granularity": "document",
            "relevance_grades": {"doc-a": 1},
        }
    )
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="unique"):
        load_benchmark(path)


def test_report_store_writes_summary_cases_and_errors_without_overwrite(
    tmp_path: Path,
) -> None:
    """Every persisted run has summary, case detail, and error analysis."""
    result = _result()
    destination = tmp_path / "report"

    persist_report(result, destination)

    assert (destination / "summary.json").is_file()
    assert len((destination / "cases.jsonl").read_text().splitlines()) == 2
    assert len((destination / "errors.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ArtifactCompatibilityError, match="already"):
        persist_report(result, destination)


def _result() -> EvaluationRunResult:
    metric = RetrievalCaseMetrics(
        recall_at_k={1: 1.0},
        precision_at_k={1: 1.0},
        ndcg_at_k={1: 1.0},
        reciprocal_rank=1.0,
        first_relevant_rank=1,
    )
    cases = [
        EvaluationCaseResult(
            case_id="ok",
            success=True,
            retrieval_metrics=metric,
        ),
        EvaluationCaseResult(
            case_id="failed",
            success=False,
            error_stage="retrieval",
            error_type="retrieval_error",
        ),
    ]
    return EvaluationRunResult(
        summary=EvaluationSummary(
            run_id="run",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            code_version="0.17.0",
            benchmark_name="fixture",
            benchmark_sha256="a" * 64,
            strategy=RetrievalStrategy.BM25,
            cutoffs=[1],
            case_count=2,
            successful_case_count=1,
            failed_case_count=1,
            retrieval_latency=LatencySummary(count=0),
            generation_latency=LatencySummary(count=0),
            resources=EvaluationResourceUsage(
                wall_time_ms=1,
                process_cpu_time_ms=1,
                python_peak_traced_memory_bytes=1,
            ),
        ),
        cases=cases,
    )
