"""Benchmark identity and immutable report persistence tests."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from legal_agentic_rag.evaluation import (
    load_benchmark,
    load_benchmark_bundle,
    load_comparison_config,
    load_evaluation_summary,
    persist_comparison_report,
    persist_report,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas import (
    EvaluationCaseResult,
    EvaluationBenchmarkLabelStatus,
    EvaluationBenchmarkManifest,
    EvaluationCandidateResult,
    EvaluationComparisonReport,
    EvaluationMetricDirection,
    EvaluationObjective,
    EvaluationResourceUsage,
    EvaluationRunResult,
    EvaluationSelectionMode,
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


def test_benchmark_bundle_validates_manifest_identity(tmp_path: Path) -> None:
    """Manifest bytes pin benchmark bytes, count, granularity, and lineage."""
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
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
    digest = sha256(benchmark.read_bytes()).hexdigest()
    manifest = tmp_path / "benchmark.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark_name": "fixture",
                "benchmark_version": "v1",
                "label_status": "diagnostic",
                "dataset_name": "fixture/legal",
                "dataset_revision": "revision-1",
                "case_count": 1,
                "benchmark_sha256": digest,
                "target_granularities": ["chunk"],
            }
        ),
        encoding="utf-8",
    )

    cases, loaded_manifest, manifest_digest = load_benchmark_bundle(
        benchmark,
        manifest,
    )

    assert [case.case_id for case in cases] == ["case-1"]
    assert loaded_manifest.benchmark_sha256 == digest
    assert manifest_digest == sha256(manifest.read_bytes()).hexdigest()


def test_benchmark_bundle_rejects_hash_mismatch(tmp_path: Path) -> None:
    """A manifest cannot be silently reused for changed benchmark bytes."""
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        '{"case_id":"a","question":"q","target_granularity":"chunk",'
        '"relevance_grades":{"chunk-a":1}}\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "benchmark.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark_name": "fixture",
                "benchmark_version": "v1",
                "label_status": "diagnostic",
                "dataset_name": "fixture/legal",
                "dataset_revision": "revision-1",
                "case_count": 1,
                "benchmark_sha256": "0" * 64,
                "target_granularities": ["chunk"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="SHA-256"):
        load_benchmark_bundle(benchmark, manifest)


def test_benchmark_bundle_rejects_case_count_mismatch(tmp_path: Path) -> None:
    """The manifest cannot claim more evaluated cases than the exact payload."""
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        '{"case_id":"a","question":"q","target_granularity":"chunk",'
        '"relevance_grades":{"chunk-a":1}}\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "benchmark.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark_name": "fixture",
                "benchmark_version": "v1",
                "label_status": "diagnostic",
                "dataset_name": "fixture/legal",
                "dataset_revision": "revision-1",
                "case_count": 2,
                "benchmark_sha256": sha256(
                    benchmark.read_bytes()
                ).hexdigest(),
                "target_granularities": ["chunk"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="case count"):
        load_benchmark_bundle(benchmark, manifest)


def test_trusted_benchmark_manifest_requires_review_provenance() -> None:
    """A trusted label status cannot exist without auditable provenance."""
    with pytest.raises(ValidationError, match="timestamped provenance"):
        EvaluationBenchmarkManifest(
            benchmark_name="fixture",
            benchmark_version="v1",
            label_status="human_reviewed",
            dataset_name="fixture/legal",
            dataset_revision="revision-1",
            case_count=1,
            benchmark_sha256="a" * 64,
            target_granularities=["chunk"],
        )


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


def test_comparison_config_paths_are_relative_to_config_file(
    tmp_path: Path,
) -> None:
    """Portable comparison specs resolve report directories predictably."""
    config_path = tmp_path / "comparison.json"
    config_path.write_text(
        json.dumps(
            {
                "comparison_name": "fixture",
                "candidates": [
                    {
                        "candidate_id": "a",
                        "report_directory": "reports/a",
                    },
                    {
                        "candidate_id": "b",
                        "report_directory": "reports/b",
                    },
                ],
                "objectives": [
                    {
                        "metric": "retrieval.ndcg@1",
                        "direction": "maximize",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_comparison_config(config_path)

    assert config.candidates[0].report_directory == (
        tmp_path / "reports/a"
    ).resolve()


def test_summary_loader_and_comparison_persistence_are_immutable(
    tmp_path: Path,
) -> None:
    """Comparison inputs and output use validated immutable JSON contracts."""
    run = _result()
    report_directory = tmp_path / "run"
    persist_report(run, report_directory)

    loaded = load_evaluation_summary(report_directory)
    assert loaded.run_id == "run"

    comparison = EvaluationComparisonReport(
        comparison_id="comparison",
        comparison_name="fixture",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        benchmark_name=loaded.benchmark_name,
        benchmark_sha256=loaded.benchmark_sha256,
        benchmark_manifest_sha256=loaded.benchmark_manifest_sha256,
        benchmark_version=loaded.benchmark_version,
        benchmark_label_status=loaded.benchmark_label_status,
        case_count=loaded.case_count,
        objectives=[
            EvaluationObjective(
                metric="retrieval.mrr",
                direction=EvaluationMetricDirection.MAXIMIZE,
            )
        ],
        selection_mode=EvaluationSelectionMode.PARETO_ONLY,
        candidates=[
            _comparison_candidate("a"),
            _comparison_candidate("b"),
        ],
        pareto_candidate_ids=["a", "b"],
    )
    destination = tmp_path / "comparison"

    persist_comparison_report(comparison, destination)

    assert (destination / "comparison.json").is_file()
    with pytest.raises(ArtifactCompatibilityError, match="already"):
        persist_comparison_report(comparison, destination)


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
            benchmark_manifest_sha256="b" * 64,
            benchmark_version="v1",
            benchmark_label_status=EvaluationBenchmarkLabelStatus.DIAGNOSTIC,
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


def _comparison_candidate(candidate_id: str) -> EvaluationCandidateResult:
    return EvaluationCandidateResult(
        candidate_id=candidate_id,
        run_id=f"run-{candidate_id}",
        code_version="0.25.0",
        strategy=RetrievalStrategy.BM25,
        runtime_config_sha256=candidate_id * 64,
        component_provenance={"generator": {"backend": "extractive"}},
        metric_values={"retrieval.mrr": 1.0},
        eligible=True,
    )
