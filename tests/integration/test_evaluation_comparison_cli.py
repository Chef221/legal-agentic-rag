"""End-to-end comparison CLI wiring over immutable summary reports."""

from datetime import UTC, datetime
import json
from pathlib import Path

from legal_agentic_rag.schemas import (
    EvaluationBenchmarkLabelStatus,
    EvaluationResourceUsage,
    EvaluationSummary,
    LatencySummary,
    RetrievalStrategy,
)
from legal_agentic_rag.serving.cli import compare_main


def test_comparison_cli_loads_compares_and_persists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The public CLI produces one inspectable comparison artifact."""
    for candidate_id, quality in (("a", 0.9), ("b", 0.8)):
        directory = tmp_path / candidate_id
        directory.mkdir()
        summary = _summary(candidate_id, quality)
        (directory / "summary.json").write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
    config_path = tmp_path / "comparison-config.json"
    config_path.write_text(
        json.dumps(
            {
                "comparison_name": "fixture",
                "candidates": [
                    {"candidate_id": "a", "report_directory": "a"},
                    {"candidate_id": "b", "report_directory": "b"},
                ],
                "objectives": [
                    {
                        "metric": "retrieval.ndcg@1",
                        "direction": "maximize",
                    }
                ],
                "selection_mode": "lexicographic",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "legal-rag-compare",
            "--comparison",
            str(config_path),
            "--output",
            str(output),
        ],
    )

    compare_main()

    payload = json.loads(
        (output / "comparison.json").read_text(encoding="utf-8")
    )
    assert payload["selected_candidate_id"] == "a"
    assert payload["pareto_candidate_ids"] == ["a"]


def _summary(run_id: str, quality: float) -> EvaluationSummary:
    return EvaluationSummary(
        run_id=run_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        code_version="0.25.0",
        benchmark_name="fixture.jsonl",
        benchmark_sha256="a" * 64,
        benchmark_manifest_sha256="b" * 64,
        benchmark_version="v1",
        benchmark_label_status=(
            EvaluationBenchmarkLabelStatus.HUMAN_REVIEWED
        ),
        benchmark_verified_at=datetime(2026, 1, 1, tzinfo=UTC),
        benchmark_label_provenance_reference="review-protocol:v1",
        dataset_name="fixture/legal",
        dataset_revision="revision-1",
        strategy=RetrievalStrategy.BM25,
        cutoffs=[1],
        case_count=1,
        successful_case_count=1,
        failed_case_count=0,
        retrieval_metrics={"ndcg@1": quality},
        metric_case_counts={"ndcg@1": 1},
        retrieval_latency=LatencySummary(
            count=1,
            mean_ms=1,
            p50_ms=1,
            p95_ms=1,
            max_ms=1,
        ),
        generation_latency=LatencySummary(count=0),
        resources=EvaluationResourceUsage(
            wall_time_ms=1,
            process_cpu_time_ms=1,
            python_peak_traced_memory_bytes=1,
        ),
        runtime_config_sha256=run_id * 64,
        component_provenance={"retrieval": {"candidate": run_id}},
    )
