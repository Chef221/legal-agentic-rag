"""Reproducible evaluation runner over the immutable online runtime."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import fmean
from time import perf_counter, process_time
import tracemalloc
from typing import Protocol
import unicodedata
from uuid import uuid4

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration import EvaluationConfig
from legal_agentic_rag.contracts import GenerationEvaluator, RetrievalEvaluator
from legal_agentic_rag.evaluation.metrics import (
    StandardGenerationEvaluator,
    StandardRetrievalEvaluator,
    ranked_target_ids,
)
from legal_agentic_rag.schemas import (
    AgentRunResult,
    ArtifactManifest,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationResourceUsage,
    EvaluationRunResult,
    EvaluationSummary,
    LatencySummary,
    RetrievalQuery,
    RetrievalResponse,
)


class _EvaluationRuntime(Protocol):
    @property
    def manifests(self) -> dict[str, ArtifactManifest]: ...

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse: ...

    def answer(self, query: RetrievalQuery) -> AgentRunResult: ...


class EvaluationRunner:
    """Run labeled cases without coupling metrics to backend implementations."""

    def __init__(
        self,
        runtime: _EvaluationRuntime,
        config: EvaluationConfig,
        *,
        retrieval_evaluator: RetrievalEvaluator | None = None,
        generation_evaluator: GenerationEvaluator | None = None,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._retrieval = retrieval_evaluator or StandardRetrievalEvaluator()
        self._generation = generation_evaluator or StandardGenerationEvaluator()

    def run(
        self,
        cases: Sequence[EvaluationCase],
        *,
        benchmark_name: str,
        benchmark_sha256: str,
    ) -> EvaluationRunResult:
        """Evaluate bounded cases and aggregate only available metrics."""
        selected = list(cases)
        if self._config.max_cases is not None:
            selected = selected[: self._config.max_cases]
        run_id = str(uuid4())
        wall_started = perf_counter()
        cpu_started = process_time()
        owns_tracer = not tracemalloc.is_tracing()
        if owns_tracer:
            tracemalloc.start()
        tracemalloc.reset_peak()
        results = [self._run_case(case, run_id) for case in selected]
        _, peak_memory = tracemalloc.get_traced_memory()
        if owns_tracer:
            tracemalloc.stop()
        resources = EvaluationResourceUsage(
            wall_time_ms=(perf_counter() - wall_started) * 1000,
            process_cpu_time_ms=(process_time() - cpu_started) * 1000,
            python_peak_traced_memory_bytes=peak_memory,
        )
        summary = self._summary(
            results,
            run_id=run_id,
            benchmark_name=benchmark_name,
            benchmark_sha256=benchmark_sha256,
            resources=resources,
        )
        return EvaluationRunResult(summary=summary, cases=results)

    def _run_case(
        self,
        case: EvaluationCase,
        run_id: str,
    ) -> EvaluationCaseResult:
        query = self._query(case, run_id)
        retrieval_started = perf_counter()
        try:
            response = self._runtime.retrieve(query)
            retrieval_latency = (perf_counter() - retrieval_started) * 1000
            metrics = self._retrieval.evaluate(
                case,
                response,
                self._config.cutoffs,
            )
            ranked_ids = ranked_target_ids(case, response)
            missing = sorted(set(case.relevance_grades) - set(ranked_ids))
        except Exception as error:
            if self._config.fail_fast:
                raise
            return self._failure(case, "retrieval", error)

        answer_response = None
        generation_metrics = None
        generation_latency = None
        if self._config.run_generation:
            generation_started = perf_counter()
            try:
                answer_response = self._runtime.answer(
                    query.model_copy(
                        update={"query_id": f"{query.query_id}:answer"}
                    )
                ).response
                generation_latency = (
                    perf_counter() - generation_started
                ) * 1000
                generation_metrics = self._generation.evaluate(
                    case,
                    answer_response,
                )
            except Exception as error:
                if self._config.fail_fast:
                    raise
                return EvaluationCaseResult(
                    case_id=case.case_id,
                    success=False,
                    retrieved_ids=ranked_ids,
                    missing_relevant_ids=missing,
                    retrieval_metrics=metrics,
                    retrieval_latency_ms=retrieval_latency,
                    error_stage="generation",
                    error_type=_error_type(error),
                )
        return EvaluationCaseResult(
            case_id=case.case_id,
            success=True,
            retrieved_ids=ranked_ids,
            missing_relevant_ids=missing,
            retrieval_metrics=metrics,
            generation_metrics=generation_metrics,
            answer_response=answer_response,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
        )

    def _query(self, case: EvaluationCase, run_id: str) -> RetrievalQuery:
        normalized = unicodedata.normalize(
            "NFC",
            " ".join(case.question.split()),
        )
        return RetrievalQuery(
            query_id=f"{run_id}:{case.case_id}",
            original_question=case.question,
            normalized_question=normalized,
            top_k=max(self._config.cutoffs),
            candidate_k=self._config.candidate_k,
            requested_strategy=self._config.strategy,
            metadata={"source": "evaluation"},
        )

    @staticmethod
    def _failure(
        case: EvaluationCase,
        stage: str,
        error: Exception,
    ) -> EvaluationCaseResult:
        return EvaluationCaseResult(
            case_id=case.case_id,
            success=False,
            error_stage=stage,
            error_type=_error_type(error),
        )

    def _summary(
        self,
        results: list[EvaluationCaseResult],
        *,
        run_id: str,
        benchmark_name: str,
        benchmark_sha256: str,
        resources: EvaluationResourceUsage,
    ) -> EvaluationSummary:
        retrieval_values: dict[str, list[float]] = {}
        generation_values: dict[str, list[float]] = {}
        for result in results:
            if result.retrieval_metrics is not None:
                metric = result.retrieval_metrics
                retrieval_values.setdefault("mrr", []).append(
                    metric.reciprocal_rank
                )
                for cutoff, value in metric.recall_at_k.items():
                    retrieval_values.setdefault(
                        f"recall@{cutoff}", []
                    ).append(value)
                for cutoff, value in metric.precision_at_k.items():
                    retrieval_values.setdefault(
                        f"precision@{cutoff}", []
                    ).append(value)
                for cutoff, value in metric.ndcg_at_k.items():
                    retrieval_values.setdefault(
                        f"ndcg@{cutoff}", []
                    ).append(value)
            if result.generation_metrics is not None:
                for name, value in result.generation_metrics.model_dump().items():
                    if value is not None:
                        generation_values.setdefault(name, []).append(value)
        all_values = retrieval_values | generation_values
        warnings = ["semantic_generation_metrics_require_gold_or_human_labels"]
        if self._config.run_generation and not generation_values:
            warnings.append("no_generation_labels_available")
        manifests = self._runtime.manifests
        return EvaluationSummary(
            run_id=run_id,
            created_at=datetime.now(UTC),
            code_version=__version__,
            benchmark_name=benchmark_name,
            benchmark_sha256=benchmark_sha256,
            strategy=self._config.strategy,
            cutoffs=self._config.cutoffs,
            case_count=len(results),
            successful_case_count=sum(result.success for result in results),
            failed_case_count=sum(not result.success for result in results),
            retrieval_metrics={
                name: fmean(values)
                for name, values in retrieval_values.items()
            },
            generation_metrics={
                name: fmean(values)
                for name, values in generation_values.items()
            },
            metric_case_counts={
                name: len(values) for name, values in all_values.items()
            },
            retrieval_latency=_latency_summary(
                [
                    result.retrieval_latency_ms
                    for result in results
                    if result.retrieval_latency_ms is not None
                ]
            ),
            generation_latency=_latency_summary(
                [
                    result.generation_latency_ms
                    for result in results
                    if result.generation_latency_ms is not None
                ]
            ),
            resources=resources,
            artifact_versions={
                key: manifest.artifact_version
                for key, manifest in manifests.items()
            },
            warnings=warnings,
        )


def _latency_summary(values: list[float]) -> LatencySummary:
    if not values:
        return LatencySummary(count=0)
    ordered = sorted(values)
    return LatencySummary(
        count=len(ordered),
        mean_ms=fmean(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        max_ms=ordered[-1],
    )


def _percentile(values: list[float], fraction: float) -> float:
    index = max(0, min(len(values) - 1, round((len(values) - 1) * fraction)))
    return values[index]


def _error_type(error: Exception) -> str:
    name = type(error).__name__
    return "".join(
        f"_{character.casefold()}" if character.isupper() else character
        for character in name
    ).lstrip("_")
