"""Evaluation metrics, runner, benchmark loading, and report persistence."""

from legal_agentic_rag.evaluation.comparison import EvaluationComparisonService
from legal_agentic_rag.evaluation.metrics import (
    StandardGenerationEvaluator,
    StandardRetrievalEvaluator,
    score_text_answer,
)
from legal_agentic_rag.evaluation.provenance import evaluation_runtime_provenance
from legal_agentic_rag.evaluation.report_store import (
    load_benchmark,
    load_benchmark_bundle,
    load_comparison_config,
    load_evaluation_summary,
    persist_comparison_report,
    persist_report,
)
from legal_agentic_rag.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "EvaluationComparisonService",
    "StandardGenerationEvaluator",
    "StandardRetrievalEvaluator",
    "score_text_answer",
    "evaluation_runtime_provenance",
    "load_benchmark",
    "load_benchmark_bundle",
    "load_comparison_config",
    "load_evaluation_summary",
    "persist_comparison_report",
    "persist_report",
]
