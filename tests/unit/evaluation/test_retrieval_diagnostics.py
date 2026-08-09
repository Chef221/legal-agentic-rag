"""Tests for answer-level, non-gold retrieval diagnostics."""

import json
from pathlib import Path

import pytest

from legal_agentic_rag.evaluation import RetrievalDiagnosticsRunner
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import (
    QueryAnalysis,
    QueryIntent,
    QueryVariant,
    QueryVariantKind,
    RetrievalDiagnosticSignal,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)


class _Runtime:
    def __init__(self) -> None:
        self.comparison_calls = 0

    def retrieve_comparison(
        self,
        query: RetrievalQuery,
        *,
        include_reranker: bool = False,
    ) -> list[RetrievalResponse]:
        self.comparison_calls += 1
        strategies = [
            RetrievalStrategy.BM25,
            RetrievalStrategy.DENSE,
            RetrievalStrategy.HYBRID,
        ]
        if include_reranker:
            strategies.append(RetrievalStrategy.HYBRID_RERANK)
        return [self._response(query, strategy) for strategy in strategies]

    @staticmethod
    def _response(
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
    ) -> RetrievalResponse:
        identities = {
            RetrievalStrategy.BM25: [("chunk-a", "doc-a"), ("chunk-b", "doc-b")],
            RetrievalStrategy.DENSE: [("chunk-b", "doc-b"), ("chunk-c", "doc-b")],
            RetrievalStrategy.HYBRID: [("chunk-b", "doc-b"), ("chunk-c", "doc-b")],
            RetrievalStrategy.HYBRID_RERANK: [
                ("chunk-c", "doc-b"),
                ("chunk-d", "doc-d"),
            ],
        }[strategy]
        enriched = query.model_copy(
            update={
                "query_analysis": QueryAnalysis(
                    intent=QueryIntent.REFERENCE_LOOKUP,
                    article_numbers=["10"],
                ),
                "query_variants": [
                    QueryVariant(
                        variant_id="qv-1",
                        text=query.normalized_question,
                        kind=QueryVariantKind.NORMALIZED,
                    )
                ],
            }
        )
        hits = [
            RetrievalHit(
                chunk_id=chunk_id,
                document_id=document_id,
                rank=rank,
                score=1 / rank,
                strategy=strategy,
                text="quy định về hồ sơ doanh nghiệp",
                metadata={"article_number": "9"},
                retrieval_trace=RetrievalTrace(),
            )
            for rank, (chunk_id, document_id) in enumerate(identities, 1)
        ]
        return RetrievalResponse(
            query=enriched,
            strategy=strategy,
            hits=hits,
            latency_ms=float(len(hits)),
            warnings=["fixture_warning"] if strategy == RetrievalStrategy.DENSE else [],
        )


def _questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "q1": {
                    "question": "Điều 10 quy định thời hạn bao lâu?",
                    "answer": "Ba mươi ngày kể từ khi nhận yêu cầu.",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_diagnostics_reports_branch_overlap_and_non_gold_signals(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "development.json"
    output = tmp_path / "diagnostics"
    _questions(questions)

    runtime = _Runtime()
    report = RetrievalDiagnosticsRunner(
        runtime, application_config_sha256="a" * 64
    ).run(questions, output, top_k=2, candidate_k=4)

    case = report.cases[0]
    assert case.bm25_dense_overlap_count == 1
    assert case.bm25_dense_jaccard == pytest.approx(1 / 3)
    assert case.hybrid_document_diversity == 0.5
    assert case.explicit_reference_match is False
    assert RetrievalDiagnosticSignal.EXPLICIT_REFERENCE_NOT_RETRIEVED in case.signals
    assert RetrievalDiagnosticSignal.LOW_ANSWER_TERM_COVERAGE in case.signals
    assert RetrievalDiagnosticSignal.RETRIEVAL_WARNING in case.signals
    persisted = (output / "retrieval_diagnostics.json").read_text(encoding="utf-8")
    assert "Điều 10 quy định" not in persisted
    assert "Ba mươi ngày" not in persisted
    assert "quy định về hồ sơ" not in persisted
    assert "answer_term_coverage_is_not_retrieval_relevance_gold" in persisted
    assert runtime.comparison_calls == 1


def test_diagnostics_is_immutable_and_records_runtime_errors(tmp_path: Path) -> None:
    class _FailingRuntime:
        def retrieve_comparison(
            self,
            query: RetrievalQuery,
            *,
            include_reranker: bool = False,
        ) -> list[RetrievalResponse]:
            raise RuntimeError("backend failed")

    questions = tmp_path / "development.json"
    output = tmp_path / "diagnostics"
    _questions(questions)
    runner = RetrievalDiagnosticsRunner(
        _FailingRuntime(), application_config_sha256="b" * 64
    )
    report = runner.run(questions, output, top_k=1, candidate_k=1)

    assert report.failed_case_count == 1
    assert report.cases[0].error_type == "runtime_error"
    assert report.signal_counts[RetrievalDiagnosticSignal.RETRIEVAL_ERROR] == 1
    with pytest.raises(ArtifactCompatibilityError, match="already exists"):
        runner.run(questions, output, top_k=1, candidate_k=1)


def test_diagnostics_without_reference_answer_omits_answer_coverage(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "public.json"
    questions.write_text(
        json.dumps({"q-public": {"question": "Điều 10 quy định gì?"}}),
        encoding="utf-8",
    )

    report = RetrievalDiagnosticsRunner(
        _Runtime(), application_config_sha256="c" * 64
    ).run(questions, tmp_path / "diagnostics", top_k=2, candidate_k=4)

    case = report.cases[0]
    assert case.answer_term_coverage is None
    assert report.mean_answer_term_coverage is None
    assert RetrievalDiagnosticSignal.LOW_ANSWER_TERM_COVERAGE not in case.signals


def test_diagnostics_optionally_compares_hybrid_with_reranker(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "development.json"
    _questions(questions)

    report = RetrievalDiagnosticsRunner(
        _Runtime(), application_config_sha256="d" * 64
    ).run(
        questions,
        tmp_path / "diagnostics",
        top_k=2,
        candidate_k=4,
        include_reranker=True,
    )

    case = report.cases[0]
    assert report.schema_version == "1.1"
    assert report.include_reranker is True
    assert len(case.branches) == 4
    assert case.hybrid_rerank_overlap_count == 1
    assert case.hybrid_rerank_jaccard == pytest.approx(1 / 3)
    assert case.hybrid_rerank_document_diversity == 1.0
    assert case.mean_absolute_rank_change == 1.0
    assert case.hybrid_rerank_answer_term_coverage_delta == pytest.approx(0.0)
    assert report.mean_hybrid_rerank_jaccard == pytest.approx(1 / 3)
