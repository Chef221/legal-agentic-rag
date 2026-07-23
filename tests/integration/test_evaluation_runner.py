"""Integration test from labeled cases through runtime calls and reports."""

from types import SimpleNamespace

from legal_agentic_rag.configuration import EvaluationConfig
from legal_agentic_rag.evaluation import EvaluationRunner
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    EvaluationCase,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


class _Runtime:
    manifests = {}

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.BM25,
            hits=[
                RetrievalHit(
                    chunk_id="chunk-gold",
                    document_id="doc-gold",
                    rank=1,
                    score=1,
                    strategy=RetrievalStrategy.BM25,
                    text="Điều luật liên quan",
                ),
                RetrievalHit(
                    chunk_id="chunk-other",
                    document_id="doc-other",
                    rank=2,
                    score=0.5,
                    strategy=RetrievalStrategy.BM25,
                    text="Điều luật khác",
                ),
            ],
            latency_ms=2,
        )

    def answer(self, query: RetrievalQuery) -> object:
        return SimpleNamespace(
            response=AnswerResponse(
                question=query.original_question,
                answer="Câu trả lời tham chiếu",
                citations=[
                    Citation(
                        evidence_id="E1",
                        chunk_id="chunk-gold",
                        document_id="doc-gold",
                    )
                ],
                insufficient_evidence=False,
                retrieval_strategy=RetrievalStrategy.BM25,
                trace_id=query.query_id,
            )
        )


def test_runner_aggregates_available_metrics_and_resources() -> None:
    """One framework evaluates retrieval, generation, latency, and resources."""
    case = EvaluationCase(
        case_id="case-1",
        question="Quy định là gì?",
        target_granularity="chunk",
        relevance_grades={"chunk-gold": 2},
        reference_answer="Câu trả lời tham chiếu",
        expected_citation_chunk_ids=["chunk-gold"],
        should_abstain=False,
    )
    runner = EvaluationRunner(
        _Runtime(),
        EvaluationConfig(
            cutoffs=[1, 2],
            strategy=RetrievalStrategy.BM25,
            candidate_k=2,
        ),
    )

    result = runner.run(
        [case],
        benchmark_name="fixture.jsonl",
        benchmark_sha256="a" * 64,
    )

    assert result.summary.case_count == 1
    assert result.summary.failed_case_count == 0
    assert result.summary.retrieval_metrics["recall@1"] == 1.0
    assert result.summary.retrieval_metrics["mrr"] == 1.0
    assert result.summary.generation_metrics["exact_match"] == 1.0
    assert result.summary.generation_metrics["citation_recall"] == 1.0
    assert result.summary.metric_case_counts["exact_match"] == 1
    assert result.summary.retrieval_latency.count == 1
    assert result.summary.generation_latency.count == 1
    assert result.summary.resources.wall_time_ms >= 0
    assert result.cases[0].retrieved_ids == [
        "chunk-gold",
        "chunk-other",
    ]
