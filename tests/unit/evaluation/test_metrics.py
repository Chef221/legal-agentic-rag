"""Exact metric tests for retrieval and generation evaluation."""

from legal_agentic_rag.evaluation import (
    StandardGenerationEvaluator,
    StandardRetrievalEvaluator,
)
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    EvaluationCase,
    EvaluationTargetGranularity,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


def _response(identities: list[tuple[str, str]]) -> RetrievalResponse:
    query = RetrievalQuery(
        query_id="query",
        original_question="Câu hỏi",
        normalized_question="Câu hỏi",
        top_k=10,
        candidate_k=10,
    )
    return RetrievalResponse(
        query=query,
        strategy=RetrievalStrategy.HYBRID,
        hits=[
            RetrievalHit(
                chunk_id=chunk_id,
                document_id=document_id,
                rank=rank,
                score=1 / rank,
                strategy=RetrievalStrategy.HYBRID,
                text="Căn cứ pháp luật",
            )
            for rank, (chunk_id, document_id) in enumerate(identities, 1)
        ],
    )


def test_retrieval_metrics_use_explicit_grades_and_standard_denominators() -> None:
    """Recall, Precision, MRR, and NDCG use stable labeled identities."""
    case = EvaluationCase(
        case_id="case-1",
        question="Câu hỏi",
        target_granularity=EvaluationTargetGranularity.CHUNK,
        relevance_grades={"chunk-a": 3, "chunk-b": 1},
    )

    metrics = StandardRetrievalEvaluator().evaluate(
        case,
        _response(
            [
                ("chunk-b", "doc-1"),
                ("chunk-x", "doc-2"),
                ("chunk-a", "doc-3"),
            ]
        ),
        [1, 3],
    )

    assert metrics.recall_at_k == {1: 0.5, 3: 1.0}
    assert metrics.precision_at_k == {1: 1.0, 3: 2 / 3}
    assert metrics.reciprocal_rank == 1.0
    assert metrics.first_relevant_rank == 1
    assert metrics.ndcg_at_k[1] == 1 / 7
    assert 0 < metrics.ndcg_at_k[3] < 1


def test_document_metrics_deduplicate_multiple_chunks_from_same_document() -> None:
    """Document-level labels do not count multiple chunks as separate ranks."""
    case = EvaluationCase(
        case_id="case-doc",
        question="Câu hỏi",
        target_granularity=EvaluationTargetGranularity.DOCUMENT,
        relevance_grades={"doc-b": 1},
    )

    metrics = StandardRetrievalEvaluator().evaluate(
        case,
        _response(
            [
                ("chunk-a1", "doc-a"),
                ("chunk-a2", "doc-a"),
                ("chunk-b1", "doc-b"),
            ]
        ),
        [2],
    )

    assert metrics.first_relevant_rank == 2
    assert metrics.reciprocal_rank == 0.5
    assert metrics.recall_at_k[2] == 1.0


def test_generation_metrics_only_score_available_labels() -> None:
    """Missing semantic labels remain null instead of becoming fake zeros."""
    case = EvaluationCase(
        case_id="case-answer",
        question="Câu hỏi",
        target_granularity=EvaluationTargetGranularity.CHUNK,
        relevance_grades={"chunk-a": 1},
        reference_answer="Doanh nghiệp phải nộp thuế.",
        expected_citation_chunk_ids=["chunk-a", "chunk-b"],
        should_abstain=False,
    )
    response = AnswerResponse(
        question="Câu hỏi",
        answer=" doanh nghiệp   phải nộp thuế. ",
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk-a",
                document_id="doc-a",
            ),
            Citation(
                evidence_id="E2",
                chunk_id="chunk-x",
                document_id="doc-x",
            ),
        ],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        trace_id="trace",
    )

    metrics = StandardGenerationEvaluator().evaluate(case, response)

    assert metrics.exact_match == 1.0
    assert metrics.abstention_accuracy == 1.0
    assert metrics.citation_precision == 0.5
    assert metrics.citation_recall == 0.5


def test_generation_metrics_are_null_without_generation_labels() -> None:
    """The framework never claims correctness without reference labels."""
    case = EvaluationCase(
        case_id="unlabeled",
        question="Câu hỏi",
        target_granularity="chunk",
        relevance_grades={"chunk-a": 1},
    )
    response = AnswerResponse(
        question="Câu hỏi",
        answer="Câu trả lời",
        insufficient_evidence=True,
        retrieval_strategy=RetrievalStrategy.BM25,
        trace_id="trace",
    )

    metrics = StandardGenerationEvaluator().evaluate(case, response)

    assert all(value is None for value in metrics.model_dump().values())
