"""Integration across the closed tool registry without an Agent framework."""

from dataclasses import dataclass

import numpy as np

from legal_agentic_rag.generation import (
    ContextBuilder,
    ExtractiveAnswerGenerator,
    RuleBasedCitationVerifier,
    RuleBasedContextGrader,
)
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import FixedRetriever
from legal_agentic_rag.schemas import (
    AnswerResponse,
    CitationVerificationResult,
    ContextGrade,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    ToolInvocationRequest,
    ToolName,
)
from legal_agentic_rag.tools import build_fixed_tool_registry


@dataclass
class _Branch:
    strategy: RetrievalStrategy
    source_artifact_identity: tuple[str, str, str] = (
        "legal_chunks",
        "1.0",
        "tool-chunks-hash",
    )

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        return RetrievalResponse(
            query=query,
            strategy=self.strategy,
            hits=[
                RetrievalHit(
                    chunk_id="chunk-tool",
                    document_id="doc-tool",
                    rank=1,
                    score=2.0 if self.strategy == RetrievalStrategy.BM25 else 0.9,
                    strategy=self.strategy,
                    text="Doanh nghiệp phải đáp ứng điều kiện theo Điều 10.",
                    metadata={
                        "token_count": 10,
                        "document_title": "Luật thử nghiệm",
                        "document_number": "01/2026/QH",
                        "effect_status": "còn hiệu lực",
                        "structure": {"article_number": "10"},
                    },
                )
            ],
            artifact_versions={f"{self.strategy.value}_index": "1.0"},
        )


class _CrossEncoder:
    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object:
        return np.asarray([1.0 for _ in inputs], dtype=np.float32)


def _request(
    invocation_id: str,
    tool_name: ToolName,
    payload: dict[str, object],
) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        invocation_id=invocation_id,
        tool_name=tool_name,
        payload=payload,
    )


def test_registered_tools_run_retrieval_grade_generate_and_verify() -> None:
    """Typed tools compose the fixed workflow while the Agent remains absent."""
    fixed_retriever = FixedRetriever(
        _Branch(RetrievalStrategy.BM25),
        _Branch(RetrievalStrategy.DENSE),
        reranker=CrossEncoderReranker(
            model_loader=lambda config: _CrossEncoder()
        ),
    )
    registry = build_fixed_tool_registry(
        retriever=fixed_retriever,
        context_grader=RuleBasedContextGrader(),
        answer_generator=ExtractiveAnswerGenerator(),
        citation_verifier=RuleBasedCitationVerifier(),
        verification_timeout_seconds=45.0,
    )
    descriptors = {item.name: item for item in registry.descriptors()}
    assert (
        descriptors[ToolName.CITATION_VERIFICATION].timeout_seconds
        == 45.0
    )
    query = RetrievalQuery(
        query_id="tool-workflow",
        original_question="Doanh nghiệp phải đáp ứng điều kiện nào?",
        normalized_question="điều kiện doanh nghiệp",
        top_k=1,
        candidate_k=2,
    )

    retrieval_result = registry.execute(
        _request(
            "invoke-retrieval",
            ToolName.RERANK_SEARCH,
            query.model_dump(mode="json"),
        )
    )
    retrieval = RetrievalResponse.model_validate(retrieval_result.output)
    context = ContextBuilder().build(retrieval)
    grading_result = registry.execute(
        _request(
            "invoke-grade",
            ToolName.CONTEXT_GRADING,
            {
                "query": query.model_dump(mode="json"),
                "evidence": [
                    item.model_dump(mode="json") for item in context.evidence
                ],
            },
        )
    )
    grade = ContextGrade.model_validate(grading_result.output)
    generation_result = registry.execute(
        _request(
            "invoke-generation",
            ToolName.ANSWER_GENERATION,
            {
                "query": query.model_dump(mode="json"),
                "evidence": [
                    item.model_dump(mode="json") for item in context.evidence
                ],
                "retrieval_strategy": retrieval.strategy.value,
                "trace_id": query.query_id,
            },
        )
    )
    answer = AnswerResponse.model_validate(generation_result.output)
    verification_result = registry.execute(
        _request(
            "invoke-verification",
            ToolName.CITATION_VERIFICATION,
            {
                "response": answer.model_dump(mode="json"),
                "evidence": [
                    item.model_dump(mode="json") for item in context.evidence
                ],
            },
        )
    )
    verification = CitationVerificationResult.model_validate(
        verification_result.output
    )

    assert len(registry.descriptors()) == 8
    assert all(
        result.success
        for result in (
            retrieval_result,
            grading_result,
            generation_result,
            verification_result,
        )
    )
    assert retrieval.strategy == RetrievalStrategy.HYBRID_RERANK
    assert grade.is_sufficient is True
    assert answer.citations[0].article_number == "10"
    assert verification.is_valid is True
