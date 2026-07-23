"""Tests for grading, generation, and citation verification wrappers."""

from legal_agentic_rag.generation import (
    ExtractiveAnswerGenerator,
    RuleBasedCitationVerifier,
    RuleBasedContextGrader,
)
from legal_agentic_rag.schemas import (
    AnswerGenerationInput,
    CitationVerificationInput,
    ContextGradingInput,
    Evidence,
    RetrievalQuery,
    RetrievalStrategy,
    ToolName,
)
from legal_agentic_rag.tools import (
    AnswerGenerationTool,
    CitationVerificationTool,
    ContextGradingTool,
    TypedTool,
)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="generation-tool",
        original_question="Quy định thế nào?",
        normalized_question="quy định",
        top_k=1,
        candidate_k=1,
    )


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Không áp dụng trong trường hợp ngoại lệ.",
        article_number="5",
    )


def test_generation_wrappers_preserve_typed_contracts_and_boundaries() -> None:
    """Each wrapper exposes one capability and delegates only supplied data."""
    grader_tool = ContextGradingTool(RuleBasedContextGrader())
    generator_tool = AnswerGenerationTool(ExtractiveAnswerGenerator())
    verifier_tool = CitationVerificationTool(RuleBasedCitationVerifier())
    query = _query()
    evidence = [_evidence()]

    grade = grader_tool.invoke(
        ContextGradingInput(query=query, evidence=evidence)
    )
    answer = generator_tool.invoke(
        AnswerGenerationInput(
            query=query,
            evidence=evidence,
            retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
            trace_id=query.query_id,
        )
    )
    verification = verifier_tool.invoke(
        CitationVerificationInput(response=answer, evidence=evidence)
    )

    assert grade.is_sufficient is True
    assert answer.citations[0].evidence_id == "E1"
    assert verification.is_valid is True
    assert [
        grader_tool.name,
        generator_tool.name,
        verifier_tool.name,
    ] == [
        ToolName.CONTEXT_GRADING,
        ToolName.ANSWER_GENERATION,
        ToolName.CITATION_VERIFICATION,
    ]
    assert all(
        isinstance(tool, TypedTool)
        for tool in (grader_tool, generator_tool, verifier_tool)
    )
