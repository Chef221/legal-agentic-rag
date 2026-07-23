"""Integration test across fixed tools and bounded Agent orchestration."""

from dataclasses import dataclass

from legal_agentic_rag.agent import DeterministicAgentWorkflow
from legal_agentic_rag.contracts import AgentWorkflow
from legal_agentic_rag.generation import (
    ExtractiveAnswerGenerator,
    RuleBasedCitationVerifier,
    RuleBasedContextGrader,
)
from legal_agentic_rag.schemas import (
    AgentStopReason,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.tools import build_fixed_tool_registry


@dataclass
class _FixedRetriever:
    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        strategy = query.requested_strategy
        assert strategy is not None
        return RetrievalResponse(
            query=query,
            strategy=strategy,
            hits=[
                RetrievalHit(
                    chunk_id="chunk-agent-integration",
                    document_id="doc-agent-integration",
                    rank=1,
                    score=1.0,
                    strategy=strategy,
                    text="Điều 10. Doanh nghiệp phải tuân thủ quy định pháp luật.",
                    metadata={
                        "document_title": "Luật thử nghiệm",
                        "document_number": "01/2026/QH",
                        "source_url": "https://example.invalid/law",
                        "structure": {"article_number": "10"},
                    },
                )
            ],
            artifact_versions={"legal_chunks": "1.0"},
        )


def test_fixed_registry_runs_through_agent_to_verified_answer() -> None:
    """The Agent composes retrieval, grading, generation, and verification tools."""
    registry = build_fixed_tool_registry(
        retriever=_FixedRetriever(),
        context_grader=RuleBasedContextGrader(),
        answer_generator=ExtractiveAnswerGenerator(),
        citation_verifier=RuleBasedCitationVerifier(),
    )
    workflow = DeterministicAgentWorkflow(registry)
    query = RetrievalQuery(
        query_id="agent-integration",
        original_question="Doanh nghiệp phải tuân thủ điều gì?",
        normalized_question="doanh nghiệp tuân thủ",
        top_k=1,
        candidate_k=2,
    )

    result = workflow.run(query)

    assert isinstance(workflow, AgentWorkflow)
    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert result.response.citations[0].article_number == "10"
    assert result.state.selected_evidence[0].chunk_id == (
        "chunk-agent-integration"
    )
    assert result.response.metadata["agent"]["attempt_count"] == 1
