"""Integration test across fixed tools and bounded Agent orchestration."""

from dataclasses import dataclass

from legal_agentic_rag.agent import DeterministicAgentWorkflow
from legal_agentic_rag.contracts import AgentWorkflow
from legal_agentic_rag.generation import (
    ExtractiveAnswerGenerator,
    ModelBackedCitationVerifier,
    RuleBasedCitationVerifier,
    RuleBasedContextGrader,
)
from legal_agentic_rag.schemas import (
    AgentStopReason,
    AnswerResponse,
    Citation,
    QueryAnalysis,
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


@dataclass
class _ReferenceRetriever:
    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        strategy = query.requested_strategy
        assert strategy is not None
        hits = [
            RetrievalHit(
                chunk_id="wrong-reference",
                document_id="wrong-document",
                rank=1,
                score=2.0,
                strategy=strategy,
                text="Điều 113. Nội dung thuộc văn bản khác.",
                metadata={
                    "document_number": "145/2020/NĐ-CP",
                    "structure": {"article_number": "113"},
                },
            ),
            RetrievalHit(
                chunk_id="exact-reference",
                document_id="exact-document",
                rank=2,
                score=1.0,
                strategy=strategy,
                text="Điều 113. Người lao động được nghỉ hằng năm.",
                metadata={
                    "document_number": "45/2019/QH14",
                    "structure": {"article_number": "113"},
                },
            ),
        ]
        return RetrievalResponse(
            query=query,
            strategy=strategy,
            hits=hits,
            artifact_versions={"legal_chunks": "1.0"},
        )


class _UnsupportedSyntheticGenerator:
    def generate(
        self,
        query: RetrievalQuery,
        evidence: list,
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
    ) -> AnswerResponse:
        source = evidence[0]
        return AnswerResponse(
            question=query.original_question,
            answer="Doanh nghiệp được miễn nghĩa vụ trong 99 năm. [E1]",
            citations=[
                Citation(
                    evidence_id=source.evidence_id,
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    document_title=source.document_title,
                    document_number=source.document_number,
                    article_number=source.article_number,
                    source_url=source.source_url,
                )
            ],
            insufficient_evidence=False,
            retrieval_strategy=retrieval_strategy,
            trace_id=trace_id,
            metadata={"semantic_synthesis": True},
        )


class _SupportedSyntheticGenerator:
    def generate(
        self,
        query: RetrievalQuery,
        evidence: list,
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
    ) -> AnswerResponse:
        source = evidence[0]
        return AnswerResponse(
            question=query.original_question,
            answer="Doanh nghiệp phải tuân thủ quy định pháp luật. [E1]",
            citations=[
                Citation(
                    evidence_id=source.evidence_id,
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    document_title=source.document_title,
                    document_number=source.document_number,
                    article_number=source.article_number,
                    source_url=source.source_url,
                )
            ],
            insufficient_evidence=False,
            retrieval_strategy=retrieval_strategy,
            trace_id=trace_id,
            metadata={"semantic_synthesis": True},
        )


@dataclass
class _SemanticProvider:
    label: str
    provider_name: str = "fixture"
    provider_version: str = "1.0"
    model_name: str = "fixture-semantic-verifier"
    model_revision: str = "fixture-revision"

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        assert "CLAIMS_AND_CITED_EVIDENCE_JSON" in user_prompt
        assert system_instruction
        return (
            '{"assessments":[{"claim_id":"C1","label":'
            f'"{self.label}"'
            "}]}"
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
    assert "citation_verification" in result.response.metadata


def test_agent_selects_exact_user_reference_before_higher_raw_rank() -> None:
    """Applicability selection is preserved through grading and generation."""
    registry = build_fixed_tool_registry(
        retriever=_ReferenceRetriever(),
        context_grader=RuleBasedContextGrader(),
        answer_generator=ExtractiveAnswerGenerator(),
        citation_verifier=RuleBasedCitationVerifier(),
    )
    workflow = DeterministicAgentWorkflow(registry)
    query = RetrievalQuery(
        query_id="agent-reference-integration",
        original_question="Điều 113 Luật 45/2019/QH14 quy định gì?",
        normalized_question="Điều 113 Luật 45/2019/QH14 quy định gì?",
        query_analysis=QueryAnalysis(
            document_numbers=["45/2019/QH14"],
            article_numbers=["113"],
        ),
        top_k=2,
        candidate_k=2,
    )

    result = workflow.run(query)

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert [
        item.chunk_id for item in result.state.selected_evidence
    ][:2] == ["exact-reference", "wrong-reference"]
    assert result.state.selected_evidence[0].metadata[
        "evidence_selection"
    ]["applicability"] == "explicit_match"
    assert result.state.context_grade is not None
    assert result.state.context_grade.metadata["reference_coverage"] == {
        "document": True,
        "article": True,
    }


def test_agent_abstains_when_synthesized_claim_is_not_grounded() -> None:
    """Claim-level failure prevents a structurally cited answer from escaping."""
    registry = build_fixed_tool_registry(
        retriever=_FixedRetriever(),
        context_grader=RuleBasedContextGrader(),
        answer_generator=_UnsupportedSyntheticGenerator(),
        citation_verifier=RuleBasedCitationVerifier(),
    )
    result = DeterministicAgentWorkflow(registry).run(
        RetrievalQuery(
            query_id="unsupported-claim-integration",
            original_question="Doanh nghiệp phải thực hiện nghĩa vụ nào?",
            normalized_question="nghĩa vụ doanh nghiệp",
            top_k=1,
            candidate_k=1,
        )
    )

    assert result.stop_reason == AgentStopReason.CITATION_VERIFICATION_FAILED
    assert result.response.insufficient_evidence is True
    verification = result.response.metadata["citation_verification"]
    assert verification["claim_level_verification_performed"] is True
    assert verification["claim_coverage_score"] == 0.0
    assert "unsupported_claim:C1" in verification["errors"]


def test_agent_abstains_when_semantic_model_contradicts_grounded_claim() -> None:
    """A semantic failure after hard checks cannot escape as a final answer."""
    verifier = ModelBackedCitationVerifier(
        RuleBasedCitationVerifier(),
        _SemanticProvider("contradicted"),
        max_structured_output_retries=0,
    )
    registry = build_fixed_tool_registry(
        retriever=_FixedRetriever(),
        context_grader=RuleBasedContextGrader(),
        answer_generator=_SupportedSyntheticGenerator(),
        citation_verifier=verifier,
    )

    result = DeterministicAgentWorkflow(registry).run(
        RetrievalQuery(
            query_id="semantic-failure-integration",
            original_question="Doanh nghiệp phải tuân thủ điều gì?",
            normalized_question="doanh nghiệp tuân thủ",
            top_k=1,
            candidate_k=1,
        )
    )

    assert result.stop_reason == AgentStopReason.CITATION_VERIFICATION_FAILED
    assert result.response.insufficient_evidence is True
    verification = result.response.metadata["citation_verification"]
    assert verification["semantic_verification"]["is_valid"] is False
    assert "semantic_contradicted:C1" in verification["errors"]
