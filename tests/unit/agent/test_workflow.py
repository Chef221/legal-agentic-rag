"""Unit tests for bounded Agent state transitions and failure paths."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from legal_agentic_rag.agent import DeterministicAgentWorkflow
from legal_agentic_rag.configuration import AgentConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    OperationTimeoutError,
    RetrievalError,
)
from legal_agentic_rag.generation import (
    ExtractiveAnswerGenerator,
    RuleBasedCitationVerifier,
    RuleBasedContextGrader,
)
from legal_agentic_rag.schemas import (
    AgentStopReason,
    AnswerResponse,
    CitationVerificationResult,
    Evidence,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    ToolName,
)
from legal_agentic_rag.tools import (
    AnswerGenerationTool,
    CitationVerificationTool,
    ContextGradingTool,
    RetrievalTool,
    ToolRegistry,
    build_fixed_tool_registry,
)


@dataclass
class _SequencedRetriever:
    """Return an evidence hit only on configured calls or raise a domain error."""

    hit_calls: set[int] = field(default_factory=set)
    error: Exception | None = None
    errors_by_call: dict[int, Exception] = field(default_factory=dict)
    calls: list[RetrievalStrategy] = field(default_factory=list)

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        strategy = query.requested_strategy
        assert strategy is not None
        self.calls.append(strategy)
        call_number = len(self.calls)
        if call_number in self.errors_by_call:
            raise self.errors_by_call[call_number]
        if self.error is not None:
            raise self.error
        hits = (
            [
                RetrievalHit(
                    chunk_id=f"chunk-{call_number}",
                    document_id="doc-1",
                    rank=1,
                    score=1.0,
                    strategy=strategy,
                    text="Điều 10 quy định doanh nghiệp phải thực hiện nghĩa vụ.",
                    metadata={
                        "document_number": "01/2026/QH",
                        "structure": {"article_number": "10"},
                    },
                )
            ]
            if call_number in self.hit_calls
            else []
        )
        return RetrievalResponse(
            query=query,
            strategy=strategy,
            hits=hits,
            warnings=[] if hits else ["no_matches"],
        )


class _InvalidCitationVerifier:
    def verify(
        self,
        response: AnswerResponse,
        evidence: Sequence[Evidence],
    ) -> CitationVerificationResult:
        return CitationVerificationResult(
            is_valid=False,
            invalid_citations=list(response.citations),
            errors=["forced_invalid_citation"],
        )


class _AbstainingGenerator:
    """Return a valid generator-level abstention despite sufficient context."""

    def generate(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
    ) -> AnswerResponse:
        return AnswerResponse(
            question=query.original_question,
            answer="Khong du can cu de tra loi.",
            insufficient_evidence=True,
            warnings=["model_reported_insufficient_evidence"],
            retrieval_strategy=retrieval_strategy,
            trace_id=trace_id,
        )


class _WarningGenerator:
    """Attach recovery telemetry to an otherwise grounded response."""

    def generate(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
    ) -> AnswerResponse:
        response = ExtractiveAnswerGenerator().generate(
            query,
            evidence,
            retrieval_strategy,
            trace_id,
        )
        return response.model_copy(
            update={"warnings": ["grounding_repair_unresolved"]}
        )


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="agent-workflow",
        original_question="Doanh nghiệp phải thực hiện nghĩa vụ nào?",
        normalized_question="nghĩa vụ doanh nghiệp",
        top_k=1,
        candidate_k=2,
    )


def _workflow(
    retriever: _SequencedRetriever,
    *,
    config: AgentConfig | None = None,
    verifier: object | None = None,
    generator: object | None = None,
) -> DeterministicAgentWorkflow:
    registry = build_fixed_tool_registry(
        retriever=retriever,
        context_grader=RuleBasedContextGrader(),
        answer_generator=generator or ExtractiveAnswerGenerator(),
        citation_verifier=verifier or RuleBasedCitationVerifier(),
    )
    return DeterministicAgentWorkflow(registry, agent_config=config)


def test_agent_stops_after_first_sufficient_context_and_verifies_answer() -> None:
    """Sufficient first-pass evidence does not trigger unnecessary retries."""
    retriever = _SequencedRetriever(hit_calls={1})

    result = _workflow(retriever).run(_query())

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert result.response.insufficient_evidence is False
    assert result.state.retry_count == 0
    assert len(result.state.retrieval_history) == 1
    assert retriever.calls == [RetrievalStrategy.HYBRID_RERANK]
    assert [
        item["tool_name"]
        for item in result.state.metadata["tool_invocations"]
    ] == [
        ToolName.RERANK_SEARCH.value,
        ToolName.CONTEXT_GRADING.value,
        ToolName.ANSWER_GENERATION.value,
        ToolName.CITATION_VERIFICATION.value,
    ]


def test_agent_rewrites_changes_strategy_and_stops_on_second_success() -> None:
    """An insufficient first attempt retries with a new safe query and tool."""
    retriever = _SequencedRetriever(hit_calls={2})

    result = _workflow(retriever).run(_query())

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert result.state.retry_count == 1
    assert result.state.current_query == _query().original_question
    assert retriever.calls == [
        RetrievalStrategy.HYBRID_RERANK,
        RetrievalStrategy.GRAPH,
    ]
    assert [item.attempt_number for item in result.state.retrieval_history] == [
        1,
        2,
    ]


def test_agent_enforces_max_retry_and_returns_abstention() -> None:
    """Three failed context grades terminate at the fixed two-retry cap."""
    retriever = _SequencedRetriever()

    result = _workflow(retriever).run(_query())

    assert result.stop_reason == AgentStopReason.MAX_RETRY_REACHED
    assert result.response.insufficient_evidence is True
    assert result.state.retry_count == 2
    assert len(result.state.retrieval_history) == 3
    assert retriever.calls == [
        RetrievalStrategy.HYBRID_RERANK,
        RetrievalStrategy.GRAPH,
        RetrievalStrategy.HYBRID,
    ]


def test_agent_stops_when_no_new_registered_strategy_remains() -> None:
    """A shorter configured route terminates without repeating a tool."""
    retriever = _SequencedRetriever()
    config = AgentConfig(
        strategy_order=[RetrievalStrategy.HYBRID],
        max_retry=2,
    )

    result = _workflow(retriever, config=config).run(_query())

    assert result.stop_reason == AgentStopReason.NO_NEW_STRATEGY
    assert result.state.retry_count == 0
    assert retriever.calls == [RetrievalStrategy.HYBRID]


def test_agent_stops_immediately_on_timeout_or_non_retryable_error() -> None:
    """Timeout and artifact incompatibility never enter an uncontrolled loop."""
    timeout_retriever = _SequencedRetriever(
        error=OperationTimeoutError("private timeout detail")
    )
    artifact_retriever = _SequencedRetriever(
        error=ArtifactCompatibilityError("private artifact detail")
    )

    timeout = _workflow(timeout_retriever).run(_query())
    artifact = _workflow(artifact_retriever).run(_query())

    assert timeout.stop_reason == AgentStopReason.TIMEOUT
    assert artifact.stop_reason == AgentStopReason.NON_RETRYABLE_TOOL_ERROR
    assert timeout.state.retry_count == artifact.state.retry_count == 0
    assert len(timeout_retriever.calls) == len(artifact_retriever.calls) == 1
    assert "private" not in " ".join(timeout.response.warnings)
    assert "private" not in " ".join(artifact.response.warnings)


def test_agent_retries_a_sanitized_retrieval_error_then_recovers() -> None:
    """A retryable retrieval failure changes tool and can still produce an answer."""
    retriever = _SequencedRetriever(
        hit_calls={2},
        errors_by_call={1: RetrievalError("private backend detail")},
    )

    result = _workflow(retriever).run(_query())

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert result.state.retry_count == 1
    assert result.state.retrieval_history[0].error_type == "retrieval_error"
    assert "private" not in " ".join(result.response.warnings)


def test_agent_fails_closed_when_citation_verification_rejects_answer() -> None:
    """An unverified generated answer is replaced by an explicit abstention."""
    retriever = _SequencedRetriever(hit_calls={1})

    result = _workflow(
        retriever,
        verifier=_InvalidCitationVerifier(),
    ).run(_query())

    assert result.stop_reason == AgentStopReason.CITATION_VERIFICATION_FAILED
    assert result.response.insufficient_evidence is True
    assert result.response.citations == []
    assert "forced_invalid_citation" in result.response.warnings


def test_agent_preserves_failed_generation_recovery_telemetry() -> None:
    """Citation fallback reports the generator recovery path that preceded it."""
    retriever = _SequencedRetriever(hit_calls={1})

    result = _workflow(
        retriever,
        verifier=_InvalidCitationVerifier(),
        generator=_WarningGenerator(),
    ).run(_query())

    assert result.stop_reason == AgentStopReason.CITATION_VERIFICATION_FAILED
    assert "grounding_repair_unresolved" in result.response.warnings


def test_agent_preserves_generator_abstention_without_marking_it_verified() -> None:
    """A model abstention is terminal and never masquerades as a verified answer."""
    retriever = _SequencedRetriever(hit_calls={1})

    result = _workflow(
        retriever,
        generator=_AbstainingGenerator(),
    ).run(_query())

    assert result.stop_reason == AgentStopReason.GENERATION_FAILED
    assert result.response.insufficient_evidence is True
    assert result.response.citations == []
    assert "generator:insufficient_evidence" in result.response.warnings
    assert [
        item["tool_name"]
        for item in result.state.metadata["tool_invocations"]
    ] == [
        ToolName.RERANK_SEARCH.value,
        ToolName.CONTEXT_GRADING.value,
        ToolName.ANSWER_GENERATION.value,
    ]


def test_agent_calls_only_the_retrieval_tool_present_in_closed_registry() -> None:
    """Routing cannot invoke configured capabilities absent from the registry."""
    retriever = _SequencedRetriever(hit_calls={1})
    registry = ToolRegistry(
        [
            RetrievalTool(ToolName.HYBRID_SEARCH, retriever),
            ContextGradingTool(RuleBasedContextGrader()),
            AnswerGenerationTool(ExtractiveAnswerGenerator()),
            CitationVerificationTool(RuleBasedCitationVerifier()),
        ]
    )

    result = DeterministicAgentWorkflow(registry).run(_query())

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert retriever.calls == [RetrievalStrategy.HYBRID]

def test_build_fixed_tool_registry_omits_graph_when_disabled() -> None:
    """When graph_runtime_enabled is False, GRAPH_SEARCH is excluded from registry."""
    retriever = _SequencedRetriever()
    registry = build_fixed_tool_registry(
        retriever=retriever,
        context_grader=RuleBasedContextGrader(),
        answer_generator=ExtractiveAnswerGenerator(),
        citation_verifier=RuleBasedCitationVerifier(),
        graph_runtime_enabled=False,
    )

    tool_names = [descriptor.name for descriptor in registry.descriptors()]
    assert len(tool_names) == 7
    assert ToolName.GRAPH_SEARCH not in tool_names
    assert ToolName.BM25_SEARCH in tool_names
    assert ToolName.DENSE_SEARCH in tool_names
    assert ToolName.HYBRID_SEARCH in tool_names
    assert ToolName.RERANK_SEARCH in tool_names
    assert ToolName.CONTEXT_GRADING in tool_names
    assert ToolName.ANSWER_GENERATION in tool_names
    assert ToolName.CITATION_VERIFICATION in tool_names
