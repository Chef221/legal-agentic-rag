"""Bounded deterministic Agent workflow over the closed tool registry."""

from __future__ import annotations

import logging
from time import perf_counter

from legal_agentic_rag.configuration.online import (
    AgentConfig,
    EvidenceSelectionConfig,
    GenerationConfig,
)
from legal_agentic_rag.exceptions import ConfigurationError, DataValidationError
from legal_agentic_rag.generation.context_builder import ContextBuilder
from legal_agentic_rag.schemas.agent_state import (
    AgentRunResult,
    AgentState,
    AgentStopReason,
    RetrievalHistoryItem,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    CitationVerificationResult,
    ContextBuildResult,
    ContextGrade,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.schemas.tools import (
    ToolErrorType,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolName,
)
from legal_agentic_rag.agent.query_rewriter import ConservativeQueryRewriter
from legal_agentic_rag.agent.router import (
    DeterministicStrategyRouter,
    RetrievalRoute,
)
from legal_agentic_rag.tools.registry import ToolRegistry

_ABSTENTION_TEXT = (
    "Hệ thống chưa tìm thấy căn cứ pháp luật đủ rõ trong dữ liệu hiện có "
    "để trả lời chắc chắn."
)
_REQUIRED_FINAL_TOOLS = {
    ToolName.CONTEXT_GRADING,
    ToolName.ANSWER_GENERATION,
    ToolName.CITATION_VERIFICATION,
}
_RETRIEVAL_TOOLS = {
    ToolName.BM25_SEARCH,
    ToolName.DENSE_SEARCH,
    ToolName.HYBRID_SEARCH,
    ToolName.RERANK_SEARCH,
    ToolName.GRAPH_SEARCH,
}
_LOGGER = logging.getLogger(__name__)


class DeterministicAgentWorkflow:
    """Select, observe, retry, generate, and verify with registered tools only."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        agent_config: AgentConfig | None = None,
        generation_config: GenerationConfig | None = None,
        evidence_selection_config: EvidenceSelectionConfig | None = None,
        context_builder: ContextBuilder | None = None,
        router: DeterministicStrategyRouter | None = None,
        query_rewriter: ConservativeQueryRewriter | None = None,
    ) -> None:
        self._registry = registry
        self._config = agent_config or AgentConfig()
        self._router = router or DeterministicStrategyRouter(self._config)
        self._rewriter = query_rewriter or ConservativeQueryRewriter()
        self._context_builder = context_builder or ContextBuilder(
            generation_config,
            evidence_selection_config,
        )
        registered = {descriptor.name for descriptor in registry.descriptors()}
        missing = _REQUIRED_FINAL_TOOLS - registered
        if missing:
            raise ConfigurationError("Agent registry is missing required online tools")
        if not registered.intersection(_RETRIEVAL_TOOLS):
            raise ConfigurationError("Agent registry has no retrieval tool")
        self._registered_tools = registered

    def run(self, query: RetrievalQuery) -> AgentRunResult:
        """Run at most three retrieval attempts and return a verified answer."""
        started = perf_counter()
        routes = self._router.plan(query, self._registered_tools)
        if not routes:
            raise ConfigurationError("No configured Agent strategy is registered")

        history: list[RetrievalHistoryItem] = []
        invocation_trace: list[dict[str, object]] = []
        warnings: list[str] = []
        current_query = query.rewritten_question or query.normalized_question
        used_queries = {current_query}
        latest_response: RetrievalResponse | None = None
        latest_context = self._empty_context()
        latest_grade: ContextGrade | None = None
        stop_reason = AgentStopReason.NO_NEW_STRATEGY

        for attempt_index, route in enumerate(routes, start=1):
            if attempt_index > 1 and self._config.rewrite_query_on_retry:
                rewritten = self._rewriter.rewrite(
                    query,
                    current_query=current_query,
                    previously_used=used_queries,
                )
                if rewritten is not None:
                    current_query = rewritten
                    used_queries.add(rewritten)
                else:
                    warnings.append("query_rewrite_unchanged")

            routed_query = query.model_copy(
                update={
                    "rewritten_question": (
                        current_query
                        if current_query != query.normalized_question
                        else None
                    ),
                    "requested_strategy": route.strategy,
                }
            )
            retrieval_result = self._invoke(
                query.query_id,
                attempt_index,
                "retrieval",
                route.tool_name,
                routed_query.model_dump(mode="json"),
                invocation_trace,
            )
            if not retrieval_result.success:
                error = retrieval_result.error
                history.append(
                    RetrievalHistoryItem(
                        attempt_number=attempt_index,
                        query=routed_query,
                        strategy=route.strategy,
                        error_type=error.error_type.value,
                        warnings=[error.message],
                    )
                )
                warnings.append(f"retrieval:{error.error_type.value}")
                stop_reason = self._failure_stop_reason(error.error_type)
                if (
                    error.error_type == ToolErrorType.TIMEOUT
                    or not error.retryable
                ):
                    break
                stop_reason = self._bounded_stop_reason(attempt_index, routes)
                continue

            latest_response = RetrievalResponse.model_validate(
                retrieval_result.output
            )
            self._validate_retrieval(routed_query, route.strategy, latest_response)
            try:
                latest_context = self._context_builder.build(latest_response)
            except DataValidationError:
                history.append(
                    RetrievalHistoryItem(
                        attempt_number=attempt_index,
                        query=routed_query,
                        strategy=route.strategy,
                        response=latest_response,
                        error_type=ToolErrorType.DATA_VALIDATION_ERROR.value,
                    )
                )
                warnings.append("context:data_validation_error")
                stop_reason = AgentStopReason.NON_RETRYABLE_TOOL_ERROR
                break

            grading_result = self._invoke(
                query.query_id,
                attempt_index,
                "grading",
                ToolName.CONTEXT_GRADING,
                {
                    "query": routed_query.model_dump(mode="json"),
                    "evidence": [
                        item.model_dump(mode="json")
                        for item in latest_context.evidence
                    ],
                },
                invocation_trace,
            )
            if not grading_result.success:
                error = grading_result.error
                history.append(
                    RetrievalHistoryItem(
                        attempt_number=attempt_index,
                        query=routed_query,
                        strategy=route.strategy,
                        response=latest_response,
                        error_type=error.error_type.value,
                        warnings=[error.message],
                    )
                )
                warnings.append(f"grader:{error.error_type.value}")
                stop_reason = self._failure_stop_reason(error.error_type)
                if (
                    error.error_type == ToolErrorType.TIMEOUT
                    or not error.retryable
                ):
                    break
                stop_reason = self._bounded_stop_reason(attempt_index, routes)
                continue

            latest_grade = ContextGrade.model_validate(grading_result.output)
            history.append(
                RetrievalHistoryItem(
                    attempt_number=attempt_index,
                    query=routed_query,
                    strategy=route.strategy,
                    response=latest_response,
                    context_grade=latest_grade,
                    warnings=[
                        *latest_response.warnings,
                        *latest_context.warnings,
                        *latest_grade.warnings,
                    ],
                )
            )
            if latest_grade.is_sufficient:
                response, stop_reason = self._generate_and_verify(
                    query=routed_query,
                    strategy=route.strategy,
                    context=latest_context,
                    attempt_number=attempt_index,
                    invocation_trace=invocation_trace,
                )
                return self._finish(
                    query,
                    current_query,
                    history,
                    latest_response,
                    latest_context,
                    latest_grade,
                    response,
                    stop_reason,
                    warnings,
                    invocation_trace,
                    started,
                )
            warnings.append("insufficient_context")
            stop_reason = self._bounded_stop_reason(attempt_index, routes)

        strategy = (
            history[-1].strategy if history else routes[0].strategy
        )
        response = self._abstention(query, strategy, warnings)
        return self._finish(
            query,
            current_query,
            history,
            latest_response,
            latest_context,
            latest_grade,
            response,
            stop_reason,
            warnings,
            invocation_trace,
            started,
        )

    def _generate_and_verify(
        self,
        *,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        context: ContextBuildResult,
        attempt_number: int,
        invocation_trace: list[dict[str, object]],
    ) -> tuple[AnswerResponse, AgentStopReason]:
        generation = self._invoke(
            query.query_id,
            attempt_number,
            "generation",
            ToolName.ANSWER_GENERATION,
            {
                "query": query.model_dump(mode="json"),
                "evidence": [
                    item.model_dump(mode="json") for item in context.evidence
                ],
                "retrieval_strategy": strategy.value,
                "trace_id": query.query_id,
            },
            invocation_trace,
        )
        if not generation.success:
            reason = (
                AgentStopReason.TIMEOUT
                if generation.error.error_type == ToolErrorType.TIMEOUT
                else AgentStopReason.GENERATION_FAILED
            )
            return (
                self._abstention(
                    query,
                    strategy,
                    [f"generator:{generation.error.error_type.value}"],
                ),
                reason,
            )
        response = AnswerResponse.model_validate(generation.output)
        if (
            response.question != query.original_question
            or response.trace_id != query.query_id
            or response.retrieval_strategy != strategy
        ):
            return (
                self._abstention(query, strategy, ["generator:contract_mismatch"]),
                AgentStopReason.GENERATION_FAILED,
            )
        verification = self._invoke(
            query.query_id,
            attempt_number,
            "verification",
            ToolName.CITATION_VERIFICATION,
            {
                "response": response.model_dump(mode="json"),
                "evidence": [
                    item.model_dump(mode="json") for item in context.evidence
                ],
            },
            invocation_trace,
        )
        if not verification.success:
            reason = (
                AgentStopReason.TIMEOUT
                if verification.error.error_type == ToolErrorType.TIMEOUT
                else AgentStopReason.CITATION_VERIFICATION_FAILED
            )
            return (
                self._abstention(
                    query,
                    strategy,
                    [f"verifier:{verification.error.error_type.value}"],
                ),
                reason,
            )
        result = CitationVerificationResult.model_validate(verification.output)
        if not result.is_valid:
            abstention = self._abstention(
                query,
                strategy,
                ["citation_verification_failed", *result.errors],
            )
            return (
                abstention.model_copy(
                    update={
                        "metadata": {
                            **abstention.metadata,
                            "citation_verification": result.model_dump(
                                mode="json"
                            ),
                        }
                    }
                ),
                AgentStopReason.CITATION_VERIFICATION_FAILED,
            )
        return (
            response.model_copy(
                update={
                    "metadata": {
                        **response.metadata,
                        "citation_verification": result.model_dump(mode="json"),
                    }
                }
            ),
            AgentStopReason.ANSWER_VERIFIED,
        )

    def _finish(
        self,
        query: RetrievalQuery,
        current_query: str,
        history: list[RetrievalHistoryItem],
        retrieval: RetrievalResponse | None,
        context: ContextBuildResult,
        grade: ContextGrade | None,
        response: AnswerResponse,
        stop_reason: AgentStopReason,
        warnings: list[str],
        invocation_trace: list[dict[str, object]],
        started: float,
    ) -> AgentRunResult:
        total_latency_ms = (perf_counter() - started) * 1000
        all_warnings = list(
            dict.fromkeys([*warnings, *response.warnings, *context.warnings])
        )
        response_metadata = dict(response.metadata)
        response_metadata["agent"] = {
            "stop_reason": stop_reason.value,
            "attempt_count": len(history),
            "retry_count": max(0, len(history) - 1),
            "tool_invocations": invocation_trace,
            "total_latency_ms": total_latency_ms,
        }
        final_response = response.model_copy(
            update={"warnings": all_warnings, "metadata": response_metadata}
        )
        state = AgentState(
            trace_id=query.query_id,
            original_question=query.original_question,
            normalized_question=query.normalized_question,
            current_query=current_query,
            selected_strategy=(
                history[-1].strategy if history else query.requested_strategy
            ),
            retrieval_history=history,
            candidate_hits=retrieval.hits if retrieval is not None else [],
            selected_evidence=context.evidence,
            context_grade=grade,
            retry_count=max(0, len(history) - 1),
            answer=final_response.answer,
            citations=final_response.citations,
            warnings=all_warnings,
            metadata={
                "stop_reason": stop_reason.value,
                "tool_invocations": invocation_trace,
                "total_latency_ms": total_latency_ms,
            },
        )
        _LOGGER.info(
            "agent_workflow_completed",
            extra={
                "query_id": query.query_id,
                "trace_id": query.query_id,
                "strategy": (
                    state.selected_strategy.value
                    if state.selected_strategy is not None
                    else None
                ),
                "candidate_count": len(state.candidate_hits),
                "selected_evidence": len(state.selected_evidence),
                "retry_count": state.retry_count,
                "stop_reason": stop_reason.value,
                "latency_ms": total_latency_ms,
            },
        )
        return AgentRunResult(
            state=state,
            response=final_response,
            stop_reason=stop_reason,
            total_latency_ms=total_latency_ms,
        )

    def _invoke(
        self,
        trace_id: str,
        attempt_number: int,
        stage: str,
        tool_name: ToolName,
        payload: dict[str, object],
        invocation_trace: list[dict[str, object]],
    ) -> ToolInvocationResult:
        invocation_id = f"{trace_id}:{attempt_number}:{stage}"
        result = self._registry.execute(
            ToolInvocationRequest(
                invocation_id=invocation_id,
                tool_name=tool_name,
                payload=payload,
            )
        )
        invocation_trace.append(
            {
                "invocation_id": invocation_id,
                "tool_name": tool_name.value,
                "success": result.success,
                "error_type": (
                    result.error.error_type.value if result.error else None
                ),
                "latency_ms": result.latency_ms,
            }
        )
        return result

    @staticmethod
    def _validate_retrieval(
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        response: RetrievalResponse,
    ) -> None:
        if (
            response.query.query_id != query.query_id
            or response.strategy != strategy
            or response.query.requested_strategy != strategy
        ):
            raise DataValidationError(
                "Retrieval tool returned incompatible Agent state"
            )

    @staticmethod
    def _empty_context() -> ContextBuildResult:
        return ContextBuildResult(
            input_hit_count=0,
            selected_count=0,
            omitted_hit_count=0,
            duplicate_hit_count=0,
            estimated_token_count=0,
        )

    @staticmethod
    def _failure_stop_reason(error_type: ToolErrorType) -> AgentStopReason:
        if error_type == ToolErrorType.TIMEOUT:
            return AgentStopReason.TIMEOUT
        return AgentStopReason.NON_RETRYABLE_TOOL_ERROR

    def _bounded_stop_reason(
        self,
        attempt_number: int,
        routes: list[RetrievalRoute],
    ) -> AgentStopReason:
        if attempt_number >= self._config.max_retry + 1:
            return AgentStopReason.MAX_RETRY_REACHED
        if attempt_number >= len(routes):
            return AgentStopReason.NO_NEW_STRATEGY
        return AgentStopReason.NO_NEW_STRATEGY

    @staticmethod
    def _abstention(
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        warnings: list[str],
    ) -> AnswerResponse:
        return AnswerResponse(
            question=query.original_question,
            answer=_ABSTENTION_TEXT,
            insufficient_evidence=True,
            warnings=list(dict.fromkeys(warnings)),
            retrieval_strategy=strategy,
            trace_id=query.query_id,
        )
