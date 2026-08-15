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
from legal_agentic_rag.generation.claim_salvage import build_supported_claim_salvage
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
    AnswerGenerationCorrectionSignal,
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
            self._generation_payload(query, strategy, context),
            invocation_trace,
        )
        if not generation.success:
            reason = (
                AgentStopReason.TIMEOUT
                if generation.error.error_type == ToolErrorType.TIMEOUT
                else AgentStopReason.GENERATION_FAILED
            )
            abstention = self._abstention(
                query,
                strategy,
                [f"generator:{generation.error.error_type.value}"],
            )
            return (
                abstention.model_copy(
                    update={
                        "metadata": {
                            **abstention.metadata,
                            **self._generation_failure_metadata(generation),
                        }
                    }
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
        if response.insufficient_evidence:
            return (
                response.model_copy(
                    update={
                        "warnings": list(
                            dict.fromkeys(
                                [
                                    *response.warnings,
                                    "generator:insufficient_evidence",
                                ]
                            )
                        )
                    }
                ),
                AgentStopReason.GENERATION_FAILED,
            )
        verification = self._verify_response(
            query=query,
            context=context,
            response=response,
            attempt_number=attempt_number,
            phase="verification",
            invocation_trace=invocation_trace,
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
            if self._can_salvage_supported_claims(result):
                if self._can_repair_numeric_mismatch(result):
                    return self._repair_numeric_mismatch(
                        query=query,
                        strategy=strategy,
                        context=context,
                        attempt_number=attempt_number,
                        initial_response=response,
                        initial_verification=result,
                        invocation_trace=invocation_trace,
                    )
                return self._salvage_supported_claims(
                    query=query,
                    strategy=strategy,
                    context=context,
                    attempt_number=attempt_number,
                    initial_response=response,
                    initial_verification=result,
                    invocation_trace=invocation_trace,
                )
            if self._can_repair_numeric_mismatch(result):
                return self._repair_numeric_mismatch(
                    query=query,
                    strategy=strategy,
                    context=context,
                    attempt_number=attempt_number,
                    initial_response=response,
                    initial_verification=result,
                    invocation_trace=invocation_trace,
                )
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
                            "citation_verification": result.model_dump(mode="json"),
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

    @staticmethod
    def _generation_payload(
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        context: ContextBuildResult,
        correction_signal: AnswerGenerationCorrectionSignal | None = None,
    ) -> dict[str, object]:
        """Create the closed generation payload without any rejected draft."""
        payload: dict[str, object] = {
            "query": query.model_dump(mode="json"),
            "evidence": [
                item.model_dump(mode="json") for item in context.evidence
            ],
            "retrieval_strategy": strategy.value,
            "trace_id": query.query_id,
        }
        if correction_signal is not None:
            payload["correction_signal"] = correction_signal.value
        return payload

    def _verify_response(
        self,
        *,
        query: RetrievalQuery,
        context: ContextBuildResult,
        response: AnswerResponse,
        attempt_number: int,
        phase: str,
        invocation_trace: list[dict[str, object]],
    ) -> ToolInvocationResult:
        """Verify one response against exactly the already selected evidence."""
        return self._invoke(
            query.query_id,
            attempt_number,
            phase,
            ToolName.CITATION_VERIFICATION,
            {
                "response": response.model_dump(mode="json"),
                "evidence": [
                    item.model_dump(mode="json") for item in context.evidence
                ],
            },
            invocation_trace,
        )

    def _can_repair_numeric_mismatch(
        self,
        result: CitationVerificationResult,
    ) -> bool:
        """Allow exactly the narrow numeric-only repair path configured for Agent."""
        if self._config.max_numeric_mismatch_repairs < 1:
            return False
        if result.is_valid or result.invalid_citations:
            return False
        unsupported = [
            claim
            for claim in result.claim_verifications
            if claim.status.value == "unsupported"
        ]
        if not unsupported:
            return False
        if any(set(claim.errors) != {"numeric_mismatch"} for claim in unsupported):
            return False
        expected_errors = {f"unsupported_claim:{claim.claim_id}" for claim in unsupported}
        return set(result.errors) == expected_errors

    def _can_salvage_supported_claims(
        self,
        result: CitationVerificationResult,
    ) -> bool:
        """Allow one safe, deterministic removal of verifier-rejected claims."""
        if self._config.max_supported_claim_salvages < 1:
            return False
        if result.is_valid or result.invalid_citations:
            return False
        supported = [
            claim
            for claim in result.claim_verifications
            if claim.status.value == "supported"
        ]
        unsupported = [
            claim
            for claim in result.claim_verifications
            if claim.status.value == "unsupported"
        ]
        if not supported or not unsupported:
            return False
        allowed_errors = {"numeric_mismatch", "negation_mismatch"}
        if any(
            not claim.errors or not set(claim.errors).issubset(allowed_errors)
            for claim in unsupported
        ):
            return False
        expected_errors = {f"unsupported_claim:{claim.claim_id}" for claim in unsupported}
        return set(result.errors) == expected_errors

    def _salvage_supported_claims(
        self,
        *,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        context: ContextBuildResult,
        attempt_number: int,
        initial_response: AnswerResponse,
        initial_verification: CitationVerificationResult,
        invocation_trace: list[dict[str, object]],
    ) -> tuple[AnswerResponse, AgentStopReason]:
        """Keep only supported claims for a bounded non-numeric repair path.

        This branch deliberately never sends the rejected draft back to the model.
        A verifier failure, timeout, or rejection remains an abstention.
        """
        metadata: dict[str, object] = {
            "attempted": True,
            "count": 1,
            "initial_verification": self._verification_metadata(initial_verification),
        }
        salvage = build_supported_claim_salvage(initial_response, initial_verification)
        metadata.update(
            {
                "outcome": salvage.outcome,
                "retained_claim_count": salvage.retained_claim_count,
                "dropped_claim_count": salvage.dropped_claim_count,
                "dropped_error_counts": salvage.dropped_error_counts,
            }
        )
        if salvage.response is None:
            return (
                self._supported_claim_salvage_abstention(
                    query,
                    strategy,
                    metadata,
                    outcome=salvage.outcome,
                    warnings=["supported_claim_salvage_failed"],
                ),
                AgentStopReason.CITATION_VERIFICATION_FAILED,
            )
        verification = self._verify_response(
            query=query,
            context=context,
            response=salvage.response,
            attempt_number=attempt_number,
            phase="supported_claim_salvage_verification",
            invocation_trace=invocation_trace,
        )
        if not verification.success:
            outcome = (
                "verification_timeout"
                if verification.error.error_type == ToolErrorType.TIMEOUT
                else "verification_failed"
            )
            return (
                self._supported_claim_salvage_abstention(
                    query,
                    strategy,
                    metadata,
                    outcome=outcome,
                    warnings=[
                        "supported_claim_salvage_failed",
                        f"verifier:{verification.error.error_type.value}",
                    ],
                ),
                (
                    AgentStopReason.TIMEOUT
                    if verification.error.error_type == ToolErrorType.TIMEOUT
                    else AgentStopReason.CITATION_VERIFICATION_FAILED
                ),
            )
        salvage_verification = CitationVerificationResult.model_validate(
            verification.output
        )
        metadata["verification"] = self._verification_metadata(salvage_verification)
        if not salvage_verification.is_valid:
            return (
                self._supported_claim_salvage_abstention(
                    query,
                    strategy,
                    metadata,
                    outcome="verification_rejected",
                    warnings=[
                        "citation_verification_failed",
                        *salvage_verification.errors,
                    ],
                    verification=salvage_verification,
                ),
                AgentStopReason.CITATION_VERIFICATION_FAILED,
            )
        metadata["outcome"] = "succeeded"
        warnings = [*salvage.response.warnings, "supported_claim_salvage_succeeded"]
        if salvage.dropped_claim_count:
            warnings.append(f"supported_claims_dropped:{salvage.dropped_claim_count}")
        return (
            salvage.response.model_copy(
                update={
                    "warnings": list(dict.fromkeys(warnings)),
                    "metadata": {
                        **salvage.response.metadata,
                        "citation_verification": salvage_verification.model_dump(
                            mode="json"
                        ),
                        "claim_salvage": metadata,
                    },
                }
            ),
            AgentStopReason.ANSWER_VERIFIED,
        )

    def _supported_claim_salvage_abstention(
        self,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        metadata: dict[str, object],
        *,
        outcome: str,
        warnings: list[str],
        verification: CitationVerificationResult | None = None,
    ) -> AnswerResponse:
        """Return a fail-closed, content-free supported-claim salvage outcome."""
        metadata["outcome"] = outcome
        response = self._abstention(
            query,
            strategy,
            ["supported_claim_salvage_failed", *warnings],
        )
        response_metadata: dict[str, object] = {
            **response.metadata,
            "claim_salvage": metadata,
        }
        if verification is not None:
            response_metadata["citation_verification"] = verification.model_dump(
                mode="json"
            )
        return response.model_copy(update={"metadata": response_metadata})

    def _repair_numeric_mismatch(
        self,
        *,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        context: ContextBuildResult,
        attempt_number: int,
        initial_response: AnswerResponse,
        initial_verification: CitationVerificationResult,
        invocation_trace: list[dict[str, object]],
    ) -> tuple[AnswerResponse, AgentStopReason]:
        """Salvage supported claims, then regenerate once only when required."""
        repair_metadata: dict[str, object] = {
            "attempted": True,
            "count": 1,
            "initial_verification": self._verification_metadata(initial_verification),
            "salvage": {
                "attempted": False,
                "outcome": "not_started",
            },
            "model_regeneration": {
                "attempted": False,
                "outcome": "not_needed",
            },
        }
        salvage = build_supported_claim_salvage(initial_response, initial_verification)
        salvage_metadata = repair_metadata["salvage"]
        assert isinstance(salvage_metadata, dict)
        salvage_metadata.update(
            {
                "attempted": salvage.outcome != "not_applicable_no_supported_claim",
                "outcome": salvage.outcome,
                "retained_claim_count": salvage.retained_claim_count,
                "dropped_claim_count": salvage.dropped_claim_count,
                "dropped_error_counts": salvage.dropped_error_counts,
            }
        )
        if salvage.outcome == "contract_mismatch":
            return (
                self._numeric_repair_abstention(
                    query,
                    strategy,
                    repair_metadata,
                    outcome="contract_mismatch",
                    warnings=["numeric_claim_salvage:contract_mismatch"],
                ),
                AgentStopReason.CITATION_VERIFICATION_FAILED,
            )
        if salvage.response is not None:
            verification = self._verify_response(
                query=query,
                context=context,
                response=salvage.response,
                attempt_number=attempt_number,
                phase="numeric_salvage_verification",
                invocation_trace=invocation_trace,
            )
            if not verification.success:
                return (
                    self._numeric_repair_abstention(
                        query,
                        strategy,
                        repair_metadata,
                        outcome=(
                            "verification_timeout"
                            if verification.error.error_type == ToolErrorType.TIMEOUT
                            else "verification_failed"
                        ),
                        warnings=[
                            "numeric_claim_salvage_failed",
                            f"verifier:{verification.error.error_type.value}",
                        ],
                    ),
                    (
                        AgentStopReason.TIMEOUT
                        if verification.error.error_type == ToolErrorType.TIMEOUT
                        else AgentStopReason.CITATION_VERIFICATION_FAILED
                    ),
                )
            salvage_verification = CitationVerificationResult.model_validate(
                verification.output
            )
            salvage_metadata["verification"] = self._verification_metadata(
                salvage_verification
            )
            if salvage_verification.is_valid:
                salvage_metadata["outcome"] = "succeeded"
                repair_metadata["outcome"] = "salvage_succeeded"
                warnings = [
                    *salvage.response.warnings,
                    "numeric_repair_succeeded",
                    "numeric_claim_salvage_succeeded",
                ]
                if salvage.dropped_claim_count:
                    warnings.append(
                        f"numeric_claims_dropped:{salvage.dropped_claim_count}"
                    )
                return (
                    salvage.response.model_copy(
                        update={
                            "warnings": list(dict.fromkeys(warnings)),
                            "metadata": {
                                **salvage.response.metadata,
                                "citation_verification": (
                                    salvage_verification.model_dump(mode="json")
                                ),
                                "numeric_repair": repair_metadata,
                            },
                        }
                    ),
                    AgentStopReason.ANSWER_VERIFIED,
                )
            salvage_metadata["outcome"] = "verification_rejected"

        regeneration_metadata = repair_metadata["model_regeneration"]
        assert isinstance(regeneration_metadata, dict)
        regeneration_metadata.update({"attempted": True, "outcome": "started"})
        generation = self._invoke(
            query.query_id,
            attempt_number,
            "numeric_repair_generation",
            ToolName.ANSWER_GENERATION,
            self._generation_payload(
                query,
                strategy,
                context,
                AnswerGenerationCorrectionSignal.NUMERIC_MISMATCH,
            ),
            invocation_trace,
        )
        if not generation.success:
            regeneration_metadata["outcome"] = (
                "timeout"
                if generation.error.error_type == ToolErrorType.TIMEOUT
                else "generation_failed"
            )
            return (
                self._numeric_repair_abstention(
                    query,
                    strategy,
                    repair_metadata,
                    outcome=(
                        "timeout"
                        if generation.error.error_type == ToolErrorType.TIMEOUT
                        else "generation_failed"
                    ),
                    warnings=[f"generator:{generation.error.error_type.value}"],
                ),
                (
                    AgentStopReason.TIMEOUT
                    if generation.error.error_type == ToolErrorType.TIMEOUT
                    else AgentStopReason.GENERATION_FAILED
                ),
            )
        response = AnswerResponse.model_validate(generation.output)
        if (
            response.question != query.original_question
            or response.trace_id != query.query_id
            or response.retrieval_strategy != strategy
        ):
            regeneration_metadata["outcome"] = "contract_mismatch"
            return (
                self._numeric_repair_abstention(
                    query,
                    strategy,
                    repair_metadata,
                    outcome="contract_mismatch",
                    warnings=["generator:contract_mismatch"],
                ),
                AgentStopReason.GENERATION_FAILED,
            )
        if response.insufficient_evidence:
            regeneration_metadata["outcome"] = "generator_abstained"
            return (
                self._numeric_repair_abstention(
                    query,
                    strategy,
                    repair_metadata,
                    outcome="generator_abstained",
                    warnings=["generator:insufficient_evidence"],
                ),
                AgentStopReason.GENERATION_FAILED,
            )
        verification = self._verify_response(
            query=query,
            context=context,
            response=response,
            attempt_number=attempt_number,
            phase="numeric_repair_verification",
            invocation_trace=invocation_trace,
        )
        if not verification.success:
            regeneration_metadata["outcome"] = (
                "verification_timeout"
                if verification.error.error_type == ToolErrorType.TIMEOUT
                else "verification_failed"
            )
            return (
                self._numeric_repair_abstention(
                    query,
                    strategy,
                    repair_metadata,
                    outcome=(
                        "verification_timeout"
                        if verification.error.error_type == ToolErrorType.TIMEOUT
                        else "verification_failed"
                    ),
                    warnings=[f"verifier:{verification.error.error_type.value}"],
                ),
                (
                    AgentStopReason.TIMEOUT
                    if verification.error.error_type == ToolErrorType.TIMEOUT
                    else AgentStopReason.CITATION_VERIFICATION_FAILED
                ),
            )
        final_verification = CitationVerificationResult.model_validate(
            verification.output
        )
        repair_metadata["final_verification"] = self._verification_metadata(
            final_verification
        )
        if not final_verification.is_valid:
            regeneration_metadata["outcome"] = "verification_rejected"
            return (
                self._numeric_repair_abstention(
                    query,
                    strategy,
                    repair_metadata,
                    outcome="verification_rejected",
                    warnings=[
                        "citation_verification_failed",
                        *final_verification.errors,
                    ],
                    verification=final_verification,
                ),
                AgentStopReason.CITATION_VERIFICATION_FAILED,
            )
        regeneration_metadata["outcome"] = "succeeded"
        repair_metadata["outcome"] = "model_regeneration_succeeded"
        return (
            response.model_copy(
                update={
                    "warnings": list(
                        dict.fromkeys(
                            [*response.warnings, "numeric_repair_succeeded"]
                        )
                    ),
                    "metadata": {
                        **response.metadata,
                        "citation_verification": final_verification.model_dump(
                            mode="json"
                        ),
                        "numeric_repair": repair_metadata,
                    },
                }
            ),
            AgentStopReason.ANSWER_VERIFIED,
        )

    def _numeric_repair_abstention(
        self,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        repair_metadata: dict[str, object],
        *,
        outcome: str,
        warnings: list[str],
        verification: CitationVerificationResult | None = None,
    ) -> AnswerResponse:
        """Package a content-free repair diagnostic with the fail-closed response."""
        repair_metadata["outcome"] = outcome
        response = self._abstention(query, strategy, ["numeric_repair_failed", *warnings])
        metadata = {
            **response.metadata,
            "numeric_repair": repair_metadata,
        }
        if verification is not None:
            metadata["citation_verification"] = verification.model_dump(
                mode="json"
            )
        return response.model_copy(update={"metadata": metadata})

    @staticmethod
    def _verification_metadata(
        result: CitationVerificationResult,
    ) -> dict[str, object]:
        """Keep verification diagnostics without draft or legal-content text."""
        return {
            "is_valid": result.is_valid,
            "valid_citation_count": len(result.valid_citations),
            "invalid_citation_count": len(result.invalid_citations),
            "claim_level_verification_performed": (
                result.claim_level_verification_performed
            ),
            "claim_coverage_score": result.claim_coverage_score,
            "claim_verifications": [
                {
                    "claim_id": claim.claim_id,
                    "evidence_ids": claim.evidence_ids,
                    "status": claim.status.value,
                    "lexical_support_score": claim.lexical_support_score,
                    "numeric_match": claim.numeric_match,
                    "negation_match": claim.negation_match,
                    "errors": claim.errors,
                }
                for claim in result.claim_verifications
            ],
            "semantic_verification": (
                result.semantic_verification.model_dump(mode="json")
                if result.semantic_verification is not None
                else None
            ),
            "errors": result.errors,
            "warnings": result.warnings,
        }

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
        response_metadata["context"] = {
            "input_hit_count": context.input_hit_count,
            "selected_count": context.selected_count,
            "omitted_hit_count": context.omitted_hit_count,
            "duplicate_hit_count": context.duplicate_hit_count,
            "estimated_token_count": context.estimated_token_count,
            "truncated": context.truncated,
            "warnings": context.warnings,
            "selection_trace": [
                item.model_dump(mode="json")
                for item in context.selection_trace
            ],
        }
        response_metadata["selected_evidence"] = [
            {
                "evidence_id": item.evidence_id,
                "chunk_id": item.chunk_id,
            }
            for item in context.evidence
        ]
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
                "generation_failure_code": (
                    result.error.generation_failure_code.value
                    if result.error is not None
                    and result.error.generation_failure_code is not None
                    else None
                ),
                "latency_ms": result.latency_ms,
            }
        )
        return result

    @staticmethod
    def _generation_failure_metadata(
        generation: ToolInvocationResult,
    ) -> dict[str, object]:
        """Persist only a closed structured-output failure code, never a draft."""
        error = generation.error
        if (
            error is None
            or error.error_type is not ToolErrorType.MODEL_ERROR
            or error.generation_failure_code is None
        ):
            return {}
        return {
            "generation_failure": {
                "error_type": error.error_type.value,
                "failure_code": error.generation_failure_code.value,
            }
        }

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
