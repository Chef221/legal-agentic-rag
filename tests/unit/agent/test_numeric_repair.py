"""Deterministic guards for the one-shot numeric-mismatch Agent repair."""

from collections.abc import Sequence

import pytest

from legal_agentic_rag.agent import DeterministicAgentWorkflow
from legal_agentic_rag.configuration import AgentConfig
from legal_agentic_rag.exceptions import ModelError, OperationTimeoutError
from legal_agentic_rag.generation import RuleBasedContextGrader
from legal_agentic_rag.schemas import (
    AgentStopReason,
    AnswerGenerationCorrectionSignal,
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimSupportStatus,
    ClaimVerification,
    Evidence,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.tools import build_fixed_tool_registry


class _Retriever:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.calls += 1
        strategy = query.requested_strategy
        assert strategy is not None
        return RetrievalResponse(
            query=query,
            strategy=strategy,
            hits=[
                RetrievalHit(
                    chunk_id="repair-chunk",
                    document_id="repair-document",
                    rank=1,
                    score=1.0,
                    strategy=strategy,
                    text="Người lao động được nghỉ 12 ngày.",
                    metadata={"structure": {"article_number": "10"}},
                )
            ],
        )


class _Generator:
    def __init__(
        self,
        *,
        repair_outcome: str = "valid",
        multi_claim_initial_response: bool = False,
    ) -> None:
        self.calls: list[AnswerGenerationCorrectionSignal | None] = []
        self._repair_outcome = repair_outcome
        self._multi_claim_initial_response = multi_claim_initial_response

    def generate(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
        correction_signal: AnswerGenerationCorrectionSignal | None = None,
    ) -> AnswerResponse:
        self.calls.append(correction_signal)
        if correction_signal is not None:
            if self._repair_outcome == "model_error":
                raise ModelError("fixture repair failure")
            if self._repair_outcome == "timeout":
                raise OperationTimeoutError("fixture repair timeout")
            if self._repair_outcome == "abstain":
                return AnswerResponse(
                    question=query.original_question,
                    answer="Không đủ căn cứ.",
                    insufficient_evidence=True,
                    retrieval_strategy=retrieval_strategy,
                    trace_id=trace_id,
                )
            if self._repair_outcome == "contract_mismatch":
                return _response(
                    query,
                    evidence[0],
                    retrieval_strategy,
                    trace_id="wrong-trace",
                )
        if correction_signal is None and self._multi_claim_initial_response:
            return _multi_claim_response(query, retrieval_strategy, trace_id)
        return _response(query, evidence[0], retrieval_strategy, trace_id)


class _Verifier:
    def __init__(self, results: list[CitationVerificationResult | Exception]) -> None:
        self._results = list(results)
        self.calls = 0

    def verify(
        self,
        response: AnswerResponse,
        evidence: Sequence[Evidence],
    ) -> CitationVerificationResult:
        self.calls += 1
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="numeric-repair",
        original_question="Người lao động được nghỉ bao nhiêu ngày?",
        normalized_question="người lao động được nghỉ bao nhiêu ngày",
        top_k=1,
        candidate_k=1,
    )


def _response(
    query: RetrievalQuery,
    evidence: Evidence,
    strategy: RetrievalStrategy,
    trace_id: str,
) -> AnswerResponse:
    return AnswerResponse(
        question=query.original_question,
        answer="Người lao động được nghỉ 12 ngày. [E1]",
        citations=[
            Citation(
                evidence_id=evidence.evidence_id,
                chunk_id=evidence.chunk_id,
                document_id=evidence.document_id,
                document_title=evidence.document_title,
                document_number=evidence.document_number,
                article_number=evidence.article_number,
                source_url=evidence.source_url,
            )
        ],
        insufficient_evidence=False,
        retrieval_strategy=strategy,
        trace_id=trace_id,
        metadata={"semantic_synthesis": True},
    )


def _verification(*, errors: list[str], valid: bool) -> CitationVerificationResult:
    claim_errors = [value for value in errors if value != "identity_mismatch"]
    claim = ClaimVerification(
        claim_id="C1",
        claim_text="content omitted from persisted repair metadata",
        evidence_ids=["E1"],
        status=(
            ClaimSupportStatus.SUPPORTED if not claim_errors else ClaimSupportStatus.UNSUPPORTED
        ),
        lexical_support_score=1.0,
        numeric_match="numeric_mismatch" not in claim_errors,
        negation_match="negation_mismatch" not in claim_errors,
        errors=claim_errors,
    )
    top_level = [] if valid else ["unsupported_claim:C1"]
    invalid = []
    if "identity_mismatch" in errors:
        invalid = [Citation(evidence_id="E1", chunk_id="wrong", document_id="wrong")]
    return CitationVerificationResult(
        is_valid=valid,
        invalid_citations=invalid,
        claim_verifications=[claim],
        claim_coverage_score=1.0 if valid else 0.0,
        claim_level_verification_performed=True,
        errors=top_level,
    )


def _multi_claim_response(
    query: RetrievalQuery,
    strategy: RetrievalStrategy,
    trace_id: str,
) -> AnswerResponse:
    return AnswerResponse(
        question=query.original_question,
        answer=(
            "Nguoi lao dong duoc nghi 12 ngay. [E1] "
            "Nguoi lao dong duoc huong nguyen luong. [E2] "
            "Nguoi lao dong duoc nghi 17 ngay. [E3]"
        ),
        citations=[
            Citation(evidence_id="E1", chunk_id="chunk-1", document_id="document-1"),
            Citation(evidence_id="E2", chunk_id="chunk-2", document_id="document-2"),
            Citation(evidence_id="E3", chunk_id="chunk-3", document_id="document-3"),
        ],
        insufficient_evidence=False,
        retrieval_strategy=strategy,
        trace_id=trace_id,
        metadata={"semantic_synthesis": True},
    )


def _partially_supported_numeric_verification() -> CitationVerificationResult:
    return CitationVerificationResult(
        is_valid=False,
        claim_verifications=[
            ClaimVerification(
                claim_id="C1",
                claim_text="Nguoi lao dong duoc nghi 12 ngay.",
                evidence_ids=["E1"],
                status=ClaimSupportStatus.SUPPORTED,
                lexical_support_score=1.0,
                numeric_match=True,
                negation_match=True,
            ),
            ClaimVerification(
                claim_id="C2",
                claim_text="Nguoi lao dong duoc huong nguyen luong.",
                evidence_ids=["E2"],
                status=ClaimSupportStatus.SUPPORTED,
                lexical_support_score=1.0,
                numeric_match=True,
                negation_match=True,
            ),
            ClaimVerification(
                claim_id="C3",
                claim_text="Nguoi lao dong duoc nghi 17 ngay.",
                evidence_ids=["E3"],
                status=ClaimSupportStatus.UNSUPPORTED,
                lexical_support_score=0.5,
                numeric_match=False,
                negation_match=True,
                errors=["numeric_mismatch"],
            ),
        ],
        claim_coverage_score=2 / 3,
        claim_level_verification_performed=True,
        errors=["unsupported_claim:C3"],
    )


def _workflow(
    generator: _Generator,
    verifier: _Verifier,
) -> tuple[DeterministicAgentWorkflow, _Retriever]:
    retriever = _Retriever()
    registry = build_fixed_tool_registry(
        retriever=retriever,
        context_grader=RuleBasedContextGrader(),
        answer_generator=generator,
        citation_verifier=verifier,
    )
    return (
        DeterministicAgentWorkflow(
            registry,
            agent_config=AgentConfig(
                max_retry=2,
                max_numeric_mismatch_repairs=1,
                strategy_order=[RetrievalStrategy.HYBRID],
                rewrite_query_on_retry=False,
            ),
        ),
        retriever,
    )


def test_numeric_only_failure_repairs_once_with_same_evidence() -> None:
    """One numeric-only rejection regenerates once without a second retrieval."""
    generator = _Generator()
    verifier = _Verifier(
        [
            _verification(errors=["numeric_mismatch"], valid=False),
            _verification(errors=[], valid=True),
        ]
    )
    workflow, retriever = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert retriever.calls == 1
    assert generator.calls == [None, AnswerGenerationCorrectionSignal.NUMERIC_MISMATCH]
    assert verifier.calls == 2
    assert result.state.retry_count == 0
    assert result.response.metadata["numeric_repair"]["outcome"] == "model_regeneration_succeeded"
    assert CitationVerificationResult.model_validate(
        result.response.metadata["citation_verification"]
    ).is_valid is True
    assert "claim_text" not in str(result.response.metadata["numeric_repair"])
    assert "numeric_repair_succeeded" in result.response.warnings
    assert [item["invocation_id"] for item in result.state.metadata["tool_invocations"]][-2:] == [
        "numeric-repair:1:numeric_repair_generation",
        "numeric-repair:1:numeric_repair_verification",
    ]


def test_numeric_salvage_keeps_only_supported_claims_without_model_retry() -> None:
    """Supported claims are retained verbatim before a model regeneration is considered."""
    generator = _Generator(multi_claim_initial_response=True)
    verifier = _Verifier(
        [
            _partially_supported_numeric_verification(),
            _verification(errors=[], valid=True),
        ]
    )
    workflow, retriever = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert retriever.calls == 1
    assert generator.calls == [None]
    assert verifier.calls == 2
    assert result.response.answer == (
        "Nguoi lao dong duoc nghi 12 ngay [E1]. "
        "Nguoi lao dong duoc huong nguyen luong [E2]."
    )
    assert [citation.evidence_id for citation in result.response.citations] == ["E1", "E2"]
    repair = result.response.metadata["numeric_repair"]
    assert repair["outcome"] == "salvage_succeeded"
    assert repair["salvage"]["dropped_claim_count"] == 1
    assert repair["model_regeneration"] == {
        "attempted": False,
        "outcome": "not_needed",
    }
    assert "numeric_claim_salvage_succeeded" in result.response.warnings
    assert "numeric_claims_dropped:1" in result.response.warnings
    assert "claim_text" not in str(repair)
    assert [item["invocation_id"] for item in result.state.metadata["tool_invocations"]][-1] == (
        "numeric-repair:1:numeric_salvage_verification"
    )


def test_numeric_salvage_verifier_timeout_never_calls_model_fallback() -> None:
    """A salvage verification failure is fail-closed rather than model-assisted."""
    generator = _Generator(multi_claim_initial_response=True)
    verifier = _Verifier(
        [
            _partially_supported_numeric_verification(),
            OperationTimeoutError("fixture salvage verifier timeout"),
        ]
    )
    workflow, retriever = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == AgentStopReason.TIMEOUT
    assert result.response.insufficient_evidence is True
    assert retriever.calls == 1
    assert generator.calls == [None]
    assert verifier.calls == 2
    repair = result.response.metadata["numeric_repair"]
    assert repair["outcome"] == "verification_timeout"
    assert repair["model_regeneration"]["attempted"] is False


def test_normal_verified_response_keeps_the_m49_1_verification_contract() -> None:
    """A non-repair success preserves the full CitationVerificationResult payload."""
    generator = _Generator()
    verifier = _Verifier([_verification(errors=[], valid=True)])
    workflow, _ = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert "numeric_repair" not in result.response.metadata
    assert CitationVerificationResult.model_validate(
        result.response.metadata["citation_verification"]
    ).is_valid is True


@pytest.mark.parametrize(
    "error",
    ["negation_mismatch", "insufficient_lexical_support", "identity_mismatch"],
)
def test_non_numeric_or_identity_failure_never_repairs(error: str) -> None:
    """Mixed hard failures follow the M49.1 fail-closed path immediately."""
    generator = _Generator()
    verifier = _Verifier([_verification(errors=["numeric_mismatch", error], valid=False)])
    workflow, retriever = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == AgentStopReason.CITATION_VERIFICATION_FAILED
    assert result.response.insufficient_evidence is True
    assert retriever.calls == 1
    assert generator.calls == [None]
    assert verifier.calls == 1
    assert "numeric_repair" not in result.response.metadata


def test_unrelated_top_level_failure_never_repairs() -> None:
    """A matching numeric claim does not override an independent hard failure."""
    generator = _Generator()
    initial = _verification(errors=["numeric_mismatch"], valid=False).model_copy(
        update={"errors": ["unsupported_claim:C1", "hard_failure"]}
    )
    verifier = _Verifier([initial])
    workflow, retriever = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == AgentStopReason.CITATION_VERIFICATION_FAILED
    assert retriever.calls == 1
    assert generator.calls == [None]
    assert verifier.calls == 1


def test_second_numeric_failure_stops_after_one_repair() -> None:
    """A rejected repair cannot loop into a third generation invocation."""
    generator = _Generator()
    verifier = _Verifier(
        [
            _verification(errors=["numeric_mismatch"], valid=False),
            _verification(errors=["numeric_mismatch"], valid=False),
        ]
    )
    workflow, retriever = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == AgentStopReason.CITATION_VERIFICATION_FAILED
    assert result.response.insufficient_evidence is True
    assert retriever.calls == 1
    assert len(generator.calls) == 2
    assert verifier.calls == 2
    assert result.response.metadata["numeric_repair"]["outcome"] == "verification_rejected"


@pytest.mark.parametrize(
    ("outcome", "stop_reason"),
    [
        ("model_error", AgentStopReason.GENERATION_FAILED),
        ("timeout", AgentStopReason.TIMEOUT),
        ("abstain", AgentStopReason.GENERATION_FAILED),
        ("contract_mismatch", AgentStopReason.GENERATION_FAILED),
    ],
)
def test_repair_generation_failures_remain_fail_closed(
    outcome: str,
    stop_reason: AgentStopReason,
) -> None:
    """Repair errors, abstention, and contract drift do not expose the draft."""
    generator = _Generator(repair_outcome=outcome)
    verifier = _Verifier([_verification(errors=["numeric_mismatch"], valid=False)])
    workflow, retriever = _workflow(generator, verifier)

    result = workflow.run(_query())

    assert result.stop_reason == stop_reason
    assert result.response.insufficient_evidence is True
    assert retriever.calls == 1
    assert len(generator.calls) == 2
    assert verifier.calls == 1
    assert result.response.metadata["numeric_repair"]["outcome"] in {
        "generation_failed", "timeout", "generator_abstained", "contract_mismatch"
    }
