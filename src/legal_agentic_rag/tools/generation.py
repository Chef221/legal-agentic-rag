"""Typed wrappers over context grading, generation, and citation verification."""

from __future__ import annotations

from legal_agentic_rag.contracts.answer_generator import AnswerGenerator
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.contracts.context_grader import ContextGrader
from legal_agentic_rag.exceptions import ConfigurationError
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    CitationVerificationResult,
    ContextGrade,
)
from legal_agentic_rag.schemas.tools import (
    AnswerGenerationInput,
    CitationVerificationInput,
    ContextGradingInput,
    ToolName,
)


class ContextGradingTool:
    """Expose structural or model-based grading through its existing contract."""

    def __init__(
        self,
        grader: ContextGrader,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ConfigurationError("Tool timeout must be positive")
        self._grader = grader
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> ToolName:
        """Return the fixed context-grading tool name."""
        return ToolName.CONTEXT_GRADING

    @property
    def description(self) -> str:
        """Describe context sufficiency grading without overstating semantics."""
        return (
            "Grade selected legal evidence using the configured ContextGrader."
        )

    @property
    def input_model(self) -> type[ContextGradingInput]:
        """Return context grading input schema."""
        return ContextGradingInput

    @property
    def output_model(self) -> type[ContextGrade]:
        """Return context grade output schema."""
        return ContextGrade

    @property
    def timeout_seconds(self) -> float:
        """Return the grading invocation budget."""
        return self._timeout_seconds

    def invoke(self, payload: ContextGradingInput) -> ContextGrade:
        """Grade only the query and evidence supplied in the typed payload."""
        return self._grader.grade(payload.query, payload.evidence)


class AnswerGenerationTool:
    """Expose grounded generation without granting retrieval or corpus access."""

    def __init__(
        self,
        generator: AnswerGenerator,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ConfigurationError("Tool timeout must be positive")
        self._generator = generator
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> ToolName:
        """Return the fixed answer-generation tool name."""
        return ToolName.ANSWER_GENERATION

    @property
    def description(self) -> str:
        """Describe grounded generation over supplied evidence only."""
        return (
            "Generate a Vietnamese legal answer using only supplied evidence."
        )

    @property
    def input_model(self) -> type[AnswerGenerationInput]:
        """Return grounded generation input schema."""
        return AnswerGenerationInput

    @property
    def output_model(self) -> type[AnswerResponse]:
        """Return answer response output schema."""
        return AnswerResponse

    @property
    def timeout_seconds(self) -> float:
        """Return the generation invocation budget."""
        return self._timeout_seconds

    def invoke(self, payload: AnswerGenerationInput) -> AnswerResponse:
        """Generate from exactly the supplied typed query and evidence."""
        return self._generator.generate(
            payload.query,
            payload.evidence,
            payload.retrieval_strategy,
            payload.trace_id,
        )


class CitationVerificationTool:
    """Expose citation checks without retrieval or generation side effects."""

    def __init__(
        self,
        verifier: CitationVerifier,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ConfigurationError("Tool timeout must be positive")
        self._verifier = verifier
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> ToolName:
        """Return the fixed citation-verification tool name."""
        return ToolName.CITATION_VERIFICATION

    @property
    def description(self) -> str:
        """Describe exact referential citation checking."""
        return (
            "Verify answer citations against the exact supplied legal evidence."
        )

    @property
    def input_model(self) -> type[CitationVerificationInput]:
        """Return citation verification input schema."""
        return CitationVerificationInput

    @property
    def output_model(self) -> type[CitationVerificationResult]:
        """Return citation verification output schema."""
        return CitationVerificationResult

    @property
    def timeout_seconds(self) -> float:
        """Return the verification invocation budget."""
        return self._timeout_seconds

    def invoke(
        self,
        payload: CitationVerificationInput,
    ) -> CitationVerificationResult:
        """Verify only the supplied response and evidence."""
        return self._verifier.verify(payload.response, payload.evidence)
