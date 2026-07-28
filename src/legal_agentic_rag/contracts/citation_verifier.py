"""Protocol for rule-based citation verification."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    CitationVerificationResult,
    Evidence,
)


@runtime_checkable
class CitationVerifier(Protocol):
    """Verify citation identity and answer grounding against supplied evidence."""

    def verify(
        self, response: AnswerResponse, evidence: Sequence[Evidence]
    ) -> CitationVerificationResult:
        """Return citation and claim-grounding verification results."""
        ...
