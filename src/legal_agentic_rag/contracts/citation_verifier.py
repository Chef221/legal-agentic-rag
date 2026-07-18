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
    """Verify answer citations against the exact supplied evidence set."""

    def verify(
        self, response: AnswerResponse, evidence: Sequence[Evidence]
    ) -> CitationVerificationResult:
        """Return structural and referential citation verification results."""
        ...
