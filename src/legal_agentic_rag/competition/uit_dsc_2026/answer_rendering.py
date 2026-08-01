"""Score-facing answer rendering for UIT DSC 2026 Task 2."""

from __future__ import annotations

import re

from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas import AnswerResponse

_EVIDENCE_MARKER = re.compile(r"\[(E[1-9]\d*)\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


def render_competition_answer(response: AnswerResponse) -> str:
    """Remove only verified internal markers from public answer prose."""
    markers = _EVIDENCE_MARKER.findall(response.answer)
    allowed = {citation.evidence_id for citation in response.citations}
    unknown = sorted(set(markers) - allowed)
    if unknown:
        raise DataValidationError(
            "Competition answer contains an unverified evidence marker"
        )
    rendered = _EVIDENCE_MARKER.sub(" ", response.answer)
    rendered = " ".join(rendered.split())
    rendered = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", rendered).strip()
    if not rendered:
        raise DataValidationError(
            "Competition answer is empty after removing evidence markers"
        )
    return rendered
