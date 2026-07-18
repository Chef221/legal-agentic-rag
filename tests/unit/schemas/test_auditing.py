"""Tests for structured offline audit issues."""

import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity


def test_audit_issue_is_json_serializable() -> None:
    """Audit records retain raw values without dataset-specific schema leakage."""
    issue = AuditIssue(
        issue_type="missing_content",
        severity=AuditSeverity.WARNING,
        record_id="doc-1",
        message="Metadata record has no content record",
        raw_value={"id": "doc-1"},
    )
    assert issue.model_dump(mode="json")["severity"] == "warning"


def test_audit_issue_requires_message() -> None:
    """Silently emitted audit issues are not valid."""
    with pytest.raises(ValidationError):
        AuditIssue(issue_type="missing_content", severity="warning", message=" ")
