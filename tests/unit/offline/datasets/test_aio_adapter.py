"""Tests for isolation of AIO raw field access."""

from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.offline.datasets.aio import AioRecordAdapter


def test_adapter_reads_each_logical_identity_without_mutation() -> None:
    """Audit accessors read raw identifiers but leave input untouched."""
    metadata = {"id": 42, "title": "Văn bản"}
    relationship = {
        "doc_id": " doc-1 ",
        "other_doc_id": "doc-2",
        "relationship": " Sửa đổi ",
    }
    adapter = AioRecordAdapter()

    assert adapter.identifier(DatasetComponent.METADATA, metadata) == "42"
    assert adapter.relationship(relationship) == (
        "doc-1",
        "doc-2",
        "Sửa đổi",
    )
    assert metadata == {"id": 42, "title": "Văn bản"}


def test_adapter_declares_only_boundary_required_fields() -> None:
    """The adapter requires raw identity/payload fields, not future core fields."""
    adapter = AioRecordAdapter()
    assert adapter.required_fields(DatasetComponent.CONTENT) == {
        "id",
        "content_html",
    }
    assert adapter.as_identifier({"nested": "id"}) is None
