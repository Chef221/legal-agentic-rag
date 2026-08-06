"""Tests for official context adaptation and reproducible ingestion."""

import json
from datetime import UTC, datetime
from pathlib import Path

from legal_agentic_rag.competition import (
    UitDsc2026ContextAdapter,
    UitDsc2026CorpusIngestor,
)
from legal_agentic_rag.configuration import OFFICIAL_CORPUS_DATASET_NAME
from legal_agentic_rag.schemas import (
    ArtifactType,
    CompetitionContext,
)


def _write_contexts(source: Path) -> None:
    source.mkdir()
    records = [
        {
            "id": 740,
            "name": "Quyết định kiểm thử",
            "link": "https://example.invalid/document-740",
            "passage": "  Điều 1. Nội dung pháp luật thứ nhất.\r\n",
        },
        {
            "id": 741,
            "name": "Quyết định kiểm thử",
            "link": "https://example.invalid/document-741",
            "passage": "<p>Điều 2. Nội dung pháp luật thứ hai dài hơn.</p>",
        },
        {
            "id": 742,
            "link": "https://example.invalid/document-742",
            "passage": "",
        },
    ]
    for index, record in enumerate(records, start=1):
        (source / f"context_{index:04d}.json").write_text(
            json.dumps(record, ensure_ascii=False),
            encoding="utf-8",
        )


def test_context_adapter_preserves_only_documented_meaning() -> None:
    """Raw context content maps exactly without invented legal metadata."""
    context = CompetitionContext(
        context_id="740",
        title="  Tiêu đề giữ nguyên  ",
        source_url="https://example.invalid/document-740",
        passage="  Điều 1. Không được sửa nội dung.  ",
    )

    document = UitDsc2026ContextAdapter().to_document(context)

    assert document.document_id == "740"
    assert document.title == "Tiêu đề giữ nguyên"
    assert document.clean_text == "  Điều 1. Không được sửa nội dung.  "
    assert document.source_url == "https://example.invalid/document-740"
    assert document.source_dataset == OFFICIAL_CORPUS_DATASET_NAME
    assert document.document_number is None
    assert document.effect_status is None
    assert document.raw_metadata == {}


def test_context_adapter_keeps_missing_title_and_blank_content_explicit() -> None:
    """Audited optional/blank values do not become invented core metadata."""
    context = CompetitionContext(
        context_id="742",
        title=None,
        source_url="https://example.invalid/document-742",
        passage="",
    )

    document = UitDsc2026ContextAdapter().to_document(context)

    assert document.title is None
    assert document.clean_text is None
    assert document.has_content is False


def test_corpus_ingestion_builds_consistent_audit_and_manifests(
    tmp_path: Path,
) -> None:
    """Every official member has one document on one pinned lineage."""
    source = tmp_path / "contexts"
    _write_contexts(source)
    now = datetime(2026, 8, 1, 8, 30, tzinfo=UTC)

    result = UitDsc2026CorpusIngestor(clock=lambda: now).ingest(source)

    assert [document.document_id for document in result.normalized_documents] == [
        "740",
        "741",
        "742",
    ]
    assert [document.document_id for document in result.cleaned_documents] == [
        "740",
        "741",
        "742",
    ]
    assert result.normalized_documents[0].clean_text == (
        "  Điều 1. Nội dung pháp luật thứ nhất.\r\n"
    )
    assert result.cleaned_documents[0].clean_text == (
        "Điều 1. Nội dung pháp luật thứ nhất."
    )
    assert result.cleaned_documents[1].clean_text == (
        "Điều 2. Nội dung pháp luật thứ hai dài hơn."
    )
    assert result.cleaned_documents[2].clean_text is None
    assert result.cleaned_documents[2].has_content is False
    assert result.dataset_manifest.loaded_at == now
    assert result.dataset_manifest.dataset_name == OFFICIAL_CORPUS_DATASET_NAME
    assert result.dataset_manifest.dataset_revision.startswith("sha256:")
    assert result.dataset_manifest.record_counts == {
        "contexts": 3,
        "contexts_with_content": 2,
    }
    assert result.normalized_manifest.artifact_type == (
        ArtifactType.NORMALIZED_DOCUMENTS
    )
    assert result.normalized_manifest.record_count == 3
    assert result.normalized_manifest.dataset_revision == (
        result.dataset_manifest.dataset_revision
    )
    assert result.cleaned_manifest.artifact_type == ArtifactType.CLEANED_DOCUMENTS
    assert result.cleaned_manifest.metadata["text_modified"] is True
    assert result.cleaned_manifest.metadata["modified_context_count"] == 2
    assert result.cleaned_manifest.metadata["html_markup_context_count"] == 1
    assert result.cleaned_manifest.dataset_revision == (
        result.dataset_manifest.dataset_revision
    )
    assert result.audit.member_count == 3
    assert result.audit.unique_context_count == 3
    assert result.audit.content_context_count == 2
    assert result.audit.blank_passage_count == 1
    assert result.audit.missing_title_count == 1
    assert result.audit.duplicate_title_count == 1
    assert result.audit.duplicate_source_url_count == 0
    assert result.audit.total_passage_characters == sum(
        len(document.clean_text or "")
        for document in result.normalized_documents
    )
    assert result.audit.total_cleaned_characters == sum(
        len(document.clean_text or "") for document in result.cleaned_documents
    )
