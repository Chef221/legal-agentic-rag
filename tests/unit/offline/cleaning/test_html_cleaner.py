"""Unit tests for conservative, deterministic legal HTML cleaning."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from legal_agentic_rag.configuration import HtmlCleaningConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.offline.cleaning import LegalHtmlCleaner
from legal_agentic_rag.schemas import ArtifactManifest, ArtifactType, LegalDocument


def _document(
    document_id: str = "doc-1", content_html: str | None = "<p>Điều 1.</p>"
) -> LegalDocument:
    return LegalDocument(
        document_id=document_id,
        title="Luật mẫu",
        content_html=content_html,
        clean_text=None,
        has_content=content_html is not None,
        source_dataset="aio",
    )


def _manifest(
    record_count: int = 1,
    artifact_type: ArtifactType = ArtifactType.NORMALIZED_DOCUMENTS,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=artifact_type,
        artifact_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=record_count,
        processing_config_hash="normalization-hash",
        code_version="0.3.0",
    )


def test_cleaner_removes_web_noise_and_preserves_legal_text(
    load_html_fixture,
) -> None:
    """Noise disappears while legal markers, values, and tables remain visible."""
    clean_text = LegalHtmlCleaner().clean_html(
        load_html_fixture("legal_document_with_noise.html")
    )

    assert "window.secretTrackingValue" not in clean_text
    assert ".tracking" not in clean_text
    assert "Trang chủ" not in clean_text
    assert "Nội dung vô hình" not in clean_text
    assert "Quảng cáo ẩn" not in clean_text
    assert "Mã theo dõi 12345" not in clean_text
    assert "Luật mẫu số 01/2026/QH" in clean_text
    assert "Chương I" in clean_text
    assert "Điều 1. Phạm vi áp dụng" in clean_text
    assert "1. Không áp dụng" in clean_text
    assert "trừ trường hợp" in clean_text
    assert "10.000.000 đồng" in clean_text
    assert "01/01/2026" in clean_text
    assert "Hành vi | Mức phạt" in clean_text
    assert "Vi phạm A | 5.000.000 đồng" in clean_text
    assert "CHỦ TỊCH" in clean_text
    assert "Nguyễn Văn A" in clean_text


def test_cleaner_is_deterministic_and_normalizes_unicode_and_entities() -> None:
    """Equivalent calls produce NFC text without losing Vietnamese diacritics."""
    html = "<p>Điều 2.&nbsp;Không xóa số 123 &amp; dấu câu.</p>"
    cleaner = LegalHtmlCleaner()

    first = cleaner.clean_html(html)
    second = cleaner.clean_html(html)

    assert first == second
    assert first == "Điều 2. Không xóa số 123 & dấu câu."


def test_cleaner_tolerates_malformed_html_and_removes_control_characters() -> None:
    """A partial HTML tree remains cleanable without inventing structure."""
    html = "<div><p>Điều 3.\x00 Không áp dụng.<p>Khoản 1 chưa hết hiệu lực"

    clean_text = LegalHtmlCleaner().clean_html(html)

    assert clean_text == "Điều 3. Không áp dụng.\nKhoản 1 chưa hết hiệu lực"


def test_batch_cleaning_preserves_inputs_and_builds_manifest() -> None:
    """A cleaning run returns typed documents, issues, counts, and provenance."""
    documents = [
        _document(),
        _document("doc-2", None),
        _document("doc-3", "<script>x</script>"),
    ]
    before = deepcopy(documents)
    cleaner = LegalHtmlCleaner(
        clock=lambda: datetime(2026, 7, 18, 2, tzinfo=UTC)
    )

    result = cleaner.clean(documents=documents, source_manifest=_manifest(3))

    assert documents == before
    assert [document.document_id for document in result.documents] == [
        "doc-1",
        "doc-2",
        "doc-3",
    ]
    assert result.documents[0].clean_text == "Điều 1."
    assert result.documents[0].content_html == "<p>Điều 1.</p>"
    assert result.cleaned_document_count == 1
    assert result.missing_content_count == 1
    assert result.empty_output_count == 1
    assert {issue.issue_type for issue in result.issues} == {
        "empty_clean_text",
        "missing_content",
    }
    assert all(
        issue.metadata["stage"] == "html_cleaning" for issue in result.issues
    )
    assert result.manifest.artifact_type == ArtifactType.CLEANED_DOCUMENTS
    assert result.manifest.record_count == 3
    assert (
        result.manifest.metadata["source_processing_config_hash"]
        == "normalization-hash"
    )
    assert len(result.manifest.processing_config_hash) == 64


def test_cleaner_rejects_incompatible_or_inconsistent_input() -> None:
    """Cleaning cannot silently accept the wrong upstream artifact boundary."""
    cleaner = LegalHtmlCleaner()
    with pytest.raises(ArtifactCompatibilityError):
        cleaner.clean(
            documents=[_document()],
            source_manifest=_manifest(artifact_type=ArtifactType.CLEANED_DOCUMENTS),
        )
    with pytest.raises(DataValidationError, match="record count"):
        cleaner.clean(documents=[_document()], source_manifest=_manifest(2))
    with pytest.raises(DataValidationError, match="IDs must be unique"):
        cleaner.clean(
            documents=[_document(), _document()], source_manifest=_manifest(2)
        )


def test_exact_noise_tokens_are_configurable_without_substring_matching() -> None:
    """Dataset-independent policy removes only explicit class tokens."""
    cleaner = LegalHtmlCleaner(
        HtmlCleaningConfig(noise_class_tokens=frozenset({"remove-me"}))
    )
    html = (
        '<div class="remove-me">noise</div>'
        '<div class="remove-me-not">Điều 4.</div>'
    )

    assert cleaner.clean_html(html) == "Điều 4."
