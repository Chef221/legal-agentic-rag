"""Unit tests for conservative AIO document normalization."""

from copy import deepcopy
from datetime import UTC, date, datetime

import pytest

from legal_agentic_rag.configuration import DocumentNormalizationConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.offline.datasets.aio import AioDocumentNormalizer
from legal_agentic_rag.schemas import DatasetManifest


def _manifest(dataset_name: str = "th1nhng0/vietnamese-legal-documents") -> DatasetManifest:
    return DatasetManifest(
        schema_version="1.0",
        dataset_name=dataset_name,
        dataset_revision="fixture-revision",
        loaded_at=datetime(2026, 7, 18, tzinfo=UTC),
        configs=["metadata", "content", "relationships"],
        record_counts={"metadata": 1, "content": 1, "relationships": 0},
        processing_config_hash="dataset-config-hash",
    )


def test_normalizer_maps_all_fields_and_preserves_raw_data_and_html() -> None:
    """Known AIO fields become unified fields without mutating legal content."""
    metadata = {
        "id": 101,
        "title": "  Luật mẫu  ",
        "so_ky_hieu": " 01/2026/QH ",
        "loai_van_ban": "Luật",
        "ngay_ban_hanh": "01/01/2026",
        "ngay_co_hieu_luc": "2026-02-01",
        "ngay_het_hieu_luc": "",
        "tinh_trang_hieu_luc": "Còn hiệu lực",
        "co_quan_ban_hanh": "Quốc hội",
        "chuc_danh": "Chủ tịch",
        "nguoi_ky": "Nguyễn Văn A",
        "nganh": "Tư pháp",
        "linh_vuc": "Pháp luật",
        "pham_vi": "Toàn quốc",
        "thong_tin_ap_dung": "Áp dụng chung",
        "ngay_dang_cong_bao": "03/01/2026",
        "source_url": " https://example.test/legal/101 ",
        "unknown_source_field": {"kept": True},
    }
    content_html = "  <p>Điều 1. Phạm vi áp dụng.</p>  "
    metadata_before = deepcopy(metadata)
    config = DocumentNormalizationConfig(
        effect_status_mapping={"Còn hiệu lực": "effective"},
        document_type_mapping={"Luật": "law"},
    )

    result = AioDocumentNormalizer(
        config,
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC),
    ).normalize(
        metadata_records=[metadata],
        content_records=[{"id": 101, "content_html": content_html}],
        dataset_manifest=_manifest(),
    )

    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.document_id == "101"
    assert document.title == "Luật mẫu"
    assert document.document_number == "01/2026/QH"
    assert document.document_type == "law"
    assert document.issuance_date == date(2026, 1, 1)
    assert document.effective_date == date(2026, 2, 1)
    assert document.expiry_date is None
    assert document.effect_status == "effective"
    assert document.issuing_authority == "Quốc hội"
    assert document.position_title == "Chủ tịch"
    assert document.signer == "Nguyễn Văn A"
    assert document.sector == "Tư pháp"
    assert document.legal_field == "Pháp luật"
    assert document.scope == "Toàn quốc"
    assert document.application_info == "Áp dụng chung"
    assert document.publication_date == date(2026, 1, 3)
    assert document.source_url == "https://example.test/legal/101"
    assert document.content_html == content_html
    assert document.clean_text is None
    assert document.has_content is True
    assert document.source_dataset == "aio"
    assert document.raw_metadata["unknown_source_field"] == {"kept": True}
    assert metadata == metadata_before
    assert result.issues == []
    assert result.manifest.record_count == 1
    assert len(result.manifest.processing_config_hash) == 64


def test_normalizer_reports_invalid_and_ambiguous_records_without_guessing() -> None:
    """Bad dates, IDs, URLs, metadata, and joins remain explicit issues."""
    metadata = [
        {
            "id": "doc-1",
            "title": ["not", "text"],
            "ngay_ban_hanh": "2026-03-01",
            "ngay_co_hieu_luc": "2026-02-01",
            "ngay_het_hieu_luc": "not-a-date",
            "source_url": "not-a-url",
        },
        {"id": " ", "title": "Invalid ID"},
        {"id": "doc-dup", "title": "First"},
        {"id": "doc-dup", "title": "Second"},
        {"id": "doc-missing", "title": "Ambiguous content"},
    ]
    content = [
        {"id": "doc-1", "content_html": ""},
        {"id": "doc-missing", "content_html": "<p>First</p>"},
        {"id": "doc-missing", "content_html": "<p>Second</p>"},
        {"id": "orphan", "content_html": "<p>Orphan</p>"},
        {"id": {"bad": "id"}, "content_html": "<p>Invalid ID</p>"},
    ]

    result = AioDocumentNormalizer(
        clock=lambda: datetime(2026, 7, 18, 1, tzinfo=UTC)
    ).normalize(
        metadata_records=metadata,
        content_records=content,
        dataset_manifest=_manifest(),
    )

    assert [document.document_id for document in result.documents] == [
        "doc-1",
        "doc-missing",
    ]
    assert all(document.has_content is False for document in result.documents)
    assert result.input_metadata_count == 5
    assert result.input_content_count == 5
    assert result.rejected_metadata_count == 3
    assert result.orphan_content_count == 1
    assert result.ambiguous_content_count == 1
    issue_types = {issue.issue_type for issue in result.issues}
    assert {
        "invalid_metadata_id",
        "duplicate_metadata_id",
        "ambiguous_content",
        "orphan_content",
        "invalid_content_id",
        "invalid_content",
        "invalid_metadata_value",
        "invalid_date",
        "invalid_source_url",
        "effective_before_issuance",
    } <= issue_types
    doc_1 = next(document for document in result.documents if document.document_id == "doc-1")
    assert doc_1.title is None
    assert doc_1.expiry_date is None
    assert doc_1.issuance_date == date(2026, 3, 1)
    assert doc_1.effective_date == date(2026, 2, 1)
    assert doc_1.source_url is None
    assert result.manifest.metadata["rejected_metadata_count"] == 3


def test_normalizer_rejects_manifest_for_another_dataset() -> None:
    """The AIO-specific adapter cannot normalize records from another source."""
    with pytest.raises(DataValidationError, match="another dataset"):
        AioDocumentNormalizer().normalize(
            metadata_records=[],
            content_records=[],
            dataset_manifest=_manifest("another/dataset"),
        )
