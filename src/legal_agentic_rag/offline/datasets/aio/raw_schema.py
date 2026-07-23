"""Raw AIO names isolated from the rest of the application."""

from legal_agentic_rag.contracts.dataset_source import DatasetComponent

AIO_DATASET_NAME = "th1nhng0/vietnamese-legal-documents"

IDENTIFIER_FIELDS: dict[DatasetComponent, str] = {
    DatasetComponent.METADATA: "id",
    DatasetComponent.CONTENT: "id",
}

REQUIRED_FIELDS: dict[DatasetComponent, frozenset[str]] = {
    DatasetComponent.METADATA: frozenset({"id"}),
    DatasetComponent.CONTENT: frozenset({"id", "content_html"}),
    DatasetComponent.RELATIONSHIPS: frozenset(
        {"doc_id", "other_doc_id", "relationship"}
    ),
}

CONTENT_FIELD = "content_html"
RELATIONSHIP_SOURCE_FIELD = "doc_id"
RELATIONSHIP_TARGET_FIELD = "other_doc_id"
RELATIONSHIP_LABEL_FIELD = "relationship"

METADATA_TITLE_FIELD = "title"
METADATA_NUMBER_FIELD = "so_ky_hieu"
METADATA_TYPE_FIELD = "loai_van_ban"
METADATA_ISSUED_DATE_FIELD = "ngay_ban_hanh"
METADATA_EFFECTIVE_DATE_FIELD = "ngay_co_hieu_luc"
METADATA_EXPIRY_DATE_FIELD = "ngay_het_hieu_luc"
METADATA_EFFECT_STATUS_FIELD = "tinh_trang_hieu_luc"
METADATA_AUTHORITY_FIELD = "co_quan_ban_hanh"
METADATA_POSITION_FIELD = "chuc_danh"
METADATA_SIGNER_FIELD = "nguoi_ky"
METADATA_SECTOR_FIELD = "nganh"
METADATA_LEGAL_FIELD = "linh_vuc"
METADATA_SCOPE_FIELD = "pham_vi"
METADATA_APPLICATION_FIELD = "thong_tin_ap_dung"
METADATA_PUBLICATION_DATE_FIELD = "ngay_dang_cong_bao"
METADATA_SOURCE_URL_FIELD = "source_url"
