"""Build bounded legal-context text for cross-encoder candidate scoring."""

from __future__ import annotations

from typing import Any

from legal_agentic_rag.schemas.retrieval import RetrievalHit

_METADATA_LABELS = (
    ("document_title", "Tên văn bản"),
    ("document_number", "Số ký hiệu"),
    ("document_type", "Loại văn bản"),
    ("issuing_authority", "Cơ quan ban hành"),
    ("legal_field", "Lĩnh vực"),
    ("effect_status", "Tình trạng hiệu lực"),
    ("effective_date", "Ngày hiệu lực"),
    ("expiry_date", "Ngày hết hiệu lực"),
)
_STRUCTURE_LABELS = (
    ("part", "Phần"),
    ("chapter", "Chương"),
    ("section", "Mục"),
    ("subsection", "Tiểu mục"),
    ("article_number", "Điều"),
    ("article_title", "Tên điều"),
    ("clause_numbers", "Khoản"),
    ("point_numbers", "Điểm"),
)
_HEADING_TYPE_MAP = {
    "PART": "Phần",
    "CHAPTER": "Chương",
    "SECTION": "Mục",
    "SUBSECTION": "Tiểu mục",
    "APPENDIX": "Phụ lục",
}


def build_legal_rerank_text(hit: RetrievalHit) -> str:
    """Prepend trusted unified legal metadata to one immutable chunk text."""
    lines: list[str] = []

    # 1. V2 Document Identity Support
    doc_identity = hit.metadata.get("document_identity")
    if isinstance(doc_identity, dict):
        title = _display_value(doc_identity.get("title"))
        if title is not None:
            lines.append(f"Tên văn bản: {title}")
        doc_num = _display_value(doc_identity.get("document_number"))
        if doc_num is not None:
            lines.append(f"Số ký hiệu: {doc_num}")

    # 2. Legacy Flat Metadata Support
    for field_name, label in _METADATA_LABELS:
        value = _display_value(hit.metadata.get(field_name))
        if value is not None:
            lines.append(f"{label}: {value}")

    # 3. V2 Hierarchy Support
    hierarchy = hit.metadata.get("hierarchy")
    if isinstance(hierarchy, dict):
        heading_path = hierarchy.get("heading_path")
        if isinstance(heading_path, list):
            for item in heading_path:
                if isinstance(item, dict):
                    h_text = _format_v2_heading_item(item)
                    if h_text:
                        lines.append(h_text)

        art_lbl = _display_value(hierarchy.get("article_label"))
        if art_lbl is not None:
            lines.append(f"Điều: {art_lbl}")

        cl_lbl = _display_value(hierarchy.get("clause_label"))
        if cl_lbl is not None:
            lines.append(f"Khoản: {cl_lbl}")

        pt_lbl = _display_value(hierarchy.get("point_label"))
        if pt_lbl is not None:
            lines.append(f"Điểm: {pt_lbl}")

    # 4. Legacy Structure Support
    structure = hit.metadata.get("structure")
    if isinstance(structure, dict):
        for field_name, label in _STRUCTURE_LABELS:
            value = _display_value(structure.get(field_name))
            if value is not None:
                lines.append(f"{label}: {value}")

    lines.extend(("Nội dung:", hit.text))
    return "\n".join(lines)


def _format_v2_heading_item(item: dict[str, Any]) -> str | None:
    raw_type = str(item.get("type", "")).strip().upper()
    type_label = _HEADING_TYPE_MAP.get(raw_type, str(item.get("type", "")).strip())
    label = str(item.get("label", "")).strip() if item.get("label") is not None else ""
    title = str(item.get("title", "")).strip() if item.get("title") is not None else ""

    prefix = f"{type_label} {label}".strip() if (type_label or label) else ""
    if prefix and title:
        return f"{prefix}: {title}"
    if prefix:
        return prefix
    if title:
        return title
    return None


def _display_value(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        normalized = [item.strip() for item in value if item.strip()]
        return ", ".join(normalized) or None
    return None
