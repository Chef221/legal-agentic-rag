"""Build bounded legal-context text for cross-encoder candidate scoring."""

from __future__ import annotations

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


def build_legal_rerank_text(hit: RetrievalHit) -> str:
    """Prepend trusted unified legal metadata to one immutable chunk text."""
    lines: list[str] = []
    for field_name, label in _METADATA_LABELS:
        value = _display_value(hit.metadata.get(field_name))
        if value is not None:
            lines.append(f"{label}: {value}")

    structure = hit.metadata.get("structure")
    if isinstance(structure, dict):
        for field_name, label in _STRUCTURE_LABELS:
            value = _display_value(structure.get(field_name))
            if value is not None:
                lines.append(f"{label}: {value}")

    lines.extend(("Nội dung:", hit.text))
    return "\n".join(lines)


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
