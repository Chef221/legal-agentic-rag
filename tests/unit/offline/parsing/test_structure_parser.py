"""Unit tests for deterministic Vietnamese legal structure parsing."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalBlockType,
    LegalDocument,
)


def _document(document_id: str, clean_text: str | None) -> LegalDocument:
    return LegalDocument(
        document_id=document_id,
        title="Văn bản mẫu",
        content_html="<p>fixture</p>" if clean_text is not None else None,
        clean_text=clean_text,
        has_content=clean_text is not None,
        source_dataset="aio",
    )


def _manifest(
    record_count: int = 1,
    artifact_type: ArtifactType = ArtifactType.CLEANED_DOCUMENTS,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=artifact_type,
        artifact_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=record_count,
        processing_config_hash="cleaning-hash",
        code_version="0.4.0",
    )


def test_parser_builds_standard_hierarchy_and_preserves_all_text(
    load_clean_text_fixture,
) -> None:
    """Standard markers become ordered blocks with traceable parents."""
    clean_text = load_clean_text_fixture("legal_structure_standard.txt")
    document = _document("doc-standard", clean_text)

    result = LegalStructureParser(
        clock=lambda: datetime(2026, 7, 18, 3, tzinfo=UTC)
    ).parse(documents=[document], source_manifest=_manifest())

    assert [block.block_type for block in result.blocks] == [
        LegalBlockType.DOCUMENT,
        LegalBlockType.CHAPTER,
        LegalBlockType.SECTION,
        LegalBlockType.ARTICLE,
        LegalBlockType.CLAUSE,
        LegalBlockType.POINT,
        LegalBlockType.POINT,
        LegalBlockType.CLAUSE,
        LegalBlockType.ARTICLE,
        LegalBlockType.APPENDIX,
        LegalBlockType.TABLE,
    ]
    chapter, section, article = result.blocks[1:4]
    assert chapter.block_number == "I"
    assert chapter.title == "QUY ĐỊNH CHUNG"
    assert section.block_number == "1"
    assert section.title == "PHẠM VI ÁP DỤNG"
    assert article.block_number == "1"
    assert article.title == "Phạm vi điều chỉnh"
    assert article.structure.chapter == "Chương I"
    assert article.structure.section == "Mục 1"
    assert article.structure.article_number == "1"
    assert article.structure.structure_path == ["Chương I", "Mục 1", "Điều 1"]

    clause = result.blocks[4]
    first_point = result.blocks[5]
    assert clause.parent_block_id == article.block_id
    assert clause.structure.clause_numbers == ["1"]
    assert first_point.parent_block_id == clause.block_id
    assert first_point.structure.point_numbers == ["a"]
    assert "không áp dụng" in clause.text
    assert "10.000.000 đồng" in result.blocks[7].text
    assert "trừ trường hợp" in result.blocks[7].text

    second_article = result.blocks[8]
    assert second_article.title == "Đối tượng áp dụng"
    assert "phải thực hiện" in second_article.text
    table = result.blocks[-1]
    assert table.block_type == LegalBlockType.TABLE
    assert table.parent_block_id == result.blocks[-2].block_id
    assert "Hành vi | Mức phạt" in table.text
    assert result.diagnostics[0].text_coverage == 1.0
    assert result.manifest.artifact_type == ArtifactType.LEGAL_BLOCKS
    assert result.manifest.record_count == 11
    assert result.issues == []


def test_parser_handles_missing_levels_roman_articles_and_discontinuous_numbers() -> None:
    """An article can stand alone and numbering gaps never drop legal text."""
    clean_text = "\n".join(
        [
            "Điều IV. Quy định độc lập",
            "Khoản 1. Nội dung thứ nhất.",
            "Điểm a) Không áp dụng trong trường hợp đặc biệt.",
            "3. Nội dung thứ ba vẫn được giữ.",
            "Đoạn không đánh số vẫn thuộc khoản 3.",
        ]
    )

    result = LegalStructureParser().parse(
        documents=[_document("doc-missing-levels", clean_text)],
        source_manifest=_manifest(),
    )

    assert [block.block_type for block in result.blocks] == [
        LegalBlockType.ARTICLE,
        LegalBlockType.CLAUSE,
        LegalBlockType.POINT,
        LegalBlockType.CLAUSE,
    ]
    assert result.blocks[0].block_number == "IV"
    assert result.blocks[1].block_number == "1"
    assert result.blocks[2].block_number == "a"
    assert result.blocks[3].block_number == "3"
    assert "Đoạn không đánh số" in result.blocks[3].text
    assert result.diagnostics[0].text_coverage == 1.0


def test_parser_preserves_malformed_marker_and_reports_diagnostic() -> None:
    """Unsupported numbering remains ordinary text with an explicit issue."""
    text = "Điều A. Marker không hợp lệ\nNội dung không được sửa hoặc bỏ."

    result = LegalStructureParser().parse(
        documents=[_document("doc-malformed", text)],
        source_manifest=_manifest(),
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].block_type == LegalBlockType.DOCUMENT
    assert result.blocks[0].text == text
    assert {issue.issue_type for issue in result.issues} == {
        "no_legal_structure",
        "unrecognized_structure_marker",
    }
    assert result.diagnostics[0].text_coverage == 1.0


def test_parser_keeps_no_structure_document_as_one_document_block() -> None:
    """Administrative prose without markers remains usable and fully covered."""
    text = "THÔNG BÁO\nCơ quan có thẩm quyền chưa áp dụng quy định mới."

    result = LegalStructureParser().parse(
        documents=[_document("doc-prose", text)], source_manifest=_manifest()
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].block_type == LegalBlockType.DOCUMENT
    assert result.blocks[0].text == text
    assert result.unstructured_document_count == 1
    assert result.structured_document_count == 0
    assert result.diagnostics[0].text_coverage == 1.0
    assert [issue.issue_type for issue in result.issues] == ["no_legal_structure"]


def test_parser_is_deterministic_and_does_not_mutate_documents() -> None:
    """The same clean text yields stable IDs, hierarchy, and source documents."""
    documents = [_document("doc-repeat", "Điều 1: Không sửa nội dung.")]
    before = deepcopy(documents)
    parser = LegalStructureParser(clock=lambda: datetime(2026, 7, 18, tzinfo=UTC))

    first = parser.parse(documents=documents, source_manifest=_manifest())
    second = parser.parse(documents=documents, source_manifest=_manifest())

    assert documents == before
    assert first.blocks == second.blocks
    assert first.diagnostics == second.diagnostics
    assert first.manifest.processing_config_hash == (
        second.manifest.processing_config_hash
    )


def test_parser_classifies_missing_clean_text_without_dropping_document() -> None:
    """Documents without cleaned content remain visible in typed diagnostics."""
    documents = [
        _document("doc-structured", "Điều 1. Có nội dung."),
        _document("doc-missing", None),
    ]

    result = LegalStructureParser().parse(
        documents=documents, source_manifest=_manifest(2)
    )

    assert result.input_document_count == 2
    assert result.parsed_document_count == 1
    assert result.missing_clean_text_count == 1
    assert len(result.documents) == 2
    assert [block.document_id for block in result.blocks] == ["doc-structured"]
    assert result.diagnostics[1].text_coverage == 0.0
    assert result.issues[0].issue_type == "missing_clean_text"


def test_parser_rejects_wrong_artifact_count_and_duplicate_ids() -> None:
    """Upstream artifact and document identity mismatches fail explicitly."""
    parser = LegalStructureParser()
    document = _document("doc-1", "Điều 1. Nội dung.")
    with pytest.raises(ArtifactCompatibilityError):
        parser.parse(
            documents=[document],
            source_manifest=_manifest(artifact_type=ArtifactType.LEGAL_BLOCKS),
        )
    with pytest.raises(DataValidationError, match="record count"):
        parser.parse(documents=[document], source_manifest=_manifest(2))
    with pytest.raises(DataValidationError, match="IDs must be unique"):
        parser.parse(documents=[document, document], source_manifest=_manifest(2))
