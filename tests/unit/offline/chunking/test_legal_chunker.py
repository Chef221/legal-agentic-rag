"""Unit tests for article-first legal chunking and validation."""

from copy import deepcopy
from datetime import UTC, date, datetime

import pytest

from legal_agentic_rag.configuration import ChunkingConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.offline.chunking import LegalChunker
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalBlock,
    LegalBlockType,
    LegalDocument,
    LegalStructure,
)


def _document(document_id: str = "doc-1") -> LegalDocument:
    return LegalDocument(
        document_id=document_id,
        title="Luật mẫu",
        document_number="01/2026/QH",
        document_type="Luật",
        issuance_date=date(2026, 1, 1),
        effective_date=date(2026, 2, 1),
        effect_status="Còn hiệu lực",
        issuing_authority="Quốc hội",
        legal_field="Pháp luật",
        source_url="https://example.test/doc-1",
        clean_text="fixture",
        has_content=True,
        source_dataset="aio",
    )


def _structure(
    *,
    article: str | None = None,
    article_title: str | None = None,
    clause: str | None = None,
    point: str | None = None,
) -> LegalStructure:
    path = ["Chương I"]
    if article is not None:
        path.append(f"Điều {article}")
    if clause is not None:
        path.append(f"Khoản {clause}")
    if point is not None:
        path.append(f"Điểm {point}")
    return LegalStructure(
        chapter="Chương I",
        article_number=article,
        article_title=article_title,
        clause_numbers=[clause] if clause is not None else [],
        point_numbers=[point] if point is not None else [],
        structure_path=path,
    )


def _block(
    block_id: str,
    block_type: LegalBlockType,
    text: str,
    order_index: int,
    *,
    parent_block_id: str | None = None,
    number: str | None = None,
    structure: LegalStructure | None = None,
    document_id: str = "doc-1",
) -> LegalBlock:
    return LegalBlock(
        block_id=block_id,
        document_id=document_id,
        block_type=block_type,
        block_number=number,
        text=text,
        parent_block_id=parent_block_id,
        order_index=order_index,
        structure=structure or LegalStructure(),
    )


def _manifest(
    record_count: int,
    artifact_type: ArtifactType = ArtifactType.LEGAL_BLOCKS,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=artifact_type,
        artifact_version="1.0",
        dataset_name="th1nhng0/vietnamese-legal-documents",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        record_count=record_count,
        processing_config_hash="parsing-hash",
        code_version="0.5.0",
    )


def _article_blocks(clause_texts: list[str]) -> list[LegalBlock]:
    article_id = "article-1"
    blocks = [
        _block(
            article_id,
            LegalBlockType.ARTICLE,
            "Điều 1. Phạm vi áp dụng",
            0,
            number="1",
            structure=_structure(article="1", article_title="Phạm vi áp dụng"),
        )
    ]
    for index, clause_text in enumerate(clause_texts, start=1):
        blocks.append(
            _block(
                f"clause-{index}",
                LegalBlockType.CLAUSE,
                clause_text,
                index,
                parent_block_id=article_id,
                number=str(index),
                structure=_structure(
                    article="1",
                    article_title="Phạm vi áp dụng",
                    clause=str(index),
                ),
            )
        )
    return blocks


def test_article_within_limit_becomes_one_chunk_with_document_metadata() -> None:
    """One complete article is the preferred retrieval unit."""
    blocks = _article_blocks(
        [
            "1. Không áp dụng đối với trường hợp đặc biệt.",
            "2. Mức phạt là 10.000.000 đồng.",
        ]
    )
    config = ChunkingConfig(max_tokens=100, min_tokens=1, overlap_tokens=10)

    result = LegalChunker(config).chunk(
        documents=[_document()], blocks=blocks, source_manifest=_manifest(3)
    )

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.metadata["chunk_strategy"] == "article"
    assert chunk.metadata["source_block_ids"] == [
        "article-1",
        "clause-1",
        "clause-2",
    ]
    assert chunk.structure.article_number == "1"
    assert chunk.structure.clause_numbers == ["1", "2"]
    assert chunk.document_title == "Luật mẫu"
    assert chunk.document_number == "01/2026/QH"
    assert chunk.effect_status == "Còn hiệu lực"
    assert chunk.source_dataset == "aio"
    assert "Văn bản: Luật mẫu" in chunk.search_text
    assert "Số ký hiệu: 01/2026/QH" in chunk.search_text
    assert "Điều 1: Phạm vi áp dụng" in chunk.search_text
    assert "Không áp dụng" in chunk.text
    assert "10.000.000 đồng" in chunk.text
    assert result.diagnostics[0].block_coverage == 1.0
    assert result.article_chunk_count == 1
    assert result.manifest.artifact_type == ArtifactType.LEGAL_CHUNKS


def test_long_article_groups_consecutive_clauses_without_token_split() -> None:
    """Clause boundaries are preferred when a full article exceeds the limit."""
    blocks = _article_blocks(
        [
            "1. Nội dung một không áp dụng.",
            "2. Nội dung hai phải thực hiện.",
            "3. Nội dung ba chưa hết hiệu lực.",
        ]
    )
    config = ChunkingConfig(max_tokens=18, min_tokens=1, overlap_tokens=3)

    result = LegalChunker(config).chunk(
        documents=[_document()], blocks=blocks, source_manifest=_manifest(4)
    )

    assert len(result.chunks) >= 2
    assert all(
        chunk.metadata["chunk_strategy"] == "clause_group"
        for chunk in result.chunks
    )
    assert all(chunk.token_count <= 18 for chunk in result.chunks)
    assert result.clause_fallback_chunk_count == len(result.chunks)
    assert result.token_fallback_chunk_count == 0
    covered = {
        source_id
        for chunk in result.chunks
        for source_id in chunk.metadata["source_block_ids"]
    }
    assert covered == {block.block_id for block in blocks}


def test_oversized_clause_uses_overlapping_token_fallback() -> None:
    """A clause too large for one chunk is split only at token boundaries."""
    long_clause = "1. " + " ".join(f"từ{i}" for i in range(30))
    blocks = _article_blocks([long_clause])
    config = ChunkingConfig(max_tokens=10, min_tokens=1, overlap_tokens=2)

    result = LegalChunker(config).chunk(
        documents=[_document()], blocks=blocks, source_manifest=_manifest(2)
    )

    assert len(result.chunks) > 1
    assert all(
        chunk.metadata["chunk_strategy"] == "token_fallback"
        for chunk in result.chunks
    )
    assert all(1 <= chunk.token_count <= 10 for chunk in result.chunks)
    assert all(chunk.metadata["split_count"] == len(result.chunks) for chunk in result.chunks)
    assert [chunk.metadata["split_index"] for chunk in result.chunks] == list(
        range(len(result.chunks))
    )
    first_tokens = result.chunks[0].text.split()
    second_tokens = result.chunks[1].text.split()
    assert set(first_tokens[-2:]) & set(second_tokens[:3])
    assert result.token_fallback_chunk_count == len(result.chunks)
    assert result.diagnostics[0].block_coverage == 1.0


def test_non_article_block_becomes_standalone_chunk() -> None:
    """Preamble and unstructured legal text are never silently discarded."""
    block = _block(
        "document-1",
        LegalBlockType.DOCUMENT,
        "THÔNG BÁO\nNội dung chưa áp dụng quy định mới.",
        0,
    )

    result = LegalChunker(
        ChunkingConfig(max_tokens=100, min_tokens=1, overlap_tokens=10)
    ).chunk(documents=[_document()], blocks=[block], source_manifest=_manifest(1))

    assert len(result.chunks) == 1
    assert result.chunks[0].metadata["chunk_strategy"] == "standalone_block"
    assert result.chunks[0].text == block.text
    assert result.standalone_chunk_count == 1


def test_chunker_is_deterministic_and_does_not_mutate_inputs() -> None:
    """Stable content yields stable chunk IDs and leaves source models intact."""
    documents = [_document()]
    blocks = _article_blocks(["1. Nội dung áp dụng."])
    documents_before = deepcopy(documents)
    blocks_before = deepcopy(blocks)
    chunker = LegalChunker(
        ChunkingConfig(max_tokens=100, min_tokens=1, overlap_tokens=10),
        clock=lambda: datetime(2026, 7, 18, tzinfo=UTC),
    )

    first = chunker.chunk(
        documents=documents, blocks=blocks, source_manifest=_manifest(2)
    )
    second = chunker.chunk(
        documents=documents, blocks=blocks, source_manifest=_manifest(2)
    )

    assert documents == documents_before
    assert blocks == blocks_before
    assert first.chunks == second.chunks
    assert first.diagnostics == second.diagnostics
    assert first.manifest.processing_config_hash == (
        second.manifest.processing_config_hash
    )


def test_document_without_blocks_remains_in_diagnostics() -> None:
    """A parser output without blocks produces a warning, not a dropped document."""
    result = LegalChunker().chunk(
        documents=[_document()], blocks=[], source_manifest=_manifest(0)
    )

    assert result.chunks == []
    assert result.documents_without_chunks_count == 1
    assert result.diagnostics[0].block_coverage == 0.0
    assert result.issues[0].issue_type == "missing_legal_blocks"


def test_chunker_rejects_incompatible_manifest_and_invalid_block_graph() -> None:
    """Artifact, count, identity, and parent errors fail before chunking."""
    chunker = LegalChunker()
    block = _block("block-1", LegalBlockType.DOCUMENT, "Nội dung", 0)
    with pytest.raises(ArtifactCompatibilityError):
        chunker.chunk(
            documents=[_document()],
            blocks=[block],
            source_manifest=_manifest(1, ArtifactType.LEGAL_CHUNKS),
        )
    with pytest.raises(DataValidationError, match="record count"):
        chunker.chunk(
            documents=[_document()], blocks=[block], source_manifest=_manifest(2)
        )
    bad_parent = _block(
        "block-2",
        LegalBlockType.CLAUSE,
        "1. Nội dung",
        0,
        parent_block_id="missing",
    )
    with pytest.raises(DataValidationError, match="parent"):
        chunker.chunk(
            documents=[_document()],
            blocks=[bad_parent],
            source_manifest=_manifest(1),
        )
