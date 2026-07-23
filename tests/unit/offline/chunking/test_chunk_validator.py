"""Unit tests for post-build legal chunk validation."""

from legal_agentic_rag.configuration import ChunkingConfig
from legal_agentic_rag.offline.chunking import LegalChunkValidator
from legal_agentic_rag.schemas import (
    LegalBlock,
    LegalBlockType,
    LegalChunk,
    LegalDocument,
)


def test_validator_reports_token_metadata_and_coverage_errors() -> None:
    """Validator surfaces production invariants without changing chunks."""
    document = LegalDocument(
        document_id="doc-1",
        title="Luật đúng",
        has_content=True,
        source_dataset="aio",
    )
    block = LegalBlock(
        block_id="block-1",
        document_id="doc-1",
        block_type=LegalBlockType.DOCUMENT,
        text="Nội dung pháp lý",
        order_index=0,
    )
    chunk = LegalChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        text="Nội dung pháp lý",
        search_text="Nội dung pháp lý",
        token_count=99,
        document_title="Sai tiêu đề",
        source_dataset="aio",
        metadata={
            "chunk_strategy": "standalone_block",
            "source_block_ids": [],
        },
    )

    issues = LegalChunkValidator(
        ChunkingConfig(max_tokens=10, min_tokens=1, overlap_tokens=2)
    ).validate(documents=[document], blocks=[block], chunks=[chunk])

    issue_types = {issue.issue_type for issue in issues}
    assert {
        "chunk_too_long",
        "metadata_inheritance_mismatch",
        "token_count_mismatch",
        "uncovered_source_block",
    } <= issue_types
