"""Integration tests for memory-bounded parsing and chunk artifact creation."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.offline.chunking import LegalChunker
from legal_agentic_rag.offline.document_processing import (
    StreamingDocumentProcessor,
)
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.runtime import load_model_artifact
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalBlock,
    LegalChunk,
    LegalDocument,
)


class _OnePassDocuments:
    def __init__(self, documents: list[LegalDocument]) -> None:
        self._documents = documents
        self.iteration_count = 0

    def __iter__(self) -> Iterator[LegalDocument]:
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise AssertionError("document corpus was iterated more than once")
        yield from self._documents


def _documents() -> list[LegalDocument]:
    return [
        LegalDocument(
            document_id="doc-1",
            title="Văn bản thứ nhất",
            content_html="<p>fixture</p>",
            clean_text=(
                "Chương I\n"
                "QUY ĐỊNH CHUNG\n"
                "Điều 1. Phạm vi điều chỉnh\n"
                "1. Quy định này áp dụng cho doanh nghiệp.\n"
                "2. Không áp dụng trong trường hợp được miễn."
            ),
            has_content=True,
            source_dataset="aio",
        ),
        LegalDocument(
            document_id="doc-2",
            title="Văn bản thứ hai",
            content_html="<p>fixture</p>",
            clean_text=(
                "Điều 2. Thời hạn\n"
                "1. Thời hạn thực hiện là 30 ngày.\n"
                "2. Trừ trường hợp pháp luật có quy định khác."
            ),
            has_content=True,
            source_dataset="aio",
        ),
    ]


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.CLEANED_DOCUMENTS,
        artifact_version="1.0",
        dataset_name="fixture",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        record_count=2,
        processing_config_hash="cleaned-hash",
        code_version="fixture",
        metadata={"payload_sha256": "cleaned-payload-hash"},
    )


def test_streaming_processing_matches_existing_parser_and_chunker(
    tmp_path: Path,
) -> None:
    """One-pass production orchestration preserves established legal output."""
    documents = _documents()
    source = _manifest()
    clock = lambda: datetime(2026, 7, 25, tzinfo=UTC)
    parser = LegalStructureParser(clock=clock)
    chunker = LegalChunker(clock=clock)
    direct_parsing = parser.parse(
        documents=documents,
        source_manifest=source,
    )
    direct_chunking = chunker.chunk(
        documents=documents,
        blocks=direct_parsing.blocks,
        source_manifest=direct_parsing.manifest,
    )
    one_pass = _OnePassDocuments(documents)

    result = StreamingDocumentProcessor(
        parser,
        chunker,
        progress_interval_documents=1,
    ).process(
        documents=one_pass,
        source_manifest=source,
        normalized_processing_config_hash="normalized-hash",
        blocks_destination=tmp_path / "blocks",
        chunks_destination=tmp_path / "chunks",
    )
    blocks, stored_block_manifest = load_model_artifact(
        tmp_path / "blocks",
        expected_type=ArtifactType.LEGAL_BLOCKS,
        record_type=LegalBlock,
    )
    chunks, stored_chunk_manifest = load_model_artifact(
        tmp_path / "chunks",
        expected_type=ArtifactType.LEGAL_CHUNKS,
        record_type=LegalChunk,
    )

    assert one_pass.iteration_count == 1
    assert blocks == direct_parsing.blocks
    assert chunks == direct_chunking.chunks
    assert stored_block_manifest == result.block_manifest
    assert stored_chunk_manifest == result.chunk_manifest
    assert (
        stored_chunk_manifest.metadata[
            "runtime_normalized_processing_config_hash"
        ]
        == "normalized-hash"
    )


def test_streaming_processing_does_not_publish_duplicate_input(
    tmp_path: Path,
) -> None:
    """A validation failure leaves neither blocks nor chunks visible."""
    document = _documents()[0]
    source = _manifest().model_copy(update={"record_count": 2})
    processor = StreamingDocumentProcessor(
        LegalStructureParser(),
        LegalChunker(),
        progress_interval_documents=1,
    )

    with pytest.raises(DataValidationError, match="duplicate"):
        processor.process(
            documents=[document, document],
            source_manifest=source,
            normalized_processing_config_hash="normalized-hash",
            blocks_destination=tmp_path / "blocks",
            chunks_destination=tmp_path / "chunks",
        )

    assert not (tmp_path / "blocks").exists()
    assert not (tmp_path / "chunks").exists()
