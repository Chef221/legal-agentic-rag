"""Memory-bounded legal parsing and chunking artifact orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import logging
from pathlib import Path

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.offline.chunking import (
    ChunkedLegalDocument,
    LegalChunker,
)
from legal_agentic_rag.offline.parsing import (
    LegalStructureParser,
    ParsedLegalDocument,
)
from legal_agentic_rag.runtime.artifact_store import ModelArtifactWriter
from legal_agentic_rag.schemas.auditing import AuditSeverity
from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalDocument,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

_LOGGER = logging.getLogger(__name__)


@dataclass
class _ParsingCounters:
    document_count: int = 0
    block_count: int = 0
    issue_count: int = 0
    parsed_count: int = 0
    missing_count: int = 0
    structured_count: int = 0
    source_characters: int = 0
    covered_characters: int = 0


@dataclass
class _ChunkingCounters:
    document_count: int = 0
    block_count: int = 0
    chunk_count: int = 0
    documents_with_chunks: int = 0
    issue_count: int = 0
    strategy_counts: dict[str, int] = field(
        default_factory=lambda: {
            "article": 0,
            "clause_group": 0,
            "token_fallback": 0,
            "standalone_block": 0,
        }
    )


@dataclass(frozen=True)
class StreamingDocumentProcessingResult:
    """Final manifests and issue counts from bounded document processing."""

    block_manifest: ArtifactManifest
    chunk_manifest: ArtifactManifest
    parser_issue_count: int
    chunking_issue_count: int


class StreamingDocumentProcessor:
    """Parse and chunk one document at a time into atomic JSONL artifacts."""

    def __init__(
        self,
        parser: LegalStructureParser,
        chunker: LegalChunker,
        *,
        progress_interval_documents: int,
    ) -> None:
        if progress_interval_documents <= 0:
            raise ValueError("progress interval must be positive")
        self._parser = parser
        self._chunker = chunker
        self._progress_interval = progress_interval_documents

    def process(
        self,
        *,
        documents: Iterable[LegalDocument],
        source_manifest: ArtifactManifest,
        normalized_processing_config_hash: str,
        blocks_destination: Path,
        chunks_destination: Path,
    ) -> StreamingDocumentProcessingResult:
        """Create legal-block and legal-chunk artifacts in a single bounded pass."""
        self._validate_cleaned_manifest(source_manifest)
        parsing = _ParsingCounters()
        chunking = _ChunkingCounters()
        seen_document_ids: set[str] = set()
        _LOGGER.info(
            "streaming_document_processing_started",
            extra={
                "dataset_name": source_manifest.dataset_name,
                "document_count": source_manifest.record_count,
            },
        )
        with (
            ModelArtifactWriter(blocks_destination) as block_writer,
            ModelArtifactWriter(chunks_destination) as chunk_writer,
        ):
            for document in documents:
                self._validate_document_identity(document, seen_document_ids)
                parsed = self._parser.parse_document(document)
                block_writer.write_many(parsed.blocks)
                self._update_parsing(parsing, document, parsed)

                chunked = self._chunker.chunk_document(
                    document,
                    parsed.blocks,
                )
                chunk_writer.write_many(chunked.chunks)
                self._update_chunking(chunking, parsed.blocks, chunked)
                if parsing.document_count % self._progress_interval == 0:
                    self._log_progress(parsing, chunking)

            self._validate_document_count(parsing.document_count, source_manifest)
            block_manifest = self._parser.build_manifest(
                source_manifest=source_manifest,
                block_count=parsing.block_count,
                document_count=parsing.document_count,
                issue_count=parsing.issue_count,
                parsed_count=parsing.parsed_count,
                missing_count=parsing.missing_count,
                structured_count=parsing.structured_count,
                unstructured_count=(
                    parsing.parsed_count - parsing.structured_count
                ),
                source_characters=parsing.source_characters,
                covered_characters=parsing.covered_characters,
            )
            stored_block_manifest = block_writer.finalize(block_manifest)
            chunk_manifest = self._chunker.build_manifest(
                source_manifest=stored_block_manifest,
                chunk_count=chunking.chunk_count,
                document_count=chunking.document_count,
                block_count=chunking.block_count,
                documents_with_chunks=chunking.documents_with_chunks,
                issue_count=chunking.issue_count,
                strategy_counts=chunking.strategy_counts,
            )
            chunk_manifest = self._with_normalized_lineage(
                chunk_manifest,
                normalized_processing_config_hash,
            )
            stored_chunk_manifest = chunk_writer.finalize(chunk_manifest)

        self._log_completed(parsing, chunking)
        return StreamingDocumentProcessingResult(
            block_manifest=stored_block_manifest,
            chunk_manifest=stored_chunk_manifest,
            parser_issue_count=parsing.issue_count,
            chunking_issue_count=chunking.issue_count,
        )

    def chunk_existing_blocks(
        self,
        *,
        documents: Iterable[LegalDocument],
        blocks: Iterable[LegalBlock],
        source_manifest: ArtifactManifest,
        normalized_processing_config_hash: str,
        chunks_destination: Path,
    ) -> StreamingDocumentProcessingResult:
        """Chunk a persisted block stream without materializing the corpus."""
        if source_manifest.artifact_type != ArtifactType.LEGAL_BLOCKS:
            raise ArtifactCompatibilityError(
                "Streaming chunking requires a legal-blocks artifact"
            )
        chunking = _ChunkingCounters()
        seen_document_ids: set[str] = set()
        grouped_blocks = self._blocks_by_document(blocks)
        current_group = next(grouped_blocks, None)
        with ModelArtifactWriter(chunks_destination) as chunk_writer:
            for document in documents:
                self._validate_document_identity(document, seen_document_ids)
                document_blocks: list[LegalBlock] = []
                if current_group is not None:
                    block_document_id, candidate_blocks = current_group
                    if block_document_id == document.document_id:
                        document_blocks = candidate_blocks
                        current_group = next(grouped_blocks, None)
                chunked = self._chunker.chunk_document(
                    document,
                    document_blocks,
                )
                chunk_writer.write_many(chunked.chunks)
                self._update_chunking(chunking, document_blocks, chunked)
                if chunking.document_count % self._progress_interval == 0:
                    self._log_progress(None, chunking)
            if current_group is not None:
                raise DataValidationError(
                    "Legal blocks are not aligned with cleaned documents"
                )
            if chunking.block_count != source_manifest.record_count:
                raise DataValidationError(
                    "Streamed legal-block count differs from manifest"
                )
            chunk_manifest = self._chunker.build_manifest(
                source_manifest=source_manifest,
                chunk_count=chunking.chunk_count,
                document_count=chunking.document_count,
                block_count=chunking.block_count,
                documents_with_chunks=chunking.documents_with_chunks,
                issue_count=chunking.issue_count,
                strategy_counts=chunking.strategy_counts,
            )
            chunk_manifest = self._with_normalized_lineage(
                chunk_manifest,
                normalized_processing_config_hash,
            )
            stored_chunk_manifest = chunk_writer.finalize(chunk_manifest)
        return StreamingDocumentProcessingResult(
            block_manifest=source_manifest,
            chunk_manifest=stored_chunk_manifest,
            parser_issue_count=self._manifest_count(
                source_manifest,
                "parser_issue_count",
            ),
            chunking_issue_count=chunking.issue_count,
        )

    @staticmethod
    def _blocks_by_document(
        blocks: Iterable[LegalBlock],
    ) -> Iterator[tuple[str, list[LegalBlock]]]:
        current_document_id: str | None = None
        current_blocks: list[LegalBlock] = []
        for block in blocks:
            if current_document_id is None:
                current_document_id = block.document_id
            if block.document_id != current_document_id:
                yield current_document_id, current_blocks
                current_document_id = block.document_id
                current_blocks = []
            if block.order_index != len(current_blocks):
                raise DataValidationError(
                    "Legal block order is not contiguous within document"
                )
            current_blocks.append(block)
        if current_document_id is not None:
            yield current_document_id, current_blocks

    @staticmethod
    def _update_parsing(
        counters: _ParsingCounters,
        document: LegalDocument,
        parsed: ParsedLegalDocument,
    ) -> None:
        diagnostic = parsed.diagnostic
        counters.document_count += 1
        counters.block_count += len(parsed.blocks)
        counters.issue_count += len(parsed.issues)
        if document.clean_text is None:
            counters.missing_count += 1
        else:
            counters.parsed_count += 1
        counters.structured_count += int(diagnostic.has_recognized_structure)
        counters.source_characters += diagnostic.source_non_whitespace_characters
        counters.covered_characters += diagnostic.covered_non_whitespace_characters

    @staticmethod
    def _update_chunking(
        counters: _ChunkingCounters,
        blocks: list[LegalBlock],
        chunked: ChunkedLegalDocument,
    ) -> None:
        counters.document_count += 1
        counters.block_count += len(blocks)
        counters.chunk_count += len(chunked.chunks)
        counters.documents_with_chunks += int(chunked.diagnostic.has_chunks)
        counters.issue_count += len(chunked.issues)
        if any(
            issue.severity == AuditSeverity.ERROR for issue in chunked.issues
        ):
            raise DataValidationError("Streaming legal chunks failed validation")
        for chunk in chunked.chunks:
            strategy = chunk.metadata.get("chunk_strategy")
            if not isinstance(strategy, str) or strategy not in counters.strategy_counts:
                raise DataValidationError(
                    "Legal chunk has an unsupported chunk strategy"
                )
            counters.strategy_counts[strategy] += 1

    @staticmethod
    def _validate_cleaned_manifest(manifest: ArtifactManifest) -> None:
        if manifest.artifact_type != ArtifactType.CLEANED_DOCUMENTS:
            raise ArtifactCompatibilityError(
                "Streaming document processing requires cleaned documents"
            )

    @staticmethod
    def _validate_document_identity(
        document: LegalDocument,
        seen_document_ids: set[str],
    ) -> None:
        if document.document_id in seen_document_ids:
            raise DataValidationError(
                "Streaming document input contains duplicate IDs"
            )
        seen_document_ids.add(document.document_id)

    @staticmethod
    def _validate_document_count(
        actual_count: int,
        manifest: ArtifactManifest,
    ) -> None:
        if actual_count != manifest.record_count:
            raise DataValidationError(
                "Streamed document count differs from source manifest"
            )

    @staticmethod
    def _with_normalized_lineage(
        manifest: ArtifactManifest,
        normalized_processing_config_hash: str,
    ) -> ArtifactManifest:
        return manifest.model_copy(
            update={
                "metadata": {
                    **manifest.metadata,
                    "runtime_normalized_processing_config_hash": (
                        normalized_processing_config_hash
                    ),
                }
            }
        )

    @staticmethod
    def _manifest_count(manifest: ArtifactManifest, field_name: str) -> int:
        value = manifest.metadata.get(field_name, 0)
        return value if isinstance(value, int) else 0

    @staticmethod
    def _log_progress(
        parsing: _ParsingCounters | None,
        chunking: _ChunkingCounters,
    ) -> None:
        _LOGGER.info(
            "streaming_document_processing_progress",
            extra={
                "document_count": chunking.document_count,
                "block_count": (
                    parsing.block_count if parsing is not None else chunking.block_count
                ),
                "chunk_count": chunking.chunk_count,
            },
        )

    @staticmethod
    def _log_completed(
        parsing: _ParsingCounters,
        chunking: _ChunkingCounters,
    ) -> None:
        _LOGGER.info(
            "streaming_document_processing_completed",
            extra={
                "document_count": parsing.document_count,
                "block_count": parsing.block_count,
                "chunk_count": chunking.chunk_count,
                "parser_issue_count": parsing.issue_count,
                "chunking_issue_count": chunking.issue_count,
            },
        )
