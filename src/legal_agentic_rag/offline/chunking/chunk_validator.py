"""Validation of legal chunks against their document and block sources."""

from collections import Counter
from collections.abc import Iterable

from legal_agentic_rag.configuration.offline import ChunkingConfig
from legal_agentic_rag.offline.chunking.tokenizer import UnicodeWordTokenizer
from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity
from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalChunk,
    LegalDocument,
)


class LegalChunkValidator:
    """Report chunk integrity and metadata inheritance issues without mutation."""

    def __init__(
        self,
        config: ChunkingConfig,
        *,
        tokenizer: UnicodeWordTokenizer | None = None,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer or UnicodeWordTokenizer()

    def validate(
        self,
        *,
        documents: Iterable[LegalDocument],
        blocks: Iterable[LegalBlock],
        chunks: Iterable[LegalChunk],
    ) -> list[AuditIssue]:
        """Validate IDs, limits, source coverage, and inherited metadata."""
        document_list = list(documents)
        block_list = list(blocks)
        chunk_list = list(chunks)
        documents_by_id = {
            document.document_id: document for document in document_list
        }
        blocks_by_id = {block.block_id: block for block in block_list}
        issues: list[AuditIssue] = []
        chunk_id_counts = Counter(chunk.chunk_id for chunk in chunk_list)
        covered_block_ids: set[str] = set()
        chunks_by_document: dict[str, list[LegalChunk]] = {}
        token_fallback_groups: dict[tuple[str, ...], list[LegalChunk]] = {}

        for chunk in chunk_list:
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
            if chunk_id_counts[chunk.chunk_id] > 1:
                issues.append(
                    self._issue(
                        "duplicate_chunk_id",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Chunk ID is not unique",
                    )
                )
            document = documents_by_id.get(chunk.document_id)
            if document is None:
                issues.append(
                    self._issue(
                        "unknown_chunk_document",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Chunk references an unknown document",
                    )
                )
                continue
            actual_token_count = self._tokenizer.count(chunk.text)
            if actual_token_count != chunk.token_count:
                issues.append(
                    self._issue(
                        "token_count_mismatch",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Stored token count does not match configured tokenizer",
                        metadata={"actual_token_count": actual_token_count},
                    )
                )
            if chunk.token_count > self._config.max_tokens:
                issues.append(
                    self._issue(
                        "chunk_too_long",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Chunk exceeds configured maximum token count",
                        metadata={"token_count": chunk.token_count},
                    )
                )
            if chunk.token_count < self._config.min_tokens:
                issues.append(
                    self._issue(
                        "chunk_below_minimum",
                        AuditSeverity.INFO,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Chunk is shorter than the preferred minimum token count",
                        metadata={"token_count": chunk.token_count},
                    )
                )
            search_token_count = self._tokenizer.count(chunk.search_text)
            if search_token_count > self._config.max_search_tokens:
                issues.append(
                    self._issue(
                        "search_text_too_long",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Search text exceeds the configured embedding-input budget",
                        metadata={"search_text_token_count": search_token_count},
                    )
                )
            if chunk.metadata.get("search_text_token_count") != search_token_count:
                issues.append(
                    self._issue(
                        "search_text_token_count_mismatch",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Stored search-text token count is missing or incorrect",
                        metadata={"actual_search_text_token_count": search_token_count},
                    )
                )
            if not chunk.search_text.endswith(chunk.text):
                issues.append(
                    self._issue(
                        "chunk_text_missing_from_search_text",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Search text does not preserve the complete chunk text",
                    )
                )
            self._validate_metadata_inheritance(chunk, document, issues)
            source_ids = chunk.metadata.get("source_block_ids")
            if not isinstance(source_ids, list):
                issues.append(
                    self._issue(
                        "missing_source_blocks",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Chunk does not list its source blocks",
                    )
                )
                continue
            valid_source_blocks: list[LegalBlock] = []
            for source_id in source_ids:
                if not isinstance(source_id, str):
                    continue
                block = blocks_by_id.get(source_id)
                if block is None:
                    issues.append(
                        self._issue(
                            "unknown_source_block",
                            AuditSeverity.ERROR,
                            chunk.document_id,
                            chunk.chunk_id,
                            "Chunk references an unknown source block",
                        )
                    )
                elif block.document_id != chunk.document_id:
                    issues.append(
                        self._issue(
                            "source_document_mismatch",
                            AuditSeverity.ERROR,
                            chunk.document_id,
                            chunk.chunk_id,
                            "Chunk and source block belong to different documents",
                        )
                    )
                else:
                    covered_block_ids.add(source_id)
                    valid_source_blocks.append(block)
            if len(valid_source_blocks) != len(source_ids):
                continue
            strategy = chunk.metadata.get("chunk_strategy")
            if strategy == "token_fallback":
                token_fallback_groups.setdefault(tuple(source_ids), []).append(chunk)
            elif chunk.text != self._join_blocks(valid_source_blocks):
                issues.append(
                    self._issue(
                        "chunk_text_mismatch",
                        AuditSeverity.ERROR,
                        chunk.document_id,
                        chunk.chunk_id,
                        "Chunk text does not equal its non-split source block text",
                    )
                )

        for source_ids, fallback_chunks in token_fallback_groups.items():
            source_blocks = [blocks_by_id[source_id] for source_id in source_ids]
            expected_fragments = self._tokenizer.split(
                self._join_blocks(source_blocks),
                max_tokens=self._config.max_tokens,
                overlap_tokens=self._config.overlap_tokens,
            )
            ordered_chunks = sorted(
                fallback_chunks,
                key=lambda chunk: self._split_index(chunk),
            )
            actual_fragments = [chunk.text for chunk in ordered_chunks]
            actual_indexes = [self._split_index(chunk) for chunk in ordered_chunks]
            split_counts = {
                chunk.metadata.get("split_count") for chunk in ordered_chunks
            }
            if (
                actual_fragments != expected_fragments
                or actual_indexes != list(range(len(expected_fragments)))
                or split_counts != {len(expected_fragments)}
            ):
                issues.append(
                    self._issue(
                        "token_fallback_mismatch",
                        AuditSeverity.ERROR,
                        source_blocks[0].document_id,
                        None,
                        "Token fallback chunks do not match configured windows",
                        metadata={"source_block_ids": list(source_ids)},
                    )
                )

        for document_id, document_chunks in chunks_by_document.items():
            indexes = [chunk.chunk_index for chunk in document_chunks]
            if indexes != list(range(len(document_chunks))):
                issues.append(
                    self._issue(
                        "non_contiguous_chunk_indexes",
                        AuditSeverity.ERROR,
                        document_id,
                        None,
                        "Chunk indexes are not contiguous in document order",
                    )
                )
        for block in block_list:
            if block.block_id not in covered_block_ids:
                issues.append(
                    self._issue(
                        "uncovered_source_block",
                        AuditSeverity.ERROR,
                        block.document_id,
                        None,
                        "Source block is not represented by any chunk",
                        metadata={"source_block_id": block.block_id},
                    )
                )
        return issues

    @staticmethod
    def _join_blocks(blocks: list[LegalBlock]) -> str:
        return "\n".join(block.text for block in blocks)

    @staticmethod
    def _split_index(chunk: LegalChunk) -> int:
        value = chunk.metadata.get("split_index")
        return value if isinstance(value, int) else -1

    @staticmethod
    def _validate_metadata_inheritance(
        chunk: LegalChunk,
        document: LegalDocument,
        issues: list[AuditIssue],
    ) -> None:
        expected_values = {
            "document_title": document.title,
            "document_number": document.document_number,
            "document_type": document.document_type,
            "issuance_date": document.issuance_date,
            "effective_date": document.effective_date,
            "expiry_date": document.expiry_date,
            "effect_status": document.effect_status,
            "issuing_authority": document.issuing_authority,
            "legal_field": document.legal_field,
            "source_url": document.source_url,
            "source_dataset": document.source_dataset,
        }
        mismatched_fields = [
            field_name
            for field_name, expected in expected_values.items()
            if getattr(chunk, field_name) != expected
        ]
        if mismatched_fields:
            issues.append(
                LegalChunkValidator._issue(
                    "metadata_inheritance_mismatch",
                    AuditSeverity.ERROR,
                    chunk.document_id,
                    chunk.chunk_id,
                    "Chunk metadata differs from its source document",
                    metadata={"fields": mismatched_fields},
                )
            )

    @staticmethod
    def _issue(
        issue_type: str,
        severity: AuditSeverity,
        document_id: str,
        chunk_id: str | None,
        message: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> AuditIssue:
        issue_metadata: dict[str, object] = {"stage": "legal_chunking"}
        if chunk_id is not None:
            issue_metadata["chunk_id"] = chunk_id
        issue_metadata.update(metadata or {})
        return AuditIssue(
            issue_type=issue_type,
            severity=severity,
            record_id=document_id,
            message=message,
            metadata=issue_metadata,
        )
