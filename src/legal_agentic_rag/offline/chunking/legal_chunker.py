"""Article-first legal chunking with clause and token fallbacks."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import logging

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import ChunkingConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    ConfigurationError,
    DataValidationError,
)
from legal_agentic_rag.offline.chunking.chunk_validator import (
    LegalChunkValidator,
)
from legal_agentic_rag.offline.chunking.tokenizer import UnicodeWordTokenizer
from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity
from legal_agentic_rag.schemas.chunking import (
    DocumentChunkingDiagnostic,
    LegalChunkingResult,
)
from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalBlockType,
    LegalChunk,
    LegalDocument,
    LegalStructure,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

Clock = Callable[[], datetime]
_LOGGER = logging.getLogger(__name__)


@dataclass
class _ChunkDraft:
    source_blocks: list[LegalBlock]
    text: str
    strategy: str
    split_index: int = 0
    split_count: int = 1


@dataclass
class ChunkedLegalDocument:
    """Memory-bounded chunker output for one parsed legal document."""

    chunks: list[LegalChunk]
    diagnostic: DocumentChunkingDiagnostic
    issues: list[AuditIssue]


class LegalChunker:
    """Build retrieval chunks while preserving legal boundaries and provenance."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        tokenizer: UnicodeWordTokenizer | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or ChunkingConfig()
        self._tokenizer = tokenizer or UnicodeWordTokenizer()
        if self._tokenizer.name != self._config.tokenizer_name:
            raise ConfigurationError(
                "Configured tokenizer name does not match chunker tokenizer"
            )
        self._validator = LegalChunkValidator(
            self._config,
            tokenizer=self._tokenizer,
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    def chunk(
        self,
        *,
        documents: Iterable[LegalDocument],
        blocks: Iterable[LegalBlock],
        source_manifest: ArtifactManifest,
    ) -> LegalChunkingResult:
        """Create and validate chunks for a complete legal-block artifact."""
        document_list = list(documents)
        block_list = list(blocks)
        self._validate_input(document_list, block_list, source_manifest)
        _LOGGER.info(
            "legal_chunking_started",
            extra={
                "dataset_name": source_manifest.dataset_name,
                "document_count": len(document_list),
                "block_count": len(block_list),
            },
        )
        blocks_by_document: dict[str, list[LegalBlock]] = {
            document.document_id: [] for document in document_list
        }
        for block in block_list:
            blocks_by_document[block.document_id].append(block)
        for document_blocks in blocks_by_document.values():
            document_blocks.sort(key=lambda block: block.order_index)

        chunks: list[LegalChunk] = []
        diagnostics: list[DocumentChunkingDiagnostic] = []
        issues: list[AuditIssue] = []
        for document in document_list:
            document_blocks = blocks_by_document[document.document_id]
            chunked = self.chunk_document(document, document_blocks)
            chunks.extend(chunked.chunks)
            diagnostics.append(chunked.diagnostic)
            issues.extend(chunked.issues)
        strategy_counts = self._strategy_counts(chunks)
        documents_with_chunks = sum(item.has_chunks for item in diagnostics)
        manifest = self.build_manifest(
            source_manifest=source_manifest,
            chunk_count=len(chunks),
            document_count=len(document_list),
            block_count=len(block_list),
            documents_with_chunks=documents_with_chunks,
            issue_count=len(issues),
            strategy_counts=strategy_counts,
        )
        result = LegalChunkingResult(
            documents=document_list,
            blocks=block_list,
            chunks=chunks,
            diagnostics=diagnostics,
            issues=issues,
            manifest=manifest,
            input_document_count=len(document_list),
            input_block_count=len(block_list),
            documents_with_chunks_count=documents_with_chunks,
            documents_without_chunks_count=len(document_list)
            - documents_with_chunks,
            article_chunk_count=strategy_counts["article"],
            clause_fallback_chunk_count=strategy_counts["clause_group"],
            token_fallback_chunk_count=strategy_counts["token_fallback"],
            standalone_chunk_count=strategy_counts["standalone_block"],
        )
        _LOGGER.info(
            "legal_chunking_completed",
            extra={
                "dataset_name": source_manifest.dataset_name,
                "document_count": len(document_list),
                "chunk_count": len(chunks),
                "token_fallback_chunk_count": strategy_counts["token_fallback"],
                "issue_count": len(issues),
            },
        )
        return result

    def chunk_document(
        self,
        document: LegalDocument,
        blocks: list[LegalBlock],
    ) -> ChunkedLegalDocument:
        """Create and validate chunks for one document in bounded memory."""
        document_chunks = self._chunk_document(document, blocks)
        covered_ids = {
            source_id
            for chunk in document_chunks
            for source_id in self._source_block_ids(chunk)
        }
        diagnostic = DocumentChunkingDiagnostic(
            document_id=document.document_id,
            source_block_count=len(blocks),
            covered_block_count=len(covered_ids),
            chunk_count=len(document_chunks),
            article_unit_count=sum(
                block.block_type == LegalBlockType.ARTICLE for block in blocks
            ),
            token_fallback_chunk_count=sum(
                chunk.metadata.get("chunk_strategy") == "token_fallback"
                for chunk in document_chunks
            ),
            block_coverage=(len(covered_ids) / len(blocks) if blocks else 0.0),
            has_chunks=bool(document_chunks),
        )
        issues: list[AuditIssue] = []
        if not blocks:
            issues.append(
                self._issue(
                    "missing_legal_blocks",
                    AuditSeverity.WARNING,
                    document.document_id,
                    "Document has no parsed legal blocks to chunk",
                )
            )
        validation_issues = self._validator.validate(
            documents=[document],
            blocks=blocks,
            chunks=document_chunks,
        )
        if any(
            issue.severity == AuditSeverity.ERROR
            for issue in validation_issues
        ):
            raise DataValidationError("Generated legal chunks failed validation")
        issues.extend(validation_issues)
        return ChunkedLegalDocument(
            chunks=document_chunks,
            diagnostic=diagnostic,
            issues=issues,
        )

    def _chunk_document(
        self,
        document: LegalDocument,
        blocks: list[LegalBlock],
    ) -> list[LegalChunk]:
        children: dict[str, list[LegalBlock]] = {
            block.block_id: [] for block in blocks
        }
        for block in blocks:
            if block.parent_block_id is not None:
                children[block.parent_block_id].append(block)
        consumed: set[str] = set()
        drafts: list[_ChunkDraft] = []
        for block in blocks:
            if block.block_id in consumed:
                continue
            if block.block_type == LegalBlockType.ARTICLE:
                article_blocks = self._subtree(block, children)
                consumed.update(item.block_id for item in article_blocks)
                drafts.extend(self._article_drafts(block, article_blocks, children))
            else:
                consumed.add(block.block_id)
                drafts.extend(self._standalone_drafts([block]))
        return [
            self._legal_chunk(document, draft, chunk_index)
            for chunk_index, draft in enumerate(drafts)
        ]

    def _article_drafts(
        self,
        article: LegalBlock,
        article_blocks: list[LegalBlock],
        children: dict[str, list[LegalBlock]],
    ) -> list[_ChunkDraft]:
        article_text = self._join_blocks(article_blocks)
        if self._tokenizer.count(article_text) <= self._config.max_tokens:
            return [_ChunkDraft(article_blocks, article_text, "article")]

        units: list[list[LegalBlock]] = [[article]]
        for child in sorted(
            children[article.block_id], key=lambda block: block.order_index
        ):
            units.append(self._subtree(child, children))
        if len(units) > 1:
            units = [units[0] + units[1], *units[2:]]

        drafts: list[_ChunkDraft] = []
        pending: list[LegalBlock] = []
        for unit in units:
            unit_text = self._join_blocks(unit)
            unit_tokens = self._tokenizer.count(unit_text)
            if unit_tokens > self._config.max_tokens:
                if pending:
                    drafts.append(
                        _ChunkDraft(
                            pending,
                            self._join_blocks(pending),
                            "clause_group",
                        )
                    )
                    pending = []
                drafts.extend(self._token_fallback_drafts(unit))
                continue
            combined = pending + unit
            if (
                pending
                and self._tokenizer.count(self._join_blocks(combined))
                > self._config.max_tokens
            ):
                drafts.append(
                    _ChunkDraft(
                        pending,
                        self._join_blocks(pending),
                        "clause_group",
                    )
                )
                pending = list(unit)
            else:
                pending = combined
        if pending:
            drafts.append(
                _ChunkDraft(
                    pending,
                    self._join_blocks(pending),
                    "clause_group",
                )
            )
        return drafts

    def _standalone_drafts(
        self, source_blocks: list[LegalBlock]
    ) -> list[_ChunkDraft]:
        text = self._join_blocks(source_blocks)
        if self._tokenizer.count(text) <= self._config.max_tokens:
            return [_ChunkDraft(source_blocks, text, "standalone_block")]
        return self._token_fallback_drafts(source_blocks)

    def _token_fallback_drafts(
        self, source_blocks: list[LegalBlock]
    ) -> list[_ChunkDraft]:
        fragments = self._tokenizer.split(
            self._join_blocks(source_blocks),
            max_tokens=self._config.max_tokens,
            overlap_tokens=self._config.overlap_tokens,
        )
        return [
            _ChunkDraft(
                source_blocks=source_blocks,
                text=fragment,
                strategy="token_fallback",
                split_index=split_index,
                split_count=len(fragments),
            )
            for split_index, fragment in enumerate(fragments)
        ]

    def _legal_chunk(
        self,
        document: LegalDocument,
        draft: _ChunkDraft,
        chunk_index: int,
    ) -> LegalChunk:
        structure = self._aggregate_structure(draft.source_blocks)
        source_ids = [block.block_id for block in draft.source_blocks]
        chunk_id = self._chunk_id(
            document.document_id,
            draft.strategy,
            source_ids,
            draft.split_index,
            draft.text,
        )
        return LegalChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            chunk_index=chunk_index,
            text=draft.text,
            search_text=self._search_text(document, structure, draft.text),
            token_count=self._tokenizer.count(draft.text),
            structure=structure,
            document_title=document.title,
            document_number=document.document_number,
            document_type=document.document_type,
            issuance_date=document.issuance_date,
            effective_date=document.effective_date,
            expiry_date=document.expiry_date,
            effect_status=document.effect_status,
            issuing_authority=document.issuing_authority,
            legal_field=document.legal_field,
            source_url=document.source_url,
            source_dataset=document.source_dataset,
            metadata={
                "source_block_ids": source_ids,
                "source_block_types": [
                    block.block_type.value for block in draft.source_blocks
                ],
                "chunk_strategy": draft.strategy,
                "tokenizer_name": self._config.tokenizer_name,
                "split_index": draft.split_index,
                "split_count": draft.split_count,
            },
        )

    @staticmethod
    def _aggregate_structure(blocks: list[LegalBlock]) -> LegalStructure:
        base = next(
            (
                block.structure
                for block in blocks
                if block.structure.article_number is not None
            ),
            blocks[0].structure,
        )
        clause_numbers: list[str] = []
        point_numbers: list[str] = []
        for block in blocks:
            for number in block.structure.clause_numbers:
                if number not in clause_numbers:
                    clause_numbers.append(number)
            for number in block.structure.point_numbers:
                if number not in point_numbers:
                    point_numbers.append(number)
        path = [
            item
            for item in base.structure_path
            if not item.casefold().startswith(("khoản ", "điểm "))
        ]
        if len(clause_numbers) == 1:
            path.append(f"Khoản {clause_numbers[0]}")
        if len(point_numbers) == 1:
            path.append(f"Điểm {point_numbers[0]}")
        payload = base.model_dump(mode="python")
        payload.update(
            {
                "clause_numbers": clause_numbers,
                "point_numbers": point_numbers,
                "structure_path": path,
            }
        )
        return LegalStructure.model_validate(payload)

    @staticmethod
    def _search_text(
        document: LegalDocument,
        structure: LegalStructure,
        text: str,
    ) -> str:
        lines: list[str] = []
        if document.title is not None:
            lines.append(f"Văn bản: {document.title}")
        if document.document_number is not None:
            lines.append(f"Số ký hiệu: {document.document_number}")
        if document.document_type is not None:
            lines.append(f"Loại văn bản: {document.document_type}")
        for value in (
            structure.part,
            structure.chapter,
            structure.section,
            structure.subsection,
        ):
            if value is not None:
                lines.append(value)
        if structure.article_number is not None:
            article = f"Điều {structure.article_number}"
            if structure.article_title is not None:
                article = f"{article}: {structure.article_title}"
            lines.append(article)
        if structure.clause_numbers:
            lines.append(f"Khoản: {', '.join(structure.clause_numbers)}")
        if structure.point_numbers:
            lines.append(f"Điểm: {', '.join(structure.point_numbers)}")
        lines.extend(["Nội dung:", text])
        return "\n".join(lines)

    @staticmethod
    def _subtree(
        root: LegalBlock,
        children: dict[str, list[LegalBlock]],
    ) -> list[LegalBlock]:
        result = [root]
        for child in sorted(
            children.get(root.block_id, []), key=lambda block: block.order_index
        ):
            result.extend(LegalChunker._subtree(child, children))
        return result

    @staticmethod
    def _join_blocks(blocks: list[LegalBlock]) -> str:
        return "\n".join(block.text for block in blocks)

    @staticmethod
    def _source_block_ids(chunk: LegalChunk) -> list[str]:
        value = chunk.metadata.get("source_block_ids")
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _chunk_id(
        document_id: str,
        strategy: str,
        source_block_ids: list[str],
        split_index: int,
        text: str,
    ) -> str:
        payload = "\0".join(
            [
                document_id,
                strategy,
                *source_block_ids,
                str(split_index),
                text,
            ]
        )
        digest = sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"chunk_{digest}"

    @staticmethod
    def _strategy_counts(chunks: list[LegalChunk]) -> dict[str, int]:
        counts = {
            "article": 0,
            "clause_group": 0,
            "token_fallback": 0,
            "standalone_block": 0,
        }
        for chunk in chunks:
            strategy = chunk.metadata.get("chunk_strategy")
            if isinstance(strategy, str) and strategy in counts:
                counts[strategy] += 1
        return counts

    @staticmethod
    def _validate_input(
        documents: list[LegalDocument],
        blocks: list[LegalBlock],
        source_manifest: ArtifactManifest,
    ) -> None:
        if source_manifest.artifact_type != ArtifactType.LEGAL_BLOCKS:
            raise ArtifactCompatibilityError(
                "Legal chunker requires a legal-blocks artifact"
            )
        if source_manifest.record_count != len(blocks):
            raise DataValidationError(
                "Source manifest record count does not match input blocks"
            )
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise DataValidationError("Legal chunker document IDs must be unique")
        block_ids = [block.block_id for block in blocks]
        if len(block_ids) != len(set(block_ids)):
            raise DataValidationError("Legal chunker block IDs must be unique")
        known_documents = set(document_ids)
        block_by_id: dict[str, LegalBlock] = {}
        indexes_by_document: dict[str, list[int]] = {
            document_id: [] for document_id in document_ids
        }
        for block in blocks:
            if block.document_id not in known_documents:
                raise DataValidationError("Legal block references an unknown document")
            indexes_by_document[block.document_id].append(block.order_index)
            if block.parent_block_id is not None:
                parent = block_by_id.get(block.parent_block_id)
                if parent is None or parent.document_id != block.document_id:
                    raise DataValidationError(
                        "Legal block parent must exist earlier in the same document"
                    )
            block_by_id[block.block_id] = block
        for indexes in indexes_by_document.values():
            if sorted(indexes) != list(range(len(indexes))):
                raise DataValidationError(
                    "Legal block order indexes must be contiguous per document"
                )

    def build_manifest(
        self,
        *,
        source_manifest: ArtifactManifest,
        chunk_count: int,
        document_count: int,
        block_count: int,
        documents_with_chunks: int,
        issue_count: int,
        strategy_counts: dict[str, int],
    ) -> ArtifactManifest:
        """Build aggregate legal-chunk provenance from streaming counters."""
        warnings = list(source_manifest.warnings)
        if issue_count:
            warnings.append(f"Legal chunking produced {issue_count} issues")
        return ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.LEGAL_CHUNKS,
            artifact_version=self._config.artifact_version,
            dataset_name=source_manifest.dataset_name,
            dataset_revision=source_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=chunk_count,
            processing_config_hash=self._config_hash(source_manifest),
            code_version=__version__,
            warnings=warnings,
            metadata={
                "source_artifact_type": source_manifest.artifact_type.value,
                "source_artifact_version": source_manifest.artifact_version,
                "source_processing_config_hash": source_manifest.processing_config_hash,
                "input_document_count": document_count,
                "input_block_count": block_count,
                "documents_with_chunks_count": documents_with_chunks,
                "documents_without_chunks_count": document_count
                - documents_with_chunks,
                "article_chunk_count": strategy_counts["article"],
                "clause_fallback_chunk_count": strategy_counts["clause_group"],
                "token_fallback_chunk_count": strategy_counts["token_fallback"],
                "standalone_chunk_count": strategy_counts["standalone_block"],
                "chunking_issue_count": issue_count,
                "tokenizer_name": self._config.tokenizer_name,
            },
        )

    def _config_hash(self, source_manifest: ArtifactManifest) -> str:
        payload = {
            "source_processing_config_hash": source_manifest.processing_config_hash,
            "chunking": self._config,
        }
        return canonical_sha256(payload)

    @staticmethod
    def _issue(
        issue_type: str,
        severity: AuditSeverity,
        document_id: str,
        message: str,
    ) -> AuditIssue:
        return AuditIssue(
            issue_type=issue_type,
            severity=severity,
            record_id=document_id,
            message=message,
            metadata={"stage": "legal_chunking"},
        )
