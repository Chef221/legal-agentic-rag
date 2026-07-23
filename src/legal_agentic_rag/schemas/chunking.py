"""Contracts emitted by deterministic legal chunking and validation."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_agentic_rag.schemas.auditing import AuditIssue
from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalBlockType,
    LegalChunk,
    LegalDocument,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

_CHUNK_STRATEGIES = {
    "article",
    "clause_group",
    "token_fallback",
    "standalone_block",
}


class DocumentChunkingDiagnostic(BaseModel):
    """Block coverage and fallback counts for one chunked document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    source_block_count: int = Field(ge=0)
    covered_block_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    article_unit_count: int = Field(ge=0)
    token_fallback_chunk_count: int = Field(ge=0)
    block_coverage: float = Field(ge=0.0, le=1.0)
    has_chunks: bool

    @model_validator(mode="after")
    def validate_coverage(self) -> "DocumentChunkingDiagnostic":
        """Keep coverage, chunk presence, and source counts consistent."""
        if self.covered_block_count > self.source_block_count:
            raise ValueError("covered blocks cannot exceed source blocks")
        expected_coverage = (
            self.covered_block_count / self.source_block_count
            if self.source_block_count
            else 0.0
        )
        if abs(self.block_coverage - expected_coverage) > 1e-9:
            raise ValueError("block_coverage must match block counts")
        if self.has_chunks != bool(self.chunk_count):
            raise ValueError("has_chunks must match chunk_count")
        if self.token_fallback_chunk_count > self.chunk_count:
            raise ValueError("token fallback count cannot exceed chunk count")
        return self


class LegalChunkingResult(BaseModel):
    """Legal chunks, validation issues, diagnostics, and artifact provenance."""

    model_config = ConfigDict(extra="forbid")

    documents: list[LegalDocument] = Field(default_factory=list)
    blocks: list[LegalBlock] = Field(default_factory=list)
    chunks: list[LegalChunk] = Field(default_factory=list)
    diagnostics: list[DocumentChunkingDiagnostic] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    manifest: ArtifactManifest
    input_document_count: int = Field(ge=0)
    input_block_count: int = Field(ge=0)
    documents_with_chunks_count: int = Field(ge=0)
    documents_without_chunks_count: int = Field(ge=0)
    article_chunk_count: int = Field(ge=0)
    clause_fallback_chunk_count: int = Field(ge=0)
    token_fallback_chunk_count: int = Field(ge=0)
    standalone_chunk_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "LegalChunkingResult":
        """Validate chunk identity, block coverage, strategies, and counts."""
        if self.manifest.artifact_type != ArtifactType.LEGAL_CHUNKS:
            raise ValueError("manifest must describe legal chunks")
        if self.manifest.record_count != len(self.chunks):
            raise ValueError("manifest record_count must equal chunk count")
        if self.input_document_count != len(self.documents):
            raise ValueError("input_document_count must equal document count")
        if self.input_block_count != len(self.blocks):
            raise ValueError("input_block_count must equal block count")
        if len(self.diagnostics) != self.input_document_count:
            raise ValueError("every input document must have one diagnostic")
        if (
            self.documents_with_chunks_count + self.documents_without_chunks_count
            != self.input_document_count
        ):
            raise ValueError("every document must be classified by chunk presence")
        strategy_total = (
            self.article_chunk_count
            + self.clause_fallback_chunk_count
            + self.token_fallback_chunk_count
            + self.standalone_chunk_count
        )
        if strategy_total != len(self.chunks):
            raise ValueError("strategy counts must equal chunk count")

        document_ids = [document.document_id for document in self.documents]
        diagnostic_ids = [item.document_id for item in self.diagnostics]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("input document IDs must be unique")
        if diagnostic_ids != document_ids:
            raise ValueError("diagnostics must follow input document order")
        known_documents = set(document_ids)

        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("source block IDs must be unique")
        block_by_id = {block.block_id: block for block in self.blocks}
        if any(block.document_id not in known_documents for block in self.blocks):
            raise ValueError("source block references an unknown document")

        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("legal chunk IDs must be unique")
        chunks_by_document: dict[str, list[LegalChunk]] = {}
        covered_ids: set[str] = set()
        actual_strategy_counts = {strategy: 0 for strategy in _CHUNK_STRATEGIES}
        for chunk in self.chunks:
            if chunk.document_id not in known_documents:
                raise ValueError("legal chunk references an unknown document")
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
            strategy = chunk.metadata.get("chunk_strategy")
            if not isinstance(strategy, str) or strategy not in _CHUNK_STRATEGIES:
                raise ValueError("legal chunk must have a supported chunk_strategy")
            actual_strategy_counts[strategy] += 1
            source_ids = chunk.metadata.get("source_block_ids")
            if (
                not isinstance(source_ids, list)
                or not source_ids
                or any(not isinstance(block_id, str) for block_id in source_ids)
            ):
                raise ValueError("legal chunk must list source_block_ids")
            for block_id in source_ids:
                block = block_by_id.get(block_id)
                if block is None:
                    raise ValueError("legal chunk references an unknown source block")
                if block.document_id != chunk.document_id:
                    raise ValueError("chunk and source blocks must share document ID")
                covered_ids.add(block_id)
        if covered_ids != set(block_ids):
            raise ValueError("every source block must be covered by a legal chunk")
        expected_strategy_counts = {
            "article": self.article_chunk_count,
            "clause_group": self.clause_fallback_chunk_count,
            "token_fallback": self.token_fallback_chunk_count,
            "standalone_block": self.standalone_chunk_count,
        }
        if actual_strategy_counts != expected_strategy_counts:
            raise ValueError("strategy fields must match result counts")

        diagnostics_by_id = {item.document_id: item for item in self.diagnostics}
        for document_id in document_ids:
            document_chunks = chunks_by_document.get(document_id, [])
            indexes = [chunk.chunk_index for chunk in document_chunks]
            if indexes != list(range(len(document_chunks))):
                raise ValueError("chunk indexes must be contiguous per document")
            diagnostic = diagnostics_by_id[document_id]
            document_block_ids = {
                block.block_id
                for block in self.blocks
                if block.document_id == document_id
            }
            document_covered_ids = covered_ids & document_block_ids
            if diagnostic.source_block_count != len(document_block_ids):
                raise ValueError("diagnostic source block count must match blocks")
            if diagnostic.covered_block_count != len(document_covered_ids):
                raise ValueError("diagnostic covered count must match chunks")
            if diagnostic.chunk_count != len(document_chunks):
                raise ValueError("diagnostic chunk count must match chunks")
            actual_token_fallback = sum(
                chunk.metadata.get("chunk_strategy") == "token_fallback"
                for chunk in document_chunks
            )
            if diagnostic.token_fallback_chunk_count != actual_token_fallback:
                raise ValueError("diagnostic token fallback count must match chunks")
            actual_article_units = sum(
                block.block_type == LegalBlockType.ARTICLE
                for block in self.blocks
                if block.document_id == document_id
            )
            if diagnostic.article_unit_count != actual_article_units:
                raise ValueError("diagnostic article unit count must match blocks")
        actual_documents_with_chunks = sum(
            bool(chunks_by_document.get(document_id)) for document_id in document_ids
        )
        if actual_documents_with_chunks != self.documents_with_chunks_count:
            raise ValueError("documents_with_chunks_count must match chunks")
        return self
