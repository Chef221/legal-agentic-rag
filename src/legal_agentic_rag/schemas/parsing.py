"""Contracts emitted by deterministic legal structure parsing."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_agentic_rag.schemas.auditing import AuditIssue
from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalBlockType,
    LegalDocument,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType


class DocumentParsingDiagnostic(BaseModel):
    """Coverage and structure findings for one parsed legal document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    block_count: int = Field(ge=0)
    recognized_structure_count: int = Field(ge=0)
    source_non_whitespace_characters: int = Field(ge=0)
    covered_non_whitespace_characters: int = Field(ge=0)
    text_coverage: float = Field(ge=0.0, le=1.0)
    has_recognized_structure: bool

    @model_validator(mode="after")
    def validate_coverage(self) -> "DocumentParsingDiagnostic":
        """Keep coverage ratio and recognized-structure flag reproducible."""
        if (
            self.covered_non_whitespace_characters
            > self.source_non_whitespace_characters
        ):
            raise ValueError("covered characters cannot exceed source characters")
        expected_coverage = (
            self.covered_non_whitespace_characters
            / self.source_non_whitespace_characters
            if self.source_non_whitespace_characters
            else 0.0
        )
        if abs(self.text_coverage - expected_coverage) > 1e-9:
            raise ValueError("text_coverage must match character counts")
        if self.has_recognized_structure != bool(
            self.recognized_structure_count
        ):
            raise ValueError(
                "has_recognized_structure must match recognized structure count"
            )
        if self.recognized_structure_count > self.block_count:
            raise ValueError("recognized structure count cannot exceed block count")
        return self


class LegalStructureParsingResult(BaseModel):
    """Parsed legal blocks, per-document diagnostics, issues, and provenance."""

    model_config = ConfigDict(extra="forbid")

    documents: list[LegalDocument] = Field(default_factory=list)
    blocks: list[LegalBlock] = Field(default_factory=list)
    diagnostics: list[DocumentParsingDiagnostic] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    manifest: ArtifactManifest
    input_document_count: int = Field(ge=0)
    parsed_document_count: int = Field(ge=0)
    missing_clean_text_count: int = Field(ge=0)
    structured_document_count: int = Field(ge=0)
    unstructured_document_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "LegalStructureParsingResult":
        """Validate block identity, hierarchy, counts, and document coverage."""
        if self.manifest.artifact_type != ArtifactType.LEGAL_BLOCKS:
            raise ValueError("manifest must describe legal blocks")
        if self.manifest.record_count != len(self.blocks):
            raise ValueError("manifest record_count must equal block count")
        if self.input_document_count != len(self.documents):
            raise ValueError("input_document_count must equal document count")
        if len(self.diagnostics) != self.input_document_count:
            raise ValueError("every input document must have one diagnostic")
        if (
            self.parsed_document_count + self.missing_clean_text_count
            != self.input_document_count
        ):
            raise ValueError("every document must be parsed or missing clean text")
        if (
            self.structured_document_count + self.unstructured_document_count
            != self.parsed_document_count
        ):
            raise ValueError("every parsed document must be classified")

        document_ids = [document.document_id for document in self.documents]
        diagnostic_ids = [item.document_id for item in self.diagnostics]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("input document IDs must be unique")
        if diagnostic_ids != document_ids:
            raise ValueError("diagnostics must follow input document order")
        if sum(item.has_recognized_structure for item in self.diagnostics) != (
            self.structured_document_count
        ):
            raise ValueError("structured document count must match diagnostics")

        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("legal block IDs must be unique")
        block_by_id = {block.block_id: block for block in self.blocks}
        known_documents = set(document_ids)
        blocks_by_document: dict[str, list[LegalBlock]] = {}
        for block in self.blocks:
            if block.document_id not in known_documents:
                raise ValueError("legal block references an unknown document")
            blocks_by_document.setdefault(block.document_id, []).append(block)
            if block.parent_block_id is not None:
                parent = block_by_id.get(block.parent_block_id)
                if parent is None:
                    raise ValueError("legal block parent must exist earlier")
                if parent.document_id != block.document_id:
                    raise ValueError("legal block parent must share document ID")
                if parent.order_index >= block.order_index:
                    raise ValueError("legal block parent must precede child")
        diagnostics_by_id = {item.document_id: item for item in self.diagnostics}
        for document_id, document_blocks in blocks_by_document.items():
            order_indexes = [block.order_index for block in document_blocks]
            if order_indexes != list(range(len(document_blocks))):
                raise ValueError("block order indexes must be contiguous per document")
            if diagnostics_by_id[document_id].block_count != len(document_blocks):
                raise ValueError("diagnostic block count must match parsed blocks")
        for diagnostic in self.diagnostics:
            if diagnostic.document_id not in blocks_by_document and diagnostic.block_count:
                raise ValueError("diagnostic reports blocks that do not exist")

        recognized_types = {
            LegalBlockType.PART,
            LegalBlockType.CHAPTER,
            LegalBlockType.SECTION,
            LegalBlockType.SUBSECTION,
            LegalBlockType.ARTICLE,
            LegalBlockType.CLAUSE,
            LegalBlockType.POINT,
            LegalBlockType.APPENDIX,
        }
        for diagnostic in self.diagnostics:
            actual_count = sum(
                block.block_type in recognized_types
                for block in blocks_by_document.get(diagnostic.document_id, [])
            )
            if diagnostic.recognized_structure_count != actual_count:
                raise ValueError(
                    "diagnostic structure count must match recognized blocks"
                )
        return self
