"""Mapping from organizer context records into reusable legal documents."""

from legal_agentic_rag.configuration.competition import (
    OFFICIAL_CORPUS_DATASET_NAME,
)
from legal_agentic_rag.schemas.competition import CompetitionContext
from legal_agentic_rag.schemas.legal_documents import LegalDocument


class UitDsc2026ContextAdapter:
    """Map audited organizer fields without inferring legal metadata."""

    def to_document(self, context: CompetitionContext) -> LegalDocument:
        """Preserve official identity, optional title, URL, and raw passage."""
        return LegalDocument(
            document_id=context.context_id,
            title=context.title,
            source_url=context.source_url,
            clean_text=context.passage,
            has_content=bool(context.passage.strip()),
            source_dataset=OFFICIAL_CORPUS_DATASET_NAME,
            raw_metadata={},
        )
