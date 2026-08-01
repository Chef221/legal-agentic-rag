"""Mapping from organizer context records into reusable legal documents."""

from legal_agentic_rag.configuration.competition import (
    OFFICIAL_CORPUS_DATASET_NAME,
)
from legal_agentic_rag.schemas.competition import CompetitionContext
from legal_agentic_rag.schemas.legal_documents import LegalDocument


class UitDsc2026ContextAdapter:
    """Map the four documented raw fields without inferring legal metadata."""

    def to_document(self, context: CompetitionContext) -> LegalDocument:
        """Preserve official identity, title, URL, and passage exactly."""
        return LegalDocument(
            document_id=context.context_id,
            title=context.title,
            source_url=context.source_url,
            clean_text=context.passage,
            has_content=True,
            source_dataset=OFFICIAL_CORPUS_DATASET_NAME,
            raw_metadata={},
        )
