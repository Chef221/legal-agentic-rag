"""Competition-specific adapters kept outside the reusable RAG core."""

from legal_agentic_rag.competition.uit_dsc_2026 import (
    ContextSourceIdentity,
    UitDsc2026ContextAdapter,
    UitDsc2026CorpusIngestor,
    UitDsc2026DataLoader,
    UitDsc2026PassageCleaner,
)

__all__ = [
    "ContextSourceIdentity",
    "UitDsc2026ContextAdapter",
    "UitDsc2026CorpusIngestor",
    "UitDsc2026DataLoader",
    "UitDsc2026PassageCleaner",
]
