"""AIO Vietnamese legal documents dataset integration."""

from legal_agentic_rag.offline.datasets.aio.adapter import AioRecordAdapter
from legal_agentic_rag.offline.datasets.aio.normalizer import AioDocumentNormalizer
from legal_agentic_rag.offline.datasets.aio.source import AioDatasetSource
from legal_agentic_rag.offline.datasets.aio.relationship_normalizer import (
    AioRelationshipNormalizer,
)

__all__ = [
    "AioDatasetSource",
    "AioDocumentNormalizer",
    "AioRecordAdapter",
    "AioRelationshipNormalizer",
]
