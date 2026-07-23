"""Concrete embedding providers kept behind backend-neutral contracts."""

from legal_agentic_rag.embeddings.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)

__all__ = ["SentenceTransformerEmbeddingProvider"]
