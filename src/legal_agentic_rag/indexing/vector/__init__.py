"""Exact NumPy vector indexing, persistence, and build orchestration."""

from legal_agentic_rag.indexing.vector.builder import VectorIndexBuilder
from legal_agentic_rag.indexing.vector.numpy_backend import NumpyVectorBackend
from legal_agentic_rag.indexing.vector.serving_metadata import (
    SQLiteVectorChunkStore,
    prepare_vector_serving_metadata,
)

__all__ = [
    "NumpyVectorBackend",
    "prepare_vector_serving_metadata",
    "SQLiteVectorChunkStore",
    "VectorIndexBuilder",
]
