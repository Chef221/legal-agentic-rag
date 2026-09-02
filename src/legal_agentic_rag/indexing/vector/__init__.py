"""Exact NumPy vector indexing, persistence, and build orchestration."""

from legal_agentic_rag.indexing.vector.builder import VectorIndexBuilder
from legal_agentic_rag.indexing.vector.numpy_backend import NumpyVectorBackend
from legal_agentic_rag.indexing.vector.serving_metadata import (
    SQLiteVectorChunkStore,
    prepare_vector_serving_metadata,
)
from legal_agentic_rag.indexing.vector.v2_precomputed_backend import (
    V2PrecomputedDenseBackend,
)
from legal_agentic_rag.indexing.vector.v2_retrieval_unit_store import (
    V2RetrievalUnitStore,
)

__all__ = [
    "NumpyVectorBackend",
    "prepare_vector_serving_metadata",
    "SQLiteVectorChunkStore",
    "V2PrecomputedDenseBackend",
    "V2RetrievalUnitStore",
    "VectorIndexBuilder",
]
