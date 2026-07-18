"""Backend-neutral protocol contracts."""

from legal_agentic_rag.contracts.answer_generator import AnswerGenerator
from legal_agentic_rag.contracts.bm25_backend import BM25Backend
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.contracts.context_grader import ContextGrader
from legal_agentic_rag.contracts.dataset_source import DatasetComponent, DatasetSource
from legal_agentic_rag.contracts.embedding_provider import EmbeddingProvider
from legal_agentic_rag.contracts.graph_backend import GraphBackend
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.contracts.vector_backend import VectorBackend

__all__ = [
    "AnswerGenerator",
    "BM25Backend",
    "CitationVerifier",
    "ContextGrader",
    "DatasetComponent",
    "DatasetSource",
    "EmbeddingProvider",
    "GraphBackend",
    "Reranker",
    "VectorBackend",
]
