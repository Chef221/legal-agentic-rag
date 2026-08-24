"""Concrete reranker providers behind the backend-neutral contract."""

from legal_agentic_rag.reranking.cross_encoder import CrossEncoderReranker
from legal_agentic_rag.reranking.factory import build_reranker
from legal_agentic_rag.reranking.jina_native import JinaNativeReranker
from legal_agentic_rag.reranking.legal_context import build_legal_rerank_text

__all__ = [
    "CrossEncoderReranker",
    "JinaNativeReranker",
    "build_legal_rerank_text",
    "build_reranker",
]
