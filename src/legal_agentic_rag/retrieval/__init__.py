"""Online retrieval orchestration over backend-neutral contracts."""

from legal_agentic_rag.retrieval.dense import DenseRetriever
from legal_agentic_rag.retrieval.fixed import FixedRetriever, HybridRetriever
from legal_agentic_rag.retrieval.graph import GraphExpandedRetriever
from legal_agentic_rag.retrieval.rerank import RerankingRetriever
from legal_agentic_rag.retrieval.rrf import reciprocal_rank_fusion

__all__ = [
    "DenseRetriever",
    "FixedRetriever",
    "GraphExpandedRetriever",
    "HybridRetriever",
    "RerankingRetriever",
    "reciprocal_rank_fusion",
]
