"""Online retrieval orchestration over backend-neutral contracts."""

from legal_agentic_rag.retrieval.dense import DenseRetriever
from legal_agentic_rag.retrieval.fixed import FixedRetriever, HybridRetriever
from legal_agentic_rag.retrieval.graph import GraphExpandedRetriever
from legal_agentic_rag.retrieval.multi_query import (
    QueryBranchResult,
    fuse_query_branches,
)
from legal_agentic_rag.retrieval.query_understanding import (
    QueryUnderstandingService,
)
from legal_agentic_rag.retrieval.rerank import RerankingRetriever
from legal_agentic_rag.retrieval.rrf import reciprocal_rank_fusion

__all__ = [
    "DenseRetriever",
    "FixedRetriever",
    "GraphExpandedRetriever",
    "HybridRetriever",
    "QueryBranchResult",
    "QueryUnderstandingService",
    "RerankingRetriever",
    "fuse_query_branches",
    "reciprocal_rank_fusion",
]
