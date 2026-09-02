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
from legal_agentic_rag.retrieval.v2_branches import (
    V2BM25RetrievalBranch,
    V2DenseRetrievalBranch,
    build_v2_fixed_retriever,
)

__all__ = [
    "DenseRetriever",
    "FixedRetriever",
    "GraphExpandedRetriever",
    "HybridRetriever",
    "QueryBranchResult",
    "QueryUnderstandingService",
    "RerankingRetriever",
    "V2BM25RetrievalBranch",
    "V2DenseRetrievalBranch",
    "build_v2_fixed_retriever",
    "fuse_query_branches",
    "reciprocal_rank_fusion",
]
