"""Public unified schema models."""

from legal_agentic_rag.schemas.agent_state import AgentState, RetrievalHistoryItem
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ContextGrade,
    Evidence,
)
from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity
from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalBlockType,
    LegalChunk,
    LegalDocument,
    LegalStructure,
)
from legal_agentic_rag.schemas.legal_relationships import LegalRelationship
from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    ArtifactType,
    ArtifactValidationResult,
    DatasetManifest,
)
from legal_agentic_rag.schemas.retrieval import (
    GraphPathStep,
    RetrievalFilters,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)

__all__ = [
    "AgentState",
    "AnswerResponse",
    "ArtifactManifest",
    "ArtifactType",
    "ArtifactValidationResult",
    "AuditIssue",
    "AuditSeverity",
    "Citation",
    "CitationVerificationResult",
    "ContextGrade",
    "DatasetManifest",
    "Evidence",
    "GraphPathStep",
    "LegalBlock",
    "LegalBlockType",
    "LegalChunk",
    "LegalDocument",
    "LegalRelationship",
    "LegalStructure",
    "RetrievalFilters",
    "RetrievalHistoryItem",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalResponse",
    "RetrievalStrategy",
    "RetrievalTrace",
]
