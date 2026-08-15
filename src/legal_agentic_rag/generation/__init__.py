"""Fixed grounded answer generation and verification pipeline."""

from legal_agentic_rag.generation.citation_verifier import (
    RuleBasedCitationVerifier,
)
from legal_agentic_rag.generation.claim_salvage import (
    SupportedClaimSalvageResult,
    build_supported_claim_salvage,
)
from legal_agentic_rag.generation.claim_grounding import (
    RuleBasedClaimGroundingVerifier,
)
from legal_agentic_rag.generation.context_builder import ContextBuilder
from legal_agentic_rag.generation.context_grader import RuleBasedContextGrader
from legal_agentic_rag.generation.evidence_selector import EvidenceSelector
from legal_agentic_rag.generation.extractive_generator import (
    ExtractiveAnswerGenerator,
)
from legal_agentic_rag.generation.factory import (
    build_answer_generator,
    build_citation_verifier,
    build_generation_components,
)
from legal_agentic_rag.generation.model_generator import (
    ModelBackedAnswerGenerator,
)
from legal_agentic_rag.generation.openai_compatible import (
    OpenAICompatibleChatProvider,
)
from legal_agentic_rag.generation.service import FixedRAGService
from legal_agentic_rag.generation.semantic_verifier import (
    ModelBackedCitationVerifier,
)
from legal_agentic_rag.generation.transformers_provider import (
    TransformersChatProvider,
)

__all__ = [
    "ContextBuilder",
    "EvidenceSelector",
    "ExtractiveAnswerGenerator",
    "FixedRAGService",
    "ModelBackedAnswerGenerator",
    "ModelBackedCitationVerifier",
    "SupportedClaimSalvageResult",
    "OpenAICompatibleChatProvider",
    "RuleBasedCitationVerifier",
    "RuleBasedClaimGroundingVerifier",
    "RuleBasedContextGrader",
    "TransformersChatProvider",
    "build_answer_generator",
    "build_citation_verifier",
    "build_supported_claim_salvage",
    "build_generation_components",
]
