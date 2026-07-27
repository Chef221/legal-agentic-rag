"""Fixed grounded answer generation and verification pipeline."""

from legal_agentic_rag.generation.citation_verifier import (
    RuleBasedCitationVerifier,
)
from legal_agentic_rag.generation.context_builder import ContextBuilder
from legal_agentic_rag.generation.context_grader import RuleBasedContextGrader
from legal_agentic_rag.generation.extractive_generator import (
    ExtractiveAnswerGenerator,
)
from legal_agentic_rag.generation.factory import build_answer_generator
from legal_agentic_rag.generation.model_generator import (
    ModelBackedAnswerGenerator,
)
from legal_agentic_rag.generation.openai_compatible import (
    OpenAICompatibleChatProvider,
)
from legal_agentic_rag.generation.service import FixedRAGService
from legal_agentic_rag.generation.transformers_provider import (
    TransformersChatProvider,
)

__all__ = [
    "ContextBuilder",
    "ExtractiveAnswerGenerator",
    "FixedRAGService",
    "ModelBackedAnswerGenerator",
    "OpenAICompatibleChatProvider",
    "RuleBasedCitationVerifier",
    "RuleBasedContextGrader",
    "TransformersChatProvider",
    "build_answer_generator",
]
