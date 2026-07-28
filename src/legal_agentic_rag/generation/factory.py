"""Configuration-driven construction of generation and verification boundaries."""

from legal_agentic_rag.configuration.online import (
    ClaimVerificationConfig,
    GenerationConfig,
    SemanticVerificationConfig,
)
from legal_agentic_rag.contracts.answer_generator import AnswerGenerator
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.generation.citation_verifier import (
    RuleBasedCitationVerifier,
)
from legal_agentic_rag.generation.extractive_generator import (
    ExtractiveAnswerGenerator,
)
from legal_agentic_rag.generation.model_generator import (
    ModelBackedAnswerGenerator,
)
from legal_agentic_rag.generation.openai_compatible import (
    OpenAICompatibleChatProvider,
)
from legal_agentic_rag.generation.transformers_provider import (
    TransformersChatProvider,
)
from legal_agentic_rag.generation.semantic_verifier import (
    ModelBackedCitationVerifier,
)


def build_answer_generator(
    config: GenerationConfig,
    *,
    provider: ChatModelProvider | None = None,
) -> AnswerGenerator:
    """Build only the explicitly configured grounded generator."""
    if config.backend == "extractive":
        return ExtractiveAnswerGenerator()
    if config.backend == "transformers":
        return ModelBackedAnswerGenerator(
            provider or TransformersChatProvider(config),
            max_structured_output_retries=(
                config.max_structured_output_retries
            ),
        )
    return ModelBackedAnswerGenerator(
        provider or OpenAICompatibleChatProvider(config),
        max_structured_output_retries=config.max_structured_output_retries,
    )


def build_citation_verifier(
    claim_config: ClaimVerificationConfig,
    semantic_config: SemanticVerificationConfig,
    *,
    provider: ChatModelProvider | None = None,
) -> CitationVerifier:
    """Compose deterministic hard checks with an optional semantic model."""
    base_verifier = RuleBasedCitationVerifier(claim_config)
    if semantic_config.backend == "disabled":
        return base_verifier
    provider_config = semantic_config.as_generation_config()
    semantic_provider = provider
    if semantic_provider is None:
        if semantic_config.backend == "transformers":
            semantic_provider = TransformersChatProvider(provider_config)
        else:
            semantic_provider = OpenAICompatibleChatProvider(provider_config)
    return ModelBackedCitationVerifier(
        base_verifier,
        semantic_provider,
        max_structured_output_retries=(
            semantic_config.max_structured_output_retries
        ),
    )


def build_generation_components(
    generation_config: GenerationConfig,
    claim_config: ClaimVerificationConfig,
    semantic_config: SemanticVerificationConfig,
) -> tuple[AnswerGenerator, CitationVerifier]:
    """Build generator and verifier while reusing compatible local weights."""
    answer_provider: ChatModelProvider | None = None
    semantic_provider: ChatModelProvider | None = None
    if (
        generation_config.backend == "transformers"
        and semantic_config.backend == "transformers"
    ):
        primary_provider = TransformersChatProvider(generation_config)
        verifier_provider_config = semantic_config.as_generation_config()
        if primary_provider.can_share_runtime_with(verifier_provider_config):
            answer_provider = primary_provider
            semantic_provider = primary_provider.with_shared_runtime(
                verifier_provider_config
            )
    return (
        build_answer_generator(
            generation_config,
            provider=answer_provider,
        ),
        build_citation_verifier(
            claim_config,
            semantic_config,
            provider=semantic_provider,
        ),
    )
