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
    claim_config: ClaimVerificationConfig | None = None,
) -> AnswerGenerator:
    """Build only the explicitly configured grounded generator."""
    if config.backend == "extractive":
        return ExtractiveAnswerGenerator()
    if config.backend == "transformers":
        grounding_verifier = (
            RuleBasedCitationVerifier(
                claim_config or ClaimVerificationConfig()
            )
            if config.max_grounding_repair_retries
            or config.grounding_failure_policy
            in {"supported_claims", "supported_claims_or_top_evidence"}
            else None
        )
        return ModelBackedAnswerGenerator(
            provider or TransformersChatProvider(config),
            max_structured_output_retries=(
                config.max_structured_output_retries
            ),
            max_model_error_retries=config.max_model_error_retries,
            model_failure_policy=config.model_failure_policy,
            answer_style=config.answer_style,
            prompt_schema_mode=config.prompt_schema_mode,
            grounding_verifier=grounding_verifier,
            max_grounding_repair_retries=(
                config.max_grounding_repair_retries
            ),
            grounding_failure_policy=config.grounding_failure_policy,
            extractive_fallback_max_evidence=(
                config.extractive_fallback_max_evidence
            ),
            salvage_rendering=config.salvage_rendering,
        )
    grounding_verifier = (
        RuleBasedCitationVerifier(claim_config or ClaimVerificationConfig())
        if config.max_grounding_repair_retries
        or config.grounding_failure_policy
        in {"supported_claims", "supported_claims_or_top_evidence"}
        else None
    )
    return ModelBackedAnswerGenerator(
        provider or OpenAICompatibleChatProvider(config),
        max_structured_output_retries=config.max_structured_output_retries,
        max_model_error_retries=config.max_model_error_retries,
        model_failure_policy=config.model_failure_policy,
        answer_style=config.answer_style,
        prompt_schema_mode=config.prompt_schema_mode,
        grounding_verifier=grounding_verifier,
        max_grounding_repair_retries=config.max_grounding_repair_retries,
        grounding_failure_policy=config.grounding_failure_policy,
        extractive_fallback_max_evidence=(
            config.extractive_fallback_max_evidence
        ),
        salvage_rendering=config.salvage_rendering,
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
            claim_config=claim_config,
        ),
        build_citation_verifier(
            claim_config,
            semantic_config,
            provider=semantic_provider,
        ),
    )
