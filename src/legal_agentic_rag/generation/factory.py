"""Configuration-driven construction of the answer-generator boundary."""

from legal_agentic_rag.configuration.online import GenerationConfig
from legal_agentic_rag.contracts.answer_generator import AnswerGenerator
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


def build_answer_generator(config: GenerationConfig) -> AnswerGenerator:
    """Build only the explicitly configured grounded generator."""
    if config.backend == "extractive":
        return ExtractiveAnswerGenerator()
    if config.backend == "transformers":
        return ModelBackedAnswerGenerator(
            TransformersChatProvider(config),
            max_structured_output_retries=(
                config.max_structured_output_retries
            ),
        )
    return ModelBackedAnswerGenerator(
        OpenAICompatibleChatProvider(config),
        max_structured_output_retries=config.max_structured_output_retries,
    )
