"""Composition of the approved fixed tools without dynamic discovery."""

from __future__ import annotations

from typing import Protocol

from legal_agentic_rag.contracts.answer_generator import AnswerGenerator
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.contracts.context_grader import ContextGrader
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalResponse
from legal_agentic_rag.tools.generation import (
    AnswerGenerationTool,
    CitationVerificationTool,
    ContextGradingTool,
)
from legal_agentic_rag.tools.registry import ToolRegistry
from legal_agentic_rag.tools.retrieval import fixed_retrieval_tools


class _Retriever(Protocol):
    def search(self, query: RetrievalQuery) -> RetrievalResponse: ...


def build_fixed_tool_registry(
    *,
    retriever: _Retriever,
    context_grader: ContextGrader,
    answer_generator: AnswerGenerator,
    citation_verifier: CitationVerifier,
    retrieval_timeout_seconds: float = 30.0,
    generation_timeout_seconds: float = 30.0,
) -> ToolRegistry:
    """Build exactly the eight approved tools from injected fixed services."""
    tools = fixed_retrieval_tools(
        retriever,
        timeout_seconds=retrieval_timeout_seconds,
    )
    return ToolRegistry(
        [
            *tools,
            ContextGradingTool(
                context_grader,
                timeout_seconds=generation_timeout_seconds,
            ),
            AnswerGenerationTool(
                answer_generator,
                timeout_seconds=generation_timeout_seconds,
            ),
            CitationVerificationTool(
                citation_verifier,
                timeout_seconds=generation_timeout_seconds,
            ),
        ]
    )
