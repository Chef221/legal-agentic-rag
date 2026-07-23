"""Backend-neutral contract for bounded Agent orchestration."""

from typing import Protocol, runtime_checkable

from legal_agentic_rag.schemas.agent_state import AgentRunResult
from legal_agentic_rag.schemas.retrieval import RetrievalQuery


@runtime_checkable
class AgentWorkflow(Protocol):
    """Run registered online tools and return a traceable terminal result."""

    def run(self, query: RetrievalQuery) -> AgentRunResult:
        """Execute one bounded Agent workflow."""
        ...
