"""Minimal structural contract consumed by the closed tool registry."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from legal_agentic_rag.schemas.tools import ToolName


@runtime_checkable
class TypedTool(Protocol):
    """Expose one typed capability without defining a generic base class."""

    @property
    def name(self) -> ToolName:
        """Return the closed-set tool identity."""
        ...

    @property
    def description(self) -> str:
        """Return a concise capability and boundary description."""
        ...

    @property
    def input_model(self) -> type[BaseModel]:
        """Return the Pydantic input contract."""
        ...

    @property
    def output_model(self) -> type[BaseModel]:
        """Return the Pydantic output contract."""
        ...

    @property
    def timeout_seconds(self) -> float:
        """Return the configured invocation time budget."""
        ...

    def invoke(self, payload: BaseModel) -> BaseModel:
        """Execute one already-validated typed payload."""
        ...
