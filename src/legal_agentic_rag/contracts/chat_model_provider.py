"""Backend-neutral contract for one structured chat-model completion."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatModelProvider(Protocol):
    """Generate text without exposing a concrete model server to core code."""

    @property
    def provider_name(self) -> str:
        """Return the concrete provider identity."""
        ...

    @property
    def provider_version(self) -> str:
        """Return the provider contract version."""
        ...

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        ...

    @property
    def model_revision(self) -> str:
        """Return the pinned model revision."""
        ...

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        """Return one model completion for the supplied bounded prompt."""
        ...
