"""Tests for bounded local Transformers answer generation."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from legal_agentic_rag.configuration import GenerationConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ModelError,
)
from legal_agentic_rag.generation import (
    ModelBackedAnswerGenerator,
    TransformersChatProvider,
    build_answer_generator,
)


def _config(**updates: object) -> GenerationConfig:
    payload: dict[str, object] = {
        "backend": "transformers",
        "model_name": "fixture-model",
        "model_revision": "fixture-revision",
        "device": "cpu",
        "torch_dtype": "float32",
        "max_input_tokens": 16,
        "max_output_tokens": 32,
    }
    payload.update(updates)
    return GenerationConfig.model_validate(payload)


class _Tokenizer:
    pad_token_id = None
    eos_token_id = 0

    def __init__(self, *, input_length: int = 3) -> None:
        self.input_length = input_length
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "fixture prompt"

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, Any]:
        assert prompt == "fixture prompt"
        assert return_tensors == "pt"
        return {
            "input_ids": torch.arange(self.input_length).reshape(1, -1),
            "attention_mask": torch.ones((1, self.input_length), dtype=torch.long),
        }

    @staticmethod
    def decode(
        token_ids: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert token_ids.tolist() == [7, 8]
        assert skip_special_tokens is True
        return '{"answer":"fixture"}'


class _Model:
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def generate(self, **options: Any) -> torch.Tensor:
        self.options = options
        return torch.cat(
            (
                options["input_ids"],
                torch.tensor([[7, 8]], device=options["input_ids"].device),
            ),
            dim=1,
        )


def test_provider_builds_chat_prompt_and_decodes_only_generated_tokens() -> None:
    """Local inference preserves roles and excludes prompt tokens from output."""
    tokenizer = _Tokenizer()
    model = _Model()
    provider = TransformersChatProvider(
        _config(),
        runtime_loader=lambda: (torch, tokenizer, model),
    )

    completion = provider.complete(
        system_instruction="system",
        user_prompt="user",
    )

    assert completion == '{"answer":"fixture"}'
    assert tokenizer.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert model.options["do_sample"] is False
    assert model.options["max_new_tokens"] == 32
    assert model.options["pad_token_id"] == 0


def test_provider_rejects_oversized_prompt_without_truncating_legal_text() -> None:
    """Input over the explicit bound fails closed instead of silent truncation."""
    provider = TransformersChatProvider(
        _config(max_input_tokens=2),
        runtime_loader=lambda: (torch, _Tokenizer(input_length=3), _Model()),
    )

    with pytest.raises(ModelError, match="input-token limit"):
        provider.complete(system_instruction="system", user_prompt="user")


def test_provider_classifies_lazy_initialization_failure() -> None:
    """A local model load failure uses the backend-initialization taxonomy."""
    def fail() -> tuple[Any, Any, Any]:
        raise OSError("fixture failure")

    provider = TransformersChatProvider(_config(), runtime_loader=fail)

    with pytest.raises(BackendInitializationError, match="initialization"):
        provider.complete(system_instruction="system", user_prompt="user")


def test_factory_selects_model_generator_for_transformers_backend() -> None:
    """The core factory remains backend-neutral and configuration-driven."""
    assert isinstance(
        build_answer_generator(_config()),
        ModelBackedAnswerGenerator,
    )
