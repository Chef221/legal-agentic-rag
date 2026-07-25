"""Tests for the dependency-free OpenAI-compatible model provider."""

import json
import socket
from urllib.error import HTTPError

import pytest

from legal_agentic_rag.configuration import GenerationConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ExternalServiceError,
    ModelError,
    OperationTimeoutError,
)
from legal_agentic_rag.generation import (
    ExtractiveAnswerGenerator,
    ModelBackedAnswerGenerator,
    OpenAICompatibleChatProvider,
    build_answer_generator,
)


def _config(**updates: object) -> GenerationConfig:
    payload: dict[str, object] = {
        "backend": "openai_compatible",
        "endpoint_url": "http://127.0.0.1:8001/v1/chat/completions",
        "model_name": "fixture-model",
        "model_revision": "fixture-revision",
        "timeout_seconds": 12.0,
        "max_output_tokens": 256,
    }
    payload.update(updates)
    return GenerationConfig.model_validate(payload)


def test_provider_sends_json_mode_request_and_parses_content() -> None:
    """The provider sends bounded settings and reads only assistant content."""
    captured: dict[str, object] = {}

    def send(request, timeout_seconds: float) -> bytes:  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout_seconds
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return json.dumps(
            {
                "choices": [
                    {"message": {"content": '{"answer":"fixture"}'}}
                ]
            }
        ).encode("utf-8")

    provider = OpenAICompatibleChatProvider(
        _config(),
        request_sender=send,
    )
    content = provider.complete(
        system_instruction="system",
        user_prompt="user",
    )

    assert content == '{"answer":"fixture"}'
    assert captured["timeout"] == 12.0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "fixture-model"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 256
    assert payload["response_format"] == {"type": "json_object"}


def test_provider_reads_secret_only_from_named_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing secret fails without placing its value in configuration."""
    monkeypatch.delenv("FIXTURE_MODEL_KEY", raising=False)
    with pytest.raises(BackendInitializationError, match="unavailable"):
        OpenAICompatibleChatProvider(
            _config(api_key_env="FIXTURE_MODEL_KEY")
        )


def test_provider_classifies_timeout_http_and_invalid_envelope() -> None:
    """Transport and completion-envelope failures use project exceptions."""
    def timeout(*args: object) -> bytes:
        raise socket.timeout()

    def http_error(*args: object) -> bytes:
        raise HTTPError("http://model", 503, "unavailable", {}, None)

    def invalid(*args: object) -> bytes:
        return b'{"choices":[]}'

    with pytest.raises(OperationTimeoutError):
        OpenAICompatibleChatProvider(
            _config(), request_sender=timeout
        ).complete(system_instruction="system", user_prompt="user")
    with pytest.raises(ExternalServiceError, match="503"):
        OpenAICompatibleChatProvider(
            _config(), request_sender=http_error
        ).complete(system_instruction="system", user_prompt="user")
    with pytest.raises(ModelError, match="envelope"):
        OpenAICompatibleChatProvider(
            _config(), request_sender=invalid
        ).complete(system_instruction="system", user_prompt="user")


def test_generator_factory_preserves_safe_default_and_explicit_model_mode() -> None:
    """Configuration selects model inference only when explicitly requested."""
    assert isinstance(
        build_answer_generator(GenerationConfig()),
        ExtractiveAnswerGenerator,
    )
    assert isinstance(
        build_answer_generator(_config()),
        ModelBackedAnswerGenerator,
    )
