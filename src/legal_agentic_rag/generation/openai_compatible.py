"""OpenAI-compatible chat-completions provider using the standard library."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from legal_agentic_rag.configuration.online import GenerationConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ExternalServiceError,
    ModelError,
    OperationTimeoutError,
)

RequestSender = Callable[[Request, float], bytes]


def _send_request(request: Request, timeout_seconds: float) -> bytes:
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        return response.read()


class OpenAICompatibleChatProvider:
    """Call one explicitly configured OpenAI-compatible model endpoint."""

    provider_name = "openai_compatible"
    provider_version = "1.0"

    def __init__(
        self,
        config: GenerationConfig,
        *,
        request_sender: RequestSender | None = None,
    ) -> None:
        if config.backend != "openai_compatible":
            raise BackendInitializationError(
                "OpenAI-compatible provider received incompatible configuration"
            )
        if (
            config.endpoint_url is None
            or config.model_name is None
            or config.model_revision is None
        ):
            raise BackendInitializationError(
                "OpenAI-compatible provider configuration is incomplete"
            )
        self._endpoint_url = config.endpoint_url
        self.model_name = config.model_name
        self.model_revision = config.model_revision
        self._timeout_seconds = config.timeout_seconds
        self._temperature = config.temperature
        self._max_output_tokens = config.max_output_tokens
        self._request_sender = request_sender or _send_request
        self._api_key = self._load_api_key(config.api_key_env)

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        """Return the assistant content from one JSON-mode completion."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            self._endpoint_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            raw_response = self._request_sender(
                request,
                self._timeout_seconds,
            )
        except (TimeoutError, socket.timeout) as error:
            raise OperationTimeoutError(
                "Model endpoint exceeded the configured timeout"
            ) from error
        except HTTPError as error:
            raise ExternalServiceError(
                f"Model endpoint returned HTTP status {error.code}"
            ) from error
        except URLError as error:
            raise ExternalServiceError("Model endpoint is unavailable") from error
        except OSError as error:
            raise ExternalServiceError("Model endpoint request failed") from error

        return self._parse_response(raw_response)

    @staticmethod
    def _load_api_key(environment_name: str | None) -> str | None:
        if environment_name is None:
            return None
        value = os.environ.get(environment_name)
        if value is None or not value.strip():
            raise BackendInitializationError(
                "Configured model API key environment variable is unavailable"
            )
        return value

    @staticmethod
    def _parse_response(raw_response: bytes) -> str:
        try:
            payload: Any = json.loads(raw_response.decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise ModelError(
                "Model endpoint returned an invalid completion envelope"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise ModelError("Model endpoint returned empty completion content")
        return content.strip()
