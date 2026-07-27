"""Local Hugging Face Transformers chat-model provider."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
import logging
from threading import RLock
from time import perf_counter
from typing import Any

from legal_agentic_rag.configuration.online import GenerationConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ModelError,
)

_LOGGER = logging.getLogger(__name__)
RuntimeLoader = Callable[[], tuple[Any, Any, Any]]


class TransformersChatProvider:
    """Run one pinned causal language model without exposing it to core code."""

    provider_name = "transformers"

    def __init__(
        self,
        config: GenerationConfig,
        *,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        if config.backend != "transformers":
            raise BackendInitializationError(
                "Transformers provider received incompatible configuration"
            )
        if config.model_name is None or config.model_revision is None:
            raise BackendInitializationError(
                "Transformers provider configuration is incomplete"
            )
        try:
            self.provider_version = version("transformers")
        except PackageNotFoundError as error:
            raise BackendInitializationError(
                "Transformers package is unavailable"
            ) from error
        self.model_name = config.model_name
        self.model_revision = config.model_revision
        self._device = config.device
        self._torch_dtype = config.torch_dtype
        self._local_files_only = config.local_files_only
        self._max_input_tokens = config.max_input_tokens
        self._max_output_tokens = config.max_output_tokens
        self._temperature = config.temperature
        self._runtime_loader = runtime_loader or self._load_runtime
        self._runtime: tuple[Any, Any, Any] | None = None
        self._lock = RLock()

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        """Return one deterministic, bounded local chat completion."""
        started = perf_counter()
        with self._lock:
            torch, tokenizer, model = self._require_runtime()
            try:
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"]
                input_length = int(input_ids.shape[-1])
                if input_length > self._max_input_tokens:
                    raise ModelError(
                        "Model prompt exceeds the configured input-token limit"
                    )
                model_inputs = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                }
                generation_options: dict[str, Any] = {
                    "max_new_tokens": self._max_output_tokens,
                    "do_sample": self._temperature > 0,
                }
                if self._temperature > 0:
                    generation_options["temperature"] = self._temperature
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id
                if pad_token_id is not None:
                    generation_options["pad_token_id"] = pad_token_id
                with torch.inference_mode():
                    output_ids = model.generate(
                        **model_inputs,
                        **generation_options,
                    )
                completion = tokenizer.decode(
                    output_ids[0, input_length:],
                    skip_special_tokens=True,
                ).strip()
            except ModelError:
                raise
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise ModelError(
                    "Transformers model completion failed"
                ) from error
        if not completion:
            raise ModelError("Transformers model returned empty completion content")
        _LOGGER.info(
            "transformers_chat_completion_completed",
            extra={"latency_ms": (perf_counter() - started) * 1000},
        )
        return completion

    def _require_runtime(self) -> tuple[Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        started = perf_counter()
        try:
            self._runtime = self._runtime_loader()
        except BackendInitializationError:
            raise
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise BackendInitializationError(
                "Transformers model initialization failed"
            ) from error
        _LOGGER.info(
            "transformers_chat_model_initialized",
            extra={
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "device": self._device,
                "latency_ms": (perf_counter() - started) * 1000,
            },
        )
        return self._runtime

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise BackendInitializationError(
                "Transformers generation dependencies are unavailable"
            ) from error

        dtype = getattr(torch, self._torch_dtype)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                local_files_only=self._local_files_only,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                local_files_only=self._local_files_only,
                trust_remote_code=False,
                torch_dtype=dtype,
            )
            model.to(self._device)
            model.eval()
        except (OSError, RuntimeError, ValueError) as error:
            raise BackendInitializationError(
                "Transformers model initialization failed"
            ) from error
        return torch, tokenizer, model
