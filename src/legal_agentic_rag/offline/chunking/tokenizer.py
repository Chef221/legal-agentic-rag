"""Deterministic tokenizer boundaries for legal chunk construction."""

from collections.abc import Callable
import re
from typing import Protocol

from legal_agentic_rag.configuration.offline import EmbeddingConfig
from legal_agentic_rag.exceptions import BackendInitializationError, DataValidationError

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class _TokenizerEncoding(dict[str, object]):
    """Structural marker for Hugging Face tokenizer return values."""


class _TokenizerBackend(Protocol):
    def __call__(self, text: str, **kwargs: object) -> _TokenizerEncoding: ...


TokenizerLoader = Callable[[EmbeddingConfig], _TokenizerBackend]


class UnicodeWordTokenizer:
    """Count Unicode words and punctuation without external model packages."""

    name = "unicode_word_v1"

    @property
    def identity(self) -> dict[str, str]:
        """Return the complete dependency-free tokenizer identity."""
        return {"tokenizer_name": self.name}

    def count(self, text: str) -> int:
        """Return the deterministic token count for one text value."""
        return sum(1 for _ in _TOKEN_PATTERN.finditer(text))

    def split(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split text at token spans with a bounded sliding-window overlap."""
        matches = list(_TOKEN_PATTERN.finditer(text))
        if not matches:
            return []
        if len(matches) <= max_tokens:
            return [text.strip()]
        step = max_tokens - overlap_tokens
        fragments: list[str] = []
        start = 0
        while start < len(matches):
            end = min(start + max_tokens, len(matches))
            char_start = matches[start].start()
            char_end = matches[end - 1].end()
            fragment = text[char_start:char_end].strip()
            if fragment:
                fragments.append(fragment)
            if end == len(matches):
                break
            start += step
        return fragments


class EmbeddingModelTokenizer:
    """Budget chunks with the exact revision-pinned embedding tokenizer."""

    name = "embedding_model_v1"

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        tokenizer_loader: TokenizerLoader | None = None,
    ) -> None:
        self._config = config
        self._loader = tokenizer_loader or self._load_tokenizer
        self._tokenizer: _TokenizerBackend | None = None

    @property
    def identity(self) -> dict[str, str]:
        """Pin the tokenizer to the same model identity as corpus embeddings."""
        return {
            "tokenizer_name": self.name,
            "tokenizer_model_name": self._config.model_name,
            "tokenizer_model_revision": self._config.model_revision,
            "tokenizer_document_prefix": self._config.document_prefix,
        }

    def count(self, text: str) -> int:
        """Count prefix, content, and special tokens exactly as embedding input."""
        encoded = self._require_tokenizer()(
            f"{self._config.document_prefix} {text}",
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )
        length = encoded.get("length")
        if isinstance(length, int):
            return length
        if isinstance(length, list) and len(length) == 1 and isinstance(length[0], int):
            return length[0]
        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, list):
            return len(input_ids)
        raise DataValidationError("Embedding tokenizer returned no usable token count")

    def split(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split on source spans while enforcing the exact embedding budget."""
        matches = list(_TOKEN_PATTERN.finditer(text))
        if not matches:
            return []
        if self.count(text) <= max_tokens:
            return [text.strip()]

        fragments: list[str] = []
        start = 0
        while start < len(matches):
            low = start + 1
            high = len(matches)
            accepted_end: int | None = None
            while low <= high:
                middle = (low + high) // 2
                candidate = text[matches[start].start() : matches[middle - 1].end()]
                if self.count(candidate) <= max_tokens:
                    accepted_end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if accepted_end is None:
                raise DataValidationError(
                    "One legal token span exceeds the embedding model window"
                )
            fragment = text[
                matches[start].start() : matches[accepted_end - 1].end()
            ].strip()
            if fragment:
                fragments.append(fragment)
            if accepted_end == len(matches):
                break
            start = max(start + 1, accepted_end - overlap_tokens)
        return fragments

    def _require_tokenizer(self) -> _TokenizerBackend:
        if self._tokenizer is None:
            self._tokenizer = self._loader(self._config)
        return self._tokenizer

    @staticmethod
    def _load_tokenizer(config: EmbeddingConfig) -> _TokenizerBackend:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise BackendInitializationError(
                "transformers dependency is unavailable for chunking"
            ) from error
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                revision=config.model_revision,
                local_files_only=config.local_files_only,
                trust_remote_code=False,
                use_fast=True,
            )
        except Exception as error:
            raise BackendInitializationError(
                "Embedding tokenizer could not be initialized"
            ) from error
        return tokenizer
