"""Pinned Sentence Transformers provider for multilingual E5 embeddings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
import logging
from typing import Protocol

import numpy as np

from legal_agentic_rag.configuration.offline import EmbeddingConfig
from legal_agentic_rag.exceptions import BackendInitializationError, ModelError

_LOGGER = logging.getLogger(__name__)


class _EmbeddingModel(Protocol):
    max_seq_length: int

    def get_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


ModelLoader = Callable[[EmbeddingConfig], _EmbeddingModel]


class SentenceTransformerEmbeddingProvider:
    """Encode passages and queries with a revision-pinned multilingual E5 model."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._config = config or EmbeddingConfig()
        self._model_loader = model_loader or self._load_sentence_transformer
        self._model: _EmbeddingModel | None = None

    @property
    def provider_name(self) -> str:
        """Return the concrete provider package identity."""
        return "sentence-transformers"

    @property
    def provider_version(self) -> str:
        """Return the installed provider version used for reproducibility."""
        try:
            return version(self.provider_name)
        except PackageNotFoundError as error:
            raise BackendInitializationError(
                "sentence-transformers dependency is unavailable"
            ) from error

    @property
    def model_name(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._config.model_name

    @property
    def model_revision(self) -> str:
        """Return the immutable Hugging Face model revision."""
        return self._config.model_revision

    @property
    def dimension(self) -> int:
        """Return the pinned output dimension without loading model weights."""
        return self._config.expected_dimension

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        """Embed legal chunk search texts using the E5 passage prefix."""
        if batch_size <= 0:
            raise ModelError("Embedding batch size must be positive")
        values = self._validate_texts(texts)
        if not values:
            return []
        prefixed = [f"{self._config.document_prefix} {text}" for text in values]
        return self._encode(prefixed, batch_size=batch_size)

    def embed_query(self, text: str) -> list[float]:
        """Embed one normalized question using the E5 query prefix."""
        values = self._validate_texts([text])
        prefixed = [f"{self._config.query_prefix} {values[0]}"]
        return self._encode(prefixed, batch_size=1)[0]

    def _encode(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        model = self._require_model()
        try:
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self._config.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise ModelError("Embedding model failed to encode text") from error
        vectors = np.asarray(encoded, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (
            len(texts),
            self._config.expected_dimension,
        ):
            raise ModelError("Embedding model returned an invalid vector shape")
        if not np.isfinite(vectors).all():
            raise ModelError("Embedding model returned non-finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ModelError("Embedding model returned a zero vector")
        normalized = vectors / norms
        return normalized.tolist()

    def _require_model(self) -> _EmbeddingModel:
        if self._model is None:
            try:
                model = self._model_loader(self._config)
                model.max_seq_length = self._config.max_sequence_length
                if (
                    model.get_embedding_dimension()
                    != self._config.expected_dimension
                ):
                    raise BackendInitializationError(
                        "Embedding model dimension does not match configuration"
                    )
            except Exception as error:
                if isinstance(error, BackendInitializationError):
                    raise
                raise BackendInitializationError(
                    "Embedding model could not be initialized"
                ) from error
            self._model = model
            _LOGGER.info(
                "embedding_model_initialized",
                extra={
                    "model_name": self.model_name,
                    "model_revision": self.model_revision,
                    "device": self._config.device,
                },
            )
        return self._model

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        values = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ModelError("Embedding input text must not be empty")
        return values

    @staticmethod
    def _load_sentence_transformer(config: EmbeddingConfig) -> _EmbeddingModel:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise BackendInitializationError(
                "sentence-transformers dependency is unavailable"
            ) from error
        return SentenceTransformer(
            config.model_name,
            revision=config.model_revision,
            device=config.device,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
