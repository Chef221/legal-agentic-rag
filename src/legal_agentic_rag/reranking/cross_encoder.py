"""Revision-pinned Sentence Transformers cross-encoder reranker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
import logging
from time import perf_counter
from typing import Protocol

import numpy as np

from legal_agentic_rag.configuration.online import RerankerConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ModelError,
    RetrievalError,
)
from legal_agentic_rag.reranking.legal_context import (
    build_legal_rerank_text,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)


class _CrossEncoderModel(Protocol):
    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object: ...


ModelLoader = Callable[[RerankerConfig], _CrossEncoderModel]


class CrossEncoderReranker:
    """Rerank a bounded candidate set with revision-pinned model logits."""

    def __init__(
        self,
        config: RerankerConfig | None = None,
        *,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._config = config or RerankerConfig()
        self._model_loader = model_loader or self._load_cross_encoder
        self._model: _CrossEncoderModel | None = None

    @property
    def provider_name(self) -> str:
        """Return the concrete provider package identity."""
        return "sentence-transformers"

    @property
    def provider_version(self) -> str:
        """Return the installed provider version."""
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

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> RetrievalResponse:
        """Score query-candidate pairs and return final top-k hits."""
        if query.requested_strategy not in (None, RetrievalStrategy.RERANK):
            raise RetrievalError("Reranker received a non-rerank request")
        values = list(candidates)
        self._validate_candidates(query, values)
        started = perf_counter()
        if not values:
            return RetrievalResponse(
                query=query.model_copy(
                    update={"requested_strategy": RetrievalStrategy.RERANK}
                ),
                strategy=RetrievalStrategy.RERANK,
                warnings=["no_rerank_candidates"],
            )
        question = query.rewritten_question or query.normalized_question
        pairs = [
            (question, self._candidate_text(hit))
            for hit in values
        ]
        scores = self._predict(pairs)
        order = sorted(
            range(len(values)),
            key=lambda index: (
                -float(scores[index]),
                values[index].rank,
                values[index].chunk_id,
            ),
        )[: query.top_k]
        hits = [
            self._reranked_hit(values[index], rank, float(scores[index]))
            for rank, index in enumerate(order, start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "cross_encoder_rerank_completed",
            extra={
                "query_id": query.query_id,
                "strategy": RetrievalStrategy.RERANK.value,
                "candidate_count": len(values),
                "selected_count": len(hits),
                "model_name": self.model_name,
                "reranker_input_mode": self._config.input_mode,
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResponse(
            query=query.model_copy(
                update={"requested_strategy": RetrievalStrategy.RERANK}
            ),
            strategy=RetrievalStrategy.RERANK,
            hits=hits,
            latency_ms=latency_ms,
        )

    def _predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        try:
            output = self._require_model().predict(
                pairs,
                batch_size=self._config.batch_size,
                show_progress_bar=False,
                activation_fn=self._identity,
                apply_softmax=False,
                convert_to_numpy=True,
            )
        except BackendInitializationError:
            raise
        except Exception as error:
            raise ModelError("Cross-encoder failed to score candidates") from error
        scores = np.asarray(output, dtype=np.float32)
        if scores.shape == (len(pairs), 1):
            scores = scores[:, 0]
        if scores.shape != (len(pairs),):
            raise ModelError("Cross-encoder returned an invalid score shape")
        if not np.isfinite(scores).all():
            raise ModelError("Cross-encoder returned non-finite scores")
        return scores

    def _require_model(self) -> _CrossEncoderModel:
        if self._model is None:
            try:
                self._model = self._model_loader(self._config)
            except Exception as error:
                raise BackendInitializationError(
                    "Cross-encoder model could not be initialized"
                ) from error
            _LOGGER.info(
                "cross_encoder_model_initialized",
                extra={
                    "model_name": self.model_name,
                    "model_revision": self.model_revision,
                    "device": self._config.device,
                },
            )
        return self._model

    def _validate_candidates(
        self,
        query: RetrievalQuery,
        candidates: list[RetrievalHit],
    ) -> None:
        limit = min(query.candidate_k, self._config.max_candidates)
        if len(candidates) > limit:
            raise RetrievalError("Reranker candidate set exceeds the configured limit")
        chunk_ids = [hit.chunk_id for hit in candidates]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RetrievalError("Reranker candidates contain duplicate chunk IDs")

    @staticmethod
    def _reranked_hit(
        hit: RetrievalHit,
        rank: int,
        score: float,
    ) -> RetrievalHit:
        trace = hit.retrieval_trace.model_copy(
            update={"reranker_score": score}
        )
        return hit.model_copy(
            update={
                "rank": rank,
                "score": score,
                "strategy": RetrievalStrategy.RERANK,
                "retrieval_trace": trace,
            }
        )

    @staticmethod
    def _identity(value: object) -> object:
        """Keep raw logits rather than applying an output activation."""
        return value

    def _candidate_text(self, hit: RetrievalHit) -> str:
        if self._config.input_mode == "text_only":
            return hit.text
        return build_legal_rerank_text(hit)

    @staticmethod
    def _load_cross_encoder(config: RerankerConfig) -> _CrossEncoderModel:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise BackendInitializationError(
                "sentence-transformers dependency is unavailable"
            ) from error
        prompt_options: dict[str, object] = {}
        if config.prompt_name is not None and config.instruction is not None:
            prompt_options = {
                "prompts": {config.prompt_name: config.instruction},
                "default_prompt_name": config.prompt_name,
            }
        return CrossEncoder(
            config.model_name,
            revision=config.model_revision,
            device=config.device,
            max_length=config.max_length,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
            model_kwargs={"torch_dtype": config.torch_dtype},
            **prompt_options,
        )
