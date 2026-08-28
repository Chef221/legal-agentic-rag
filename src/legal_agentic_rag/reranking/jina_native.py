"""Native listwise Jina reranker backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import logging
import math
import operator
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from legal_agentic_rag.configuration.online import RerankerConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ConfigurationError,
    ModelError,
    RetrievalError,
)
from legal_agentic_rag.reranking.legal_context import build_legal_rerank_text
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)


def compute_projector_state_sha256(module: Any) -> str:
    """Compute deterministic SHA256 over module state dict according to exact S7 specification."""
    state_hash = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().float().cpu().contiguous()
        state_hash.update(name.encode("utf-8"))
        state_hash.update(str(value.dtype).encode("utf-8"))
        state_hash.update(str(tuple(value.shape)).encode("utf-8"))
        state_hash.update(value.numpy().tobytes())
    return state_hash.hexdigest().lower()


class _JinaRerankModel(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_embeddings: bool = False,
    ) -> Any: ...


JinaModelLoader = Callable[[RerankerConfig], _JinaRerankModel]


class JinaNativeReranker:
    """Rerank a bounded candidate set using native Jina listwise cross-attention."""

    def __init__(
        self,
        config: RerankerConfig | None = None,
        *,
        model_loader: JinaModelLoader | None = None,
    ) -> None:
        self._config = config or RerankerConfig(backend="jina_native_listwise")
        self._model_loader = model_loader or self._load_jina_model
        self._model: _JinaRerankModel | None = None
        self._actual_device: str = self._config.device
        self._actual_parameter_count: int | None = self._config.expected_parameter_count

    @property
    def provider_name(self) -> str:
        """Return the concrete provider package identity."""
        return "transformers-jina"

    @property
    def provider_version(self) -> str:
        """Return the installed transformers package version."""
        try:
            return version("transformers")
        except PackageNotFoundError as error:
            raise BackendInitializationError(
                "transformers dependency is unavailable"
            ) from error

    @property
    def model_name(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._config.model_name

    @property
    def model_revision(self) -> str | None:
        """Return the immutable Hugging Face model revision."""
        return self._config.model_revision

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> RetrievalResponse:
        """Score query-candidate pairs natively and return top-k reranked hits."""
        if query.requested_strategy not in (
            None,
            RetrievalStrategy.RERANK,
            RetrievalStrategy.HYBRID_RERANK,
        ):
            raise RetrievalError(f"Reranker received incompatible strategy request: {query.requested_strategy}")
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
        question = query.rewritten_question or query.original_question
        serialized_documents = [build_legal_rerank_text(hit) for hit in values]

        model = self._ensure_model()
        is_cuda = self._config.device.casefold().startswith("cuda")
        is_o2 = self._config.projector_checkpoint_path is not None

        if is_cuda:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

        try:
            import torch
            with torch.no_grad():
                if is_o2 and is_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        native_results = model.rerank(
                            query=question,
                            documents=serialized_documents,
                            top_n=len(serialized_documents),
                            return_embeddings=False,
                        )
                else:
                    native_results = model.rerank(
                        query=question,
                        documents=serialized_documents,
                        top_n=len(serialized_documents),
                        return_embeddings=False,
                    )
        except Exception as error:
            raise ModelError(f"Native Jina rerank execution failed: {error}") from error
        finally:
            if is_cuda:
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

        doc_scores = self._parse_native_results(native_results, expected_count=len(values))

        reranked_hits: list[RetrievalHit] = []
        for orig_idx, (hit, score) in enumerate(zip(values, doc_scores)):
            if not math.isfinite(score):
                raise ModelError(f"Reranker produced non-finite score {score} for chunk {hit.chunk_id}")
            trace = hit.retrieval_trace.model_copy(
                update={"reranker_score": float(score)}
            )
            reranked_hits.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    rank=orig_idx + 1,
                    score=float(score),
                    strategy=RetrievalStrategy.RERANK,
                    text=hit.text,
                    metadata=dict(hit.metadata),
                    retrieval_trace=trace,
                )
            )

        reranked_hits.sort(key=lambda h: (-h.score, h.rank, h.chunk_id))
        top_k_hits = reranked_hits[: query.top_k]
        for rank_idx, hit in enumerate(top_k_hits, start=1):
            hit.rank = rank_idx

        latency_ms = (perf_counter() - started) * 1000
        _LOGGER.info(
            "jina_reranking_completed",
            extra={
                "backend": self._config.backend,
                "model_name": self._config.model_name,
                "model_revision": self._config.model_revision,
                "requested_device": self._config.device,
                "actual_parameter_device": self._actual_device,
                "configured_dtype": self._config.torch_dtype,
                "actual_parameter_count": self._actual_parameter_count,
                "candidate_count": len(values),
                "selected_count": len(top_k_hits),
                "native_context_cap": self._config.native_context_cap,
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResponse(
            query=query.model_copy(
                update={"requested_strategy": RetrievalStrategy.RERANK}
            ),
            strategy=RetrievalStrategy.RERANK,
            hits=top_k_hits,
            latency_ms=latency_ms,
            artifact_versions=self._artifact_versions(values),
        )

    def _ensure_model(self) -> _JinaRerankModel:
        if self._model is None:
            self._model = self._model_loader(self._config)
        return self._model

    def _load_jina_model(self, config: RerankerConfig) -> _JinaRerankModel:
        """Load official Jina reranker model via transformers AutoModel."""
        _LOGGER.info(
            "jina_reranker_load_started",
            extra={
                "model_name": config.model_name,
                "model_revision": config.model_revision,
                "device": config.device,
                "torch_dtype": config.torch_dtype,
                "native_context_cap": config.native_context_cap,
            },
        )
        is_cuda = config.device.casefold().startswith("cuda")
        if is_cuda and config.torch_dtype != "float16":
            raise ConfigurationError(f"CUDA Jina reranking requires float16, got {config.torch_dtype}")
        if not is_cuda and config.torch_dtype != "float32":
            raise ConfigurationError(f"CPU Jina reranking requires float32, got {config.torch_dtype}")

        try:
            from transformers import AutoModel
            import torch
        except ImportError as error:
            raise BackendInitializationError(
                "transformers or torch is unavailable for Jina reranker"
            ) from error

        dtype = torch.float16 if is_cuda else torch.float32
        try:
            model = AutoModel.from_pretrained(
                config.model_name,
                revision=config.model_revision,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=config.local_files_only,
            )
        except Exception as error:
            raise BackendInitializationError(
                f"Failed to load Jina model from {config.model_name}@{config.model_revision}: {error}"
            ) from error

        try:
            model = model.to(config.device)
            model.eval()
        except Exception as error:
            raise BackendInitializationError(
                f"Failed to move Jina model to device {config.device}: {error}"
            ) from error

        if not hasattr(model, "rerank") or not callable(model.rerank):
            raise BackendInitializationError(
                f"Loaded Jina model {type(model)} does not expose a callable .rerank() method"
            )

        if not hasattr(model, "_ensure_tokenizer") or not callable(model._ensure_tokenizer):
            raise BackendInitializationError(
                f"Loaded Jina model {type(model)} does not expose a callable _ensure_tokenizer method"
            )

        model._ensure_tokenizer()

        if not hasattr(model, "_tokenizer") or model._tokenizer is None:
            raise BackendInitializationError(
                f"Loaded Jina model {type(model)} has no _tokenizer after _ensure_tokenizer()"
            )

        context_cap = config.native_context_cap or 12288
        try:
            model._tokenizer.model_max_length = context_cap
        except Exception as error:
            raise BackendInitializationError(f"Failed to set tokenizer model_max_length to {context_cap}: {error}") from error

        if getattr(model._tokenizer, "model_max_length", None) != context_cap:
            raise BackendInitializationError(
                f"Tokenizer model_max_length {getattr(model._tokenizer, 'model_max_length', None)} != {context_cap}"
            )

        try:
            actual_param_device = next(model.parameters()).device
            self._actual_device = str(actual_param_device)
            if is_cuda and actual_param_device.type != "cuda":
                raise BackendInitializationError(
                    f"CUDA placement gate violation: requested {config.device} but parameter device is {actual_param_device}"
                )
        except StopIteration:
            self._actual_device = config.device

        actual_params = sum(p.numel() for p in model.parameters())
        self._actual_parameter_count = actual_params

        if config.expected_parameter_count is not None:
            if actual_params != config.expected_parameter_count:
                raise ModelError(
                    f"Jina parameter gate violation: expected {config.expected_parameter_count} params, got {actual_params}"
                )

        if config.projector_checkpoint_path is not None:
            _LOGGER.info(
                "jina_o2_projector_load_started",
                extra={
                    "checkpoint_path": str(config.projector_checkpoint_path),
                    "expected_file_sha256": config.projector_checkpoint_sha256,
                    "expected_state_sha256": config.expected_projector_state_sha256,
                    "expected_projector_params": config.expected_projector_parameter_count,
                },
            )
            if (
                not hasattr(model, "projector")
                or not hasattr(model.projector, "parameters")
                or not hasattr(model.projector, "state_dict")
                or not hasattr(model.projector, "load_state_dict")
            ):
                raise BackendInitializationError(
                    f"Loaded Jina model {type(model)} does not expose a valid projector submodule"
                )

            for p in model.parameters():
                p.requires_grad_(False)

            try:
                model.projector.float()
            except Exception as error:
                raise BackendInitializationError(
                    f"Failed to convert projector to float32: {error}"
                ) from error

            actual_proj_params = sum(p.numel() for p in model.projector.parameters())
            if config.expected_projector_parameter_count is not None:
                if actual_proj_params != config.expected_projector_parameter_count:
                    raise ModelError(
                        f"O2 projector parameter count mismatch: expected {config.expected_projector_parameter_count}, got {actual_proj_params}"
                    )

            ckpt_path = Path(config.projector_checkpoint_path)
            if not ckpt_path.exists() or not ckpt_path.is_file():
                raise BackendInitializationError(
                    f"O2 projector checkpoint path '{ckpt_path}' does not exist or is not a regular file"
                )

            h = hashlib.sha256()
            with open(ckpt_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            file_sha = h.hexdigest().lower()
            if config.projector_checkpoint_sha256 is not None:
                if file_sha != config.projector_checkpoint_sha256.lower():
                    raise ModelError(
                        f"O2 projector checkpoint file SHA256 mismatch: expected {config.projector_checkpoint_sha256}, got {file_sha}"
                    )

            try:
                import safetensors.torch
            except ImportError as error:
                raise BackendInitializationError(
                    "safetensors dependency is unavailable for O2 projector loading"
                ) from error

            try:
                checkpoint_state = safetensors.torch.load_file(str(ckpt_path), device="cpu")
            except Exception as error:
                raise BackendInitializationError(
                    f"Failed to load safetensors checkpoint '{ckpt_path}': {error}"
                ) from error

            actual_keys = sorted(checkpoint_state.keys())
            expected_keys = ["0.weight", "2.weight"]
            if actual_keys != expected_keys:
                raise ModelError(
                    f"O2 projector checkpoint keys mismatch: expected {expected_keys}, got {actual_keys}"
                )

            try:
                model.projector.load_state_dict(checkpoint_state, strict=True)
            except Exception as error:
                raise ModelError(
                    f"Failed to load O2 projector state dict into model.projector: {error}"
                ) from error

            for p in model.projector.parameters():
                if p.dtype != torch.float32:
                    raise ModelError(f"O2 projector parameter dtype {p.dtype} is not float32")

            actual_state_sha = compute_projector_state_sha256(model.projector)
            if config.expected_projector_state_sha256 is not None:
                if actual_state_sha != config.expected_projector_state_sha256.lower():
                    raise ModelError(
                        f"O2 projector loaded state SHA256 mismatch: expected {config.expected_projector_state_sha256}, got {actual_state_sha}"
                    )

            _LOGGER.info(
                "jina_o2_projector_load_completed",
                extra={
                    "actual_file_sha256": file_sha,
                    "actual_state_sha256": actual_state_sha,
                    "actual_projector_params": actual_proj_params,
                },
            )

        _LOGGER.info("jina_reranker_load_completed")
        return model

    def _parse_native_results(
        self,
        native_results: Any,
        expected_count: int,
    ) -> list[float]:
        """Deterministically map native Jina rerank result items back to candidate indices matching exact V4."""
        doc_scores = [0.0] * expected_count
        seen_indices: set[int] = set()

        items: list[Any]
        if isinstance(native_results, list):
            items = native_results
        elif hasattr(native_results, "results") and isinstance(native_results.results, list):
            items = native_results.results
        else:
            raise ModelError(f"Unexpected native Jina rerank output format: {type(native_results)}")

        if len(items) != expected_count:
            raise ModelError(
                f"Jina rerank returned {len(items)} items for {expected_count} input documents"
            )

        for item in items:
            orig_idx: Any = None
            sc: Any = None

            if isinstance(item, dict):
                if "index" not in item:
                    raise ModelError(f"Missing 'index' in Jina rerank result dictionary: {item}")
                orig_idx = item["index"]
                if "relevance_score" in item:
                    sc = item["relevance_score"]
                elif "score" in item:
                    sc = item["score"]
                else:
                    raise ModelError(f"Missing relevance_score/score in Jina rerank dictionary: {item}")
            elif hasattr(item, "index") and (hasattr(item, "relevance_score") or hasattr(item, "score")):
                orig_idx = getattr(item, "index")
                sc = getattr(item, "relevance_score", getattr(item, "score", None))
                if sc is None:
                    raise ModelError(f"Missing relevance_score/score on Jina rerank object: {item}")
            else:
                raise ModelError(f"Unexpected item type in Jina rerank output: {type(item)}")

            if isinstance(orig_idx, bool):
                raise ModelError(f"Invalid non-integer candidate index {orig_idx!r} in Jina rerank results")

            try:
                norm_idx = operator.index(orig_idx)
            except (TypeError, ValueError) as error:
                raise ModelError(f"Invalid non-integer candidate index {orig_idx!r} in Jina rerank results") from error

            orig_idx = int(norm_idx)
            if orig_idx < 0 or orig_idx >= expected_count:
                raise ModelError(
                    f"Out-of-range candidate index {orig_idx} in Jina rerank results (expected 0..{expected_count-1})"
                )
            if orig_idx in seen_indices:
                raise ModelError(f"Duplicate candidate index {orig_idx} in Jina rerank results")

            try:
                sc_float = float(sc)
            except (ValueError, TypeError) as err:
                raise ModelError(f"Cannot cast candidate score {sc!r} to float: {err}") from err

            if not math.isfinite(sc_float):
                raise ModelError(f"Non-finite candidate score {sc_float} at index {orig_idx}")

            seen_indices.add(orig_idx)
            doc_scores[orig_idx] = sc_float

        if len(seen_indices) != expected_count:
            missing = set(range(expected_count)) - seen_indices
            raise ModelError(f"Incomplete coverage in Jina rerank: missing indices {sorted(missing)}")

        return doc_scores

    def _validate_candidates(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> None:
        if len(candidates) > self._config.max_candidates:
            raise RetrievalError(
                f"Candidate count {len(candidates)} exceeds reranker limit {self._config.max_candidates}"
            )
        for index, candidate in enumerate(candidates, start=1):
            if candidate.rank != index:
                raise RetrievalError("Candidates must be sorted with 1-based ranks")

    @staticmethod
    def _artifact_versions(candidates: Sequence[RetrievalHit]) -> dict[str, str]:
        versions: dict[str, str] = {}
        for candidate in candidates:
            for key, value in candidate.metadata.items():
                if key.endswith("_artifact_version") and isinstance(value, str):
                    versions[key.replace("_artifact_version", "")] = value
        return versions
