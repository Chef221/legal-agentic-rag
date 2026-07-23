"""Unit tests for the pinned Sentence Transformers embedding provider."""

import numpy as np
import pytest

from legal_agentic_rag.configuration import EmbeddingConfig
from legal_agentic_rag.embeddings import SentenceTransformerEmbeddingProvider
from legal_agentic_rag.exceptions import BackendInitializationError, ModelError


class _FixtureModel:
    def __init__(
        self,
        vectors: list[list[float]] | None = None,
        *,
        dimension: int = 3,
    ) -> None:
        self.max_seq_length = 0
        self.dimension = dimension
        self.vectors = vectors
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences: list[str], **kwargs: object) -> object:
        self.calls.append((sentences, kwargs))
        if self.vectors is not None:
            return np.asarray(self.vectors[: len(sentences)], dtype=np.float32)
        return np.asarray(
            [[float(index + 1), 1.0, 0.0] for index in range(len(sentences))],
            dtype=np.float32,
        )


def _config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name="fixture/e5",
        model_revision="fixture-revision",
        expected_dimension=3,
        max_sequence_length=128,
        device="cpu",
    )


def test_provider_applies_e5_prefixes_and_normalizes_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passage/query prefixes and normalized float vectors are explicit."""
    monkeypatch.setattr(
        "legal_agentic_rag.embeddings.sentence_transformer_provider.version",
        lambda package_name: "fixture-provider-version",
    )
    model = _FixtureModel()
    provider = SentenceTransformerEmbeddingProvider(
        _config(), model_loader=lambda config: model
    )

    documents = provider.embed_documents(["Điều 1", "Điều 2"], batch_size=2)
    query = provider.embed_query("quy định nào")

    assert model.max_seq_length == 128
    assert model.calls[0][0] == ["passage: Điều 1", "passage: Điều 2"]
    assert model.calls[1][0] == ["query: quy định nào"]
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)
    assert np.isclose(np.linalg.norm(query), 1.0)
    assert provider.dimension == 3
    assert provider.provider_name == "sentence-transformers"
    assert provider.provider_version == "fixture-provider-version"
    assert provider.model_name == "fixture/e5"
    assert provider.model_revision == "fixture-revision"


def test_provider_rejects_invalid_text_batch_and_model_outputs() -> None:
    """Empty text, wrong shape, zero vectors, and non-finite values fail clearly."""
    provider = SentenceTransformerEmbeddingProvider(
        _config(), model_loader=lambda config: _FixtureModel()
    )
    with pytest.raises(ModelError):
        provider.embed_documents([""], batch_size=1)
    with pytest.raises(ModelError):
        provider.embed_documents(["Điều 1"], batch_size=0)

    wrong_shape = SentenceTransformerEmbeddingProvider(
        _config(),
        model_loader=lambda config: _FixtureModel([[1.0, 2.0]], dimension=3),
    )
    with pytest.raises(ModelError, match="shape"):
        wrong_shape.embed_query("câu hỏi")

    zero = SentenceTransformerEmbeddingProvider(
        _config(),
        model_loader=lambda config: _FixtureModel([[0.0, 0.0, 0.0]]),
    )
    with pytest.raises(ModelError, match="zero"):
        zero.embed_query("câu hỏi")

    non_finite = SentenceTransformerEmbeddingProvider(
        _config(),
        model_loader=lambda config: _FixtureModel([[1.0, float("nan"), 0.0]]),
    )
    with pytest.raises(ModelError, match="non-finite"):
        non_finite.embed_query("câu hỏi")


def test_provider_rejects_loader_and_dimension_mismatch() -> None:
    """Initialization and pinned dimension failures use backend error types."""
    def fail_loader(config: EmbeddingConfig) -> _FixtureModel:
        raise RuntimeError("fixture failure")

    failed = SentenceTransformerEmbeddingProvider(
        _config(), model_loader=fail_loader
    )
    with pytest.raises(BackendInitializationError):
        _ = failed.dimension

    mismatched = SentenceTransformerEmbeddingProvider(
        _config(), model_loader=lambda config: _FixtureModel(dimension=2)
    )
    with pytest.raises(BackendInitializationError, match="dimension"):
        _ = mismatched.dimension


def test_empty_document_batch_does_not_load_model() -> None:
    """An empty offline batch is a valid no-op without model initialization."""
    loaded = False

    def loader(config: EmbeddingConfig) -> _FixtureModel:
        nonlocal loaded
        loaded = True
        return _FixtureModel()

    provider = SentenceTransformerEmbeddingProvider(_config(), model_loader=loader)

    assert provider.embed_documents([], batch_size=4) == []
    assert loaded is False
