"""Unit tests for the bounded multilingual cross-encoder reranker."""

import numpy as np
import pytest

from legal_agentic_rag.configuration import RerankerConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ModelError,
    RetrievalError,
)
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.schemas import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalStrategy,
    RetrievalTrace,
)


class _FixtureModel:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object:
        self.calls.append((inputs, kwargs))
        return self.scores


def _query(*, top_k: int = 2, candidate_k: int = 3) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-rerank",
        original_question="Original question",
        normalized_question="normalized question",
        rewritten_question="rewritten question",
        top_k=top_k,
        candidate_k=candidate_k,
        requested_strategy=RetrievalStrategy.RERANK,
    )


def _hit(chunk_id: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=1 / (60 + rank),
        strategy=RetrievalStrategy.HYBRID,
        text=f"Legal text {chunk_id}",
        metadata={"article_number": str(rank)},
        retrieval_trace=RetrievalTrace(rrf_score=1 / (60 + rank)),
    )


def test_cross_encoder_scores_sorts_and_preserves_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw logits determine rank while candidate metadata and RRF trace survive."""
    monkeypatch.setattr(
        "legal_agentic_rag.reranking.cross_encoder.version",
        lambda package_name: "fixture-version",
    )
    model = _FixtureModel(np.asarray([0.1, 0.9, 0.9], dtype=np.float32))
    reranker = CrossEncoderReranker(model_loader=lambda config: model)
    candidates = [_hit("first", 1), _hit("second", 2), _hit("third", 3)]

    response = reranker.rerank(_query(), candidates)

    assert [hit.chunk_id for hit in response.hits] == ["second", "third"]
    assert [hit.rank for hit in response.hits] == [1, 2]
    assert response.strategy == RetrievalStrategy.RERANK
    assert response.hits[0].score == pytest.approx(0.9)
    assert response.hits[0].retrieval_trace.rrf_score == candidates[1].score
    assert response.hits[0].retrieval_trace.reranker_score == pytest.approx(0.9)
    assert response.hits[0].metadata == candidates[1].metadata
    assert model.calls[0][0][0] == ("rewritten question", "Legal text first")
    assert model.calls[0][1]["batch_size"] == 8
    assert model.calls[0][1]["apply_softmax"] is False
    identity = model.calls[0][1]["activation_fn"]
    assert callable(identity) and identity("raw") == "raw"
    assert reranker.provider_name == "sentence-transformers"
    assert reranker.provider_version == "fixture-version"
    assert reranker.model_revision == "1427fd652930e4ba29e8149678df786c240d8825"


def test_empty_candidates_do_not_load_model() -> None:
    """An empty retrieved set returns a warning without model initialization."""
    loaded = False

    def loader(config: RerankerConfig) -> _FixtureModel:
        nonlocal loaded
        loaded = True
        return _FixtureModel([])

    response = CrossEncoderReranker(model_loader=loader).rerank(
        _query(top_k=1, candidate_k=1), []
    )

    assert response.hits == []
    assert response.warnings == ["no_rerank_candidates"]
    assert loaded is False


def test_reranker_rejects_unbounded_duplicates_and_wrong_strategy() -> None:
    """Direct callers cannot bypass candidate bounds or identity requirements."""
    reranker = CrossEncoderReranker(
        RerankerConfig(max_candidates=2),
        model_loader=lambda config: _FixtureModel([0.1, 0.2]),
    )
    candidates = [_hit("one", 1), _hit("two", 2), _hit("three", 3)]
    with pytest.raises(RetrievalError, match="limit"):
        reranker.rerank(_query(candidate_k=3), candidates)
    with pytest.raises(RetrievalError, match="duplicate"):
        reranker.rerank(
            _query(top_k=1, candidate_k=2),
            [_hit("same", 1), _hit("same", 2)],
        )
    wrong = _query().model_copy(
        update={"requested_strategy": RetrievalStrategy.HYBRID}
    )
    with pytest.raises(RetrievalError, match="non-rerank"):
        reranker.rerank(wrong, [])


@pytest.mark.parametrize(
    "scores, message",
    [
        ([[0.1, 0.2]], "shape"),
        ([float("nan"), 0.2], "non-finite"),
    ],
)
def test_reranker_rejects_invalid_model_outputs(scores: object, message: str) -> None:
    """Malformed score tensors are reported as model errors."""
    reranker = CrossEncoderReranker(
        model_loader=lambda config: _FixtureModel(scores)
    )
    with pytest.raises(ModelError, match=message):
        reranker.rerank(
            _query(top_k=1, candidate_k=2),
            [_hit("one", 1), _hit("two", 2)],
        )


def test_reranker_classifies_loading_and_prediction_failures() -> None:
    """Initialization and inference failures use distinct domain exceptions."""
    def fail_loader(config: RerankerConfig) -> _FixtureModel:
        raise RuntimeError("load failed")

    with pytest.raises(BackendInitializationError):
        CrossEncoderReranker(model_loader=fail_loader).rerank(
            _query(top_k=1, candidate_k=1), [_hit("one", 1)]
        )

    class _FailingModel:
        def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object:
            raise RuntimeError("predict failed")

    with pytest.raises(ModelError):
        CrossEncoderReranker(model_loader=lambda config: _FailingModel()).rerank(
            _query(top_k=1, candidate_k=1), [_hit("one", 1)]
        )
