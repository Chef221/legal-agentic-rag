"""Unit tests for the bounded multilingual cross-encoder reranker."""

import numpy as np
import pytest

from legal_agentic_rag.configuration import RerankerConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ModelError,
    RetrievalError,
)
from legal_agentic_rag.reranking import (
    CrossEncoderReranker,
    build_legal_rerank_text,
)
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
        metadata={
            "document_title": f"Văn bản phạm vi {chunk_id}",
            "document_number": f"0{rank}/2026/QH",
            "document_type": "Luật",
            "issuing_authority": "Quốc hội",
            "legal_field": "Lao động",
            "effect_status": "Còn hiệu lực",
            "effective_date": "2026-01-01",
            "source_url": "https://example.test/not-for-model-input",
            "raw_dataset_field": "must-not-leak",
            "structure": {
                "article_number": str(rank),
                "article_title": f"Phạm vi {chunk_id}",
                "clause_numbers": ["1", "2"],
            },
        },
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
    first_pair = model.calls[0][0][0]
    assert first_pair[0] == "rewritten question"
    assert "Tên văn bản: Văn bản phạm vi first" in first_pair[1]
    assert "Số ký hiệu: 01/2026/QH" in first_pair[1]
    assert "Tình trạng hiệu lực: Còn hiệu lực" in first_pair[1]
    assert "Điều: 1" in first_pair[1]
    assert "Khoản: 1, 2" in first_pair[1]
    assert first_pair[1].endswith("Nội dung:\nLegal text first")
    assert "source_url" not in first_pair[1]
    assert "must-not-leak" not in first_pair[1]
    assert model.calls[0][1]["batch_size"] == 8
    assert model.calls[0][1]["apply_softmax"] is False
    identity = model.calls[0][1]["activation_fn"]
    assert callable(identity) and identity("raw") == "raw"
    assert reranker.provider_name == "sentence-transformers"
    assert reranker.provider_version == "fixture-version"
    assert reranker.model_revision == "1427fd652930e4ba29e8149678df786c240d8825"


def test_text_only_reranker_input_mode_preserves_reference_pair() -> None:
    """Explicit text-only mode supports controlled quality comparison."""
    model = _FixtureModel(np.asarray([0.5], dtype=np.float32))
    reranker = CrossEncoderReranker(
        RerankerConfig(input_mode="text_only"),
        model_loader=lambda config: model,
    )

    reranker.rerank(
        _query(top_k=1, candidate_k=1),
        [_hit("first", 1)],
    )

    assert model.calls[0][0] == [
        ("rewritten question", "Legal text first")
    ]


def test_qwen_reranker_loader_receives_pinned_custom_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider passes a legal retrieval instruction through its adapter."""
    captured: dict[str, object] = {}

    def fake_cross_encoder(model_name: str, **options: object) -> _FixtureModel:
        captured["model_name"] = model_name
        captured.update(options)
        return _FixtureModel([0.5])

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        fake_cross_encoder,
    )
    config = RerankerConfig(
        prompt_name="legal_retrieval",
        instruction="Given a Vietnamese legal question, rank relevant passages.",
    )

    CrossEncoderReranker._load_cross_encoder(config)

    assert captured["prompts"] == {
        "legal_retrieval": (
            "Given a Vietnamese legal question, rank relevant passages."
        )
    }
    assert captured["default_prompt_name"] == "legal_retrieval"
    assert captured["model_kwargs"] == {"torch_dtype": "float32"}


def test_legal_context_builder_uses_only_named_unified_metadata() -> None:
    """Arbitrary metadata and URLs do not enter the cross-encoder text."""
    value = build_legal_rerank_text(_hit("scope", 1))

    assert "Tên văn bản: Văn bản phạm vi scope" in value
    assert "Cơ quan ban hành: Quốc hội" in value
    assert "Tên điều: Phạm vi scope" in value
    assert "raw_dataset_field" not in value
    assert "must-not-leak" not in value
    assert "https://example.test" not in value


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
