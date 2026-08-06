"""Tests for dependency-free and model-aware chunk tokenizers."""

from legal_agentic_rag.configuration.offline import EmbeddingConfig
from legal_agentic_rag.offline.chunking import (
    EmbeddingModelTokenizer,
    UnicodeWordTokenizer,
)


def test_unicode_tokenizer_preserves_vietnamese_numbers_and_punctuation() -> None:
    """Vietnamese text, amounts, and negation participate in token windows."""
    tokenizer = UnicodeWordTokenizer()
    text = "Không áp dụng mức 10.000.000 đồng; trừ trường hợp đặc biệt."

    fragments = tokenizer.split(text, max_tokens=8, overlap_tokens=2)

    assert tokenizer.count(text) > 8
    assert len(fragments) > 1
    assert all(tokenizer.count(fragment) <= 8 for fragment in fragments)
    assert "Không" in fragments[0]
    assert any("10" in fragment for fragment in fragments)
    assert any("trừ" in fragment for fragment in fragments)


class _WordpieceFixture:
    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        pieces = [piece for word in text.split() for piece in (word, f"##{word}")]
        return {"input_ids": [0, *range(len(pieces)), 1], "length": len(pieces) + 2}


def test_embedding_tokenizer_counts_prefix_and_splits_source_text() -> None:
    """Exact model budgeting preserves source substrings without decoding."""
    config = EmbeddingConfig(
        model_name="fixture/model",
        model_revision="fixture-revision",
        document_prefix="passage:",
    )
    tokenizer = EmbeddingModelTokenizer(
        config,
        tokenizer_loader=lambda _: _WordpieceFixture(),
    )
    text = "không áp dụng trong trường hợp đặc biệt"

    assert tokenizer.count(text) == 20
    fragments = tokenizer.split(text, max_tokens=10, overlap_tokens=1)

    assert len(fragments) > 1
    assert all(fragment in text for fragment in fragments)
    assert all(tokenizer.count(fragment) <= 10 for fragment in fragments)
    assert tokenizer.identity["tokenizer_model_revision"] == "fixture-revision"
