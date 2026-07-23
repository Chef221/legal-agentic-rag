"""Tests for the dependency-free baseline chunk tokenizer."""

from legal_agentic_rag.offline.chunking import UnicodeWordTokenizer


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
