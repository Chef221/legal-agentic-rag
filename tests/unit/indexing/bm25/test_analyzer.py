"""Tests for conservative Vietnamese BM25 lexical analysis."""

from legal_agentic_rag.indexing.bm25 import UnicodeBM25Analyzer


def test_analyzer_preserves_accents_numbers_and_negation() -> None:
    """Analyzer keeps legally meaningful Vietnamese terms and numbers."""
    terms = UnicodeBM25Analyzer().analyze(
        "KHÔNG áp dụng mức 10.000.000 đồng, Điều 5!"
    )

    assert terms == [
        "không",
        "áp",
        "dụng",
        "mức",
        "10",
        "000",
        "000",
        "đồng",
        "điều",
        "5",
    ]


def test_analyzer_normalizes_canonical_unicode_and_case() -> None:
    """Equivalent Unicode and case forms produce identical terms."""
    analyzer = UnicodeBM25Analyzer()

    assert analyzer.analyze("HIỆU LỰC") == analyzer.analyze("HIỆU LỰC")

