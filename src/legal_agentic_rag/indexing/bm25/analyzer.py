"""Deterministic Vietnamese-compatible lexical analysis for BM25."""

import re
import unicodedata

_TERM_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class UnicodeBM25Analyzer:
    """Normalize lexical terms while preserving Vietnamese accents and numbers."""

    name = "unicode_word_casefold_v1"

    def analyze(self, text: str) -> list[str]:
        """Return NFC-normalized, case-insensitive word and number terms."""
        normalized = unicodedata.normalize("NFC", text).casefold()
        return [match.group(0) for match in _TERM_PATTERN.finditer(normalized)]
