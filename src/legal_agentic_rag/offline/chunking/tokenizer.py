"""Lightweight deterministic tokenizer used only by baseline chunking."""

import re

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class UnicodeWordTokenizer:
    """Count Unicode words and punctuation without external model packages."""

    name = "unicode_word_v1"

    def count(self, text: str) -> int:
        """Return the deterministic token count for one text value."""
        return sum(1 for _ in _TOKEN_PATTERN.finditer(text))

    def split(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split text at token spans with a bounded sliding-window overlap."""
        matches = list(_TOKEN_PATTERN.finditer(text))
        if not matches:
            return []
        if len(matches) <= max_tokens:
            return [text.strip()]
        step = max_tokens - overlap_tokens
        fragments: list[str] = []
        start = 0
        while start < len(matches):
            end = min(start + max_tokens, len(matches))
            char_start = matches[start].start()
            char_end = matches[end - 1].end()
            fragment = text[char_start:char_end].strip()
            if fragment:
                fragments.append(fragment)
            if end == len(matches):
                break
            start += step
        return fragments
