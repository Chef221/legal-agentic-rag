"""Deterministic cleaning for the audited UIT DSC 2026 context passages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import re
import unicodedata

_CLEANER_VERSION = "1.0"
_TVPL_PRO_NOTICE = (
    "Bạn phải đăng nhập hoặc đăng ký Thành Viên TVPL Pro để sử dụng được đầy "
    "đủ các tiện ích gia tăng liên quan đến nội dung TCVN.Mọi chi tiết xin "
    "liên hệ: ĐT: (028) 3930 3279 DĐ: 0906 22 99 66"
)
_TVPL_PRO_WITH_PLACEHOLDER = re.compile(
    rf"\.{{5,}}\s*{re.escape(_TVPL_PRO_NOTICE)}"
)
_IGNORED_SUBTREES = frozenset(
    {"iframe", "noscript", "object", "option", "script", "style", "template"}
)
_BLOCK_TAGS = frozenset(
    {
        "div",
        "h1",
        "h2",
        "h3",
        "li",
        "ol",
        "p",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_INLINE_TAGS = frozenset({"a", "b", "font", "i", "span", "strong"})
_VOID_TAGS = frozenset({"br", "hr", "img", "input", "meta"})
_KNOWN_TAGS = _IGNORED_SUBTREES | _BLOCK_TAGS | _INLINE_TAGS | _VOID_TAGS
_KNOWN_MARKUP = re.compile(
    rf"<\s*/?\s*(?:{'|'.join(sorted(_KNOWN_TAGS))})(?:\s[^<>]*?)?/?>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PassageCleaningResult:
    """Clean text plus the exact transformations applied to one passage."""

    text: str
    modified: bool
    html_markup_removed: bool
    boilerplate_occurrence_count: int
    unicode_normalized: bool
    newline_normalized: bool


class _KnownMarkupParser(HTMLParser):
    """Remove only audited HTML tags while retaining unknown legal brackets."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []
        self._ignored_depth = 0

    @property
    def text(self) -> str:
        """Return visible text collected from the organizer passage."""
        return "".join(self._parts)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Handle audited markup without treating arbitrary brackets as HTML."""
        normalized = tag.casefold()
        if self._ignored_depth:
            if normalized not in _VOID_TAGS:
                self._ignored_depth += 1
            return
        if normalized in _IGNORED_SUBTREES:
            self._line_break()
            self._ignored_depth = 1
            return
        if normalized not in _KNOWN_TAGS:
            self._parts.append(self.get_starttag_text())
            return
        if normalized == "img":
            alt = next((value for name, value in attrs if name.casefold() == "alt"), None)
            if alt and alt.strip():
                self._parts.append(alt)
        if normalized == "br" or normalized in _BLOCK_TAGS:
            self._line_break()

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Handle self-closing audited markup and preserve unknown markup."""
        normalized = tag.casefold()
        if self._ignored_depth:
            return
        if normalized not in _KNOWN_TAGS:
            self._parts.append(self.get_starttag_text())
            return
        if normalized == "img":
            alt = next((value for name, value in attrs if name.casefold() == "alt"), None)
            if alt and alt.strip():
                self._parts.append(alt)
        if normalized == "br" or normalized in _BLOCK_TAGS:
            self._line_break()

    def handle_endtag(self, tag: str) -> None:
        """Close ignored markup or emit a visible block boundary."""
        normalized = tag.casefold()
        if self._ignored_depth:
            self._ignored_depth -= 1
            if not self._ignored_depth:
                self._line_break()
            return
        if normalized not in _KNOWN_TAGS:
            self._parts.append(f"</{tag}>")
        elif normalized in _BLOCK_TAGS:
            self._line_break()

    def handle_data(self, data: str) -> None:
        """Retain visible legal text outside explicitly ignored subtrees."""
        if not self._ignored_depth:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        """Decode standard HTML entities in passages containing real markup."""
        if not self._ignored_depth:
            self._parts.append(unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        """Decode numeric HTML entities in passages containing real markup."""
        if not self._ignored_depth:
            self._parts.append(unescape(f"&#{name};"))

    def _line_break(self) -> None:
        if not self._parts or not self._parts[-1].endswith("\n"):
            self._parts.append("\n")


class UitDsc2026PassageCleaner:
    """Clean only source artifacts confirmed by the official corpus audit."""

    version = _CLEANER_VERSION

    def clean(self, passage: str) -> PassageCleaningResult:
        """Return deterministic clean text without interpreting legal meaning."""
        without_boilerplate, occurrence_count = self._remove_boilerplate(passage)
        html_markup_removed = bool(_KNOWN_MARKUP.search(without_boilerplate))
        visible = (
            self._remove_known_markup(without_boilerplate)
            if html_markup_removed
            else without_boilerplate
        )
        unicode_normalized = visible != unicodedata.normalize("NFC", visible)
        newline_normalized = "\r" in visible
        clean_text = self._normalize_text(visible)
        return PassageCleaningResult(
            text=clean_text,
            modified=clean_text != passage,
            html_markup_removed=html_markup_removed,
            boilerplate_occurrence_count=occurrence_count,
            unicode_normalized=unicode_normalized,
            newline_normalized=newline_normalized,
        )

    @staticmethod
    def policy_identity() -> dict[str, object]:
        """Return a stable payload included in cleaned-artifact lineage."""
        return {
            "cleaner": "uit_dsc_2026_passage",
            "cleaner_version": _CLEANER_VERSION,
            "unicode_normalization": "NFC",
            "known_html_tags": sorted(_KNOWN_TAGS),
            "ignored_html_subtrees": sorted(_IGNORED_SUBTREES),
            "boilerplate_sha256": sha256(
                _TVPL_PRO_NOTICE.encode("utf-8")
            ).hexdigest(),
        }

    @staticmethod
    def _remove_boilerplate(value: str) -> tuple[str, int]:
        without_placeholder, prefixed_count = _TVPL_PRO_WITH_PLACEHOLDER.subn(
            " ", value
        )
        plain_count = without_placeholder.count(_TVPL_PRO_NOTICE)
        return (
            without_placeholder.replace(_TVPL_PRO_NOTICE, " "),
            prefixed_count + plain_count,
        )

    @staticmethod
    def _remove_known_markup(value: str) -> str:
        parser = _KnownMarkupParser()
        parser.feed(value)
        parser.close()
        return parser.text

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        characters: list[str] = []
        for character in normalized:
            if character in {"\n", "\r", "\t"}:
                characters.append(character)
                continue
            category = unicodedata.category(character)
            if category in {"Cc", "Cf"}:
                continue
            characters.append(" " if category == "Zs" else character)
        visible = "".join(characters).replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        previous_blank = True
        for raw_line in visible.splitlines():
            line = raw_line.strip()
            if line:
                lines.append(line)
                previous_blank = False
            elif not previous_blank:
                lines.append("")
                previous_blank = True
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)
