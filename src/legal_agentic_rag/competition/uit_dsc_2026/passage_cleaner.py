"""Deterministic cleaning for the audited UIT DSC 2026 context passages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import re
import unicodedata

_CLEANER_VERSION = "1.2"
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
        "huongdan",
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
_NAKED_SCRIPT_BLOCK_START = re.compile(
    r"^\s*(?:"
    r"\$\(document\)\.ready\s*\(\s*function\s*\(\)\s*\{|"
    r"function\s+(?:DeleteDoc|ScrollNoiDungTA|ShowGuid|ShowPopupVBTraiNgiem)"
    r"\s*\([^)]*\)\s*\{|"
    r"(?:\.download\s+\.divAttachFiles|\.huong-dan-dieu-khoan(?:-d)?)"
    r"\s*(?:\{)?"
    r")",
    re.IGNORECASE,
)
_NAKED_SCRIPT_SINGLE_LINE = re.compile(
    r"^\s*(?:"
    r"var\s+(?:breadcrumbs|_scrollTopNoiDung(?:TA)?)\s*=|"
    r"_gaq\.push\s*\("
    r")",
    re.IGNORECASE,
)
_AUDITED_WEB_UI_LINES = frozenset(
    {
        "Bản án liên quan",
        "Hỏi đáp pháp luật",
        "PHÁP LUẬT DOANH NGHIỆP",
    }
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
        newline_normalized = "\r" in without_boilerplate
        without_naked_script, naked_script_removed = self._remove_naked_script(
            without_boilerplate
        )
        html_markup_removed = bool(_KNOWN_MARKUP.search(without_naked_script))
        visible = (
            self._remove_known_markup(without_naked_script)
            if html_markup_removed
            else without_naked_script
        )
        if naked_script_removed:
            visible = self._remove_audited_web_ui_lines(visible)
        unicode_normalized = visible != unicodedata.normalize("NFC", visible)
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
            "naked_script_block_start_patterns": [
                _NAKED_SCRIPT_BLOCK_START.pattern,
                _NAKED_SCRIPT_SINGLE_LINE.pattern,
            ],
            "audited_web_ui_lines": sorted(_AUDITED_WEB_UI_LINES),
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
    def _remove_naked_script(value: str) -> tuple[str, bool]:
        """Remove only balanced JavaScript blocks audited in official passages."""
        lines = value.splitlines()
        kept: list[str] = []
        index = 0
        removed = False
        while index < len(lines):
            line = lines[index]
            if _NAKED_SCRIPT_SINGLE_LINE.match(line):
                removed = True
                index += 1
                continue
            if not _NAKED_SCRIPT_BLOCK_START.match(line):
                kept.append(line)
                index += 1
                continue

            end = UitDsc2026PassageCleaner._balanced_script_end(lines, index)
            if end is None:
                # Fail closed: an unbalanced candidate may contain legal text later.
                kept.append(line)
                index += 1
                continue
            removed = True
            index = end + 1
        return "\n".join(kept), removed

    @staticmethod
    def _balanced_script_end(lines: list[str], start: int) -> int | None:
        depth = 0
        opened = False
        for index in range(start, len(lines)):
            delta, saw_open = UitDsc2026PassageCleaner._javascript_brace_delta(
                lines[index]
            )
            depth += delta
            opened = opened or saw_open
            if opened and depth <= 0:
                return index
        return None

    @staticmethod
    def _javascript_brace_delta(line: str) -> tuple[int, bool]:
        """Count structural braces while ignoring quoted JavaScript strings."""
        depth = 0
        saw_open = False
        quote: str | None = None
        escaped = False
        index = 0
        while index < len(line):
            character = line[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if quote is not None:
                if character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                index += 1
                continue
            if character in {"'", '"', "`"}:
                quote = character
            elif character == "/" and index + 1 < len(line) and line[index + 1] == "/":
                break
            elif character == "{":
                depth += 1
                saw_open = True
            elif character == "}":
                depth -= 1
            index += 1
        return depth, saw_open

    @staticmethod
    def _remove_audited_web_ui_lines(value: str) -> str:
        return "\n".join(
            line
            for line in value.splitlines()
            if line.strip() not in _AUDITED_WEB_UI_LINES
        )

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
