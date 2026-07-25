"""Conservative HTML-to-text cleaning for normalized legal documents."""

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from html.parser import HTMLParser
import logging
import re
import unicodedata

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import HtmlCleaningConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity
from legal_agentic_rag.schemas.cleaning import HtmlCleaningResult
from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType

_LOGGER = logging.getLogger(__name__)
Clock = Callable[[], datetime]
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+")
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "caption",
        "dd",
        "dir",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "tr",
        "ul",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class _LegalTextParser(HTMLParser):
    """Collect visible text while preserving coarse legal layout boundaries."""

    def __init__(self, config: HtmlCleaningConfig) -> None:
        super().__init__(convert_charrefs=True)
        self._config = config
        self._parts: list[str] = []
        self._ignored_depth = 0
        self._table_cell_index = 0

    @property
    def text(self) -> str:
        """Return collected text before Unicode and whitespace normalization."""
        return "".join(self._parts)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Apply explicit noise rules and emit stable structural separators."""
        normalized_tag = tag.casefold()
        if self._ignored_depth:
            if normalized_tag not in _VOID_TAGS:
                self._ignored_depth += 1
            return
        if self._is_noise(normalized_tag, attrs):
            self._line_break()
            if normalized_tag not in _VOID_TAGS:
                self._ignored_depth = 1
            return
        if normalized_tag == "br":
            self._line_break()
        elif normalized_tag == "tr":
            self._line_break()
            self._table_cell_index = 0
        elif normalized_tag in {"td", "th"}:
            if self._table_cell_index:
                self._parts.append(" | ")
            self._table_cell_index += 1
        elif normalized_tag in _BLOCK_TAGS:
            self._line_break()

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Handle self-closing tags without changing ignored nesting depth."""
        if self._ignored_depth:
            return
        normalized_tag = tag.casefold()
        if self._is_noise(normalized_tag, attrs):
            self._line_break()
            return
        if normalized_tag in _BLOCK_TAGS or normalized_tag == "br":
            self._line_break()

    def handle_endtag(self, tag: str) -> None:
        """Close ignored subtrees or preserve visible block boundaries."""
        normalized_tag = tag.casefold()
        if self._ignored_depth:
            if normalized_tag not in _VOID_TAGS:
                self._ignored_depth -= 1
                if not self._ignored_depth:
                    self._line_break()
            return
        if normalized_tag in _BLOCK_TAGS:
            self._line_break()

    def handle_data(self, data: str) -> None:
        """Keep visible text exactly until the normalization pass."""
        if not self._ignored_depth:
            self._parts.append(data)

    def _is_noise(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> bool:
        if tag in self._config.remove_tags:
            return True
        attr_map = {name.casefold(): value or "" for name, value in attrs}
        if "hidden" in attr_map:
            return True
        if attr_map.get("aria-hidden", "").strip().casefold() == "true":
            return True
        inline_style = re.sub(r"\s+", "", attr_map.get("style", "").casefold())
        if "display:none" in inline_style or "visibility:hidden" in inline_style:
            return True
        class_tokens = {
            token.casefold() for token in attr_map.get("class", "").split()
        }
        element_id = attr_map.get("id", "").strip().casefold()
        return bool(class_tokens & self._config.noise_class_tokens) or (
            bool(element_id) and element_id in self._config.noise_id_tokens
        )

    def _line_break(self) -> None:
        if not self._parts or not self._parts[-1].endswith("\n"):
            self._parts.append("\n")


class LegalHtmlCleaner:
    """Clean normalized legal documents without interpreting legal structure."""

    def __init__(
        self,
        config: HtmlCleaningConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or HtmlCleaningConfig()
        self._clock = clock or (lambda: datetime.now(UTC))

    def clean_html(self, content_html: str) -> str:
        """Convert one HTML payload into deterministic, layout-aware plain text."""
        parser = _LegalTextParser(self._config)
        parser.feed(content_html)
        parser.close()
        return self._normalize_text(parser.text)

    def clean(
        self,
        *,
        documents: Iterable[LegalDocument],
        source_manifest: ArtifactManifest,
    ) -> HtmlCleaningResult:
        """Clean a complete normalized artifact and retain every document."""
        input_documents = list(documents)
        self._validate_input(input_documents, source_manifest)
        _LOGGER.info(
            "html_cleaning_started",
            extra={
                "dataset_name": source_manifest.dataset_name,
                "document_count": len(input_documents),
            },
        )
        cleaned_documents: list[LegalDocument] = []
        issues: list[AuditIssue] = []
        cleaned_count = 0
        missing_count = 0
        empty_count = 0
        for document in input_documents:
            clean_text: str | None = None
            if document.content_html is None:
                missing_count += 1
                issues.append(
                    self._issue(
                        "missing_content",
                        AuditSeverity.WARNING,
                        document.document_id,
                        "Document has no HTML content to clean",
                    )
                )
            else:
                candidate = self.clean_html(document.content_html)
                if candidate:
                    clean_text = candidate
                    cleaned_count += 1
                else:
                    empty_count += 1
                    issues.append(
                        self._issue(
                            "empty_clean_text",
                            AuditSeverity.ERROR,
                            document.document_id,
                            "HTML contained no visible legal text after cleaning",
                        )
                    )
            payload = document.model_dump(mode="python")
            payload["clean_text"] = clean_text
            cleaned_documents.append(LegalDocument.model_validate(payload))

        manifest = self._manifest(
            source_manifest=source_manifest,
            record_count=len(cleaned_documents),
            cleaned_count=cleaned_count,
            missing_count=missing_count,
            empty_count=empty_count,
            issue_count=len(issues),
        )
        result = HtmlCleaningResult(
            documents=cleaned_documents,
            issues=issues,
            manifest=manifest,
            input_document_count=len(input_documents),
            cleaned_document_count=cleaned_count,
            missing_content_count=missing_count,
            empty_output_count=empty_count,
        )
        _LOGGER.info(
            "html_cleaning_completed",
            extra={
                "dataset_name": source_manifest.dataset_name,
                "document_count": len(cleaned_documents),
                "cleaned_document_count": cleaned_count,
                "issue_count": len(issues),
            },
        )
        return result

    @staticmethod
    def _validate_input(
        documents: list[LegalDocument], source_manifest: ArtifactManifest
    ) -> None:
        if source_manifest.artifact_type != ArtifactType.NORMALIZED_DOCUMENTS:
            raise ArtifactCompatibilityError(
                "HTML cleaner requires a normalized-documents artifact"
            )
        if source_manifest.record_count != len(documents):
            raise DataValidationError(
                "Source manifest record count does not match input documents"
            )
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise DataValidationError("HTML cleaner input document IDs must be unique")

    def _manifest(
        self,
        *,
        source_manifest: ArtifactManifest,
        record_count: int,
        cleaned_count: int,
        missing_count: int,
        empty_count: int,
        issue_count: int,
    ) -> ArtifactManifest:
        warnings = list(source_manifest.warnings)
        if issue_count:
            warnings.append(f"HTML cleaning produced {issue_count} issues")
        return ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.CLEANED_DOCUMENTS,
            artifact_version=self._config.artifact_version,
            dataset_name=source_manifest.dataset_name,
            dataset_revision=source_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=record_count,
            processing_config_hash=self._config_hash(source_manifest),
            code_version=__version__,
            warnings=warnings,
            metadata={
                "source_artifact_type": source_manifest.artifact_type.value,
                "source_artifact_version": source_manifest.artifact_version,
                "source_processing_config_hash": (
                    source_manifest.processing_config_hash
                ),
                "cleaned_document_count": cleaned_count,
                "missing_content_count": missing_count,
                "empty_output_count": empty_count,
                "cleaning_issue_count": issue_count,
            },
        )

    def _config_hash(self, source_manifest: ArtifactManifest) -> str:
        payload = {
            "source_processing_config_hash": source_manifest.processing_config_hash,
            "html_cleaning": self._config,
        }
        return canonical_sha256(payload)

    @staticmethod
    def _issue(
        issue_type: str,
        severity: AuditSeverity,
        document_id: str,
        message: str,
    ) -> AuditIssue:
        return AuditIssue(
            issue_type=issue_type,
            severity=severity,
            record_id=document_id,
            message=message,
            metadata={"stage": "html_cleaning"},
        )

    def _normalize_text(self, value: str) -> str:
        normalized = unicodedata.normalize(
            self._config.unicode_normalization_form, value
        )
        visible_characters: list[str] = []
        for character in normalized:
            if character in {"\n", "\r", "\t"}:
                visible_characters.append(character)
                continue
            category = unicodedata.category(character)
            if category in {"Cc", "Cf"}:
                continue
            visible_characters.append(" " if category == "Zs" else character)
        visible = "".join(visible_characters).replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        lines: list[str] = []
        previous_was_blank = True
        for raw_line in visible.splitlines():
            line = _HORIZONTAL_WHITESPACE.sub(" ", raw_line).strip()
            if line:
                lines.append(line)
                previous_was_blank = False
            elif not previous_was_blank:
                lines.append("")
                previous_was_blank = True
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)
