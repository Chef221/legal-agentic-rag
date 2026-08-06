"""Conservative line-based parsing of Vietnamese legal structure markers."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import logging
import re

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import LegalStructureParserConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity
from legal_agentic_rag.schemas.legal_documents import (
    LegalBlock,
    LegalBlockType,
    LegalDocument,
    LegalStructure,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.parsing import (
    DocumentParsingDiagnostic,
    LegalStructureParsingResult,
)

Clock = Callable[[], datetime]
_LOGGER = logging.getLogger(__name__)
_NUMBER = r"(?:\d+[A-ZĐ]?|[IVXLCDM]+)"
_CLAUSE_NUMBER = r"\d+[A-ZĐ]?"
_DELIMITER = r"[.:\-–—]"
_PART_PATTERN = re.compile(
    rf"^PHẦN(?:\s+THỨ)?\s+(?P<number>[^\s.:\-–—]+)\s*"
    rf"(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_CHAPTER_PATTERN = re.compile(
    rf"^CHƯƠNG\s+(?P<number>{_NUMBER})\s*"
    rf"(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_SECTION_PATTERN = re.compile(
    rf"^MỤC\s+(?P<number>{_NUMBER})\s*"
    rf"(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_SUBSECTION_PATTERN = re.compile(
    rf"^TIỂU\s+MỤC\s+(?P<number>{_NUMBER})\s*"
    rf"(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_ARTICLE_PATTERN = re.compile(
    rf"^ĐIỀU\s+(?P<number>{_NUMBER})(?!\w)\s*"
    rf"(?:(?:{_DELIMITER})\s*)?(?P<title>.*)$",
    re.IGNORECASE,
)
_EXPLICIT_CLAUSE_PATTERN = re.compile(
    rf"^KHOẢN\s+(?P<number>{_CLAUSE_NUMBER})\s*"
    rf"(?:(?:{_DELIMITER})\s*)?(?P<body>.*)$",
    re.IGNORECASE,
)
_CLAUSE_PATTERN = re.compile(
    rf"^(?P<number>{_CLAUSE_NUMBER})(?P<delimiter>[.)])\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_EXPLICIT_POINT_PATTERN = re.compile(
    rf"^ĐIỂM\s+(?P<number>[A-ZĐ])\s*"
    rf"(?:(?:[.):\-–—])\s*)?(?P<body>.*)$",
    re.IGNORECASE,
)
_POINT_PATTERN = re.compile(
    r"^(?P<number>[A-ZĐ])[.)]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_APPENDIX_PATTERN = re.compile(
    rf"^PHỤ\s+LỤC(?:\s+(?P<number>{_NUMBER}))?\s*"
    rf"(?:(?:{_DELIMITER})\s*)?(?P<title>.*)$",
    re.IGNORECASE,
)
_POTENTIAL_MARKER_PATTERN = re.compile(
    r"^(?:PHẦN|CHƯƠNG|TIỂU\s+MỤC|MỤC|ĐIỀU|KHOẢN|ĐIỂM|PHỤ\s+LỤC)\s+"
    r"(?:\d|[IVXLCDM]+\b|[A-ZĐ][.):])",
    re.IGNORECASE,
)
_BARE_MARKER_PATTERN = re.compile(
    r"^(?:PHẦN|CHƯƠNG|TIỂU\s+MỤC|MỤC|ĐIỀU|KHOẢN|ĐIỂM|PHỤ\s+LỤC)$",
    re.IGNORECASE,
)
_STRUCTURE_TYPES = frozenset(
    {
        LegalBlockType.PART,
        LegalBlockType.CHAPTER,
        LegalBlockType.SECTION,
        LegalBlockType.SUBSECTION,
        LegalBlockType.ARTICLE,
        LegalBlockType.CLAUSE,
        LegalBlockType.POINT,
        LegalBlockType.APPENDIX,
    }
)
_TITLE_BEARING_TYPES = frozenset(
    {
        LegalBlockType.PART,
        LegalBlockType.CHAPTER,
        LegalBlockType.SECTION,
        LegalBlockType.SUBSECTION,
        LegalBlockType.ARTICLE,
        LegalBlockType.APPENDIX,
    }
)


@dataclass(frozen=True)
class _Marker:
    block_type: LegalBlockType
    level: int
    number: str | None
    label: str
    title: str | None = None


@dataclass
class _DraftBlock:
    block_type: LegalBlockType
    number: str | None
    title: str | None
    lines: list[str]
    parent_index: int | None
    structure: LegalStructure

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class ParsedLegalDocument:
    """Memory-bounded parser output for one cleaned legal document."""

    blocks: list[LegalBlock]
    diagnostic: DocumentParsingDiagnostic
    issues: list[AuditIssue] = field(default_factory=list)


class LegalStructureParser:
    """Parse cleaned legal text into deterministic non-overlapping blocks."""

    def __init__(
        self,
        config: LegalStructureParserConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or LegalStructureParserConfig()
        self._clock = clock or (lambda: datetime.now(UTC))

    def parse(
        self,
        *,
        documents: Iterable[LegalDocument],
        source_manifest: ArtifactManifest,
    ) -> LegalStructureParsingResult:
        """Parse every document in a compatible cleaned-documents artifact."""
        input_documents = list(documents)
        self._validate_input(input_documents, source_manifest)
        _LOGGER.info(
            "legal_structure_parsing_started",
            extra={
                "dataset_name": source_manifest.dataset_name,
                "document_count": len(input_documents),
            },
        )
        blocks: list[LegalBlock] = []
        diagnostics: list[DocumentParsingDiagnostic] = []
        issues: list[AuditIssue] = []
        for document in input_documents:
            parsed = self._parse_document(document)
            blocks.extend(parsed.blocks)
            diagnostics.append(parsed.diagnostic)
            issues.extend(parsed.issues)

        parsed_count = sum(document.clean_text is not None for document in input_documents)
        structured_count = sum(item.has_recognized_structure for item in diagnostics)
        missing_count = len(input_documents) - parsed_count
        unstructured_count = parsed_count - structured_count
        manifest = self.build_manifest(
            source_manifest=source_manifest,
            block_count=len(blocks),
            document_count=len(diagnostics),
            issue_count=len(issues),
            parsed_count=parsed_count,
            missing_count=missing_count,
            structured_count=structured_count,
            unstructured_count=unstructured_count,
            source_characters=sum(
                item.source_non_whitespace_characters for item in diagnostics
            ),
            covered_characters=sum(
                item.covered_non_whitespace_characters for item in diagnostics
            ),
        )
        result = LegalStructureParsingResult(
            documents=input_documents,
            blocks=blocks,
            diagnostics=diagnostics,
            issues=issues,
            manifest=manifest,
            input_document_count=len(input_documents),
            parsed_document_count=parsed_count,
            missing_clean_text_count=missing_count,
            structured_document_count=structured_count,
            unstructured_document_count=unstructured_count,
        )
        _LOGGER.info(
            "legal_structure_parsing_completed",
            extra={
                "dataset_name": source_manifest.dataset_name,
                "document_count": len(input_documents),
                "block_count": len(blocks),
                "structured_document_count": structured_count,
                "issue_count": len(issues),
            },
        )
        return result

    def parse_document(self, document: LegalDocument) -> ParsedLegalDocument:
        """Parse one document without retaining any corpus-level state."""
        return self._parse_document(document)

    def _parse_document(self, document: LegalDocument) -> ParsedLegalDocument:
        clean_text = document.clean_text
        if clean_text is None:
            return ParsedLegalDocument(
                blocks=[],
                diagnostic=DocumentParsingDiagnostic(
                    document_id=document.document_id,
                    block_count=0,
                    recognized_structure_count=0,
                    source_non_whitespace_characters=0,
                    covered_non_whitespace_characters=0,
                    text_coverage=0.0,
                    has_recognized_structure=False,
                ),
                issues=[
                    self._issue(
                        "missing_clean_text",
                        AuditSeverity.WARNING,
                        document.document_id,
                        "Document has no clean text to parse",
                    )
                ],
            )

        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        drafts: list[_DraftBlock] = []
        active: dict[int, int] = {}
        issues: list[AuditIssue] = []
        line_index = 0
        while line_index < len(lines):
            line = lines[line_index]
            if self._is_table_row(line):
                self._append_table_row(line, drafts, active)
                line_index += 1
                continue
            active.pop(8, None)
            marker_source_lines = [line]
            if (
                _BARE_MARKER_PATTERN.fullmatch(line)
                and line_index + 1 < len(lines)
            ):
                joined = f"{line} {lines[line_index + 1]}"
                joined_marker = self._classify_marker(joined, drafts, active)
                if joined_marker is not None:
                    marker_source_lines.append(lines[line_index + 1])
                    line_index += 1
                    marker = joined_marker
                else:
                    marker = self._classify_marker(line, drafts, active)
            else:
                marker = self._classify_marker(line, drafts, active)
            if marker is not None:
                title_line: str | None = None
                if (
                    marker.block_type in _TITLE_BEARING_TYPES
                    and marker.title is None
                    and line_index + 1 < len(lines)
                    and self._looks_like_title(lines[line_index + 1])
                ):
                    title_line = lines[line_index + 1]
                    marker = _Marker(
                        block_type=marker.block_type,
                        level=marker.level,
                        number=marker.number,
                        label=marker.label,
                        title=title_line,
                    )
                    line_index += 1
                self._start_marker_block(
                    marker,
                    (
                        [*marker_source_lines, title_line]
                        if title_line is not None
                        else marker_source_lines
                    ),
                    drafts,
                    active,
                )
            else:
                if (
                    self._config.emit_unrecognized_marker_issues
                    and _POTENTIAL_MARKER_PATTERN.match(line)
                ):
                    issues.append(
                        self._issue(
                            "unrecognized_structure_marker",
                            AuditSeverity.WARNING,
                            document.document_id,
                            "Potential legal marker was preserved as ordinary text",
                            metadata={"line_index": line_index},
                        )
                    )
                self._append_ordinary_line(line, drafts, active)
            line_index += 1

        blocks = self._build_blocks(document.document_id, drafts)
        source_characters = self._non_whitespace_count(clean_text)
        covered_characters = sum(
            self._non_whitespace_count(block.text) for block in blocks
        )
        if covered_characters != source_characters:
            issues.append(
                self._issue(
                    "text_coverage_gap",
                    AuditSeverity.ERROR,
                    document.document_id,
                    "Parsed blocks do not cover all non-whitespace source text",
                    metadata={
                        "source_non_whitespace_characters": source_characters,
                        "covered_non_whitespace_characters": covered_characters,
                    },
                )
            )
        recognized_count = sum(
            block.block_type in _STRUCTURE_TYPES for block in blocks
        )
        if not recognized_count:
            issues.append(
                self._issue(
                    "no_legal_structure",
                    AuditSeverity.INFO,
                    document.document_id,
                    "No explicit legal hierarchy marker was recognized",
                )
            )
        coverage = covered_characters / source_characters if source_characters else 0.0
        return ParsedLegalDocument(
            blocks=blocks,
            diagnostic=DocumentParsingDiagnostic(
                document_id=document.document_id,
                block_count=len(blocks),
                recognized_structure_count=recognized_count,
                source_non_whitespace_characters=source_characters,
                covered_non_whitespace_characters=covered_characters,
                text_coverage=coverage,
                has_recognized_structure=bool(recognized_count),
            ),
            issues=issues,
        )

    def _classify_marker(
        self,
        line: str,
        drafts: list[_DraftBlock],
        active: Mapping[int, int],
    ) -> _Marker | None:
        patterns = (
            (_PART_PATTERN, LegalBlockType.PART, 1, "Phần"),
            (_CHAPTER_PATTERN, LegalBlockType.CHAPTER, 2, "Chương"),
            (_SUBSECTION_PATTERN, LegalBlockType.SUBSECTION, 4, "Tiểu mục"),
            (_SECTION_PATTERN, LegalBlockType.SECTION, 3, "Mục"),
            (_ARTICLE_PATTERN, LegalBlockType.ARTICLE, 5, "Điều"),
            (_APPENDIX_PATTERN, LegalBlockType.APPENDIX, 1, "Phụ lục"),
        )
        for pattern, block_type, level, prefix in patterns:
            match = pattern.match(line)
            if match is not None:
                number = self._group(match, "number")
                title = self._group(match, "title")
                label = prefix if number is None else f"{prefix} {number}"
                return _Marker(block_type, level, number, label, title)

        has_article = any(
            drafts[index].block_type == LegalBlockType.ARTICLE
            for index in active.values()
        )
        if not has_article:
            return None
        explicit_clause = _EXPLICIT_CLAUSE_PATTERN.match(line)
        if explicit_clause is not None:
            number = self._group(explicit_clause, "number")
            return _Marker(
                LegalBlockType.CLAUSE,
                6,
                number,
                f"Khoản {number}",
            )
        implicit_clause = _CLAUSE_PATTERN.match(line)
        if (
            implicit_clause is not None
            and self._is_valid_implicit_clause(implicit_clause)
        ):
            number = self._group(implicit_clause, "number")
            return _Marker(
                LegalBlockType.CLAUSE,
                6,
                number,
                f"Khoản {number}",
            )
        for pattern in (_EXPLICIT_POINT_PATTERN, _POINT_PATTERN):
            match = pattern.match(line)
            if match is not None:
                number = self._group(match, "number")
                return _Marker(
                    LegalBlockType.POINT,
                    7,
                    number,
                    f"Điểm {number}",
                )
        return None

    @staticmethod
    def _is_valid_implicit_clause(match: re.Match[str]) -> bool:
        """Reject years, decimals, and tariff codes masquerading as clauses."""
        body = match.group("body").strip()
        if not body:
            return False
        return not (
            match.group("delimiter") == "."
            and body[0].isdigit()
        )

    def _start_marker_block(
        self,
        marker: _Marker,
        lines: list[str],
        drafts: list[_DraftBlock],
        active: dict[int, int],
    ) -> None:
        for level in [level for level in active if level >= marker.level]:
            active.pop(level)
        parent_index = active[max(active)] if active else None
        draft_index = len(drafts)
        active[marker.level] = draft_index
        drafts.append(
            _DraftBlock(
                block_type=marker.block_type,
                number=marker.number,
                title=marker.title,
                lines=lines,
                parent_index=parent_index,
                structure=self._structure_snapshot(drafts, active, marker),
            )
        )

    def _append_ordinary_line(
        self,
        line: str,
        drafts: list[_DraftBlock],
        active: dict[int, int],
    ) -> None:
        if not active:
            drafts.append(
                _DraftBlock(
                    block_type=LegalBlockType.DOCUMENT,
                    number=None,
                    title=None,
                    lines=[line],
                    parent_index=None,
                    structure=LegalStructure(),
                )
            )
            active[0] = len(drafts) - 1
            return
        drafts[active[max(active)]].lines.append(line)

    def _append_table_row(
        self,
        line: str,
        drafts: list[_DraftBlock],
        active: dict[int, int],
    ) -> None:
        if 8 in active:
            drafts[active[8]].lines.append(line)
            return
        parent_index = active[max(active)] if active else None
        structure = (
            drafts[parent_index].structure.model_copy(deep=True)
            if parent_index is not None
            else LegalStructure()
        )
        drafts.append(
            _DraftBlock(
                block_type=LegalBlockType.TABLE,
                number=None,
                title=None,
                lines=[line],
                parent_index=parent_index,
                structure=structure,
            )
        )
        active[8] = len(drafts) - 1

    @staticmethod
    def _structure_snapshot(
        drafts: list[_DraftBlock],
        active: Mapping[int, int],
        marker: _Marker,
    ) -> LegalStructure:
        entries: list[tuple[LegalBlockType, str | None, str, str | None]] = []
        for level in sorted(active):
            if level == marker.level:
                entries.append(
                    (marker.block_type, marker.number, marker.label, marker.title)
                )
            else:
                draft = drafts[active[level]]
                label_prefix = {
                    LegalBlockType.DOCUMENT: "Văn bản",
                    LegalBlockType.PART: "Phần",
                    LegalBlockType.CHAPTER: "Chương",
                    LegalBlockType.SECTION: "Mục",
                    LegalBlockType.SUBSECTION: "Tiểu mục",
                    LegalBlockType.ARTICLE: "Điều",
                    LegalBlockType.CLAUSE: "Khoản",
                    LegalBlockType.POINT: "Điểm",
                    LegalBlockType.APPENDIX: "Phụ lục",
                }.get(draft.block_type, draft.block_type.value)
                label = (
                    label_prefix
                    if draft.number is None
                    else f"{label_prefix} {draft.number}"
                )
                entries.append((draft.block_type, draft.number, label, draft.title))
        values: dict[str, object] = {"structure_path": []}
        for block_type, number, label, title in entries:
            if block_type == LegalBlockType.PART:
                values["part"] = label
            elif block_type == LegalBlockType.CHAPTER:
                values["chapter"] = label
            elif block_type == LegalBlockType.SECTION:
                values["section"] = label
            elif block_type == LegalBlockType.SUBSECTION:
                values["subsection"] = label
            elif block_type == LegalBlockType.ARTICLE:
                values["article_number"] = number
                values["article_title"] = title
            elif block_type == LegalBlockType.CLAUSE and number is not None:
                values["clause_numbers"] = [number]
            elif block_type == LegalBlockType.POINT and number is not None:
                values["point_numbers"] = [number]
            if block_type != LegalBlockType.DOCUMENT:
                values["structure_path"].append(label)  # type: ignore[union-attr]
        return LegalStructure.model_validate(values)

    def _looks_like_title(self, line: str) -> bool:
        if not line or len(line) > self._config.maximum_title_characters:
            return False
        if len(line.split()) > self._config.maximum_title_words:
            return False
        if self._is_table_row(line):
            return False
        if _BARE_MARKER_PATTERN.fullmatch(line):
            return False
        if line.endswith((".", ";", ":")):
            return False
        if any(
            pattern.match(line)
            for pattern in (
                _PART_PATTERN,
                _CHAPTER_PATTERN,
                _SUBSECTION_PATTERN,
                _SECTION_PATTERN,
                _ARTICLE_PATTERN,
                _APPENDIX_PATTERN,
                _EXPLICIT_CLAUSE_PATTERN,
                _CLAUSE_PATTERN,
                _EXPLICIT_POINT_PATTERN,
                _POINT_PATTERN,
            )
        ):
            return False
        return True

    @staticmethod
    def _build_blocks(
        document_id: str, drafts: list[_DraftBlock]
    ) -> list[LegalBlock]:
        block_ids = [
            LegalStructureParser._block_id(
                document_id,
                order_index,
                draft.block_type,
                draft.text,
            )
            for order_index, draft in enumerate(drafts)
        ]
        return [
            LegalBlock(
                block_id=block_ids[order_index],
                document_id=document_id,
                block_type=draft.block_type,
                block_number=draft.number,
                title=draft.title,
                text=draft.text,
                parent_block_id=(
                    block_ids[draft.parent_index]
                    if draft.parent_index is not None
                    else None
                ),
                order_index=order_index,
                structure=draft.structure,
            )
            for order_index, draft in enumerate(drafts)
        ]

    @staticmethod
    def _block_id(
        document_id: str,
        order_index: int,
        block_type: LegalBlockType,
        text: str,
    ) -> str:
        payload = f"{document_id}\0{order_index}\0{block_type.value}\0{text}"
        digest = sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"block_{digest}"

    @staticmethod
    def _validate_input(
        documents: list[LegalDocument], source_manifest: ArtifactManifest
    ) -> None:
        if source_manifest.artifact_type != ArtifactType.CLEANED_DOCUMENTS:
            raise ArtifactCompatibilityError(
                "Legal structure parser requires a cleaned-documents artifact"
            )
        if source_manifest.record_count != len(documents):
            raise DataValidationError(
                "Source manifest record count does not match input documents"
            )
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise DataValidationError(
                "Legal structure parser input document IDs must be unique"
            )

    def build_manifest(
        self,
        *,
        source_manifest: ArtifactManifest,
        block_count: int,
        document_count: int,
        issue_count: int,
        parsed_count: int,
        missing_count: int,
        structured_count: int,
        unstructured_count: int,
        source_characters: int,
        covered_characters: int,
    ) -> ArtifactManifest:
        """Build aggregate legal-block provenance from streaming counters."""
        warnings = list(source_manifest.warnings)
        if issue_count:
            warnings.append(f"Legal structure parsing produced {issue_count} issues")
        return ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.LEGAL_BLOCKS,
            artifact_version=self._config.artifact_version,
            dataset_name=source_manifest.dataset_name,
            dataset_revision=source_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=block_count,
            processing_config_hash=self._config_hash(source_manifest),
            code_version=__version__,
            warnings=warnings,
            metadata={
                "source_artifact_type": source_manifest.artifact_type.value,
                "source_artifact_version": source_manifest.artifact_version,
                "source_processing_config_hash": source_manifest.processing_config_hash,
                "input_document_count": document_count,
                "parsed_document_count": parsed_count,
                "missing_clean_text_count": missing_count,
                "structured_document_count": structured_count,
                "unstructured_document_count": unstructured_count,
                "parser_issue_count": issue_count,
                "source_non_whitespace_characters": source_characters,
                "covered_non_whitespace_characters": covered_characters,
            },
        )

    def _config_hash(self, source_manifest: ArtifactManifest) -> str:
        payload = {
            "source_processing_config_hash": source_manifest.processing_config_hash,
            "legal_structure_parser": self._config,
        }
        return canonical_sha256(payload)

    @staticmethod
    def _issue(
        issue_type: str,
        severity: AuditSeverity,
        document_id: str,
        message: str,
        *,
        metadata: dict[str, int] | None = None,
    ) -> AuditIssue:
        issue_metadata: dict[str, int | str] = {"stage": "legal_structure_parsing"}
        issue_metadata.update(metadata or {})
        return AuditIssue(
            issue_type=issue_type,
            severity=severity,
            record_id=document_id,
            message=message,
            metadata=issue_metadata,
        )

    @staticmethod
    def _group(match: re.Match[str], name: str) -> str | None:
        value = match.groupdict().get(name)
        normalized = value.strip() if value is not None else ""
        return normalized or None

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return " | " in line

    @staticmethod
    def _non_whitespace_count(value: str) -> int:
        return sum(not character.isspace() for character in value)
