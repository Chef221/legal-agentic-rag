"""Deterministic, read-only auditing of raw AIO dataset components."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
import logging
import re

from legal_agentic_rag.configuration.offline import DatasetAuditConfig
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.offline.datasets.aio.adapter import AioRecordAdapter
from legal_agentic_rag.offline.datasets.aio import raw_schema
from legal_agentic_rag.schemas.auditing import (
    AuditFieldProfile,
    AuditIssue,
    AuditSeverity,
    ComponentAuditSummary,
    DatasetAuditReport,
    JoinAuditSummary,
)
from legal_agentic_rag.schemas.manifests import DatasetManifest

RawRecord = Mapping[str, object]
Clock = Callable[[], datetime]
_LEGAL_MARKER = re.compile(r"\b(?:Điều|Chương|Mục|Khoản)\s+\w+", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
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
_LOGGER = logging.getLogger(__name__)


@dataclass
class _FieldAccumulator:
    present_count: int = 0
    null_count: int = 0
    observed_types: Counter[str] = field(default_factory=Counter)


@dataclass
class _ComponentAccumulator:
    component: DatasetComponent
    total_records: int = 0
    ids: Counter[str] = field(default_factory=Counter)
    empty_ids: int = 0
    malformed_ids: int = 0
    fields: dict[str, _FieldAccumulator] = field(default_factory=dict)
    baseline_fields: frozenset[str] | None = None


class _HtmlBalanceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.unbalanced = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        return None

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if not self.stack or tag not in self.stack:
            self.unbalanced = True
            return
        if self.stack[-1] != tag:
            self.unbalanced = True
        while self.stack:
            opened = self.stack.pop()
            if opened == tag:
                break


class DatasetAuditService:
    """Audit raw streams without converting them to unified legal records."""

    def __init__(
        self,
        config: DatasetAuditConfig | None = None,
        *,
        adapter: AioRecordAdapter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or DatasetAuditConfig()
        self._adapter = adapter or AioRecordAdapter()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._issues: list[AuditIssue] = []
        self._effect_statuses: Counter[str] = Counter()

    def audit(
        self,
        *,
        metadata_records: Iterable[RawRecord],
        content_records: Iterable[RawRecord],
        relationship_records: Iterable[RawRecord],
        manifest: DatasetManifest,
    ) -> DatasetAuditReport:
        """Return schema, identity, join, value, and relationship findings."""
        _LOGGER.info(
            "dataset_audit_started",
            extra={"dataset_name": manifest.dataset_name},
        )
        self._issues = []
        self._effect_statuses = Counter()
        metadata = self._scan_metadata(metadata_records)
        content, content_by_id = self._scan_content(content_records)
        relationships, edges, labels = self._scan_relationships(
            relationship_records
        )

        metadata_ids = set(metadata.ids)
        content_ids = set(content.ids)
        missing_content = sorted(metadata_ids - content_ids)
        orphan_content = sorted(content_ids - metadata_ids)
        multiple_content = sorted(
            record_id for record_id, count in content_by_id.items() if count > 1
        )
        for record_id in missing_content:
            self._issue(
                "missing_content",
                AuditSeverity.WARNING,
                record_id,
                "Metadata record has no matching content record",
            )
        for record_id in orphan_content:
            self._issue(
                "orphan_content",
                AuditSeverity.ERROR,
                record_id,
                "Content record has no matching metadata record",
            )
        for record_id in multiple_content:
            self._issue(
                "multiple_content_records",
                AuditSeverity.WARNING,
                record_id,
                "Metadata ID has multiple content records",
                metadata={"count": content_by_id[record_id]},
            )

        invalid_sources, invalid_targets = self._audit_edges(edges, metadata_ids)
        components = {
            accumulator.component.value: self._summary(accumulator)
            for accumulator in (metadata, content, relationships)
        }
        report = DatasetAuditReport(
            schema_version="1.0",
            audit_config_hash=self._config_hash(),
            dataset_manifest=manifest,
            created_at=self._clock(),
            components=components,
            joins=JoinAuditSummary(
                metadata_with_content=len(metadata_ids & content_ids),
                metadata_without_content=len(missing_content),
                orphan_content_ids=len(orphan_content),
                metadata_with_multiple_content_records=len(multiple_content),
                invalid_relationship_sources=invalid_sources,
                invalid_relationship_targets=invalid_targets,
            ),
            effect_status_distribution=dict(sorted(self._effect_statuses.items())),
            relationship_distribution=dict(sorted(labels.items())),
            issues=self._issues,
        )
        _LOGGER.info(
            "dataset_audit_completed",
            extra={
                "dataset_name": manifest.dataset_name,
                "metadata_count": metadata.total_records,
                "content_count": content.total_records,
                "relationship_count": relationships.total_records,
                "issue_count": len(self._issues),
            },
        )
        return report

    def _scan_metadata(self, records: Iterable[RawRecord]) -> _ComponentAccumulator:
        component = DatasetComponent.METADATA
        accumulator = _ComponentAccumulator(component)
        for index, record in enumerate(records):
            self._profile_record(accumulator, record, index)
            record_id = self._profile_id(accumulator, record, index)
            self._audit_metadata_values(record, record_id)
        self._add_duplicate_id_issues(accumulator)
        return accumulator

    def _scan_content(
        self, records: Iterable[RawRecord]
    ) -> tuple[_ComponentAccumulator, Counter[str]]:
        component = DatasetComponent.CONTENT
        accumulator = _ComponentAccumulator(component)
        content_by_id: Counter[str] = Counter()
        content_hashes: dict[str, tuple[str | None, int]] = {}
        for index, record in enumerate(records):
            self._profile_record(accumulator, record, index)
            record_id = self._profile_id(accumulator, record, index)
            if record_id is not None:
                content_by_id[record_id] += 1
            raw_content = self._adapter.content(record)
            self._audit_content_value(
                raw_content, record_id, index, content_hashes
            )
        self._add_duplicate_id_issues(accumulator)
        for digest, (first_id, count) in sorted(content_hashes.items()):
            if count > 1:
                self._issue(
                    "duplicate_content",
                    AuditSeverity.WARNING,
                    first_id,
                    "Identical raw content occurs more than once",
                    metadata={
                        "component": DatasetComponent.CONTENT.value,
                        "content_hash": digest,
                        "count": count,
                    },
                )
        return accumulator, content_by_id

    def _scan_relationships(
        self, records: Iterable[RawRecord]
    ) -> tuple[
        _ComponentAccumulator,
        Counter[tuple[str, str, str]],
        Counter[str],
    ]:
        component = DatasetComponent.RELATIONSHIPS
        accumulator = _ComponentAccumulator(component)
        edges: Counter[tuple[str, str, str]] = Counter()
        labels: Counter[str] = Counter()
        for index, record in enumerate(records):
            self._profile_record(accumulator, record, index)
            source, target, label = self._adapter.relationship(record)
            raw_endpoints = (
                (
                    "source",
                    source,
                    record.get(raw_schema.RELATIONSHIP_SOURCE_FIELD),
                ),
                (
                    "target",
                    target,
                    record.get(raw_schema.RELATIONSHIP_TARGET_FIELD),
                ),
            )
            for endpoint_name, endpoint, raw_endpoint in raw_endpoints:
                if endpoint is None:
                    is_empty = raw_endpoint is None or (
                        isinstance(raw_endpoint, str) and not raw_endpoint.strip()
                    )
                    issue_type = (
                        "empty_relationship_endpoint"
                        if is_empty
                        else "malformed_relationship_endpoint"
                    )
                    if is_empty:
                        accumulator.empty_ids += 1
                    else:
                        accumulator.malformed_ids += 1
                    self._issue(
                        issue_type,
                        AuditSeverity.ERROR,
                        source,
                        f"Relationship {endpoint_name} ID is empty or malformed",
                        raw_value=raw_endpoint,
                        metadata={
                            "record_index": index,
                            "endpoint": endpoint_name,
                            "target_id": target,
                            "relationship": label,
                        },
                    )
            if label is None:
                self._issue(
                    "empty_relationship_label",
                    AuditSeverity.ERROR,
                    source,
                    "Relationship label is empty",
                    metadata={"record_index": index, "target_id": target},
                )
                continue
            labels[label] += 1
            if self._config.known_relationship_labels and (
                label not in self._config.known_relationship_labels
            ):
                self._issue(
                    "unknown_relationship_label",
                    AuditSeverity.WARNING,
                    source,
                    "Relationship label is not in the configured accepted set",
                    raw_value=label,
                    metadata={"record_index": index, "target_id": target},
                )
            if source is None or target is None:
                continue
            edge = (source, target, label)
            edges[edge] += 1
            accumulator.ids["\u241f".join(edge)] += 1
            if source == target:
                self._issue(
                    "self_loop",
                    AuditSeverity.WARNING,
                    source,
                    "Relationship points from a document to itself",
                    metadata={"target_id": target, "relationship": label},
                )
        for edge, count in sorted(edges.items()):
            if count > 1:
                source, target, label = edge
                self._issue(
                    "duplicate_edge",
                    AuditSeverity.WARNING,
                    source,
                    "Identical relationship edge occurs more than once",
                    metadata={
                        "component": DatasetComponent.RELATIONSHIPS.value,
                        "target_id": target,
                        "relationship": label,
                        "count": count,
                    },
                )
        self._audit_reciprocal_edges(edges)
        return accumulator, edges, labels

    def _profile_record(
        self,
        accumulator: _ComponentAccumulator,
        record: RawRecord,
        index: int,
    ) -> None:
        accumulator.total_records += 1
        field_names = frozenset(key for key in record if isinstance(key, str))
        required = self._adapter.required_fields(accumulator.component)
        for missing in sorted(required - field_names):
            self._issue(
                "missing_required_field",
                AuditSeverity.ERROR,
                None,
                "Raw record is missing a required field",
                metadata={
                    "component": accumulator.component.value,
                    "record_index": index,
                    "field_name": missing,
                },
            )
        if accumulator.baseline_fields is None:
            accumulator.baseline_fields = field_names
        elif field_names != accumulator.baseline_fields:
            self._issue(
                "inconsistent_record_schema",
                AuditSeverity.WARNING,
                None,
                "Raw record fields differ from the first observed record",
                metadata={
                    "component": accumulator.component.value,
                    "record_index": index,
                    "missing_fields": sorted(accumulator.baseline_fields - field_names),
                    "unexpected_fields": sorted(field_names - accumulator.baseline_fields),
                },
            )
        for field_name in sorted(field_names):
            value = record[field_name]
            profile = accumulator.fields.setdefault(field_name, _FieldAccumulator())
            profile.present_count += 1
            if value is None:
                profile.null_count += 1
            profile.observed_types[self._type_name(value)] += 1

    def _profile_id(
        self,
        accumulator: _ComponentAccumulator,
        record: RawRecord,
        index: int,
    ) -> str | None:
        field_name = raw_schema.IDENTIFIER_FIELDS[accumulator.component]
        raw_id = record.get(field_name)
        record_id = self._adapter.identifier(accumulator.component, record)
        if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
            accumulator.empty_ids += 1
            self._issue(
                "empty_id",
                AuditSeverity.ERROR,
                None,
                "Record ID is empty",
                metadata={
                    "component": accumulator.component.value,
                    "record_index": index,
                },
            )
            return None
        if record_id is None or any(character.isspace() for character in record_id):
            accumulator.malformed_ids += 1
            self._issue(
                "malformed_id",
                AuditSeverity.ERROR,
                record_id,
                "Record ID is not a supported non-whitespace scalar",
                raw_value=self._json_value(raw_id),
                metadata={
                    "component": accumulator.component.value,
                    "record_index": index,
                },
            )
            return None
        accumulator.ids[record_id] += 1
        return record_id

    def _audit_metadata_values(
        self, record: RawRecord, record_id: str | None
    ) -> None:
        required_values = (
            (raw_schema.METADATA_TITLE_FIELD, "missing_title"),
            (raw_schema.METADATA_NUMBER_FIELD, "missing_document_number"),
            (raw_schema.METADATA_TYPE_FIELD, "missing_document_type"),
            (raw_schema.METADATA_SOURCE_URL_FIELD, "missing_source_url"),
        )
        for field_name, issue_type in required_values:
            value = self._adapter.metadata_value(record, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                self._issue(
                    issue_type,
                    AuditSeverity.WARNING,
                    record_id,
                    f"Metadata field '{field_name}' is empty",
                    metadata={"field_name": field_name},
                )

        parsed_dates: dict[str, date | None] = {}
        for field_name in (
            raw_schema.METADATA_ISSUED_DATE_FIELD,
            raw_schema.METADATA_EFFECTIVE_DATE_FIELD,
            raw_schema.METADATA_EXPIRY_DATE_FIELD,
        ):
            raw_value = self._adapter.metadata_value(record, field_name)
            parsed, invalid = self._parse_date(raw_value)
            parsed_dates[field_name] = parsed
            if invalid:
                self._issue(
                    "invalid_date",
                    AuditSeverity.WARNING,
                    record_id,
                    f"Metadata date '{field_name}' cannot be parsed",
                    raw_value=self._json_value(raw_value),
                    metadata={"field_name": field_name},
                )
        issued = parsed_dates[raw_schema.METADATA_ISSUED_DATE_FIELD]
        effective = parsed_dates[raw_schema.METADATA_EFFECTIVE_DATE_FIELD]
        expiry = parsed_dates[raw_schema.METADATA_EXPIRY_DATE_FIELD]
        if issued is not None and effective is not None and effective < issued:
            self._issue(
                "effective_before_issued",
                AuditSeverity.WARNING,
                record_id,
                "Effective date is earlier than issued date",
            )
        if effective is not None and expiry is not None and expiry < effective:
            self._issue(
                "expiry_before_effective",
                AuditSeverity.WARNING,
                record_id,
                "Expiry date is earlier than effective date",
            )
        effect_status = self._adapter.metadata_value(
            record, raw_schema.METADATA_EFFECT_STATUS_FIELD
        )
        if isinstance(effect_status, str) and effect_status.strip():
            self._effect_statuses[effect_status.strip()] += 1
        if (
            self._config.known_effect_statuses
            and isinstance(effect_status, str)
            and effect_status.strip()
            and effect_status.strip() not in self._config.known_effect_statuses
        ):
            self._issue(
                "unknown_effect_status",
                AuditSeverity.WARNING,
                record_id,
                "Effect status is not in the configured accepted set",
                raw_value=effect_status,
            )

    def _audit_content_value(
        self,
        raw_content: object,
        record_id: str | None,
        index: int,
        hashes: dict[str, tuple[str | None, int]],
    ) -> None:
        if not isinstance(raw_content, str) or not raw_content.strip():
            self._issue(
                "empty_content",
                AuditSeverity.ERROR,
                record_id,
                "Raw content is empty or is not text",
                metadata={"record_index": index},
            )
            return
        length = len(raw_content)
        if length < self._config.minimum_content_characters:
            self._issue(
                "short_content",
                AuditSeverity.WARNING,
                record_id,
                "Raw content is shorter than the configured threshold",
                metadata={"character_count": length},
            )
        if length > self._config.maximum_content_characters:
            self._issue(
                "long_content",
                AuditSeverity.WARNING,
                record_id,
                "Raw content is longer than the configured threshold",
                metadata={"character_count": length},
            )
        digest = sha256(raw_content.encode("utf-8")).hexdigest()
        first_id, count = hashes.get(digest, (record_id, 0))
        hashes[digest] = (first_id, count + 1)

        parser = _HtmlBalanceParser()
        try:
            parser.feed(raw_content)
            parser.close()
        except Exception:
            parser.unbalanced = True
        if parser.unbalanced or parser.stack:
            self._issue(
                "malformed_html",
                AuditSeverity.WARNING,
                record_id,
                "Raw HTML contains unbalanced tags",
            )
        if self._only_element(raw_content, "nav"):
            self._issue(
                "navigation_only_content",
                AuditSeverity.WARNING,
                record_id,
                "Raw content contains only navigation markup",
            )
        if self._only_element(raw_content, "table"):
            self._issue(
                "table_only_content",
                AuditSeverity.WARNING,
                record_id,
                "Raw content contains only table markup",
            )
        visible_text = _TAG.sub(" ", raw_content)
        if not _LEGAL_MARKER.search(visible_text):
            self._issue(
                "missing_legal_structure_marker",
                AuditSeverity.INFO,
                record_id,
                "No common Vietnamese legal structure marker was detected",
            )

    def _audit_edges(
        self,
        edges: Counter[tuple[str, str, str]],
        metadata_ids: set[str],
    ) -> tuple[int, int]:
        invalid_sources = 0
        invalid_targets = 0
        for (source, target, label), count in sorted(edges.items()):
            if source not in metadata_ids:
                invalid_sources += count
                self._issue(
                    "invalid_relationship_source",
                    AuditSeverity.ERROR,
                    source,
                    "Relationship source has no matching metadata record",
                    metadata={
                        "target_id": target,
                        "relationship": label,
                        "count": count,
                    },
                )
            if target not in metadata_ids:
                invalid_targets += count
                self._issue(
                    "invalid_relationship_target",
                    AuditSeverity.ERROR,
                    source,
                    "Relationship target has no matching metadata record",
                    metadata={
                        "target_id": target,
                        "relationship": label,
                        "count": count,
                    },
                )
        return invalid_sources, invalid_targets

    def _audit_reciprocal_edges(
        self, edges: Counter[tuple[str, str, str]]
    ) -> None:
        endpoint_pairs = {(source, target) for source, target, _ in edges}
        for source, target in sorted(endpoint_pairs):
            if source < target and (target, source) in endpoint_pairs:
                self._issue(
                    "reciprocal_edge",
                    AuditSeverity.INFO,
                    source,
                    "Relationship endpoints also occur in the reverse direction",
                    metadata={"target_id": target},
                )

    def _add_duplicate_id_issues(
        self, accumulator: _ComponentAccumulator
    ) -> None:
        for record_id, count in sorted(accumulator.ids.items()):
            if count > 1:
                self._issue(
                    "duplicate_id",
                    AuditSeverity.ERROR,
                    record_id,
                    "Record ID occurs more than once",
                    metadata={
                        "component": accumulator.component.value,
                        "count": count,
                    },
                )

    def _summary(self, accumulator: _ComponentAccumulator) -> ComponentAuditSummary:
        profiles = [
            AuditFieldProfile(
                field_name=name,
                present_count=value.present_count,
                null_count=value.null_count,
                observed_types=dict(sorted(value.observed_types.items())),
            )
            for name, value in sorted(accumulator.fields.items())
        ]
        return ComponentAuditSummary(
            component=accumulator.component.value,
            total_records=accumulator.total_records,
            unique_ids=len(accumulator.ids),
            duplicate_ids=sum(1 for count in accumulator.ids.values() if count > 1),
            empty_ids=accumulator.empty_ids,
            malformed_ids=accumulator.malformed_ids,
            field_profiles=profiles,
        )

    def _issue(
        self,
        issue_type: str,
        severity: AuditSeverity,
        record_id: str | None,
        message: str,
        *,
        raw_value: object = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._issues.append(
            AuditIssue(
                issue_type=issue_type,
                severity=severity,
                record_id=record_id,
                message=message,
                raw_value=self._json_value(raw_value),
                metadata={
                    key: self._json_value(value)
                    for key, value in (metadata or {}).items()
                },
            )
        )

    @staticmethod
    def _parse_date(value: object) -> tuple[date | None, bool]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None, False
        if isinstance(value, datetime):
            return value.date(), False
        if isinstance(value, date):
            return value, False
        if not isinstance(value, str):
            return None, True
        normalized = value.strip()
        for format_string in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(normalized, format_string).date(), False
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date(), False
        except ValueError:
            return None, True

    @staticmethod
    def _only_element(html: str, tag: str) -> bool:
        if not re.search(fr"<{tag}\b", html, flags=re.IGNORECASE):
            return False
        outside = re.sub(
            fr"<{tag}\b[^>]*>.*?</{tag}>",
            "",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return not _TAG.sub("", outside).strip()

    @staticmethod
    def _type_name(value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, Mapping):
            return "object"
        if isinstance(value, (list, tuple)):
            return "array"
        return type(value).__name__

    @staticmethod
    def _json_value(value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): DatasetAuditService._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [DatasetAuditService._json_value(item) for item in value]
        return str(value)

    def _config_hash(self) -> str:
        payload = self._config.model_dump(mode="json")
        payload["known_effect_statuses"] = sorted(self._config.known_effect_statuses)
        payload["known_relationship_labels"] = sorted(
            self._config.known_relationship_labels
        )
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()
