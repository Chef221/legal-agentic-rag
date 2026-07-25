"""Conservative conversion of raw AIO records into unified legal documents."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime
import logging
from urllib.parse import urlsplit

from pydantic import JsonValue

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import DocumentNormalizationConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.offline.datasets.aio.adapter import (
    AioContentValues,
    AioMetadataValues,
    AioRecordAdapter,
)
from legal_agentic_rag.offline.datasets.aio.raw_schema import AIO_DATASET_NAME
from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity
from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    ArtifactType,
    DatasetManifest,
)
from legal_agentic_rag.schemas.normalization import DocumentNormalizationResult

RawRecord = Mapping[str, object]
Clock = Callable[[], datetime]
_LOGGER = logging.getLogger(__name__)


class AioDocumentNormalizer:
    """Join AIO metadata/content while refusing ambiguous legal source records."""

    def __init__(
        self,
        config: DocumentNormalizationConfig | None = None,
        *,
        adapter: AioRecordAdapter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or DocumentNormalizationConfig()
        self._adapter = adapter or AioRecordAdapter()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._issues: list[AuditIssue] = []

    def normalize(
        self,
        *,
        metadata_records: Iterable[RawRecord],
        content_records: Iterable[RawRecord],
        dataset_manifest: DatasetManifest,
    ) -> DocumentNormalizationResult:
        """Normalize one completed AIO ingestion into unified documents."""
        if dataset_manifest.dataset_name != AIO_DATASET_NAME:
            raise DataValidationError(
                "AioDocumentNormalizer received a manifest for another dataset"
            )
        self._issues = []
        _LOGGER.info(
            "document_normalization_started",
            extra={"dataset_name": dataset_manifest.dataset_name},
        )

        metadata_by_id, metadata_counts, input_metadata_count, invalid_metadata = (
            self._index_metadata(metadata_records)
        )
        content_by_id, content_counts, input_content_count = self._index_content(
            content_records
        )
        duplicate_metadata_ids = {
            document_id
            for document_id, count in metadata_counts.items()
            if count > 1
        }
        ambiguous_content_ids = {
            document_id for document_id, count in content_counts.items() if count > 1
        }
        for document_id in sorted(duplicate_metadata_ids):
            self._issue(
                "duplicate_metadata_id",
                AuditSeverity.ERROR,
                document_id,
                "All metadata records for this duplicate ID were rejected",
                metadata={"count": metadata_counts[document_id]},
            )
        for document_id in sorted(ambiguous_content_ids):
            self._issue(
                "ambiguous_content",
                AuditSeverity.ERROR,
                document_id,
                "Multiple content records exist; none was selected or merged",
                metadata={"count": content_counts[document_id]},
            )

        known_metadata_ids = set(metadata_counts)
        orphan_content_ids = sorted(set(content_counts) - known_metadata_ids)
        for document_id in orphan_content_ids:
            self._issue(
                "orphan_content",
                AuditSeverity.WARNING,
                document_id,
                "Content has no matching metadata and was not normalized",
                metadata={"count": content_counts[document_id]},
            )

        documents: list[LegalDocument] = []
        for document_id in sorted(set(metadata_by_id) - duplicate_metadata_ids):
            metadata = metadata_by_id[document_id]
            raw_content = (
                content_by_id.get(document_id)
                if content_counts.get(document_id, 0) == 1
                else None
            )
            documents.append(
                self._normalize_document(metadata, raw_content, dataset_manifest)
            )

        rejected_metadata_count = invalid_metadata + sum(
            metadata_counts[document_id] for document_id in duplicate_metadata_ids
        )
        manifest = self._artifact_manifest(
            dataset_manifest=dataset_manifest,
            document_count=len(documents),
            input_metadata_count=input_metadata_count,
            input_content_count=input_content_count,
            rejected_metadata_count=rejected_metadata_count,
            orphan_content_count=len(orphan_content_ids),
            ambiguous_content_count=len(ambiguous_content_ids),
        )
        result = DocumentNormalizationResult(
            documents=documents,
            issues=self._issues,
            manifest=manifest,
            input_metadata_count=input_metadata_count,
            input_content_count=input_content_count,
            rejected_metadata_count=rejected_metadata_count,
            orphan_content_count=len(orphan_content_ids),
            ambiguous_content_count=len(ambiguous_content_ids),
        )
        _LOGGER.info(
            "document_normalization_completed",
            extra={
                "dataset_name": dataset_manifest.dataset_name,
                "document_count": len(documents),
                "issue_count": len(self._issues),
                "rejected_metadata_count": rejected_metadata_count,
            },
        )
        return result

    def _index_metadata(
        self, records: Iterable[RawRecord]
    ) -> tuple[dict[str, AioMetadataValues], Counter[str], int, int]:
        metadata_by_id: dict[str, AioMetadataValues] = {}
        counts: Counter[str] = Counter()
        input_count = 0
        invalid_count = 0
        for index, record in enumerate(records):
            input_count += 1
            values = self._adapter.normalization_metadata(record)
            if not self._valid_identifier(values.document_id):
                invalid_count += 1
                self._issue(
                    "invalid_metadata_id",
                    AuditSeverity.ERROR,
                    values.document_id,
                    "Metadata record cannot be normalized without a valid ID",
                    raw_value=values.raw_document_id,
                    metadata={"record_index": index},
                )
                continue
            document_id = values.document_id
            counts[document_id] += 1
            metadata_by_id.setdefault(document_id, values)
        return metadata_by_id, counts, input_count, invalid_count

    def _index_content(
        self, records: Iterable[RawRecord]
    ) -> tuple[dict[str, AioContentValues], Counter[str], int]:
        content_by_id: dict[str, AioContentValues] = {}
        counts: Counter[str] = Counter()
        input_count = 0
        for index, record in enumerate(records):
            input_count += 1
            values = self._adapter.normalization_content(record)
            if not self._valid_identifier(values.document_id):
                self._issue(
                    "invalid_content_id",
                    AuditSeverity.ERROR,
                    values.document_id,
                    "Content record cannot be joined without a valid ID",
                    raw_value=values.raw_document_id,
                    metadata={"record_index": index},
                )
                continue
            document_id = values.document_id
            counts[document_id] += 1
            content_by_id.setdefault(document_id, values)
        return content_by_id, counts, input_count

    def _normalize_document(
        self,
        metadata: AioMetadataValues,
        content: AioContentValues | None,
        dataset_manifest: DatasetManifest,
    ) -> LegalDocument:
        document_id = metadata.document_id
        if document_id is None:
            raise DataValidationError("Validated metadata unexpectedly has no ID")
        issuance_date = self._date(
            metadata.issuance_date, "issuance_date", document_id
        )
        effective_date = self._date(
            metadata.effective_date, "effective_date", document_id
        )
        expiry_date = self._date(metadata.expiry_date, "expiry_date", document_id)
        publication_date = self._date(
            metadata.publication_date, "publication_date", document_id
        )
        self._audit_date_order(
            document_id, issuance_date, effective_date, expiry_date
        )
        document_type = self._mapped_text(
            metadata.document_type,
            "document_type",
            document_id,
            self._config.document_type_mapping,
        )
        effect_status = self._mapped_text(
            metadata.effect_status,
            "effect_status",
            document_id,
            self._config.effect_status_mapping,
        )
        content_html = self._content(content, document_id)
        return LegalDocument(
            document_id=document_id,
            title=self._text(metadata.title, "title", document_id),
            document_number=self._text(
                metadata.document_number, "document_number", document_id
            ),
            document_type=document_type,
            issuance_date=issuance_date,
            effective_date=effective_date,
            expiry_date=expiry_date,
            effect_status=effect_status,
            issuing_authority=self._text(
                metadata.issuing_authority, "issuing_authority", document_id
            ),
            position_title=self._text(
                metadata.position_title, "position_title", document_id
            ),
            signer=self._text(metadata.signer, "signer", document_id),
            sector=self._text(metadata.sector, "sector", document_id),
            legal_field=self._text(
                metadata.legal_field, "legal_field", document_id
            ),
            scope=self._text(metadata.scope, "scope", document_id),
            application_info=self._text(
                metadata.application_info, "application_info", document_id
            ),
            publication_date=publication_date,
            source_url=self._url(metadata.source_url, document_id),
            content_html=content_html,
            clean_text=None,
            has_content=content_html is not None,
            source_dataset=self._config.source_dataset,
            raw_metadata=self._json_mapping(metadata.raw_metadata),
        )

    def _content(
        self, content: AioContentValues | None, document_id: str
    ) -> str | None:
        if content is None:
            self._issue(
                "missing_content",
                AuditSeverity.WARNING,
                document_id,
                "No single unambiguous content record was available",
            )
            return None
        value = content.content_html
        if not isinstance(value, str) or not value.strip():
            self._issue(
                "invalid_content",
                AuditSeverity.ERROR,
                document_id,
                "Content payload is empty or is not text",
                raw_value=value,
            )
            return None
        return value

    def _text(
        self, value: object, field_name: str, document_id: str
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._issue(
                "coerced_metadata_value",
                AuditSeverity.INFO,
                document_id,
                "Scalar metadata value was converted to text",
                raw_value=value,
                metadata={"field_name": field_name},
            )
            return str(value)
        self._issue(
            "invalid_metadata_value",
            AuditSeverity.WARNING,
            document_id,
            "Metadata value is not a supported text scalar",
            raw_value=value,
            metadata={"field_name": field_name},
        )
        return None

    def _mapped_text(
        self,
        value: object,
        field_name: str,
        document_id: str,
        mapping: dict[str, str],
    ) -> str | None:
        normalized = self._text(value, field_name, document_id)
        if normalized is None:
            return None
        return mapping.get(normalized, normalized)

    def _date(
        self, value: object, field_name: str, document_id: str
    ) -> date | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            for format_string in (
                "%d/%m/%Y",
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%Y/%m/%d",
            ):
                try:
                    return datetime.strptime(normalized, format_string).date()
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(
                    normalized.replace("Z", "+00:00")
                ).date()
            except ValueError:
                pass
        self._issue(
            "invalid_date",
            AuditSeverity.WARNING,
            document_id,
            "Date value could not be parsed and was normalized to null",
            raw_value=value,
            metadata={"field_name": field_name},
        )
        return None

    def _url(self, value: object, document_id: str) -> str | None:
        normalized = self._text(value, "source_url", document_id)
        if normalized is None:
            return None
        parsed = urlsplit(normalized)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            self._issue(
                "invalid_source_url",
                AuditSeverity.WARNING,
                document_id,
                "Source URL is not an absolute HTTP(S) URL",
                raw_value=normalized,
            )
            return None
        return normalized

    def _audit_date_order(
        self,
        document_id: str,
        issuance_date: date | None,
        effective_date: date | None,
        expiry_date: date | None,
    ) -> None:
        if (
            issuance_date is not None
            and effective_date is not None
            and effective_date < issuance_date
        ):
            self._issue(
                "effective_before_issuance",
                AuditSeverity.WARNING,
                document_id,
                "Effective date precedes issuance date; values were preserved",
            )
        if (
            effective_date is not None
            and expiry_date is not None
            and expiry_date < effective_date
        ):
            self._issue(
                "expiry_before_effective",
                AuditSeverity.WARNING,
                document_id,
                "Expiry date precedes effective date; values were preserved",
            )

    def _artifact_manifest(
        self,
        *,
        dataset_manifest: DatasetManifest,
        document_count: int,
        input_metadata_count: int,
        input_content_count: int,
        rejected_metadata_count: int,
        orphan_content_count: int,
        ambiguous_content_count: int,
    ) -> ArtifactManifest:
        warnings = list(dataset_manifest.warnings)
        if self._issues:
            warnings.append(f"Normalization produced {len(self._issues)} issues")
        return ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
            artifact_version=self._config.artifact_version,
            dataset_name=dataset_manifest.dataset_name,
            dataset_revision=dataset_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=document_count,
            processing_config_hash=self._config_hash(dataset_manifest),
            code_version=__version__,
            warnings=warnings,
            metadata={
                "input_metadata_count": input_metadata_count,
                "input_content_count": input_content_count,
                "rejected_metadata_count": rejected_metadata_count,
                "orphan_content_count": orphan_content_count,
                "ambiguous_content_count": ambiguous_content_count,
                "normalization_issue_count": len(self._issues),
            },
        )

    def _config_hash(self, dataset_manifest: DatasetManifest) -> str:
        payload = {
            "dataset_processing_config_hash": (
                dataset_manifest.processing_config_hash
            ),
            "normalization": self._config,
        }
        return canonical_sha256(payload)

    def _issue(
        self,
        issue_type: str,
        severity: AuditSeverity,
        document_id: str | None,
        message: str,
        *,
        raw_value: object = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        issue_metadata = {"stage": "normalization"}
        issue_metadata.update(metadata or {})
        self._issues.append(
            AuditIssue(
                issue_type=issue_type,
                severity=severity,
                record_id=document_id,
                message=message,
                raw_value=self._json_value(raw_value),
                metadata={
                    key: self._json_value(value)
                    for key, value in issue_metadata.items()
                },
            )
        )

    @staticmethod
    def _valid_identifier(value: str | None) -> bool:
        return value is not None and not any(character.isspace() for character in value)

    @classmethod
    def _json_mapping(cls, value: Mapping[str, object]) -> dict[str, JsonValue]:
        return {str(key): cls._json_value(item) for key, item in value.items()}

    @classmethod
    def _json_value(cls, value: object) -> JsonValue:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._json_value(item) for item in value]
        return str(value)
