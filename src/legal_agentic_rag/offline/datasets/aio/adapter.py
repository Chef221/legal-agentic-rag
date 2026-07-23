"""Read-only accessors for raw AIO ingestion, audit, and normalization."""

from collections.abc import Mapping
from dataclasses import dataclass

from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.offline.datasets.aio import raw_schema


@dataclass(frozen=True, slots=True)
class AioMetadataValues:
    """Raw metadata projected onto stable normalization inputs."""

    document_id: str | None
    raw_document_id: object
    title: object
    document_number: object
    document_type: object
    issuance_date: object
    effective_date: object
    expiry_date: object
    effect_status: object
    issuing_authority: object
    position_title: object
    signer: object
    sector: object
    legal_field: object
    scope: object
    application_info: object
    publication_date: object
    source_url: object
    raw_metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class AioContentValues:
    """Raw content identity and payload used for a conservative join."""

    document_id: str | None
    raw_document_id: object
    content_html: object


class AioRecordAdapter:
    """Expose known AIO fields without mutating or normalizing raw records."""

    def required_fields(self, component: DatasetComponent) -> frozenset[str]:
        """Return the minimum raw fields required for a logical component."""
        return raw_schema.REQUIRED_FIELDS[component]

    def identifier(
        self, component: DatasetComponent, record: Mapping[str, object]
    ) -> str | None:
        """Return a comparable raw identifier, or null when it is unusable."""
        field_name = raw_schema.IDENTIFIER_FIELDS.get(component)
        if field_name is None:
            return None
        return self.as_identifier(record.get(field_name))

    def as_identifier(self, value: object) -> str | None:
        """Convert scalar raw IDs to text solely for audit comparison."""
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None
        normalized = str(value).strip()
        return normalized or None

    def content(self, record: Mapping[str, object]) -> object:
        """Return the raw HTML payload without modifying it."""
        return record.get(raw_schema.CONTENT_FIELD)

    def relationship(
        self, record: Mapping[str, object]
    ) -> tuple[str | None, str | None, str | None]:
        """Return raw relationship endpoints and label for audit comparison."""
        source = self.as_identifier(
            record.get(raw_schema.RELATIONSHIP_SOURCE_FIELD)
        )
        target = self.as_identifier(
            record.get(raw_schema.RELATIONSHIP_TARGET_FIELD)
        )
        raw_label = record.get(raw_schema.RELATIONSHIP_LABEL_FIELD)
        label = raw_label.strip() if isinstance(raw_label, str) else None
        return source, target, label or None

    def metadata_value(
        self, record: Mapping[str, object], field_name: str
    ) -> object:
        """Return one explicitly requested raw metadata value."""
        return record.get(field_name)

    def normalization_metadata(
        self, record: Mapping[str, object]
    ) -> AioMetadataValues:
        """Project raw AIO metadata without parsing or discarding source fields."""
        return AioMetadataValues(
            document_id=self.identifier(DatasetComponent.METADATA, record),
            raw_document_id=record.get(
                raw_schema.IDENTIFIER_FIELDS[DatasetComponent.METADATA]
            ),
            title=record.get(raw_schema.METADATA_TITLE_FIELD),
            document_number=record.get(raw_schema.METADATA_NUMBER_FIELD),
            document_type=record.get(raw_schema.METADATA_TYPE_FIELD),
            issuance_date=record.get(raw_schema.METADATA_ISSUED_DATE_FIELD),
            effective_date=record.get(raw_schema.METADATA_EFFECTIVE_DATE_FIELD),
            expiry_date=record.get(raw_schema.METADATA_EXPIRY_DATE_FIELD),
            effect_status=record.get(raw_schema.METADATA_EFFECT_STATUS_FIELD),
            issuing_authority=record.get(raw_schema.METADATA_AUTHORITY_FIELD),
            position_title=record.get(raw_schema.METADATA_POSITION_FIELD),
            signer=record.get(raw_schema.METADATA_SIGNER_FIELD),
            sector=record.get(raw_schema.METADATA_SECTOR_FIELD),
            legal_field=record.get(raw_schema.METADATA_LEGAL_FIELD),
            scope=record.get(raw_schema.METADATA_SCOPE_FIELD),
            application_info=record.get(raw_schema.METADATA_APPLICATION_FIELD),
            publication_date=record.get(raw_schema.METADATA_PUBLICATION_DATE_FIELD),
            source_url=record.get(raw_schema.METADATA_SOURCE_URL_FIELD),
            raw_metadata=dict(record),
        )

    def normalization_content(
        self, record: Mapping[str, object]
    ) -> AioContentValues:
        """Project one raw content record for joining without altering its HTML."""
        return AioContentValues(
            document_id=self.identifier(DatasetComponent.CONTENT, record),
            raw_document_id=record.get(
                raw_schema.IDENTIFIER_FIELDS[DatasetComponent.CONTENT]
            ),
            content_html=self.content(record),
        )
