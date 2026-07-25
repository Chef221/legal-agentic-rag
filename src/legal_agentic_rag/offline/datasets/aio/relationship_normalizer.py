"""AIO-boundary conversion into unified directed legal relationships."""

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
import logging

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import (
    RelationshipNormalizationConfig,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.offline.datasets.aio.adapter import AioRecordAdapter
from legal_agentic_rag.offline.datasets.aio.raw_schema import AIO_DATASET_NAME
from legal_agentic_rag.schemas.auditing import AuditIssue, AuditSeverity
from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.legal_relationships import LegalRelationship
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.relationship_processing import (
    RelationshipNormalizationResult,
)

RawRecord = Mapping[str, object]
Clock = Callable[[], datetime]
_LOGGER = logging.getLogger(__name__)


class AioRelationshipNormalizer:
    """Validate AIO edges and isolate all raw field knowledge at the adapter."""

    def __init__(
        self,
        config: RelationshipNormalizationConfig | None = None,
        *,
        adapter: AioRecordAdapter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or RelationshipNormalizationConfig()
        self._adapter = adapter or AioRecordAdapter()
        self._clock = clock or (lambda: datetime.now(UTC))

    def normalize(
        self,
        *,
        relationship_records: Iterable[RawRecord],
        documents: Iterable[LegalDocument],
        document_manifest: ArtifactManifest,
    ) -> RelationshipNormalizationResult:
        """Normalize valid directed edges against the known document corpus."""
        document_list = list(documents)
        self._validate_documents(document_list, document_manifest)
        document_ids = {document.document_id for document in document_list}
        issues: list[AuditIssue] = []
        relationships: list[LegalRelationship] = []
        seen: set[tuple[str, str, str, str]] = set()
        input_count = 0
        duplicate_count = 0

        for record_index, record in enumerate(relationship_records):
            input_count += 1
            source_id, target_id, raw_label = self._adapter.relationship(record)
            issue = self._invalid_issue(
                source_id,
                target_id,
                raw_label,
                document_ids,
                record_index,
            )
            if issue is not None:
                issues.append(issue)
                continue
            assert source_id is not None and target_id is not None
            assert raw_label is not None
            canonical_type = self._config.relationship_type_mapping.get(raw_label)
            identity = (
                source_id,
                target_id,
                raw_label,
                canonical_type or "",
            )
            if identity in seen:
                duplicate_count += 1
                issues.append(
                    AuditIssue(
                        issue_type="duplicate_relationship",
                        severity=AuditSeverity.WARNING,
                        record_id=f"{source_id}->{target_id}",
                        message="Duplicate directed relationship was rejected",
                        raw_value=raw_label,
                        metadata={"record_index": record_index},
                    )
                )
                continue
            seen.add(identity)
            relationships.append(
                LegalRelationship(
                    source_document_id=source_id,
                    target_document_id=target_id,
                    relationship_type=canonical_type,
                    raw_relationship=raw_label,
                    is_directed=True,
                    source_dataset=self._config.source_dataset,
                    metadata={"record_index": record_index},
                )
            )

        relationships.sort(key=self._identity)
        rejected_count = input_count - len(relationships)
        warnings: list[str] = []
        if rejected_count:
            warnings.append(f"rejected_relationship_count:{rejected_count}")
        if any(item.relationship_type is None for item in relationships):
            warnings.append("unmapped_relationship_labels_preserved")
        manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.RELATIONSHIP_MAPPING,
            artifact_version=self._config.artifact_version,
            dataset_name=document_manifest.dataset_name,
            dataset_revision=document_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=len(relationships),
            processing_config_hash=self._config_hash(),
            code_version=__version__,
            backend="aio_relationship_adapter",
            warnings=warnings,
            metadata={
                "input_count": input_count,
                "rejected_count": rejected_count,
                "duplicate_count": duplicate_count,
                "source_artifact_type": document_manifest.artifact_type.value,
                "source_artifact_version": document_manifest.artifact_version,
                "source_processing_config_hash": (
                    document_manifest.processing_config_hash
                ),
                "direction": "directed",
            },
        )
        _LOGGER.info(
            "relationship_normalization_completed",
            extra={
                "document_count": len(document_list),
                "relationship_count": len(relationships),
                "rejected_count": rejected_count,
            },
        )
        return RelationshipNormalizationResult(
            relationships=relationships,
            issues=issues,
            manifest=manifest,
            input_count=input_count,
            rejected_count=rejected_count,
            duplicate_count=duplicate_count,
        )

    def _invalid_issue(
        self,
        source_id: str | None,
        target_id: str | None,
        raw_label: str | None,
        document_ids: set[str],
        record_index: int,
    ) -> AuditIssue | None:
        metadata = {"record_index": record_index}
        record_id = (
            f"{source_id or '<missing>'}->{target_id or '<missing>'}"
        )
        if source_id is None or target_id is None:
            return AuditIssue(
                issue_type="invalid_relationship_endpoint",
                severity=AuditSeverity.ERROR,
                record_id=record_id,
                message="Relationship requires valid source and target identifiers",
                metadata=metadata,
            )
        if raw_label is None:
            return AuditIssue(
                issue_type="missing_relationship_label",
                severity=AuditSeverity.ERROR,
                record_id=record_id,
                message="Relationship requires a non-empty source label",
                metadata=metadata,
            )
        if source_id not in document_ids or target_id not in document_ids:
            return AuditIssue(
                issue_type="orphan_relationship_endpoint",
                severity=AuditSeverity.ERROR,
                record_id=record_id,
                message="Relationship endpoint is absent from normalized documents",
                raw_value=raw_label,
                metadata={
                    **metadata,
                    "missing_source": source_id not in document_ids,
                    "missing_target": target_id not in document_ids,
                },
            )
        if source_id == target_id:
            return AuditIssue(
                issue_type="relationship_self_loop",
                severity=AuditSeverity.WARNING,
                record_id=record_id,
                message="Self-loop was rejected by the baseline graph policy",
                raw_value=raw_label,
                metadata=metadata,
            )
        return None

    def _validate_documents(
        self,
        documents: list[LegalDocument],
        manifest: ArtifactManifest,
    ) -> None:
        if manifest.artifact_type != ArtifactType.NORMALIZED_DOCUMENTS:
            raise ArtifactCompatibilityError(
                "Relationship normalization requires normalized documents"
            )
        if manifest.dataset_name != AIO_DATASET_NAME or any(
            document.source_dataset != self._config.source_dataset
            for document in documents
        ):
            raise ArtifactCompatibilityError(
                "AIO relationship normalization received another data source"
            )
        document_ids = [document.document_id for document in documents]
        if (
            manifest.record_count != len(documents)
            or len(document_ids) != len(set(document_ids))
        ):
            raise ArtifactCompatibilityError(
                "Normalized document payload does not match its manifest"
            )

    @staticmethod
    def _identity(
        relationship: LegalRelationship,
    ) -> tuple[str, str, str, str]:
        return (
            relationship.source_document_id,
            relationship.target_document_id,
            relationship.raw_relationship,
            relationship.relationship_type or "",
        )

    def _config_hash(self) -> str:
        return canonical_sha256(self._config)
