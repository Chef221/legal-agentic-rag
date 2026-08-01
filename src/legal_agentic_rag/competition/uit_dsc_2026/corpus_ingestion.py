"""Official UIT DSC 2026 context ingestion into unified documents."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.context_adapter import (
    UitDsc2026ContextAdapter,
)
from legal_agentic_rag.competition.uit_dsc_2026.loader import (
    ContextSourceIdentity,
    UitDsc2026DataLoader,
)
from legal_agentic_rag.configuration.competition import (
    OFFICIAL_CORPUS_DATASET_NAME,
)
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas.competition import (
    CompetitionCorpusAuditReport,
    CompetitionCorpusIngestionResult,
)
from legal_agentic_rag.schemas.manifests import (
    ArtifactManifest,
    ArtifactType,
    DatasetManifest,
)

_ADAPTER_VERSION = "1.0"
_PASSED_CHECKS = [
    "utf8_json",
    "one_object_per_member",
    "exact_raw_fields",
    "non_blank_required_text",
    "unique_context_ids",
    "one_to_one_document_mapping",
]


class UitDsc2026CorpusIngestor:
    """Build one strict, in-memory normalized corpus with pinned lineage."""

    def __init__(
        self,
        *,
        loader: UitDsc2026DataLoader | None = None,
        adapter: UitDsc2026ContextAdapter | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._loader = loader or UitDsc2026DataLoader()
        self._adapter = adapter or UitDsc2026ContextAdapter()
        self._clock = clock or (lambda: datetime.now(UTC))

    def inspect_source(self, source: Path) -> ContextSourceIdentity:
        """Return the exact canonical identity without ingesting the corpus."""
        return self._loader.inspect_context_source(source)

    def ingest(self, source: Path) -> CompetitionCorpusIngestionResult:
        """Validate and map an official ZIP or extracted context directory."""
        identity = self._loader.inspect_context_source(source)
        contexts = list(self._loader.iter_contexts(source))
        verified_identity = self._loader.inspect_context_source(source)
        if verified_identity != identity:
            raise DataValidationError(
                "Official context source changed during ingestion"
            )
        documents = [self._adapter.to_document(context) for context in contexts]
        if len(documents) != identity.member_count:
            raise DataValidationError(
                "Official context member count differs from normalized documents"
            )

        passage_lengths = [len(context.passage) for context in contexts]
        titles = [context.title for context in contexts]
        source_urls = [context.source_url for context in contexts]
        processing_hash = canonical_sha256(
            {
                "adapter": "uit_dsc_2026_context",
                "adapter_version": _ADAPTER_VERSION,
                "mapping": {
                    "context_id": "document_id",
                    "title": "title",
                    "source_url": "source_url",
                    "passage": "clean_text",
                },
                "infer_missing_legal_metadata": False,
            }
        )
        created_at = self._clock()
        dataset_manifest = DatasetManifest(
            schema_version="1.0",
            dataset_name=OFFICIAL_CORPUS_DATASET_NAME,
            dataset_revision=identity.revision,
            loaded_at=created_at,
            configs=["contexts"],
            record_counts={"contexts": len(documents)},
            processing_config_hash=processing_hash,
            code_version=__version__,
        )
        normalized_manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.NORMALIZED_DOCUMENTS,
            artifact_version="1.0",
            dataset_name=OFFICIAL_CORPUS_DATASET_NAME,
            dataset_revision=identity.revision,
            created_at=created_at,
            record_count=len(documents),
            processing_config_hash=processing_hash,
            code_version=__version__,
            metadata={
                "adapter_version": _ADAPTER_VERSION,
                "source_kind": identity.source_kind,
                "source_member_count": identity.member_count,
                "passage_is_plain_text": True,
            },
        )
        cleaned_processing_hash = canonical_sha256(
            {
                "source_processing_config_hash": processing_hash,
                "operation": "official_plain_text_passthrough",
                "text_modified": False,
            }
        )
        cleaned_manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.CLEANED_DOCUMENTS,
            artifact_version="1.0",
            dataset_name=OFFICIAL_CORPUS_DATASET_NAME,
            dataset_revision=identity.revision,
            created_at=created_at,
            record_count=len(documents),
            processing_config_hash=cleaned_processing_hash,
            code_version=__version__,
            metadata={
                "source_artifact_type": ArtifactType.NORMALIZED_DOCUMENTS.value,
                "source_processing_config_hash": processing_hash,
                "operation": "official_plain_text_passthrough",
                "text_modified": False,
            },
        )
        audit = CompetitionCorpusAuditReport(
            dataset_name=OFFICIAL_CORPUS_DATASET_NAME,
            dataset_revision=identity.revision,
            source_kind=identity.source_kind,
            member_count=identity.member_count,
            record_count=len(documents),
            unique_context_count=len({context.context_id for context in contexts}),
            total_passage_characters=sum(passage_lengths),
            minimum_passage_characters=min(passage_lengths),
            maximum_passage_characters=max(passage_lengths),
            duplicate_title_count=len(titles) - len(set(titles)),
            duplicate_source_url_count=len(source_urls) - len(set(source_urls)),
            passed_checks=list(_PASSED_CHECKS),
        )
        return CompetitionCorpusIngestionResult(
            documents=documents,
            dataset_manifest=dataset_manifest,
            normalized_manifest=normalized_manifest,
            cleaned_manifest=cleaned_manifest,
            audit=audit,
        )
