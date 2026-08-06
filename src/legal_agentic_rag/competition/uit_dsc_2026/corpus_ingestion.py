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
from legal_agentic_rag.competition.uit_dsc_2026.passage_cleaner import (
    UitDsc2026PassageCleaner,
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
from legal_agentic_rag.schemas.legal_documents import LegalDocument

_ADAPTER_VERSION = "2.0"
_PASSED_CHECKS = [
    "utf8_json",
    "one_object_per_member",
    "audited_required_and_optional_fields",
    "canonical_non_negative_context_ids",
    "unique_context_ids",
    "one_to_one_document_mapping",
    "deterministic_passage_cleaning",
]


class UitDsc2026CorpusIngestor:
    """Build one strict, in-memory normalized corpus with pinned lineage."""

    def __init__(
        self,
        *,
        loader: UitDsc2026DataLoader | None = None,
        adapter: UitDsc2026ContextAdapter | None = None,
        cleaner: UitDsc2026PassageCleaner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._loader = loader or UitDsc2026DataLoader()
        self._adapter = adapter or UitDsc2026ContextAdapter()
        self._cleaner = cleaner or UitDsc2026PassageCleaner()
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
        normalized_documents = [
            self._adapter.to_document(context) for context in contexts
        ]
        cleaning_results = [
            self._cleaner.clean(context.passage) for context in contexts
        ]
        cleaned_documents = [
            self._with_clean_text(document, cleaning.text)
            for document, cleaning in zip(
                normalized_documents,
                cleaning_results,
                strict=True,
            )
        ]
        if len(normalized_documents) != identity.member_count:
            raise DataValidationError(
                "Official context member count differs from normalized documents"
            )

        passage_lengths = [len(context.passage) for context in contexts]
        cleaned_lengths = [len(result.text) for result in cleaning_results]
        titles = [
            context.title for context in contexts if context.title is not None
        ]
        source_urls = [context.source_url for context in contexts]
        content_context_count = sum(
            bool(context.passage.strip()) for context in contexts
        )
        blank_passage_count = len(contexts) - content_context_count
        missing_title_count = sum(context.title is None for context in contexts)
        duplicate_passage_count = len(contexts) - len(
            {context.passage for context in contexts}
        )
        html_markup_context_count = sum(
            result.html_markup_removed for result in cleaning_results
        )
        boilerplate_context_count = sum(
            result.boilerplate_occurrence_count > 0 for result in cleaning_results
        )
        boilerplate_occurrence_count = sum(
            result.boilerplate_occurrence_count for result in cleaning_results
        )
        modified_context_count = sum(result.modified for result in cleaning_results)
        unicode_normalized_context_count = sum(
            result.unicode_normalized for result in cleaning_results
        )
        newline_normalized_context_count = sum(
            result.newline_normalized for result in cleaning_results
        )
        warnings = self._audit_warnings(
            blank_passage_count=blank_passage_count,
            missing_title_count=missing_title_count,
            duplicate_passage_count=duplicate_passage_count,
            html_markup_context_count=html_markup_context_count,
            boilerplate_occurrence_count=boilerplate_occurrence_count,
        )
        processing_hash = canonical_sha256(
            {
                "adapter": "uit_dsc_2026_context",
                "adapter_version": _ADAPTER_VERSION,
                "mapping": {
                    "context_id": "document_id",
                    "title": "title",
                    "source_url": "source_url",
                    "passage": "normalized_document.clean_text",
                },
                "context_id_canonicalization": "non_negative_int_or_non_blank_string",
                "title_is_optional": True,
                "blank_passage_is_allowed": True,
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
            record_counts={
                "contexts": len(normalized_documents),
                "contexts_with_content": content_context_count,
            },
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
            record_count=len(normalized_documents),
            processing_config_hash=processing_hash,
            code_version=__version__,
            warnings=list(warnings),
            metadata={
                "adapter_version": _ADAPTER_VERSION,
                "source_kind": identity.source_kind,
                "source_member_count": identity.member_count,
                "raw_passage_preserved": True,
                "content_context_count": content_context_count,
                "blank_passage_count": blank_passage_count,
                "missing_title_count": missing_title_count,
            },
        )
        cleaned_processing_hash = canonical_sha256(
            {
                "source_processing_config_hash": processing_hash,
                "cleaning_policy": self._cleaner.policy_identity(),
            }
        )
        cleaned_manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.CLEANED_DOCUMENTS,
            artifact_version="1.0",
            dataset_name=OFFICIAL_CORPUS_DATASET_NAME,
            dataset_revision=identity.revision,
            created_at=created_at,
            record_count=len(cleaned_documents),
            processing_config_hash=cleaned_processing_hash,
            code_version=__version__,
            warnings=list(warnings),
            metadata={
                "source_artifact_type": ArtifactType.NORMALIZED_DOCUMENTS.value,
                "source_processing_config_hash": processing_hash,
                "operation": "uit_dsc_2026_passage_cleaning",
                "cleaner_version": self._cleaner.version,
                "text_modified": bool(modified_context_count),
                "modified_context_count": modified_context_count,
                "html_markup_context_count": html_markup_context_count,
                "boilerplate_context_count": boilerplate_context_count,
                "boilerplate_occurrence_count": boilerplate_occurrence_count,
                "unicode_normalized_context_count": (
                    unicode_normalized_context_count
                ),
                "newline_normalized_context_count": (
                    newline_normalized_context_count
                ),
            },
        )
        audit = CompetitionCorpusAuditReport(
            dataset_name=OFFICIAL_CORPUS_DATASET_NAME,
            dataset_revision=identity.revision,
            source_kind=identity.source_kind,
            member_count=identity.member_count,
            record_count=len(normalized_documents),
            unique_context_count=len({context.context_id for context in contexts}),
            content_context_count=content_context_count,
            blank_passage_count=blank_passage_count,
            missing_title_count=missing_title_count,
            total_passage_characters=sum(passage_lengths),
            total_cleaned_characters=sum(cleaned_lengths),
            minimum_passage_characters=min(passage_lengths),
            maximum_passage_characters=max(passage_lengths),
            duplicate_title_count=len(titles) - len(set(titles)),
            duplicate_source_url_count=len(source_urls) - len(set(source_urls)),
            duplicate_passage_count=duplicate_passage_count,
            html_markup_context_count=html_markup_context_count,
            boilerplate_context_count=boilerplate_context_count,
            boilerplate_occurrence_count=boilerplate_occurrence_count,
            modified_context_count=modified_context_count,
            passed_checks=list(_PASSED_CHECKS),
            warnings=warnings,
        )
        return CompetitionCorpusIngestionResult(
            normalized_documents=normalized_documents,
            cleaned_documents=cleaned_documents,
            dataset_manifest=dataset_manifest,
            normalized_manifest=normalized_manifest,
            cleaned_manifest=cleaned_manifest,
            audit=audit,
        )

    @staticmethod
    def _with_clean_text(document: LegalDocument, clean_text: str) -> LegalDocument:
        payload = document.model_dump(mode="python")
        payload["clean_text"] = clean_text or None
        payload["has_content"] = bool(clean_text.strip())
        return LegalDocument.model_validate(payload)

    @staticmethod
    def _audit_warnings(
        *,
        blank_passage_count: int,
        missing_title_count: int,
        duplicate_passage_count: int,
        html_markup_context_count: int,
        boilerplate_occurrence_count: int,
    ) -> list[str]:
        warnings: list[str] = []
        if blank_passage_count:
            warnings.append(
                f"{blank_passage_count} official contexts have blank passages"
            )
        if missing_title_count:
            warnings.append(
                f"{missing_title_count} official contexts have no title"
            )
        if duplicate_passage_count:
            warnings.append(
                f"{duplicate_passage_count} official contexts duplicate passage text"
            )
        if html_markup_context_count:
            warnings.append(
                f"{html_markup_context_count} official contexts contain "
                "audited HTML markup"
            )
        if boilerplate_occurrence_count:
            warnings.append(
                f"{boilerplate_occurrence_count} audited TVPL boilerplate "
                "occurrences detected"
            )
        return warnings
