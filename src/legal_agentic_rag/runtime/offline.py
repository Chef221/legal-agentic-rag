"""Complete offline composition from AIO source streams to persisted indexes."""

from __future__ import annotations

import logging
from pathlib import Path

from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.contracts import (
    BM25Backend,
    DatasetSource,
    EmbeddingProvider,
    GraphBackend,
    VectorBackend,
)
from legal_agentic_rag.embeddings import SentenceTransformerEmbeddingProvider
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    ConfigurationError,
)
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.indexing.vector import (
    NumpyVectorBackend,
    VectorIndexBuilder,
)
from legal_agentic_rag.offline.audit import (
    DatasetAuditReportWriter,
    DatasetAuditService,
)
from legal_agentic_rag.offline.chunking import LegalChunker
from legal_agentic_rag.offline.cleaning import LegalHtmlCleaner
from legal_agentic_rag.offline.datasets.aio import (
    AioDatasetSource,
    AioDocumentNormalizer,
    AioRelationshipNormalizer,
)
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.offline.relationships import (
    persist_relationship_artifact,
)
from legal_agentic_rag.schemas import ArtifactManifest, OfflineBuildResult
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.runtime.artifact_store import (
    persist_dataset_manifest,
    persist_model_artifact,
)

_LOGGER = logging.getLogger(__name__)


class OfflineBuildRuntime:
    """Run the approved AIO offline pipeline and persist immutable artifacts."""

    def __init__(
        self,
        config: ApplicationConfig,
        *,
        source: DatasetSource | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        bm25_backend: BM25Backend | None = None,
        vector_backend: VectorBackend | None = None,
        graph_backend: GraphBackend | None = None,
    ) -> None:
        if config.artifacts.allow_overwrite:
            raise ConfigurationError(
                "Runtime assembly does not overwrite immutable artifact builds"
            )
        self._config = config
        self._source = source or AioDatasetSource(config.offline.dataset)
        self._embedding_provider = (
            embedding_provider
            or SentenceTransformerEmbeddingProvider(config.offline.embedding)
        )
        self._bm25 = bm25_backend or SQLiteFTS5BM25Backend(
            config.offline.bm25
        )
        self._vector = vector_backend or NumpyVectorBackend(
            config.offline.vector_index
        )
        self._graph = graph_backend or AdjacencyGraphBackend(
            config.offline.graph_index
        )

    def build(self) -> OfflineBuildResult:
        """Build one complete artifact set from a consistent raw snapshot."""
        self._preflight_destinations()
        metadata_records = list(
            self._source.iter_records(DatasetComponent.METADATA)
        )
        content_records = list(
            self._source.iter_records(DatasetComponent.CONTENT)
        )
        relationship_records = list(
            self._source.iter_records(DatasetComponent.RELATIONSHIPS)
        )
        dataset_manifest = self._source.dataset_manifest()

        audit = DatasetAuditService(self._config.offline.audit).audit(
            metadata_records=metadata_records,
            content_records=content_records,
            relationship_records=relationship_records,
            manifest=dataset_manifest,
        )
        normalized = AioDocumentNormalizer(
            self._config.offline.normalization
        ).normalize(
            metadata_records=metadata_records,
            content_records=content_records,
            dataset_manifest=dataset_manifest,
        )
        cleaned = LegalHtmlCleaner(
            self._config.offline.html_cleaning
        ).clean(
            documents=normalized.documents,
            source_manifest=normalized.manifest,
        )
        parsed = LegalStructureParser(
            self._config.offline.legal_structure_parser
        ).parse(
            documents=cleaned.documents,
            source_manifest=cleaned.manifest,
        )
        chunked = LegalChunker(self._config.offline.chunking).chunk(
            documents=parsed.documents,
            blocks=parsed.blocks,
            source_manifest=parsed.manifest,
        )
        relationships = AioRelationshipNormalizer(
            self._config.offline.relationship_normalization
        ).normalize(
            relationship_records=relationship_records,
            documents=normalized.documents,
            document_manifest=normalized.manifest,
        )

        self._bm25.build(chunked.chunks, chunked.manifest)
        VectorIndexBuilder(
            self._embedding_provider,
            self._vector,
            self._config.offline.vector_index,
        ).build(chunked.chunks, chunked.manifest)
        self._graph.build(
            normalized.documents,
            relationships.relationships,
            document_manifest=normalized.manifest,
            relationship_manifest=relationships.manifest,
        )

        root = self._config.artifacts.root_path
        output_paths: dict[str, str] = {}
        output_paths["dataset_manifest"] = str(
            persist_dataset_manifest(dataset_manifest, root)
        )
        audit_directory = self._directory("audit_directory")
        audit_paths = DatasetAuditReportWriter().write(
            audit,
            audit_directory,
        )
        output_paths["audit"] = str(audit_paths["data_audit.json"])

        stored_manifests: list[ArtifactManifest] = []
        for key, records, manifest, directory_field in (
            (
                "normalized_documents",
                normalized.documents,
                normalized.manifest,
                "normalized_documents_directory",
            ),
            (
                "cleaned_documents",
                cleaned.documents,
                cleaned.manifest,
                "cleaned_documents_directory",
            ),
            (
                "legal_blocks",
                parsed.blocks,
                parsed.manifest,
                "legal_blocks_directory",
            ),
            (
                "legal_chunks",
                chunked.chunks,
                chunked.manifest.model_copy(
                    update={
                        "metadata": {
                            **chunked.manifest.metadata,
                            "runtime_normalized_processing_config_hash": (
                                normalized.manifest.processing_config_hash
                            ),
                        }
                    }
                ),
                "legal_chunks_directory",
            ),
        ):
            destination = self._directory(directory_field)
            stored = persist_model_artifact(
                records=records,
                destination=destination,
                manifest=manifest,
            )
            stored_manifests.append(stored)
            output_paths[key] = str(destination)

        relationship_destination = self._directory("relationships_directory")
        relationship_manifest = persist_relationship_artifact(
            relationships=relationships.relationships,
            destination=relationship_destination,
            manifest=relationships.manifest,
        )
        stored_manifests.append(relationship_manifest)
        output_paths["relationship_mapping"] = str(relationship_destination)

        for backend, directory_field, key in (
            (self._bm25, "bm25_directory", "bm25_index"),
            (self._vector, "vector_directory", "vector_index"),
            (self._graph, "graph_directory", "graph_index"),
        ):
            destination = self._directory(directory_field)
            manifest = backend.persist(destination)
            stored_manifests.append(manifest)
            output_paths[key] = str(destination)

        processing_issue_count = sum(
            len(items)
            for items in (
                normalized.issues,
                cleaned.issues,
                parsed.issues,
                chunked.issues,
                relationships.issues,
            )
        )
        result = OfflineBuildResult(
            dataset_manifest=dataset_manifest,
            artifact_manifests={
                manifest.artifact_type.value: manifest
                for manifest in stored_manifests
            },
            output_paths=output_paths,
            audit_issue_count=len(audit.issues),
            processing_issue_count=processing_issue_count,
        )
        _LOGGER.info(
            "offline_runtime_build_completed",
            extra={
                "dataset_name": dataset_manifest.dataset_name,
                "document_count": len(normalized.documents),
                "chunk_count": len(chunked.chunks),
                "relationship_count": len(relationships.relationships),
                "audit_issue_count": len(audit.issues),
                "processing_issue_count": processing_issue_count,
            },
        )
        return result

    def _preflight_destinations(self) -> None:
        root = self._config.artifacts.root_path
        targets = [
            root / "dataset_manifest.json",
            *[
                self._directory(field)
                for field in (
                    "audit_directory",
                    "normalized_documents_directory",
                    "cleaned_documents_directory",
                    "legal_blocks_directory",
                    "legal_chunks_directory",
                    "relationships_directory",
                    "bm25_directory",
                    "vector_directory",
                    "graph_directory",
                )
            ],
        ]
        existing = [path for path in targets if path.exists()]
        if existing:
            raise ArtifactCompatibilityError(
                "Runtime artifact destination already contains a build"
            )

    def _directory(self, field_name: str) -> Path:
        return self._config.artifacts.directory(field_name)
