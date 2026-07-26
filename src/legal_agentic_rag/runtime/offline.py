"""Complete offline composition from AIO source streams to persisted indexes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
import gc
import logging
from pathlib import Path

from pydantic import ValidationError

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.configuration.hashing import canonical_sha256
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
from legal_agentic_rag.offline.document_processing import (
    StreamingDocumentProcessor,
)
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.offline.relationships import (
    load_relationship_artifact,
    persist_relationship_artifact,
)
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    DatasetAuditReport,
    DatasetManifest,
    LegalBlock,
    LegalChunk,
    LegalDocument,
    OfflineBuildState,
    OfflineBuildResult,
)
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.runtime.artifact_store import (
    load_artifact_manifest,
    load_dataset_manifest,
    load_model_artifact,
    persist_dataset_manifest,
    persist_model_artifact,
    stream_model_artifact,
)
from legal_agentic_rag.runtime.build_validation import (
    ArtifactSetValidator,
    persist_build_validation_report,
)

_LOGGER = logging.getLogger(__name__)
_BUILD_STATE_FILENAME = "build_state.json"
_COMPATIBLE_RESUME_UPGRADES = {("0.20.0", "0.20.1")}


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
        self._bm25: BM25Backend | None = bm25_backend or SQLiteFTS5BM25Backend(
            config.offline.bm25
        )
        self._vector: VectorBackend | None = vector_backend or NumpyVectorBackend(
            config.offline.vector_index
        )
        self._graph: GraphBackend | None = graph_backend or AdjacencyGraphBackend(
            config.offline.graph_index
        )

    def build(self) -> OfflineBuildResult:
        """Build or safely resume one immutable artifact set stage by stage."""
        resuming = self._preflight_destinations()
        root = self._config.artifacts.root_path
        output_paths: dict[str, str] = {}
        manifests: dict[ArtifactType, ArtifactManifest] = {}
        processing_issue_count = 0
        relationship_records: Iterable[Mapping[str, object]] | None = None

        if resuming:
            dataset_manifest = load_dataset_manifest(root)
            self._validate_resume_dataset(dataset_manifest)
            audit = self._load_audit_report()
            audit_issue_count = len(audit.issues)
            del audit
        else:
            self._persist_build_state()
            bounded_passes = (
                self._config.offline.execution.bounded_source_passes
            )
            if bounded_passes:
                self._observe_source_counts()
                metadata_records = self._source.iter_records(
                    DatasetComponent.METADATA
                )
                content_records = self._source.iter_records(
                    DatasetComponent.CONTENT
                )
                audit_relationship_records = self._source.iter_records(
                    DatasetComponent.RELATIONSHIPS
                )
            else:
                metadata_records = list(
                    self._source.iter_records(DatasetComponent.METADATA)
                )
                content_records = list(
                    self._source.iter_records(DatasetComponent.CONTENT)
                )
                audit_relationship_records = list(
                    self._source.iter_records(DatasetComponent.RELATIONSHIPS)
                )
            dataset_manifest = self._source.dataset_manifest()
            output_paths["dataset_manifest"] = str(
                persist_dataset_manifest(dataset_manifest, root)
            )
            audit = DatasetAuditService(self._config.offline.audit).audit(
                metadata_records=metadata_records,
                content_records=content_records,
                relationship_records=audit_relationship_records,
                manifest=dataset_manifest,
            )
            audit_paths = DatasetAuditReportWriter().write(
                audit,
                self._directory("audit_directory"),
            )
            output_paths["audit"] = str(audit_paths["data_audit.json"])
            audit_issue_count = len(audit.issues)
            del audit, audit_relationship_records
            if bounded_passes:
                metadata_records = self._source.iter_records(
                    DatasetComponent.METADATA
                )
                content_records = self._source.iter_records(
                    DatasetComponent.CONTENT
                )
            normalized_result = AioDocumentNormalizer(
                self._config.offline.normalization
            ).normalize(
                metadata_records=metadata_records,
                content_records=content_records,
                dataset_manifest=dataset_manifest,
            )
            normalized_documents = normalized_result.documents
            normalized_manifest = persist_model_artifact(
                records=normalized_documents,
                destination=self._directory("normalized_documents_directory"),
                manifest=normalized_result.manifest,
            )
            processing_issue_count += len(normalized_result.issues)
            del normalized_result, metadata_records, content_records
            self._release_stage_memory("raw_ingestion")

        output_paths.setdefault(
            "dataset_manifest",
            str(root / "dataset_manifest.json"),
        )
        output_paths.setdefault(
            "audit",
            str(self._directory("audit_directory") / "data_audit.json"),
        )
        if resuming:
            needs_normalized_payload = any(
                not self._directory(field).exists()
                for field in (
                    "relationships_directory",
                    "graph_directory",
                    "cleaned_documents_directory",
                )
            )
            if needs_normalized_payload:
                normalized_documents, normalized_manifest = load_model_artifact(
                    self._directory("normalized_documents_directory"),
                    expected_type=ArtifactType.NORMALIZED_DOCUMENTS,
                    record_type=LegalDocument,
                )
            else:
                normalized_documents = []
                normalized_manifest = load_artifact_manifest(
                    self._directory("normalized_documents_directory"),
                    expected_type=ArtifactType.NORMALIZED_DOCUMENTS,
                    verify_payload=True,
                )
            processing_issue_count += self._manifest_issue_count(
                normalized_manifest,
                "normalization_issue_count",
            )
        manifests[ArtifactType.NORMALIZED_DOCUMENTS] = normalized_manifest
        output_paths["normalized_documents"] = str(
            self._directory("normalized_documents_directory")
        )
        normalized_hash = normalized_manifest.processing_config_hash
        document_count = normalized_manifest.record_count

        relationship_directory = self._directory("relationships_directory")
        graph_directory = self._directory("graph_directory")
        relationship_values = None
        if relationship_directory.exists():
            if graph_directory.exists():
                relationship_manifest = load_artifact_manifest(
                    relationship_directory,
                    expected_type=ArtifactType.RELATIONSHIP_MAPPING,
                )
            else:
                relationship_values, relationship_manifest = (
                    load_relationship_artifact(
                        source=relationship_directory,
                        supplied_manifest=load_artifact_manifest(
                            relationship_directory,
                            expected_type=ArtifactType.RELATIONSHIP_MAPPING,
                        ),
                    )
                )
            processing_issue_count += self._relationship_issue_count(
                relationship_manifest
            )
        else:
            if relationship_records is None:
                relationship_records = self._source.iter_records(
                    DatasetComponent.RELATIONSHIPS
                )
            relationship_result = AioRelationshipNormalizer(
                self._config.offline.relationship_normalization
            ).normalize(
                relationship_records=relationship_records,
                documents=normalized_documents,
                document_manifest=normalized_manifest,
            )
            relationship_values = relationship_result.relationships
            relationship_manifest = persist_relationship_artifact(
                relationships=relationship_values,
                destination=relationship_directory,
                manifest=relationship_result.manifest,
            )
            processing_issue_count += len(relationship_result.issues)
            del relationship_result
        manifests[ArtifactType.RELATIONSHIP_MAPPING] = relationship_manifest
        output_paths["relationship_mapping"] = str(relationship_directory)
        relationship_count = relationship_manifest.record_count

        if graph_directory.exists():
            graph_manifest = load_artifact_manifest(
                graph_directory,
                expected_type=ArtifactType.GRAPH_INDEX,
            )
        else:
            if relationship_values is None:
                relationship_values, relationship_manifest = (
                    load_relationship_artifact(
                        source=relationship_directory,
                        supplied_manifest=relationship_manifest,
                    )
                )
            graph_backend = self._require_backend(self._graph, "graph")
            graph_backend.build(
                normalized_documents,
                relationship_values,
                document_manifest=normalized_manifest,
                relationship_manifest=relationship_manifest,
            )
            graph_manifest = graph_backend.persist(graph_directory)
            self._graph = None
            del graph_backend
        manifests[ArtifactType.GRAPH_INDEX] = graph_manifest
        output_paths["graph_index"] = str(graph_directory)
        del relationship_values, relationship_records
        self._release_stage_memory("relationship_graph")

        cleaned_directory = self._directory("cleaned_documents_directory")
        blocks_directory = self._directory("legal_blocks_directory")
        chunks_directory = self._directory("legal_chunks_directory")
        if chunks_directory.exists():
            chunk_manifest = load_artifact_manifest(
                chunks_directory,
                expected_type=ArtifactType.LEGAL_CHUNKS,
                verify_payload=True,
            )
            processing_issue_count += self._manifest_issue_count(
                chunk_manifest,
                "chunking_issue_count",
            )
            cleaned_manifest = load_artifact_manifest(
                cleaned_directory,
                expected_type=ArtifactType.CLEANED_DOCUMENTS,
                verify_payload=True,
            )
            block_manifest = load_artifact_manifest(
                blocks_directory,
                expected_type=ArtifactType.LEGAL_BLOCKS,
                verify_payload=True,
            )
            processing_issue_count += self._manifest_issue_count(
                cleaned_manifest,
                "cleaning_issue_count",
            )
            processing_issue_count += self._manifest_issue_count(
                block_manifest,
                "parser_issue_count",
            )
            del normalized_documents
            self._release_stage_memory("normalized_documents")
        else:
            if cleaned_directory.exists():
                cleaned_manifest = load_artifact_manifest(
                    cleaned_directory,
                    expected_type=ArtifactType.CLEANED_DOCUMENTS,
                    verify_payload=True,
                )
                processing_issue_count += self._manifest_issue_count(
                    cleaned_manifest,
                    "cleaning_issue_count",
                )
            else:
                cleaned_result = LegalHtmlCleaner(
                    self._config.offline.html_cleaning
                ).clean(
                    documents=normalized_documents,
                    source_manifest=normalized_manifest,
                )
                cleaned_documents = cleaned_result.documents
                cleaned_manifest = persist_model_artifact(
                    records=cleaned_documents,
                    destination=cleaned_directory,
                    manifest=cleaned_result.manifest,
                )
                processing_issue_count += len(cleaned_result.issues)
                del cleaned_result, cleaned_documents
            manifests[ArtifactType.CLEANED_DOCUMENTS] = cleaned_manifest
            output_paths["cleaned_documents"] = str(cleaned_directory)
            del normalized_documents
            self._release_stage_memory("normalized_html")

            cleaned_documents, verified_cleaned_manifest = stream_model_artifact(
                cleaned_directory,
                expected_type=ArtifactType.CLEANED_DOCUMENTS,
                record_type=LegalDocument,
            )
            if verified_cleaned_manifest != cleaned_manifest:
                raise ArtifactCompatibilityError(
                    "Cleaned document manifest changed during processing"
                )
            processing_documents = self._documents_without_html(
                cleaned_documents
            )
            processor = StreamingDocumentProcessor(
                LegalStructureParser(
                    self._config.offline.legal_structure_parser
                ),
                LegalChunker(self._config.offline.chunking),
                progress_interval_documents=(
                    self._config.offline.execution
                    .document_processing_progress_interval
                ),
            )
            if blocks_directory.exists():
                blocks, block_manifest = stream_model_artifact(
                    blocks_directory,
                    expected_type=ArtifactType.LEGAL_BLOCKS,
                    record_type=LegalBlock,
                )
                processed = processor.chunk_existing_blocks(
                    documents=processing_documents,
                    blocks=blocks,
                    source_manifest=block_manifest,
                    normalized_processing_config_hash=normalized_hash,
                    chunks_destination=chunks_directory,
                )
            else:
                processed = processor.process(
                    documents=processing_documents,
                    source_manifest=cleaned_manifest,
                    normalized_processing_config_hash=normalized_hash,
                    blocks_destination=blocks_directory,
                    chunks_destination=chunks_directory,
                )
            block_manifest = processed.block_manifest
            chunk_manifest = processed.chunk_manifest
            processing_issue_count += (
                processed.parser_issue_count
                + processed.chunking_issue_count
            )
            manifests[ArtifactType.LEGAL_BLOCKS] = block_manifest
            output_paths["legal_blocks"] = str(blocks_directory)
            del processed, processor, processing_documents
            self._release_stage_memory("document_processing")

        manifests.setdefault(ArtifactType.CLEANED_DOCUMENTS, cleaned_manifest)
        manifests.setdefault(ArtifactType.LEGAL_BLOCKS, block_manifest)
        manifests[ArtifactType.LEGAL_CHUNKS] = chunk_manifest
        output_paths["cleaned_documents"] = str(
            self._directory("cleaned_documents_directory")
        )
        output_paths["legal_blocks"] = str(
            self._directory("legal_blocks_directory")
        )
        output_paths["legal_chunks"] = str(chunks_directory)
        chunk_count = chunk_manifest.record_count

        bm25_directory = self._directory("bm25_directory")
        if bm25_directory.exists():
            bm25_manifest = load_artifact_manifest(
                bm25_directory,
                expected_type=ArtifactType.BM25_INDEX,
            )
        else:
            bm25_backend = self._require_backend(self._bm25, "BM25")
            chunks, verified_chunk_manifest = stream_model_artifact(
                chunks_directory,
                expected_type=ArtifactType.LEGAL_CHUNKS,
                record_type=LegalChunk,
            )
            if verified_chunk_manifest != chunk_manifest:
                raise ArtifactCompatibilityError(
                    "Legal chunk manifest changed during BM25 build"
                )
            bm25_backend.build(chunks, chunk_manifest)
            bm25_manifest = bm25_backend.persist(bm25_directory)
            close = getattr(bm25_backend, "close", None)
            if callable(close):
                close()
            self._bm25 = None
            del bm25_backend
            self._release_stage_memory("bm25")
        manifests[ArtifactType.BM25_INDEX] = bm25_manifest
        output_paths["bm25_index"] = str(bm25_directory)

        vector_directory = self._directory("vector_directory")
        if vector_directory.exists():
            vector_manifest = load_artifact_manifest(
                vector_directory,
                expected_type=ArtifactType.VECTOR_INDEX,
            )
        else:
            vector_backend = self._require_backend(self._vector, "vector")
            chunks, verified_chunk_manifest = stream_model_artifact(
                chunks_directory,
                expected_type=ArtifactType.LEGAL_CHUNKS,
                record_type=LegalChunk,
            )
            if verified_chunk_manifest != chunk_manifest:
                raise ArtifactCompatibilityError(
                    "Legal chunk manifest changed during vector build"
                )
            vector_manifest = VectorIndexBuilder(
                self._embedding_provider,
                vector_backend,
                self._config.offline.vector_index,
            ).build_persisted(
                chunks,
                chunk_manifest,
                vector_directory,
            )
            self._vector = None
            del vector_backend
        manifests[ArtifactType.VECTOR_INDEX] = vector_manifest
        output_paths["vector_index"] = str(vector_directory)
        self._release_stage_memory("vector")

        validation_report = ArtifactSetValidator(
            self._config.artifacts,
            self._config.build_validation,
        ).validate()
        validation_path = persist_build_validation_report(
            validation_report,
            root,
            self._config.build_validation.report_filename,
        )
        output_paths["build_validation"] = str(validation_path)
        if not validation_report.is_valid:
            raise ArtifactCompatibilityError(
                "Offline artifact set failed post-build validation"
            )
        result = OfflineBuildResult(
            dataset_manifest=dataset_manifest,
            artifact_manifests={
                artifact_type.value: manifest
                for artifact_type, manifest in manifests.items()
            },
            output_paths=output_paths,
            audit_issue_count=audit_issue_count,
            processing_issue_count=processing_issue_count,
            validation_report=validation_report,
        )
        _LOGGER.info(
            "offline_runtime_build_completed",
            extra={
                "dataset_name": dataset_manifest.dataset_name,
                "document_count": document_count,
                "chunk_count": chunk_count,
                "relationship_count": relationship_count,
                "audit_issue_count": audit_issue_count,
                "processing_issue_count": processing_issue_count,
                "resumed": resuming,
            },
        )
        return result

    def _preflight_destinations(self) -> bool:
        root = self._config.artifacts.root_path
        targets = [
            root / _BUILD_STATE_FILENAME,
            root / "dataset_manifest.json",
            root / self._config.build_validation.report_filename,
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
        if not existing:
            return False
        if not self._config.offline.execution.resume_partial_build:
            raise ArtifactCompatibilityError(
                "Runtime artifact destination already contains a build"
            )
        if (root / self._config.build_validation.report_filename).exists():
            raise ArtifactCompatibilityError(
                "Completed or failed validation report cannot be resumed"
            )
        required = (
            root / _BUILD_STATE_FILENAME,
            root / "dataset_manifest.json",
            self._directory("audit_directory"),
            self._directory("normalized_documents_directory"),
        )
        if not all(path.exists() for path in required):
            raise ArtifactCompatibilityError(
                "Partial build cannot resume before normalized checkpoint"
            )
        self._validate_partial_stage_dependencies()
        self._validate_build_state()
        return True

    def _persist_build_state(self) -> None:
        root = self._config.artifacts.root_path
        root.mkdir(parents=True, exist_ok=True)
        path = root / _BUILD_STATE_FILENAME
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                state = OfflineBuildState(
                    application_config_hash=self._config_hash(),
                    code_version=__version__,
                    created_at=datetime.now(UTC),
                )
                stream.write(state.model_dump_json() + "\n")
        except OSError as error:
            raise ArtifactCompatibilityError(
                "Build state could not be persisted"
            ) from error

    def _validate_build_state(self) -> None:
        try:
            state = OfflineBuildState.model_validate_json(
                (
                    self._config.artifacts.root_path / _BUILD_STATE_FILENAME
                ).read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Partial build state is missing or invalid"
            ) from error
        if state.schema_version != "1.1":
            raise ArtifactCompatibilityError(
                "Partial build state uses an incompatible hash format"
            )
        if state.application_config_hash != self._config_hash():
            raise ArtifactCompatibilityError(
                "Partial build configuration is incompatible"
            )
        if state.code_version != __version__:
            if (state.code_version, __version__) not in _COMPATIBLE_RESUME_UPGRADES:
                raise ArtifactCompatibilityError(
                    "Partial build code version is incompatible"
                )
            self._upgrade_build_state(state)

    def _upgrade_build_state(self, state: OfflineBuildState) -> None:
        """Atomically record an explicitly supported recovery-only code upgrade."""
        path = self._config.artifacts.root_path / _BUILD_STATE_FILENAME
        temporary = path.with_name(f".{path.name}.tmp")
        upgraded = state.model_copy(update={"code_version": __version__})
        try:
            temporary.write_text(
                upgraded.model_dump_json() + "\n",
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ArtifactCompatibilityError(
                "Partial build state could not be upgraded"
            ) from error
        _LOGGER.info(
            "offline_build_state_upgraded",
            extra={
                "previous_code_version": state.code_version,
                "code_version": __version__,
            },
        )

    def _validate_partial_stage_dependencies(self) -> None:
        paths = {
            name: self._directory(field)
            for name, field in (
                ("cleaned", "cleaned_documents_directory"),
                ("blocks", "legal_blocks_directory"),
                ("chunks", "legal_chunks_directory"),
                ("relationships", "relationships_directory"),
                ("bm25", "bm25_directory"),
                ("vector", "vector_directory"),
                ("graph", "graph_directory"),
            )
        }
        invalid = (
            (paths["blocks"].exists() and not paths["cleaned"].exists())
            or (
                paths["chunks"].exists()
                and not all(
                    paths[name].exists() for name in ("cleaned", "blocks")
                )
            )
            or (
                paths["bm25"].exists() and not paths["chunks"].exists()
            )
            or (
                paths["vector"].exists() and not paths["chunks"].exists()
            )
            or (
                paths["graph"].exists()
                and not paths["relationships"].exists()
            )
        )
        if invalid:
            raise ArtifactCompatibilityError(
                "Partial build stage dependencies are incomplete"
            )

    def _config_hash(self) -> str:
        return canonical_sha256(self._config)

    def _validate_resume_dataset(self, manifest: DatasetManifest) -> None:
        if (
            manifest.dataset_name != self._source.dataset_name
            or manifest.dataset_revision != self._source.dataset_revision
        ):
            raise ArtifactCompatibilityError(
                "Partial build dataset identity is incompatible"
            )

    def _load_audit_report(self) -> DatasetAuditReport:
        try:
            return DatasetAuditReport.model_validate_json(
                (
                    self._directory("audit_directory") / "data_audit.json"
                ).read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError(
                "Partial build audit report is missing or invalid"
            ) from error

    def _release_stage_memory(self, stage: str) -> None:
        if self._config.offline.execution.release_stage_memory:
            collected = gc.collect()
            _LOGGER.info(
                "offline_stage_memory_released",
                extra={"stage": stage, "collected_objects": collected},
            )

    def _observe_source_counts(self) -> None:
        """Complete one count-only pass before bounded repeatable source reads."""
        for component in DatasetComponent:
            for _ in self._source.iter_records(component):
                pass

    @staticmethod
    def _manifest_issue_count(
        manifest: ArtifactManifest,
        metadata_key: str,
    ) -> int:
        value = manifest.metadata.get(metadata_key, 0)
        return value if isinstance(value, int) and value >= 0 else 0

    @staticmethod
    def _relationship_issue_count(manifest: ArtifactManifest) -> int:
        rejected = manifest.metadata.get("rejected_count", 0)
        duplicates = manifest.metadata.get("duplicate_count", 0)
        return sum(
            value
            for value in (rejected, duplicates)
            if isinstance(value, int) and value >= 0
        )

    @staticmethod
    def _documents_without_html(
        documents: Iterable[LegalDocument],
    ) -> Iterable[LegalDocument]:
        """Drop HTML references after the cleaned artifact checksum passes."""
        for document in documents:
            document.content_html = None
            yield document

    @staticmethod
    def _require_backend(backend: object | None, label: str) -> object:
        if backend is None:
            raise ArtifactCompatibilityError(
                f"{label} backend is unavailable for this build stage"
            )
        return backend

    def _directory(self, field_name: str) -> Path:
        return self._config.artifacts.directory(field_name)
