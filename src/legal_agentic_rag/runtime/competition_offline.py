"""Staged, resumable offline assembly for the official UIT DSC corpus."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026 import UitDsc2026CorpusIngestor
from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.contracts import (
    BM25Backend,
    EmbeddingProvider,
    VectorBackend,
)
from legal_agentic_rag.embeddings import SentenceTransformerEmbeddingProvider
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.vector import NumpyVectorBackend, VectorIndexBuilder
from legal_agentic_rag.offline.chunking import LegalChunker
from legal_agentic_rag.offline.chunking.tokenizer import EmbeddingModelTokenizer
from legal_agentic_rag.offline.document_processing import StreamingDocumentProcessor
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.runtime.artifact_store import (
    load_artifact_manifest,
    load_dataset_manifest,
    persist_dataset_manifest,
    persist_model_artifact,
    stream_model_artifact,
)
from legal_agentic_rag.runtime.build_validation import (
    COMPETITION_REQUIRED_ARTIFACT_TYPES,
    ArtifactSetValidator,
    persist_build_validation_report,
)
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    BuildValidationReport,
    CompetitionBuildStage,
    CompetitionBuildState,
    CompetitionCorpusAuditReport,
    CompetitionOfflineBuildResult,
    DatasetManifest,
    LegalBlock,
    LegalChunk,
    LegalDocument,
)

_LOGGER = logging.getLogger(__name__)
_STATE_FILENAME = "competition_build_state.json"
_AUDIT_FILENAME = "corpus_audit.json"


class CompetitionOfflineBuildRuntime:
    """Build official artifacts stage by stage with fail-closed recovery."""

    def __init__(
        self,
        config: ApplicationConfig,
        source: Path,
        *,
        ingestor: UitDsc2026CorpusIngestor | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        bm25_backend: BM25Backend | None = None,
        vector_backend: VectorBackend | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._source = source
        self._ingestor = ingestor or UitDsc2026CorpusIngestor()
        self._embedding_provider = embedding_provider
        self._bm25 = bm25_backend
        self._vector = vector_backend
        self._clock = clock or (lambda: datetime.now(UTC))

    def build(
        self,
        *,
        through: CompetitionBuildStage = CompetitionBuildStage.VALIDATION,
    ) -> CompetitionOfflineBuildResult:
        """Build or resume official artifacts through one requested stage."""
        source_identity = self._ingestor.inspect_source(self._source)
        state, resumed = self._prepare_state(source_identity.revision)
        if CompetitionBuildStage.CORPUS not in state.completed_stages:
            self._build_corpus_stage()
            self._validate_corpus_stage(source_identity.revision)
            state = self._complete_stage(state, CompetitionBuildStage.CORPUS)
        else:
            self._validate_corpus_stage(source_identity.revision)
        if through == CompetitionBuildStage.CORPUS:
            return self._build_result(state, resumed)

        if CompetitionBuildStage.DOCUMENT_PROCESSING not in state.completed_stages:
            self._build_document_stage()
            self._validate_document_stage()
            state = self._complete_stage(
                state, CompetitionBuildStage.DOCUMENT_PROCESSING
            )
        else:
            self._validate_document_stage()
        if through == CompetitionBuildStage.DOCUMENT_PROCESSING:
            return self._build_result(state, resumed)

        if CompetitionBuildStage.BM25 not in state.completed_stages:
            self._build_bm25_stage()
            self._validate_bm25_stage()
            state = self._complete_stage(state, CompetitionBuildStage.BM25)
        else:
            self._validate_bm25_stage()
        if through == CompetitionBuildStage.BM25:
            return self._build_result(state, resumed)

        if CompetitionBuildStage.VECTOR not in state.completed_stages:
            self._build_vector_stage()
            self._validate_vector_stage()
            state = self._complete_stage(state, CompetitionBuildStage.VECTOR)
        else:
            self._validate_vector_stage()
        if through == CompetitionBuildStage.VECTOR:
            return self._build_result(state, resumed)

        report = self._validation_report()
        if CompetitionBuildStage.VALIDATION not in state.completed_stages:
            state = self._complete_stage(state, CompetitionBuildStage.VALIDATION)

        return CompetitionOfflineBuildResult(
            artifact_root=str(self._root),
            source_revision=source_identity.revision,
            resumed=resumed,
            completed_stages=state.completed_stages,
            validation_report=report,
        )

    def _build_result(
        self,
        state: CompetitionBuildState,
        resumed: bool,
    ) -> CompetitionOfflineBuildResult:
        report = (
            self._validation_report()
            if CompetitionBuildStage.VALIDATION in state.completed_stages
            else None
        )
        return CompetitionOfflineBuildResult(
            artifact_root=str(self._root),
            source_revision=state.source_revision,
            resumed=resumed,
            completed_stages=state.completed_stages,
            validation_report=report,
        )

    @property
    def _root(self) -> Path:
        return self._config.artifacts.root_path.resolve()

    def _directory(self, field: str) -> Path:
        return self._config.artifacts.directory(field).resolve()

    def _config_hash(self) -> str:
        return canonical_sha256(self._config)

    def _prepare_state(self, source_revision: str) -> tuple[CompetitionBuildState, bool]:
        root = self._root
        state_path = root / _STATE_FILENAME
        if state_path.exists():
            state = self._load_state(state_path)
            if (
                state.source_revision != source_revision
                or state.application_config_hash != self._config_hash()
                or state.code_version != __version__
            ):
                raise ArtifactCompatibilityError(
                    "Partial competition build identity is incompatible"
                )
            return state, True
        if root.exists() and any(root.iterdir()):
            raise ArtifactCompatibilityError(
                "Non-empty artifact root has no competition build state"
            )
        root.mkdir(parents=True, exist_ok=True)
        now = self._clock()
        state = CompetitionBuildState(
            source_revision=source_revision,
            application_config_hash=self._config_hash(),
            code_version=__version__,
            created_at=now,
            updated_at=now,
        )
        self._write_json_atomic(state_path, state.model_dump(mode="json"))
        return state, False

    def _complete_stage(
        self,
        state: CompetitionBuildState,
        stage: CompetitionBuildStage,
    ) -> CompetitionBuildState:
        expected = list(CompetitionBuildStage)[len(state.completed_stages)]
        if stage != expected:
            raise ArtifactCompatibilityError("Competition build stage order is invalid")
        updated = state.model_copy(
            update={
                "completed_stages": [*state.completed_stages, stage],
                "updated_at": self._clock(),
            }
        )
        self._write_json_atomic(
            self._root / _STATE_FILENAME, updated.model_dump(mode="json")
        )
        _LOGGER.info("competition_build_stage_completed", extra={"stage": stage.value})
        return updated

    def _build_corpus_stage(self) -> None:
        result = self._ingestor.ingest(self._source)
        dataset_path = self._root / "dataset_manifest.json"
        if dataset_path.exists():
            stored = load_dataset_manifest(self._root)
            if not self._same_dataset_identity(stored, result.dataset_manifest):
                raise ArtifactCompatibilityError("Dataset manifest is incompatible")
        else:
            persist_dataset_manifest(result.dataset_manifest, self._root)
        self._persist_or_validate_documents(
            result.normalized_documents,
            result.normalized_manifest,
            "normalized_documents_directory",
        )
        self._persist_or_validate_documents(
            result.cleaned_documents,
            result.cleaned_manifest,
            "cleaned_documents_directory",
        )
        self._persist_or_validate_audit(result.audit)

    def _build_document_stage(self) -> None:
        blocks_directory = self._directory("legal_blocks_directory")
        chunks_directory = self._directory("legal_chunks_directory")
        if chunks_directory.exists() and not blocks_directory.exists():
            raise ArtifactCompatibilityError(
                "Legal chunks cannot resume without legal blocks"
            )
        if blocks_directory.exists() and chunks_directory.exists():
            self._validate_model_artifact("legal_blocks_directory", ArtifactType.LEGAL_BLOCKS)
            self._validate_model_artifact("legal_chunks_directory", ArtifactType.LEGAL_CHUNKS)
            return
        documents, cleaned_manifest = stream_model_artifact(
            self._directory("cleaned_documents_directory"),
            expected_type=ArtifactType.CLEANED_DOCUMENTS,
            record_type=LegalDocument,
        )
        normalized_manifest = load_artifact_manifest(
            self._directory("normalized_documents_directory"),
            expected_type=ArtifactType.NORMALIZED_DOCUMENTS,
            verify_payload=True,
        )
        processor = StreamingDocumentProcessor(
            LegalStructureParser(self._config.offline.legal_structure_parser),
            LegalChunker(
                self._config.offline.chunking,
                tokenizer=(
                    EmbeddingModelTokenizer(self._config.offline.embedding)
                    if self._config.offline.chunking.tokenizer_name
                    == "embedding_model_v1"
                    else None
                ),
            ),
            progress_interval_documents=self._config.offline.index_build.batch_size,
        )
        if blocks_directory.exists():
            blocks, block_manifest = stream_model_artifact(
                blocks_directory,
                expected_type=ArtifactType.LEGAL_BLOCKS,
                record_type=LegalBlock,
            )
            processor.chunk_existing_blocks(
                documents=documents,
                blocks=blocks,
                source_manifest=block_manifest,
                normalized_processing_config_hash=(
                    normalized_manifest.processing_config_hash
                ),
                chunks_destination=chunks_directory,
            )
        else:
            processor.process(
                documents=documents,
                source_manifest=cleaned_manifest,
                normalized_processing_config_hash=(
                    normalized_manifest.processing_config_hash
                ),
                blocks_destination=blocks_directory,
                chunks_destination=chunks_directory,
            )

    def _build_bm25_stage(self) -> None:
        destination = self._directory("bm25_directory")
        if destination.exists():
            self._validate_bm25_stage()
            return
        chunks, manifest = stream_model_artifact(
            self._directory("legal_chunks_directory"),
            expected_type=ArtifactType.LEGAL_CHUNKS,
            record_type=LegalChunk,
        )
        backend = self._bm25 or SQLiteFTS5BM25Backend(self._config.offline.bm25)
        backend.build(chunks, manifest)
        backend.persist(destination)
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    def _build_vector_stage(self) -> None:
        destination = self._directory("vector_directory")
        if destination.exists():
            self._validate_vector_stage()
            return
        chunks, manifest = stream_model_artifact(
            self._directory("legal_chunks_directory"),
            expected_type=ArtifactType.LEGAL_CHUNKS,
            record_type=LegalChunk,
        )
        provider = self._embedding_provider or SentenceTransformerEmbeddingProvider(
            self._config.offline.embedding
        )
        backend = self._vector or NumpyVectorBackend(
            self._config.offline.vector_index
        )
        VectorIndexBuilder(
            provider, backend, self._config.offline.vector_index
        ).build_persisted(chunks, manifest, destination)

    def _validation_report(self) -> BuildValidationReport:
        report_path = self._root / self._config.build_validation.report_filename
        if report_path.exists():
            try:
                report = BuildValidationReport.model_validate_json(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as error:
                raise ArtifactCompatibilityError(
                    "Existing build validation report is invalid"
                ) from error
        else:
            report = ArtifactSetValidator(
                self._config.artifacts,
                self._config.build_validation,
                required_artifact_types=COMPETITION_REQUIRED_ARTIFACT_TYPES,
                clock=self._clock,
            ).validate()
            persist_build_validation_report(
                report,
                self._root,
                self._config.build_validation.report_filename,
            )
        if not report.is_valid:
            raise ArtifactCompatibilityError("Official artifact set failed validation")
        return report

    def _validate_corpus_stage(self, source_revision: str) -> None:
        dataset = load_dataset_manifest(self._root)
        if dataset.dataset_revision != source_revision:
            raise ArtifactCompatibilityError("Official dataset revision changed")
        self._validate_model_artifact(
            "normalized_documents_directory", ArtifactType.NORMALIZED_DOCUMENTS
        )
        self._validate_model_artifact(
            "cleaned_documents_directory", ArtifactType.CLEANED_DOCUMENTS
        )
        audit_path = self._directory("audit_directory") / _AUDIT_FILENAME
        try:
            audit = CompetitionCorpusAuditReport.model_validate_json(
                audit_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ArtifactCompatibilityError("Official corpus audit is invalid") from error
        if audit.dataset_revision != source_revision:
            raise ArtifactCompatibilityError("Official corpus audit revision changed")

    def _validate_bm25_stage(self) -> None:
        destination = self._directory("bm25_directory")
        manifest = load_artifact_manifest(
            destination, expected_type=ArtifactType.BM25_INDEX
        )
        backend = self._bm25 or SQLiteFTS5BM25Backend(self._config.offline.bm25)
        backend.load(destination, manifest)
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    def _validate_document_stage(self) -> None:
        self._validate_model_artifact(
            "legal_blocks_directory", ArtifactType.LEGAL_BLOCKS
        )
        self._validate_model_artifact(
            "legal_chunks_directory", ArtifactType.LEGAL_CHUNKS
        )

    def _validate_vector_stage(self) -> None:
        destination = self._directory("vector_directory")
        manifest = load_artifact_manifest(
            destination, expected_type=ArtifactType.VECTOR_INDEX
        )
        backend = self._vector or NumpyVectorBackend(
            self._config.offline.vector_index
        )
        backend.load(destination, manifest)

    def _validate_model_artifact(self, field: str, artifact_type: ArtifactType) -> None:
        load_artifact_manifest(
            self._directory(field),
            expected_type=artifact_type,
            verify_payload=True,
        )

    def _persist_or_validate_documents(
        self,
        documents: list[LegalDocument],
        expected: ArtifactManifest,
        field: str,
    ) -> None:
        destination = self._directory(field)
        if destination.exists():
            stored = load_artifact_manifest(
                destination,
                expected_type=expected.artifact_type,
                verify_payload=True,
            )
            if not self._same_manifest_identity(stored, expected):
                raise ArtifactCompatibilityError("Document artifact is incompatible")
            return
        persist_model_artifact(
            records=documents, destination=destination, manifest=expected
        )

    def _persist_or_validate_audit(self, audit: CompetitionCorpusAuditReport) -> None:
        directory = self._directory("audit_directory")
        path = directory / _AUDIT_FILENAME
        if path.exists():
            try:
                stored = CompetitionCorpusAuditReport.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as error:
                raise ArtifactCompatibilityError("Corpus audit is invalid") from error
            if stored != audit:
                raise ArtifactCompatibilityError("Corpus audit is incompatible")
            return
        directory.mkdir(exist_ok=True)
        self._write_json_atomic(path, audit.model_dump(mode="json"))

    @staticmethod
    def _same_manifest_identity(
        stored: ArtifactManifest, expected: ArtifactManifest
    ) -> bool:
        return (
            stored.artifact_type == expected.artifact_type
            and stored.artifact_version == expected.artifact_version
            and stored.dataset_name == expected.dataset_name
            and stored.dataset_revision == expected.dataset_revision
            and stored.record_count == expected.record_count
            and stored.processing_config_hash == expected.processing_config_hash
            and stored.code_version == expected.code_version
        )

    @staticmethod
    def _same_dataset_identity(
        stored: DatasetManifest, expected: DatasetManifest
    ) -> bool:
        return (
            stored.dataset_name == expected.dataset_name
            and stored.dataset_revision == expected.dataset_revision
            and stored.configs == expected.configs
            and stored.record_counts == expected.record_counts
            and stored.processing_config_hash == expected.processing_config_hash
            and stored.code_version == expected.code_version
        )

    @staticmethod
    def _load_state(path: Path) -> CompetitionBuildState:
        try:
            return CompetitionBuildState.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ArtifactCompatibilityError(
                "Competition build state is missing or invalid"
            ) from error

    @staticmethod
    def _write_json_atomic(path: Path, payload: object) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ArtifactCompatibilityError("Recovery state could not be persisted") from error
