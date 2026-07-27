"""End-to-end fixture build and online Agent runtime assembly."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from legal_agentic_rag.configuration import (
    AgentConfig,
    ApplicationConfig,
    ArtifactConfig,
    BuildValidationConfig,
    ChunkingConfig,
    DatasetSourceConfig,
    EmbeddingConfig,
    OfflineConfig,
    OfflineExecutionConfig,
    OnlineConfig,
    RelationshipNormalizationConfig,
    RetrievalConfig,
    StartupValidationConfig,
)
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
)
from legal_agentic_rag.indexing.vector import (
    NumpyVectorBackend,
    prepare_vector_serving_metadata,
)
from legal_agentic_rag.runtime import (
    ArtifactSetValidator,
    OfflineBuildRuntime,
    OnlineRuntimeFactory,
    load_artifact_manifest,
    load_model_artifact,
)
from legal_agentic_rag.schemas import (
    AgentStopReason,
    ArtifactType,
    DatasetManifest,
    LegalDocument,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)


class _FixtureSource:
    dataset_name = "th1nhng0/vietnamese-legal-documents"
    dataset_revision = "runtime-fixture"

    def __init__(self) -> None:
        self._records: dict[DatasetComponent, list[dict[str, object]]] = {
            DatasetComponent.METADATA: [
                {
                    "id": "doc-tax",
                    "title": "Luật thuế mẫu",
                    "so_ky_hieu": "01/2026/QH",
                    "loai_van_ban": "Luật",
                    "ngay_ban_hanh": "01/01/2026",
                    "ngay_co_hieu_luc": "01/02/2026",
                    "ngay_het_hieu_luc": None,
                    "tinh_trang_hieu_luc": "Còn hiệu lực",
                    "source_url": "https://example.invalid/doc-tax",
                },
                {
                    "id": "doc-company",
                    "title": "Luật doanh nghiệp mẫu",
                    "so_ky_hieu": "02/2026/QH",
                    "loai_van_ban": "Luật",
                    "ngay_ban_hanh": "02/01/2026",
                    "ngay_co_hieu_luc": "02/02/2026",
                    "ngay_het_hieu_luc": None,
                    "tinh_trang_hieu_luc": "Còn hiệu lực",
                    "source_url": "https://example.invalid/doc-company",
                },
            ],
            DatasetComponent.CONTENT: [
                {
                    "id": "doc-tax",
                    "content_html": (
                        "<article><p>Điều 1. Nghĩa vụ thuế</p>"
                        "<p>Doanh nghiệp phải nộp thuế đúng thời hạn.</p></article>"
                    ),
                },
                {
                    "id": "doc-company",
                    "content_html": (
                        "<article><p>Điều 2. Đăng ký doanh nghiệp</p>"
                        "<p>Doanh nghiệp phải đăng ký theo quy định.</p></article>"
                    ),
                },
            ],
            DatasetComponent.RELATIONSHIPS: [
                {
                    "doc_id": "doc-tax",
                    "other_doc_id": "doc-company",
                    "relationship": "Sửa đổi",
                }
            ],
        }

    def iter_records(
        self,
        component: DatasetComponent,
        limit: int | None = None,
    ) -> Sequence[Mapping[str, object]]:
        values = self._records[component]
        return values if limit is None else values[:limit]

    def dataset_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            schema_version="1.0",
            dataset_name=self.dataset_name,
            dataset_revision=self.dataset_revision,
            loaded_at=datetime(2026, 7, 23, tzinfo=UTC),
            configs=["metadata", "content", "relationships"],
            record_counts={
                component.value: len(records)
                for component, records in self._records.items()
            },
            processing_config_hash="runtime-dataset-hash",
            code_version="0.17.0",
        )


class _FixtureEmbeddingProvider:
    provider_name = "fixture-embedding"
    provider_version = "1.0"
    model_name = "fixture-legal-embedding"
    model_revision = "fixture-revision"
    dimension = 2

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "thuế" in text.casefold() else [0.0, 1.0]


class _FixtureReranker:
    provider_name = "fixture-reranker"
    provider_version = "1.0"
    model_name = "fixture-legal-reranker"
    model_revision = "fixture-revision"

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> RetrievalResponse:
        values = list(candidates)[: query.top_k]
        hits = [
            hit.model_copy(
                update={
                    "rank": rank,
                    "score": float(len(values) - rank + 1),
                    "strategy": RetrievalStrategy.RERANK,
                    "retrieval_trace": hit.retrieval_trace.model_copy(
                        update={
                            "reranker_score": float(
                                len(values) - rank + 1
                            )
                        }
                    ),
                }
            )
            for rank, hit in enumerate(values, start=1)
        ]
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.RERANK,
            hits=hits,
        )


def _config(
    root: Path,
    *,
    resume_partial_build: bool = False,
    bounded_source_passes: bool = False,
) -> ApplicationConfig:
    return ApplicationConfig(
        artifacts=ArtifactConfig(root_path=root),
        offline=OfflineConfig(
            dataset=DatasetSourceConfig(
                dataset_name="th1nhng0/vietnamese-legal-documents",
                dataset_revision="runtime-fixture",
            ),
            chunking=ChunkingConfig(
                max_tokens=100,
                min_tokens=1,
                overlap_tokens=10,
            ),
            embedding=EmbeddingConfig(
                model_name="fixture-legal-embedding",
                model_revision="fixture-revision",
                expected_dimension=2,
            ),
            relationship_normalization=RelationshipNormalizationConfig(
                relationship_type_mapping={"Sửa đổi": "amends"}
            ),
            execution=OfflineExecutionConfig(
                resume_partial_build=resume_partial_build,
                bounded_source_passes=bounded_source_passes,
            ),
        ),
        online=OnlineConfig(
            retrieval=RetrievalConfig(top_k=1, candidate_k=2),
            agent=AgentConfig(max_retry=2),
        ),
        build_validation=BuildValidationConfig(
            require_pinned_dataset_revision=True,
            require_full_corpus=True,
            expected_record_counts={
                "metadata": 2,
                "content": 2,
                "relationships": 1,
            },
        ),
    )


def test_fixture_offline_build_loads_into_answering_agent(
    tmp_path: Path,
) -> None:
    """A single artifact set flows from raw fixture records to verified answer."""
    config = _config(tmp_path / "artifacts")
    provider = _FixtureEmbeddingProvider()
    offline = OfflineBuildRuntime(
        config,
        source=_FixtureSource(),
        embedding_provider=provider,
    )

    build = offline.build()
    online = OnlineRuntimeFactory(
        config,
        embedding_provider=provider,
        reranker=_FixtureReranker(),
    ).build()
    result = online.answer(
        RetrievalQuery(
            query_id="runtime-query",
            original_question="Doanh nghiệp phải nộp thuế khi nào?",
            normalized_question="doanh nghiệp nộp thuế",
            top_k=1,
            candidate_k=2,
        )
    )

    assert len(build.artifact_manifests) == 8
    assert build.validation_report.is_valid is True
    assert build.validation_report.is_full_corpus is True
    assert (
        config.artifacts.root_path
        / config.build_validation.report_filename
    ).is_file()
    cleaned_records, _ = load_model_artifact(
        config.artifacts.directory("cleaned_documents_directory"),
        expected_type=ArtifactType.CLEANED_DOCUMENTS,
        record_type=LegalDocument,
    )
    assert all(document.content_html is not None for document in cleaned_records)
    assert len(online.manifests) == 4
    assert len(online.tool_descriptors()) == 8
    assert result.stop_reason == AgentStopReason.ANSWER_VERIFIED
    assert result.response.insufficient_evidence is False
    assert result.response.citations[0].document_id == "doc-tax"
    assert result.response.citations[0].article_number == "1"
    assert "nộp thuế đúng thời hạn" in result.response.answer
    with pytest.raises(ArtifactCompatibilityError, match="already"):
        offline.build()


def test_online_runtime_rejects_tampered_chunk_payload(
    tmp_path: Path,
) -> None:
    """Startup fails before serving when a source artifact checksum changes."""
    config = _config(tmp_path / "artifacts")
    provider = _FixtureEmbeddingProvider()
    OfflineBuildRuntime(
        config,
        source=_FixtureSource(),
        embedding_provider=provider,
    ).build()

    class _WrongDimensionProvider(_FixtureEmbeddingProvider):
        dimension = 3

    with pytest.raises(ArtifactCompatibilityError, match="embedding"):
        OnlineRuntimeFactory(
            config,
            embedding_provider=_WrongDimensionProvider(),
            reranker=_FixtureReranker(),
        ).build()

    chunk_payload = (
        config.artifacts.directory("legal_chunks_directory")
        / "records.jsonl"
    )
    with chunk_payload.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    validation = ArtifactSetValidator(
        config.artifacts,
        config.build_validation,
    ).validate()
    assert validation.is_valid is False
    assert (
        validation.artifact_results["legal_chunks"].errors
        == ["legal_chunks payload checksum mismatch"]
    )

    with pytest.raises(ArtifactCompatibilityError, match="checksum"):
        OnlineRuntimeFactory(
            config,
            embedding_provider=provider,
            reranker=_FixtureReranker(),
        ).build()


def test_validated_report_startup_reuses_prior_deep_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching immutable report skips repeated corpus-sized integrity scans."""
    config = _config(tmp_path / "artifacts")
    provider = _FixtureEmbeddingProvider()
    OfflineBuildRuntime(
        config,
        source=_FixtureSource(),
        embedding_provider=provider,
    ).build()
    vector_directory = config.artifacts.directory("vector_directory")
    vector_manifest = load_artifact_manifest(
        vector_directory,
        expected_type=ArtifactType.VECTOR_INDEX,
    )
    prepare_vector_serving_metadata(
        vector_directory=vector_directory,
        destination=config.artifacts.directory("vector_serving_directory"),
        vector_manifest=vector_manifest,
    )
    config.online.startup_validation = StartupValidationConfig(
        mode="validated_report"
    )
    config.online.vector_runtime.require_serving_metadata = True

    def unexpected_scan(*args: object, **kwargs: object) -> object:
        raise AssertionError("deep integrity scan must not run")

    monkeypatch.setattr(
        "legal_agentic_rag.runtime.artifact_store._sha256_file",
        unexpected_scan,
    )
    monkeypatch.setattr(
        "legal_agentic_rag.indexing.bm25.artifact_store._sha256_file",
        unexpected_scan,
    )
    monkeypatch.setattr(
        "legal_agentic_rag.indexing.vector.artifact_store._validate_checksum",
        unexpected_scan,
    )
    monkeypatch.setattr(
        "legal_agentic_rag.indexing.vector.artifact_store._validate_vector_rows",
        unexpected_scan,
    )
    monkeypatch.setattr(
        "legal_agentic_rag.indexing.vector.artifact_store.JsonlChunkStore.load",
        unexpected_scan,
    )
    monkeypatch.setattr(
        "legal_agentic_rag.indexing.graph.adjacency_backend."
        "AdjacencyGraphBackend._sha256_file",
        unexpected_scan,
    )

    online = OnlineRuntimeFactory(
        config,
        embedding_provider=provider,
        reranker=_FixtureReranker(),
    ).build()

    assert len(online.manifests) == 4


def test_validated_report_startup_rejects_manifest_drift(
    tmp_path: Path,
) -> None:
    """Fast startup fails closed when a current manifest differs from its report."""
    config = _config(tmp_path / "artifacts")
    provider = _FixtureEmbeddingProvider()
    OfflineBuildRuntime(
        config,
        source=_FixtureSource(),
        embedding_provider=provider,
    ).build()
    config.online.startup_validation = StartupValidationConfig(
        mode="validated_report"
    )
    manifest_path = (
        config.artifacts.directory("bm25_directory") / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["warnings"] = ["manifest drift"]
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactCompatibilityError, match="report"):
        OnlineRuntimeFactory(
            config,
            embedding_provider=provider,
            reranker=_FixtureReranker(),
        ).build()


def test_offline_runtime_resumes_from_validated_stage_checkpoints(
    tmp_path: Path,
) -> None:
    """A failed late index stage resumes without rebuilding persisted stages."""
    config = _config(
        tmp_path / "resumable-artifacts",
        resume_partial_build=True,
        bounded_source_passes=True,
    )
    provider = _FixtureEmbeddingProvider()

    class _FailingVectorBackend(NumpyVectorBackend):
        def build_persisted(  # type: ignore[no-untyped-def]
            self,
            *args: object,
            **kwargs: object,
        ):
            raise BackendInitializationError("fixture vector failure")

    with pytest.raises(BackendInitializationError, match="fixture vector"):
        OfflineBuildRuntime(
            config,
            source=_FixtureSource(),
            embedding_provider=provider,
            vector_backend=_FailingVectorBackend(),
        ).build()

    assert config.artifacts.directory("legal_chunks_directory").is_dir()
    assert config.artifacts.directory("bm25_directory").is_dir()
    assert not config.artifacts.directory("vector_directory").exists()
    assert not (
        config.artifacts.root_path
        / config.build_validation.report_filename
    ).exists()
    state_path = config.artifacts.root_path / "build_state.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["schema_version"] == "1.1"

    legacy_payload = {**state_payload, "schema_version": "1.0"}
    state_path.write_text(
        json.dumps(legacy_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactCompatibilityError, match="hash format"):
        OfflineBuildRuntime(
            config,
            source=_FixtureSource(),
            embedding_provider=provider,
        ).build()
    state_path.write_text(
        json.dumps(state_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    incompatible = _config(
        config.artifacts.root_path,
        resume_partial_build=True,
        bounded_source_passes=True,
    )
    incompatible.offline.chunking.max_tokens = 99
    with pytest.raises(ArtifactCompatibilityError, match="configuration"):
        OfflineBuildRuntime(
            incompatible,
            source=_FixtureSource(),
            embedding_provider=provider,
        ).build()

    resumed = OfflineBuildRuntime(
        config,
        source=_FixtureSource(),
        embedding_provider=provider,
    ).build()

    upgraded_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert upgraded_state["code_version"] == "0.20.5"
    assert resumed.validation_report.is_valid is True
    assert resumed.validation_report.is_full_corpus is True
    assert resumed.artifact_manifests["vector_index"].record_count == 2
