"""End-to-end fixture build and online Agent runtime assembly."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from legal_agentic_rag.configuration import (
    AgentConfig,
    ApplicationConfig,
    ArtifactConfig,
    ChunkingConfig,
    DatasetSourceConfig,
    EmbeddingConfig,
    OfflineConfig,
    OnlineConfig,
    RelationshipNormalizationConfig,
    RetrievalConfig,
)
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.runtime import (
    OfflineBuildRuntime,
    OnlineRuntimeFactory,
)
from legal_agentic_rag.schemas import (
    AgentStopReason,
    DatasetManifest,
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


def _config(root: Path) -> ApplicationConfig:
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
        ),
        online=OnlineConfig(
            retrieval=RetrievalConfig(top_k=1, candidate_k=2),
            agent=AgentConfig(max_retry=2),
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

    with pytest.raises(ArtifactCompatibilityError, match="checksum"):
        OnlineRuntimeFactory(
            config,
            embedding_provider=provider,
            reranker=_FixtureReranker(),
        ).build()
