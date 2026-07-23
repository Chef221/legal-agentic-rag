"""Integration tests for FastAPI lifecycle, contracts, and mounted UI."""

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from legal_agentic_rag.configuration import (
    ApplicationConfig,
    ArtifactConfig,
    DatasetSourceConfig,
    OfflineConfig,
    OnlineConfig,
    ServingConfig,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    RetrievalError,
)
from legal_agentic_rag.schemas import (
    AgentRunResult,
    AgentState,
    AgentStopReason,
    AnswerResponse,
    ArtifactManifest,
    ArtifactType,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.serving import create_app


def _config(
    tmp_path: Path,
    *,
    ui_enabled: bool = False,
    max_question_characters: int = 4_000,
) -> ApplicationConfig:
    return ApplicationConfig(
        artifacts=ArtifactConfig(root_path=tmp_path / "artifacts"),
        offline=OfflineConfig(
            dataset=DatasetSourceConfig(dataset_name="fixture-corpus")
        ),
        online=OnlineConfig(),
        serving=ServingConfig(
            ui_enabled=ui_enabled,
            max_question_characters=max_question_characters,
        ),
    )


def _manifest(artifact_type: ArtifactType) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1",
        artifact_type=artifact_type,
        artifact_version=f"{artifact_type.value}-v1",
        dataset_name="fixture-corpus",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        record_count=2,
        processing_config_hash="config-hash",
        backend="fixture",
    )


class _Runtime:
    def __init__(self, *, retrieval_failure: Exception | None = None) -> None:
        self.retrieval_failure = retrieval_failure
        self.retrieval_calls = 0
        self.answer_calls = 0
        self._manifests = {
            artifact_type.value: _manifest(artifact_type)
            for artifact_type in (
                ArtifactType.LEGAL_CHUNKS,
                ArtifactType.BM25_INDEX,
                ArtifactType.VECTOR_INDEX,
                ArtifactType.GRAPH_INDEX,
            )
        }

    @property
    def manifests(self) -> dict[str, ArtifactManifest]:
        return dict(self._manifests)

    def tool_descriptors(self) -> list[object]:
        return [object()] * 8

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        self.retrieval_calls += 1
        if self.retrieval_failure is not None:
            raise self.retrieval_failure
        return RetrievalResponse(
            query=query,
            strategy=query.requested_strategy or RetrievalStrategy.HYBRID,
            artifact_versions={
                key: manifest.artifact_version
                for key, manifest in self._manifests.items()
            },
        )

    def answer(self, query: RetrievalQuery) -> AgentRunResult:
        self.answer_calls += 1
        strategy = query.requested_strategy or RetrievalStrategy.HYBRID_RERANK
        answer = "Chưa đủ căn cứ pháp luật để trả lời."
        return AgentRunResult(
            state=AgentState(
                trace_id=query.query_id,
                original_question=query.original_question,
                normalized_question=query.normalized_question,
                current_query=query.normalized_question,
                selected_strategy=strategy,
                answer=answer,
            ),
            response=AnswerResponse(
                question=query.original_question,
                answer=answer,
                insufficient_evidence=True,
                retrieval_strategy=strategy,
                trace_id=query.query_id,
                warnings=["Không đủ evidence."],
            ),
            stop_reason=AgentStopReason.NO_NEW_STRATEGY,
        )


def test_api_loads_runtime_once_and_serves_all_public_contracts(
    tmp_path: Path,
) -> None:
    """One startup runtime serves health, retrieval, and answering requests."""
    runtime = _Runtime()
    load_count = 0

    def load_runtime() -> _Runtime:
        nonlocal load_count
        load_count += 1
        return runtime

    app = create_app(_config(tmp_path), runtime_loader=load_runtime)
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        retrieval = client.post(
            "/api/v1/retrieve",
            json={
                "question": "Doanh nghiệp nộp thuế khi nào?",
                "top_k": 2,
                "candidate_k": 5,
                "requested_strategy": "hybrid",
            },
        )
        answer = client.post(
            "/api/v1/answer",
            json={"question": "Doanh nghiệp nộp thuế khi nào?"},
        )

    assert load_count == 1
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert health.json()["tool_count"] == 8
    assert len(health.json()["artifacts"]) == 4
    assert retrieval.status_code == 200
    assert retrieval.json()["query"]["top_k"] == 2
    assert retrieval.json()["strategy"] == "hybrid"
    assert answer.status_code == 200
    assert answer.json()["insufficient_evidence"] is True
    assert runtime.retrieval_calls == 1
    assert runtime.answer_calls == 1


def test_api_returns_sanitized_validation_and_domain_errors(
    tmp_path: Path,
) -> None:
    """HTTP errors never echo invalid text or internal backend details."""
    secret_detail = "private C:\\secret\\index.sqlite failed"
    runtime = _Runtime(
        retrieval_failure=RetrievalError(secret_detail),
    )
    app = create_app(
        _config(tmp_path, max_question_characters=10),
        runtime_loader=lambda: runtime,
    )

    with TestClient(app) as client:
        invalid_contract = client.post(
            "/api/v1/retrieve",
            json={"question": "", "unknown": "sensitive input"},
        )
        too_long = client.post(
            "/api/v1/retrieve",
            json={"question": "câu hỏi pháp luật quá dài"},
        )
        backend_failure = client.post(
            "/api/v1/retrieve",
            json={"question": "thuế"},
        )

    assert invalid_contract.status_code == 422
    assert invalid_contract.json()["error"]["error_type"] == "invalid_request"
    assert "sensitive input" not in invalid_contract.text
    assert too_long.status_code == 400
    assert too_long.json()["error"]["error_type"] == "invalid_user_input"
    assert backend_failure.status_code == 503
    assert backend_failure.json()["error"]["error_type"] == "retrieval_error"
    assert secret_detail not in backend_failure.text


def test_api_fails_fast_when_runtime_cannot_load(tmp_path: Path) -> None:
    """An incompatible artifact set prevents readiness and request handling."""

    def fail_startup() -> _Runtime:
        raise ArtifactCompatibilityError("tampered private artifact path")

    app = create_app(_config(tmp_path), runtime_loader=fail_startup)

    with pytest.raises(ArtifactCompatibilityError, match="tampered"):
        with TestClient(app):
            pass


def test_gradio_is_mounted_without_creating_a_second_runtime(
    tmp_path: Path,
) -> None:
    """The optional local UI shares the exact FastAPI serving lifecycle."""
    runtime = _Runtime()
    load_count = 0

    def load_runtime() -> _Runtime:
        nonlocal load_count
        load_count += 1
        return runtime

    app = create_app(
        _config(tmp_path, ui_enabled=True),
        runtime_loader=load_runtime,
    )

    with TestClient(app) as client:
        ui = client.get("/ui/", follow_redirects=True)
        health = client.get("/api/v1/health")

    assert load_count == 1
    assert ui.status_code == 200
    assert "Vietnamese Legal Agentic RAG" in ui.text
    assert health.status_code == 200
