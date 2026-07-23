"""Unit tests for the narrow service boundary over OnlineRuntime."""

from datetime import UTC, datetime

import pytest

from legal_agentic_rag.configuration import OnlineConfig, ServingConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    InvalidUserInputError,
)
from legal_agentic_rag.schemas import (
    AgentRunResult,
    AgentState,
    AgentStopReason,
    AnswerResponse,
    ArtifactManifest,
    ArtifactType,
    LegalQuestionRequest,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.serving.query_service import ServingService


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="chunks-v1",
        dataset_name="fixture-corpus",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        record_count=2,
        processing_config_hash="config-hash",
        backend="jsonl",
    )


class _Runtime:
    def __init__(self) -> None:
        self.last_query: RetrievalQuery | None = None
        self._manifests = {"legal_chunks": _manifest()}

    @property
    def manifests(self) -> dict[str, ArtifactManifest]:
        return self._manifests

    def tool_descriptors(self) -> list[object]:
        return [object(), object()]

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        self.last_query = query
        return RetrievalResponse(
            query=query,
            strategy=query.requested_strategy or RetrievalStrategy.HYBRID,
        )

    def answer(self, query: RetrievalQuery) -> AgentRunResult:
        self.last_query = query
        strategy = query.requested_strategy or RetrievalStrategy.HYBRID
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
            ),
            stop_reason=AgentStopReason.NO_NEW_STRATEGY,
        )


def _service(
    runtime: _Runtime | None = None,
    *,
    serving: ServingConfig | None = None,
) -> tuple[ServingService, _Runtime]:
    actual_runtime = runtime or _Runtime()
    return (
        ServingService(
            actual_runtime,
            serving or ServingConfig(),
            OnlineConfig(),
            id_factory=lambda: "query-fixed",
        ),
        actual_runtime,
    )


def test_service_normalizes_question_and_applies_defaults() -> None:
    """Serving creates one traceable query while preserving original accents."""
    service, runtime = _service()

    response = service.retrieve(
        LegalQuestionRequest(
            question="Thue\u0302\u0301   phải  nộp khi nào?",
            requested_strategy=RetrievalStrategy.DENSE,
        )
    )

    assert response.query.query_id == "query-fixed"
    assert response.query.original_question == (
        "Thue\u0302\u0301   phải  nộp khi nào?"
    )
    assert response.query.normalized_question == "Thuế phải nộp khi nào?"
    assert response.query.top_k == 10
    assert response.query.candidate_k == 100
    assert response.query.metadata == {"source": "serving"}
    assert runtime.last_query == response.query


def test_service_delegates_answer_and_returns_public_response() -> None:
    """The public service strips internal Agent state from its answer."""
    service, _ = _service()

    response = service.answer(LegalQuestionRequest(question="Câu hỏi"))

    assert response.trace_id == "query-fixed"
    assert response.insufficient_evidence is True


def test_service_enforces_question_and_retrieval_limits() -> None:
    """Request values cannot exceed the process-level serving policy."""
    service, _ = _service(
        serving=ServingConfig(
            max_question_characters=5,
            max_top_k=10,
            max_candidate_k=100,
        )
    )

    with pytest.raises(InvalidUserInputError, match="Question"):
        service.retrieve(LegalQuestionRequest(question="quá dài"))
    with pytest.raises(InvalidUserInputError, match="limits"):
        service.retrieve(
            LegalQuestionRequest(
                question="hỏi",
                top_k=11,
                candidate_k=11,
            )
        )


def test_health_reports_only_loaded_artifact_identity() -> None:
    """Readiness exposes versions and counts, not local artifact paths."""
    service, _ = _service()

    health = service.health()

    assert health.status.value == "ready"
    assert health.dataset_name == "fixture-corpus"
    assert health.tool_count == 2
    assert health.artifacts[0].artifact_version == "chunks-v1"
    assert "path" not in health.model_dump_json()


def test_health_fails_when_runtime_has_no_validated_artifacts() -> None:
    """A process without loaded artifacts is not ready for legal queries."""
    runtime = _Runtime()
    runtime._manifests = {}
    service, _ = _service(runtime)

    with pytest.raises(BackendInitializationError):
        service.health()
