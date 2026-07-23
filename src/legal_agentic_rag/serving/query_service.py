"""Serving boundary for query creation, retrieval, answers, and health."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
import unicodedata
from uuid import uuid4

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.online import OnlineConfig
from legal_agentic_rag.configuration.serving import ServingConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    InvalidUserInputError,
)
from legal_agentic_rag.schemas import (
    AgentRunResult,
    AnswerResponse,
    ArtifactHealth,
    ArtifactManifest,
    HealthResponse,
    LegalQuestionRequest,
    RetrievalQuery,
    RetrievalResponse,
    ServiceStatus,
)


class _OnlineRuntime(Protocol):
    @property
    def manifests(self) -> dict[str, ArtifactManifest]: ...

    def tool_descriptors(self) -> list[object]: ...

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse: ...

    def answer(self, query: RetrievalQuery) -> AgentRunResult: ...


class ServingService:
    """Expose a narrow validated application boundary over OnlineRuntime."""

    def __init__(
        self,
        runtime: _OnlineRuntime,
        serving_config: ServingConfig,
        online_config: OnlineConfig,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._runtime = runtime
        self._serving = serving_config
        self._online = online_config
        self._id_factory = id_factory or (lambda: str(uuid4()))

    def retrieve(self, request: LegalQuestionRequest) -> RetrievalResponse:
        """Run one fixed strategy request through the loaded runtime."""
        return self._runtime.retrieve(self.create_query(request))

    def answer(self, request: LegalQuestionRequest) -> AnswerResponse:
        """Return only the public verified answer contract."""
        return self.answer_result(request).response

    def answer_result(self, request: LegalQuestionRequest) -> AgentRunResult:
        """Return answer plus internal state for the local diagnostic UI."""
        return self._runtime.answer(self.create_query(request))

    def create_query(self, request: LegalQuestionRequest) -> RetrievalQuery:
        """Create one bounded unified query without losing Vietnamese text."""
        question = request.question.strip()
        if len(question) > self._serving.max_question_characters:
            raise InvalidUserInputError("Question exceeds serving limit")
        normalized_question = unicodedata.normalize(
            "NFC",
            " ".join(question.split()),
        )
        top_k = request.top_k or self._online.retrieval.top_k
        candidate_k = (
            request.candidate_k or self._online.retrieval.candidate_k
        )
        maximum_candidate_k = min(
            self._serving.max_candidate_k,
            self._online.reranker.max_candidates,
        )
        if (
            top_k > self._serving.max_top_k
            or candidate_k > maximum_candidate_k
            or candidate_k < top_k
        ):
            raise InvalidUserInputError(
                "Requested retrieval limits exceed serving policy"
            )
        query_id = self._id_factory().strip()
        if not query_id:
            raise InvalidUserInputError("Query identity could not be created")
        return RetrievalQuery(
            query_id=query_id,
            original_question=question,
            normalized_question=normalized_question,
            filters=request.filters,
            top_k=top_k,
            candidate_k=candidate_k,
            requested_strategy=request.requested_strategy,
            metadata={"source": "serving"},
        )

    def health(self) -> HealthResponse:
        """Return non-sensitive readiness and loaded artifact identities."""
        manifests = sorted(
            self._runtime.manifests.values(),
            key=lambda item: item.artifact_type.value,
        )
        if not manifests:
            raise BackendInitializationError("Online runtime has no artifacts")
        dataset_name = manifests[0].dataset_name
        dataset_revision = manifests[0].dataset_revision
        return HealthResponse(
            status=ServiceStatus.READY,
            service_version=__version__,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            artifacts=[
                ArtifactHealth(
                    artifact_type=manifest.artifact_type,
                    artifact_version=manifest.artifact_version,
                    record_count=manifest.record_count,
                    backend=manifest.backend,
                    model_name=manifest.model_name,
                )
                for manifest in manifests
            ],
            tool_count=len(self._runtime.tool_descriptors()),
        )
