"""Fixed retrieval-to-answer orchestration with fail-closed verification."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from legal_agentic_rag.configuration.online import (
    ContextGradingConfig,
    GenerationConfig,
)
from legal_agentic_rag.contracts.answer_generator import AnswerGenerator
from legal_agentic_rag.contracts.citation_verifier import CitationVerifier
from legal_agentic_rag.contracts.context_grader import ContextGrader
from legal_agentic_rag.exceptions import ModelError, RetrievalError
from legal_agentic_rag.generation.citation_verifier import (
    RuleBasedCitationVerifier,
)
from legal_agentic_rag.generation.context_builder import ContextBuilder
from legal_agentic_rag.generation.context_grader import RuleBasedContextGrader
from legal_agentic_rag.generation.extractive_generator import (
    ExtractiveAnswerGenerator,
)
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    CitationVerificationResult,
    ContextBuildResult,
    ContextGrade,
)
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalResponse

_ABSTENTION_TEXT = (
    "Hệ thống chưa tìm thấy căn cứ pháp luật đủ rõ trong dữ liệu hiện có "
    "để trả lời chắc chắn."
)
_LOGGER = logging.getLogger(__name__)


class _Retriever(Protocol):
    def search(self, query: RetrievalQuery) -> RetrievalResponse: ...


class FixedRAGService:
    """Run fixed retrieval, context selection, grading, generation, and verification."""

    def __init__(
        self,
        retriever: _Retriever,
        *,
        context_builder: ContextBuilder | None = None,
        context_grader: ContextGrader | None = None,
        answer_generator: AnswerGenerator | None = None,
        citation_verifier: CitationVerifier | None = None,
        generation_config: GenerationConfig | None = None,
        grading_config: ContextGradingConfig | None = None,
    ) -> None:
        generation = generation_config or GenerationConfig()
        self._retriever = retriever
        self._context_builder = context_builder or ContextBuilder(generation)
        self._context_grader = context_grader or RuleBasedContextGrader(
            grading_config
        )
        self._answer_generator = answer_generator or ExtractiveAnswerGenerator()
        self._citation_verifier = citation_verifier or RuleBasedCitationVerifier()

    def answer(self, query: RetrievalQuery) -> AnswerResponse:
        """Return a verified grounded answer or a fail-closed abstention."""
        started = perf_counter()
        retrieval = self._retriever.search(query)
        self._validate_retrieval(query, retrieval)
        context = self._context_builder.build(retrieval)
        grade = self._context_grader.grade(query, context.evidence)
        trace_id = query.query_id
        if not grade.is_sufficient:
            response = self._abstention(
                query,
                retrieval,
                context,
                grade,
                ["insufficient_context"],
            )
        else:
            generated = self._answer_generator.generate(
                query,
                context.evidence,
                retrieval.strategy,
                trace_id,
            )
            self._validate_generated(query, retrieval, generated)
            verification = self._citation_verifier.verify(
                generated, context.evidence
            )
            if verification.is_valid:
                response = self._package(
                    generated,
                    retrieval,
                    context,
                    grade,
                    verification,
                )
            else:
                response = self._abstention(
                    query,
                    retrieval,
                    context,
                    grade,
                    [
                        "citation_verification_failed",
                        *verification.errors,
                    ],
                    verification=verification,
                )
        latency_ms = (perf_counter() - started) * 1000
        metadata = dict(response.metadata)
        metadata["total_latency_ms"] = latency_ms
        final_response = response.model_copy(update={"metadata": metadata})
        _LOGGER.info(
            "fixed_rag_completed",
            extra={
                "query_id": query.query_id,
                "trace_id": trace_id,
                "strategy": retrieval.strategy.value,
                "candidate_count": len(retrieval.hits),
                "selected_evidence": len(context.evidence),
                "insufficient_evidence": final_response.insufficient_evidence,
                "latency_ms": latency_ms,
            },
        )
        return final_response

    @staticmethod
    def _validate_retrieval(
        query: RetrievalQuery,
        response: RetrievalResponse,
    ) -> None:
        if (
            response.query.query_id != query.query_id
            or response.query.original_question != query.original_question
            or response.query.normalized_question != query.normalized_question
        ):
            raise RetrievalError("Retriever returned a response for another query")

    @staticmethod
    def _validate_generated(
        query: RetrievalQuery,
        retrieval: RetrievalResponse,
        response: AnswerResponse,
    ) -> None:
        if (
            response.question != query.original_question
            or response.trace_id != query.query_id
            or response.retrieval_strategy != retrieval.strategy
        ):
            raise ModelError("Answer generator returned incompatible response metadata")

    def _abstention(
        self,
        query: RetrievalQuery,
        retrieval: RetrievalResponse,
        context: ContextBuildResult,
        grade: ContextGrade,
        warnings: list[str],
        *,
        verification: CitationVerificationResult | None = None,
    ) -> AnswerResponse:
        response = AnswerResponse(
            question=query.original_question,
            answer=_ABSTENTION_TEXT,
            insufficient_evidence=True,
            warnings=warnings,
            retrieval_strategy=retrieval.strategy,
            trace_id=query.query_id,
        )
        checked = verification or self._citation_verifier.verify(
            response, context.evidence
        )
        return self._package(
            response,
            retrieval,
            context,
            grade,
            checked,
        )

    @staticmethod
    def _package(
        response: AnswerResponse,
        retrieval: RetrievalResponse,
        context: ContextBuildResult,
        grade: ContextGrade,
        verification: CitationVerificationResult,
    ) -> AnswerResponse:
        warnings = list(response.warnings)
        warnings.extend(f"retrieval:{item}" for item in retrieval.warnings)
        warnings.extend(f"context:{item}" for item in context.warnings)
        warnings.extend(f"grader:{item}" for item in grade.warnings)
        warnings.extend(f"verifier:{item}" for item in verification.warnings)
        metadata = dict(response.metadata)
        metadata.update(
            {
                "retrieval": {
                    "strategy": retrieval.strategy.value,
                    "hit_count": len(retrieval.hits),
                    "latency_ms": retrieval.latency_ms,
                    "artifact_versions": retrieval.artifact_versions,
                },
                "context": context.model_dump(mode="json", exclude={"evidence"}),
                "context_grade": grade.model_dump(mode="json"),
                "citation_verification": verification.model_dump(mode="json"),
                "selected_evidence_ids": [
                    item.evidence_id for item in context.evidence
                ],
                "evidence_retrieval_trace": {
                    item.evidence_id: item.metadata.get("retrieval_trace")
                    for item in context.evidence
                },
            }
        )
        return response.model_copy(
            update={
                "warnings": list(dict.fromkeys(warnings)),
                "metadata": metadata,
            }
        )
