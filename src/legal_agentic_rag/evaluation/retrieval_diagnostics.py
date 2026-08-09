"""Answer-level retrieval diagnostics without fabricated relevance labels."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
from statistics import fmean
from typing import Protocol
import unicodedata

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.schemas import (
    CompetitionQuestion,
    QueryAnalysis,
    RetrievalBranchDiagnostic,
    RetrievalDiagnosticCase,
    RetrievalDiagnosticSignal,
    RetrievalDiagnosticsReport,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)
_REPORT_FILENAME = "retrieval_diagnostics.json"


class _DiagnosticRuntime(Protocol):
    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse: ...


class RetrievalDiagnosticsRunner:
    """Run BM25, dense, and hybrid observations over official questions."""

    def __init__(
        self,
        runtime: _DiagnosticRuntime,
        *,
        application_config_sha256: str,
        loader: UitDsc2026DataLoader | None = None,
    ) -> None:
        self._runtime = runtime
        self._config_hash = application_config_sha256
        self._loader = loader or UitDsc2026DataLoader()

    def run(
        self,
        question_source: Path,
        output_directory: Path,
        *,
        top_k: int = 20,
        candidate_k: int = 100,
        max_cases: int | None = None,
        include_reranker: bool = False,
        low_document_diversity_threshold: float = 0.4,
        low_answer_term_coverage_threshold: float = 0.25,
    ) -> RetrievalDiagnosticsReport:
        """Persist content-free signals; answer overlap remains non-gold."""
        if top_k <= 0 or top_k > 100 or candidate_k < top_k or candidate_k > 100:
            raise DataValidationError(
                "Diagnostic limits must satisfy 1 <= top_k <= candidate_k <= 100"
            )
        if max_cases is not None and max_cases <= 0:
            raise DataValidationError("Diagnostic max_cases must be positive")
        if not 0 <= low_document_diversity_threshold <= 1:
            raise DataValidationError("Document diversity threshold must be in [0, 1]")
        if not 0 <= low_answer_term_coverage_threshold <= 1:
            raise DataValidationError("Answer coverage threshold must be in [0, 1]")
        source_hash = _file_sha256(question_source)
        questions = self._loader.load_questions(
            question_source, require_reference_answers=False
        )
        if _file_sha256(question_source) != source_hash:
            raise DataValidationError("Diagnostic question source changed while loading")
        selected = questions[:max_cases] if max_cases is not None else questions
        cases: list[RetrievalDiagnosticCase] = []
        for index, question in enumerate(selected, 1):
            cases.append(self._run_case(
                question,
                top_k=top_k,
                candidate_k=candidate_k,
                include_reranker=include_reranker,
                low_document_diversity_threshold=low_document_diversity_threshold,
                low_answer_term_coverage_threshold=low_answer_term_coverage_threshold,
            ))
            if index % 25 == 0 or index == len(selected):
                _LOGGER.info(
                    "retrieval_diagnostics_progress",
                    extra={
                        "question_count": len(selected),
                        "completed_question_count": index,
                    },
                )
        if _file_sha256(question_source) != source_hash:
            raise DataValidationError("Diagnostic question source changed during run")
        report = self._report(
            cases,
            question_source_sha256=source_hash,
            top_k=top_k,
            candidate_k=candidate_k,
            max_cases=max_cases,
            include_reranker=include_reranker,
            low_document_diversity_threshold=low_document_diversity_threshold,
            low_answer_term_coverage_threshold=low_answer_term_coverage_threshold,
        )
        _persist_report(output_directory, report)
        return report

    def _run_case(
        self,
        question: CompetitionQuestion,
        *,
        top_k: int,
        candidate_k: int,
        include_reranker: bool,
        low_document_diversity_threshold: float,
        low_answer_term_coverage_threshold: float,
    ) -> RetrievalDiagnosticCase:
        normalized = unicodedata.normalize("NFC", " ".join(question.question.split()))
        responses: list[RetrievalResponse] = []
        try:
            strategies = [
                RetrievalStrategy.BM25,
                RetrievalStrategy.DENSE,
                RetrievalStrategy.HYBRID,
            ]
            if include_reranker:
                strategies.append(RetrievalStrategy.HYBRID_RERANK)
            for strategy in strategies:
                query_id = f"diagnostic:{question.question_id}:{strategy.value}"
                response = self._runtime.retrieve(
                    RetrievalQuery(
                        query_id=query_id,
                        original_question=question.question,
                        normalized_question=normalized,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        requested_strategy=strategy,
                        metadata={"source": "retrieval_diagnostics"},
                    )
                )
                if (
                    response.strategy != strategy
                    or response.query.query_id != query_id
                ):
                    raise DataValidationError(
                        "Diagnostic runtime returned an incompatible response"
                    )
                responses.append(response)
        except Exception as error:
            return RetrievalDiagnosticCase(
                question_id=question.question_id,
                success=False,
                reranker_included=include_reranker,
                signals=[RetrievalDiagnosticSignal.RETRIEVAL_ERROR],
                error_type=_error_type(error),
            )
        by_strategy = {response.strategy: response for response in responses}
        bm25 = by_strategy[RetrievalStrategy.BM25]
        dense = by_strategy[RetrievalStrategy.DENSE]
        hybrid = by_strategy[RetrievalStrategy.HYBRID]
        sparse_ids = {hit.chunk_id for hit in bm25.hits}
        dense_ids = {hit.chunk_id for hit in dense.hits}
        overlap = len(sparse_ids & dense_ids)
        union = len(sparse_ids | dense_ids)
        jaccard = overlap / union if union else 0.0
        unique_documents = len({hit.document_id for hit in hybrid.hits})
        diversity = unique_documents / len(hybrid.hits) if hybrid.hits else 0.0
        analysis = hybrid.query.query_analysis
        explicit_match = _explicit_reference_match(hybrid, analysis)
        answer_coverage = (
            _answer_term_coverage(question.reference_answer, hybrid)
            if question.reference_answer is not None
            else None
        )
        reranked = by_strategy.get(RetrievalStrategy.HYBRID_RERANK)
        reranked_ids = (
            {hit.chunk_id for hit in reranked.hits}
            if reranked is not None
            else set()
        )
        reranked_overlap = (
            len({hit.chunk_id for hit in hybrid.hits} & reranked_ids)
            if reranked is not None
            else None
        )
        reranked_union = (
            len({hit.chunk_id for hit in hybrid.hits} | reranked_ids)
            if reranked is not None
            else 0
        )
        reranked_jaccard = (
            reranked_overlap / reranked_union
            if reranked is not None and reranked_union
            else (0.0 if reranked is not None else None)
        )
        reranked_diversity = (
            len({hit.document_id for hit in reranked.hits}) / len(reranked.hits)
            if reranked is not None and reranked.hits
            else (0.0 if reranked is not None else None)
        )
        reranked_coverage = (
            _answer_term_coverage(question.reference_answer, reranked)
            if reranked is not None and question.reference_answer is not None
            else None
        )
        reranked_coverage_delta = (
            reranked_coverage - answer_coverage
            if reranked_coverage is not None and answer_coverage is not None
            else None
        )
        signals: list[RetrievalDiagnosticSignal] = []
        if not bm25.hits:
            signals.append(RetrievalDiagnosticSignal.NO_BM25_HITS)
        if not dense.hits:
            signals.append(RetrievalDiagnosticSignal.NO_DENSE_HITS)
        if not hybrid.hits:
            signals.append(RetrievalDiagnosticSignal.NO_HYBRID_HITS)
        if sparse_ids and dense_ids and overlap == 0:
            signals.append(RetrievalDiagnosticSignal.ZERO_BRANCH_OVERLAP)
        if hybrid.hits and diversity < low_document_diversity_threshold:
            signals.append(RetrievalDiagnosticSignal.LOW_DOCUMENT_DIVERSITY)
        if explicit_match is False:
            signals.append(
                RetrievalDiagnosticSignal.EXPLICIT_REFERENCE_NOT_RETRIEVED
            )
        if (
            answer_coverage is not None
            and answer_coverage < low_answer_term_coverage_threshold
        ):
            signals.append(RetrievalDiagnosticSignal.LOW_ANSWER_TERM_COVERAGE)
        if any(response.warnings for response in responses):
            signals.append(RetrievalDiagnosticSignal.RETRIEVAL_WARNING)
        return RetrievalDiagnosticCase(
            question_id=question.question_id,
            success=True,
            reranker_included=include_reranker,
            query_intent=analysis.intent.value if analysis is not None else None,
            query_variant_count=len(hybrid.query.query_variants),
            branches=[_branch(response) for response in responses],
            bm25_dense_overlap_count=overlap,
            bm25_dense_jaccard=jaccard,
            hybrid_document_diversity=diversity,
            explicit_reference_match=explicit_match,
            answer_term_coverage=answer_coverage,
            hybrid_rerank_overlap_count=reranked_overlap,
            hybrid_rerank_jaccard=reranked_jaccard,
            hybrid_rerank_document_diversity=reranked_diversity,
            reranked_explicit_reference_match=(
                _explicit_reference_match(reranked, analysis)
                if reranked is not None
                else None
            ),
            hybrid_rerank_answer_term_coverage=reranked_coverage,
            hybrid_rerank_answer_term_coverage_delta=reranked_coverage_delta,
            mean_absolute_rank_change=(
                _mean_absolute_rank_change(hybrid, reranked)
                if reranked is not None
                else None
            ),
            signals=signals,
        )

    def _report(
        self,
        cases: list[RetrievalDiagnosticCase],
        **identity: object,
    ) -> RetrievalDiagnosticsReport:
        include_reranker = bool(identity.get("include_reranker", False))
        successful = [case for case in cases if case.success]
        coverage = [
            case.answer_term_coverage
            for case in successful
            if case.answer_term_coverage is not None
        ]
        reranker_jaccards = [
            case.hybrid_rerank_jaccard
            for case in successful
            if case.hybrid_rerank_jaccard is not None
        ]
        reranker_diversities = [
            case.hybrid_rerank_document_diversity
            for case in successful
            if case.hybrid_rerank_document_diversity is not None
        ]
        reranker_coverage_deltas = [
            case.hybrid_rerank_answer_term_coverage_delta
            for case in successful
            if case.hybrid_rerank_answer_term_coverage_delta is not None
        ]
        reranker_rank_changes = [
            case.mean_absolute_rank_change
            for case in successful
            if case.mean_absolute_rank_change is not None
        ]
        signal_counts = Counter(
            signal for case in cases for signal in case.signals
        )
        return RetrievalDiagnosticsReport(
            created_at=datetime.now(UTC),
            code_version=__version__,
            application_config_sha256=self._config_hash,
            question_count=len(cases),
            successful_case_count=len(successful),
            failed_case_count=len(cases) - len(successful),
            mean_bm25_dense_jaccard=(
                fmean(case.bm25_dense_jaccard for case in successful)
                if successful
                else 0.0
            ),
            mean_hybrid_document_diversity=(
                fmean(case.hybrid_document_diversity for case in successful)
                if successful
                else 0.0
            ),
            mean_answer_term_coverage=fmean(coverage) if coverage else None,
            mean_hybrid_rerank_jaccard=(
                fmean(reranker_jaccards)
                if reranker_jaccards
                else (0.0 if include_reranker else None)
            ),
            mean_hybrid_rerank_document_diversity=(
                fmean(reranker_diversities)
                if reranker_diversities
                else (0.0 if include_reranker else None)
            ),
            mean_hybrid_rerank_answer_term_coverage_delta=(
                fmean(reranker_coverage_deltas)
                if reranker_coverage_deltas
                else None
            ),
            mean_absolute_rank_change=(
                fmean(reranker_rank_changes) if reranker_rank_changes else None
            ),
            signal_counts=dict(signal_counts),
            cases=cases,
            warnings=[
                "answer_term_coverage_is_not_retrieval_relevance_gold",
                "diagnostic_signals_are_hypotheses_not_failure_labels",
                "report_omits_question_answer_and_legal_text",
            ],
            **identity,
        )


def _branch(response: RetrievalResponse) -> RetrievalBranchDiagnostic:
    return RetrievalBranchDiagnostic(
        strategy=response.strategy,
        hit_count=len(response.hits),
        unique_document_count=len({hit.document_id for hit in response.hits}),
        latency_ms=response.latency_ms,
        chunk_ids=[hit.chunk_id for hit in response.hits],
        document_ids=[hit.document_id for hit in response.hits],
        warnings=response.warnings,
    )


def _explicit_reference_match(
    response: RetrievalResponse,
    analysis: QueryAnalysis | None,
) -> bool | None:
    if analysis is None or not analysis.has_explicit_legal_reference:
        return None
    document_numbers = {_normalized_identity(value) for value in analysis.document_numbers}
    article_numbers = {_normalized_identity(value) for value in analysis.article_numbers}
    clause_numbers = {_normalized_identity(value) for value in analysis.clause_numbers}
    point_numbers = {_normalized_identity(value) for value in analysis.point_numbers}
    for hit in response.hits:
        metadata = hit.metadata
        document_match = not document_numbers or _normalized_identity(
            str(metadata.get("document_number", ""))
        ) in document_numbers
        article_match = not article_numbers or _normalized_identity(
            str(metadata.get("article_number", ""))
        ) in article_numbers
        clause_match = not clause_numbers or _normalized_identity(
            str(metadata.get("clause_number", ""))
        ) in clause_numbers
        point_match = not point_numbers or _normalized_identity(
            str(metadata.get("point_number", ""))
        ) in point_numbers
        if document_match and article_match and clause_match and point_match:
            return True
    return False


def _answer_term_coverage(answer: str, response: RetrievalResponse) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    retrieved_tokens = _tokens(" ".join(hit.text for hit in response.hits))
    return len(answer_tokens & retrieved_tokens) / len(answer_tokens)


def _mean_absolute_rank_change(
    hybrid: RetrievalResponse,
    reranked: RetrievalResponse,
) -> float | None:
    hybrid_ranks = {hit.chunk_id: hit.rank for hit in hybrid.hits}
    reranked_ranks = {hit.chunk_id: hit.rank for hit in reranked.hits}
    shared = hybrid_ranks.keys() & reranked_ranks.keys()
    if not shared:
        return None
    return fmean(
        abs(hybrid_ranks[chunk_id] - reranked_ranks[chunk_id])
        for chunk_id in shared
    )


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return {
        token
        for token in "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
        if token
    }


def _normalized_identity(value: str) -> str:
    return unicodedata.normalize("NFC", "".join(value.split())).casefold()


def _error_type(error: Exception) -> str:
    name = type(error).__name__
    return "".join(
        f"_{character.casefold()}" if character.isupper() else character
        for character in name
    ).lstrip("_")


def _file_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactCompatibilityError("Diagnostic source is unreadable") from error


def _persist_report(output_directory: Path, report: RetrievalDiagnosticsReport) -> None:
    output = output_directory.resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists() or temporary.exists():
        raise ArtifactCompatibilityError("Retrieval diagnostic output already exists")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        (temporary / _REPORT_FILENAME).write_text(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
    except OSError as error:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink(missing_ok=True)
            temporary.rmdir()
        raise ArtifactCompatibilityError(
            "Retrieval diagnostic report could not be persisted"
        ) from error
