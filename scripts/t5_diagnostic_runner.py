"""T5 Diagnostic Execution Harness for baseline error decomposition.

Runs the exact frozen production baseline while capturing pre-rerank candidates,
post-rerank hits, branch telemetry, generator rejection logs, overlap proxies,
and per-question official scores without modifying production serving behavior.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable, TypeVar
import unicodedata

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
)

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
    render_competition_answer,
)
from legal_agentic_rag.competition.uit_dsc_2026.loader import (
    UitDsc2026DataLoader,
)
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.retrieval.fixed import FixedRetriever, _RetrievalBranch
from legal_agentic_rag.retrieval.rerank import RerankingRetriever
from legal_agentic_rag.schemas import (
    AgentRunResult,
    AgentState,
    AgentStopReason,
    AnswerResponse,
    Citation,
    CompetitionBatchManifest,
    CompetitionBatchRecord,
    ContextGrade,
    Evidence,
    LegalQuestionRequest,
    QueryAnalysis,
    QueryIntent,
    QueryVariant,
    RetrievalHit,
    RetrievalHistoryItem,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)
from legal_agentic_rag.serving.query_service import ServingService

_LOGGER = logging.getLogger(__name__)

# Execution authority constants
FROZEN_BASELINE_SOURCE_SHA = "87e71eb7661eb9cda1e63f4f0af16ef4613dadfb"
FROZEN_T4_TAG = "t4-graph-retirement-closed"
FROZEN_M45_ARCHIVE_SHA256 = (
    "7e78ad60ff2982592a9471eb8704fce44042add0496268fade3f32db1823ea7a"
)
FROZEN_M49_GENERATOR_TREE_SHA256 = (
    "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b"
)
FROZEN_TRAIN_SOURCE_SHA256 = (
    "2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988"
)
FROZEN_DEV200_ORDERED_IDS_SHA256 = (
    "694825b5961a90a284ad0364ac4f31a1a85f446519c92274a784c8e2be9a48ad"
)
FROZEN_OFFICIAL_SCORER_SHA256 = (
    "4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891"
)
T5_DIAGNOSTIC_SCHEMA_VERSION = "t5_diagnostic_record_v1"

BATCH_RECORDS_FILENAME = "results.jsonl"
DIAGNOSTIC_RECORDS_FILENAME = "diagnostics.jsonl"
BATCH_STATE_FILENAME = "batch_state.json"
BATCH_MANIFEST_FILENAME = "manifest.json"
BATCH_REPORT_FILENAME = "report.json"
_PROGRESS_INTERVAL = 25

_SHA_40_HEX_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA_64_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

T = TypeVar("T", bound=BaseModel)


class T5ExecutionIdentity(BaseModel):
    """Immutable identity pins governing an authoritative T5 baseline execution."""

    model_config = ConfigDict(extra="forbid")

    production_baseline_source_sha: str = FROZEN_BASELINE_SOURCE_SHA
    measurement_harness_source_sha: str
    t4_tag: str = FROZEN_T4_TAG
    m45_archive_sha256: str = FROZEN_M45_ARCHIVE_SHA256
    m49_generator_tree_sha256: str = FROZEN_M49_GENERATOR_TREE_SHA256
    train_source_sha256: str = FROZEN_TRAIN_SOURCE_SHA256
    dev200_ordered_ids_sha256: str = FROZEN_DEV200_ORDERED_IDS_SHA256
    official_scorer_sha256: str = FROZEN_OFFICIAL_SCORER_SHA256
    diagnostic_schema_version: str = T5_DIAGNOSTIC_SCHEMA_VERSION

    @field_validator(
        "production_baseline_source_sha",
        "measurement_harness_source_sha",
    )
    @classmethod
    def validate_git_commit_shas(cls, value: str) -> str:
        v = value.strip()
        if not _SHA_40_HEX_RE.match(v):
            raise ValueError(
                f"Git commit SHA must be exactly 40 hexadecimal characters, got '{value}'"
            )
        return v.lower()

    @field_validator(
        "m45_archive_sha256",
        "m49_generator_tree_sha256",
        "train_source_sha256",
        "dev200_ordered_ids_sha256",
        "official_scorer_sha256",
    )
    @classmethod
    def validate_sha256_digests(cls, value: str) -> str:
        v = value.strip()
        if not _SHA_64_HEX_RE.match(v):
            raise ValueError(
                f"SHA-256 digest must be exactly 64 hexadecimal characters, got '{value}'"
            )
        return v.lower()


class T5RetrievalHitTelemetry(BaseModel):
    """Normalized snapshot of one retrieval hit with trace details."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    chunk_id: str
    document_id: str
    rank: int
    score: float
    strategy: RetrievalStrategy
    text: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    retrieval_trace: RetrievalTrace | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class T5QueryBranchTelemetry(BaseModel):
    """Snapshot of one sparse or dense retrieval branch execution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    variant_id: str
    variant_text: str
    strategy: RetrievalStrategy
    hits: list[T5RetrievalHitTelemetry] = Field(default_factory=list)


class T5RerankTelemetry(BaseModel):
    """Exact pre- and post-rerank candidate pool observation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    attempt_number: int
    query_id: str
    requested_candidate_k: int
    effective_candidate_limit: int
    is_relationship_intent: bool
    pre_rerank_candidate_count: int
    pre_rerank_candidates: list[T5RetrievalHitTelemetry] = Field(
        default_factory=list
    )
    post_rerank_count: int
    post_rerank_hits: list[T5RetrievalHitTelemetry] = Field(
        default_factory=list
    )


class T5GeneratorRejectionItem(BaseModel):
    """One observed generator draft rejection logged by ModelDrivenAnswerGenerator."""

    model_config = ConfigDict(extra="forbid")

    error_type: str
    structured_output_attempt: int


class T5EvidenceOverlapProxy(BaseModel):
    """Descriptive lexical and character overlap metrics against reference answer."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    token_containment: float = 0.0
    token_jaccard: float = 0.0
    character_overlap: float = 0.0
    best_chunk_id: str | None = None
    best_document_id: str | None = None


class T5QuestionDiagnosticRecord(BaseModel):
    """Rich per-question diagnostic record for T5 error decomposition."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = T5_DIAGNOSTIC_SCHEMA_VERSION
    question_id: str
    question: str
    reference_answer: str

    # Public response summary
    public_response: AnswerResponse
    stop_reason: AgentStopReason
    total_latency_ms: float
    selected_strategy: RetrievalStrategy | None = None
    retry_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    # Detailed Agent state & trace
    query_analysis: QueryAnalysis | None = None
    query_variants: list[QueryVariant] = Field(default_factory=list)
    retrieval_history: list[RetrievalHistoryItem] = Field(default_factory=list)
    terminal_retrieval_hits: list[T5RetrievalHitTelemetry] = Field(
        default_factory=list
    )
    selected_evidence: list[Evidence] = Field(default_factory=list)
    context_grade: ContextGrade | None = None

    # Pre-rerank and branch telemetry
    rerank_telemetry: list[T5RerankTelemetry] = Field(default_factory=list)
    branch_telemetry: list[T5QueryBranchTelemetry] = Field(
        default_factory=list
    )

    # Generator forensics & Fallback breakdowns
    generator_draft_rejections: list[T5GeneratorRejectionItem] = Field(
        default_factory=list
    )
    is_generator_model_error_fallback: bool = False
    is_grounding_extractive_fallback: bool = False
    is_any_extractive_fallback: bool = False
    is_insufficient_evidence: bool = False

    # Official scoring & Overlap Proxies
    meteor_score: float
    rouge_l_score: float
    pre_rerank_overlap_proxy: T5EvidenceOverlapProxy | None = None
    post_rerank_overlap_proxy: T5EvidenceOverlapProxy | None = None
    selected_evidence_overlap_proxy: T5EvidenceOverlapProxy | None = None


class T5DiagnosticBatchState(BaseModel):
    """Resumable state for T5 diagnostic batch inference."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "t5_diagnostic_state_v1"
    execution_identity: T5ExecutionIdentity
    question_source_sha256: str
    application_config_hash: str
    code_version: str
    question_count: int
    ordered_question_ids_sha256: str
    completed_question_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class T5DiagnosticBatchManifest(BaseModel):
    """Immutable proof of completion for a T5 diagnostic batch execution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "t5_diagnostic_manifest_v1"
    execution_identity: T5ExecutionIdentity
    question_source_sha256: str
    application_config_hash: str
    code_version: str
    ordered_question_ids_sha256: str
    created_at: datetime
    record_count: int
    public_records_sha256: str
    diagnostic_records_sha256: str
    insufficient_evidence_count: int
    generator_model_error_fallback_count: int
    grounding_extractive_fallback_count: int
    extractive_fallback_count: int
    any_extractive_fallback_count: int
    grounding_repair_count: int
    official_rouge: float
    official_meteor: float
    mean_per_question_rouge: float
    mean_per_question_meteor: float
    warning_counts: dict[str, int] = Field(default_factory=dict)


def hit_to_telemetry(hit: RetrievalHit) -> T5RetrievalHitTelemetry:
    """Convert a core RetrievalHit to a structured telemetry model."""
    trace = hit.retrieval_trace
    return T5RetrievalHitTelemetry(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        rank=hit.rank,
        score=hit.score,
        strategy=hit.strategy,
        text=hit.text,
        metadata=hit.metadata,
        retrieval_trace=trace,
        bm25_rank=trace.bm25_rank if trace else None,
        dense_rank=trace.dense_rank if trace else None,
        bm25_score=trace.bm25_score if trace else None,
        dense_score=trace.dense_score if trace else None,
        rrf_score=trace.rrf_score if trace else None,
        reranker_score=trace.reranker_score if trace else None,
    )


class ActiveDiagnosticContext:
    """Thread-safe and invocation-scoped storage for observed diagnostic events."""

    def __init__(self) -> None:
        self.question_id: str = ""
        self.trace_id: str = ""
        self.rerank_events: list[T5RerankTelemetry] = []
        self.branch_events: list[T5QueryBranchTelemetry] = []
        self.rejection_events: list[T5GeneratorRejectionItem] = []

    def reset(self, question_id: str = "", trace_id: str = "") -> None:
        self.question_id = question_id
        self.trace_id = trace_id
        self.rerank_events = []
        self.branch_events = []
        self.rejection_events = []


class DiagnosticRerankerObserver(Reranker):
    """Transparent proxy around Reranker that records candidate hits without mutation."""

    def __init__(
        self,
        inner: Reranker,
        context: ActiveDiagnosticContext,
        *,
        relationship_candidate_k: int | None = None,
        max_candidates: int | None = None,
    ) -> None:
        self._inner = inner
        self._context = context
        self._relationship_candidate_k = relationship_candidate_k
        self._max_candidates = max_candidates

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def provider_version(self) -> str:
        return self._inner.provider_version

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def model_revision(self) -> str | None:
        return self._inner.model_revision

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> RetrievalResponse:
        pre_hits = list(candidates)
        is_rel = (
            query.query_analysis is not None
            and query.query_analysis.intent == QueryIntent.RELATIONSHIP
        )

        effective_limit = query.candidate_k
        if is_rel and self._relationship_candidate_k is not None:
            effective_limit = min(effective_limit, self._relationship_candidate_k)
        if self._max_candidates is not None:
            effective_limit = min(effective_limit, self._max_candidates)

        response = self._inner.rerank(query, candidates)

        telemetry = T5RerankTelemetry(
            attempt_number=len(self._context.rerank_events) + 1,
            query_id=query.query_id,
            requested_candidate_k=query.candidate_k,
            effective_candidate_limit=effective_limit,
            is_relationship_intent=is_rel,
            pre_rerank_candidate_count=len(pre_hits),
            pre_rerank_candidates=[hit_to_telemetry(h) for h in pre_hits],
            post_rerank_count=len(response.hits),
            post_rerank_hits=[hit_to_telemetry(h) for h in response.hits],
        )
        self._context.rerank_events.append(telemetry)
        return response


class DiagnosticBranchRetrieverObserver:
    """Transparent proxy around branch retrievers recording branch query and hits."""

    def __init__(
        self,
        inner: _RetrievalBranch,
        strategy: RetrievalStrategy,
        context: ActiveDiagnosticContext,
    ) -> None:
        self._inner = inner
        self._strategy = strategy
        self._context = context

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        return self._inner.source_artifact_identity

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        resp = self._inner.search(query)
        effective_text = query.rewritten_question or query.normalized_question

        variant_id = "qv-active"
        if query.query_variants:
            for v in query.query_variants:
                if v.text == effective_text:
                    variant_id = v.variant_id
                    break

        telemetry = T5QueryBranchTelemetry(
            variant_id=variant_id,
            variant_text=effective_text,
            strategy=self._strategy,
            hits=[hit_to_telemetry(h) for h in resp.hits],
        )
        self._context.branch_events.append(telemetry)
        return resp


class GeneratorRejectionLogHandler(logging.Handler):
    """Logging handler intercepting model_answer_draft_rejected warnings."""

    def __init__(self, context: ActiveDiagnosticContext) -> None:
        super().__init__()
        self._context = context

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg == "model_answer_draft_rejected":
            error_type = getattr(record, "error_type", "unknown")
            structured_output_attempt = getattr(
                record, "structured_output_attempt", 1
            )
            self._context.rejection_events.append(
                T5GeneratorRejectionItem(
                    error_type=str(error_type),
                    structured_output_attempt=int(structured_output_attempt),
                )
            )


def compute_overlap_proxy(
    reference_answer: str | None,
    candidates: Sequence[
        RetrievalHit | Evidence | T5RetrievalHitTelemetry
    ],
) -> T5EvidenceOverlapProxy:
    """Compute descriptive lexical overlap proxies between reference and candidates."""
    if not reference_answer or not reference_answer.strip() or not candidates:
        return T5EvidenceOverlapProxy(
            token_containment=0.0,
            token_jaccard=0.0,
            character_overlap=0.0,
        )

    ref_norm = unicodedata.normalize("NFC", reference_answer.casefold())
    ref_tokens = set(ref_norm.split())
    if not ref_tokens:
        return T5EvidenceOverlapProxy(
            token_containment=0.0,
            token_jaccard=0.0,
            character_overlap=0.0,
        )

    best_containment = 0.0
    best_jaccard = 0.0
    best_char_overlap = 0.0
    best_chunk_id: str | None = None
    best_doc_id: str | None = None

    # Reference character 3-grams
    ref_3grams = (
        set(ref_norm[i : i + 3] for i in range(len(ref_norm) - 2))
        if len(ref_norm) >= 3
        else {ref_norm}
    )

    for item in candidates:
        text = item.text if hasattr(item, "text") else ""
        chunk_id = item.chunk_id if hasattr(item, "chunk_id") else None
        doc_id = item.document_id if hasattr(item, "document_id") else None

        cand_norm = unicodedata.normalize("NFC", text.casefold())
        cand_tokens = set(cand_norm.split())

        intersection = ref_tokens & cand_tokens
        union = ref_tokens | cand_tokens
        containment = len(intersection) / len(ref_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0

        cand_3grams = (
            set(cand_norm[i : i + 3] for i in range(len(cand_norm) - 2))
            if len(cand_norm) >= 3
            else {cand_norm}
        )
        char_overlap = (
            len(ref_3grams & cand_3grams) / len(ref_3grams)
            if ref_3grams
            else 0.0
        )

        if containment > best_containment or (
            containment == best_containment and jaccard > best_jaccard
        ):
            best_containment = containment
            best_jaccard = jaccard
            best_char_overlap = char_overlap
            best_chunk_id = chunk_id
            best_doc_id = doc_id

    return T5EvidenceOverlapProxy(
        token_containment=round(best_containment, 6),
        token_jaccard=round(best_jaccard, 6),
        character_overlap=round(best_char_overlap, 6),
        best_chunk_id=best_chunk_id,
        best_document_id=best_doc_id,
    )


def compute_per_question_scores(
    official_scoring_module: Any | None,
    question_id: str,
    rendered_answer: str,
    reference_answer: str | None,
) -> tuple[float, float]:
    """Evaluate one question prediction using the audited official scoring module."""
    if reference_answer is None or not reference_answer.strip():
        raise DataValidationError(f"Reference answer for question {question_id} is missing or empty")
    if official_scoring_module is None:
        raise DataValidationError("Official scoring module is required for authoritative scoring")
    if not hasattr(official_scoring_module, "eval_qa"):
        raise DataValidationError("Official scoring module must expose eval_qa()")
    try:
        preds = {question_id: {"answer": rendered_answer}}
        truth = {question_id: reference_answer}
        score_values = official_scoring_module.eval_qa(preds, truth)
        if "rouge" not in score_values or "meteor" not in score_values:
            raise DataValidationError("Official scoring output missing rouge or meteor")
        return (
            float(score_values["rouge"]),
            float(score_values["meteor"]),
        )
    except Exception as error:
        if isinstance(error, DataValidationError):
            raise
        raise DataValidationError(
            f"Per-question official scoring failed for question {question_id}: {error}"
        ) from error


def instrument_service_for_diagnostics(
    service: ServingService,
    context: ActiveDiagnosticContext,
) -> GeneratorRejectionLogHandler:
    """Attach transparent diagnostic observers to the runtime without mutating contracts."""
    runtime = service._runtime
    retriever = getattr(runtime, "_retriever", None)
    if isinstance(retriever, FixedRetriever):
        hybrid_rerank = getattr(retriever, "_hybrid_rerank", None)
        if isinstance(hybrid_rerank, RerankingRetriever):
            inner_reranker = hybrid_rerank._reranker
            if not isinstance(inner_reranker, DiagnosticRerankerObserver):
                reranker_cfg = getattr(hybrid_rerank, "_config", None)
                rel_k = getattr(reranker_cfg, "relationship_candidate_k", None) if reranker_cfg else None
                max_k = getattr(reranker_cfg, "max_candidates", None) if reranker_cfg else None
                hybrid_rerank._reranker = DiagnosticRerankerObserver(
                    inner_reranker,
                    context,
                    relationship_candidate_k=rel_k,
                    max_candidates=max_k,
                )
        hybrid = getattr(retriever, "_hybrid", None)
        if hybrid is not None:
            if hasattr(hybrid, "_bm25") and not isinstance(
                hybrid._bm25, DiagnosticBranchRetrieverObserver
            ):
                hybrid._bm25 = DiagnosticBranchRetrieverObserver(
                    hybrid._bm25, RetrievalStrategy.BM25, context
                )
            if hasattr(hybrid, "_dense") and not isinstance(
                hybrid._dense, DiagnosticBranchRetrieverObserver
            ):
                hybrid._dense = DiagnosticBranchRetrieverObserver(
                    hybrid._dense, RetrievalStrategy.DENSE, context
                )

    log_handler = GeneratorRejectionLogHandler(context)
    gen_logger = logging.getLogger(
        "legal_agentic_rag.generation.model_generator"
    )
    gen_logger.addHandler(log_handler)
    return log_handler


class T5Dev200DiagnosticRunner:
    """Resumable diagnostic evaluation runner over Dev-200 with full telemetry."""

    def __init__(
        self,
        service: ServingService,
        *,
        application_config_hash: str,
        execution_identity: T5ExecutionIdentity,
        official_scoring_module: Any | None = None,
        loader: UitDsc2026DataLoader | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._service = service
        self._config_hash = application_config_hash
        self._identity = execution_identity
        self._scorer_module = official_scoring_module
        self._loader = loader or UitDsc2026DataLoader()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._context = ActiveDiagnosticContext()
        self._log_handler: GeneratorRejectionLogHandler | None = instrument_service_for_diagnostics(
            self._service, self._context
        )

    def close(self) -> None:
        """Detach logging handler from global logger."""
        if self._log_handler is not None:
            gen_logger = logging.getLogger("legal_agentic_rag.generation.model_generator")
            gen_logger.removeHandler(self._log_handler)
            self._log_handler = None

    def __enter__(self) -> "T5Dev200DiagnosticRunner":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def run(
        self,
        questions_source: Path,
        output_directory: Path,
        *,
        reference_answers: dict[str, str] | None = None,
    ) -> T5DiagnosticBatchManifest:
        """Execute or resume the diagnostic run on all questions in order."""
        # 1. Diagnostic schema identity gate
        if self._identity.diagnostic_schema_version != T5_DIAGNOSTIC_SCHEMA_VERSION:
            raise ArtifactCompatibilityError(
                f"Diagnostic schema version in execution identity ({self._identity.diagnostic_schema_version}) "
                f"does not match runner schema version ({T5_DIAGNOSTIC_SCHEMA_VERSION})"
            )

        # 2. Official scorer gate
        if self._scorer_module is None or not hasattr(self._scorer_module, "eval_qa"):
            raise DataValidationError("Official scoring module is required and must expose eval_qa()")

        # 3. Question source and population hash gate
        source_hash = self._sha256_file(questions_source)
        questions = self._loader.load_questions(
            questions_source,
            require_reference_answers=False,
        )
        if self._sha256_file(questions_source) != source_hash:
            raise DataValidationError("Question source modified during load")

        question_ids = [q.question_id for q in questions]
        ordered_qids_hash = sha256("\n".join(question_ids).encode("utf-8")).hexdigest()

        if ordered_qids_hash != self._identity.dev200_ordered_ids_sha256:
            raise ArtifactCompatibilityError(
                f"Question population ordered ID digest ({ordered_qids_hash}) does not match "
                f"execution identity ({self._identity.dev200_ordered_ids_sha256})"
            )

        # 4. Reference answer (truth) coverage gate
        truth_map = dict(reference_answers or {})
        for q in questions:
            if q.reference_answer and q.question_id not in truth_map:
                truth_map[q.question_id] = q.reference_answer

        missing_truth_ids = [
            q.question_id for q in questions
            if q.question_id not in truth_map or not str(truth_map[q.question_id]).strip()
        ]
        if missing_truth_ids:
            raise DataValidationError(
                f"Missing gold reference answers for {len(missing_truth_ids)} questions "
                f"(first missing: {missing_truth_ids[:5]})"
            )

        output = output_directory.resolve()
        output.mkdir(parents=True, exist_ok=True)

        results_path = output / BATCH_RECORDS_FILENAME
        diag_path = output / DIAGNOSTIC_RECORDS_FILENAME
        state_path = output / BATCH_STATE_FILENAME
        manifest_path = output / BATCH_MANIFEST_FILENAME
        report_path = output / BATCH_REPORT_FILENAME

        if manifest_path.exists():
            manifest = self._load_manifest(manifest_path)
            public_records = self._load_public_records(results_path)
            diag_records = self._load_diag_records(diag_path)
            self._validate_complete(
                public_records,
                diag_records,
                question_ids,
                manifest,
                source_hash,
                ordered_qids_hash,
                results_path,
                diag_path,
            )
            return manifest

        state = self._prepare_state(
            state_path,
            source_hash=source_hash,
            ordered_qids_hash=ordered_qids_hash,
            question_count=len(questions),
            output=output,
        )

        diag_records = self._load_diag_records(diag_path) if diag_path.exists() else []
        self._validate_diag_prefix(diag_records, question_ids)

        public_records = self._load_public_records(results_path) if results_path.exists() else []

        # Canonical diagnostic-first reconciliation
        if len(public_records) > len(diag_records):
            raise ArtifactCompatibilityError(
                "Public records cannot exceed canonical diagnostic records"
            )

        # Check existing public records match diagnostic public_response
        for i, pub_rec in enumerate(public_records):
            diag_rec = diag_records[i]
            if (
                pub_rec.question_id != diag_rec.question_id
                or pub_rec.response.model_dump_json() != diag_rec.public_response.model_dump_json()
            ):
                raise ArtifactCompatibilityError(
                    f"Public record {pub_rec.question_id} conflicts with canonical diagnostic response"
                )

        # Reconstruct missing public records directly from diagnostic records
        if len(public_records) < len(diag_records):
            missing_public = [
                CompetitionBatchRecord(question_id=d.question_id, response=d.public_response)
                for d in diag_records[len(public_records):]
            ]
            with results_path.open("a", encoding="utf-8") as pub_stream:
                for pub_rec in missing_public:
                    pub_stream.write(pub_rec.model_dump_json() + "\n")
                pub_stream.flush()
                os.fsync(pub_stream.fileno())
            public_records.extend(missing_public)

        # Reconcile state
        completed_ids = [d.question_id for d in diag_records]
        if state.completed_question_ids != completed_ids:
            state = state.model_copy(
                update={
                    "completed_question_ids": completed_ids,
                    "updated_at": self._clock(),
                }
            )
            self._write_json_atomic(state_path, state.model_dump(mode="json"))

        start_index = len(diag_records)

        with (
            results_path.open("a", encoding="utf-8") as pub_stream,
            diag_path.open("a", encoding="utf-8") as diag_stream,
        ):
            for question in questions[start_index:]:
                ref_ans = truth_map[question.question_id]
                self._context.reset(question_id=question.question_id)

                req = LegalQuestionRequest(question=question.question)
                agent_result: AgentRunResult = self._service.answer_result(req)
                response = agent_result.response
                state_obj = agent_result.state

                # Diagnostic overlap proxies
                pre_candidates = [
                    cand
                    for ev in self._context.rerank_events
                    for cand in ev.pre_rerank_candidates
                ]
                post_hits = [
                    hit
                    for ev in self._context.rerank_events
                    for hit in ev.post_rerank_hits
                ]

                pre_overlap = compute_overlap_proxy(ref_ans, pre_candidates)
                post_overlap = compute_overlap_proxy(ref_ans, post_hits)
                sel_overlap = compute_overlap_proxy(
                    ref_ans, state_obj.selected_evidence
                )

                # Score-facing rendering for official scoring
                rendered_pred = render_competition_answer(response)

                r_score, m_score = compute_per_question_scores(
                    self._scorer_module,
                    question.question_id,
                    rendered_pred,
                    ref_ans,
                )

                # Query analysis and variants from actual run
                first_hist = (
                    state_obj.retrieval_history[0]
                    if state_obj.retrieval_history
                    else None
                )
                actual_query = first_hist.query if first_hist else None
                q_analysis = (
                    actual_query.query_analysis
                    if actual_query and isinstance(actual_query.query_analysis, QueryAnalysis)
                    else None
                )
                q_variants = actual_query.query_variants if actual_query else []

                is_gen_fb = "generator_model_error_fallback" in response.warnings
                is_ground_fb = "extractive_fallback_applied" in response.warnings

                diag_record = T5QuestionDiagnosticRecord(
                    question_id=question.question_id,
                    question=question.question,
                    reference_answer=ref_ans,
                    public_response=response,
                    stop_reason=agent_result.stop_reason,
                    total_latency_ms=agent_result.total_latency_ms,
                    selected_strategy=state_obj.selected_strategy,
                    retry_count=state_obj.retry_count,
                    warnings=response.warnings,
                    metadata=response.metadata,
                    query_analysis=q_analysis,
                    query_variants=list(q_variants),
                    retrieval_history=state_obj.retrieval_history,
                    terminal_retrieval_hits=[
                        hit_to_telemetry(h) for h in state_obj.candidate_hits
                    ],
                    selected_evidence=state_obj.selected_evidence,
                    context_grade=state_obj.context_grade,
                    rerank_telemetry=list(self._context.rerank_events),
                    branch_telemetry=list(self._context.branch_events),
                    generator_draft_rejections=list(
                        self._context.rejection_events
                    ),
                    is_generator_model_error_fallback=is_gen_fb,
                    is_grounding_extractive_fallback=is_ground_fb,
                    is_any_extractive_fallback=(is_gen_fb or is_ground_fb),
                    is_insufficient_evidence=response.insufficient_evidence,
                    meteor_score=m_score,
                    rouge_l_score=r_score,
                    pre_rerank_overlap_proxy=pre_overlap,
                    post_rerank_overlap_proxy=post_overlap,
                    selected_evidence_overlap_proxy=sel_overlap,
                )

                pub_record = CompetitionBatchRecord(
                    question_id=question.question_id,
                    response=response,
                )

                # 1. Write canonical diagnostic record & fsync
                diag_stream.write(diag_record.model_dump_json() + "\n")
                diag_stream.flush()
                os.fsync(diag_stream.fileno())
                diag_records.append(diag_record)

                # 2. Write public record projection & fsync
                pub_stream.write(pub_record.model_dump_json() + "\n")
                pub_stream.flush()
                os.fsync(pub_stream.fileno())
                public_records.append(pub_record)

                # 3. Update state atomically
                state = state.model_copy(
                    update={
                        "completed_question_ids": [
                            *state.completed_question_ids,
                            question.question_id,
                        ],
                        "updated_at": self._clock(),
                    }
                )
                self._write_json_atomic(
                    state_path, state.model_dump(mode="json")
                )

                if (
                    len(public_records) % _PROGRESS_INTERVAL == 0
                    or len(public_records) == len(questions)
                ):
                    _LOGGER.info(
                        "t5_diagnostic_batch_progress",
                        extra={
                            "question_count": len(questions),
                            "completed_question_count": len(public_records),
                        },
                    )

        self._validate_prefix(public_records, question_ids)
        self._validate_diag_prefix(diag_records, question_ids)

        insufficient_count = sum(
            r.public_response.insufficient_evidence for r in diag_records
        )
        gen_fallback_count = sum(
            r.is_generator_model_error_fallback for r in diag_records
        )
        grounding_fb_count = sum(
            r.is_grounding_extractive_fallback for r in diag_records
        )
        any_extractive_count = sum(
            r.is_any_extractive_fallback for r in diag_records
        )
        repair_count = sum(
            "grounding_repair_attempted" in r.warnings for r in diag_records
        )

        all_warnings = [w for r in diag_records for w in r.warnings]
        warning_counts = dict(Counter(all_warnings).most_common())

        # Full-dataset official score authority
        preds_all = {
            r.question_id: {
                "answer": render_competition_answer(r.public_response)
            }
            for r in diag_records
        }
        truth_all = {
            q.question_id: truth_map[q.question_id]
            for q in questions
            if q.question_id in truth_map
        }

        if set(preds_all) != set(truth_all) or set(preds_all) != set(question_ids) or len(preds_all) != len(questions):
            raise DataValidationError("Incomplete prediction or truth coverage for full-dataset official scoring")

        try:
            full_scores = self._scorer_module.eval_qa(preds_all, truth_all)
            if "rouge" not in full_scores or "meteor" not in full_scores:
                raise DataValidationError("Official scoring output missing rouge or meteor")
            official_rouge = float(full_scores["rouge"])
            official_meteor = float(full_scores["meteor"])
        except Exception as error:
            if isinstance(error, DataValidationError):
                raise
            raise DataValidationError(f"Full-dataset official scoring failed: {error}") from error

        meteor_scores = [r.meteor_score for r in diag_records]
        rouge_scores = [r.rouge_l_score for r in diag_records]
        mean_m = sum(meteor_scores) / len(meteor_scores)
        mean_r = sum(rouge_scores) / len(rouge_scores)

        manifest = T5DiagnosticBatchManifest(
            execution_identity=self._identity,
            question_source_sha256=source_hash,
            application_config_hash=self._config_hash,
            code_version=__version__,
            ordered_question_ids_sha256=ordered_qids_hash,
            created_at=self._clock(),
            record_count=len(public_records),
            public_records_sha256=self._sha256_file(results_path),
            diagnostic_records_sha256=self._sha256_file(diag_path),
            insufficient_evidence_count=insufficient_count,
            generator_model_error_fallback_count=gen_fallback_count,
            grounding_extractive_fallback_count=grounding_fb_count,
            extractive_fallback_count=any_extractive_count,
            any_extractive_fallback_count=any_extractive_count,
            grounding_repair_count=repair_count,
            official_rouge=official_rouge,
            official_meteor=official_meteor,
            mean_per_question_rouge=mean_r,
            mean_per_question_meteor=mean_m,
            warning_counts=warning_counts,
        )

        # 1. Write summary report FIRST
        report = {
            "experiment": "t5-1-dev200-baseline-replay",
            "execution_identity": self._identity.model_dump(mode="json"),
            "record_count": len(public_records),
            "insufficient_evidence_count": insufficient_count,
            "generator_model_error_fallback_count": gen_fallback_count,
            "grounding_extractive_fallback_count": grounding_fb_count,
            "extractive_fallback_count": any_extractive_count,
            "any_extractive_fallback_count": any_extractive_count,
            "grounding_repair_count": repair_count,
            "official_rouge": official_rouge,
            "official_meteor": official_meteor,
            "mean_per_question_rouge": mean_r,
            "mean_per_question_meteor": mean_m,
            "warning_counts": warning_counts,
            "manifest": manifest.model_dump(mode="json"),
        }
        self._write_json_atomic(report_path, report)

        # 2. Write manifest LAST as the final completion marker
        self._write_json_exclusive(
            manifest_path, manifest.model_dump(mode="json")
        )
        return manifest

    def _prepare_state(
        self,
        path: Path,
        *,
        source_hash: str,
        ordered_qids_hash: str,
        question_count: int,
        output: Path,
    ) -> T5DiagnosticBatchState:
        if path.exists():
            state = self._load_state(path)
            if (
                state.question_source_sha256 != source_hash
                or state.ordered_question_ids_sha256 != ordered_qids_hash
                or state.application_config_hash != self._config_hash
                or state.code_version != __version__
                or state.question_count != question_count
                or state.execution_identity.model_dump_json() != self._identity.model_dump_json()
            ):
                raise ArtifactCompatibilityError(
                    "T5 diagnostic batch recovery identity is incompatible"
                )
            return state

        unexpected = [
            p
            for p in output.iterdir()
            if p.name not in {BATCH_RECORDS_FILENAME, DIAGNOSTIC_RECORDS_FILENAME}
        ]
        if unexpected or (output / BATCH_RECORDS_FILENAME).exists() or (output / DIAGNOSTIC_RECORDS_FILENAME).exists():
            raise ArtifactCompatibilityError(
                "Diagnostic batch output has no compatible recovery state"
            )

        now = self._clock()
        state = T5DiagnosticBatchState(
            execution_identity=self._identity,
            question_source_sha256=source_hash,
            application_config_hash=self._config_hash,
            code_version=__version__,
            question_count=question_count,
            ordered_question_ids_sha256=ordered_qids_hash,
            created_at=now,
            updated_at=now,
        )
        self._write_json_atomic(path, state.model_dump(mode="json"))
        return state

    def _validate_complete(
        self,
        pub_records: list[CompetitionBatchRecord],
        diag_records: list[T5QuestionDiagnosticRecord],
        question_ids: list[str],
        manifest: T5DiagnosticBatchManifest,
        source_hash: str,
        ordered_qids_hash: str,
        results_path: Path,
        diag_path: Path,
    ) -> None:
        if (
            manifest.question_source_sha256 != source_hash
            or manifest.ordered_question_ids_sha256 != ordered_qids_hash
            or manifest.application_config_hash != self._config_hash
            or manifest.code_version != __version__
            or manifest.execution_identity.model_dump_json() != self._identity.model_dump_json()
        ):
            raise ArtifactCompatibilityError(
                "Completed diagnostic batch manifest identity is incompatible"
            )
        self._validate_prefix(pub_records, question_ids)
        self._validate_diag_prefix(diag_records, question_ids)
        if (
            len(pub_records) != len(question_ids)
            or len(diag_records) != len(question_ids)
            or manifest.record_count != len(pub_records)
            or manifest.public_records_sha256 != self._sha256_file(results_path)
            or manifest.diagnostic_records_sha256 != self._sha256_file(diag_path)
        ):
            raise ArtifactCompatibilityError(
                "Completed diagnostic batch record counts or checksums do not match manifest"
            )

    @staticmethod
    def _validate_prefix(
        records: list[CompetitionBatchRecord], question_ids: list[str]
    ) -> None:
        ids = [r.question_id for r in records]
        if ids != question_ids[: len(ids)]:
            raise ArtifactCompatibilityError(
                "Public batch records are not an ordered prefix of official questions"
            )

    @staticmethod
    def _validate_diag_prefix(
        records: list[T5QuestionDiagnosticRecord], question_ids: list[str]
    ) -> None:
        ids = [r.question_id for r in records]
        if ids != question_ids[: len(ids)]:
            raise ArtifactCompatibilityError(
                "Diagnostic records are not an ordered prefix of official questions"
            )

    @classmethod
    def _load_jsonl_with_trailing_recovery(
        cls,
        path: Path,
        record_cls: type[T],
    ) -> list[T]:
        if not path.exists():
            return []

        raw_bytes = path.read_bytes()
        if not raw_bytes:
            return []

        ends_with_newline = raw_bytes.endswith(b"\n")
        lines_bytes = raw_bytes.splitlines(keepends=True)
        if not lines_bytes:
            return []

        records: list[T] = []
        normalized_bytes = bytearray()
        needs_rewrite = False

        for i, line_b in enumerate(lines_bytes):
            is_last_line = (i == len(lines_bytes) - 1)
            line_str = line_b.decode("utf-8", errors="replace").strip()
            if not line_str:
                raise ArtifactCompatibilityError(
                    f"JSONL checkpoint {path.name} contains a blank record on line {i+1}"
                )

            # Step 1: Check syntax validity
            try:
                parsed_json = json.loads(line_str)
            except json.JSONDecodeError as syntax_err:
                # Syntax error on the LAST line with NO trailing newline in the file -> Interrupted partial write
                if is_last_line and not ends_with_newline:
                    _LOGGER.warning(
                        "interrupted_trailing_jsonl_line_truncated",
                        extra={"path": str(path), "truncated_bytes": len(line_b)},
                    )
                    needs_rewrite = True
                    break
                else:
                    # Syntax error in middle or on newline-terminated line -> FAIL CLOSED
                    raise ArtifactCompatibilityError(
                        f"JSONL checkpoint {path.name} contains invalid JSON syntax on line {i+1}: {syntax_err}"
                    ) from syntax_err

            # Step 2: Validate against Pydantic schema
            try:
                record = record_cls.model_validate(parsed_json)
                records.append(record)
                line_to_append = line_b if line_b.endswith(b"\n") else line_b + b"\n"
                normalized_bytes.extend(line_to_append)
                if is_last_line and not line_b.endswith(b"\n"):
                    needs_rewrite = True
            except (ValidationError, ValueError) as schema_err:
                # Schema errors ALWAYS fail closed (corruption, not partial write)
                raise ArtifactCompatibilityError(
                    f"JSONL checkpoint {path.name} contains schema-invalid record on line {i+1}: {schema_err}"
                ) from schema_err

        if needs_rewrite:
            temp_path = path.with_name(f".{path.name}.tmp")
            try:
                with temp_path.open("wb") as stream:
                    stream.write(bytes(normalized_bytes))
                    stream.flush()
                    os.fsync(stream.fileno())
                temp_path.replace(path)
            except OSError as error:
                temp_path.unlink(missing_ok=True)
                raise ArtifactCompatibilityError(
                    f"Failed to durably persist normalized JSONL checkpoint {path.name}"
                ) from error

        return records

    @classmethod
    def _load_public_records(cls, path: Path) -> list[CompetitionBatchRecord]:
        return cls._load_jsonl_with_trailing_recovery(path, CompetitionBatchRecord)

    @classmethod
    def _load_diag_records(cls, path: Path) -> list[T5QuestionDiagnosticRecord]:
        return cls._load_jsonl_with_trailing_recovery(path, T5QuestionDiagnosticRecord)

    @staticmethod
    def _load_state(path: Path) -> T5DiagnosticBatchState:
        try:
            return T5DiagnosticBatchState.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ArtifactCompatibilityError(
                "Diagnostic batch state is invalid"
            ) from error

    @staticmethod
    def _load_manifest(path: Path) -> T5DiagnosticBatchManifest:
        try:
            return T5DiagnosticBatchManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ArtifactCompatibilityError(
                "Diagnostic batch manifest is invalid"
            ) from error

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, payload: object) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        payload, ensure_ascii=False, indent=2, sort_keys=True
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ArtifactCompatibilityError(
                f"State could not be persisted: {error}"
            ) from error

    @classmethod
    def _write_json_exclusive(cls, path: Path, payload: object) -> None:
        if path.exists():
            raise ArtifactCompatibilityError(
                "Diagnostic manifest already exists"
            )
        cls._write_json_atomic(path, payload)
