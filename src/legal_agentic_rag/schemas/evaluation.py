"""Versioned benchmark, per-case metric, and evaluation report schemas."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from legal_agentic_rag.schemas.answering import AnswerResponse
from legal_agentic_rag.schemas.retrieval import RetrievalStrategy


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


def _sha256(value: str) -> str:
    normalized = value.casefold()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("SHA-256 value must be hexadecimal")
    return normalized


class EvaluationTargetGranularity(StrEnum):
    """Identity level used by retrieval relevance labels."""

    CHUNK = "chunk"
    DOCUMENT = "document"


class EvaluationMetricDirection(StrEnum):
    """Optimization direction explicitly selected by the evaluator."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class EvaluationSelectionMode(StrEnum):
    """Whether comparison only reports Pareto candidates or selects one."""

    PARETO_ONLY = "pareto_only"
    LEXICOGRAPHIC = "lexicographic"


class EvaluationBenchmarkLabelStatus(StrEnum):
    """Declared review level of labels in an evaluation benchmark."""

    DIAGNOSTIC = "diagnostic"
    HUMAN_REVIEWED = "human_reviewed"
    COMPETITION_OFFICIAL = "competition_official"


class EvaluationBenchmarkManifest(BaseModel):
    """Pinned identity and declared label provenance for one benchmark."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    benchmark_name: str
    benchmark_version: str
    label_status: EvaluationBenchmarkLabelStatus
    dataset_name: str
    dataset_revision: str
    case_count: int = Field(gt=0)
    benchmark_sha256: str = Field(min_length=64, max_length=64)
    target_granularities: list[EvaluationTargetGranularity] = Field(
        min_length=1
    )
    verified_at: datetime | None = None
    label_provenance_reference: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "benchmark_name",
        "benchmark_version",
        "dataset_name",
        "dataset_revision",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("benchmark_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("target_granularities")
    @classmethod
    def validate_granularities(
        cls,
        values: list[EvaluationTargetGranularity],
    ) -> list[EvaluationTargetGranularity]:
        if len(values) != len(set(values)):
            raise ValueError("target granularities must be unique")
        return values

    @field_validator("label_provenance_reference", mode="before")
    @classmethod
    def normalize_provenance_reference(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("verified_at")
    @classmethod
    def validate_verified_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("verified_at must include timezone information")
        return value

    @model_validator(mode="after")
    def validate_label_provenance(self) -> "EvaluationBenchmarkManifest":
        trusted = self.label_status in {
            EvaluationBenchmarkLabelStatus.HUMAN_REVIEWED,
            EvaluationBenchmarkLabelStatus.COMPETITION_OFFICIAL,
        }
        if trusted and (
            self.verified_at is None
            or self.verified_at.tzinfo is None
            or self.label_provenance_reference is None
        ):
            raise ValueError(
                "trusted benchmark labels require timestamped provenance"
            )
        return self


class EvaluationCase(BaseModel):
    """One labeled question without competition-specific field names."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    target_granularity: EvaluationTargetGranularity
    relevance_grades: dict[str, int]
    reference_answer: str | None = None
    expected_citation_chunk_ids: list[str] = Field(default_factory=list)
    should_abstain: bool | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("case_id", "question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("reference_answer", mode="before")
    @classmethod
    def normalize_reference(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("relevance_grades")
    @classmethod
    def validate_relevance(cls, values: dict[str, int]) -> dict[str, int]:
        if not values:
            raise ValueError("relevance_grades must not be empty")
        normalized = {_non_empty(key): value for key, value in values.items()}
        if any(value <= 0 for value in normalized.values()):
            raise ValueError("relevance grades must be positive")
        return normalized

    @field_validator("expected_citation_chunk_ids", "tags")
    @classmethod
    def validate_unique_text(cls, values: list[str]) -> list[str]:
        normalized = [_non_empty(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("values must be unique")
        return normalized


class RetrievalCaseMetrics(BaseModel):
    """Transparent metrics for one ranked retrieval response."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    reciprocal_rank: float = Field(ge=0, le=1)
    first_relevant_rank: int | None = Field(default=None, ge=1)


class GenerationCaseMetrics(BaseModel):
    """Automatic generation metrics computed only when labels exist."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    exact_match: float | None = Field(default=None, ge=0, le=1)
    meteor: float | None = Field(default=None, ge=0, le=1)
    rouge_l: float | None = Field(default=None, ge=0, le=1)
    abstention_accuracy: float | None = Field(default=None, ge=0, le=1)
    citation_precision: float | None = Field(default=None, ge=0, le=1)
    citation_recall: float | None = Field(default=None, ge=0, le=1)


class EvaluationCaseResult(BaseModel):
    """Compact inspectable outcome for one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    success: bool
    retrieved_ids: list[str] = Field(default_factory=list)
    missing_relevant_ids: list[str] = Field(default_factory=list)
    retrieval_metrics: RetrievalCaseMetrics | None = None
    generation_metrics: GenerationCaseMetrics | None = None
    answer_response: AnswerResponse | None = None
    retrieval_latency_ms: float | None = Field(default=None, ge=0)
    generation_latency_ms: float | None = Field(default=None, ge=0)
    error_stage: str | None = None
    error_type: str | None = None

    @model_validator(mode="after")
    def validate_success(self) -> "EvaluationCaseResult":
        if self.success and (
            self.retrieval_metrics is None
            or self.error_stage is not None
            or self.error_type is not None
        ):
            raise ValueError("successful result requires retrieval metrics")
        if not self.success and (
            self.error_stage is None or self.error_type is None
        ):
            raise ValueError("failed result requires a sanitized error")
        return self


class LatencySummary(BaseModel):
    """Distribution summary for successful measured operations."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    count: int = Field(ge=0)
    mean_ms: float | None = Field(default=None, ge=0)
    p50_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    max_ms: float | None = Field(default=None, ge=0)


class EvaluationResourceUsage(BaseModel):
    """Portable process-level resource observations."""

    model_config = ConfigDict(extra="forbid")

    wall_time_ms: float = Field(ge=0)
    process_cpu_time_ms: float = Field(ge=0)
    python_peak_traced_memory_bytes: int = Field(ge=0)
    accelerator_name: str | None = Field(default=None, min_length=1)
    accelerator_peak_memory_bytes: int | None = Field(default=None, ge=0)


class EvaluationSummary(BaseModel):
    """Versioned aggregate report without inventing unavailable metrics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.3"
    run_id: str
    created_at: datetime
    code_version: str
    benchmark_name: str
    benchmark_sha256: str
    benchmark_manifest_sha256: str = Field(min_length=64, max_length=64)
    benchmark_version: str = Field(min_length=1)
    benchmark_label_status: EvaluationBenchmarkLabelStatus
    benchmark_verified_at: datetime | None = None
    benchmark_label_provenance_reference: str | None = Field(
        default=None,
        min_length=1,
    )
    dataset_name: str | None = Field(default=None, min_length=1)
    dataset_revision: str | None = Field(default=None, min_length=1)
    strategy: RetrievalStrategy
    cutoffs: list[int]
    case_count: int = Field(ge=0)
    successful_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    generation_metrics: dict[str, float] = Field(default_factory=dict)
    metric_case_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_latency: LatencySummary
    generation_latency: LatencySummary
    resources: EvaluationResourceUsage
    artifact_versions: dict[str, str] = Field(default_factory=dict)
    runtime_config_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    component_provenance: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "benchmark_sha256",
        "benchmark_manifest_sha256",
        "runtime_config_sha256",
    )
    @classmethod
    def validate_sha256_fields(cls, value: str | None) -> str | None:
        return _sha256(value) if value is not None else None

    @field_validator("benchmark_verified_at")
    @classmethod
    def validate_benchmark_verified_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError(
                "benchmark_verified_at must include timezone information"
            )
        return value

    @model_validator(mode="after")
    def validate_trusted_benchmark_provenance(self) -> "EvaluationSummary":
        if (
            self.benchmark_label_status
            != EvaluationBenchmarkLabelStatus.DIAGNOSTIC
            and (
                self.benchmark_verified_at is None
                or self.benchmark_verified_at.tzinfo is None
                or self.benchmark_label_provenance_reference is None
            )
        ):
            raise ValueError(
                "trusted benchmark summary requires label provenance"
            )
        return self


class EvaluationRunResult(BaseModel):
    """Summary plus per-case results consumed by report persistence."""

    model_config = ConfigDict(extra="forbid")

    summary: EvaluationSummary
    cases: list[EvaluationCaseResult]

    @model_validator(mode="after")
    def validate_counts(self) -> "EvaluationRunResult":
        if self.summary.case_count != len(self.cases):
            raise ValueError("summary case count must match case results")
        return self


class EvaluationObjective(BaseModel):
    """Persisted comparison objective without assuming a competition metric."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    metric: str = Field(min_length=1)
    direction: EvaluationMetricDirection
    eligibility_threshold: float | None = None
    maximum_regression: float | None = Field(default=None, ge=0)


class EvaluationCandidateResult(BaseModel):
    """Comparable metric projection and eligibility of one evaluated candidate."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    strategy: RetrievalStrategy
    runtime_config_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    component_provenance: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_versions: dict[str, str] = Field(default_factory=dict)
    metric_values: dict[str, float]
    eligible: bool
    exclusion_reasons: list[str] = Field(default_factory=list)
    dominated_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_eligibility(self) -> "EvaluationCandidateResult":
        """Keep explicit eligibility aligned with exclusion reasons."""
        if self.eligible == bool(self.exclusion_reasons):
            raise ValueError(
                "eligible must be false exactly when exclusion reasons exist"
            )
        return self


class EvaluationComparisonReport(BaseModel):
    """Immutable multi-run comparison with conservative selection semantics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.1"
    comparison_id: str = Field(min_length=1)
    comparison_name: str = Field(min_length=1)
    created_at: datetime
    benchmark_name: str = Field(min_length=1)
    benchmark_sha256: str = Field(min_length=64, max_length=64)
    benchmark_manifest_sha256: str = Field(min_length=64, max_length=64)
    benchmark_version: str = Field(min_length=1)
    benchmark_label_status: EvaluationBenchmarkLabelStatus
    benchmark_verified_at: datetime | None = None
    benchmark_label_provenance_reference: str | None = Field(
        default=None,
        min_length=1,
    )
    case_count: int = Field(gt=0)
    objectives: list[EvaluationObjective] = Field(min_length=1)
    selection_mode: EvaluationSelectionMode
    baseline_candidate_id: str | None = None
    candidates: list[EvaluationCandidateResult] = Field(min_length=2)
    pareto_candidate_ids: list[str] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("benchmark_sha256", "benchmark_manifest_sha256")
    @classmethod
    def validate_sha256_fields(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("benchmark_verified_at")
    @classmethod
    def validate_benchmark_verified_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError(
                "benchmark_verified_at must include timezone information"
            )
        return value

    @model_validator(mode="after")
    def validate_candidate_references(self) -> "EvaluationComparisonReport":
        """Require all Pareto and selected identities to reference candidates."""
        identities = [candidate.candidate_id for candidate in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("comparison candidate IDs must be unique")
        known = set(identities)
        if not set(self.pareto_candidate_ids) <= known:
            raise ValueError("Pareto identities must reference candidates")
        if self.selected_candidate_id not in (None, *known):
            raise ValueError("selected identity must reference a candidate")
        if self.baseline_candidate_id not in (None, *known):
            raise ValueError("baseline identity must reference a candidate")
        if (
            self.selection_mode == EvaluationSelectionMode.PARETO_ONLY
            and self.selected_candidate_id is not None
        ):
            raise ValueError("pareto-only comparison cannot select a candidate")
        if (
            self.benchmark_label_status
            != EvaluationBenchmarkLabelStatus.DIAGNOSTIC
            and (
                self.benchmark_verified_at is None
                or self.benchmark_verified_at.tzinfo is None
                or self.benchmark_label_provenance_reference is None
            )
        ):
            raise ValueError(
                "trusted comparison requires benchmark label provenance"
            )
        return self


class RetrievalDiagnosticSignal(StrEnum):
    """Observable retrieval risk signals that do not claim gold relevance."""

    NO_BM25_HITS = "no_bm25_hits"
    NO_DENSE_HITS = "no_dense_hits"
    NO_HYBRID_HITS = "no_hybrid_hits"
    ZERO_BRANCH_OVERLAP = "zero_branch_overlap"
    LOW_DOCUMENT_DIVERSITY = "low_document_diversity"
    EXPLICIT_REFERENCE_NOT_RETRIEVED = "explicit_reference_not_retrieved"
    LOW_ANSWER_TERM_COVERAGE = "low_answer_term_coverage"
    RETRIEVAL_WARNING = "retrieval_warning"
    RETRIEVAL_ERROR = "retrieval_error"


class RetrievalBranchDiagnostic(BaseModel):
    """Content-free identity and count projection of one retrieval branch."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    strategy: RetrievalStrategy
    hit_count: int = Field(ge=0)
    unique_document_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "RetrievalBranchDiagnostic":
        if self.hit_count != len(self.chunk_ids):
            raise ValueError("diagnostic hit count must match chunk IDs")
        if self.hit_count != len(self.document_ids):
            raise ValueError("diagnostic hit count must match document IDs")
        if self.unique_document_count != len(set(self.document_ids)):
            raise ValueError("diagnostic document count must match document IDs")
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("diagnostic chunk IDs must be unique")
        return self


class RetrievalDiagnosticCase(BaseModel):
    """Per-question retrieval observations without a relevance judgment."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    question_id: str = Field(min_length=1)
    success: bool
    reranker_included: bool = False
    query_intent: str | None = None
    query_variant_count: int = Field(default=0, ge=0)
    branches: list[RetrievalBranchDiagnostic] = Field(default_factory=list)
    bm25_dense_overlap_count: int = Field(default=0, ge=0)
    bm25_dense_jaccard: float = Field(default=0, ge=0, le=1)
    hybrid_document_diversity: float = Field(default=0, ge=0, le=1)
    explicit_reference_match: bool | None = None
    answer_term_coverage: float | None = Field(default=None, ge=0, le=1)
    hybrid_rerank_overlap_count: int | None = Field(default=None, ge=0)
    hybrid_rerank_jaccard: float | None = Field(default=None, ge=0, le=1)
    hybrid_rerank_document_diversity: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    reranked_explicit_reference_match: bool | None = None
    hybrid_rerank_answer_term_coverage: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    hybrid_rerank_answer_term_coverage_delta: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    mean_absolute_rank_change: float | None = Field(default=None, ge=0)
    signals: list[RetrievalDiagnosticSignal] = Field(default_factory=list)
    error_type: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "RetrievalDiagnosticCase":
        if self.success == (self.error_type is not None):
            raise ValueError("diagnostic error type must match failed outcome")
        expected_branch_count = 4 if self.reranker_included else 3
        if self.success and len(self.branches) != expected_branch_count:
            raise ValueError(
                "successful diagnostic has an incompatible branch count"
            )
        reranker_metrics = (
            self.hybrid_rerank_overlap_count,
            self.hybrid_rerank_jaccard,
            self.hybrid_rerank_document_diversity,
        )
        if self.success and self.reranker_included != all(
            value is not None for value in reranker_metrics
        ):
            raise ValueError(
                "reranker diagnostic metrics must match reranker inclusion"
            )
        if not self.reranker_included and any(
            value is not None
            for value in (
                *reranker_metrics,
                self.reranked_explicit_reference_match,
                self.hybrid_rerank_answer_term_coverage,
                self.hybrid_rerank_answer_term_coverage_delta,
                self.mean_absolute_rank_change,
            )
        ):
            raise ValueError(
                "non-reranker diagnostic cannot contain reranker metrics"
            )
        if (self.hybrid_rerank_answer_term_coverage is None) != (
            self.hybrid_rerank_answer_term_coverage_delta is None
        ):
            raise ValueError(
                "reranked answer coverage and delta must be present together"
            )
        if len(self.signals) != len(set(self.signals)):
            raise ValueError("diagnostic signals must be unique")
        return self


class RetrievalDiagnosticsReport(BaseModel):
    """Immutable aggregate for answer-level, non-gold retrieval diagnostics."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.1"
    created_at: datetime
    code_version: str = Field(min_length=1)
    question_source_sha256: str = Field(min_length=64, max_length=64)
    application_config_sha256: str = Field(min_length=64, max_length=64)
    question_count: int = Field(gt=0)
    successful_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    top_k: int = Field(gt=0)
    candidate_k: int = Field(gt=0)
    include_reranker: bool = False
    max_cases: int | None = Field(default=None, gt=0)
    low_document_diversity_threshold: float = Field(ge=0, le=1)
    low_answer_term_coverage_threshold: float = Field(ge=0, le=1)
    mean_bm25_dense_jaccard: float = Field(ge=0, le=1)
    mean_hybrid_document_diversity: float = Field(ge=0, le=1)
    mean_answer_term_coverage: float | None = Field(default=None, ge=0, le=1)
    mean_hybrid_rerank_jaccard: float | None = Field(default=None, ge=0, le=1)
    mean_hybrid_rerank_document_diversity: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    mean_hybrid_rerank_answer_term_coverage_delta: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    mean_absolute_rank_change: float | None = Field(default=None, ge=0)
    signal_counts: dict[RetrievalDiagnosticSignal, int] = Field(default_factory=dict)
    cases: list[RetrievalDiagnosticCase] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("question_source_sha256", "application_config_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("diagnostic timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def validate_case_counts(self) -> "RetrievalDiagnosticsReport":
        if self.question_count != len(self.cases):
            raise ValueError("diagnostic question count must match cases")
        if self.successful_case_count + self.failed_case_count != self.question_count:
            raise ValueError("diagnostic outcome counts must match cases")
        identities = [case.question_id for case in self.cases]
        if len(identities) != len(set(identities)):
            raise ValueError("diagnostic question IDs must be unique")
        if self.top_k > self.candidate_k:
            raise ValueError("diagnostic top_k cannot exceed candidate_k")
        observed_signal_counts = Counter(
            signal for case in self.cases for signal in case.signals
        )
        if self.signal_counts != dict(observed_signal_counts):
            raise ValueError("diagnostic signal counts must match cases")
        if any(
            case.reranker_included != self.include_reranker
            for case in self.cases
            if case.success
        ):
            raise ValueError(
                "diagnostic report and successful cases disagree on reranker"
            )
        required_reranker_means = (
            self.mean_hybrid_rerank_jaccard,
            self.mean_hybrid_rerank_document_diversity,
        )
        if self.include_reranker != all(
            value is not None for value in required_reranker_means
        ):
            raise ValueError(
                "reranker aggregate metrics must match reranker inclusion"
            )
        if not self.include_reranker and any(
            value is not None
            for value in (
                *required_reranker_means,
                self.mean_hybrid_rerank_answer_term_coverage_delta,
                self.mean_absolute_rank_change,
            )
        ):
            raise ValueError(
                "non-reranker report cannot contain reranker aggregates"
            )
        return self
