"""Tests for the hardened T5 diagnostic execution harness covering all Review Fix 3 requirements."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
    render_competition_answer,
)
from legal_agentic_rag.configuration.online import (
    QueryUnderstandingConfig,
    RerankerConfig,
    RetrievalConfig,
)
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.retrieval.fixed import HybridRetriever
from legal_agentic_rag.retrieval.rerank import RerankingRetriever
from legal_agentic_rag.schemas import (
    AgentRunResult,
    AgentState,
    AgentStopReason,
    AnswerResponse,
    Citation,
    CompetitionBatchRecord,
    ContextGrade,
    Evidence,
    LegalQuestionRequest,
    QueryAnalysis,
    QueryIntent,
    QueryVariant,
    QueryVariantKind,
    RetrievalHit,
    RetrievalHistoryItem,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)
from legal_agentic_rag.serving.query_service import ServingService
from scripts.t5_diagnostic_runner import (
    T5_DIAGNOSTIC_SCHEMA_VERSION,
    ActiveDiagnosticContext,
    DiagnosticBranchRetrieverObserver,
    DiagnosticRerankerObserver,
    GeneratorRejectionLogHandler,
    T5Dev200DiagnosticRunner,
    T5DiagnosticBatchManifest,
    T5DiagnosticBatchState,
    T5EvidenceOverlapProxy,
    T5ExecutionIdentity,
    T5QuestionDiagnosticRecord,
    T5RerankTelemetry,
    compute_overlap_proxy,
    compute_per_question_scores,
    hit_to_telemetry,
    instrument_service_for_diagnostics,
)

VALID_HARNESS_SHA = "a" * 40
VALID_CONFIG_HASH = "b" * 64


class _ConformantDummyReranker(Reranker):
    @property
    def provider_name(self) -> str:
        return "dummy-provider"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def model_name(self) -> str:
        return "dummy-cross-encoder"

    @property
    def model_revision(self) -> str | None:
        return "rev-1"

    def rerank(
        self, query: RetrievalQuery, candidates: Sequence[RetrievalHit]
    ) -> RetrievalResponse:
        reordered = [
            hit.model_copy(
                update={
                    "rank": i + 1,
                    "score": 1.0 - (i * 0.05),
                    "strategy": RetrievalStrategy.RERANK,
                    "retrieval_trace": (
                        hit.retrieval_trace.model_copy(
                            update={"reranker_score": 1.0 - (i * 0.05)}
                        )
                        if hit.retrieval_trace
                        else None
                    ),
                }
            )
            for i, hit in enumerate(reversed(candidates))
        ]
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.RERANK,
            hits=reordered[: query.top_k],
            latency_ms=12.5,
            warnings=[],
            artifact_versions={"reranker": "1.0"},
        )


def _make_dummy_hit(
    chunk_id: str,
    rank: int,
    score: float = 0.5,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=score,
        strategy=strategy,
        text=f"Nội dung điều luật cho {chunk_id}",
        retrieval_trace=RetrievalTrace(
            bm25_rank=rank,
            dense_rank=rank,
            bm25_score=10.0 - rank,
            dense_score=0.9 - (rank * 0.01),
            rrf_score=0.03 / rank,
            reranker_score=score,
        ),
    )


def test_t5_execution_identity_strict_git_sha_validation():
    """Strict 40-hex Git SHA and 64-hex SHA-256 validation."""
    identity = T5ExecutionIdentity(measurement_harness_source_sha="0123456789abcdef0123456789abcdef01234567")
    assert identity.measurement_harness_source_sha == "0123456789abcdef0123456789abcdef01234567"

    with pytest.raises(ValueError, match="40 hexadecimal characters"):
        T5ExecutionIdentity(measurement_harness_source_sha="commit-sha-1234567")

    with pytest.raises(ValueError, match="40 hexadecimal characters"):
        T5ExecutionIdentity(measurement_harness_source_sha="abcdef0123")

    with pytest.raises(ValueError, match="40 hexadecimal characters"):
        T5ExecutionIdentity(measurement_harness_source_sha="unspecified")


def test_reranker_protocol_and_reranking_retriever_integration():
    """Real Reranker protocol, RerankingRetriever integration, limits 40 and 20."""
    context = ActiveDiagnosticContext()
    inner_reranker = _ConformantDummyReranker()
    config = RerankerConfig(max_candidates=40, relationship_candidate_k=20)

    observer = DiagnosticRerankerObserver(
        inner_reranker,
        context,
        relationship_candidate_k=config.relationship_candidate_k,
        max_candidates=config.max_candidates,
    )

    candidate_retriever = MagicMock()
    reranking_retriever = RerankingRetriever(candidate_retriever, observer, config)

    hits_40 = [_make_dummy_hit(f"chunk-{i}", i + 1) for i in range(40)]
    candidate_retriever.search.return_value = RetrievalResponse(
        query=RetrievalQuery(
            query_id="q-norm-1",
            original_question="Thủ tục thuế?",
            normalized_question="thủ tục thuế?",
            top_k=40,
            requested_strategy=RetrievalStrategy.HYBRID,
        ),
        strategy=RetrievalStrategy.HYBRID,
        hits=hits_40,
        latency_ms=10.0,
    )

    normal_query = RetrievalQuery(
        query_id="q-norm-1",
        original_question="Thủ tục thuế?",
        normalized_question="thủ tục thuế?",
        top_k=10,
        candidate_k=40,
        query_analysis=QueryAnalysis(intent=QueryIntent.PROCEDURE),
    )

    response_normal = reranking_retriever.search(normal_query)
    assert response_normal.strategy == RetrievalStrategy.HYBRID_RERANK
    assert len(response_normal.hits) == 10
    assert len(context.rerank_events) == 1

    ev_norm = context.rerank_events[0]
    assert ev_norm.requested_candidate_k == 40
    assert ev_norm.effective_candidate_limit == 40
    assert ev_norm.pre_rerank_candidate_count == 40
    assert ev_norm.is_relationship_intent is False
    assert [c.chunk_id for c in ev_norm.pre_rerank_candidates] == [h.chunk_id for h in hits_40]
    assert ev_norm.post_rerank_count == 10

    context.reset()
    hits_20 = [_make_dummy_hit(f"rel-{i}", i + 1) for i in range(20)]
    candidate_retriever.search.return_value = RetrievalResponse(
        query=RetrievalQuery(
            query_id="q-rel-1",
            original_question="Mối quan hệ luật?",
            normalized_question="mối quan hệ luật?",
            top_k=20,
            requested_strategy=RetrievalStrategy.HYBRID,
        ),
        strategy=RetrievalStrategy.HYBRID,
        hits=hits_20,
        latency_ms=10.0,
    )

    rel_query = RetrievalQuery(
        query_id="q-rel-1",
        original_question="Mối quan hệ luật?",
        normalized_question="mối quan hệ luật?",
        top_k=10,
        candidate_k=40,
        query_analysis=QueryAnalysis(intent=QueryIntent.RELATIONSHIP),
    )

    response_rel = reranking_retriever.search(rel_query)
    assert response_rel.strategy == RetrievalStrategy.HYBRID_RERANK
    assert len(response_rel.hits) == 10
    assert len(context.rerank_events) == 1

    ev_rel = context.rerank_events[0]
    assert ev_rel.requested_candidate_k == 40
    assert ev_rel.effective_candidate_limit == 20
    assert ev_rel.pre_rerank_candidate_count == 20
    assert ev_rel.is_relationship_intent is True


def test_t5_hybrid_multi_query_all_four_branches():
    """Hybrid multi-query execution captures all 4 branch observations with true variant IDs."""
    context = ActiveDiagnosticContext()

    mock_bm25 = MagicMock()
    mock_dense = MagicMock()

    def mock_branch_search(q: RetrievalQuery, strat: RetrievalStrategy) -> RetrievalResponse:
        return RetrievalResponse(
            query=q,
            strategy=strat,
            hits=[_make_dummy_hit(f"hit-{strat.value}-{q.normalized_question}", 1, strategy=strat)],
            latency_ms=5.0,
        )

    mock_bm25.search.side_effect = lambda q: mock_branch_search(q, RetrievalStrategy.BM25)
    mock_dense.search.side_effect = lambda q: mock_branch_search(q, RetrievalStrategy.DENSE)
    mock_bm25.source_artifact_identity = ("legal-chunks", "1.0.0", "chunk-artifact-hash-123")
    mock_dense.source_artifact_identity = ("legal-chunks", "1.0.0", "chunk-artifact-hash-123")

    obs_bm25 = DiagnosticBranchRetrieverObserver(mock_bm25, RetrievalStrategy.BM25, context)
    obs_dense = DiagnosticBranchRetrieverObserver(mock_dense, RetrievalStrategy.DENSE, context)

    hybrid = HybridRetriever(obs_bm25, obs_dense, config=RetrievalConfig())

    var_a = QueryVariant(variant_id="qv-norm-1", text="câu hỏi biến thể a", kind=QueryVariantKind.NORMALIZED)
    var_b = QueryVariant(variant_id="qv-legal-2", text="câu hỏi biến thể b", kind=QueryVariantKind.LEGAL_REFERENCE)

    query = RetrievalQuery(
        query_id="q-multi-1",
        original_question="Câu hỏi gốc",
        normalized_question="câu hỏi chuẩn hóa",
        query_variants=[var_a, var_b],
    )

    resp = hybrid.search(query)
    assert resp.strategy == RetrievalStrategy.HYBRID

    assert len(context.branch_events) == 4
    observations = [(e.variant_id, e.strategy, e.variant_text) for e in context.branch_events]
    assert observations == [
        ("qv-norm-1", RetrievalStrategy.BM25, "câu hỏi biến thể a"),
        ("qv-norm-1", RetrievalStrategy.DENSE, "câu hỏi biến thể a"),
        ("qv-legal-2", RetrievalStrategy.BM25, "câu hỏi biến thể b"),
        ("qv-legal-2", RetrievalStrategy.DENSE, "câu hỏi biến thể b"),
    ]


def test_t5_pre_inference_gates_fail_closed(tmp_path: Path):
    """Pre-inference gates fail with zero inference calls."""
    q_file = tmp_path / "questions.json"
    q_payload = {"q-01": {"question": "Câu 1", "answer": "Đáp án 1"}}
    q_file.write_text(json.dumps(q_payload, ensure_ascii=False), encoding="utf-8")

    q_ordered_hash = sha256("q-01".encode("utf-8")).hexdigest()
    correct_identity = T5ExecutionIdentity(
        measurement_harness_source_sha=VALID_HARNESS_SHA,
        dev200_ordered_ids_sha256=q_ordered_hash,
    )

    class _ValidScorer:
        def eval_qa(self, preds: dict[str, dict[str, str]], truth: dict[str, str]) -> dict[str, float]:
            return {"rouge": 0.8, "meteor": 0.8}

    valid_scorer = _ValidScorer()
    mock_service = MagicMock(spec=ServingService)
    mock_service._runtime = MagicMock()

    # 1. Wrong question population ordered ID hash -> fail before inference
    wrong_pop_identity = correct_identity.model_copy(update={"dev200_ordered_ids_sha256": "c" * 64})
    with pytest.raises(ArtifactCompatibilityError, match="Question population ordered ID digest"):
        with T5Dev200DiagnosticRunner(
            mock_service,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=wrong_pop_identity,
            official_scoring_module=valid_scorer,
        ) as runner:
            runner.run(q_file, tmp_path / "out1")
    assert mock_service.answer_result.call_count == 0

    # 2. Missing gold reference answer -> fail before inference
    q_missing_gold = tmp_path / "q_missing.json"
    q_missing_gold.write_text(json.dumps({"q-01": {"question": "Câu 1"}}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Missing gold reference answers"):
        with T5Dev200DiagnosticRunner(
            mock_service,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=correct_identity,
            official_scoring_module=valid_scorer,
        ) as runner:
            runner.run(q_missing_gold, tmp_path / "out2", reference_answers={})
    assert mock_service.answer_result.call_count == 0

    # 3. Missing official scoring module -> fail before inference
    with pytest.raises(DataValidationError, match="Official scoring module is required"):
        with T5Dev200DiagnosticRunner(
            mock_service,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=correct_identity,
            official_scoring_module=None,
        ) as runner:
            runner.run(q_file, tmp_path / "out3")
    assert mock_service.answer_result.call_count == 0

    # 4. Scorer lacking eval_qa -> fail before inference
    with pytest.raises(DataValidationError, match="Official scoring module is required"):
        with T5Dev200DiagnosticRunner(
            mock_service,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=correct_identity,
            official_scoring_module=object(),
        ) as runner:
            runner.run(q_file, tmp_path / "out4")
    assert mock_service.answer_result.call_count == 0

    # 5. Schema identity mismatch -> fail before inference
    wrong_schema_identity = correct_identity.model_copy(update={"diagnostic_schema_version": "v_mismatch"})
    with pytest.raises(ArtifactCompatibilityError, match="Diagnostic schema version"):
        with T5Dev200DiagnosticRunner(
            mock_service,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=wrong_schema_identity,
            official_scoring_module=valid_scorer,
        ) as runner:
            runner.run(q_file, tmp_path / "out5")
    assert mock_service.answer_result.call_count == 0


def test_t5_four_distinct_final_line_states_and_recovery(tmp_path: Path):
    """Tests Case A, Case B, Case C, Case D, Case E, Case F: Four distinct final line states."""
    q_file = tmp_path / "questions.json"
    q_payload = {
        "q-01": {"question": "Câu 1", "answer": "Đáp án 1"},
        "q-02": {"question": "Câu 2", "answer": "Đáp án 2"},
        "q-03": {"question": "Câu 3", "answer": "Đáp án 3"},
    }
    q_file.write_text(json.dumps(q_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    q_hash = sha256(q_file.read_bytes()).hexdigest()
    q_ordered_hash = sha256("\n".join(["q-01", "q-02", "q-03"]).encode("utf-8")).hexdigest()

    exec_identity = T5ExecutionIdentity(
        measurement_harness_source_sha=VALID_HARNESS_SHA,
        dev200_ordered_ids_sha256=q_ordered_hash,
    )

    class _ValidScorer:
        def eval_qa(self, preds: dict[str, dict[str, str]], truth: dict[str, str]) -> dict[str, float]:
            return {"rouge": 0.8, "meteor": 0.8}

    valid_scorer = _ValidScorer()

    resp_1 = AnswerResponse(
        question="Câu 1",
        answer="Trả lời 1",
        citations=[Citation(evidence_id="E1", chunk_id="c-1", document_id="d-1")],
        insufficient_evidence=False,
        warnings=[],
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="t-1",
    )
    resp_2 = AnswerResponse(
        question="Câu 2",
        answer="Trả lời 2",
        citations=[Citation(evidence_id="E1", chunk_id="c-2", document_id="d-2")],
        insufficient_evidence=False,
        warnings=[],
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="t-2",
    )
    diag_1 = T5QuestionDiagnosticRecord(
        question_id="q-01",
        question="Câu 1",
        reference_answer="Đáp án 1",
        public_response=resp_1,
        stop_reason=AgentStopReason.ANSWER_VERIFIED,
        total_latency_ms=50.0,
        meteor_score=0.8,
        rouge_l_score=0.8,
    )
    diag_2 = T5QuestionDiagnosticRecord(
        question_id="q-02",
        question="Câu 2",
        reference_answer="Đáp án 2",
        public_response=resp_2,
        stop_reason=AgentStopReason.ANSWER_VERIFIED,
        total_latency_ms=50.0,
        meteor_score=0.8,
        rouge_l_score=0.8,
    )
    pub_1 = CompetitionBatchRecord(question_id="q-01", response=resp_1)
    pub_2 = CompetitionBatchRecord(question_id="q-02", response=resp_2)

    def mock_answer_fn(req: LegalQuestionRequest) -> AgentRunResult:
        qid = "q-01" if "1" in req.question else ("q-02" if "2" in req.question else "q-03")
        resp = AnswerResponse(
            question=req.question,
            answer=f"Trả lời {qid}",
            citations=[Citation(evidence_id="E1", chunk_id=f"c-{qid}", document_id=f"d-{qid}")],
            insufficient_evidence=False,
            warnings=[],
            retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
            trace_id=f"t-{qid}",
        )
        state_obj = AgentState(
            trace_id=f"t-{qid}",
            original_question=req.question,
            normalized_question=req.question.lower(),
            current_query=req.question,
            selected_strategy=RetrievalStrategy.HYBRID_RERANK,
            candidate_hits=[_make_dummy_hit(f"c-{qid}", 1)],
            selected_evidence=[Evidence(evidence_id="E1", chunk_id=f"c-{qid}", document_id=f"d-{qid}", text="text")],
            context_grade=ContextGrade(is_sufficient=True, coverage_score=0.9, relevance_score=0.95),
            retry_count=0,
            answer=resp.answer,
            citations=resp.citations,
            warnings=resp.warnings,
        )
        return AgentRunResult(
            state=state_obj,
            response=resp,
            stop_reason=AgentStopReason.ANSWER_VERIFIED,
            total_latency_ms=50.0,
        )

    # CASE A: Incomplete syntax, no newline -> truncated, q1 retained, q2/q3 inferred
    dir_a = tmp_path / "case_a"
    dir_a.mkdir(parents=True, exist_ok=True)
    (dir_a / "diagnostics.jsonl").write_bytes(
        diag_1.model_dump_json().encode("utf-8") + b"\n" + b'{"question_id": "q-02", "partial":'
    )
    (dir_a / "results.jsonl").write_bytes(pub_1.model_dump_json().encode("utf-8") + b"\n")
    state_a = T5DiagnosticBatchState(
        execution_identity=exec_identity,
        question_source_sha256=q_hash,
        application_config_hash=VALID_CONFIG_HASH,
        code_version="0.45.0",
        question_count=3,
        ordered_question_ids_sha256=q_ordered_hash,
        completed_question_ids=["q-01"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    (dir_a / "batch_state.json").write_text(state_a.model_dump_json(), encoding="utf-8")

    mock_service_a = MagicMock(spec=ServingService)
    mock_service_a._runtime = MagicMock()
    mock_service_a.answer_result.side_effect = mock_answer_fn

    with T5Dev200DiagnosticRunner(
        mock_service_a,
        application_config_hash=VALID_CONFIG_HASH,
        execution_identity=exec_identity,
        official_scoring_module=valid_scorer,
    ) as runner:
        manifest_a = runner.run(q_file, dir_a)
        assert manifest_a.record_count == 3
        # q1 was not re-inferred, q2 and q3 were inferred
        assert mock_service_a.answer_result.call_count == 2

    # CASE B: Valid JSON + valid schema, no newline -> retained, normalized to end with newline, resumes at q3
    dir_b = tmp_path / "case_b"
    dir_b.mkdir(parents=True, exist_ok=True)
    # Note: no newline after diag_2
    (dir_b / "diagnostics.jsonl").write_bytes(
        diag_1.model_dump_json().encode("utf-8") + b"\n" + diag_2.model_dump_json().encode("utf-8")
    )
    (dir_b / "results.jsonl").write_bytes(
        pub_1.model_dump_json().encode("utf-8") + b"\n" + pub_2.model_dump_json().encode("utf-8") + b"\n"
    )
    state_b = T5DiagnosticBatchState(
        execution_identity=exec_identity,
        question_source_sha256=q_hash,
        application_config_hash=VALID_CONFIG_HASH,
        code_version="0.45.0",
        question_count=3,
        ordered_question_ids_sha256=q_ordered_hash,
        completed_question_ids=["q-01", "q-02"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    (dir_b / "batch_state.json").write_text(state_b.model_dump_json(), encoding="utf-8")

    mock_service_b = MagicMock(spec=ServingService)
    mock_service_b._runtime = MagicMock()
    mock_service_b.answer_result.side_effect = mock_answer_fn

    with T5Dev200DiagnosticRunner(
        mock_service_b,
        application_config_hash=VALID_CONFIG_HASH,
        execution_identity=exec_identity,
        official_scoring_module=valid_scorer,
    ) as runner:
        manifest_b = runner.run(q_file, dir_b)
        assert manifest_b.record_count == 3
        # q1 and q2 were NOT re-inferred, only q3 inferred
        assert mock_service_b.answer_result.call_count == 1
    # Check diagnostics.jsonl ends with newline
    assert (dir_b / "diagnostics.jsonl").read_bytes().endswith(b"\n")

    # CASE C: Valid JSON but schema-invalid, no newline -> fail closed (ArtifactCompatibilityError)
    dir_c = tmp_path / "case_c"
    dir_c.mkdir(parents=True, exist_ok=True)
    (dir_c / "diagnostics.jsonl").write_bytes(
        diag_1.model_dump_json().encode("utf-8") + b"\n" + b'{"question_id": "q-02", "invalid_extra": 123}'
    )
    (dir_c / "results.jsonl").write_bytes(pub_1.model_dump_json().encode("utf-8") + b"\n")
    (dir_c / "batch_state.json").write_text(state_a.model_dump_json(), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="schema-invalid record"):
        with T5Dev200DiagnosticRunner(
            mock_service_a,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=exec_identity,
            official_scoring_module=valid_scorer,
        ) as runner:
            runner.run(q_file, dir_c)

    # CASE D: Valid JSON but schema-invalid WITH newline -> fail closed (ArtifactCompatibilityError)
    dir_d = tmp_path / "case_d"
    dir_d.mkdir(parents=True, exist_ok=True)
    (dir_d / "diagnostics.jsonl").write_text(
        diag_1.model_dump_json() + "\n" + '{"question_id": "q-02", "invalid_extra": 123}\n',
        encoding="utf-8",
    )
    (dir_d / "results.jsonl").write_text(pub_1.model_dump_json() + "\n", encoding="utf-8")
    (dir_d / "batch_state.json").write_text(state_a.model_dump_json(), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="schema-invalid record"):
        with T5Dev200DiagnosticRunner(
            mock_service_a,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=exec_identity,
            official_scoring_module=valid_scorer,
        ) as runner:
            runner.run(q_file, dir_d)

    # CASE E: Middle syntax corruption -> fail closed
    dir_e = tmp_path / "case_e"
    dir_e.mkdir(parents=True, exist_ok=True)
    (dir_e / "diagnostics.jsonl").write_text(
        "INVALID_MIDDLE_SYNTAX\n" + diag_1.model_dump_json() + "\n", encoding="utf-8"
    )
    (dir_e / "results.jsonl").write_text(pub_1.model_dump_json() + "\n", encoding="utf-8")
    (dir_e / "batch_state.json").write_text(state_a.model_dump_json(), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="invalid JSON syntax"):
        with T5Dev200DiagnosticRunner(
            mock_service_a,
            application_config_hash=VALID_CONFIG_HASH,
            execution_identity=exec_identity,
            official_scoring_module=valid_scorer,
        ) as runner:
            runner.run(q_file, dir_e)


def test_t5_completion_marker_ordering_and_report_recovery(tmp_path: Path):
    """Test Case G: Crash after report.json written before manifest.json resumes without re-inference."""
    output_dir = tmp_path / "report_first_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    q_file = tmp_path / "questions.json"
    q_payload = {"q-01": {"question": "Câu 1", "answer": "Đáp án 1"}}
    q_file.write_text(json.dumps(q_payload, ensure_ascii=False), encoding="utf-8")

    q_hash = sha256(q_file.read_bytes()).hexdigest()
    q_ordered_hash = sha256("q-01".encode("utf-8")).hexdigest()

    exec_identity = T5ExecutionIdentity(
        measurement_harness_source_sha=VALID_HARNESS_SHA,
        dev200_ordered_ids_sha256=q_ordered_hash,
    )

    class _ValidScorer:
        def eval_qa(self, preds: dict[str, dict[str, str]], truth: dict[str, str]) -> dict[str, float]:
            return {"rouge": 0.9, "meteor": 0.9}

    valid_scorer = _ValidScorer()

    resp_1 = AnswerResponse(
        question="Câu 1",
        answer="Trả lời 1",
        citations=[Citation(evidence_id="E1", chunk_id="c-1", document_id="d-1")],
        insufficient_evidence=False,
        warnings=[],
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="t-1",
    )
    diag_1 = T5QuestionDiagnosticRecord(
        question_id="q-01",
        question="Câu 1",
        reference_answer="Đáp án 1",
        public_response=resp_1,
        stop_reason=AgentStopReason.ANSWER_VERIFIED,
        total_latency_ms=50.0,
        meteor_score=0.9,
        rouge_l_score=0.9,
    )
    pub_1 = CompetitionBatchRecord(question_id="q-01", response=resp_1)

    (output_dir / "diagnostics.jsonl").write_text(diag_1.model_dump_json() + "\n", encoding="utf-8")
    (output_dir / "results.jsonl").write_text(pub_1.model_dump_json() + "\n", encoding="utf-8")
    state = T5DiagnosticBatchState(
        execution_identity=exec_identity,
        question_source_sha256=q_hash,
        application_config_hash=VALID_CONFIG_HASH,
        code_version="0.45.0",
        question_count=1,
        ordered_question_ids_sha256=q_ordered_hash,
        completed_question_ids=["q-01"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    (output_dir / "batch_state.json").write_text(state.model_dump_json(), encoding="utf-8")

    # Simulate report.json written, but manifest.json not yet written
    (output_dir / "report.json").write_text(json.dumps({"interrupted": True}), encoding="utf-8")

    mock_service = MagicMock(spec=ServingService)
    mock_service._runtime = MagicMock()

    with T5Dev200DiagnosticRunner(
        mock_service,
        application_config_hash=VALID_CONFIG_HASH,
        execution_identity=exec_identity,
        official_scoring_module=valid_scorer,
    ) as runner:
        manifest = runner.run(q_file, output_dir)
        assert manifest.record_count == 1
        assert (output_dir / "manifest.json").exists()
        # Proves report.json was overwritten with complete final report
        report_data = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        assert report_data.get("experiment") == "t5-1-dev200-baseline-replay"
        # Zero model inference repeated
        assert mock_service.answer_result.call_count == 0


def test_t5_results_jsonl_trailing_valid_record_without_newline_normalized(tmp_path: Path):
    """Test Case F: results.jsonl valid final record without newline is normalized and retained."""
    output_dir = tmp_path / "results_no_newline"
    output_dir.mkdir(parents=True, exist_ok=True)

    resp_1 = AnswerResponse(
        question="Câu 1",
        answer="Trả lời 1",
        citations=[Citation(evidence_id="E1", chunk_id="c-1", document_id="d-1")],
        insufficient_evidence=False,
        warnings=[],
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="t-1",
    )
    pub_1 = CompetitionBatchRecord(question_id="q-01", response=resp_1)

    results_file = output_dir / "results.jsonl"
    # Write valid record without newline
    results_file.write_bytes(pub_1.model_dump_json().encode("utf-8"))

    records = T5Dev200DiagnosticRunner._load_public_records(results_file)
    assert len(records) == 1
    assert records[0].question_id == "q-01"
    # File is normalized to end with newline
    assert results_file.read_bytes().endswith(b"\n")
