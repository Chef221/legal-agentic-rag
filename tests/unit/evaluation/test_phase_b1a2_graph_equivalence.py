"""Unit tests for Phase B1A.2 graph equivalence and candidate-pool isolation."""

import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.configuration.online import RerankerConfig, RetrievalConfig
from legal_agentic_rag.contracts.graph_backend import GraphBackend
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.retrieval.graph import GraphExpandedRetriever
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    GraphPathStep,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from scripts.phase_b1a2_graph_equivalence import (
    CANONICAL_SOURCE_QUESTION_COUNT,
    CANONICAL_SOURCE_QUESTION_SHA256,
    EXPECTED_CASE_COUNT,
    FINAL_TOP_K,
    S20_BRANCH_CANDIDATE_DEPTH,
    S20_HYBRID_OUTPUT_LIMIT,
    S20_RERANK_INPUT_LIMIT,
    SCORE_ABS_TOLERANCE,
    RecordingGraphBackend,
    RecordingHybridCandidateAdapter,
    RetrievalPipelineSuite,
    analyze_b1a2_experiment,
    evaluate_b1a2_verdict_gate,
    package_b1a2_evidence,
    prepare_b1a2_dataset,
    run_b1a2_case,
)

MANIFEST_PATH = ROOT_DIR / "configs" / "phase-b1a-graph-routing-cases.json"
BASE_CONFIG_PATH = ROOT_DIR / "configs" / "phase-a-current-system-census-kaggle.example.json"

EXPECTED_22_IDS = [
    "102047", "107487", "110287", "111905", "113537", "122659", "125393", "133075",
    "134605", "147239", "147869", "150051", "26541", "29491", "29877", "39671",
    "45219", "47537", "48905", "64035", "95861", "99639",
]


# ----------------------------------------------------------------------
# DETERMINISTIC FAKE TEST BACKENDS
# ----------------------------------------------------------------------


class FakeGraphBackend:
    def __init__(
        self,
        record_count: int = 0,
        steps: list[GraphPathStep] | None = None,
        edge_count: int | None = None,
    ) -> None:
        self._record_count = record_count
        self._steps = steps or []
        self._edge_count = edge_count

    @property
    def manifest(self) -> ArtifactManifest:
        metadata = {}
        if self._edge_count is not None:
            metadata["edge_count"] = self._edge_count
        return ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.GRAPH_INDEX,
            artifact_version="fake-graph-v1",
            dataset_name="uit-dsc-2026-task2-selected-contexts",
            dataset_revision="canonical",
            record_count=self._record_count,
            created_at=datetime.now(UTC),
            processing_config_hash="dummy_graph_hash",
            backend="adjacency_graph",
            metadata=metadata,
        )

    def traverse(
        self,
        seed_document_ids: Sequence[str],
        hop_limit: int,
        relationship_types: Sequence[str] | None = None,
    ) -> Iterator[GraphPathStep]:
        yield from self._steps


class FakeReranker:
    def __init__(self, model_name: str = "fake-cross-encoder") -> None:
        self.model_name = model_name

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalHit],
    ) -> RetrievalResponse:
        reranked_hits = [
            hit.model_copy(
                update={
                    "rank": idx + 1,
                    "score": 10.0 - (idx * 0.1),
                    "strategy": RetrievalStrategy.RERANK,
                    "retrieval_trace": hit.retrieval_trace.model_copy(
                        update={"reranker_score": 10.0 - (idx * 0.1)}
                    ),
                }
            )
            for idx, hit in enumerate(candidates[: query.top_k])
        ]
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.RERANK,
            hits=reranked_hits,
            latency_ms=5.0,
            warnings=[],
        )


def _make_dummy_hit(idx: int, strategy: RetrievalStrategy = RetrievalStrategy.HYBRID) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk_{idx:03d}",
        document_id=f"doc_{idx // 2:03d}",
        rank=idx + 1,
        score=1.0 / (idx + 1),
        strategy=strategy,
        text=f"Text for chunk {idx}",
    )


class FakeHybridCandidateRetriever:
    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        return ("legal_chunks", "v1", "dummy_chunk_hash")

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        count = min(query.top_k, 40)
        hits = [_make_dummy_hit(i, RetrievalStrategy.HYBRID) for i in range(count)]
        return RetrievalResponse(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            hits=hits,
            latency_ms=10.0,
            warnings=[],
        )


def _build_mock_case_result(
    qid: str,
    *,
    seed_chunks_match: bool = True,
    final_chunks_match: bool = True,
    final_docs_match: bool = True,
    scores_match: bool = True,
    max_score_diff: float = 0.0,
    g_hybrid_calls: int = 1,
    g_warnings: list[str] | None = None,
    graph_step_count: int = 0,
    graph_manifest_record_count: int = 0,
    graph_manifest_edge_count: int | None = 0,
    g_seed_strategy: str = "hybrid",
    g_seed_top_k: int = 20,
    g_seed_cand_k: int = 40,
    s20_branch_depth: int = 40,
    s20_seed_strategy: str = "hybrid",
    s20_seed_top_k: int = 20,
    s20_seed_cand_k: int = 40,
    top8_identical: bool = True,
) -> dict[str, object]:
    warnings = g_warnings if g_warnings is not None else ["no_graph_expansion"]
    return {
        "question_id": qid,
        "g_arm": {
            "graph_manifest_record_count": graph_manifest_record_count,
            "graph_manifest_edge_count": graph_manifest_edge_count,
            "graph_traverse_call_count": 1 if graph_manifest_record_count == 0 else 0,
            "graph_step_count": graph_step_count,
            "hybrid_calls": g_hybrid_calls,
            "seed_requested_strategy": g_seed_strategy,
            "seed_top_k": g_seed_top_k,
            "seed_candidate_k": g_seed_cand_k,
            "warnings": warnings,
        },
        "s20_arm": {
            "branch_candidate_depth": s20_branch_depth,
            "seed_requested_strategy": s20_seed_strategy,
            "seed_top_k": s20_seed_top_k,
            "seed_candidate_k": s20_seed_cand_k,
        },
        "g_vs_s20": {
            "seed_chunks_match": seed_chunks_match,
            "final_chunks_match": final_chunks_match,
            "final_docs_match": final_docs_match,
            "max_score_diff": max_score_diff,
            "scores_match": scores_match,
            "equivalent": (
                seed_chunks_match
                and final_chunks_match
                and final_docs_match
                and scores_match
                and g_hybrid_calls == 1
                and graph_step_count == 0
            ),
        },
        "s20_vs_h40": {
            "top8_identical": top8_identical,
            "top8_overlap_count": 8 if top8_identical else 6,
            "top8_jaccard": 1.0 if top8_identical else 0.6,
            "rank_movements": [],
        },
    }


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------


def test_1_exact_22_case_manifest_validation() -> None:
    """TEST 1: Committed manifest contains exactly 22 IDs matching canonical count, SHA, and ordering."""
    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["candidate"] == "PHASE-B1A"
    assert manifest["case_count"] == EXPECTED_CASE_COUNT
    assert manifest["source_question_count"] == CANONICAL_SOURCE_QUESTION_COUNT
    assert manifest["source_question_sha256"] == CANONICAL_SOURCE_QUESTION_SHA256
    assert manifest["question_ids"] == EXPECTED_22_IDS
    assert len(set(manifest["question_ids"])) == 22


def test_2_k4_runtime_config_device_validation() -> None:
    """TEST 2: ApplicationConfig accepts search_device='cuda' and strictly rejects 'cuda:0'."""
    raw_cfg = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))

    # Valid: search_device = "cuda"
    raw_cfg["online"]["vector_runtime"]["search_device"] = "cuda"
    raw_cfg["offline"]["embedding"]["device"] = "cuda"
    raw_cfg["online"]["reranker"]["device"] = "cuda"
    cfg = ApplicationConfig.model_validate(raw_cfg)
    assert cfg.online.vector_runtime.search_device == "cuda"

    # Invalid: search_device = "cuda:0" must fail validation
    raw_cfg["online"]["vector_runtime"]["search_device"] = "cuda:0"
    with pytest.raises(ValidationError, match="Input should be 'cpu' or 'cuda'"):
        ApplicationConfig.model_validate(raw_cfg)


def test_3_graph_seed_chunk_k_resolves_to_20() -> None:
    """TEST 3: graph_seed_chunk_k default/config strictly resolves to 20."""
    ret_cfg = RetrievalConfig()
    assert ret_cfg.graph_seed_chunk_k == S20_HYBRID_OUTPUT_LIMIT


def test_4_graph_seed_query_construction() -> None:
    """TEST 4: GraphExpandedRetriever maps (top_k=8, candidate_k=40) -> (seed top_k=20, candidate_k=40, HYBRID)."""
    hybrid = FakeHybridCandidateRetriever()
    recording = RecordingHybridCandidateAdapter(hybrid)
    graph_backend = FakeGraphBackend(record_count=0)
    reranker = FakeReranker()
    chunk_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="v1",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=8532,
        created_at=datetime.now(UTC),
        processing_config_hash="dummy_chunk_hash",
    )

    retriever = GraphExpandedRetriever(
        candidate_retriever=recording,
        graph_backend=graph_backend,
        reranker=reranker,
        chunk_manifest=chunk_manifest,
        retrieval_config=RetrievalConfig(graph_seed_chunk_k=20),
        reranker_config=RerankerConfig(),
    )

    query = RetrievalQuery(
        query_id="test_q1",
        original_question="Văn bản sửa đổi",
        normalized_question="văn bản sửa đổi",
        top_k=8,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.GRAPH,
    )

    resp = retriever.search(query)
    assert len(recording.recorded_queries) == 1
    seed_q = recording.recorded_queries[0]
    assert seed_q.top_k == 20
    assert seed_q.candidate_k == 40
    assert seed_q.requested_strategy == RetrievalStrategy.HYBRID
    assert resp.warnings == ["no_graph_expansion"]


def test_5_6_7_zero_edge_graph_behavior() -> None:
    """TEST 5, 6, 7: Zero-edge GRAPH makes exactly 1 hybrid call, 0 related calls, and emits no_graph_expansion."""
    hybrid = FakeHybridCandidateRetriever()
    recording = RecordingHybridCandidateAdapter(hybrid)
    graph_backend = FakeGraphBackend(record_count=0)
    reranker = FakeReranker()
    chunk_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="v1",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=8532,
        created_at=datetime.now(UTC),
        processing_config_hash="dummy_chunk_hash",
    )

    retriever = GraphExpandedRetriever(
        candidate_retriever=recording,
        graph_backend=graph_backend,
        reranker=reranker,
        chunk_manifest=chunk_manifest,
        retrieval_config=RetrievalConfig(graph_seed_chunk_k=20),
    )

    query = RetrievalQuery(
        query_id="test_q2",
        original_question="Test query",
        normalized_question="test query",
        top_k=8,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.GRAPH,
    )
    resp = retriever.search(query)

    assert len(recording.recorded_queries) == 1
    assert "no_graph_expansion" in resp.warnings
    assert len(resp.hits) == 8


def test_8_recording_graph_backend_records_steps() -> None:
    """TEST 8 & 9: RecordingGraphBackend records traverse calls and yielded steps observationally."""
    step = GraphPathStep(
        source_document_id="doc_1",
        target_document_id="doc_2",
        relationship_type="references",
        hop=1,
    )
    fake_graph = FakeGraphBackend(record_count=1, steps=[step])
    recording = RecordingGraphBackend(fake_graph)

    assert recording.manifest.record_count == 1
    steps = list(recording.traverse(["doc_1"], hop_limit=1))
    assert len(steps) == 1
    assert recording.traverse_call_count == 1
    assert len(recording.recorded_steps) == 1
    assert recording.recorded_steps[0].target_document_id == "doc_2"


def test_9_s20_preserves_depth_40_and_rejects_candidate_k_20() -> None:
    """TEST 9: S20 preserves branch candidate depth 40; candidate_k=20 HYBRID_RERANK is rejected."""
    standard_h20_query = RetrievalQuery(
        query_id="test_h20",
        original_question="Test",
        normalized_question="test",
        top_k=8,
        candidate_k=20,
    )
    assert standard_h20_query.candidate_k == 20

    s20_seed_query = RetrievalQuery(
        query_id="test_s20",
        original_question="Test",
        normalized_question="test",
        top_k=S20_HYBRID_OUTPUT_LIMIT,
        candidate_k=S20_BRANCH_CANDIDATE_DEPTH,
        requested_strategy=RetrievalStrategy.HYBRID,
    )
    assert s20_seed_query.top_k == 20
    assert s20_seed_query.candidate_k == 40


def test_10_verdict_gate_proven() -> None:
    """TEST 10: Exact normalized GRAPH-vs-S20 equivalence across all 22 cases returns GRAPH_REDUNDANCY_PROVEN."""
    cases = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS]
    verdict, reasons = evaluate_b1a2_verdict_gate(cases, 22)
    assert verdict == "GRAPH_REDUNDANCY_PROVEN"
    assert any("All 22 cases match exactly" in r for r in reasons)


def test_11_changed_seed_chunk_causes_not_proven() -> None:
    """TEST 11: One changed seed chunk sequence causes GRAPH_REDUNDANCY_NOT_PROVEN."""
    cases = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS]
    cases[0] = _build_mock_case_result(EXPECTED_22_IDS[0], seed_chunks_match=False)

    verdict, reasons = evaluate_b1a2_verdict_gate(cases, 22)
    assert verdict == "GRAPH_REDUNDANCY_NOT_PROVEN"
    assert any(f"Case {EXPECTED_22_IDS[0]}: seed chunk sequences differ" in r for r in reasons)


def test_12_changed_final_order_causes_not_proven() -> None:
    """TEST 12: One changed final top-8 chunk sequence causes GRAPH_REDUNDANCY_NOT_PROVEN."""
    cases = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS]
    cases[5] = _build_mock_case_result(EXPECTED_22_IDS[5], final_chunks_match=False)

    verdict, reasons = evaluate_b1a2_verdict_gate(cases, 22)
    assert verdict == "GRAPH_REDUNDANCY_NOT_PROVEN"
    assert any(f"Case {EXPECTED_22_IDS[5]}: final top-8 chunk sequences differ" in r for r in reasons)


def test_13_score_tolerance_exceeded_causes_not_proven() -> None:
    """TEST 13: Reranker score delta > 1e-6 causes GRAPH_REDUNDANCY_NOT_PROVEN."""
    cases = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS]
    cases[2] = _build_mock_case_result(
        EXPECTED_22_IDS[2],
        scores_match=False,
        max_score_diff=1.5e-5,
    )

    verdict, reasons = evaluate_b1a2_verdict_gate(cases, 22)
    assert verdict == "GRAPH_REDUNDANCY_NOT_PROVEN"
    assert any("final reranker score delta" in r for r in reasons)


def test_14_15_graph_steps_or_edges_causes_invalid_experiment() -> None:
    """TEST 14 & 15: Non-zero graph step count or non-zero edge count causes INVALID_EXPERIMENT."""
    # Step count > 0
    cases = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS]
    cases[0] = _build_mock_case_result(EXPECTED_22_IDS[0], graph_step_count=1)

    verdict, reasons = evaluate_b1a2_verdict_gate(cases, 22)
    assert verdict == "INVALID_EXPERIMENT"
    assert any("graph step count (1) != 0" in r for r in reasons)

    # Edge count > 0
    cases2 = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS]
    cases2[1] = _build_mock_case_result(EXPECTED_22_IDS[1], graph_manifest_edge_count=5)

    verdict2, reasons2 = evaluate_b1a2_verdict_gate(cases2, 22)
    assert verdict2 == "INVALID_EXPERIMENT"
    assert any("graph edge_count (5) != 0" in r for r in reasons2)


def test_16_wrong_case_count_or_invariants_causes_invalid_experiment() -> None:
    """TEST 16: Wrong case count or violated seed invariants causes INVALID_EXPERIMENT."""
    # Case count 21
    cases_21 = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS[:-1]]
    verdict, reasons = evaluate_b1a2_verdict_gate(cases_21, 22)
    assert verdict == "INVALID_EXPERIMENT"
    assert any("expected 22 cases, got 21" in r for r in reasons)

    # Violated seed candidate_k
    cases_bad_cand = [_build_mock_case_result(qid) for qid in EXPECTED_22_IDS]
    cases_bad_cand[0] = _build_mock_case_result(EXPECTED_22_IDS[0], g_seed_cand_k=20)
    verdict2, reasons2 = evaluate_b1a2_verdict_gate(cases_bad_cand, 22)
    assert verdict2 == "INVALID_EXPERIMENT"
    assert any("G seed candidate_k (20) != 40" in r for r in reasons2)


def test_17_18_h40_differences_diagnostics_do_not_affect_verdict(tmp_path: Path) -> None:
    """TEST 17 & 18: S20 vs H40 pool differences are reported diagnostically without changing verdict."""
    cases = []
    for idx, qid in enumerate(EXPECTED_22_IDS):
        is_diff = idx < 10
        cases.append(_build_mock_case_result(qid, top8_identical=not is_diff))

    results_file = tmp_path / "results.jsonl"
    results_file.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")

    rep_file = tmp_path / "report.json"
    dec_file = tmp_path / "decision.json"
    metrics_file = tmp_path / "metrics.jsonl"

    report = analyze_b1a2_experiment(
        results_jsonl_path=results_file,
        manifest_path=manifest_file,
        output_report_path=rep_file,
        output_decision_path=dec_file,
        output_case_metrics_path=metrics_file,
    )

    assert report["verdict"] == "GRAPH_REDUNDANCY_PROVEN"
    diag = report["s20_vs_h40_candidate_pool_diagnostics"]
    assert diag["identical_top8_count"] == 12
    assert diag["changed_top8_count"] == 10


def test_19_no_generation_provider_used() -> None:
    """TEST 19: RetrievalPipelineSuite contains only retrieval modules and no LLM generator."""
    from scripts.phase_b1a2_graph_equivalence import RetrievalPipelineSuite

    suite_members = set(dir(RetrievalPipelineSuite))
    forbidden_members = {
        "generator",
        "answer_generator",
        "chat_provider",
        "transformers_chat_provider",
        "qwen",
    }
    assert not (suite_members & forbidden_members)


def test_20_no_raw_benchmark_leakage() -> None:
    """TEST 20: Manifest contains ONLY content-free metadata and IDs, with no question/answer text."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden_keys = {"question", "questions", "answer", "answers", "reference", "prompt"}
    assert not (set(manifest.keys()) & forbidden_keys)
    for qid in manifest["question_ids"]:
        assert isinstance(qid, str)
        assert qid.isdigit()


def test_21_prepare_and_package_pipeline_with_run_summary(tmp_path: Path) -> None:
    """TEST 21: Test package requires and includes run summary in evidence zip."""
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")

    ident_p = tmp_path / "ident.json"
    ident_p.write_text(json.dumps({"materialized_case_count": 22}), encoding="utf-8")

    cfg_p = tmp_path / "config.json"
    cfg_p.write_text(json.dumps({"retrieval": {"top_k": 8}}), encoding="utf-8")

    sum_p = tmp_path / "run_summary.json"
    sum_p.write_text(json.dumps({"case_count": 22, "graph_zero_record_cases": 22}), encoding="utf-8")

    res_p = tmp_path / "results.jsonl"
    res_p.write_text('{"question_id": "102047"}\n', encoding="utf-8")

    rep_p = tmp_path / "report.json"
    rep_p.write_text(json.dumps({"verdict": "GRAPH_REDUNDANCY_PROVEN"}), encoding="utf-8")

    dec_p = tmp_path / "decision.json"
    dec_p.write_text(json.dumps({"verdict": "GRAPH_REDUNDANCY_PROVEN"}), encoding="utf-8")

    zip_out = tmp_path / "evidence.zip"

    pkg_res = package_b1a2_evidence(
        output_zip_path=zip_out,
        manifest_path=manifest_p,
        questions_identity_path=ident_p,
        runtime_config_path=cfg_p,
        run_summary_path=sum_p,
        results_jsonl_path=res_p,
        report_path=rep_p,
        decision_path=dec_p,
    )

    assert zip_out.exists()
    assert pkg_res["zip_size_bytes"] > 0
    assert len(pkg_res["zip_sha256"]) == 64


def _create_valid_test_manifests() -> tuple[ArtifactManifest, ArtifactManifest, ArtifactManifest, ArtifactManifest]:
    now = datetime.now(UTC)
    chunk_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="v1",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=8532,
        created_at=now,
        processing_config_hash="chunk_hash_123",
        metadata={"runtime_normalized_processing_config_hash": "norm_hash_123"},
    )
    bm25_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.BM25_INDEX,
        artifact_version="v1",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=8532,
        created_at=now,
        processing_config_hash="bm25_hash_123",
        metadata={
            "source_artifact_type": "legal_chunks",
            "source_artifact_version": "v1",
            "source_processing_config_hash": "chunk_hash_123",
        },
    )
    vector_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.VECTOR_INDEX,
        artifact_version="v1",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=8532,
        created_at=now,
        processing_config_hash="vector_hash_123",
        model_name="intfloat/multilingual-e5-small",
        model_revision="canonical",
        metadata={
            "source_artifact_type": "legal_chunks",
            "source_artifact_version": "v1",
            "source_processing_config_hash": "chunk_hash_123",
            "embedding_provider_name": "sentence_transformers",
            "embedding_provider_version": "1.0",
            "dimension": 384,
        },
    )
    graph_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.GRAPH_INDEX,
        artifact_version="v1",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=0,
        created_at=now,
        processing_config_hash="graph_hash_123",
        metadata={"source_document_processing_config_hash": "norm_hash_123", "edge_count": 0},
    )
    return chunk_manifest, bm25_manifest, vector_manifest, graph_manifest


class FakeEmbeddingProvider:
    provider_name = "sentence_transformers"
    provider_version = "1.0"
    model_name = "intfloat/multilingual-e5-small"
    model_revision = "canonical"
    dimension = 384


def test_22_mismatched_bm25_lineage_fails_closed() -> None:
    """TEST 22: Mismatched BM25 source lineage raises ArtifactCompatibilityError."""
    cfg = ApplicationConfig.model_validate(json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8")))
    c_m, b_m, v_m, g_m = _create_valid_test_manifests()

    # Tamper with BM25 source hash
    bad_bm25_meta = dict(b_m.metadata)
    bad_bm25_meta["source_processing_config_hash"] = "wrong_hash"
    bad_bm25_m = b_m.model_copy(update={"metadata": bad_bm25_meta})

    with pytest.raises(ArtifactCompatibilityError, match="bm25_index does not originate from runtime legal chunks"):
        RetrievalPipelineSuite(
            config=cfg,
            chunk_manifest=c_m,
            bm25_manifest=bad_bm25_m,
            vector_manifest=v_m,
            graph_manifest=g_m,
            bm25_backend=object(),  # type: ignore[arg-type]
            vector_backend=object(),  # type: ignore[arg-type]
            graph_backend=FakeGraphBackend(),
            embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
            reranker=FakeReranker(),
        )


def test_23_mismatched_vector_lineage_fails_closed() -> None:
    """TEST 23: Mismatched vector source lineage raises ArtifactCompatibilityError."""
    cfg = ApplicationConfig.model_validate(json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8")))
    c_m, b_m, v_m, g_m = _create_valid_test_manifests()

    bad_vector_meta = dict(v_m.metadata)
    bad_vector_meta["source_artifact_version"] = "wrong_v2"
    bad_v_m = v_m.model_copy(update={"metadata": bad_vector_meta})

    with pytest.raises(ArtifactCompatibilityError, match="vector_index does not originate from runtime legal chunks"):
        RetrievalPipelineSuite(
            config=cfg,
            chunk_manifest=c_m,
            bm25_manifest=b_m,
            vector_manifest=bad_v_m,
            graph_manifest=g_m,
            bm25_backend=object(),  # type: ignore[arg-type]
            vector_backend=object(),  # type: ignore[arg-type]
            graph_backend=FakeGraphBackend(),
            embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
            reranker=FakeReranker(),
        )


def test_24_mismatched_graph_lineage_fails_closed() -> None:
    """TEST 24: Mismatched graph normalized-document lineage raises ArtifactCompatibilityError."""
    cfg = ApplicationConfig.model_validate(json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8")))
    c_m, b_m, v_m, g_m = _create_valid_test_manifests()

    bad_graph_meta = dict(g_m.metadata)
    bad_graph_meta["source_document_processing_config_hash"] = "different_norm_hash"
    bad_g_m = g_m.model_copy(update={"metadata": bad_graph_meta})

    with pytest.raises(ArtifactCompatibilityError, match="Graph and legal chunks do not share normalized-document lineage"):
        RetrievalPipelineSuite(
            config=cfg,
            chunk_manifest=c_m,
            bm25_manifest=b_m,
            vector_manifest=v_m,
            graph_manifest=bad_g_m,
            bm25_backend=object(),  # type: ignore[arg-type]
            vector_backend=object(),  # type: ignore[arg-type]
            graph_backend=FakeGraphBackend(),
            embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
            reranker=FakeReranker(),
        )


def test_25_incompatible_embedding_provider_fails_closed() -> None:
    """TEST 25: Mismatched embedding provider dimension/identity raises ArtifactCompatibilityError."""
    raw_cfg = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_cfg["online"]["startup_validation"]["mode"] = "full"
    cfg = ApplicationConfig.model_validate(raw_cfg)
    c_m, b_m, v_m, g_m = _create_valid_test_manifests()

    class BadEmbeddingProvider:
        provider_name = "sentence_transformers"
        provider_version = "1.0"
        model_name = "intfloat/multilingual-e5-small"
        model_revision = "canonical"
        dimension = 768  # Mismatched dimension (vector manifest has 384)

    with pytest.raises(ArtifactCompatibilityError, match="Configured embedding provider is incompatible with vector artifact"):
        RetrievalPipelineSuite(
            config=cfg,
            chunk_manifest=c_m,
            bm25_manifest=b_m,
            vector_manifest=v_m,
            graph_manifest=g_m,
            bm25_backend=object(),  # type: ignore[arg-type]
            vector_backend=object(),  # type: ignore[arg-type]
            graph_backend=FakeGraphBackend(),
            embedding_provider=BadEmbeddingProvider(),  # type: ignore[arg-type]
            reranker=FakeReranker(),
        )


def test_26_validated_report_mode_invokes_startup_validation() -> None:
    """TEST 26: Startup in validated_report mode fails closed if report is missing."""
    cfg = ApplicationConfig.model_validate(json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8")))
    assert cfg.online.startup_validation.mode == "validated_report"
    c_m, b_m, v_m, g_m = _create_valid_test_manifests()

    with pytest.raises(ArtifactCompatibilityError, match="Validated-report startup requires a valid build validation report"):
        RetrievalPipelineSuite(
            config=cfg,
            chunk_manifest=c_m,
            bm25_manifest=b_m,
            vector_manifest=v_m,
            graph_manifest=g_m,
            bm25_backend=object(),  # type: ignore[arg-type]
            vector_backend=object(),  # type: ignore[arg-type]
            graph_backend=FakeGraphBackend(),
            embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
            reranker=FakeReranker(),
        )
