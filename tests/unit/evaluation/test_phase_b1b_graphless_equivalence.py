"""Unit tests for Phase B1B graphless equivalence verification tooling."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import zipfile

import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from legal_agentic_rag.configuration import (
    ApplicationConfig,
    ArtifactConfig,
    OnlineConfig,
)
from legal_agentic_rag.configuration.online import (
    AgentConfig,
    EvidenceSelectionConfig,
    GenerationConfig,
    QueryUnderstandingConfig,
    RerankerConfig,
    RetrievalConfig,
    SemanticVerificationConfig,
    StartupValidationConfig,
)
from legal_agentic_rag.agent.router import (
    DeterministicStrategyRouter,
    RetrievalRoute,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    QueryAnalysis,
    QueryIntent,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.schemas.tools import ToolDescriptor, ToolName
import scripts.phase_b1b_graphless_equivalence as b1b_script
from scripts.phase_b1b_graphless_equivalence import (
    CANONICAL_B1A2_EXECUTION_COMMIT,
    CANONICAL_B1A2_RESULTS_SHA256,
    CANONICAL_SOURCE_QUESTION_COUNT,
    CANONICAL_SOURCE_QUESTION_SHA256,
    EXPECTED_22_IDS,
    EXPECTED_CASE_COUNT,
    FINAL_TOP_K,
    S20_BRANCH_CANDIDATE_DEPTH,
    S20_HYBRID_OUTPUT_LIMIT,
    SCORE_ABS_TOLERANCE,
    CaseEquivalenceResult,
    CaseMaterialization,
    EvaluatedHit,
    FrozenB1A2Baseline,
    RecordingBranchRetriever,
    RecordingCandidateRetriever,
    compare_hit_lists,
    create_graphless_staging_root,
    load_and_verify_b1a2_baseline,
    materialize_cases_from_benchmark,
    normalize_question_text,
    package_b1b_evidence,
    run_b1b_verification_protocol,
    sha256_bytes,
    sha256_file,
    validate_graphless_staging_root,
)


def _make_eval_hit(
    chunk_id: str,
    doc_id: str = "doc1",
    score: float = 0.9,
    rank: int = 1,
) -> EvaluatedHit:
    return EvaluatedHit(
        rank=rank,
        chunk_id=chunk_id,
        document_id=doc_id,
        score=score,
        strategy=RetrievalStrategy.HYBRID_RERANK.value,
    )


def _build_dummy_b1a2_baseline(
    base_dir: Path,
    verdict: str = "GRAPH_REDUNDANCY_PROVEN",
    commit: str = CANONICAL_B1A2_EXECUTION_COMMIT,
    custom_results_bytes: bytes | None = None,
    omit_summary: bool = False,
    summary_override: dict[str, object] | None = None,
) -> Path:
    results_dir = base_dir / "results"
    evidence_dir = base_dir / "evidence"
    results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    if custom_results_bytes is not None:
        (results_dir / "phase_b1a2_retrieval_results.jsonl").write_bytes(custom_results_bytes)
    else:
        lines = []
        for idx, qid in enumerate(EXPECTED_22_IDS):
            lines.append(
                json.dumps(
                    {
                        "question_id": qid,
                        "s20_arm": {
                            "final_hits": [
                                {
                                    "rank": 1,
                                    "chunk_id": f"chunk-{qid}",
                                    "document_id": f"doc-{qid}",
                                    "score": 0.95,
                                    "strategy": "hybrid_rerank",
                                }
                            ]
                        },
                    }
                )
            )
        (results_dir / "phase_b1a2_retrieval_results.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    (results_dir / "phase_b1a2_decision_report.json").write_text(
        json.dumps({"verdict": verdict, "b1b_design_authorized": True}), encoding="utf-8"
    )

    if not omit_summary:
        res_sha = sha256_file(results_dir / "phase_b1a2_retrieval_results.jsonl")
        summary_payload: dict[str, object] = {
            "execution_git_commit": commit,
            "case_count": 22,
            "results_sha256": res_sha,
        }
        if summary_override is not None:
            summary_payload.update(summary_override)
        (evidence_dir / "phase_b1a2_run_summary.json").write_text(
            json.dumps(summary_payload), encoding="utf-8"
        )
    return base_dir


def test_01_b1a2_frozen_baseline_parsing_success(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir)

    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")
    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        baseline = load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)
        assert baseline.decision_verdict == "GRAPH_REDUNDANCY_PROVEN"
        assert baseline.execution_git_commit == CANONICAL_B1A2_EXECUTION_COMMIT
        assert baseline.case_count == 22
        assert len(baseline.expected_s20_hits) == 22
        assert baseline.expected_s20_hits["102047"][0].chunk_id == "chunk-102047"


def test_02_wrong_baseline_execution_commit_rejected(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir, commit="wrongcommit1234567890123456789012345678")
    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")
    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        with pytest.raises(DataValidationError, match="execution commit mismatch"):
            load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)


def test_03_wrong_b1a2_results_sha_rejected(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir)
    with pytest.raises(DataValidationError, match="results SHA256 mismatch"):
        load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)


def test_04_non_graph_redundancy_proven_baseline_rejected(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir, verdict="INCONCLUSIVE")
    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")
    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        with pytest.raises(DataValidationError, match="GRAPH_REDUNDANCY_PROVEN"):
            load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)


def test_05_exact_22_manifest_order_enforcement(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8"
    )
    questions_file = tmp_path / "dev.json"
    dummy_dev = {qid: {"question": f"Question {qid}"} for qid in EXPECTED_22_IDS}
    for i in range(len(dummy_dev), CANONICAL_SOURCE_QUESTION_COUNT):
        dummy_dev[f"extra_{i}"] = {"question": "extra"}
    questions_file.write_text(json.dumps(dummy_dev), encoding="utf-8")
    dev_sha = sha256_file(questions_file)

    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha):
        cases, m_sha, q_sha = materialize_cases_from_benchmark(manifest_file, questions_file)
        assert len(cases) == 22
        assert [c.question_id for c in cases] == EXPECTED_22_IDS
        assert m_sha == sha256_file(manifest_file)
        assert q_sha == dev_sha


def test_06_duplicate_missing_reordered_ids_rejected(tmp_path: Path) -> None:
    questions_file = tmp_path / "dev.json"
    questions_file.write_text(json.dumps({}), encoding="utf-8")

    # Missing ID (only 21 IDs)
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({"question_ids": EXPECTED_22_IDS[:-1]}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="must have exactly 22 IDs"):
        materialize_cases_from_benchmark(m1, questions_file)

    # Reordered IDs
    m2 = tmp_path / "m2.json"
    reordered = list(reversed(EXPECTED_22_IDS))
    m2.write_text(json.dumps({"question_ids": reordered}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="IDs/order mismatch"):
        materialize_cases_from_benchmark(m2, questions_file)

    # Duplicate IDs
    m3 = tmp_path / "m3.json"
    dupes = EXPECTED_22_IDS[:-1] + [EXPECTED_22_IDS[0]]
    m3.write_text(json.dumps({"question_ids": dupes}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicate IDs"):
        materialize_cases_from_benchmark(m3, questions_file)


def test_07_and_08_graphless_staging_root_excludes_graph_and_relationships(tmp_path: Path) -> None:
    source_root = tmp_path / "source_artifacts"
    source_root.mkdir()
    (source_root / "legal_chunks").mkdir()
    (source_root / "bm25").mkdir()
    (source_root / "vector").mkdir()
    (source_root / "graph").mkdir()
    (source_root / "relationships").mkdir()
    (source_root / "build_validation.json").write_text("{}", encoding="utf-8")

    staging_root = tmp_path / "staging"
    inventory = create_graphless_staging_root(source_root, staging_root)

    assert not (staging_root / "graph").exists()
    assert not (staging_root / "relationships").exists()
    assert (staging_root / "legal_chunks").exists()
    assert (staging_root / "bm25").exists()
    assert (staging_root / "vector").exists()
    assert (staging_root / "build_validation.json").exists()

    inventory_names = [item["name"] for item in inventory]
    assert "graph" not in inventory_names
    assert "relationships" not in inventory_names
    assert "legal_chunks" in inventory_names


def test_09_runtime_starts_from_graphless_staging_fixture(tmp_path: Path) -> None:
    from legal_agentic_rag.runtime.online import OnlineRuntimeFactory

    source_root = tmp_path / "artifacts"
    source_root.mkdir()
    for name in ("legal_chunks", "bm25", "vector"):
        (source_root / name).mkdir()

    (source_root / "legal_chunks" / "chunks.jsonl").write_bytes(b"")
    empty_sha = sha256_bytes(b"")

    for atype, dirname in (
        (ArtifactType.LEGAL_CHUNKS, "legal_chunks"),
        (ArtifactType.BM25_INDEX, "bm25"),
        (ArtifactType.VECTOR_INDEX, "vector"),
    ):
        model_name = None
        model_revision = None
        if atype == ArtifactType.LEGAL_CHUNKS:
            meta = {"payload_file": "chunks.jsonl", "payload_sha256": empty_sha}
        elif atype == ArtifactType.VECTOR_INDEX:
            meta = {
                "source_artifact_type": "legal_chunks",
                "source_artifact_version": "1.0",
                "source_processing_config_hash": "hash",
                "dimension": 384,
                "embedding_provider_name": "test_emb",
                "embedding_provider_version": "1.0",
            }
            model_name = "test_model"
            model_revision = "rev"
        else:
            meta = {
                "source_artifact_type": "legal_chunks",
                "source_artifact_version": "1.0",
                "source_processing_config_hash": "hash",
            }
        manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=atype,
            artifact_version="1.0",
            dataset_name="uit-dsc-2026-task2-selected-contexts",
            dataset_revision="canonical",
            record_count=10,
            created_at=datetime.now(UTC),
            processing_config_hash="hash",
            code_version="0.50.7",
            model_name=model_name,
            model_revision=model_revision,
            metadata=meta,
        )
        (source_root / dirname / "manifest.json").write_text(
            manifest.model_dump_json(), encoding="utf-8"
        )

    staging_root = tmp_path / "staging"
    create_graphless_staging_root(source_root, staging_root)

    config = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=staging_root),
        online=OnlineConfig(),
    )

    mock_emb = MagicMock()
    mock_emb.provider_name = "test_emb"
    mock_emb.provider_version = "1.0"
    mock_emb.model_name = "test_model"
    mock_emb.model_revision = "rev"
    mock_emb.dimension = 384

    mock_rerank = MagicMock()
    mock_rerank.model_name = "test_rerank"

    with patch("legal_agentic_rag.indexing.bm25.SQLiteFTS5BM25Backend.load"), \
         patch("legal_agentic_rag.indexing.vector.NumpyVectorBackend.load"), \
         patch("legal_agentic_rag.runtime.online.validate_startup_report"):
        runtime = OnlineRuntimeFactory(
            config,
            embedding_provider=mock_emb,
            reranker=mock_rerank,
        ).build()
        assert set(runtime.manifests.keys()) == {
            ArtifactType.LEGAL_CHUNKS.value,
            ArtifactType.BM25_INDEX.value,
            ArtifactType.VECTOR_INDEX.value,
        }
        assert "graph_index" not in runtime.manifests


def test_10_verification_never_references_runtime_tool_registry() -> None:
    source_lines = inspect.getsource(b1b_script)
    assert "runtime.tool_registry" not in source_lines
    assert "runtime.tool_descriptors()" in source_lines


def test_11_route_gate_checks_exact_strategy_and_tool_pairs() -> None:
    router = DeterministicStrategyRouter(
        AgentConfig(max_retry=2),
        QueryUnderstandingConfig(adaptive_routing_enabled=True),
    )
    analysis = QueryAnalysis(intent=QueryIntent.RELATIONSHIP)
    query = RetrievalQuery(
        query_id="q1",
        original_question="Văn bản A sửa đổi văn bản B?",
        normalized_question="văn bản a sửa đổi văn bản b",
        top_k=8,
        candidate_k=40,
        query_analysis=analysis,
    )
    tools = {
        ToolName.RELATIONSHIP_RERANK_SEARCH,
        ToolName.RERANK_SEARCH,
        ToolName.HYBRID_SEARCH,
    }
    routes = router.plan(query, tools)
    assert len(routes) == 3
    assert routes[0].tool_name == ToolName.RELATIONSHIP_RERANK_SEARCH
    assert routes[0].strategy == RetrievalStrategy.HYBRID_RERANK
    assert routes[1].tool_name == ToolName.RERANK_SEARCH
    assert routes[1].strategy == RetrievalStrategy.HYBRID_RERANK
    assert routes[2].tool_name == ToolName.HYBRID_SEARCH
    assert routes[2].strategy == RetrievalStrategy.HYBRID


def test_12_comparison_to_persisted_baseline_hits_passes() -> None:
    actual = [_make_eval_hit("c1", "d1", 0.9500001, 1), _make_eval_hit("c2", "d2", 0.85, 2)]
    expected = [_make_eval_hit("c1", "d1", 0.9500002, 1), _make_eval_hit("c2", "d2", 0.85, 2)]
    chunks_match, docs_match, scores_match, diffs, max_diff = compare_hit_lists(
        actual, expected, tolerance=1e-6
    )
    assert chunks_match is True
    assert docs_match is True
    assert scores_match is True
    assert max_diff < 1e-6


def test_13_changed_chunk_order_fails_comparison() -> None:
    actual = [_make_eval_hit("c2", "d1", 0.95, 1), _make_eval_hit("c1", "d2", 0.85, 2)]
    expected = [_make_eval_hit("c1", "d1", 0.95, 1), _make_eval_hit("c2", "d2", 0.85, 2)]
    chunks_match, docs_match, scores_match, _, _ = compare_hit_lists(actual, expected)
    assert chunks_match is False


def test_14_changed_document_order_fails_comparison() -> None:
    actual = [_make_eval_hit("c1", "d2", 0.95, 1)]
    expected = [_make_eval_hit("c1", "d1", 0.95, 1)]
    chunks_match, docs_match, scores_match, _, _ = compare_hit_lists(actual, expected)
    assert chunks_match is True
    assert docs_match is False


def test_15_score_delta_exceeding_tolerance_fails_comparison() -> None:
    actual = [_make_eval_hit("c1", "d1", 0.95, 1)]
    expected = [_make_eval_hit("c1", "d1", 0.95001, 1)]
    _, _, scores_match, diffs, max_diff = compare_hit_lists(actual, expected, tolerance=1e-6)
    assert scores_match is False
    assert max_diff > 1e-6


def test_16_observational_candidate_retriever_records_candidate_query_and_fused_count() -> None:
    mock_inner = MagicMock()
    mock_resp = RetrievalResponse(
        query=RetrievalQuery(
            query_id="q1",
            original_question="q",
            normalized_question="q",
            top_k=20,
            candidate_k=40,
            requested_strategy=RetrievalStrategy.HYBRID,
        ),
        strategy=RetrievalStrategy.HYBRID,
        hits=[
            RetrievalHit(
                chunk_id=f"c{i}",
                document_id=f"d{i}",
                rank=i + 1,
                score=1.0 / (i + 1),
                strategy=RetrievalStrategy.HYBRID,
                text="sample text",
            )
            for i in range(15)
        ],
        latency_ms=10.0,
    )
    mock_inner.search.return_value = mock_resp
    mock_inner.source_artifact_identity = ("legal_chunks", "1.0", "hash")

    recorder = RecordingCandidateRetriever(mock_inner)
    test_q = RetrievalQuery(
        query_id="q1",
        original_question="q",
        normalized_question="q",
        top_k=20,
        candidate_k=40,
        requested_strategy=RetrievalStrategy.HYBRID,
    )
    res = recorder.search(test_q)

    assert recorder.last_candidate_query == test_q
    assert recorder.last_candidate_query.top_k == 20
    assert recorder.last_candidate_query.candidate_k == 40
    assert recorder.last_candidate_query.requested_strategy == RetrievalStrategy.HYBRID
    assert recorder.last_candidate_response == mock_resp
    assert len(recorder.last_candidate_response.hits) == 15


def test_17_missing_mandatory_b1a2_run_summary_rejected(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2_no_summary"
    _build_dummy_b1a2_baseline(base_dir, omit_summary=True)
    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")
    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        with pytest.raises(DataValidationError, match="Missing mandatory B1A.2 run summary"):
            load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)


def test_18_missing_execution_git_commit_in_summary_rejected(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2_bad_summary"
    _build_dummy_b1a2_baseline(base_dir, summary_override={"case_count": 22})
    # Remove execution_git_commit from written summary
    summary_path = base_dir / "evidence" / "phase_b1a2_run_summary.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary_data["execution_git_commit"]
    summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")
    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        with pytest.raises(DataValidationError, match="missing mandatory 'execution_git_commit'"):
            load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)


def test_19_source_root_equals_staging_root_create_rejected(tmp_path: Path) -> None:
    same_dir = tmp_path / "same_root"
    same_dir.mkdir()
    with pytest.raises(ArtifactCompatibilityError, match="resolve to the same path"):
        create_graphless_staging_root(same_dir, same_dir)


def test_20_validation_of_existing_staging_root_produces_non_empty_inventory(tmp_path: Path) -> None:
    staging_dir = tmp_path / "valid_staging"
    staging_dir.mkdir()
    (staging_dir / "legal_chunks").mkdir()
    (staging_dir / "bm25").mkdir()
    (staging_dir / "vector").mkdir()

    inventory = validate_graphless_staging_root(staging_dir)
    assert len(inventory) == 3
    inv_names = {item["name"] for item in inventory}
    assert inv_names == {"legal_chunks", "bm25", "vector"}


def test_21_graph_or_relationships_in_staging_root_fails_validation(tmp_path: Path) -> None:
    staging_dir = tmp_path / "bad_staging"
    staging_dir.mkdir()
    (staging_dir / "legal_chunks").mkdir()
    (staging_dir / "bm25").mkdir()
    (staging_dir / "vector").mkdir()
    (staging_dir / "graph").mkdir()

    with pytest.raises(ArtifactCompatibilityError, match="illegally contains 'graph'"):
        validate_graphless_staging_root(staging_dir)

    (staging_dir / "graph").rmdir()
    (staging_dir / "relationships").mkdir()

    with pytest.raises(ArtifactCompatibilityError, match="illegally contains 'relationships'"):
        validate_graphless_staging_root(staging_dir)


def test_22_evidence_package_inventory_and_zip(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    (out_dir / "execution").mkdir(parents=True)
    (out_dir / "baseline").mkdir(parents=True)
    (out_dir / "configs").mkdir(parents=True)
    (out_dir / "results").mkdir(parents=True)

    (out_dir / "execution" / "b1b_execution_identity.json").write_text("{}", encoding="utf-8")
    (out_dir / "execution" / "graphless_root_inventory.json").write_text("[]", encoding="utf-8")
    (out_dir / "baseline" / "b1a2_baseline_identity.json").write_text("{}", encoding="utf-8")
    (out_dir / "configs" / "runtime_config.json").write_text("{}", encoding="utf-8")
    (out_dir / "configs" / "phase-b1a-graph-routing-cases.json").write_text("{}", encoding="utf-8")
    (out_dir / "results" / "phase_b1b_retrieval_results.jsonl").write_text("", encoding="utf-8")
    (out_dir / "results" / "phase_b1b_case_metrics.jsonl").write_text("", encoding="utf-8")
    (out_dir / "results" / "phase_b1b_equivalence_report.json").write_text("{}", encoding="utf-8")
    (out_dir / "results" / "phase_b1b_decision_report.json").write_text("{}", encoding="utf-8")

    zip_path = out_dir / "phase-b1b-graphless-equivalence-evidence.zip"
    zip_sha, zip_size = package_b1b_evidence(zip_path, out_dir)

    assert zip_path.is_file()
    assert len(zip_sha) == 64
    assert zip_size > 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "execution/b1b_execution_identity.json" in names
        assert "execution/graphless_root_inventory.json" in names
        assert "baseline/b1a2_baseline_identity.json" in names
        assert "configs/runtime_config.json" in names
        assert "configs/phase-b1a-graph-routing-cases.json" in names
        assert "results/phase_b1b_retrieval_results.jsonl" in names
        assert "results/phase_b1b_case_metrics.jsonl" in names
        assert "results/phase_b1b_equivalence_report.json" in names
        assert "results/phase_b1b_decision_report.json" in names


def test_23_end_to_end_mocked_protocol_execution_success(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    source_root.mkdir()
    for name in ("legal_chunks", "bm25", "vector"):
        (source_root / name).mkdir()
    (source_root / "legal_chunks" / "chunks.jsonl").write_bytes(b"")
    empty_sha = sha256_bytes(b"")

    for atype, dirname in (
        (ArtifactType.LEGAL_CHUNKS, "legal_chunks"),
        (ArtifactType.BM25_INDEX, "bm25"),
        (ArtifactType.VECTOR_INDEX, "vector"),
    ):
        meta: dict[str, object] = {}
        model_name = None
        model_revision = None
        if atype == ArtifactType.LEGAL_CHUNKS:
            meta = {"payload_file": "chunks.jsonl", "payload_sha256": empty_sha}
        elif atype == ArtifactType.VECTOR_INDEX:
            meta = {
                "source_artifact_type": "legal_chunks",
                "source_artifact_version": "1.0",
                "source_processing_config_hash": "hash",
                "dimension": 384,
                "embedding_provider_name": "test_emb",
                "embedding_provider_version": "1.0",
            }
            model_name = "test_model"
            model_revision = "rev"
        else:
            meta = {
                "source_artifact_type": "legal_chunks",
                "source_artifact_version": "1.0",
                "source_processing_config_hash": "hash",
            }
        manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=atype,
            artifact_version="1.0",
            dataset_name="uit-dsc-2026-task2-selected-contexts",
            dataset_revision="canonical",
            record_count=10,
            created_at=datetime.now(UTC),
            processing_config_hash="hash",
            code_version="0.50.7",
            model_name=model_name,
            model_revision=model_revision,
            metadata=meta,
        )
        (source_root / dirname / "manifest.json").write_text(
            manifest.model_dump_json(), encoding="utf-8"
        )

    b1a2_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(b1a2_dir)
    real_b1a2_sha = sha256_file(b1a2_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8"
    )
    questions_file = tmp_path / "dev.json"
    dummy_dev = {qid: {"question": f"Văn bản {qid} sửa đổi văn bản khác"} for qid in EXPECTED_22_IDS}
    for i in range(len(dummy_dev), CANONICAL_SOURCE_QUESTION_COUNT):
        dummy_dev[f"extra_{i}"] = {"question": "extra"}
    questions_file.write_text(json.dumps(dummy_dev), encoding="utf-8")
    dev_sha = sha256_file(questions_file)

    config = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=source_root),
        online=OnlineConfig(),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")

    out_dir = tmp_path / "output"

    mock_emb = MagicMock()
    mock_emb.provider_name = "test_emb"
    mock_emb.provider_version = "1.0"
    mock_emb.model_name = "test_model"
    mock_emb.model_revision = "rev"
    mock_emb.dimension = 384
    mock_emb.embed_query.return_value = [0.1] * 384

    mock_rerank = MagicMock()
    mock_rerank.model_name = "test_rerank"

    def _mock_rerank_candidates(q, cand_resp):
        hits = [
            RetrievalHit(
                chunk_id=f"chunk-{q.query_id}",
                document_id=f"doc-{q.query_id}",
                rank=1,
                score=0.95,
                strategy=RetrievalStrategy.HYBRID_RERANK,
                text="sample text",
            )
        ]
        return RetrievalResponse(
            query=q,
            strategy=RetrievalStrategy.HYBRID_RERANK,
            hits=hits,
            latency_ms=5.0,
        )

    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_B1A2_RESULTS_SHA256", real_b1a2_sha), \
         patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha), \
         patch("legal_agentic_rag.indexing.bm25.SQLiteFTS5BM25Backend.load") as mock_bm25_load, \
         patch("legal_agentic_rag.indexing.vector.NumpyVectorBackend.load") as mock_vec_load, \
         patch("scripts.phase_b1b_graphless_equivalence.SentenceTransformerEmbeddingProvider", return_value=mock_emb), \
         patch("scripts.phase_b1b_graphless_equivalence.CrossEncoderReranker", return_value=mock_rerank), \
         patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory.build") as mock_runtime_build, \
         patch("legal_agentic_rag.retrieval.rerank.RerankingRetriever.rerank_candidates", side_effect=_mock_rerank_candidates):

        # Setup mock runtime returned by factory
        mock_rt = MagicMock()
        mock_rt.manifests = {
            "legal_chunks": MagicMock(),
            "bm25_index": MagicMock(),
            "vector_index": MagicMock(),
        }
        mock_rt.tool_descriptors.return_value = [
            ToolDescriptor(
                name=ToolName.RELATIONSHIP_RERANK_SEARCH,
                description="desc",
                input_schema={},
                output_schema={},
                timeout_seconds=60.0,
            ),
            ToolDescriptor(
                name=ToolName.RERANK_SEARCH,
                description="desc",
                input_schema={},
                output_schema={},
                timeout_seconds=60.0,
            ),
            ToolDescriptor(
                name=ToolName.HYBRID_SEARCH,
                description="desc",
                input_schema={},
                output_schema={},
                timeout_seconds=60.0,
            ),
        ]
        mock_runtime_build.return_value = mock_rt

        # Mock backends for retrieval execution
        mock_b = MagicMock()
        mock_b.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        def _mock_bm25_search(q):
            return RetrievalResponse(
                query=q,
                strategy=RetrievalStrategy.BM25,
                hits=[
                    RetrievalHit(
                        chunk_id="c1", document_id="d1", rank=1, score=1.0, strategy=RetrievalStrategy.BM25, text="t"
                    )
                ],
                latency_ms=1.0,
            )
        mock_b.search.side_effect = _mock_bm25_search
        mock_bm25_load.return_value = mock_b

        mock_v = MagicMock()
        mock_v.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        mock_v.embedding_provider_name = "test_emb"
        mock_v.embedding_provider_version = "1.0"
        mock_v.model_name = "test_model"
        mock_v.model_revision = "rev"
        mock_v.dimension = 384
        def _mock_dense_search(q, q_vec=None):
            return RetrievalResponse(
                query=q,
                strategy=RetrievalStrategy.DENSE,
                hits=[
                    RetrievalHit(
                        chunk_id="c1", document_id="d1", rank=1, score=1.0, strategy=RetrievalStrategy.DENSE, text="t"
                    )
                ],
                latency_ms=1.0,
            )
        mock_v.search.side_effect = _mock_dense_search
        mock_vec_load.return_value = mock_v

        report, decision, verdict = run_b1b_verification_protocol(
            config_path=config_path,
            manifest_path=manifest_file,
            questions_path=questions_file,
            baseline_dir_or_zip=b1a2_dir,
            output_dir=out_dir,
        )

        assert verdict == "B1B_EQUIVALENCE_PASS", f"Reasons: {report.get('reasons')}"
        assert decision["verdict"] == "B1B_EQUIVALENCE_PASS"
        assert decision["b1b_verified"] is True
        assert report["aggregate_protocol_counts"]["branch_depth_40_passes"] == 22
        assert report["aggregate_protocol_counts"]["candidate_query_invariant_passes"] == 22
        assert report["aggregate_protocol_counts"]["fusion_limit_passes"] == 22
        assert report["aggregate_protocol_counts"]["final_topk_passes"] == 22
        assert report["aggregate_protocol_counts"]["route_plan_passes"] == 22


def test_24_retrieval_model_error_yields_invalid_experiment(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    source_root.mkdir()
    for name in ("legal_chunks", "bm25", "vector"):
        (source_root / name).mkdir()
    (source_root / "legal_chunks" / "chunks.jsonl").write_bytes(b"")
    empty_sha = sha256_bytes(b"")

    for atype, dirname in (
        (ArtifactType.LEGAL_CHUNKS, "legal_chunks"),
        (ArtifactType.BM25_INDEX, "bm25"),
        (ArtifactType.VECTOR_INDEX, "vector"),
    ):
        meta = {"payload_file": "chunks.jsonl", "payload_sha256": empty_sha} if atype == ArtifactType.LEGAL_CHUNKS else {
            "source_artifact_type": "legal_chunks", "source_artifact_version": "1.0", "source_processing_config_hash": "hash"
        }
        manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=atype,
            artifact_version="1.0",
            dataset_name="uit-dsc-2026-task2-selected-contexts",
            dataset_revision="canonical",
            record_count=10,
            created_at=datetime.now(UTC),
            processing_config_hash="hash",
            code_version="0.50.7",
            metadata=meta,
        )
        (source_root / dirname / "manifest.json").write_text(
            manifest.model_dump_json(), encoding="utf-8"
        )

    b1a2_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(b1a2_dir)
    real_b1a2_sha = sha256_file(b1a2_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")
    questions_file = tmp_path / "dev.json"
    dummy_dev = {qid: {"question": f"Văn bản {qid} sửa đổi"} for qid in EXPECTED_22_IDS}
    for i in range(len(dummy_dev), CANONICAL_SOURCE_QUESTION_COUNT):
        dummy_dev[f"extra_{i}"] = {"question": "extra"}
    questions_file.write_text(json.dumps(dummy_dev), encoding="utf-8")
    dev_sha = sha256_file(questions_file)

    config = ApplicationConfig(artifacts=ArtifactConfig(root_path=source_root), online=OnlineConfig())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")

    out_dir = tmp_path / "output"

    with patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_B1A2_RESULTS_SHA256", real_b1a2_sha), \
         patch("scripts.phase_b1b_graphless_equivalence.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha), \
         patch("legal_agentic_rag.indexing.bm25.SQLiteFTS5BM25Backend.load"), \
         patch("legal_agentic_rag.indexing.vector.NumpyVectorBackend.load"), \
         patch("scripts.phase_b1b_graphless_equivalence.SentenceTransformerEmbeddingProvider"), \
         patch("scripts.phase_b1b_graphless_equivalence.CrossEncoderReranker"), \
         patch("legal_agentic_rag.runtime.online.OnlineRuntimeFactory.build") as mock_runtime_build, \
         patch("legal_agentic_rag.tools.retrieval.RetrievalTool.invoke", side_effect=RuntimeError("CUDA out of memory")):

        mock_rt = MagicMock()
        mock_rt.manifests = {
            "legal_chunks": MagicMock(),
            "bm25_index": MagicMock(),
            "vector_index": MagicMock(),
        }
        mock_rt.tool_descriptors.return_value = [
            ToolDescriptor(
                name=ToolName.RELATIONSHIP_RERANK_SEARCH,
                description="desc",
                input_schema={},
                output_schema={},
                timeout_seconds=60.0,
            ),
            ToolDescriptor(
                name=ToolName.RERANK_SEARCH,
                description="desc",
                input_schema={},
                output_schema={},
                timeout_seconds=60.0,
            ),
            ToolDescriptor(
                name=ToolName.HYBRID_SEARCH,
                description="desc",
                input_schema={},
                output_schema={},
                timeout_seconds=60.0,
            ),
        ]
        mock_runtime_build.return_value = mock_rt

        report, decision, verdict = run_b1b_verification_protocol(
            config_path=config_path,
            manifest_path=manifest_file,
            questions_path=questions_file,
            baseline_dir_or_zip=b1a2_dir,
            output_dir=out_dir,
        )

        assert verdict == "INVALID_EXPERIMENT"
        assert decision["verdict"] == "INVALID_EXPERIMENT"
        assert decision["b1b_verified"] is False
        assert report["retrieval_model_error_count"] > 0
