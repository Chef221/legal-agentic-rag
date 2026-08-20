"""Unit tests for Phase B1B graphless equivalence verification tooling."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
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
from legal_agentic_rag.schemas.tools import ToolName
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
    compare_hit_lists,
    create_graphless_staging_root,
    load_and_verify_b1a2_baseline,
    materialize_cases_from_benchmark,
    normalize_question_text,
    package_b1b_evidence,
    run_b1b_verification_protocol,
    sha256_bytes,
    sha256_file,
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
) -> Path:
    results_dir = base_dir / "results"
    evidence_dir = base_dir / "evidence"
    results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # If custom results provided, write them, otherwise build standard
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
    (evidence_dir / "phase_b1a2_run_summary.json").write_text(
        json.dumps({"execution_git_commit": commit, "case_count": 22}), encoding="utf-8"
    )
    return base_dir


def test_01_b1a2_frozen_baseline_parsing_success(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir)

    # Patch canonical SHA to match our dummy generated file
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
    from legal_agentic_rag.serving.config_loader import load_application_config

    source_root = tmp_path / "artifacts"
    source_root.mkdir()
    for name in ("legal_chunks", "bm25", "vector"):
        (source_root / name).mkdir()

    # Create chunks payload
    (source_root / "legal_chunks" / "chunks.jsonl").write_bytes(b"")
    empty_sha = sha256_bytes(b"")

    # Create dummy manifests
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


def test_10_tool_surface_excludes_graph_search(tmp_path: Path) -> None:
    from legal_agentic_rag.tools.factory import build_fixed_tool_registry
    registry = build_fixed_tool_registry(
        retriever=MagicMock(),
        context_grader=MagicMock(),
        answer_generator=MagicMock(),
        citation_verifier=MagicMock(),
    )
    descriptor_names = [d.name.value for d in registry.descriptors()]
    assert "graph_search" not in descriptor_names
    assert "relationship_rerank_search" in descriptor_names


def test_11_route_plan_exact_first_second_third_tools() -> None:
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


def test_16_provenance_failure_returns_invalid_experiment(tmp_path: Path) -> None:
    dummy_cfg = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=tmp_path),
        online=OnlineConfig(),
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(dummy_cfg.model_dump(mode="json")), encoding="utf-8")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")
    questions_file = tmp_path / "dev.json"
    questions_file.write_text(json.dumps({}), encoding="utf-8")

    # Pass non-existent baseline dir
    report, decision, verdict = run_b1b_verification_protocol(
        config_path=config_file,
        manifest_path=manifest_file,
        questions_path=questions_file,
        baseline_dir_or_zip=tmp_path / "missing_baseline",
        output_dir=tmp_path / "output",
    )
    assert verdict == "INVALID_EXPERIMENT"
    assert decision["verdict"] == "INVALID_EXPERIMENT"
    assert decision["b1b_verified"] is False


def test_17_evidence_package_inventory_and_zip(tmp_path: Path) -> None:
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


def test_18_runtime_config_explicitly_resolves_relationship_rerank_fusion_k() -> None:
    from legal_agentic_rag.configuration.online import RetrievalConfig
    cfg = RetrievalConfig()
    assert cfg.relationship_rerank_fusion_k == 20

    cfg_custom = RetrievalConfig(relationship_rerank_fusion_k=15)
    assert cfg_custom.relationship_rerank_fusion_k == 15
