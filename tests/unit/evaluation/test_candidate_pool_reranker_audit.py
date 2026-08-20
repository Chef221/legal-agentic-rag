"""Unit tests for Stage R1 S20 vs H40 Candidate-Pool / Reranker Mechanics Audit."""

from __future__ import annotations

from datetime import UTC, datetime
import json
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
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.retrieval.fixed import HybridRetriever
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    QueryAnalysis,
    QueryIntent,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)
from scripts.candidate_pool_reranker_audit import (
    CANONICAL_B1A2_EXECUTION_COMMIT,
    CANONICAL_B1A2_MEMBER_SHA256,
    CANONICAL_B1A2_RESULTS_SHA256,
    CANONICAL_B1A2_ZIP_SHA256,
    CANONICAL_SOURCE_QUESTION_COUNT,
    CANONICAL_SOURCE_QUESTION_SHA256,
    EXPECTED_22_IDS,
    EXPECTED_CASE_COUNT,
    FINAL_TOP_K,
    CandidatePoolAuditPipeline,
    EvaluatedHit,
    FrozenB1A2Baseline,
    RecordingBranchRetriever,
    compute_aggregate_audit_metrics,
    create_graphless_staging_root,
    get_fused_rank_bucket,
    load_and_verify_b1a2_baseline,
    package_audit_evidence,
    run_candidate_pool_audit_protocol,
    run_case_candidate_pool_audit,
    sha256_bytes,
    sha256_file,
    validate_graphless_staging_root,
)


def _build_dummy_b1a2_bundle(
    bundle_dir: Path,
    *,
    verdict: str = "GRAPH_REDUNDANCY_PROVEN",
    results_sha_override: str | None = None,
    execution_commit: str = CANONICAL_B1A2_EXECUTION_COMMIT,
    missing_summary_sha: bool = False,
    alter_s20_final_ids: list[str] | None = None,
    alter_h40_final_ids: list[str] | None = None,
    drop_question_id: str | None = None,
    corrupt_member: str | None = None,
    delete_member: str | None = None,
) -> dict[str, str]:
    """Helper to create a synthetic B1A.2 evidence directory matching all 8 canonical members."""
    configs_dir = bundle_dir / "configs"
    evidence_dir = bundle_dir / "evidence"
    results_dir = bundle_dir / "results"

    for d in (configs_dir, evidence_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. configs/phase-b1a-graph-routing-cases.json
    cases_file = configs_dir / "phase-b1a-graph-routing-cases.json"
    cases_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}, indent=2) + "\n", encoding="utf-8")

    # 2. evidence/materialized_questions_identity.json
    mat_file = evidence_dir / "materialized_questions_identity.json"
    mat_file.write_text(json.dumps({"count": 22, "mode": "canonical"}, indent=2) + "\n", encoding="utf-8")

    # 3. configs/runtime_config.json
    cfg_file = configs_dir / "runtime_config.json"
    cfg_file.write_text(json.dumps({"version": "b1a2_runtime"}, indent=2) + "\n", encoding="utf-8")

    # 4. results/phase_b1a2_retrieval_results.jsonl
    results_file = results_dir / "phase_b1a2_retrieval_results.jsonl"
    lines = []
    for idx, qid in enumerate(EXPECTED_22_IDS):
        if drop_question_id and qid == drop_question_id:
            continue

        is_identical = idx < 5

        s20_seed_hits = [
            {
                "rank": r,
                "chunk_id": f"chunk-{qid}-{r}" if not (not is_identical and r == 25) else f"tail-chunk-{qid}-25",
                "document_id": f"doc-{qid}-{r}" if not (not is_identical and r == 25) else f"tail-doc-{qid}-25",
                "score": round(1.0 / (r + 10), 8),
                "strategy": "hybrid",
            }
            for r in range(1, 21)
        ]

        if alter_s20_final_ids and qid in alter_s20_final_ids:
            s20_final_hits = [
                {
                    "rank": r,
                    "chunk_id": f"altered-{qid}-{r}",
                    "document_id": f"doc-{qid}-{r}",
                    "score": round(0.95 - (r * 0.01), 8),
                    "strategy": "hybrid_rerank",
                }
                for r in range(1, 9)
            ]
        else:
            s20_final_hits = [
                {
                    "rank": r,
                    "chunk_id": f"chunk-{qid}-{r}",
                    "document_id": f"doc-{qid}-{r}",
                    "score": round(0.95 - (r * 0.01), 8),
                    "strategy": "hybrid_rerank",
                }
                for r in range(1, 9)
            ]

        if is_identical:
            h40_final_hits = list(s20_final_hits)
        else:
            if alter_h40_final_ids and qid in alter_h40_final_ids:
                h40_final_hits = [
                    {
                        "rank": r,
                        "chunk_id": f"altered-h40-{qid}-{r}",
                        "document_id": f"doc-{qid}-{r}",
                        "score": round(0.95 - (r * 0.01), 8),
                        "strategy": "hybrid_rerank",
                    }
                    for r in range(1, 9)
                ]
            else:
                h40_final_hits = [
                    {
                        "rank": r,
                        "chunk_id": f"chunk-{qid}-{r}",
                        "document_id": f"doc-{qid}-{r}",
                        "score": round(0.95 - (r * 0.01), 8),
                        "strategy": "hybrid_rerank",
                    }
                    for r in range(1, 8)
                ] + [
                    {
                        "rank": 8,
                        "chunk_id": f"tail-chunk-{qid}-25",
                        "document_id": f"tail-doc-{qid}-25",
                        "score": 0.875,
                        "strategy": "hybrid_rerank",
                    }
                ]

        line_obj = {
            "question_id": qid,
            "s20_arm": {
                "seed_hits": s20_seed_hits,
                "final_hits": s20_final_hits,
            },
            "h40_arm": {
                "final_hits": h40_final_hits,
            },
            "s20_vs_h40": {
                "top8_identical": is_identical,
                "overlap_count": 8 if is_identical else 7,
                "jaccard": 1.0 if is_identical else 7.0 / 9.0,
            },
        }
        lines.append(json.dumps(line_obj))

    results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    actual_res_sha = sha256_file(results_file)

    # 5. evidence/phase_b1a2_run_summary.json
    summary_file = evidence_dir / "phase_b1a2_run_summary.json"
    summary_data = {
        "execution_git_commit": execution_commit,
        "results_sha256": None if missing_summary_sha else (results_sha_override or actual_res_sha),
    }
    summary_file.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")

    # 6. results/phase_b1a2_graph_equivalence_report.json
    eq_file = results_dir / "phase_b1a2_graph_equivalence_report.json"
    eq_file.write_text(json.dumps({"status": "equivalence_verified"}, indent=2) + "\n", encoding="utf-8")

    # 7. results/phase_b1a2_decision_report.json
    decision_file = results_dir / "phase_b1a2_decision_report.json"
    decision_file.write_text(
        json.dumps({
            "audit_id": "PHASE-B1A2-GRAPH-EQUIVALENCE",
            "verdict": verdict,
            "results_sha256": actual_res_sha,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    # 8. results/phase_b1a2_case_metrics.jsonl
    metrics_file = results_dir / "phase_b1a2_case_metrics.jsonl"
    metrics_file.write_text("\n".join([json.dumps({"qid": qid, "m": idx}) for idx, qid in enumerate(EXPECTED_22_IDS)]) + "\n", encoding="utf-8")

    member_shas = {
        rel: sha256_file(bundle_dir / rel)
        for rel in CANONICAL_B1A2_MEMBER_SHA256
        if (bundle_dir / rel).is_file()
    }

    # Handle corruptions or deletions
    if corrupt_member:
        target = bundle_dir / corrupt_member
        target.write_text("corrupted content\n", encoding="utf-8")

    if delete_member:
        target = bundle_dir / delete_member
        if target.exists():
            target.unlink()

    return member_shas


def _build_dummy_b1a2_zip_from_bundle(bundle_dir: Path, zip_path: Path) -> Path:
    """Pack an existing bundle directory into a zip archive."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in CANONICAL_B1A2_MEMBER_SHA256:
            p = bundle_dir / rel
            if p.is_file():
                z.write(p, arcname=rel)
    return zip_path


def _setup_mock_staging_root(root: Path) -> Path:
    """Create a minimal valid graphless staging directory."""
    root.mkdir(parents=True, exist_ok=True)
    chunks_dir = root / "legal_chunks"
    bm25_dir = root / "bm25"
    vec_dir = root / "vector"

    for d in (chunks_dir, bm25_dir, vec_dir):
        d.mkdir(parents=True, exist_ok=True)

    c_man = ArtifactManifest(
        schema_version="1.0.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        processing_config_hash="hash1",
        code_version="0.50.7",
        created_at=datetime.now(UTC),
        record_count=100,
        backend="sqlite",
    )
    (chunks_dir / "manifest.json").write_text(
        json.dumps(c_man.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    b_man = ArtifactManifest(
        schema_version="1.0.0",
        artifact_type=ArtifactType.BM25_INDEX,
        artifact_version="1.0",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        processing_config_hash="hash1",
        code_version="0.50.7",
        created_at=datetime.now(UTC),
        record_count=100,
        backend="sqlite_fts5",
        metadata={"source_artifact_type": "legal_chunks", "source_artifact_version": "1.0", "source_processing_config_hash": "hash1"},
    )
    (bm25_dir / "manifest.json").write_text(
        json.dumps(b_man.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    v_man = ArtifactManifest(
        schema_version="1.0.0",
        artifact_type=ArtifactType.VECTOR_INDEX,
        artifact_version="1.0",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        processing_config_hash="hash1",
        code_version="0.50.7",
        created_at=datetime.now(UTC),
        record_count=100,
        backend="numpy",
        model_name="test_model",
        model_revision="rev",
        metadata={
            "source_artifact_type": "legal_chunks",
            "source_artifact_version": "1.0",
            "source_processing_config_hash": "hash1",
            "dimension": 384,
            "embedding_provider_name": "sentence-transformers",
            "embedding_provider_version": "1.0",
        },
    )
    (vec_dir / "manifest.json").write_text(
        json.dumps(v_man.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    # Add dummy report
    report_file = root / "build_validation_report.json"
    report_file.write_text(
        json.dumps({
            "is_valid": True,
            "validation_mode": "report",
            "manifest_checksums": {
                str(chunks_dir / "manifest.json"): "hash",
                str(bm25_dir / "manifest.json"): "hash",
                str(vec_dir / "manifest.json"): "hash",
            },
        }),
        encoding="utf-8",
    )
    return root


# ======================================================================
# FIX 1 & 2 REGRESSION TESTS: ZIP AND EXTRACTED BUNDLE BASELINE MODES
# ======================================================================


def test_01_canonical_zip_mode_accepted(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)
    zip_path = tmp_path / "b1a2.zip"
    _build_dummy_b1a2_zip_from_bundle(bundle_dir, zip_path)
    real_zip_sha = sha256_file(zip_path)

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_ZIP_SHA256", real_zip_sha), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]):
        baseline = load_and_verify_b1a2_baseline(zip_path, EXPECTED_22_IDS)

    assert isinstance(baseline, FrozenB1A2Baseline)
    assert baseline.source_kind == "canonical_zip"
    assert baseline.observed_zip_sha256 == real_zip_sha
    assert baseline.canonical_zip_sha256_expected == real_zip_sha
    assert baseline.case_count == 22
    assert baseline.decision_verdict == "GRAPH_REDUNDANCY_PROVEN"


def test_02_canonical_extracted_bundle_mode_accepted(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]):
        baseline = load_and_verify_b1a2_baseline(bundle_dir, EXPECTED_22_IDS)

    assert isinstance(baseline, FrozenB1A2Baseline)
    assert baseline.source_kind == "canonical_extracted_bundle"
    assert baseline.observed_zip_sha256 is None
    assert baseline.canonical_zip_sha256_expected == CANONICAL_B1A2_ZIP_SHA256
    assert baseline.canonical_member_sha256 == shas
    assert baseline.case_count == 22
    assert baseline.decision_verdict == "GRAPH_REDUNDANCY_PROVEN"


def test_03_extracted_bundle_missing_one_member_rejected(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir, delete_member="evidence/materialized_questions_identity.json")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]):
        with pytest.raises(DataValidationError, match="missing required member"):
            load_and_verify_b1a2_baseline(bundle_dir, EXPECTED_22_IDS)


def test_04_extracted_bundle_corrupt_member_sha_rejected(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)
    # Corrupt one member
    (bundle_dir / "configs" / "runtime_config.json").write_text("corrupted\n", encoding="utf-8")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas):
        with pytest.raises(DataValidationError, match="SHA mismatch"):
            load_and_verify_b1a2_baseline(bundle_dir, EXPECTED_22_IDS)


def test_05_wrong_zip_sha_rejected(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _build_dummy_b1a2_bundle(bundle_dir)
    zip_path = tmp_path / "b1a2.zip"
    _build_dummy_b1a2_zip_from_bundle(bundle_dir, zip_path)

    with pytest.raises(DataValidationError, match="baseline ZIP SHA mismatch"):
        load_and_verify_b1a2_baseline(zip_path, EXPECTED_22_IDS)


def test_06_wrong_run_summary_results_sha_rejected(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir, results_sha_override="bad_sha")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]):
        with pytest.raises(DataValidationError, match="run summary results SHA mismatch"):
            load_and_verify_b1a2_baseline(bundle_dir, EXPECTED_22_IDS)


def test_07_wrong_execution_commit_rejected(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir, execution_commit="bad_commit")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]):
        with pytest.raises(DataValidationError, match="execution commit mismatch"):
            load_and_verify_b1a2_baseline(bundle_dir, EXPECTED_22_IDS)


def test_08_staging_root_passed_becomes_pipeline_runtime_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source_artifacts"
    _setup_mock_staging_root(source_root)
    staging_root = tmp_path / "custom_staging"
    _setup_mock_staging_root(staging_root)

    app_config = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=source_root),
        online=OnlineConfig(),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(app_config.model_dump(mode="json")), encoding="utf-8")

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")

    dummy_dev = {qid: {"question": f"Văn bản {qid}"} for qid in EXPECTED_22_IDS}
    for i in range(len(dummy_dev), CANONICAL_SOURCE_QUESTION_COUNT):
        dummy_dev[f"extra_{i}"] = {"question": "extra"}
    questions_file = tmp_path / "dev.json"
    questions_file.write_text(json.dumps(dummy_dev), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)
    dev_sha = sha256_file(questions_file)

    out_dir = tmp_path / "out"

    dummy_case_res = {
        "question_id": "1",
        "reproduction_gates": {
            "seed_prefix_match": True,
            "s20_chunks_match": True,
            "s20_docs_match": True,
            "s20_scores_match": True,
            "h40_chunks_match": True,
            "h40_docs_match": True,
            "h40_scores_match": True,
            "branch_depth_fidelity": True,
        },
        "tail_entrants": [],
        "derived_s20_final_hits": [],
        "derived_h40_final_hits": [],
        "s20_vs_h40_comparison": {
            "top8_identical": True,
            "tail_entrant_count": 0,
            "top8_overlap_count": 8,
            "top8_jaccard": 1.0,
        },
        "score_cutoff_margin_diagnostics": {
            "s20_top8_cutoff_score": 0.9,
            "h40_top8_cutoff_score": 0.9,
            "entrant_vs_displaced_margin": None,
        },
    }
    dummy_case_met = {
        "question_id": "1",
        "top8_identical": True,
        "tail_entrant_count": 0,
        "overlap_count": 8,
        "jaccard": 1.0,
    }

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha), \
         patch("scripts.candidate_pool_reranker_audit.run_case_candidate_pool_audit", return_value=(dummy_case_res, dummy_case_met, [])), \
         patch("scripts.candidate_pool_reranker_audit.CandidatePoolAuditPipeline") as mock_pipe_cls:

        mock_instance = MagicMock()
        mock_pipe_cls.return_value = mock_instance

        report, decision, verdict = run_candidate_pool_audit_protocol(
            config_path=config_path,
            manifest_path=manifest_file,
            questions_path=questions_file,
            baseline_evidence_path=bundle_dir,
            output_dir=out_dir,
            staging_root=staging_root,
        )

        call_config = mock_pipe_cls.call_args[0][0]
        assert call_config.artifacts.root_path == staging_root
        assert call_config.artifacts.root_path != source_root

        persisted_cfg = json.loads((out_dir / "configs" / "runtime_config.json").read_text(encoding="utf-8"))
        assert Path(persisted_cfg["artifacts"]["root_path"]) == staging_root


# ======================================================================
# BRANCH DEPTH & CASE MECHANICS TESTS
# ======================================================================


def test_09_recording_branch_retriever_captures_queries() -> None:
    inner_mock = MagicMock()
    inner_mock.source_artifact_identity = ("bm25", "1.0", "hash")
    inner_mock.search.return_value = RetrievalResponse(
        query=RetrievalQuery(query_id="1", original_question="q", normalized_question="q"),
        strategy=RetrievalStrategy.BM25,
        hits=[],
        latency_ms=1.0,
    )

    recorder = RecordingBranchRetriever(inner_mock)
    assert recorder.source_artifact_identity == ("bm25", "1.0", "hash")
    assert len(recorder.recorded_queries) == 0

    q = RetrievalQuery(
        query_id="1", original_question="q", normalized_question="q", top_k=40, candidate_k=40, requested_strategy=RetrievalStrategy.BM25
    )
    resp = recorder.search(q)

    assert len(recorder.recorded_queries) == 1
    assert recorder.recorded_queries[0] == q
    assert resp.strategy == RetrievalStrategy.BM25


def test_10_single_case_audit_observes_real_branch_depth_40(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]):
        baseline = load_and_verify_b1a2_baseline(bundle_dir, EXPECTED_22_IDS)

    qid = "102047"
    q_text = "Văn bản 102047 sửa đổi điều khoản nào?"

    mock_pipeline = MagicMock()
    mock_pipeline.query_understanding.enrich.return_value = RetrievalQuery(
        query_id=qid,
        original_question=q_text,
        normalized_question=q_text,
        top_k=8,
        candidate_k=40,
        query_analysis=QueryAnalysis(intent=QueryIntent.RELATIONSHIP),
    )

    bm25_inner = MagicMock()
    dense_inner = MagicMock()
    rec_bm25 = RecordingBranchRetriever(bm25_inner)
    rec_dense = RecordingBranchRetriever(dense_inner)
    mock_pipeline.recording_bm25 = rec_bm25
    mock_pipeline.recording_dense = rec_dense

    fused_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{qid}-{r}",
            document_id=f"doc-{qid}-{r}",
            rank=r,
            score=round(1.0 / (r + 10), 8),
            strategy=RetrievalStrategy.HYBRID,
            text="sample text",
        )
        for r in range(1, 41)
    ]

    def _mock_hybrid_search(q):
        rec_bm25.search(RetrievalQuery(
            query_id=q.query_id, original_question=q.original_question, normalized_question=q.normalized_question,
            top_k=40, candidate_k=40, requested_strategy=RetrievalStrategy.BM25
        ))
        rec_dense.search(RetrievalQuery(
            query_id=q.query_id, original_question=q.original_question, normalized_question=q.normalized_question,
            top_k=40, candidate_k=40, requested_strategy=RetrievalStrategy.DENSE
        ))
        return RetrievalResponse(
            query=q,
            strategy=RetrievalStrategy.HYBRID,
            hits=fused_hits,
            latency_ms=5.0,
        )

    mock_pipeline.hybrid_retriever.search.side_effect = _mock_hybrid_search

    scored_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{qid}-{r}",
            document_id=f"doc-{qid}-{r}",
            rank=r,
            score=round(0.95 - (r * 0.01), 8),
            strategy=RetrievalStrategy.RERANK,
            text="sample text",
        )
        for r in range(1, 41)
    ]
    mock_pipeline.reranker.rerank.return_value = RetrievalResponse(
        query=RetrievalQuery(query_id=qid, original_question=q_text, normalized_question=q_text),
        strategy=RetrievalStrategy.RERANK,
        hits=scored_hits,
        latency_ms=10.0,
    )

    case_res, case_met, reasons = run_case_candidate_pool_audit(
        mock_pipeline, qid, q_text, baseline
    )

    assert len(reasons) == 0
    assert case_res["reproduction_gates"]["branch_depth_fidelity"] is True
    obs = case_res["branch_depth_observations"]
    assert obs["sparse_query_count"] == 1
    assert obs["dense_query_count"] == 1
    assert obs["sparse_candidate_depths"] == [40]
    assert obs["dense_candidate_depths"] == [40]
    assert obs["all_sparse_depth_40"] is True
    assert obs["all_dense_depth_40"] is True


def test_11_branch_depth_failure_triggers_invalid_experiment(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]):
        baseline = load_and_verify_b1a2_baseline(bundle_dir, EXPECTED_22_IDS)

    qid = "102047"
    q_text = "Văn bản 102047 sửa đổi điều khoản nào?"

    mock_pipeline = MagicMock()
    mock_pipeline.query_understanding.enrich.return_value = RetrievalQuery(
        query_id=qid,
        original_question=q_text,
        normalized_question=q_text,
        top_k=8,
        candidate_k=40,
    )

    bm25_inner = MagicMock()
    dense_inner = MagicMock()
    rec_bm25 = RecordingBranchRetriever(bm25_inner)
    rec_dense = RecordingBranchRetriever(dense_inner)
    mock_pipeline.recording_bm25 = rec_bm25
    mock_pipeline.recording_dense = rec_dense

    fused_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{qid}-{r}",
            document_id=f"doc-{qid}-{r}",
            rank=r,
            score=round(1.0 / (r + 10), 8),
            strategy=RetrievalStrategy.HYBRID,
            text="sample text",
        )
        for r in range(1, 41)
    ]

    def _mock_hybrid_search_bad_depth(q):
        rec_bm25.search(RetrievalQuery(
            query_id=q.query_id, original_question=q.original_question, normalized_question=q.normalized_question,
            top_k=20, candidate_k=20, requested_strategy=RetrievalStrategy.BM25
        ))
        rec_dense.search(RetrievalQuery(
            query_id=q.query_id, original_question=q.original_question, normalized_question=q.normalized_question,
            top_k=40, candidate_k=40, requested_strategy=RetrievalStrategy.DENSE
        ))
        return RetrievalResponse(
            query=q, strategy=RetrievalStrategy.HYBRID, hits=fused_hits, latency_ms=5.0
        )

    mock_pipeline.hybrid_retriever.search.side_effect = _mock_hybrid_search_bad_depth

    scored_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{qid}-{r}",
            document_id=f"doc-{qid}-{r}",
            rank=r,
            score=round(0.95 - (r * 0.01), 8),
            strategy=RetrievalStrategy.RERANK,
            text="sample text",
        )
        for r in range(1, 41)
    ]
    mock_pipeline.reranker.rerank.return_value = RetrievalResponse(
        query=RetrievalQuery(query_id=qid, original_question=q_text, normalized_question=q_text),
        strategy=RetrievalStrategy.RERANK,
        hits=scored_hits,
        latency_ms=10.0,
    )

    case_res, case_met, reasons = run_case_candidate_pool_audit(
        mock_pipeline, qid, q_text, baseline
    )

    assert case_res["reproduction_gates"]["branch_depth_fidelity"] is False
    assert case_res["branch_depth_observations"]["all_sparse_depth_40"] is False
    assert len(reasons) > 0


# ======================================================================
# FULL PROTOCOL & EVIDENCE REGRESSION TESTS
# ======================================================================


def test_12_end_to_end_audit_pass_extracted_bundle(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    _setup_mock_staging_root(source_root)

    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")

    dummy_dev = {qid: {"question": f"Văn bản {qid}"} for qid in EXPECTED_22_IDS}
    for i in range(len(dummy_dev), CANONICAL_SOURCE_QUESTION_COUNT):
        dummy_dev[f"extra_{i}"] = {"question": "extra"}
    questions_file = tmp_path / "dev.json"
    questions_file.write_text(json.dumps(dummy_dev), encoding="utf-8")
    dev_sha = sha256_file(questions_file)

    config = ApplicationConfig(artifacts=ArtifactConfig(root_path=source_root), online=OnlineConfig())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")

    out_dir = tmp_path / "output"

    mock_emb = MagicMock()
    mock_emb.provider_name = "sentence-transformers"
    mock_emb.provider_version = "1.0"
    mock_emb.model_name = "test_model"
    mock_emb.model_revision = "rev"
    mock_emb.dimension = 384

    mock_rerank = MagicMock()
    mock_rerank.model_name = "test_rerank"

    def _score_for_chunk(cid: str) -> float:
        if "tail-chunk" in cid:
            return 0.875
        r = int(cid.split("-")[-1])
        return round(0.95 - (r * 0.01), 8)

    def _mock_rerank(q, candidates):
        values = list(candidates)
        ordered_cands = sorted(values, key=lambda c: (-_score_for_chunk(c.chunk_id), c.rank, c.chunk_id))
        hits = [
            RetrievalHit(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                rank=idx,
                score=_score_for_chunk(c.chunk_id),
                strategy=RetrievalStrategy.RERANK,
                text="sample text",
            )
            for idx, c in enumerate(ordered_cands, start=1)
        ]
        return RetrievalResponse(
            query=q, strategy=RetrievalStrategy.RERANK, hits=hits, latency_ms=10.0
        )

    mock_rerank.rerank.side_effect = _mock_rerank

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha), \
         patch("scripts.candidate_pool_reranker_audit.SQLiteFTS5BM25Backend") as mock_bm25_cls, \
         patch("scripts.candidate_pool_reranker_audit.NumpyVectorBackend") as mock_vec_cls, \
         patch("scripts.candidate_pool_reranker_audit.SentenceTransformerEmbeddingProvider", return_value=mock_emb), \
         patch("scripts.candidate_pool_reranker_audit.CrossEncoderReranker", return_value=mock_rerank):

        mock_bm25 = MagicMock()
        mock_bm25.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        def _mock_bm25_search(q):
            return RetrievalResponse(
                query=q,
                strategy=RetrievalStrategy.BM25,
                hits=[
                    RetrievalHit(
                        chunk_id=f"chunk-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-chunk-{q.query_id}-25",
                        document_id=f"doc-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-doc-{q.query_id}-25",
                        rank=r,
                        score=1.0 / r,
                        strategy=RetrievalStrategy.BM25,
                        text="sample",
                    )
                    for r in range(1, q.candidate_k + 1)
                ],
                latency_ms=1.0,
            )
        mock_bm25.search.side_effect = _mock_bm25_search
        mock_bm25_cls.return_value = mock_bm25

        mock_vec = MagicMock()
        mock_vec.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        mock_vec.embedding_provider_name = "sentence-transformers"
        mock_vec.embedding_provider_version = "1.0"
        mock_vec.model_name = "test_model"
        mock_vec.model_revision = "rev"
        mock_vec.dimension = 384
        def _mock_dense_search(q, q_vec=None):
            return RetrievalResponse(
                query=q,
                strategy=RetrievalStrategy.DENSE,
                hits=[
                    RetrievalHit(
                        chunk_id=f"chunk-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-chunk-{q.query_id}-25",
                        document_id=f"doc-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-doc-{q.query_id}-25",
                        rank=r,
                        score=1.0 / r,
                        strategy=RetrievalStrategy.DENSE,
                        text="sample",
                    )
                    for r in range(1, q.candidate_k + 1)
                ],
                latency_ms=1.0,
            )
        mock_vec.search.side_effect = _mock_dense_search
        mock_vec_cls.return_value = mock_vec

        report, decision, verdict = run_candidate_pool_audit_protocol(
            config_path=config_path,
            manifest_path=manifest_file,
            questions_path=questions_file,
            baseline_evidence_path=bundle_dir,
            output_dir=out_dir,
        )

        assert verdict == "CANDIDATE_POOL_AUDIT_PASS", f"Reasons: {report.get('reasons')}"
        assert decision["audit_verified"] is True
        assert decision["h40_promotion_authorized"] is False
        assert decision["summary"]["identical_top8_cases"] == 5
        assert decision["summary"]["changed_top8_cases"] == 17
        assert decision["summary"]["total_tail_entrants"] == 17
        assert decision["summary"]["branch_depth_passes"] == 22

        # Check evidence identity fields
        exec_ident = json.loads((out_dir / "execution" / "audit_execution_identity.json").read_text(encoding="utf-8"))
        assert exec_ident["canonical_b1a2_source_kind"] == "canonical_extracted_bundle"
        assert exec_ident["canonical_b1a2_observed_zip_sha256"] is None

        base_ident = json.loads((out_dir / "baseline" / "b1a2_baseline_identity.json").read_text(encoding="utf-8"))
        assert base_ident["source_kind"] == "canonical_extracted_bundle"
        assert base_ident["observed_zip_sha256"] is None
        assert base_ident["canonical_member_sha256"] == shas

        zip_path = out_dir / "candidate-pool-reranker-audit-evidence.zip"
        assert zip_path.is_file()
        with zipfile.ZipFile(zip_path) as z:
            names = set(z.namelist())
            assert "results/candidate_pool_audit_report.json" in names
            assert "results/candidate_pool_decision_report.json" in names
            assert "results/candidate_pool_case_results.jsonl" in names
            assert "results/candidate_pool_case_metrics.jsonl" in names
            assert "configs/runtime_config.json" in names
            assert "execution/audit_execution_identity.json" in names
            assert "baseline/b1a2_baseline_identity.json" in names
            assert "execution/graphless_root_inventory.json" in names


def test_13_drift_detected_when_mechanics_diverge(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    _setup_mock_staging_root(source_root)

    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir, alter_s20_final_ids=[EXPECTED_22_IDS[0]])

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")

    dummy_dev = {qid: {"question": f"Văn bản {qid}"} for qid in EXPECTED_22_IDS}
    for i in range(len(dummy_dev), CANONICAL_SOURCE_QUESTION_COUNT):
        dummy_dev[f"extra_{i}"] = {"question": "extra"}
    questions_file = tmp_path / "dev.json"
    questions_file.write_text(json.dumps(dummy_dev), encoding="utf-8")
    dev_sha = sha256_file(questions_file)

    config = ApplicationConfig(artifacts=ArtifactConfig(root_path=source_root), online=OnlineConfig())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")

    out_dir = tmp_path / "output"

    mock_emb = MagicMock()
    mock_emb.provider_name = "sentence-transformers"
    mock_emb.provider_version = "1.0"
    mock_emb.model_name = "test_model"
    mock_emb.model_revision = "rev"
    mock_emb.dimension = 384

    mock_rerank = MagicMock()
    mock_rerank.model_name = "test_rerank"

    def _score_for_chunk(cid: str) -> float:
        if "tail-chunk" in cid:
            return 0.875
        r = int(cid.split("-")[-1])
        return round(0.95 - (r * 0.01), 8)

    def _mock_rerank(q, candidates):
        values = list(candidates)
        ordered_cands = sorted(values, key=lambda c: (-_score_for_chunk(c.chunk_id), c.rank, c.chunk_id))
        hits = [
            RetrievalHit(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                rank=idx,
                score=_score_for_chunk(c.chunk_id),
                strategy=RetrievalStrategy.RERANK,
                text="sample text",
            )
            for idx, c in enumerate(ordered_cands, start=1)
        ]
        return RetrievalResponse(
            query=q, strategy=RetrievalStrategy.RERANK, hits=hits, latency_ms=10.0
        )

    mock_rerank.rerank.side_effect = _mock_rerank

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha), \
         patch("scripts.candidate_pool_reranker_audit.SQLiteFTS5BM25Backend") as mock_bm25_cls, \
         patch("scripts.candidate_pool_reranker_audit.NumpyVectorBackend") as mock_vec_cls, \
         patch("scripts.candidate_pool_reranker_audit.SentenceTransformerEmbeddingProvider", return_value=mock_emb), \
         patch("scripts.candidate_pool_reranker_audit.CrossEncoderReranker", return_value=mock_rerank):

        mock_bm25 = MagicMock()
        mock_bm25.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        def _mock_bm25_search_drift(q):
            return RetrievalResponse(
                query=q,
                strategy=RetrievalStrategy.BM25,
                hits=[
                    RetrievalHit(
                        chunk_id=f"chunk-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-chunk-{q.query_id}-25",
                        document_id=f"doc-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-doc-{q.query_id}-25",
                        rank=r,
                        score=1.0 / r,
                        strategy=RetrievalStrategy.BM25,
                        text="sample",
                    )
                    for r in range(1, q.candidate_k + 1)
                ],
                latency_ms=1.0,
            )
        mock_bm25.search.side_effect = _mock_bm25_search_drift
        mock_bm25_cls.return_value = mock_bm25

        mock_vec = MagicMock()
        mock_vec.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        mock_vec.embedding_provider_name = "sentence-transformers"
        mock_vec.embedding_provider_version = "1.0"
        mock_vec.model_name = "test_model"
        mock_vec.model_revision = "rev"
        mock_vec.dimension = 384
        def _mock_dense_search_drift(q, q_vec=None):
            return RetrievalResponse(
                query=q,
                strategy=RetrievalStrategy.DENSE,
                hits=[
                    RetrievalHit(
                        chunk_id=f"chunk-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-chunk-{q.query_id}-25",
                        document_id=f"doc-{q.query_id}-{r}" if not (EXPECTED_22_IDS.index(q.query_id) >= 5 and r == 25) else f"tail-doc-{q.query_id}-25",
                        rank=r,
                        score=1.0 / r,
                        strategy=RetrievalStrategy.DENSE,
                        text="sample",
                    )
                    for r in range(1, q.candidate_k + 1)
                ],
                latency_ms=1.0,
            )
        mock_vec.search.side_effect = _mock_dense_search_drift
        mock_vec_cls.return_value = mock_vec

        report, decision, verdict = run_candidate_pool_audit_protocol(
            config_path=config_path,
            manifest_path=manifest_file,
            questions_path=questions_file,
            baseline_evidence_path=bundle_dir,
            output_dir=out_dir,
        )

        assert verdict == "CANDIDATE_POOL_DRIFT_DETECTED"
        assert decision["audit_verified"] is False
        assert decision["h40_promotion_authorized"] is False


def test_14_retrieval_model_error_yields_invalid_experiment(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    _setup_mock_staging_root(source_root)

    bundle_dir = tmp_path / "bundle"
    shas = _build_dummy_b1a2_bundle(bundle_dir)

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")

    dummy_dev = {qid: {"question": f"Văn bản {qid}"} for qid in EXPECTED_22_IDS}
    for i in range(len(dummy_dev), CANONICAL_SOURCE_QUESTION_COUNT):
        dummy_dev[f"extra_{i}"] = {"question": "extra"}
    questions_file = tmp_path / "dev.json"
    questions_file.write_text(json.dumps(dummy_dev), encoding="utf-8")
    dev_sha = sha256_file(questions_file)

    config = ApplicationConfig(artifacts=ArtifactConfig(root_path=source_root), online=OnlineConfig())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")

    out_dir = tmp_path / "output"

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_MEMBER_SHA256", shas), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", shas["results/phase_b1a2_retrieval_results.jsonl"]), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha), \
         patch("scripts.candidate_pool_reranker_audit.SQLiteFTS5BM25Backend", side_effect=RuntimeError("BM25 failure")):

        report, decision, verdict = run_candidate_pool_audit_protocol(
            config_path=config_path,
            manifest_path=manifest_file,
            questions_path=questions_file,
            baseline_evidence_path=bundle_dir,
            output_dir=out_dir,
        )

        assert verdict == "INVALID_EXPERIMENT"
        assert decision["audit_verified"] is False
        assert decision["h40_promotion_authorized"] is False
