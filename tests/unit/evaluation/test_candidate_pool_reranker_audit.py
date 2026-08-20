"""Unit tests for Stage R1 S20 vs H40 Candidate-Pool / Reranker Mechanics Audit."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, PropertyMock, patch
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
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTrace,
)
from scripts.candidate_pool_reranker_audit import (
    CANONICAL_B1A2_EXECUTION_COMMIT,
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
    compute_aggregate_audit_metrics,
    get_fused_rank_bucket,
    load_and_verify_b1a2_baseline,
    package_audit_evidence,
    run_candidate_pool_audit_protocol,
    run_case_candidate_pool_audit,
    sha256_bytes,
    sha256_file,
    validate_graphless_staging_root,
)


def _build_dummy_b1a2_baseline(
    output_dir: Path,
    *,
    omit_summary: bool = False,
    summary_override: dict[str, object] | None = None,
    results_corrupt: bool = False,
    decision_override: dict[str, object] | None = None,
    custom_ids: list[str] | None = None,
    alter_seed_ids: list[str] | None = None,
    alter_s20_final_ids: list[str] | None = None,
    alter_h40_final_ids: list[str] | None = None,
) -> None:
    """Helper to construct dummy B1A.2 baseline evidence directory."""
    results_dir = output_dir / "results"
    evidence_dir = output_dir / "evidence"
    results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    ids = custom_ids or EXPECTED_22_IDS

    results_lines = []
    for idx, qid in enumerate(ids):
        # Default: First 5 cases have identical s20 and h40 top8; remaining 17 have changed top8
        is_identical = idx < 5

        s20_seed_hits = [
            {
                "chunk_id": f"chunk-{qid}-{r}" if not (alter_seed_ids and qid in alter_seed_ids and r == 1) else f"altered-{qid}",
                "document_id": f"doc-{qid}-{r}",
                "rank": r,
                "score": round(1.0 / (r + 10), 8),
                "strategy": "hybrid",
            }
            for r in range(1, 21)
        ]

        s20_final_hits = [
            {
                "chunk_id": f"chunk-{qid}-{r}" if not (alter_s20_final_ids and qid in alter_s20_final_ids and r == 1) else f"altered-{qid}",
                "document_id": f"doc-{qid}-{r}",
                "rank": r,
                "score": round(0.95 - (r * 0.01), 8),
                "strategy": "hybrid_rerank",
            }
            for r in range(1, 9)
        ]

        if is_identical:
            h40_final_hits = list(s20_final_hits)
        else:
            # Replace 8th hit with a tail entrant from fused rank 25
            h40_final_hits = [
                {
                    "chunk_id": f"chunk-{qid}-{r}" if not (alter_h40_final_ids and qid in alter_h40_final_ids and r == 1) else f"altered-{qid}",
                    "document_id": f"doc-{qid}-{r}",
                    "rank": r,
                    "score": round(0.95 - (r * 0.01), 8),
                    "strategy": "hybrid_rerank",
                }
                for r in range(1, 8)
            ] + [
                {
                    "chunk_id": f"tail-chunk-{qid}-25",
                    "document_id": f"tail-doc-{qid}-25",
                    "rank": 8,
                    "score": round(0.875, 8),
                    "strategy": "hybrid_rerank",
                }
            ]

        case_row = {
            "question_id": qid,
            "query_intent": "relationship",
            "query_variants_count": 1,
            "s20_arm": {
                "candidate_count": 20,
                "branch_candidate_depth": 40,
                "seed_hits": s20_seed_hits,
                "final_hits": s20_final_hits,
                "latency_ms": 10.0,
            },
            "h40_arm": {
                "candidate_count": 40,
                "final_hits": h40_final_hits,
                "latency_ms": 15.0,
            },
            "s20_vs_h40": {
                "top8_identical": is_identical,
                "top8_overlap_count": 8 if is_identical else 7,
                "top8_jaccard": 1.0 if is_identical else round(7.0 / 9.0, 4),
            },
        }
        results_lines.append(json.dumps(case_row))

    results_content = "\n".join(results_lines) + "\n"
    if results_corrupt:
        results_content = "corrupt content"
    results_file = results_dir / "phase_b1a2_retrieval_results.jsonl"
    results_file.write_text(results_content, encoding="utf-8")
    actual_results_sha = sha256_file(results_file)

    decision_data = {
        "verdict": "GRAPH_REDUNDANCY_PROVEN",
        "created_at": datetime.now(UTC).isoformat(),
    }
    if decision_override:
        decision_data.update(decision_override)
    (results_dir / "phase_b1a2_decision_report.json").write_text(
        json.dumps(decision_data, indent=2), encoding="utf-8"
    )

    if not omit_summary:
        summary_data = {
            "execution_git_commit": CANONICAL_B1A2_EXECUTION_COMMIT,
            "results_sha256": actual_results_sha,
            "case_count": len(ids),
            "created_at": datetime.now(UTC).isoformat(),
        }
        if summary_override:
            summary_data.update(summary_override)
        (evidence_dir / "phase_b1a2_run_summary.json").write_text(
            json.dumps(summary_data, indent=2), encoding="utf-8"
        )


def _setup_mock_staging_root(staging_root: Path) -> None:
    """Create mock graphless staging directory with manifests."""
    staging_root.mkdir(parents=True, exist_ok=True)
    for name in ("legal_chunks", "bm25", "vector"):
        (staging_root / name).mkdir(parents=True, exist_ok=True)

    chunk_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=100,
        created_at=datetime.now(UTC),
        processing_config_hash="hash",
        code_version="0.50.7",
        metadata={
            "payload_file": "chunks.jsonl",
            "payload_sha256": sha256_bytes(b""),
        },
    )
    (staging_root / "legal_chunks" / "manifest.json").write_text(
        chunk_manifest.model_dump_json(), encoding="utf-8"
    )
    (staging_root / "legal_chunks" / "chunks.jsonl").write_bytes(b"")

    bm25_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.BM25_INDEX,
        artifact_version="1.0",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=100,
        created_at=datetime.now(UTC),
        processing_config_hash="hash",
        code_version="0.50.7",
        metadata={
            "source_artifact_type": "legal_chunks",
            "source_artifact_version": "1.0",
            "source_processing_config_hash": "hash",
        },
    )
    (staging_root / "bm25" / "manifest.json").write_text(
        bm25_manifest.model_dump_json(), encoding="utf-8"
    )

    vector_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.VECTOR_INDEX,
        artifact_version="1.0",
        dataset_name="uit-dsc-2026-task2-selected-contexts",
        dataset_revision="canonical",
        record_count=100,
        created_at=datetime.now(UTC),
        processing_config_hash="hash",
        code_version="0.50.7",
        model_name="test_model",
        model_revision="rev",
        metadata={
            "source_artifact_type": "legal_chunks",
            "source_artifact_version": "1.0",
            "source_processing_config_hash": "hash",
            "dimension": 384,
            "embedding_provider_name": "sentence-transformers",
            "embedding_provider_version": "1.0",
        },
    )
    (staging_root / "vector" / "manifest.json").write_text(
        vector_manifest.model_dump_json(), encoding="utf-8"
    )


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------


def test_01_fused_rank_bucket_boundaries() -> None:
    assert get_fused_rank_bucket(21) == "21-25"
    assert get_fused_rank_bucket(25) == "21-25"
    assert get_fused_rank_bucket(26) == "26-30"
    assert get_fused_rank_bucket(30) == "26-30"
    assert get_fused_rank_bucket(31) == "31-35"
    assert get_fused_rank_bucket(35) == "31-35"
    assert get_fused_rank_bucket(36) == "36-40"
    assert get_fused_rank_bucket(40) == "36-40"
    assert get_fused_rank_bucket(20) == "unknown"
    assert get_fused_rank_bucket(41) == "unknown"


def test_02_load_and_verify_b1a2_baseline_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir)
    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        baseline = load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)
        assert baseline.case_count == 22
        assert baseline.decision_verdict == "GRAPH_REDUNDANCY_PROVEN"
        assert baseline.execution_git_commit == CANONICAL_B1A2_EXECUTION_COMMIT
        assert len(baseline.expected_s20_seed_hits) == 22
        assert len(baseline.expected_s20_final_hits) == 22
        assert len(baseline.expected_h40_final_hits) == 22


def test_03_load_and_verify_b1a2_baseline_zip(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2_dir"
    _build_dummy_b1a2_baseline(base_dir)
    real_results_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    zip_path = tmp_path / "b1a2.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in base_dir.rglob("*"):
            if f.is_file():
                z.write(f, arcname=str(f.relative_to(base_dir)))

    real_zip_sha = sha256_file(zip_path)

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_ZIP_SHA256", real_zip_sha), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_results_sha):
        baseline = load_and_verify_b1a2_baseline(zip_path, EXPECTED_22_IDS)
        assert baseline.case_count == 22


def test_04_b1a2_zip_sha_mismatch_rejected(tmp_path: Path) -> None:
    zip_path = tmp_path / "dummy.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("test.txt", "data")

    with pytest.raises(DataValidationError, match="B1A.2 baseline ZIP SHA mismatch"):
        load_and_verify_b1a2_baseline(zip_path, EXPECTED_22_IDS)


def test_05_b1a2_missing_summary_results_sha_rejected(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2_bad_summary"
    _build_dummy_b1a2_baseline(base_dir)
    summary_path = base_dir / "evidence" / "phase_b1a2_run_summary.json"
    s_data = json.loads(summary_path.read_text(encoding="utf-8"))
    del s_data["results_sha256"]
    summary_path.write_text(json.dumps(s_data), encoding="utf-8")

    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")
    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        with pytest.raises(DataValidationError, match="missing mandatory 'results_sha256'"):
            load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)


def test_06_b1a2_wrong_commit_rejected(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2_bad_commit"
    _build_dummy_b1a2_baseline(base_dir, summary_override={"execution_git_commit": "0" * 40})
    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        with pytest.raises(DataValidationError, match="execution commit mismatch"):
            load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)


def test_07_validate_graphless_staging_root_success_and_rejection(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    _setup_mock_staging_root(staging_dir)

    inv = validate_graphless_staging_root(staging_dir)
    assert len(inv) == 3
    assert {item["name"] for item in inv} == {"bm25", "legal_chunks", "vector"}

    # Forbidden graph directory rejected
    (staging_dir / "graph").mkdir()
    with pytest.raises(ArtifactCompatibilityError, match="MUST NOT contain forbidden artifact 'graph'"):
        validate_graphless_staging_root(staging_dir)


def test_08_pipeline_construction_and_autospecced_signatures(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    _setup_mock_staging_root(staging_dir)

    config = ApplicationConfig(
        artifacts=ArtifactConfig(root_path=staging_dir),
        online=OnlineConfig(),
    )

    with patch(
        "scripts.candidate_pool_reranker_audit.SQLiteFTS5BM25Backend",
        autospec=True,
    ) as mock_bm25_cls, patch(
        "scripts.candidate_pool_reranker_audit.NumpyVectorBackend",
        autospec=True,
    ) as mock_vec_cls, patch(
        "scripts.candidate_pool_reranker_audit.SentenceTransformerEmbeddingProvider",
        autospec=True,
    ) as mock_emb_cls, patch(
        "scripts.candidate_pool_reranker_audit.CrossEncoderReranker",
        autospec=True,
    ) as mock_rerank_cls:

        mock_bm25_inst = mock_bm25_cls.return_value
        type(mock_bm25_inst).source_artifact_identity = PropertyMock(
            return_value=("legal_chunks", "1.0", "hash")
        )

        mock_vec_inst = mock_vec_cls.return_value
        type(mock_vec_inst).source_artifact_identity = PropertyMock(
            return_value=("legal_chunks", "1.0", "hash")
        )
        type(mock_vec_inst).embedding_provider_name = PropertyMock(
            return_value="sentence-transformers"
        )
        type(mock_vec_inst).embedding_provider_version = PropertyMock(
            return_value="1.0"
        )
        type(mock_vec_inst).model_name = PropertyMock(return_value="test_model")
        type(mock_vec_inst).model_revision = PropertyMock(return_value="rev")
        type(mock_vec_inst).dimension = PropertyMock(return_value=384)

        mock_emb_inst = mock_emb_cls.return_value
        type(mock_emb_inst).provider_name = PropertyMock(
            return_value="sentence-transformers"
        )
        type(mock_emb_inst).provider_version = PropertyMock(return_value="1.0")
        type(mock_emb_inst).model_name = PropertyMock(return_value="test_model")
        type(mock_emb_inst).model_revision = PropertyMock(return_value="rev")
        type(mock_emb_inst).dimension = PropertyMock(return_value=384)

        mock_rerank_inst = mock_rerank_cls.return_value
        type(mock_rerank_inst).model_name = PropertyMock(return_value="test_reranker")

        pipeline = CandidatePoolAuditPipeline(config)

        mock_bm25_cls.assert_called_once()
        mock_vec_cls.assert_called_once()
        mock_emb_cls.assert_called_once_with(config.offline.embedding)
        assert isinstance(pipeline.hybrid_retriever, HybridRetriever)


def test_09_single_case_candidate_pool_audit_mechanics(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir)
    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        baseline = load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)

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

    # 40 fused candidates
    fused_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{qid}-{r}",
            document_id=f"doc-{qid}-{r}",
            rank=r,
            score=round(1.0 / (r + 10), 8),
            strategy=RetrievalStrategy.HYBRID,
            text="sample text",
            retrieval_trace=RetrievalTrace(
                bm25_rank=r,
                bm25_score=1.0,
                bm25_rrf_contribution=0.01,
                dense_rank=r,
                dense_score=0.9,
                dense_rrf_contribution=0.01,
            ),
        )
        for r in range(1, 41)
    ]
    mock_pipeline.hybrid_retriever.search.return_value = RetrievalResponse(
        query=RetrievalQuery(query_id=qid, original_question=q_text, normalized_question=q_text),
        strategy=RetrievalStrategy.HYBRID,
        hits=fused_hits,
        latency_ms=5.0,
    )

    # Scored candidates: same order as final hits
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
    assert case_res["reproduction_gates"]["seed_prefix_match"] is True
    assert case_res["reproduction_gates"]["s20_chunks_match"] is True
    assert case_res["reproduction_gates"]["h40_chunks_match"] is True
    assert len(case_res["fused_candidates_40"]) == 40
    assert len(case_res["derived_s20_final_hits"]) == 8
    assert len(case_res["derived_h40_final_hits"]) == 8
    assert case_met["top8_identical"] is True


def test_10_changed_case_diagnostics_and_tail_entrants(tmp_path: Path) -> None:
    base_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(base_dir)
    real_sha = sha256_file(base_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_sha):
        baseline = load_and_verify_b1a2_baseline(base_dir, EXPECTED_22_IDS)

    # Case index 6 is a changed case in our dummy baseline generator
    qid = EXPECTED_22_IDS[6]
    q_text = f"Văn bản {qid} sửa đổi quy định gì?"

    mock_pipeline = MagicMock()
    mock_pipeline.query_understanding.enrich.return_value = RetrievalQuery(
        query_id=qid, original_question=q_text, normalized_question=q_text, top_k=8, candidate_k=40
    )

    # 40 fused candidates where rank 25 is 'tail-chunk-{qid}-25'
    fused_hits = [
        RetrievalHit(
            chunk_id=f"chunk-{qid}-{r}" if r != 25 else f"tail-chunk-{qid}-25",
            document_id=f"doc-{qid}-{r}" if r != 25 else f"tail-doc-{qid}-25",
            rank=r,
            score=round(1.0 / (r + 10), 8),
            strategy=RetrievalStrategy.HYBRID,
            text="sample text",
        )
        for r in range(1, 41)
    ]
    mock_pipeline.hybrid_retriever.search.return_value = RetrievalResponse(
        query=RetrievalQuery(query_id=qid, original_question=q_text, normalized_question=q_text),
        strategy=RetrievalStrategy.HYBRID,
        hits=fused_hits,
        latency_ms=5.0,
    )

    def _score_for_chunk_10(cid: str) -> float:
        if "tail-chunk" in cid:
            return 0.875
        r = int(cid.split("-")[-1])
        return round(0.95 - (r * 0.01), 8)

    ordered_fused = sorted(fused_hits, key=lambda c: (-_score_for_chunk_10(c.chunk_id), c.rank, c.chunk_id))
    scored_hits = [
        RetrievalHit(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            rank=idx,
            score=_score_for_chunk_10(c.chunk_id),
            strategy=RetrievalStrategy.RERANK,
            text="sample text",
        )
        for idx, c in enumerate(ordered_fused, start=1)
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
    assert case_met["top8_identical"] is False
    assert case_met["tail_entrant_count"] == 1
    assert case_met["tail_entrant_fused_ranks"] == [25]
    assert case_met["tail_entrant_buckets"] == ["21-25"]
    assert case_met["displaced_s20_count"] == 1
    assert case_met["displaced_s20_fused_ranks"] == [8]

    # Margin check
    assert case_met["entrant_margin"] == round(0.875 - 0.870, 8)


def test_11_aggregate_metrics_calculation() -> None:
    # 2 cases: 1 identical, 1 changed with 2 entrants
    case_results = [
        {
            "question_id": "q1",
            "derived_s20_final_hits": [{"document_id": "docA"}, {"document_id": "docB"}],
            "derived_h40_final_hits": [{"document_id": "docA"}, {"document_id": "docB"}],
            "s20_vs_h40_comparison": {
                "top8_identical": True,
                "tail_entrant_count": 0,
                "top8_overlap_count": 8,
                "top8_jaccard": 1.0,
            },
            "tail_entrants": [],
            "score_cutoff_margin_diagnostics": {
                "s20_top8_cutoff_score": 0.88,
                "h40_top8_cutoff_score": 0.88,
                "entrant_vs_displaced_margin": None,
            },
        },
        {
            "question_id": "q2",
            "derived_s20_final_hits": [{"document_id": "docA"}, {"document_id": "docB"}],
            "derived_h40_final_hits": [{"document_id": "docA"}, {"document_id": "docC"}],
            "s20_vs_h40_comparison": {
                "top8_identical": False,
                "tail_entrant_count": 1,
                "top8_overlap_count": 7,
                "top8_jaccard": 0.7778,
            },
            "tail_entrants": [{"fused_rank": 26, "fused_rank_bucket": "26-30"}],
            "score_cutoff_margin_diagnostics": {
                "s20_top8_cutoff_score": 0.85,
                "h40_top8_cutoff_score": 0.87,
                "entrant_vs_displaced_margin": 0.02,
            },
        },
    ]

    case_metrics = [
        {"question_id": "q1", "top8_identical": True, "overlap_count": 8, "jaccard": 1.0, "tail_entrant_count": 0},
        {"question_id": "q2", "top8_identical": False, "overlap_count": 7, "jaccard": 0.7778, "tail_entrant_count": 1},
    ]

    agg = compute_aggregate_audit_metrics(case_results, case_metrics)
    assert agg["total_case_count"] == 2
    assert agg["identical_top8_cases"] == 1
    assert agg["changed_top8_cases"] == 1
    assert agg["total_tail_entrants"] == 1
    assert agg["cases_with_tail_entrants"] == 1
    assert agg["entrant_fused_rank_bucket_counts"]["26-30"] == 1
    assert agg["document_level_churn_count"] == 1


def test_12_full_protocol_end_to_end_audit_pass(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    _setup_mock_staging_root(source_root)

    b1a2_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(b1a2_dir)
    real_b1a2_sha = sha256_file(b1a2_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"question_ids": EXPECTED_22_IDS}), encoding="utf-8")

    dummy_dev = {qid: {"question": f"Văn bản {qid} sửa đổi"} for qid in EXPECTED_22_IDS}
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

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_b1a2_sha), \
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
            baseline_dir_or_zip=b1a2_dir,
            output_dir=out_dir,
        )

        assert verdict == "CANDIDATE_POOL_AUDIT_PASS", f"Reasons: {report.get('reasons')}"
        assert decision["audit_verified"] is True
        assert decision["h40_promotion_authorized"] is False
        assert decision["summary"]["identical_top8_cases"] == 5
        assert decision["summary"]["changed_top8_cases"] == 17
        assert decision["summary"]["total_tail_entrants"] == 17

        # Check evidence ZIP created and valid
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


def test_13_drift_detected_when_mechanics_diverge(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    _setup_mock_staging_root(source_root)

    # Baseline with altered s20 final ids in case 1
    b1a2_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(b1a2_dir, alter_s20_final_ids=[EXPECTED_22_IDS[0]])
    real_b1a2_sha = sha256_file(b1a2_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

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

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_b1a2_sha), \
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
            baseline_dir_or_zip=b1a2_dir,
            output_dir=out_dir,
        )

        assert verdict == "CANDIDATE_POOL_DRIFT_DETECTED"
        assert decision["audit_verified"] is False
        assert decision["h40_promotion_authorized"] is False


def test_14_retrieval_model_error_yields_invalid_experiment(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    _setup_mock_staging_root(source_root)

    b1a2_dir = tmp_path / "b1a2"
    _build_dummy_b1a2_baseline(b1a2_dir)
    real_b1a2_sha = sha256_file(b1a2_dir / "results" / "phase_b1a2_retrieval_results.jsonl")

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

    with patch("scripts.candidate_pool_reranker_audit.CANONICAL_B1A2_RESULTS_SHA256", real_b1a2_sha), \
         patch("scripts.candidate_pool_reranker_audit.CANONICAL_SOURCE_QUESTION_SHA256", dev_sha), \
         patch("scripts.candidate_pool_reranker_audit.SQLiteFTS5BM25Backend") as mock_bm25_cls, \
         patch("scripts.candidate_pool_reranker_audit.NumpyVectorBackend") as mock_vec_cls, \
         patch("scripts.candidate_pool_reranker_audit.SentenceTransformerEmbeddingProvider", return_value=mock_emb), \
         patch("scripts.candidate_pool_reranker_audit.CrossEncoderReranker", return_value=mock_rerank):

        mock_bm25 = MagicMock()
        mock_bm25.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        mock_bm25.search.side_effect = RuntimeError("CUDA out of memory in BM25")
        mock_bm25_cls.return_value = mock_bm25

        mock_vec = MagicMock()
        mock_vec.source_artifact_identity = ("legal_chunks", "1.0", "hash")
        mock_vec.embedding_provider_name = "sentence-transformers"
        mock_vec.embedding_provider_version = "1.0"
        mock_vec.model_name = "test_model"
        mock_vec.model_revision = "rev"
        mock_vec.dimension = 384
        mock_vec_cls.return_value = mock_vec

        report, decision, verdict = run_candidate_pool_audit_protocol(
            config_path=config_path,
            manifest_path=manifest_file,
            questions_path=questions_file,
            baseline_dir_or_zip=b1a2_dir,
            output_dir=out_dir,
        )

        assert verdict == "INVALID_EXPERIMENT"
        assert decision["audit_verified"] is False
        assert decision["h40_promotion_authorized"] is False
        assert decision["summary"]["retrieval_model_error_count"] > 0
