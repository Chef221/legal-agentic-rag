#!/usr/bin/env python3
"""Phase B1A.2: Graph Equivalence and Candidate-Pool Isolation protocol tooling.

Evaluates whether the current zero-edge GRAPH retrieval path is behaviorally
equivalent to a seed-equivalent direct hybrid-rerank path (ARM S20), while
isolating the candidate pool effect from standard HYBRID_RERANK@40 (ARM H40).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
from statistics import fmean
import unicodedata
import zipfile

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.configuration.online import RerankerConfig, RetrievalConfig
from legal_agentic_rag.contracts.embedding_provider import EmbeddingProvider
from legal_agentic_rag.contracts.graph_backend import GraphBackend
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.embeddings.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)
from legal_agentic_rag.exceptions import DataValidationError, RetrievalError
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.graph import AdjacencyGraphBackend
from legal_agentic_rag.indexing.vector import NumpyVectorBackend
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import (
    DenseRetriever,
    FixedRetriever,
    QueryUnderstandingService,
)
from legal_agentic_rag.retrieval.fixed import HybridRetriever
from legal_agentic_rag.retrieval.graph import GraphExpandedRetriever
from legal_agentic_rag.retrieval.rerank import RerankingRetriever
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.runtime.startup_validation import validate_startup_report
from legal_agentic_rag.serving.config_loader import load_application_config
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)

SCORE_ABS_TOLERANCE = 1e-6
EXPECTED_CASE_COUNT = 22
CANONICAL_SOURCE_QUESTION_COUNT = 991
CANONICAL_SOURCE_QUESTION_SHA256 = (
    "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"
)

S20_BRANCH_CANDIDATE_DEPTH = 40
S20_HYBRID_OUTPUT_LIMIT = 20
S20_RERANK_INPUT_LIMIT = 20
FINAL_TOP_K = 8


def sha256_file(path: Path) -> str:
    """Compute deterministic SHA-256 hex digest for a file."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest for bytes."""
    return sha256(data).hexdigest()


def normalize_question_text(question: str) -> str:
    """Apply standard ServingService question normalization."""
    stripped = question.strip()
    nfc = unicodedata.normalize("NFC", stripped)
    return " ".join(nfc.split())


# ----------------------------------------------------------------------
# RECORDING HYBRID ADAPTER
# ----------------------------------------------------------------------


class RecordingHybridCandidateAdapter:
    """Observational wrapper around HybridRetriever to audit graph seed calls."""

    def __init__(self, inner: HybridRetriever) -> None:
        self._inner = inner
        self.recorded_queries: list[RetrievalQuery] = []
        self.recorded_responses: list[RetrievalResponse] = []

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        return self._inner.source_artifact_identity

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.recorded_queries.append(query.model_copy(deep=True))
        response = self._inner.search(query)
        self.recorded_responses.append(response.model_copy(deep=True))
        return response


# ----------------------------------------------------------------------
# 1. PREPARE SUBCOMMAND
# ----------------------------------------------------------------------


def prepare_b1a2_dataset(
    development_path: Path,
    manifest_path: Path,
    output_path: Path,
    identity_output_path: Path | None = None,
) -> dict[str, object]:
    """Materialize the exact 22-question subset preserving canonical order."""
    dev_sha = sha256_file(development_path)
    if dev_sha != CANONICAL_SOURCE_QUESTION_SHA256:
        raise DataValidationError(
            f"Source development.json SHA mismatch: expected {CANONICAL_SOURCE_QUESTION_SHA256}, got {dev_sha}"
        )

    raw_dev = json.loads(development_path.read_text(encoding="utf-8"))
    if not isinstance(raw_dev, Mapping):
        raise DataValidationError("development.json root must be a mapping")
    if len(raw_dev) != CANONICAL_SOURCE_QUESTION_COUNT:
        raise DataValidationError(
            f"Source development.json question count mismatch: expected {CANONICAL_SOURCE_QUESTION_COUNT}, got {len(raw_dev)}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_ids: list[str] = manifest.get("question_ids", [])
    if len(target_ids) != EXPECTED_CASE_COUNT:
        raise DataValidationError(
            f"Manifest question_ids count mismatch: expected {EXPECTED_CASE_COUNT}, got {len(target_ids)}"
        )
    if len(set(target_ids)) != len(target_ids):
        raise DataValidationError("Manifest question_ids contain duplicates")

    dev_key_order = list(raw_dev.keys())
    for qid in target_ids:
        if qid not in raw_dev:
            raise DataValidationError(f"Question ID {qid} not found in development.json")

    ordered_subset = {
        qid: raw_dev[qid]
        for qid in dev_key_order
        if qid in set(target_ids)
    }

    if list(ordered_subset.keys()) != target_ids:
        raise DataValidationError(
            "Target IDs order in manifest does not match canonical development.json order"
        )

    output_bytes = json.dumps(ordered_subset, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    output_path.write_bytes(output_bytes)
    output_sha = sha256_bytes(output_bytes)

    identity = {
        "experiment_id": "PHASE-B1A.2",
        "created_at": datetime.now(UTC).isoformat(),
        "source_question_count": len(raw_dev),
        "source_question_sha256": dev_sha,
        "materialized_case_count": len(ordered_subset),
        "materialized_case_sha256": output_sha,
        "materialized_question_ids": list(ordered_subset.keys()),
        "output_path": str(output_path),
    }

    if identity_output_path is not None:
        identity_output_path.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return identity


# ----------------------------------------------------------------------
# 2. RUN SUBCOMMAND (RETRIEVAL ONLY)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HitProjection:
    chunk_id: str
    document_id: str
    rank: int
    score: float
    strategy: str


def _project_hits(hits: Sequence[RetrievalHit]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": h.chunk_id,
            "document_id": h.document_id,
            "rank": h.rank,
            "score": round(h.score, 8),
            "strategy": h.strategy.value,
        }
        for h in hits
    ]


class RetrievalPipelineSuite:
    """Assembles all retrieval backends and components without any generation models."""

    def __init__(
        self,
        config: ApplicationConfig,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        graph_backend: GraphBackend | None = None,
        bm25_backend: SQLiteFTS5BM25Backend | None = None,
        vector_backend: NumpyVectorBackend | None = None,
        chunk_manifest: ArtifactManifest | None = None,
    ) -> None:
        self.config = config
        deep_validation = config.online.startup_validation.mode == "full"

        if chunk_manifest is None:
            chunk_manifest = load_artifact_manifest(
                config.artifacts.directory("legal_chunks_directory"),
                expected_type=ArtifactType.LEGAL_CHUNKS,
                verify_payload=deep_validation,
            )
        self.chunk_manifest = chunk_manifest

        if bm25_backend is None:
            bm25_manifest = load_artifact_manifest(
                config.artifacts.directory("bm25_directory"),
                expected_type=ArtifactType.BM25_INDEX,
            )
            bm25_backend = SQLiteFTS5BM25Backend(
                config.offline.bm25,
                runtime_config=config.online.bm25_runtime,
                verify_integrity_on_load=deep_validation,
            )
            bm25_backend.load(config.artifacts.directory("bm25_directory"), bm25_manifest)
        self.bm25 = bm25_backend

        if vector_backend is None:
            vector_manifest = load_artifact_manifest(
                config.artifacts.directory("vector_directory"),
                expected_type=ArtifactType.VECTOR_INDEX,
            )
            vector_backend = NumpyVectorBackend(
                config.offline.vector_index,
                runtime_config=config.online.vector_runtime,
                verify_integrity_on_load=deep_validation,
                serving_metadata_source=config.artifacts.directory("vector_serving_directory"),
            )
            vector_backend.load(config.artifacts.directory("vector_directory"), vector_manifest)
        self.vector = vector_backend

        if graph_backend is None:
            graph_manifest = load_artifact_manifest(
                config.artifacts.directory("graph_directory"),
                expected_type=ArtifactType.GRAPH_INDEX,
            )
            graph_backend = AdjacencyGraphBackend(
                config.offline.graph_index,
                verify_integrity_on_load=deep_validation,
            )
            graph_backend.load(config.artifacts.directory("graph_directory"), graph_manifest)
        self.graph = graph_backend

        if embedding_provider is None:
            embedding_provider = SentenceTransformerEmbeddingProvider(
                config.offline.embedding
            )
        self.embedding_provider = embedding_provider

        if reranker is None:
            reranker = CrossEncoderReranker(config.online.reranker)
        self.reranker = reranker

        self.dense = DenseRetriever(self.embedding_provider, self.vector)
        self.hybrid = HybridRetriever(
            self.bm25,
            self.dense,
            config.online.retrieval,
            query_understanding_config=config.online.query_understanding,
        )
        self.query_understanding = QueryUnderstandingService(
            config.online.query_understanding
        )
        self.reranking = RerankingRetriever(
            self.hybrid,
            self.reranker,
            config.online.reranker,
        )


def run_b1a2_case(
    pipeline: RetrievalPipelineSuite,
    question_id: str,
    question_text: str,
) -> dict[str, object]:
    """Execute ARM G, ARM S20, and ARM H40 on one question and return audit observations."""
    norm_text = normalize_question_text(question_text)
    base_query = RetrievalQuery(
        query_id=question_id,
        original_question=question_text,
        normalized_question=norm_text,
        top_k=FINAL_TOP_K,
        candidate_k=S20_BRANCH_CANDIDATE_DEPTH,
    )

    enriched_query = pipeline.query_understanding.enrich(base_query)
    q_analysis = enriched_query.query_analysis

    # ------------------------------------------------------------------
    # ARM G — CURRENT GRAPH PATH
    # ------------------------------------------------------------------
    recording_hybrid = RecordingHybridCandidateAdapter(pipeline.hybrid)
    graph_retriever = GraphExpandedRetriever(
        candidate_retriever=recording_hybrid,
        graph_backend=pipeline.graph,
        reranker=pipeline.reranker,
        chunk_manifest=pipeline.chunk_manifest,
        retrieval_config=pipeline.config.online.retrieval,
        reranker_config=pipeline.config.online.reranker,
    )

    g_query = enriched_query.model_copy(update={"requested_strategy": RetrievalStrategy.GRAPH})
    g_response = graph_retriever.search(g_query)

    g_hybrid_calls = len(recording_hybrid.recorded_queries)
    g_captured_seed_query = recording_hybrid.recorded_queries[0] if g_hybrid_calls > 0 else None
    g_captured_seed_resp = recording_hybrid.recorded_responses[0] if g_hybrid_calls > 0 else None

    # ------------------------------------------------------------------
    # ARM S20 — SEED-EQUIVALENT DIRECT PATH
    # ------------------------------------------------------------------
    maximum_seed_slots = (
        enriched_query.candidate_k - 1 if enriched_query.candidate_k > 1 else 1
    )
    s20_seed_limit = min(
        pipeline.config.online.retrieval.graph_seed_chunk_k,
        maximum_seed_slots,
    )
    s20_seed_query = enriched_query.model_copy(
        update={
            "top_k": s20_seed_limit,
            "candidate_k": S20_BRANCH_CANDIDATE_DEPTH,
            "requested_strategy": RetrievalStrategy.HYBRID,
        }
    )

    s20_seed_response = pipeline.hybrid.search(s20_seed_query)
    s20_final_response = pipeline.reranking.rerank_candidates(
        enriched_query.model_copy(update={"requested_strategy": RetrievalStrategy.HYBRID_RERANK}),
        s20_seed_response,
    )

    # ------------------------------------------------------------------
    # ARM H40 — DIAGNOSTIC NORMAL HYBRID_RERANK PATH
    # ------------------------------------------------------------------
    h40_query = enriched_query.model_copy(
        update={"requested_strategy": RetrievalStrategy.HYBRID_RERANK}
    )
    h40_final_response = pipeline.reranking.search(h40_query)

    # Projections
    g_seed_proj = _project_hits(g_captured_seed_resp.hits if g_captured_seed_resp else [])
    g_final_proj = _project_hits(g_response.hits)

    s20_seed_proj = _project_hits(s20_seed_response.hits)
    s20_final_proj = _project_hits(s20_final_response.hits)

    h40_final_proj = _project_hits(h40_final_response.hits)

    # Equivalence checks (G vs S20)
    seed_chunks_match = [h["chunk_id"] for h in g_seed_proj] == [h["chunk_id"] for h in s20_seed_proj]
    final_chunks_match = [h["chunk_id"] for h in g_final_proj] == [h["chunk_id"] for h in s20_final_proj]
    final_docs_match = [h["document_id"] for h in g_final_proj] == [h["document_id"] for h in s20_final_proj]

    score_diffs = [
        abs(g_h["score"] - s_h["score"])
        for g_h, s_h in zip(g_final_proj, s20_final_proj, strict=False)
    ]
    max_score_diff = max(score_diffs) if score_diffs else 0.0
    scores_match = max_score_diff <= SCORE_ABS_TOLERANCE

    # Diagnostics (S20 vs H40)
    s20_ids = [h["chunk_id"] for h in s20_final_proj]
    h40_ids = [h["chunk_id"] for h in h40_final_proj]
    overlap_ids = set(s20_ids) & set(h40_ids)
    union_ids = set(s20_ids) | set(h40_ids)
    jaccard = len(overlap_ids) / len(union_ids) if union_ids else 1.0

    return {
        "question_id": question_id,
        "query_intent": q_analysis.intent.value if q_analysis else None,
        "query_variants_count": len(enriched_query.query_variants),
        "g_arm": {
            "hybrid_calls": g_hybrid_calls,
            "seed_query_top_k": g_captured_seed_query.top_k if g_captured_seed_query else None,
            "seed_query_candidate_k": g_captured_seed_query.candidate_k if g_captured_seed_query else None,
            "seed_hits": g_seed_proj,
            "final_hits": g_final_proj,
            "warnings": g_response.warnings,
            "latency_ms": g_response.latency_ms,
        },
        "s20_arm": {
            "branch_candidate_depth": S20_BRANCH_CANDIDATE_DEPTH,
            "seed_query_top_k": s20_seed_query.top_k,
            "seed_query_candidate_k": s20_seed_query.candidate_k,
            "seed_hits": s20_seed_proj,
            "final_hits": s20_final_proj,
            "warnings": s20_final_response.warnings,
            "latency_ms": s20_final_response.latency_ms,
        },
        "h40_arm": {
            "final_hits": h40_final_proj,
            "latency_ms": h40_final_response.latency_ms,
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
            ),
        },
        "s20_vs_h40": {
            "top8_identical": s20_ids == h40_ids,
            "top8_overlap_count": len(overlap_ids),
            "top8_jaccard": round(jaccard, 4),
            "s20_only_chunk_ids": [cid for cid in s20_ids if cid not in overlap_ids],
            "h40_only_chunk_ids": [cid for cid in h40_ids if cid not in overlap_ids],
        },
    }


def run_b1a2_experiment(
    config_path: Path,
    questions_path: Path,
    output_jsonl_path: Path,
    summary_output_path: Path | None = None,
) -> dict[str, object]:
    """Run full B1A.2 retrieval-only execution across all 22 questions."""
    app_config = load_application_config(config_path)

    # Invariants verification on config
    ret_cfg = app_config.online.retrieval
    if ret_cfg.top_k != FINAL_TOP_K or ret_cfg.candidate_k != S20_BRANCH_CANDIDATE_DEPTH:
        raise DataValidationError(
            f"Config retrieval limits mismatch: expected top_k={FINAL_TOP_K}, candidate_k={S20_BRANCH_CANDIDATE_DEPTH}, "
            f"got top_k={ret_cfg.top_k}, candidate_k={ret_cfg.candidate_k}"
        )
    if ret_cfg.graph_seed_chunk_k != S20_HYBRID_OUTPUT_LIMIT:
        raise DataValidationError(
            f"Config graph_seed_chunk_k mismatch: expected {S20_HYBRID_OUTPUT_LIMIT}, got {ret_cfg.graph_seed_chunk_k}"
        )

    pipeline = RetrievalPipelineSuite(app_config)

    # Graph edge invariant check
    if pipeline.graph.manifest.record_count != 0:
        raise DataValidationError(
            f"Graph backend record_count is not zero: {pipeline.graph.manifest.record_count}"
        )

    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_questions, Mapping):
        raise DataValidationError("22-question source must be an object mapping")

    ordered_qids = list(raw_questions.keys())
    if len(ordered_qids) != EXPECTED_CASE_COUNT:
        raise DataValidationError(
            f"Questions file count mismatch: expected {EXPECTED_CASE_COUNT}, got {len(ordered_qids)}"
        )

    results: list[dict[str, object]] = []
    lines: list[str] = []

    print(f"Starting Phase B1A.2 retrieval-only run across {len(ordered_qids)} cases...")
    for idx, qid in enumerate(ordered_qids, start=1):
        q_record = raw_questions[qid]
        q_text = q_record.get("question") or ""
        print(f"[B1A.2] {idx:02d}/{len(ordered_qids)} Processing question {qid}...")

        case_res = run_b1a2_case(pipeline, qid, q_text)
        results.append(case_res)
        lines.append(json.dumps(case_res, ensure_ascii=False))

    output_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    output_jsonl_path.write_bytes(output_bytes)
    results_sha = sha256_bytes(output_bytes)

    summary = {
        "experiment_id": "PHASE-B1A.2",
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(results),
        "results_sha256": results_sha,
        "config_sha256": sha256_file(config_path),
        "questions_sha256": sha256_file(questions_path),
        "graph_record_count": pipeline.graph.manifest.record_count,
        "s20_branch_candidate_depth": S20_BRANCH_CANDIDATE_DEPTH,
        "s20_hybrid_output_limit": S20_HYBRID_OUTPUT_LIMIT,
        "final_top_k": FINAL_TOP_K,
    }

    if summary_output_path is not None:
        summary_output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return summary


# ----------------------------------------------------------------------
# 3. ANALYZE SUBCOMMAND & VERDICT GATE
# ----------------------------------------------------------------------


def evaluate_b1a2_verdict_gate(
    case_results: Sequence[dict[str, object]],
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> tuple[str, list[str]]:
    """Mechanically evaluate the pre-registered B1A.2 equivalence verdict."""
    reasons: list[str] = []

    # 1. Hard Protocol Checks
    hard_fail = False
    if len(case_results) != expected_case_count:
        reasons.append(f"Hard protocol failure: expected {expected_case_count} cases, got {len(case_results)}")
        hard_fail = True

    for case in case_results:
        qid = case.get("question_id")
        g_arm = case.get("g_arm") or {}
        s20_arm = case.get("s20_arm") or {}

        if g_arm.get("hybrid_calls") != 1:
            reasons.append(f"Hard protocol failure ({qid}): G hybrid calls ({g_arm.get('hybrid_calls')}) != 1")
            hard_fail = True
        if "no_graph_expansion" not in (g_arm.get("warnings") or []):
            reasons.append(f"Hard protocol failure ({qid}): G warnings missing 'no_graph_expansion'")
            hard_fail = True
        if s20_arm.get("branch_candidate_depth") != S20_BRANCH_CANDIDATE_DEPTH:
            reasons.append(
                f"Hard protocol failure ({qid}): S20 branch candidate depth ({s20_arm.get('branch_candidate_depth')}) != {S20_BRANCH_CANDIDATE_DEPTH}"
            )
            hard_fail = True

    if hard_fail:
        return "INVALID_EXPERIMENT", reasons

    # 2. Equivalence Checks (G vs S20)
    mismatches: list[str] = []
    for case in case_results:
        qid = case.get("question_id")
        comp = case.get("g_vs_s20") or {}

        if not comp.get("seed_chunks_match"):
            mismatches.append(f"Case {qid}: seed chunk sequences differ")
        if not comp.get("final_chunks_match"):
            mismatches.append(f"Case {qid}: final top-8 chunk sequences differ")
        if not comp.get("final_docs_match"):
            mismatches.append(f"Case {qid}: final top-8 document sequences differ")
        if not comp.get("scores_match"):
            mismatches.append(f"Case {qid}: final reranker score delta ({comp.get('max_score_diff')}) > {SCORE_ABS_TOLERANCE}")

    if mismatches:
        reasons.extend(mismatches)
        return "GRAPH_REDUNDANCY_NOT_PROVEN", reasons

    reasons.append(
        "All 22 cases match exactly in seed sequences, final top-8 chunk sequences, "
        f"final document sequences, and reranker scores (tolerance <= {SCORE_ABS_TOLERANCE}) "
        "with exactly 0 graph steps and single seed candidate call."
    )
    return "GRAPH_REDUNDANCY_PROVEN", reasons


def analyze_b1a2_experiment(
    results_jsonl_path: Path,
    manifest_path: Path,
    output_report_path: Path,
    output_decision_path: Path,
    output_case_metrics_path: Path | None = None,
) -> dict[str, object]:
    """Analyze retrieval observations and apply mechanical verdict gate."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids: list[str] = manifest.get("question_ids", [])

    lines = [
        line.strip()
        for line in results_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [json.loads(line) for line in lines]

    actual_ids = [str(c.get("question_id")) for c in cases]
    if actual_ids != expected_ids:
        raise DataValidationError(
            f"Case IDs mismatch canonical manifest: expected {expected_ids}, got {actual_ids}"
        )

    verdict, reasons = evaluate_b1a2_verdict_gate(cases, len(expected_ids))

    # Aggregate S20 vs H40 diagnostics
    s20_h40_identical_count = sum(1 for c in cases if c.get("s20_vs_h40", {}).get("top8_identical") is True)
    s20_h40_changed_count = len(cases) - s20_h40_identical_count
    overlaps = [float(c.get("s20_vs_h40", {}).get("top8_overlap_count", 0)) for c in cases]
    jaccards = [float(c.get("s20_vs_h40", {}).get("top8_jaccard", 0.0)) for c in cases]

    mean_overlap = fmean(overlaps) if overlaps else 0.0
    min_overlap = min(overlaps) if overlaps else 0.0
    mean_jaccard = fmean(jaccards) if jaccards else 0.0

    # Aggregate G vs S20 comparisons
    seed_matches = sum(1 for c in cases if c.get("g_vs_s20", {}).get("seed_chunks_match") is True)
    final_matches = sum(1 for c in cases if c.get("g_vs_s20", {}).get("final_chunks_match") is True)
    doc_matches = sum(1 for c in cases if c.get("g_vs_s20", {}).get("final_docs_match") is True)
    score_passes = sum(1 for c in cases if c.get("g_vs_s20", {}).get("scores_match") is True)

    report = {
        "experiment_id": "PHASE-B1A.2",
        "code_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(cases),
        "verdict": verdict,
        "reasons": reasons,
        "g_vs_s20_equivalence": {
            "seed_matches_count": seed_matches,
            "final_top8_matches_count": final_matches,
            "document_sequence_matches_count": doc_matches,
            "score_tolerance_passes_count": score_passes,
            "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        },
        "s20_vs_h40_candidate_pool_diagnostics": {
            "identical_top8_count": s20_h40_identical_count,
            "changed_top8_count": s20_h40_changed_count,
            "mean_top8_overlap_count": round(mean_overlap, 4),
            "min_top8_overlap_count": int(min_overlap),
            "mean_top8_jaccard": round(mean_jaccard, 4),
        },
        "invariants": {
            "s20_branch_candidate_depth": S20_BRANCH_CANDIDATE_DEPTH,
            "s20_hybrid_output_limit": S20_HYBRID_OUTPUT_LIMIT,
            "final_top_k": FINAL_TOP_K,
        },
    }

    decision_report = {
        "experiment_id": "PHASE-B1A.2",
        "verdict": verdict,
        "reasons": reasons,
        "b1b_design_authorized": (verdict == "GRAPH_REDUNDANCY_PROVEN"),
        "g_vs_s20_summary": {
            "exact_matches": f"{final_matches}/{len(cases)}",
            "score_passes": f"{score_passes}/{len(cases)}",
        },
        "s20_vs_h40_summary": {
            "identical_top8": f"{s20_h40_identical_count}/{len(cases)}",
            "changed_top8": f"{s20_h40_changed_count}/{len(cases)}",
            "mean_overlap": round(mean_overlap, 4),
        },
    }

    output_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_decision_path.write_text(
        json.dumps(decision_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if output_case_metrics_path is not None:
        case_metric_lines = [
            json.dumps(
                {
                    "question_id": c["question_id"],
                    "g_vs_s20_equivalent": c.get("g_vs_s20", {}).get("equivalent"),
                    "g_vs_s20_max_score_diff": c.get("g_vs_s20", {}).get("max_score_diff"),
                    "s20_vs_h40_identical": c.get("s20_vs_h40", {}).get("top8_identical"),
                    "s20_vs_h40_overlap_count": c.get("s20_vs_h40", {}).get("top8_overlap_count"),
                    "s20_vs_h40_jaccard": c.get("s20_vs_h40", {}).get("top8_jaccard"),
                },
                ensure_ascii=False,
            )
            for c in cases
        ]
        output_case_metrics_path.write_bytes(
            ("\n".join(case_metric_lines) + "\n").encode("utf-8")
        )

    return report


# ----------------------------------------------------------------------
# 4. PACKAGE SUBCOMMAND
# ----------------------------------------------------------------------


def package_b1a2_evidence(
    output_zip_path: Path,
    manifest_path: Path,
    questions_identity_path: Path,
    runtime_config_path: Path,
    results_jsonl_path: Path,
    report_path: Path,
    decision_path: Path,
    case_metrics_path: Path | None = None,
) -> dict[str, object]:
    """Bundle all B1A.2 retrieval evidence into a single ZIP archive."""
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname="configs/phase-b1a-graph-routing-cases.json")
        zf.write(questions_identity_path, arcname="evidence/materialized_questions_identity.json")
        zf.write(runtime_config_path, arcname="configs/runtime_config.json")
        zf.write(results_jsonl_path, arcname="results/phase_b1a2_retrieval_results.jsonl")
        zf.write(report_path, arcname="results/phase_b1a2_graph_equivalence_report.json")
        zf.write(decision_path, arcname="results/phase_b1a2_decision_report.json")
        if case_metrics_path and case_metrics_path.exists():
            zf.write(case_metrics_path, arcname="results/phase_b1a2_case_metrics.jsonl")

    zip_sha = sha256_file(output_zip_path)
    zip_size = output_zip_path.stat().st_size

    return {
        "zip_path": str(output_zip_path),
        "zip_sha256": zip_sha,
        "zip_size_bytes": zip_size,
    }


# ----------------------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B1A.2: Graph Equivalence and Candidate-Pool Isolation Tooling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Materialize the 22-case benchmark subset")
    p_prep.add_argument("--development", type=Path, required=True, help="Path to canonical development.json")
    p_prep.add_argument("--manifest", type=Path, required=True, help="Path to B1A cases manifest")
    p_prep.add_argument("--output", type=Path, required=True, help="Path to output 22-case JSON")
    p_prep.add_argument("--identity-output", type=Path, default=None, help="Path to output identity JSON")

    # run
    p_run = subparsers.add_parser("run", help="Run retrieval-only comparison across G, S20, and H40")
    p_run.add_argument("--config", type=Path, required=True, help="Path to runtime ApplicationConfig JSON")
    p_run.add_argument("--questions", type=Path, required=True, help="Path to 22-question input JSON")
    p_run.add_argument("--output", type=Path, required=True, help="Path to output retrieval JSONL")
    p_run.add_argument("--summary-output", type=Path, default=None, help="Path to output run summary JSON")

    # analyze
    p_an = subparsers.add_parser("analyze", help="Analyze retrieval observations and evaluate verdict gate")
    p_an.add_argument("--results", type=Path, required=True, help="Path to retrieval-only JSONL")
    p_an.add_argument("--manifest", type=Path, required=True, help="Path to B1A cases manifest")
    p_an.add_argument("--output-report", type=Path, required=True, help="Path to output equivalence report JSON")
    p_an.add_argument("--output-decision", type=Path, required=True, help="Path to output decision report JSON")
    p_an.add_argument("--output-case-metrics", type=Path, default=None, help="Path to output case metrics JSONL")

    # package
    p_pk = subparsers.add_parser("package", help="Package B1A.2 evidence ZIP")
    p_pk.add_argument("--output-zip", type=Path, required=True, help="Path to output evidence ZIP")
    p_pk.add_argument("--manifest", type=Path, required=True, help="Path to B1A cases manifest")
    p_pk.add_argument("--questions-identity", type=Path, required=True, help="Path to materialized questions identity JSON")
    p_pk.add_argument("--runtime-config", type=Path, required=True, help="Path to runtime config")
    p_pk.add_argument("--results", type=Path, required=True, help="Path to results JSONL")
    p_pk.add_argument("--report", type=Path, required=True, help="Path to equivalence report JSON")
    p_pk.add_argument("--decision", type=Path, required=True, help="Path to decision report JSON")
    p_pk.add_argument("--case-metrics", type=Path, default=None, help="Path to case metrics JSONL")

    args = parser.parse_args()

    if args.command == "prepare":
        res = prepare_b1a2_dataset(
            development_path=args.development.resolve(),
            manifest_path=args.manifest.resolve(),
            output_path=args.output.resolve(),
            identity_output_path=args.identity_output.resolve() if args.identity_output else None,
        )
        print(json.dumps(res, indent=2))

    elif args.command == "run":
        res = run_b1a2_experiment(
            config_path=args.config.resolve(),
            questions_path=args.questions.resolve(),
            output_jsonl_path=args.output.resolve(),
            summary_output_path=args.summary_output.resolve() if args.summary_output else None,
        )
        print(json.dumps(res, indent=2))

    elif args.command == "analyze":
        res = analyze_b1a2_experiment(
            results_jsonl_path=args.results.resolve(),
            manifest_path=args.manifest.resolve(),
            output_report_path=args.output_report.resolve(),
            output_decision_path=args.output_decision.resolve(),
            output_case_metrics_path=args.output_case_metrics.resolve() if args.output_case_metrics else None,
        )
        print(json.dumps(res, indent=2))

    elif args.command == "package":
        res = package_b1a2_evidence(
            output_zip_path=args.output_zip.resolve(),
            manifest_path=args.manifest.resolve(),
            questions_identity_path=args.questions_identity.resolve(),
            runtime_config_path=args.runtime_config.resolve(),
            results_jsonl_path=args.results.resolve(),
            report_path=args.report.resolve(),
            decision_path=args.decision.resolve(),
            case_metrics_path=args.case_metrics.resolve() if args.case_metrics else None,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
