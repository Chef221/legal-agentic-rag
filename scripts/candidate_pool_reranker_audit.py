#!/usr/bin/env python3
"""S20 vs H40 Candidate-Pool / Reranker Mechanics Audit — Stage R1 tooling.

Audits candidate-pool depth effects (fused top-20 vs fused top-40) under identical
query understanding, branch retrieval (BM25 + dense), RRF fusion, and cross-encoder
reranker scoring without modifying production routing or promoting H40.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
from statistics import fmean, median
import subprocess
import tempfile
import unicodedata
import zipfile

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.configuration.online import RerankerConfig, RetrievalConfig
from legal_agentic_rag.contracts.embedding_provider import EmbeddingProvider
from legal_agentic_rag.contracts.reranker import Reranker
from legal_agentic_rag.embeddings.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.bm25 import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.vector import NumpyVectorBackend
from legal_agentic_rag.reranking import CrossEncoderReranker
from legal_agentic_rag.retrieval import (
    DenseRetriever,
    QueryUnderstandingService,
)
from legal_agentic_rag.retrieval.fixed import HybridRetriever
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.runtime.startup_validation import (
    validate_competition_artifact_lineage,
    validate_startup_report,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
)
from legal_agentic_rag.serving.config_loader import load_application_config

_LOGGER = logging.getLogger(__name__)

SCORE_ABS_TOLERANCE = 1e-6
EXPECTED_CASE_COUNT = 22
CANONICAL_SOURCE_QUESTION_COUNT = 991
CANONICAL_SOURCE_QUESTION_SHA256 = (
    "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"
)

CANONICAL_B1A2_ZIP_SHA256 = (
    "1fcc9150840573023d8ae443324d431635f59b54cd8325aa3324611bc1cb7117"
)
CANONICAL_B1A2_RESULTS_SHA256 = (
    "51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a"
)
CANONICAL_B1A2_EXECUTION_COMMIT = (
    "9265f3dadcf1ef0170f0abe618519da1657fc55e"
)

EXPECTED_22_IDS = [
    "102047", "107487", "110287", "111905", "113537",
    "122659", "125393", "133075", "134605", "147239",
    "147869", "150051", "26541",  "29491",  "29877",
    "39671",  "45219",  "47537",  "48905",  "64035",
    "95861",  "99639",
]

BRANCH_CANDIDATE_DEPTH = 40
FUSED_POOL_LIMIT = 40
S20_CANDIDATE_LIMIT = 20
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


def resolve_execution_git_commit(script_path: Path) -> str:
    """Resolve immutable git commit SHA for the repository containing the script."""
    repo_dir = script_path.resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        if len(commit) != 40 or not all(c in "0123456789abcdefABCDEF" for c in commit):
            raise ValueError(f"Invalid git commit output: '{commit}'")
        return commit
    except Exception as exc:
        raise DataValidationError(
            f"Failed to resolve execution git commit SHA: {exc}"
        ) from exc


def get_fused_rank_bucket(fused_rank: int) -> str:
    """Classify fused rank into standard 5-rank diagnostic buckets."""
    if 21 <= fused_rank <= 25:
        return "21-25"
    if 26 <= fused_rank <= 30:
        return "26-30"
    if 31 <= fused_rank <= 35:
        return "31-35"
    if 36 <= fused_rank <= 40:
        return "36-40"
    return "unknown"


# ----------------------------------------------------------------------
# DATA MODELS & SCHEMAS
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluatedHit:
    """A single hit serialized for baseline comparison."""

    rank: int
    chunk_id: str
    document_id: str
    score: float
    strategy: str


@dataclass(frozen=True, slots=True)
class FrozenB1A2Baseline:
    """Loaded and verified B1A.2 baseline evidence."""

    decision_verdict: str
    results_sha256: str
    execution_git_commit: str
    case_count: int
    expected_s20_seed_hits: dict[str, list[EvaluatedHit]]
    expected_s20_final_hits: dict[str, list[EvaluatedHit]]
    expected_h40_final_hits: dict[str, list[EvaluatedHit]]
    expected_s20_vs_h40: dict[str, dict]


def load_and_verify_b1a2_baseline(
    baseline_dir_or_zip: Path,
    expected_ids: list[str],
) -> FrozenB1A2Baseline:
    """Load and verify frozen B1A.2 baseline evidence against strict gates."""
    if not baseline_dir_or_zip.exists():
        raise DataValidationError(
            f"B1A.2 baseline path does not exist: {baseline_dir_or_zip}"
        )

    temp_unpack_dir: Path | None = None
    if baseline_dir_or_zip.is_file() and baseline_dir_or_zip.suffix == ".zip":
        actual_zip_sha = sha256_file(baseline_dir_or_zip)
        if actual_zip_sha != CANONICAL_B1A2_ZIP_SHA256:
            raise DataValidationError(
                f"B1A.2 baseline ZIP SHA mismatch: expected {CANONICAL_B1A2_ZIP_SHA256}, got {actual_zip_sha}"
            )
        temp_unpack_dir = Path(tempfile.mkdtemp(prefix="b1a2_unpacked_"))
        with zipfile.ZipFile(baseline_dir_or_zip, "r") as zip_ref:
            zip_ref.extractall(temp_unpack_dir)
        base_path = temp_unpack_dir
    else:
        base_path = baseline_dir_or_zip

    try:
        results_path = base_path / "results" / "phase_b1a2_retrieval_results.jsonl"
        if not results_path.exists():
            results_path = base_path / "phase_b1a2_retrieval_results.jsonl"
        if not results_path.exists():
            raise DataValidationError(
                f"Missing B1A.2 results jsonl at {results_path}"
            )

        decision_path = base_path / "results" / "phase_b1a2_decision_report.json"
        if not decision_path.exists():
            decision_path = base_path / "phase_b1a2_decision_report.json"
        if not decision_path.exists():
            raise DataValidationError(
                f"Missing B1A.2 decision report at {decision_path}"
            )

        summary_path = base_path / "evidence" / "phase_b1a2_run_summary.json"
        if not summary_path.exists():
            summary_path = base_path / "phase_b1a2_run_summary.json"
        if not summary_path.exists():
            raise DataValidationError(
                "Missing mandatory B1A.2 run summary at evidence/phase_b1a2_run_summary.json"
            )

        actual_results_sha = sha256_file(results_path)
        if actual_results_sha != CANONICAL_B1A2_RESULTS_SHA256:
            raise DataValidationError(
                f"B1A.2 results SHA256 mismatch: expected {CANONICAL_B1A2_RESULTS_SHA256}, got {actual_results_sha}"
            )

        decision_data = json.loads(decision_path.read_text(encoding="utf-8"))
        verdict = decision_data.get("verdict")
        if verdict != "GRAPH_REDUNDANCY_PROVEN":
            raise DataValidationError(
                f"B1A.2 verdict must be 'GRAPH_REDUNDANCY_PROVEN', got '{verdict}'"
            )

        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        if "results_sha256" not in summary_data or not summary_data["results_sha256"]:
            raise DataValidationError(
                "B1A.2 run summary missing mandatory 'results_sha256' field"
            )
        if summary_data["results_sha256"] != CANONICAL_B1A2_RESULTS_SHA256:
            raise DataValidationError(
                f"B1A.2 run summary results SHA mismatch: expected {CANONICAL_B1A2_RESULTS_SHA256}, got {summary_data['results_sha256']}"
            )

        if "execution_git_commit" not in summary_data:
            raise DataValidationError(
                "B1A.2 run summary missing mandatory 'execution_git_commit' field"
            )
        commit = summary_data["execution_git_commit"]
        if commit != CANONICAL_B1A2_EXECUTION_COMMIT:
            raise DataValidationError(
                f"B1A.2 execution commit mismatch: expected {CANONICAL_B1A2_EXECUTION_COMMIT}, got {commit}"
            )

        expected_s20_seeds: dict[str, list[EvaluatedHit]] = {}
        expected_s20_finals: dict[str, list[EvaluatedHit]] = {}
        expected_h40_finals: dict[str, list[EvaluatedHit]] = {}
        expected_s20_vs_h40: dict[str, dict] = {}

        for line_idx, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if not qid:
                raise DataValidationError(f"Line {line_idx} missing question_id")

            s20_arm = row.get("s20_arm")
            if not s20_arm or "final_hits" not in s20_arm:
                raise DataValidationError(f"Line {line_idx} missing s20_arm.final_hits")

            h40_arm = row.get("h40_arm")
            if not h40_arm or "final_hits" not in h40_arm:
                raise DataValidationError(f"Line {line_idx} missing h40_arm.final_hits")

            expected_s20_seeds[qid] = [
                EvaluatedHit(
                    rank=h["rank"],
                    chunk_id=h["chunk_id"],
                    document_id=h["document_id"],
                    score=float(h["score"]),
                    strategy=h["strategy"],
                )
                for h in s20_arm.get("seed_hits", [])
            ]

            expected_s20_finals[qid] = [
                EvaluatedHit(
                    rank=h["rank"],
                    chunk_id=h["chunk_id"],
                    document_id=h["document_id"],
                    score=float(h["score"]),
                    strategy=h["strategy"],
                )
                for h in s20_arm["final_hits"]
            ]

            expected_h40_finals[qid] = [
                EvaluatedHit(
                    rank=h["rank"],
                    chunk_id=h["chunk_id"],
                    document_id=h["document_id"],
                    score=float(h["score"]),
                    strategy=h["strategy"],
                )
                for h in h40_arm["final_hits"]
            ]

            expected_s20_vs_h40[qid] = row.get("s20_vs_h40", {})

        if set(expected_s20_finals.keys()) != set(expected_ids):
            raise DataValidationError(
                f"B1A.2 results IDs do not match expected canonical 22 IDs: missing {set(expected_ids) - set(expected_s20_finals.keys())}"
            )

        return FrozenB1A2Baseline(
            decision_verdict=verdict,
            results_sha256=actual_results_sha,
            execution_git_commit=commit,
            case_count=len(expected_s20_finals),
            expected_s20_seed_hits=expected_s20_seeds,
            expected_s20_final_hits=expected_s20_finals,
            expected_h40_final_hits=expected_h40_finals,
            expected_s20_vs_h40=expected_s20_vs_h40,
        )
    finally:
        if temp_unpack_dir is not None:
            import shutil
            shutil.rmtree(temp_unpack_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# GRAPHLESS ARTIFACT STAGING & PIPELINE
# ----------------------------------------------------------------------


def validate_graphless_staging_root(staging_root: Path) -> list[dict[str, object]]:
    """Verify graphless staging directory contains exactly the 3 online artifacts."""
    if not staging_root.is_dir():
        raise ArtifactCompatibilityError(f"Staging root is not a directory: {staging_root}")

    required_artifacts = {"legal_chunks", "bm25", "vector"}
    forbidden_artifacts = {"graph", "relationships"}

    found = set()
    inventory = []
    for child in staging_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name in forbidden_artifacts:
            raise ArtifactCompatibilityError(
                f"Graphless staging root MUST NOT contain forbidden artifact '{name}'"
            )
        manifest_path = child / "manifest.json"
        if manifest_path.is_file():
            found.add(name)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            inventory.append({
                "name": name,
                "path": str(child),
                "artifact_type": manifest.get("artifact_type"),
                "artifact_version": manifest.get("artifact_version"),
                "record_count": manifest.get("record_count"),
            })

    missing = required_artifacts - found
    if missing:
        raise ArtifactCompatibilityError(
            f"Staging root is missing required graphless artifacts: {sorted(missing)}"
        )

    return sorted(inventory, key=lambda x: x["name"])


class CandidatePoolAuditPipeline:
    """Assembles query understanding, hybrid retrieval, and reranking backends."""

    config: ApplicationConfig
    chunk_manifest: ArtifactManifest
    bm25_manifest: ArtifactManifest
    vector_manifest: ArtifactManifest
    bm25_backend: SQLiteFTS5BM25Backend
    vector_backend: NumpyVectorBackend
    embedding_provider: EmbeddingProvider
    dense_retriever: DenseRetriever
    hybrid_retriever: HybridRetriever
    query_understanding: QueryUnderstandingService
    reranker: Reranker

    def __init__(
        self,
        config: ApplicationConfig,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        staging_root = config.artifacts.root_path

        chunk_dir = staging_root / "legal_chunks"
        bm25_dir = config.artifacts.directory("bm25_directory")
        vector_dir = config.artifacts.directory("vector_directory")
        vector_serving_dir = config.artifacts.directory("vector_serving_directory")

        self.chunk_manifest = load_artifact_manifest(
            chunk_dir, expected_type=ArtifactType.LEGAL_CHUNKS
        )
        self.bm25_manifest = load_artifact_manifest(
            bm25_dir, expected_type=ArtifactType.BM25_INDEX
        )
        self.vector_manifest = load_artifact_manifest(
            vector_dir, expected_type=ArtifactType.VECTOR_INDEX
        )

        validate_competition_artifact_lineage(
            (self.chunk_manifest, self.bm25_manifest, self.vector_manifest),
            config.competition,
        )

        deep_validation = config.online.startup_validation.mode == "full"

        self.bm25_backend = SQLiteFTS5BM25Backend(
            config.offline.bm25,
            runtime_config=config.online.bm25_runtime,
            verify_integrity_on_load=deep_validation,
        )
        self.bm25_backend.load(bm25_dir, self.bm25_manifest)

        self.vector_backend = NumpyVectorBackend(
            config.offline.vector_index,
            runtime_config=config.online.vector_runtime,
            verify_integrity_on_load=deep_validation,
            serving_metadata_source=vector_serving_dir,
        )
        self.vector_backend.load(vector_dir, self.vector_manifest)

        self.embedding_provider = (
            embedding_provider
            or SentenceTransformerEmbeddingProvider(config.offline.embedding)
        )

        self._validate_embedding_provider_identity()

        self.dense_retriever = DenseRetriever(
            self.embedding_provider, self.vector_backend
        )

        self.hybrid_retriever = HybridRetriever(
            self.bm25_backend,
            self.dense_retriever,
            config=config.online.retrieval,
            query_understanding_config=config.online.query_understanding,
        )

        self.query_understanding = QueryUnderstandingService(
            config.online.query_understanding
        )

        self.reranker = reranker or CrossEncoderReranker(config.online.reranker)

    def _validate_embedding_provider_identity(self) -> None:
        metadata = self.vector_manifest.metadata
        expected_dim = metadata.get("dimension")
        if (
            metadata.get("embedding_provider_name") != self.embedding_provider.provider_name
            or metadata.get("embedding_provider_version") != self.embedding_provider.provider_version
            or self.vector_manifest.model_name != self.embedding_provider.model_name
            or self.vector_manifest.model_revision != self.embedding_provider.model_revision
            or expected_dim != self.embedding_provider.dimension
        ):
            raise ArtifactCompatibilityError(
                "Configured embedding provider is incompatible with vector artifact"
            )


# ----------------------------------------------------------------------
# AUDIT EXECUTION & METRIC CALCULATION
# ----------------------------------------------------------------------


def run_case_candidate_pool_audit(
    pipeline: CandidatePoolAuditPipeline,
    question_id: str,
    question_text: str,
    baseline: FrozenB1A2Baseline,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    """Execute single-pass hybrid retrieval and reranking for one case.

    Returns:
        (case_result, case_metrics, case_reasons)
    """
    reasons: list[str] = []
    norm_text = normalize_question_text(question_text)

    base_query = RetrievalQuery(
        query_id=question_id,
        original_question=question_text,
        normalized_question=norm_text,
        top_k=FINAL_TOP_K,
        candidate_k=BRANCH_CANDIDATE_DEPTH,
    )

    enriched_query = pipeline.query_understanding.enrich(base_query)
    q_analysis = enriched_query.query_analysis

    # 1. Execute ONE top-level HYBRID search retrieving fused top 40
    candidate_query = enriched_query.model_copy(
        update={
            "top_k": FUSED_POOL_LIMIT,
            "candidate_k": BRANCH_CANDIDATE_DEPTH,
            "requested_strategy": RetrievalStrategy.HYBRID,
        }
    )
    hybrid_resp = pipeline.hybrid_retriever.search(candidate_query)
    fused_hits_40 = list(hybrid_resp.hits)

    # Project fused 40 candidates with traces
    fused_candidate_records: list[dict[str, object]] = []
    for h in fused_hits_40:
        trace = h.retrieval_trace
        fused_candidate_records.append({
            "fused_rank": h.rank,
            "chunk_id": h.chunk_id,
            "document_id": h.document_id,
            "hybrid_rrf_score": round(h.score, 8),
            "bm25_rank": trace.bm25_rank if trace else None,
            "bm25_rrf_contribution": round(trace.bm25_rrf_contribution, 8) if trace and trace.bm25_rrf_contribution is not None else None,
            "dense_rank": trace.dense_rank if trace else None,
            "dense_rrf_contribution": round(trace.dense_rrf_contribution, 8) if trace and trace.dense_rrf_contribution is not None else None,
        })

    # Gate 1: Verify current fused40[:20] chunk sequence matches frozen B1A.2 s20_arm.seed_hits
    expected_s20_seeds = baseline.expected_s20_seed_hits.get(question_id, [])
    actual_s20_seed_ids = [h.chunk_id for h in fused_hits_40[:S20_CANDIDATE_LIMIT]]
    expected_s20_seed_ids = [h.chunk_id for h in expected_s20_seeds]

    seed_prefix_match = actual_s20_seed_ids == expected_s20_seed_ids
    if not seed_prefix_match:
        reasons.append(
            f"Case {question_id} fused40[:20] prefix does not match frozen B1A.2 s20 seed hits"
        )

    # 2. Score the SAME fused40 candidate list once with CrossEncoder
    rerank_query = enriched_query.model_copy(
        update={
            "top_k": len(fused_hits_40),
            "candidate_k": BRANCH_CANDIDATE_DEPTH,
            "requested_strategy": RetrievalStrategy.RERANK,
        }
    )
    all_scored_resp = pipeline.reranker.rerank(rerank_query, fused_hits_40)
    all_scored_hits = list(all_scored_resp.hits)

    # Map chunk_id to scored hit
    scored_by_chunk = {h.chunk_id: h for h in all_scored_hits}

    scored_records_40: list[dict[str, object]] = []
    for rank_idx, h in enumerate(all_scored_hits, start=1):
        fused_orig = next((fc for fc in fused_candidate_records if fc["chunk_id"] == h.chunk_id), None)
        scored_records_40.append({
            "reranker_rank": rank_idx,
            "fused_rank": fused_orig["fused_rank"] if fused_orig else None,
            "chunk_id": h.chunk_id,
            "document_id": h.document_id,
            "reranker_score": round(h.score, 8),
        })

    # 3. Derive S20 ranking:
    # Filter scored candidates to fused ranks 1..20, apply production tie-break (-score, fused_rank, chunk_id)
    s20_candidate_hits = fused_hits_40[:S20_CANDIDATE_LIMIT]
    s20_ordered_indices = sorted(
        range(len(s20_candidate_hits)),
        key=lambda idx: (
            -float(scored_by_chunk[s20_candidate_hits[idx].chunk_id].score),
            s20_candidate_hits[idx].rank,
            s20_candidate_hits[idx].chunk_id,
        ),
    )[:FINAL_TOP_K]

    derived_s20_top8: list[dict[str, object]] = []
    for rank_idx, idx in enumerate(s20_ordered_indices, start=1):
        cand = s20_candidate_hits[idx]
        sc = scored_by_chunk[cand.chunk_id]
        derived_s20_top8.append({
            "rank": rank_idx,
            "chunk_id": cand.chunk_id,
            "document_id": cand.document_id,
            "score": round(sc.score, 8),
            "fused_rank": cand.rank,
            "strategy": RetrievalStrategy.HYBRID_RERANK.value,
        })

    # 4. Derive H40 ranking:
    # Use scored fused ranks 1..40, apply production tie-break (-score, fused_rank, chunk_id)
    h40_ordered_indices = sorted(
        range(len(fused_hits_40)),
        key=lambda idx: (
            -float(scored_by_chunk[fused_hits_40[idx].chunk_id].score),
            fused_hits_40[idx].rank,
            fused_hits_40[idx].chunk_id,
        ),
    )[:FINAL_TOP_K]

    derived_h40_top8: list[dict[str, object]] = []
    for rank_idx, idx in enumerate(h40_ordered_indices, start=1):
        cand = fused_hits_40[idx]
        sc = scored_by_chunk[cand.chunk_id]
        derived_h40_top8.append({
            "rank": rank_idx,
            "chunk_id": cand.chunk_id,
            "document_id": cand.document_id,
            "score": round(sc.score, 8),
            "fused_rank": cand.rank,
            "strategy": RetrievalStrategy.HYBRID_RERANK.value,
        })

    # Gate 2: Verify derived S20 top8 against frozen B1A.2 s20_arm.final_hits
    expected_s20_finals = baseline.expected_s20_final_hits.get(question_id, [])
    s20_chunks_match = [h["chunk_id"] for h in derived_s20_top8] == [h.chunk_id for h in expected_s20_finals]
    s20_docs_match = [h["document_id"] for h in derived_s20_top8] == [h.document_id for h in expected_s20_finals]
    s20_score_diffs = [
        abs(h["score"] - exp.score)
        for h, exp in zip(derived_s20_top8, expected_s20_finals, strict=False)
    ]
    s20_max_score_diff = max(s20_score_diffs) if s20_score_diffs else 0.0
    s20_scores_match = s20_max_score_diff <= SCORE_ABS_TOLERANCE

    if not (s20_chunks_match and s20_docs_match and s20_scores_match):
        reasons.append(
            f"Case {question_id} derived S20 final top8 differs from frozen B1A.2 (chunks={s20_chunks_match}, docs={s20_docs_match}, score_diff={s20_max_score_diff:.8f})"
        )

    # Gate 3: Verify derived H40 top8 against frozen B1A.2 h40_arm.final_hits
    expected_h40_finals = baseline.expected_h40_final_hits.get(question_id, [])
    h40_chunks_match = [h["chunk_id"] for h in derived_h40_top8] == [h.chunk_id for h in expected_h40_finals]
    h40_docs_match = [h["document_id"] for h in derived_h40_top8] == [h.document_id for h in expected_h40_finals]
    h40_score_diffs = [
        abs(h["score"] - exp.score)
        for h, exp in zip(derived_h40_top8, expected_h40_finals, strict=False)
    ]
    h40_max_score_diff = max(h40_score_diffs) if h40_score_diffs else 0.0
    h40_scores_match = h40_max_score_diff <= SCORE_ABS_TOLERANCE

    if not (h40_chunks_match and h40_docs_match and h40_scores_match):
        reasons.append(
            f"Case {question_id} derived H40 final top8 differs from frozen B1A.2 (chunks={h40_chunks_match}, docs={h40_docs_match}, score_diff={h40_max_score_diff:.8f})"
        )

    # 5. S20 vs H40 Diagnostics & Entrants Analysis
    s20_top8_ids = [h["chunk_id"] for h in derived_s20_top8]
    h40_top8_ids = [h["chunk_id"] for h in derived_h40_top8]

    top8_identical = s20_top8_ids == h40_top8_ids
    overlap_ids = set(s20_top8_ids) & set(h40_top8_ids)
    union_ids = set(s20_top8_ids) | set(h40_top8_ids)
    jaccard = len(overlap_ids) / len(union_ids) if union_ids else 1.0

    s20_only_ids = [cid for cid in s20_top8_ids if cid not in overlap_ids]
    h40_only_ids = [cid for cid in h40_top8_ids if cid not in overlap_ids]

    # Tail entrants (H40-only chunks entering top-8 from fused ranks 21..40)
    tail_entrants: list[dict[str, object]] = []
    for h in derived_h40_top8:
        if h["chunk_id"] in h40_only_ids:
            fused_rank = int(h["fused_rank"])
            tail_entrants.append({
                "chunk_id": h["chunk_id"],
                "document_id": h["document_id"],
                "fused_rank": fused_rank,
                "reranker_rank": h["rank"],
                "reranker_score": h["score"],
                "fused_rank_bucket": get_fused_rank_bucket(fused_rank),
            })

    # Displaced S20 candidates (S20 chunks displaced from top-8 when pool expanded)
    displaced_s20: list[dict[str, object]] = []
    for h in derived_s20_top8:
        if h["chunk_id"] in s20_only_ids:
            displaced_s20.append({
                "chunk_id": h["chunk_id"],
                "document_id": h["document_id"],
                "fused_rank": h["fused_rank"],
                "s20_reranker_rank": h["rank"],
                "reranker_score": h["score"],
            })

    # Cutoffs and Margins (Deterministic set-level diagnostics)
    s20_cutoff_score = derived_s20_top8[-1]["score"] if derived_s20_top8 else 0.0
    h40_cutoff_score = derived_h40_top8[-1]["score"] if derived_h40_top8 else 0.0

    min_entrant_score = (
        min(e["reranker_score"] for e in tail_entrants)
        if tail_entrants else None
    )
    max_displaced_score = (
        max(d["reranker_score"] for d in displaced_s20)
        if displaced_s20 else None
    )
    entrant_margin = (
        round(min_entrant_score - max_displaced_score, 8)
        if min_entrant_score is not None and max_displaced_score is not None
        else None
    )

    # Build full case record
    case_result = {
        "question_id": question_id,
        "query_intent": q_analysis.intent.value if q_analysis else None,
        "query_variants_count": len(enriched_query.query_variants),
        "branch_depth_observations": {
            "sparse_branch_candidate_depth": BRANCH_CANDIDATE_DEPTH,
            "dense_branch_candidate_depth": BRANCH_CANDIDATE_DEPTH,
        },
        "fused_candidates_40": fused_candidate_records,
        "cross_encoder_scored_candidates_40": scored_records_40,
        "derived_s20_final_hits": derived_s20_top8,
        "derived_h40_final_hits": derived_h40_top8,
        "reproduction_gates": {
            "seed_prefix_match": seed_prefix_match,
            "s20_chunks_match": s20_chunks_match,
            "s20_docs_match": s20_docs_match,
            "s20_scores_match": s20_scores_match,
            "s20_max_score_diff": s20_max_score_diff,
            "h40_chunks_match": h40_chunks_match,
            "h40_docs_match": h40_docs_match,
            "h40_scores_match": h40_scores_match,
            "h40_max_score_diff": h40_max_score_diff,
        },
        "s20_vs_h40_comparison": {
            "top8_identical": top8_identical,
            "top8_overlap_count": len(overlap_ids),
            "top8_jaccard": round(jaccard, 4),
            "s20_only_chunks": s20_only_ids,
            "h40_only_chunks": h40_only_ids,
            "tail_entrant_count": len(tail_entrants),
            "displaced_s20_count": len(displaced_s20),
        },
        "tail_entrants": tail_entrants,
        "displaced_s20_candidates": displaced_s20,
        "score_cutoff_margin_diagnostics": {
            "s20_top8_cutoff_score": s20_cutoff_score,
            "h40_top8_cutoff_score": h40_cutoff_score,
            "min_h40_entrant_score": min_entrant_score,
            "max_displaced_s20_score": max_displaced_score,
            "entrant_vs_displaced_margin": entrant_margin,
        },
    }

    case_metrics = {
        "question_id": question_id,
        "top8_identical": top8_identical,
        "overlap_count": len(overlap_ids),
        "jaccard": round(jaccard, 4),
        "tail_entrant_count": len(tail_entrants),
        "tail_entrant_fused_ranks": [e["fused_rank"] for e in tail_entrants],
        "tail_entrant_buckets": [e["fused_rank_bucket"] for e in tail_entrants],
        "displaced_s20_count": len(displaced_s20),
        "displaced_s20_fused_ranks": [d["fused_rank"] for d in displaced_s20],
        "entrant_margin": entrant_margin,
    }

    return case_result, case_metrics, reasons


def compute_aggregate_audit_metrics(
    case_results: list[dict[str, object]],
    case_metrics: list[dict[str, object]],
) -> dict[str, object]:
    """Compute aggregate candidate-pool mechanics and churn diagnostics."""
    total_cases = len(case_results)
    identical_cases = sum(1 for m in case_metrics if m["top8_identical"])
    changed_cases = total_cases - identical_cases

    all_entrants = [e for res in case_results for e in res["tail_entrants"]]
    cases_with_entrants = sum(1 for m in case_metrics if m["tail_entrant_count"] > 0)

    entrant_counts_changed = [m["tail_entrant_count"] for m in case_metrics if not m["top8_identical"]]
    overlaps = [m["overlap_count"] for m in case_metrics]
    jaccards = [m["jaccard"] for m in case_metrics]

    bucket_counts = {"21-25": 0, "26-30": 0, "31-35": 0, "36-40": 0}
    for e in all_entrants:
        b = e["fused_rank_bucket"]
        if b in bucket_counts:
            bucket_counts[b] += 1

    # Document-level churn
    doc_churn_total = 0
    for res in case_results:
        s20_docs = {h["document_id"] for h in res["derived_s20_final_hits"]}
        h40_docs = {h["document_id"] for h in res["derived_h40_final_hits"]}
        doc_churn_total += len(h40_docs - s20_docs)

    # Margins among changed cases
    margins = [
        res["score_cutoff_margin_diagnostics"]["entrant_vs_displaced_margin"]
        for res in case_results
        if res["score_cutoff_margin_diagnostics"]["entrant_vs_displaced_margin"] is not None
    ]
    s20_cutoffs = [res["score_cutoff_margin_diagnostics"]["s20_top8_cutoff_score"] for res in case_results]
    h40_cutoffs = [res["score_cutoff_margin_diagnostics"]["h40_top8_cutoff_score"] for res in case_results]

    # Cases ordered by greatest candidate pool churn (entrant count desc, jaccard asc)
    cases_ordered = sorted(
        [
            {
                "question_id": res["question_id"],
                "top8_identical": res["s20_vs_h40_comparison"]["top8_identical"],
                "tail_entrant_count": res["s20_vs_h40_comparison"]["tail_entrant_count"],
                "overlap_count": res["s20_vs_h40_comparison"]["top8_overlap_count"],
                "jaccard": res["s20_vs_h40_comparison"]["top8_jaccard"],
                "entrant_fused_ranks": [e["fused_rank"] for e in res["tail_entrants"]],
            }
            for res in case_results
        ],
        key=lambda x: (-x["tail_entrant_count"], x["jaccard"], x["question_id"]),
    )

    return {
        "total_case_count": total_cases,
        "identical_top8_cases": identical_cases,
        "changed_top8_cases": changed_cases,
        "total_tail_entrants": len(all_entrants),
        "cases_with_tail_entrants": cases_with_entrants,
        "tail_entrants_per_changed_case": {
            "mean": round(fmean(entrant_counts_changed), 4) if entrant_counts_changed else 0.0,
            "median": median(entrant_counts_changed) if entrant_counts_changed else 0,
            "min": min(entrant_counts_changed) if entrant_counts_changed else 0,
            "max": max(entrant_counts_changed) if entrant_counts_changed else 0,
        },
        "entrant_fused_rank_bucket_counts": bucket_counts,
        "top8_overlap": {
            "mean": round(fmean(overlaps), 4) if overlaps else 0.0,
            "median": median(overlaps) if overlaps else 0,
            "min": min(overlaps) if overlaps else 0,
            "max": max(overlaps) if overlaps else 0,
        },
        "top8_jaccard": {
            "mean": round(fmean(jaccards), 4) if jaccards else 0.0,
            "median": round(float(median(jaccards)), 4) if jaccards else 0.0,
            "min": min(jaccards) if jaccards else 0.0,
            "max": max(jaccards) if jaccards else 0.0,
        },
        "document_level_churn_count": doc_churn_total,
        "score_cutoff_margin_distributions": {
            "s20_cutoff_mean": round(fmean(s20_cutoffs), 8) if s20_cutoffs else 0.0,
            "h40_cutoff_mean": round(fmean(h40_cutoffs), 8) if h40_cutoffs else 0.0,
            "entrant_margin_mean": round(fmean(margins), 8) if margins else 0.0,
        },
        "cases_ordered_by_churn": cases_ordered,
    }


# ----------------------------------------------------------------------
# AUDIT PROTOCOL ORCHESTRATOR
# ----------------------------------------------------------------------


def run_candidate_pool_audit_protocol(
    config_path: Path,
    manifest_path: Path,
    questions_path: Path,
    baseline_dir_or_zip: Path,
    output_dir: Path,
    staging_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object], str]:
    """Execute Stage R1 Candidate-Pool / Reranker Mechanics Audit."""
    reasons: list[str] = []
    retrieval_model_error_count = 0
    created_at = datetime.now(UTC).isoformat()
    script_path = Path(__file__)
    execution_commit = resolve_execution_git_commit(script_path)

    # 1. Validate questions input
    questions_sha = sha256_file(questions_path)
    if questions_sha != CANONICAL_SOURCE_QUESTION_SHA256:
        reasons.append(
            f"Questions SHA mismatch: expected {CANONICAL_SOURCE_QUESTION_SHA256}, got {questions_sha}"
        )

    raw_dev = json.loads(questions_path.read_text(encoding="utf-8"))
    if len(raw_dev) != CANONICAL_SOURCE_QUESTION_COUNT:
        reasons.append(
            f"Questions count mismatch: expected {CANONICAL_SOURCE_QUESTION_COUNT}, got {len(raw_dev)}"
        )

    # 2. Validate manifest input
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_ids: list[str] = manifest_data.get("question_ids", [])
    if target_ids != EXPECTED_22_IDS:
        reasons.append(
            f"Target question IDs do not match canonical 22 IDs: expected {EXPECTED_22_IDS}, got {target_ids}"
        )

    # 3. Load & verify B1A.2 baseline
    baseline: FrozenB1A2Baseline | None = None
    try:
        baseline = load_and_verify_b1a2_baseline(baseline_dir_or_zip, EXPECTED_22_IDS)
    except Exception as exc:
        reasons.append(f"B1A.2 baseline verification failed: {exc}")

    # 4. Load config & staging root
    app_config = load_application_config(config_path)
    active_staging_root = staging_root or app_config.artifacts.root_path

    staging_inventory = []
    try:
        staging_inventory = validate_graphless_staging_root(active_staging_root)
    except Exception as exc:
        reasons.append(f"Graphless staging root validation failed: {exc}")

    if reasons:
        invalid_report = {
            "audit_id": "STAGE-R1-CANDIDATE-POOL-AUDIT",
            "verdict": "INVALID_EXPERIMENT",
            "reasons": reasons,
            "code_version": __version__,
            "execution_git_commit": execution_commit,
            "created_at": created_at,
        }
        invalid_decision = {
            "audit_id": "STAGE-R1-CANDIDATE-POOL-AUDIT",
            "verdict": "INVALID_EXPERIMENT",
            "audit_verified": False,
            "h40_promotion_authorized": False,
            "reasons": reasons,
            "code_version": __version__,
            "execution_git_commit": execution_commit,
            "created_at": created_at,
            "summary": {
                "total_cases": 0,
                "seed_prefix_passes": 0,
                "s20_top8_passes": 0,
                "h40_top8_passes": 0,
                "identical_top8_cases": 0,
                "changed_top8_cases": 0,
                "total_tail_entrants": 0,
                "cases_with_tail_entrants": 0,
                "document_level_churn_count": 0,
                "retrieval_model_error_count": retrieval_model_error_count,
            },
        }
        return invalid_report, invalid_decision, "INVALID_EXPERIMENT"

    assert baseline is not None

    # 5. Build pipeline
    pipeline: CandidatePoolAuditPipeline | None = None
    try:
        pipeline = CandidatePoolAuditPipeline(app_config)
    except Exception as exc:
        reasons.append(f"Retrieval stack construction failed: {exc}")
        invalid_report = {
            "audit_id": "STAGE-R1-CANDIDATE-POOL-AUDIT",
            "verdict": "INVALID_EXPERIMENT",
            "reasons": reasons,
            "code_version": __version__,
            "execution_git_commit": execution_commit,
            "created_at": created_at,
        }
        invalid_decision = {
            "audit_id": "STAGE-R1-CANDIDATE-POOL-AUDIT",
            "verdict": "INVALID_EXPERIMENT",
            "audit_verified": False,
            "h40_promotion_authorized": False,
            "reasons": reasons,
            "code_version": __version__,
            "execution_git_commit": execution_commit,
            "created_at": created_at,
            "summary": {
                "total_cases": 0,
                "seed_prefix_passes": 0,
                "s20_top8_passes": 0,
                "h40_top8_passes": 0,
                "identical_top8_cases": 0,
                "changed_top8_cases": 0,
                "total_tail_entrants": 0,
                "cases_with_tail_entrants": 0,
                "document_level_churn_count": 0,
                "retrieval_model_error_count": retrieval_model_error_count,
            },
        }
        return invalid_report, invalid_decision, "INVALID_EXPERIMENT"

    # 6. Execute audit across all 22 questions
    case_results: list[dict[str, object]] = []
    case_metrics_list: list[dict[str, object]] = []

    seed_prefix_passes = 0
    s20_top8_passes = 0
    h40_top8_passes = 0

    for qid in target_ids:
        q_data = raw_dev.get(qid, {})
        q_text = q_data.get("question", "") if isinstance(q_data, dict) else ""
        if not q_text:
            reasons.append(f"Missing question text for question_id '{qid}'")
            continue

        try:
            c_res, c_met, c_reasons = run_case_candidate_pool_audit(
                pipeline, qid, q_text, baseline
            )
            case_results.append(c_res)
            case_metrics_list.append(c_met)

            if c_res["reproduction_gates"]["seed_prefix_match"]:
                seed_prefix_passes += 1
            if (
                c_res["reproduction_gates"]["s20_chunks_match"]
                and c_res["reproduction_gates"]["s20_docs_match"]
                and c_res["reproduction_gates"]["s20_scores_match"]
            ):
                s20_top8_passes += 1
            if (
                c_res["reproduction_gates"]["h40_chunks_match"]
                and c_res["reproduction_gates"]["h40_docs_match"]
                and c_res["reproduction_gates"]["h40_scores_match"]
            ):
                h40_top8_passes += 1

            if c_reasons:
                reasons.extend(c_reasons)
        except Exception as exc:
            retrieval_model_error_count += 1
            reasons.append(f"Case {qid} execution failure: {exc}")

    # 7. Aggregate Analysis
    aggregate_metrics = (
        compute_aggregate_audit_metrics(case_results, case_metrics_list)
        if case_results else {}
    )

    # 8. Decision Determination
    if retrieval_model_error_count > 0 or len(case_results) != EXPECTED_CASE_COUNT:
        verdict = "INVALID_EXPERIMENT"
    elif (
        seed_prefix_passes != EXPECTED_CASE_COUNT
        or s20_top8_passes != EXPECTED_CASE_COUNT
        or h40_top8_passes != EXPECTED_CASE_COUNT
        or aggregate_metrics.get("identical_top8_cases") != 5
        or aggregate_metrics.get("changed_top8_cases") != 17
    ):
        verdict = "CANDIDATE_POOL_DRIFT_DETECTED"
    else:
        verdict = "CANDIDATE_POOL_AUDIT_PASS"

    audit_verified = verdict == "CANDIDATE_POOL_AUDIT_PASS"

    # Build reports
    report = {
        "audit_id": "STAGE-R1-CANDIDATE-POOL-AUDIT",
        "verdict": verdict,
        "audit_verified": audit_verified,
        "h40_promotion_authorized": False,
        "code_version": __version__,
        "execution_git_commit": execution_commit,
        "created_at": created_at,
        "reasons": reasons,
        "reproduction_summary": {
            "expected_cases": EXPECTED_CASE_COUNT,
            "evaluated_cases": len(case_results),
            "seed_prefix_passes": seed_prefix_passes,
            "s20_top8_passes": s20_top8_passes,
            "h40_top8_passes": h40_top8_passes,
            "retrieval_model_error_count": retrieval_model_error_count,
        },
        "aggregate_metrics": aggregate_metrics,
        "baseline_provenance": {
            "baseline_results_sha256": baseline.results_sha256,
            "baseline_execution_commit": baseline.execution_git_commit,
            "baseline_verdict": baseline.decision_verdict,
        },
    }

    decision = {
        "audit_id": "STAGE-R1-CANDIDATE-POOL-AUDIT",
        "verdict": verdict,
        "audit_verified": audit_verified,
        "h40_promotion_authorized": False,
        "code_version": __version__,
        "execution_git_commit": execution_commit,
        "created_at": created_at,
        "summary": {
            "total_cases": len(case_results),
            "seed_prefix_passes": seed_prefix_passes,
            "s20_top8_passes": s20_top8_passes,
            "h40_top8_passes": h40_top8_passes,
            "identical_top8_cases": aggregate_metrics.get("identical_top8_cases", 0),
            "changed_top8_cases": aggregate_metrics.get("changed_top8_cases", 0),
            "total_tail_entrants": aggregate_metrics.get("total_tail_entrants", 0),
            "cases_with_tail_entrants": aggregate_metrics.get("cases_with_tail_entrants", 0),
            "document_level_churn_count": aggregate_metrics.get("document_level_churn_count", 0),
            "retrieval_model_error_count": retrieval_model_error_count,
        },
        "reasons": reasons,
    }

    # 9. Output persistence & Evidence Packaging
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    configs_dir = output_dir / "configs"
    execution_dir = output_dir / "execution"
    baseline_dir = output_dir / "baseline"

    for d in (results_dir, configs_dir, execution_dir, baseline_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save case results JSONL
    case_results_path = results_dir / "candidate_pool_case_results.jsonl"
    with case_results_path.open("w", encoding="utf-8") as f:
        for r in case_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save case metrics JSONL
    case_metrics_path = results_dir / "candidate_pool_case_metrics.jsonl"
    with case_metrics_path.open("w", encoding="utf-8") as f:
        for m in case_metrics_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Save reports
    report_path = results_dir / "candidate_pool_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision_path = results_dir / "candidate_pool_decision_report.json"
    decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Save configs & execution metadata
    (configs_dir / "runtime_config.json").write_text(
        json.dumps(app_config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    (configs_dir / "phase-b1a-graph-routing-cases.json").write_text(
        json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8"
    )
    (execution_dir / "audit_execution_identity.json").write_text(
        json.dumps({
            "code_version": __version__,
            "execution_git_commit": execution_commit,
            "created_at": created_at,
            "questions_path": str(questions_path),
            "questions_sha256": questions_sha,
            "manifest_path": str(manifest_path),
        }, indent=2) + "\n", encoding="utf-8"
    )
    (execution_dir / "graphless_root_inventory.json").write_text(
        json.dumps(staging_inventory, indent=2) + "\n", encoding="utf-8"
    )
    (baseline_dir / "b1a2_baseline_identity.json").write_text(
        json.dumps({
            "baseline_results_sha256": baseline.results_sha256,
            "baseline_execution_commit": baseline.execution_git_commit,
            "baseline_verdict": baseline.decision_verdict,
            "case_count": baseline.case_count,
        }, indent=2) + "\n", encoding="utf-8"
    )

    package_audit_evidence(output_dir)

    return report, decision, verdict


def package_audit_evidence(output_dir: Path) -> Path:
    """Package Stage R1 audit evidence into canonical deterministic ZIP archive."""
    zip_path = output_dir / "candidate-pool-reranker-audit-evidence.zip"
    required_files = [
        "execution/audit_execution_identity.json",
        "baseline/b1a2_baseline_identity.json",
        "execution/graphless_root_inventory.json",
        "configs/runtime_config.json",
        "configs/phase-b1a-graph-routing-cases.json",
        "results/candidate_pool_case_results.jsonl",
        "results/candidate_pool_case_metrics.jsonl",
        "results/candidate_pool_audit_report.json",
        "results/candidate_pool_decision_report.json",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for rel_path_str in required_files:
            file_path = output_dir / rel_path_str
            if not file_path.is_file():
                raise DataValidationError(
                    f"Cannot package audit evidence: missing required file '{rel_path_str}' at {file_path}"
                )
            zip_file.write(file_path, arcname=rel_path_str)

    return zip_path


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for Stage R1 candidate-pool audit."""
    parser = argparse.ArgumentParser(
        description="S20 vs H40 Candidate-Pool / Reranker Mechanics Audit — Stage R1"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to application configuration JSON file",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to case manifest JSON (e.g. phase-b1a-graph-routing-cases.json)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to source questions JSON (e.g. development.json)",
    )
    parser.add_argument(
        "--baseline-evidence-dir",
        type=Path,
        required=True,
        help="Path to frozen B1A.2 evidence directory or evidence ZIP",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory for audit artifacts and evidence package",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="Optional path to graphless staging root (defaults to config artifacts root)",
    )
    return parser


def main() -> int:
    """CLI entrypoint for Stage R1 audit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    report, decision, verdict = run_candidate_pool_audit_protocol(
        config_path=args.config,
        manifest_path=args.manifest,
        questions_path=args.questions,
        baseline_dir_or_zip=args.baseline_evidence_dir,
        output_dir=args.output_dir,
        staging_root=args.staging_root,
    )

    print("\n" + "=" * 60)
    print(f"CANDIDATE-POOL / RERANKER AUDIT VERDICT: {verdict}")
    print("=" * 60)
    print(f"Audit Verified:            {decision.get('audit_verified')}")
    print(f"H40 Promotion Authorized:  {decision.get('h40_promotion_authorized')} (ALWAYS FALSE)")
    if "summary" in decision:
        s = decision["summary"]
        print(f"Total Cases:               {s.get('total_cases')}")
        print(f"Seed Prefix Passes:        {s.get('seed_prefix_passes')} / 22")
        print(f"S20 Top-8 Passes:          {s.get('s20_top8_passes')} / 22")
        print(f"H40 Top-8 Passes:          {s.get('h40_top8_passes')} / 22")
        print(f"Identical Top-8 Cases:     {s.get('identical_top8_cases')} (expected 5)")
        print(f"Changed Top-8 Cases:       {s.get('changed_top8_cases')} (expected 17)")
        print(f"Total Tail Entrants:       {s.get('total_tail_entrants')}")
        print(f"Document Churn Count:      {s.get('document_level_churn_count')}")

    if decision.get("reasons"):
        print("\nReasons / Invariants:")
        for r in decision["reasons"]:
            print(f"  - {r}")
    print("=" * 60 + "\n")

    return 0 if verdict == "CANDIDATE_POOL_AUDIT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
