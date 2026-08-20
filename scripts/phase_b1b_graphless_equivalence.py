#!/usr/bin/env python3
"""Phase B1B: Post-Change Graphless Equivalence Verification Tooling.

Verifies that the graphless competition runtime loads cleanly without graph
artifacts, registers `relationship_rerank_search` without `graph_search`, and
reproduces the exact S20 retrieval hits and ranking on the 22 relationship
cases authorized by Phase B1A.2 using the frozen B1A.2 evidence baseline.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import shutil
from statistics import fmean
import subprocess
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
    FixedRetriever,
    HybridRetriever,
    QueryUnderstandingService,
    RelationshipSeedRerankingRetriever,
    RerankingRetriever,
)
from legal_agentic_rag.agent.router import (
    DeterministicStrategyRouter,
    RetrievalRoute,
)
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.runtime.online import OnlineRuntime, OnlineRuntimeFactory
from legal_agentic_rag.runtime.startup_validation import (
    validate_competition_artifact_lineage,
    validate_startup_report,
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
from legal_agentic_rag.serving.config_loader import load_application_config

_LOGGER = logging.getLogger(__name__)

SCORE_ABS_TOLERANCE = 1e-6
EXPECTED_CASE_COUNT = 22
CANONICAL_SOURCE_QUESTION_COUNT = 991
CANONICAL_SOURCE_QUESTION_SHA256 = (
    "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"
)
CANONICAL_B1A2_RESULTS_SHA256 = (
    "51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a"
)
CANONICAL_B1A2_EXECUTION_COMMIT = "9265f3dadcf1ef0170f0abe618519da1657fc55e"

EXPECTED_22_IDS: list[str] = [
    "102047",
    "107487",
    "110287",
    "111905",
    "113537",
    "122659",
    "125393",
    "133075",
    "134605",
    "147239",
    "147869",
    "150051",
    "26541",
    "29491",
    "29877",
    "39671",
    "45219",
    "47537",
    "48905",
    "64035",
    "95861",
    "99639",
]

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


@dataclass(frozen=True, slots=True)
class EvaluatedHit:
    """A single hit serialized for equivalence evaluation."""

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
    expected_s20_hits: dict[str, list[EvaluatedHit]]


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
        import tempfile
        temp_unpack_dir = Path(tempfile.mkdtemp(prefix="b1a2_unpacked_"))
        with zipfile.ZipFile(baseline_dir_or_zip, "r") as zip_ref:
            zip_ref.extractall(temp_unpack_dir)
        base_path = temp_unpack_dir
    else:
        base_path = baseline_dir_or_zip

    try:
        # Locate results file
        results_path = base_path / "results" / "phase_b1a2_retrieval_results.jsonl"
        if not results_path.exists():
            results_path = base_path / "phase_b1a2_retrieval_results.jsonl"
        if not results_path.exists():
            raise DataValidationError(
                f"Missing B1A.2 results jsonl at {results_path}"
            )

        # Locate decision report
        decision_path = base_path / "results" / "phase_b1a2_decision_report.json"
        if not decision_path.exists():
            decision_path = base_path / "phase_b1a2_decision_report.json"
        if not decision_path.exists():
            raise DataValidationError(
                f"Missing B1A.2 decision report at {decision_path}"
            )

        # Locate run summary / baseline identity
        summary_path = base_path / "evidence" / "phase_b1a2_run_summary.json"
        if not summary_path.exists():
            summary_path = base_path / "phase_b1a2_run_summary.json"
        if not summary_path.exists():
            summary_path = base_path / "baseline" / "b1a2_baseline_identity.json"

        # Verify results SHA256
        actual_results_sha = sha256_file(results_path)
        if actual_results_sha != CANONICAL_B1A2_RESULTS_SHA256:
            raise DataValidationError(
                f"B1A.2 results SHA256 mismatch: expected {CANONICAL_B1A2_RESULTS_SHA256}, got {actual_results_sha}"
            )

        # Verify decision report
        decision_data = json.loads(decision_path.read_text(encoding="utf-8"))
        verdict = decision_data.get("verdict")
        if verdict != "GRAPH_REDUNDANCY_PROVEN":
            raise DataValidationError(
                f"B1A.2 verdict must be 'GRAPH_REDUNDANCY_PROVEN', got '{verdict}'"
            )

        # Verify execution commit from summary if present
        commit = CANONICAL_B1A2_EXECUTION_COMMIT
        if summary_path.exists():
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            commit = summary_data.get("execution_git_commit", commit)
            if commit != CANONICAL_B1A2_EXECUTION_COMMIT:
                raise DataValidationError(
                    f"B1A.2 execution commit mismatch: expected {CANONICAL_B1A2_EXECUTION_COMMIT}, got {commit}"
                )

        # Load S20 hits per question
        lines = [
            line.strip()
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != len(expected_ids):
            raise DataValidationError(
                f"B1A.2 results count ({len(lines)}) != expected case count ({len(expected_ids)})"
            )

        expected_hits: dict[str, list[EvaluatedHit]] = {}
        for line in lines:
            record = json.loads(line)
            qid = str(record["question_id"])
            s20_hits_raw = record.get("s20_arm", {}).get("final_hits", [])
            expected_hits[qid] = [
                EvaluatedHit(
                    rank=h["rank"],
                    chunk_id=h["chunk_id"],
                    document_id=h["document_id"],
                    score=float(h["score"]),
                    strategy=h["strategy"],
                )
                for h in s20_hits_raw
            ]

        # Verify all expected IDs exist
        for expected_id in expected_ids:
            if expected_id not in expected_hits:
                raise DataValidationError(
                    f"B1A.2 baseline missing expected question ID '{expected_id}'"
                )

        return FrozenB1A2Baseline(
            decision_verdict=verdict,
            results_sha256=actual_results_sha,
            execution_git_commit=commit,
            case_count=len(expected_hits),
            expected_s20_hits=expected_hits,
        )
    finally:
        if temp_unpack_dir is not None and temp_unpack_dir.exists():
            shutil.rmtree(temp_unpack_dir, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class CaseMaterialization:
    """Materialized case from manifest and benchmark questions."""

    question_id: str
    raw_question: str
    normalized_question: str


def materialize_cases_from_benchmark(
    manifest_path: Path,
    questions_path: Path,
) -> tuple[list[CaseMaterialization], str, str]:
    """Validate manifest and questions benchmark, then materialize the 22 cases."""
    if not manifest_path.exists():
        raise DataValidationError(f"Manifest path does not exist: {manifest_path}")
    if not questions_path.exists():
        raise DataValidationError(f"Questions benchmark path does not exist: {questions_path}")

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    manifest_data = json.loads(manifest_bytes.decode("utf-8"))

    case_ids: list[str] = [str(x) for x in manifest_data.get("question_ids", [])]
    if len(case_ids) != EXPECTED_CASE_COUNT:
        raise DataValidationError(
            f"Case manifest must have exactly {EXPECTED_CASE_COUNT} IDs, got {len(case_ids)}"
        )
    if len(set(case_ids)) != len(case_ids):
        raise DataValidationError("Case manifest contains duplicate IDs")
    if case_ids != EXPECTED_22_IDS:
        raise DataValidationError(
            f"Case manifest IDs/order mismatch: expected {EXPECTED_22_IDS}, got {case_ids}"
        )

    questions_bytes = questions_path.read_bytes()
    questions_sha = sha256_bytes(questions_bytes)
    if questions_sha != CANONICAL_SOURCE_QUESTION_SHA256:
        raise DataValidationError(
            f"Canonical development.json SHA mismatch: expected {CANONICAL_SOURCE_QUESTION_SHA256}, got {questions_sha}"
        )

    raw_questions_data = json.loads(questions_bytes.decode("utf-8"))
    if not isinstance(raw_questions_data, dict):
        raise DataValidationError("development.json must be a JSON object mapping IDs to payloads")
    if len(raw_questions_data) != CANONICAL_SOURCE_QUESTION_COUNT:
        raise DataValidationError(
            f"development.json question count ({len(raw_questions_data)}) != {CANONICAL_SOURCE_QUESTION_COUNT}"
        )

    cases: list[CaseMaterialization] = []
    for qid in case_ids:
        if qid not in raw_questions_data:
            raise DataValidationError(f"Question ID '{qid}' not found in development.json")
        payload = raw_questions_data[qid]
        if not isinstance(payload, dict) or "question" not in payload:
            raise DataValidationError(f"Question ID '{qid}' missing 'question' string field")
        raw_q = payload["question"]
        if not isinstance(raw_q, str) or not raw_q.strip():
            raise DataValidationError(f"Question ID '{qid}' has invalid or empty question text")
        norm_q = normalize_question_text(raw_q)
        cases.append(
            CaseMaterialization(
                question_id=qid,
                raw_question=raw_q,
                normalized_question=norm_q,
            )
        )

    return cases, manifest_sha, questions_sha


def create_graphless_staging_root(
    source_root: Path,
    staging_root: Path,
) -> list[dict[str, object]]:
    """Create an immutable graphless staging root exposing only B1B competition artifacts."""
    if not source_root.exists():
        raise DataValidationError(f"Source artifact root does not exist: {source_root}")

    staging_root.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []

    for item in source_root.iterdir():
        if item.name in ("graph", "relationships"):
            continue
        dest = staging_root / item.name
        if dest.exists() or dest.is_symlink():
            continue
        is_symlink = False
        try:
            os.symlink(item.resolve(), dest, target_is_directory=item.is_dir())
            is_symlink = True
        except (OSError, NotImplementedError):
            if item.is_dir():
                shutil.copytree(item, dest, symlinks=True)
            else:
                shutil.copy2(item, dest)
        inventory.append({
            "name": item.name,
            "is_symlink": is_symlink,
            "is_dir": item.is_dir(),
            "source_path": str(item.resolve()),
        })

    # Hard assert graph and relationships are absent
    if (staging_root / "graph").exists() or (staging_root / "graph").is_symlink():
        raise ArtifactCompatibilityError("Staging root illegally contains 'graph'")
    if (staging_root / "relationships").exists() or (staging_root / "relationships").is_symlink():
        raise ArtifactCompatibilityError("Staging root illegally contains 'relationships'")

    return inventory


@dataclass(frozen=True, slots=True)
class CaseEquivalenceResult:
    """Equivalence comparison result for a single case."""

    question_id: str
    normalized_question: str
    b1b_hits: list[EvaluatedHit]
    b1a2_expected_hits: list[EvaluatedHit]
    chunks_match: bool
    docs_match: bool
    scores_match: bool
    score_diffs: list[float]
    max_score_diff: float
    is_equivalent: bool
    observed_candidate_count: int
    route_plan: list[str]
    warnings: list[str]


def compare_hit_lists(
    actual: Sequence[EvaluatedHit],
    expected: Sequence[EvaluatedHit],
    tolerance: float = SCORE_ABS_TOLERANCE,
) -> tuple[bool, bool, bool, list[float], float]:
    """Compare hit sequences on chunks, documents, and score tolerance."""
    actual_chunks = [h.chunk_id for h in actual]
    expected_chunks = [h.chunk_id for h in expected]
    chunks_match = (actual_chunks == expected_chunks)

    actual_docs = [h.document_id for h in actual]
    expected_docs = [h.document_id for h in expected]
    docs_match = (actual_docs == expected_docs)

    score_diffs: list[float] = []
    if len(actual) == len(expected):
        for act_h, exp_h in zip(actual, expected):
            score_diffs.append(abs(act_h.score - exp_h.score))
        max_diff = max(score_diffs) if score_diffs else 0.0
        scores_match = all(d <= tolerance for d in score_diffs)
    else:
        max_diff = float("inf")
        scores_match = False

    return chunks_match, docs_match, scores_match, score_diffs, max_diff


def run_b1b_verification_protocol(
    config_path: Path,
    manifest_path: Path,
    questions_path: Path,
    baseline_dir_or_zip: Path,
    output_dir: Path,
    staging_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object], str]:
    """Execute the full B1B post-change verification protocol and determine mechanical verdict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_dir = output_dir / "execution"
    baseline_dir = output_dir / "baseline"
    configs_dir = output_dir / "configs"
    results_dir = output_dir / "results"

    execution_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    reasons: list[str] = []
    is_invalid = False
    verdict = "INVALID_EXPERIMENT"

    # Resolve execution commit & script SHA
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    try:
        execution_commit = resolve_execution_git_commit(script_path)
    except Exception as exc:
        reasons.append(f"Protocol failure: Cannot resolve execution git commit ({exc})")
        execution_commit = "unknown"
        is_invalid = True

    # 1. Load baseline B1A.2 evidence
    b1a2_baseline: FrozenB1A2Baseline | None = None
    try:
        b1a2_baseline = load_and_verify_b1a2_baseline(baseline_dir_or_zip, EXPECTED_22_IDS)
        baseline_identity = {
            "baseline_experiment_id": "PHASE-B1A.2",
            "decision_verdict": b1a2_baseline.decision_verdict,
            "results_sha256": b1a2_baseline.results_sha256,
            "execution_git_commit": b1a2_baseline.execution_git_commit,
            "case_count": b1a2_baseline.case_count,
        }
        (baseline_dir / "b1a2_baseline_identity.json").write_text(
            json.dumps(baseline_identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        reasons.append(f"Baseline provenance failure: {exc}")
        is_invalid = True

    # 2. Materialize exact 22 cases
    cases: list[CaseMaterialization] = []
    manifest_sha = ""
    questions_sha = ""
    try:
        cases, manifest_sha, questions_sha = materialize_cases_from_benchmark(
            manifest_path, questions_path
        )
        shutil.copy2(manifest_path, configs_dir / "phase-b1a-graph-routing-cases.json")
    except Exception as exc:
        reasons.append(f"Case materialization failure: {exc}")
        is_invalid = True

    # 3. Create graphless staging root and build runtime
    app_config = load_application_config(config_path)
    source_root = app_config.artifacts.root_path
    if staging_root is None:
        staging_root = output_dir / "staging" / "graphless_root"

    inventory: list[dict[str, object]] = []
    runtime: OnlineRuntime | None = None
    runtime_config: ApplicationConfig | None = None

    try:
        inventory = create_graphless_staging_root(source_root, staging_root)
        (execution_dir / "graphless_root_inventory.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Freeze runtime config targeting graphless root
        runtime_config = app_config.model_copy(
            update={
                "artifacts": app_config.artifacts.model_copy(
                    update={"root_path": staging_root}
                ),
                "online": app_config.online.model_copy(
                    update={
                        "retrieval": app_config.online.retrieval.model_copy(
                            update={
                                "top_k": FINAL_TOP_K,
                                "candidate_k": S20_BRANCH_CANDIDATE_DEPTH,
                                "relationship_rerank_fusion_k": S20_HYBRID_OUTPUT_LIMIT,
                            }
                        )
                    }
                ),
            }
        )

        (configs_dir / "runtime_config.json").write_text(
            json.dumps(runtime_config.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        runtime = OnlineRuntimeFactory(runtime_config).build()

        # Check runtime manifests
        actual_manifest_types = set(runtime.manifests.keys())
        expected_manifest_types = {
            ArtifactType.LEGAL_CHUNKS.value,
            ArtifactType.BM25_INDEX.value,
            ArtifactType.VECTOR_INDEX.value,
        }
        if actual_manifest_types != expected_manifest_types:
            raise ArtifactCompatibilityError(
                f"OnlineRuntime manifests mismatch: expected {expected_manifest_types}, got {actual_manifest_types}"
            )

        # Check tool registry
        descriptor_names = {d.name.value for d in runtime.tool_registry.descriptors()}
        if "graph_search" in descriptor_names:
            raise RetrievalError("ToolName.GRAPH_SEARCH illegally present in tool registry")
        if ToolName.RELATIONSHIP_RERANK_SEARCH.value not in descriptor_names:
            raise RetrievalError("ToolName.RELATIONSHIP_RERANK_SEARCH missing from tool registry")

    except Exception as exc:
        reasons.append(f"Runtime startup / artifact compatibility failure: {exc}")
        is_invalid = True

    # 4. If invalid before execution, record decision and exit
    if is_invalid or runtime is None or b1a2_baseline is None or runtime_config is None:
        report = {
            "experiment_id": "PHASE-B1B",
            "verdict": "INVALID_EXPERIMENT",
            "reasons": reasons,
            "code_version": __version__,
            "execution_git_commit": execution_commit,
            "created_at": datetime.now(UTC).isoformat(),
        }
        decision_report = {
            "experiment_id": "PHASE-B1B",
            "verdict": "INVALID_EXPERIMENT",
            "reasons": reasons,
            "b1b_verified": False,
        }
        (results_dir / "phase_b1b_equivalence_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (results_dir / "phase_b1b_decision_report.json").write_text(
            json.dumps(decision_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report, decision_report, "INVALID_EXPERIMENT"

    # 5. Execute 22 cases
    qu_service = QueryUnderstandingService(runtime_config.online.query_understanding)
    router = DeterministicStrategyRouter(
        runtime_config.online.agent,
        runtime_config.online.query_understanding,
    )
    registered_tool_names = {d.name for d in runtime.tool_registry.descriptors()}
    rel_tool = runtime.tool_registry.get(ToolName.RELATIONSHIP_RERANK_SEARCH)

    case_results: list[CaseEquivalenceResult] = []
    case_results_lines: list[str] = []
    case_metrics_lines: list[str] = []

    for idx, case in enumerate(cases):
        qid = case.question_id
        analysis = qu_service.analyze(case.normalized_question)
        if analysis.intent != QueryIntent.RELATIONSHIP:
            reasons.append(
                f"Protocol failure ({qid}): Query intent ({analysis.intent}) != RELATIONSHIP"
            )
            is_invalid = True

        query = RetrievalQuery(
            query_id=qid,
            original_question=case.raw_question,
            normalized_question=case.normalized_question,
            top_k=FINAL_TOP_K,
            candidate_k=S20_BRANCH_CANDIDATE_DEPTH,
            requested_strategy=None,
            query_analysis=analysis,
        )

        routes = router.plan(query, registered_tool_names)
        route_str_list = [f"{r.strategy.value}:{r.tool_name.value}" for r in routes]

        # Verify route ordering
        if len(routes) < 3:
            reasons.append(f"Protocol failure ({qid}): Router planned < 3 attempts: {route_str_list}")
            is_invalid = True
        elif (
            routes[0].tool_name != ToolName.RELATIONSHIP_RERANK_SEARCH
            or routes[1].tool_name != ToolName.RERANK_SEARCH
            or routes[2].tool_name != ToolName.HYBRID_SEARCH
        ):
            reasons.append(
                f"Protocol failure ({qid}): Route order mismatch: expected [relationship_rerank, rerank, hybrid], got {route_str_list}"
            )
            is_invalid = True

        # Execute Attempt 1 via RELATIONSHIP_RERANK_SEARCH tool
        try:
            resp: RetrievalResponse = rel_tool.invoke(query)
        except Exception as exc:
            reasons.append(f"Execution model error on case {qid}: {exc}")
            is_invalid = True
            continue

        if resp.strategy != RetrievalStrategy.HYBRID_RERANK:
            reasons.append(f"Protocol failure ({qid}): Strategy ({resp.strategy}) != HYBRID_RERANK")
            is_invalid = True

        if len(resp.hits) > FINAL_TOP_K:
            reasons.append(f"Protocol failure ({qid}): Hits count ({len(resp.hits)}) > {FINAL_TOP_K}")
            is_invalid = True

        b1b_hits = [
            EvaluatedHit(
                rank=h.rank,
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                score=float(h.score),
                strategy=h.strategy.value,
            )
            for h in resp.hits
        ]

        expected_hits = b1a2_baseline.expected_s20_hits.get(qid, [])
        (
            chunks_match,
            docs_match,
            scores_match,
            score_diffs,
            max_diff,
        ) = compare_hit_lists(b1b_hits, expected_hits, tolerance=SCORE_ABS_TOLERANCE)

        is_equiv = chunks_match and docs_match and scores_match

        # Extract candidate count if available from trace
        observed_cands = len(b1b_hits)

        case_res = CaseEquivalenceResult(
            question_id=qid,
            normalized_question=case.normalized_question,
            b1b_hits=b1b_hits,
            b1a2_expected_hits=expected_hits,
            chunks_match=chunks_match,
            docs_match=docs_match,
            scores_match=scores_match,
            score_diffs=score_diffs,
            max_score_diff=max_diff,
            is_equivalent=is_equiv,
            observed_candidate_count=observed_cands,
            route_plan=route_str_list,
            warnings=resp.warnings,
        )
        case_results.append(case_res)

        case_record = {
            "question_id": qid,
            "normalized_question": case.normalized_question,
            "b1b_final_hits": [asdict(h) for h in b1b_hits],
            "b1a2_expected_hits": [asdict(h) for h in expected_hits],
            "chunks_match": chunks_match,
            "docs_match": docs_match,
            "scores_match": scores_match,
            "max_score_diff": max_diff if max_diff != float("inf") else None,
            "is_equivalent": is_equiv,
            "route_plan": route_str_list,
            "warnings": resp.warnings,
        }
        case_results_lines.append(json.dumps(case_record, ensure_ascii=False))

        case_metric = {
            "question_id": qid,
            "is_equivalent": is_equiv,
            "chunks_match": chunks_match,
            "docs_match": docs_match,
            "scores_match": scores_match,
            "max_score_diff": max_diff if max_diff != float("inf") else None,
        }
        case_metrics_lines.append(json.dumps(case_metric, ensure_ascii=False))

        _LOGGER.info(
            "Case [%d/%d] %s: equivalent=%s (chunks=%s, docs=%s, scores=%s, max_diff=%.7f)",
            idx + 1,
            len(cases),
            qid,
            is_equiv,
            chunks_match,
            docs_match,
            scores_match,
            max_diff,
        )

    # Write retrieval results jsonl and case metrics jsonl
    (results_dir / "phase_b1b_retrieval_results.jsonl").write_bytes(
        ("\n".join(case_results_lines) + "\n").encode("utf-8")
    )
    (results_dir / "phase_b1b_case_metrics.jsonl").write_bytes(
        ("\n".join(case_metrics_lines) + "\n").encode("utf-8")
    )

    # 6. Evaluate verdict
    if is_invalid or len(case_results) != len(cases):
        verdict = "INVALID_EXPERIMENT"
    else:
        mismatches: list[str] = []
        for cr in case_results:
            if not cr.chunks_match:
                mismatches.append(f"Case {cr.question_id}: Chunk ID sequences differ from frozen B1A.2 S20")
            if not cr.docs_match:
                mismatches.append(f"Case {cr.question_id}: Document ID sequences differ from frozen B1A.2 S20")
            if not cr.scores_match:
                mismatches.append(
                    f"Case {cr.question_id}: Reranker score diff ({cr.max_score_diff}) > {SCORE_ABS_TOLERANCE}"
                )

        if mismatches:
            reasons.extend(mismatches)
            verdict = "B1B_EQUIVALENCE_FAIL"
        else:
            reasons.append(
                f"All {len(cases)} relationship cases match frozen B1A.2 S20 hits exactly on chunk IDs, "
                f"document IDs, and reranker scores (tolerance <= {SCORE_ABS_TOLERANCE}) against graphless runtime root."
            )
            verdict = "B1B_EQUIVALENCE_PASS"

    matching_count = sum(1 for cr in case_results if cr.is_equivalent)
    chunk_matches_count = sum(1 for cr in case_results if cr.chunks_match)
    doc_matches_count = sum(1 for cr in case_results if cr.docs_match)
    score_passes_count = sum(1 for cr in case_results if cr.scores_match)

    report = {
        "experiment_id": "PHASE-B1B",
        "code_version": __version__,
        "execution_git_commit": execution_commit,
        "script_sha256": script_sha,
        "manifest_sha256": manifest_sha,
        "development_sha256": questions_sha,
        "b1a2_baseline_results_sha256": b1a2_baseline.results_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(case_results),
        "verdict": verdict,
        "reasons": reasons,
        "equivalence_summary": {
            "exact_matches_count": matching_count,
            "chunk_sequence_matches_count": chunk_matches_count,
            "document_sequence_matches_count": doc_matches_count,
            "score_tolerance_passes_count": score_passes_count,
            "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        },
        "invariants": {
            "top_k": FINAL_TOP_K,
            "candidate_k": S20_BRANCH_CANDIDATE_DEPTH,
            "relationship_rerank_fusion_k": S20_HYBRID_OUTPUT_LIMIT,
            "graph_search_tool_present": False,
            "relationship_rerank_search_tool_present": True,
            "online_manifests": sorted(list(runtime.manifests.keys())),
        },
    }

    decision_report = {
        "experiment_id": "PHASE-B1B",
        "verdict": verdict,
        "reasons": reasons,
        "b1b_verified": (verdict == "B1B_EQUIVALENCE_PASS"),
        "summary": {
            "exact_matches": f"{matching_count}/{len(cases)}",
            "score_passes": f"{score_passes_count}/{len(cases)}",
        },
    }

    (results_dir / "phase_b1b_equivalence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "phase_b1b_decision_report.json").write_text(
        json.dumps(decision_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Persist execution identity
    execution_identity = {
        "experiment_id": "PHASE-B1B",
        "code_version": __version__,
        "execution_git_commit": execution_commit,
        "script_sha256": script_sha,
        "manifest_sha256": manifest_sha,
        "development_sha256": questions_sha,
        "runtime_config_sha256": sha256_file(configs_dir / "runtime_config.json"),
    }
    (execution_dir / "b1b_execution_identity.json").write_text(
        json.dumps(execution_identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return report, decision_report, verdict


def package_b1b_evidence(
    output_zip_path: Path,
    output_dir: Path,
) -> tuple[str, int]:
    """Package verification evidence into canonical zip archive."""
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path == output_zip_path:
                    continue
                # Expose only evidence directories
                rel_path = file_path.relative_to(output_dir)
                if any(rel_path.parts[0] == p for p in ("execution", "baseline", "configs", "results")):
                    zip_file.write(file_path, arcname=str(rel_path).replace("\\", "/"))

    zip_sha = sha256_file(output_zip_path)
    zip_size = output_zip_path.stat().st_size
    return zip_sha, zip_size


def main() -> None:
    """CLI entrypoint for Phase B1B verification tooling."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Phase B1B Graphless Equivalence Verification Tooling")
    parser.add_argument("--config", "-c", type=Path, required=True, help="Path to base ApplicationConfig JSON")
    parser.add_argument("--manifest", "-m", type=Path, default=Path("configs/phase-b1a-graph-routing-cases.json"), help="Path to case manifest")
    parser.add_argument("--questions", "-q", type=Path, required=True, help="Path to development.json")
    parser.add_argument("--baseline-evidence-dir", "--b1a2-dir", type=Path, required=True, help="Path to B1A.2 baseline evidence directory or zip")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("artifacts/b1b_verification"), help="Output directory")
    parser.add_argument("--staging-root", type=Path, default=None, help="Optional custom graphless staging root")

    args = parser.parse_args()

    report, decision, verdict = run_b1b_verification_protocol(
        config_path=args.config,
        manifest_path=args.manifest,
        questions_path=args.questions,
        baseline_dir_or_zip=args.baseline_evidence_dir,
        output_dir=args.output_dir,
        staging_root=args.staging_root,
    )

    zip_path = args.output_dir / "phase-b1b-graphless-equivalence-evidence.zip"
    zip_sha, zip_size = package_b1b_evidence(zip_path, args.output_dir)

    print("\n========================================================")
    print(f"PHASE B1B POST-CHANGE EQUIVALENCE VERDICT: {verdict}")
    print("========================================================")
    print(f"Exact matches: {decision.get('summary', {}).get('exact_matches', 'N/A')}")
    print(f"Evidence ZIP:  {zip_path}")
    print(f"Evidence SHA:  {zip_sha}")
    print(f"Evidence Size: {zip_size} bytes")
    print("========================================================\n")


if __name__ == "__main__":
    main()
