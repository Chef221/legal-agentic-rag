#!/usr/bin/env python3
"""Phase B1B: Post-Change Graphless Equivalence Verification Tooling.

Verifies that the graphless competition runtime loads cleanly without graph
artifacts, registers `relationship_rerank_search` without `graph_search` in public
descriptors, and reproduces the exact S20 retrieval hits and ranking on the 22
relationship cases authorized by Phase B1A.2 using the frozen B1A.2 evidence baseline.
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
from legal_agentic_rag.tools.retrieval import RetrievalTool

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

        # Locate run summary (MANDATORY per FIX 5)
        summary_path = base_path / "evidence" / "phase_b1a2_run_summary.json"
        if not summary_path.exists():
            summary_path = base_path / "phase_b1a2_run_summary.json"
        if not summary_path.exists():
            raise DataValidationError(
                "Missing mandatory B1A.2 run summary at evidence/phase_b1a2_run_summary.json"
            )

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

        # Verify run summary fields strictly without fallback defaults
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        if "execution_git_commit" not in summary_data:
            raise DataValidationError(
                "B1A.2 run summary missing mandatory 'execution_git_commit' field"
            )
        commit = summary_data["execution_git_commit"]
        if commit != CANONICAL_B1A2_EXECUTION_COMMIT:
            raise DataValidationError(
                f"B1A.2 execution commit mismatch: expected {CANONICAL_B1A2_EXECUTION_COMMIT}, got {commit}"
            )

        if summary_data.get("case_count") != EXPECTED_CASE_COUNT:
            raise DataValidationError(
                f"B1A.2 run summary case_count ({summary_data.get('case_count')}) != {EXPECTED_CASE_COUNT}"
            )

        # Check summary results SHA if recorded
        summary_results_sha = summary_data.get("results_sha256") or summary_data.get("raw_results_sha256")
        if summary_results_sha and summary_results_sha != CANONICAL_B1A2_RESULTS_SHA256:
            raise DataValidationError(
                f"B1A.2 run summary results SHA mismatch: expected {CANONICAL_B1A2_RESULTS_SHA256}, got {summary_results_sha}"
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

    if source_root.resolve() == staging_root.resolve():
        raise ArtifactCompatibilityError(
            "Cannot create staging root: source_root and staging_root resolve to the same path"
        )

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


def validate_graphless_staging_root(staging_root: Path) -> list[dict[str, object]]:
    """Validate an existing graphless staging root and generate its inventory."""
    if not staging_root.exists() or not staging_root.is_dir():
        raise ArtifactCompatibilityError(
            f"Staging root does not exist or is not a directory: {staging_root}"
        )

    if (staging_root / "graph").exists() or (staging_root / "graph").is_symlink():
        raise ArtifactCompatibilityError("Staging root illegally contains 'graph'")
    if (staging_root / "relationships").exists() or (staging_root / "relationships").is_symlink():
        raise ArtifactCompatibilityError("Staging root illegally contains 'relationships'")

    required_dirs = ["legal_chunks", "bm25", "vector"]
    for req in required_dirs:
        if not (staging_root / req).is_dir():
            raise ArtifactCompatibilityError(
                f"Staging root missing required active directory: '{req}'"
            )

    inventory: list[dict[str, object]] = []
    for item in staging_root.iterdir():
        inventory.append({
            "name": item.name,
            "is_symlink": item.is_symlink(),
            "is_dir": item.is_dir(),
            "target_path": str(item.resolve()),
        })

    if not inventory:
        raise ArtifactCompatibilityError("Staging root inventory is empty")

    return inventory


class RecordingBranchRetriever:
    """Observational proxy around a sparse or dense branch retriever."""

    def __init__(self, inner_branch: object) -> None:
        self._inner = inner_branch
        self.recorded_queries: list[RetrievalQuery] = []

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        return getattr(self._inner, "source_artifact_identity")

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.recorded_queries.append(query)
        return getattr(self._inner, "search")(query)


class RecordingCandidateRetriever:
    """Observational proxy around candidate retriever to capture pre-rerank queries/responses."""

    def __init__(self, inner_retriever: HybridRetriever) -> None:
        self._inner = inner_retriever
        self.last_candidate_query: RetrievalQuery | None = None
        self.last_candidate_response: RetrievalResponse | None = None

    @property
    def source_artifact_identity(self) -> tuple[str, str, str]:
        return self._inner.source_artifact_identity

    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        self.last_candidate_query = query
        resp = self._inner.search(query)
        self.last_candidate_response = resp
        return resp


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
    observed_candidate_query: dict[str, object]
    observed_fused_candidate_count: int
    observed_branch_candidate_depth: int
    branch_depth_match: bool
    candidate_query_match: bool
    fusion_limit_match: bool
    final_topk_match: bool
    route_plan_match: bool
    route_plan: list[dict[str, str]]
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
    retrieval_model_error_count = 0

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

    # 3. Create/Validate graphless staging root and build runtime
    app_config = load_application_config(config_path)
    config_root = app_config.artifacts.root_path

    resolved_staging: Path
    if staging_root is not None:
        resolved_staging = staging_root
    else:
        if (config_root / "graph").exists() or (config_root / "relationships").exists():
            resolved_staging = output_dir / "staging" / "graphless_root"
        else:
            resolved_staging = config_root

    inventory: list[dict[str, object]] = []
    runtime: OnlineRuntime | None = None
    runtime_config: ApplicationConfig | None = None

    try:
        if resolved_staging.resolve() == config_root.resolve():
            inventory = validate_graphless_staging_root(resolved_staging)
        else:
            if resolved_staging.exists() and any(resolved_staging.iterdir()):
                inventory = validate_graphless_staging_root(resolved_staging)
            else:
                inventory = create_graphless_staging_root(config_root, resolved_staging)

        (execution_dir / "graphless_root_inventory.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Freeze runtime config targeting graphless root
        runtime_config = app_config.model_copy(
            update={
                "artifacts": app_config.artifacts.model_copy(
                    update={"root_path": resolved_staging}
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

        # Check tool descriptors via public API runtime.tool_descriptors()
        descriptor_names = {d.name.value for d in runtime.tool_descriptors()}
        if "graph_search" in descriptor_names:
            raise RetrievalError("ToolName.GRAPH_SEARCH illegally present in tool descriptors")
        if ToolName.RELATIONSHIP_RERANK_SEARCH.value not in descriptor_names:
            raise RetrievalError("ToolName.RELATIONSHIP_RERANK_SEARCH missing from tool descriptors")

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

    # 5. Build isolated retrieval execution stack with observational candidate recording
    bm25_backend = SQLiteFTS5BM25Backend.load(resolved_staging / "bm25")
    vector_backend = NumpyVectorBackend.load(resolved_staging / "vector")
    embedding_provider = SentenceTransformerEmbeddingProvider(
        runtime_config.offline.embedding,
        runtime_config=runtime_config.online.vector_runtime,
    )
    dense_retriever = DenseRetriever(embedding_provider, vector_backend)

    recording_bm25 = RecordingBranchRetriever(bm25_backend)
    recording_dense = RecordingBranchRetriever(dense_retriever)

    hybrid_retriever = HybridRetriever(
        bm25_retriever=recording_bm25,
        dense_retriever=recording_dense,
        config=runtime_config.online.retrieval,
        query_understanding_config=runtime_config.online.query_understanding,
    )

    recording_candidate = RecordingCandidateRetriever(hybrid_retriever)
    reranker = CrossEncoderReranker(runtime_config.online.reranker)

    relationship_reranker = RelationshipSeedRerankingRetriever(
        candidate_retriever=recording_candidate,
        reranker=reranker,
        retrieval_config=runtime_config.online.retrieval,
        reranker_config=runtime_config.online.reranker,
    )

    rel_tool = RetrievalTool(
        name=ToolName.RELATIONSHIP_RERANK_SEARCH,
        retriever=relationship_reranker,
    )

    qu_service = QueryUnderstandingService(runtime_config.online.query_understanding)
    router = DeterministicStrategyRouter(
        runtime_config.online.agent,
        runtime_config.online.query_understanding,
    )
    registered_tool_names = {d.name for d in runtime.tool_descriptors()}

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
        route_plan_dicts = [
            {"strategy": r.strategy.value, "tool": r.tool_name.value} for r in routes
        ]

        # FIX 3: Check BOTH strategy and tool for exact route pairs
        route_plan_match = (
            len(routes) >= 3
            and routes[0].strategy == RetrievalStrategy.HYBRID_RERANK
            and routes[0].tool_name == ToolName.RELATIONSHIP_RERANK_SEARCH
            and routes[1].strategy == RetrievalStrategy.HYBRID_RERANK
            and routes[1].tool_name == ToolName.RERANK_SEARCH
            and routes[2].strategy == RetrievalStrategy.HYBRID
            and routes[2].tool_name == ToolName.HYBRID_SEARCH
        )
        if not route_plan_match:
            reasons.append(
                f"Protocol failure ({qid}): Route plan mismatch: expected [HYBRID_RERANK:relationship_rerank, HYBRID_RERANK:rerank, HYBRID:hybrid], got {route_plan_dicts}"
            )
            is_invalid = True

        # Clear branch recording before this query
        recording_bm25.recorded_queries.clear()
        recording_dense.recorded_queries.clear()

        # Execute Attempt 1 via RELATIONSHIP_RERANK_SEARCH tool
        try:
            resp: RetrievalResponse = rel_tool.invoke(query)
        except Exception as exc:
            reasons.append(f"Execution model error on case {qid}: {exc}")
            retrieval_model_error_count += 1
            is_invalid = True
            continue

        # FIX 4: Check retrieval:model_error warning
        if any("retrieval:model_error" in str(w) for w in resp.warnings):
            reasons.append(f"Case {qid} produced 'retrieval:model_error' warning")
            retrieval_model_error_count += 1
            is_invalid = True

        # FIX 2: Observational S20 Trace from Candidate Retriever
        observed_cand_q = recording_candidate.last_candidate_query
        observed_cand_resp = recording_candidate.last_candidate_response

        candidate_query_match = (
            observed_cand_q is not None
            and observed_cand_q.top_k == S20_HYBRID_OUTPUT_LIMIT
            and observed_cand_q.candidate_k == S20_BRANCH_CANDIDATE_DEPTH
            and observed_cand_q.requested_strategy == RetrievalStrategy.HYBRID
        )
        if not candidate_query_match:
            reasons.append(
                f"Protocol failure ({qid}): Candidate query invariant mismatch: {observed_cand_q}"
            )
            is_invalid = True

        observed_fused_count = len(observed_cand_resp.hits) if observed_cand_resp else 0
        fusion_limit_match = (observed_fused_count <= S20_HYBRID_OUTPUT_LIMIT)
        if not fusion_limit_match:
            reasons.append(
                f"Protocol failure ({qid}): Observed fused candidate count ({observed_fused_count}) > {S20_HYBRID_OUTPUT_LIMIT}"
            )
            is_invalid = True

        # Branch depth observation (all branches requested depth 40)
        branch_depths = [q.candidate_k for q in recording_bm25.recorded_queries + recording_dense.recorded_queries]
        branch_depth_match = bool(branch_depths) and all(d == S20_BRANCH_CANDIDATE_DEPTH for d in branch_depths)
        if not branch_depth_match:
            reasons.append(
                f"Protocol failure ({qid}): Branch depth mismatch: {branch_depths}"
            )
            is_invalid = True

        final_topk_match = (
            resp.strategy == RetrievalStrategy.HYBRID_RERANK
            and len(resp.hits) <= FINAL_TOP_K
        )
        if not final_topk_match:
            reasons.append(
                f"Protocol failure ({qid}): Final response invariant mismatch (hits={len(resp.hits)}, strategy={resp.strategy})"
            )
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
            observed_candidate_query={
                "top_k": observed_cand_q.top_k if observed_cand_q else None,
                "candidate_k": observed_cand_q.candidate_k if observed_cand_q else None,
                "requested_strategy": observed_cand_q.requested_strategy.value if observed_cand_q and observed_cand_q.requested_strategy else None,
            },
            observed_fused_candidate_count=observed_fused_count,
            observed_branch_candidate_depth=S20_BRANCH_CANDIDATE_DEPTH,
            branch_depth_match=branch_depth_match,
            candidate_query_match=candidate_query_match,
            fusion_limit_match=fusion_limit_match,
            final_topk_match=final_topk_match,
            route_plan_match=route_plan_match,
            route_plan=route_plan_dicts,
            warnings=resp.warnings,
        )
        case_results.append(case_res)

        case_record = {
            "question_id": qid,
            "route_plan": route_plan_dicts,
            "observed_candidate_query": case_res.observed_candidate_query,
            "observed_fused_candidate_count": observed_fused_count,
            "observed_branch_candidate_depth": S20_BRANCH_CANDIDATE_DEPTH,
            "b1b_final_hits": [asdict(h) for h in b1b_hits],
            "b1a2_expected_hits": [asdict(h) for h in expected_hits],
            "chunks_match": chunks_match,
            "docs_match": docs_match,
            "scores_match": scores_match,
            "max_score_diff": max_diff if max_diff != float("inf") else None,
            "is_equivalent": is_equiv,
            "warnings": resp.warnings,
        }
        case_results_lines.append(json.dumps(case_record, ensure_ascii=False))

        case_metric = {
            "question_id": qid,
            "is_equivalent": is_equiv,
            "chunks_match": chunks_match,
            "docs_match": docs_match,
            "scores_match": scores_match,
            "route_plan_match": route_plan_match,
            "candidate_query_match": candidate_query_match,
            "fusion_limit_match": fusion_limit_match,
            "branch_depth_match": branch_depth_match,
            "final_topk_match": final_topk_match,
            "max_score_diff": max_diff if max_diff != float("inf") else None,
        }
        case_metrics_lines.append(json.dumps(case_metric, ensure_ascii=False))

        _LOGGER.info(
            "Case [%d/%d] %s: equivalent=%s (chunks=%s, docs=%s, scores=%s, fused_cands=%d, max_diff=%.7f)",
            idx + 1,
            len(cases),
            qid,
            is_equiv,
            chunks_match,
            docs_match,
            scores_match,
            observed_fused_count,
            max_diff,
        )

    # Write retrieval results jsonl and case metrics jsonl
    (results_dir / "phase_b1b_retrieval_results.jsonl").write_bytes(
        ("\n".join(case_results_lines) + "\n").encode("utf-8")
    )
    (results_dir / "phase_b1b_case_metrics.jsonl").write_bytes(
        ("\n".join(case_metrics_lines) + "\n").encode("utf-8")
    )

    # 6. Aggregate invariant counts
    branch_depth_40_passes = sum(1 for cr in case_results if cr.branch_depth_match)
    candidate_query_invariant_passes = sum(1 for cr in case_results if cr.candidate_query_match)
    fusion_limit_passes = sum(1 for cr in case_results if cr.fusion_limit_match)
    final_topk_passes = sum(1 for cr in case_results if cr.final_topk_match)
    route_plan_passes = sum(1 for cr in case_results if cr.route_plan_match)
    matching_count = sum(1 for cr in case_results if cr.is_equivalent)
    chunk_matches_count = sum(1 for cr in case_results if cr.chunks_match)
    doc_matches_count = sum(1 for cr in case_results if cr.docs_match)
    score_passes_count = sum(1 for cr in case_results if cr.scores_match)

    # 7. Evaluate verdict
    if is_invalid or len(case_results) != len(cases) or retrieval_model_error_count > 0:
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
        elif (
            matching_count == len(cases)
            and branch_depth_40_passes == len(cases)
            and candidate_query_invariant_passes == len(cases)
            and fusion_limit_passes == len(cases)
            and final_topk_passes == len(cases)
            and route_plan_passes == len(cases)
        ):
            reasons.append(
                f"All {len(cases)} relationship cases match frozen B1A.2 S20 hits exactly on chunk IDs, "
                f"document IDs, and reranker scores (tolerance <= {SCORE_ABS_TOLERANCE}) with 100% invariant passes."
            )
            verdict = "B1B_EQUIVALENCE_PASS"
        else:
            verdict = "INVALID_EXPERIMENT"

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
        "retrieval_model_error_count": retrieval_model_error_count,
        "equivalence_summary": {
            "exact_matches_count": matching_count,
            "chunk_sequence_matches_count": chunk_matches_count,
            "document_sequence_matches_count": doc_matches_count,
            "score_tolerance_passes_count": score_passes_count,
            "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        },
        "aggregate_protocol_counts": {
            "branch_depth_40_passes": branch_depth_40_passes,
            "candidate_query_invariant_passes": candidate_query_invariant_passes,
            "fusion_limit_passes": fusion_limit_passes,
            "final_topk_passes": final_topk_passes,
            "route_plan_passes": route_plan_passes,
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
        "retrieval_model_error_count": retrieval_model_error_count,
        "summary": {
            "exact_matches": f"{matching_count}/{len(cases)}",
            "score_passes": f"{score_passes_count}/{len(cases)}",
            "branch_depth_passes": f"{branch_depth_40_passes}/{len(cases)}",
            "candidate_query_passes": f"{candidate_query_invariant_passes}/{len(cases)}",
            "fusion_limit_passes": f"{fusion_limit_passes}/{len(cases)}",
            "final_topk_passes": f"{final_topk_passes}/{len(cases)}",
            "route_plan_passes": f"{route_plan_passes}/{len(cases)}",
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
        "b1a2_results_sha256": b1a2_baseline.results_sha256,
        "b1a2_execution_commit": b1a2_baseline.execution_git_commit,
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
    print(f"Exact matches:          {decision.get('summary', {}).get('exact_matches', 'N/A')}")
    print(f"Score passes:           {decision.get('summary', {}).get('score_passes', 'N/A')}")
    print(f"Branch depth passes:    {decision.get('summary', {}).get('branch_depth_passes', 'N/A')}")
    print(f"Candidate query passes: {decision.get('summary', {}).get('candidate_query_passes', 'N/A')}")
    print(f"Fusion limit passes:    {decision.get('summary', {}).get('fusion_limit_passes', 'N/A')}")
    print(f"Route plan passes:      {decision.get('summary', {}).get('route_plan_passes', 'N/A')}")
    print(f"Evidence ZIP:           {zip_path}")
    print(f"Evidence SHA:           {zip_sha}")
    print(f"Evidence Size:          {zip_size} bytes")
    print("========================================================\n")


if __name__ == "__main__":
    main()
