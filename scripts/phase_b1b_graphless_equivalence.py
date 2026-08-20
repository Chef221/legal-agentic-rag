#!/usr/bin/env python3
"""Phase B1B: Post-Change Graphless Equivalence Verification Tooling.

Verifies that the graphless competition runtime loads cleanly without graph
artifacts, registers `relationship_rerank_search` without `graph_search`, and
reproduces the exact S20 retrieval hits and ranking on the 22 relationship
cases authorized by Phase B1A.2.
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
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.runtime.online import OnlineRuntime, OnlineRuntimeFactory
from legal_agentic_rag.runtime.startup_validation import (
    validate_competition_artifact_lineage,
    validate_startup_report,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import (
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


@dataclass(frozen=True, slots=True)
class RelationshipCase:
    """A verified relationship question for equivalence evaluation."""

    question_id: str
    raw_question: str
    normalized_question: str
    detected_intent: str


def load_relationship_cases(
    questions_path: Path,
    expected_count: int = EXPECTED_CASE_COUNT,
) -> list[RelationshipCase]:
    """Load and filter relationship cases deterministically from questions json."""
    if not questions_path.exists():
        raise DataValidationError(
            f"Questions file does not exist: {questions_path}"
        )

    content_bytes = questions_path.read_bytes()
    file_hash = sha256_bytes(content_bytes)
    try:
        raw_data = json.loads(content_bytes.decode("utf-8"))
    except Exception as error:
        raise DataValidationError(
            f"Failed to parse questions JSON from {questions_path}: {error}"
        ) from error

    if not isinstance(raw_data, dict):
        raise DataValidationError(
            f"Questions file must be a JSON object, got {type(raw_data).__name__}"
        )

    qu_service = QueryUnderstandingService()
    cases: list[RelationshipCase] = []

    for qid, payload in raw_data.items():
        if not isinstance(payload, dict) or "question" not in payload:
            continue
        raw_q = payload["question"]
        if not isinstance(raw_q, str):
            continue

        norm_q = normalize_question_text(raw_q)
        analysis = qu_service.analyze(norm_q)

        if analysis.intent == QueryIntent.RELATIONSHIP:
            cases.append(
                RelationshipCase(
                    question_id=str(qid),
                    raw_question=raw_q,
                    normalized_question=norm_q,
                    detected_intent=analysis.intent.value,
                )
            )

    cases.sort(key=lambda c: c.question_id)

    _LOGGER.info(
        "relationship_cases_loaded",
        extra={
            "total_questions_in_file": len(raw_data),
            "file_sha256": file_hash,
            "relationship_cases_found": len(cases),
            "expected_count": expected_count,
        },
    )

    if expected_count is not None and len(cases) != expected_count:
        raise DataValidationError(
            f"Expected exactly {expected_count} relationship cases, "
            f"found {len(cases)} (file={questions_path}, sha256={file_hash})"
        )

    return cases


@dataclass(frozen=True, slots=True)
class EvaluatedHit:
    """Comparable hit attributes for post-change equivalence."""

    rank: int
    chunk_id: str
    document_id: str
    score: float
    strategy: str

    @classmethod
    def from_retrieval_hit(cls, hit: RetrievalHit, rank: int) -> EvaluatedHit:
        return cls(
            rank=rank,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            score=round(hit.score, 8),
            strategy=hit.strategy.value,
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Per-case evaluation output for B1B verification."""

    question_id: str
    normalized_question: str
    s20_arm_hits: list[EvaluatedHit]
    relationship_reranker_hits: list[EvaluatedHit]
    h40_arm_hits: list[EvaluatedHit]
    s20_match: bool
    s20_score_diffs: list[float]
    warnings: list[str]


def compare_hit_lists(
    expected: Sequence[EvaluatedHit],
    actual: Sequence[EvaluatedHit],
    tolerance: float = SCORE_ABS_TOLERANCE,
) -> tuple[bool, list[float]]:
    """Compare two hit lists for identical chunk/doc IDs and score tolerance."""
    if len(expected) != len(actual):
        return False, []

    score_diffs: list[float] = []
    for exp, act in zip(expected, actual, strict=True):
        if exp.chunk_id != act.chunk_id:
            return False, score_diffs
        if exp.document_id != act.document_id:
            return False, score_diffs
        diff = abs(exp.score - act.score)
        score_diffs.append(diff)
        if diff > tolerance:
            return False, score_diffs

    return True, score_diffs


def run_b1b_verification(
    config: ApplicationConfig,
    questions_path: Path,
    output_dir: Path,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> dict[str, object]:
    """Execute complete Phase B1B post-change verification protocol."""
    started_at = datetime.now(UTC)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load cases
    cases = load_relationship_cases(
        questions_path, expected_count=expected_case_count
    )

    # 2. Build OnlineRuntime via factory to test startup without graph
    runtime_factory = OnlineRuntimeFactory(config)
    runtime = runtime_factory.build()

    # 3. Verify tool capabilities
    tool_names = set(runtime.registry.descriptors().keys())
    if ToolName.GRAPH_SEARCH.value in tool_names:
        raise RetrievalError("graph_search must NOT be present in online runtime tools")
    if ToolName.RELATIONSHIP_RERANK_SEARCH.value not in tool_names:
        raise RetrievalError("relationship_rerank_search MUST be present in online runtime tools")

    # 4. Verify registered strategy router planning for relationship intent
    qu_service = QueryUnderstandingService(config.online.query_understanding)
    for case in cases:
        analysis = qu_service.analyze(case.normalized_question)
        query = RetrievalQuery(
            text=case.normalized_question,
            top_k=FINAL_TOP_K,
            candidate_k=S20_BRANCH_CANDIDATE_DEPTH,
            query_analysis=analysis,
        )
        routes = runtime.workflow._router.plan(
            query,
            set(runtime.registry.tools.keys()),
        )
        if len(routes) < 2:
            raise RetrievalError(f"Router produced fewer than 2 routes for case {case.question_id}")
        if routes[0].tool_name != ToolName.RELATIONSHIP_RERANK_SEARCH:
            raise RetrievalError(
                f"Attempt 1 route for relationship query must be relationship_rerank_search, got {routes[0].tool_name}"
            )
        if routes[1].tool_name != ToolName.RERANK_SEARCH:
            raise RetrievalError(
                f"Attempt 2 route for relationship query must be rerank_search, got {routes[1].tool_name}"
            )

    # 5. Evaluate all cases across arms: S20 specification vs runtime relationship_reranker
    reranker = runtime_factory._reranker
    reranker_config = config.online.reranker
    bm25 = SQLiteFTS5BM25Backend(
        config.offline.bm25,
        runtime_config=config.online.bm25_runtime,
    )
    bm25_manifest = load_artifact_manifest(
        config.artifacts.directory("bm25_directory"),
        expected_type=ArtifactType.BM25_INDEX,
    )
    bm25.load(config.artifacts.directory("bm25_directory"), bm25_manifest)

    vector = NumpyVectorBackend(
        config.offline.vector_index,
        runtime_config=config.online.vector_runtime,
        serving_metadata_source=config.artifacts.directory("vector_serving_directory"),
    )
    vector_manifest = load_artifact_manifest(
        config.artifacts.directory("vector_directory"),
        expected_type=ArtifactType.VECTOR_INDEX,
    )
    vector.load(config.artifacts.directory("vector_directory"), vector_manifest)

    dense = DenseRetriever(runtime_factory._embedding_provider, vector)
    hybrid = HybridRetriever(
        bm25,
        dense,
        config.online.retrieval,
        config.online.query_understanding,
    )
    h40_reranker = RerankingRetriever(hybrid, reranker, reranker_config)

    case_results: list[CaseResult] = []
    all_s20_matches = True
    max_score_diff = 0.0

    for case in cases:
        analysis = qu_service.analyze(case.normalized_question)
        base_query = RetrievalQuery(
            text=case.normalized_question,
            top_k=FINAL_TOP_K,
            candidate_k=S20_BRANCH_CANDIDATE_DEPTH,
            query_analysis=analysis,
        )

        # Direct S20 baseline calculation
        s20_candidate_query = base_query.model_copy(
            update={
                "top_k": S20_HYBRID_OUTPUT_LIMIT,
                "requested_strategy": RetrievalStrategy.HYBRID,
            }
        )
        s20_candidate_resp = hybrid.search(s20_candidate_query)
        s20_rerank_resp = h40_reranker.rerank_candidates(
            base_query.model_copy(update={"requested_strategy": RetrievalStrategy.HYBRID_RERANK}),
            s20_candidate_resp,
        )
        s20_hits = [
            EvaluatedHit.from_retrieval_hit(hit, idx + 1)
            for idx, hit in enumerate(s20_rerank_resp.hits[:FINAL_TOP_K])
        ]

        # Runtime relationship reranker search
        rel_resp = runtime.retriever.search_relationship_rerank(base_query)
        rel_hits = [
            EvaluatedHit.from_retrieval_hit(hit, idx + 1)
            for idx, hit in enumerate(rel_resp.hits[:FINAL_TOP_K])
        ]

        # H40 diagnostic search
        h40_resp = h40_reranker.search(base_query)
        h40_hits = [
            EvaluatedHit.from_retrieval_hit(hit, idx + 1)
            for idx, hit in enumerate(h40_resp.hits[:FINAL_TOP_K])
        ]

        # Compare S20 vs relationship_reranker
        matched, diffs = compare_hit_lists(s20_hits, rel_hits, SCORE_ABS_TOLERANCE)
        if not matched:
            all_s20_matches = False
        if diffs:
            max_score_diff = max(max_score_diff, max(diffs))

        case_results.append(
            CaseResult(
                question_id=case.question_id,
                normalized_question=case.normalized_question,
                s20_arm_hits=s20_hits,
                relationship_reranker_hits=rel_hits,
                h40_arm_hits=h40_hits,
                s20_match=matched,
                s20_score_diffs=diffs,
                warnings=rel_resp.warnings,
            )
        )

    completed_at = datetime.now(UTC)
    duration_seconds = (completed_at - started_at).total_seconds()

    report: dict[str, object] = {
        "protocol": "PHASE_B1B_GRAPHLESS_EQUIVALENCE",
        "version": __version__,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": duration_seconds,
        "cases_evaluated": len(case_results),
        "all_s20_matches": all_s20_matches,
        "max_score_diff": max_score_diff,
        "tolerance": SCORE_ABS_TOLERANCE,
        "verdict": "VERIFIED_EQUIVALENT" if all_s20_matches else "EQUIVALENCE_FAILED",
        "tools_registered": list(tool_names),
        "manifests_loaded": list(runtime.manifests.keys()),
        "case_details": [
            {
                "question_id": r.question_id,
                "s20_match": r.s20_match,
                "max_score_diff": max(r.s20_score_diffs) if r.s20_score_diffs else 0.0,
                "hits_count": len(r.relationship_reranker_hits),
            }
            for r in case_results
        ],
    }

    report_file = output_dir / "phase_b1b_verification_report.json"
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _LOGGER.info(
        "b1b_verification_report_written",
        extra={
            "report_path": str(report_file),
            "verdict": report["verdict"],
            "all_s20_matches": all_s20_matches,
        },
    )
    return report


def main() -> None:
    """CLI entrypoint for Phase B1B post-change verification."""
    parser = argparse.ArgumentParser(
        description="Phase B1B Graphless Equivalence Verification Tool"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/application.json"),
        help="Path to application configuration JSON file.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/public-official.json"),
        help="Path to questions JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/b1b_verification"),
        help="Directory to save verification report.",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=EXPECTED_CASE_COUNT,
        help="Expected number of relationship cases to find.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    app_config = load_application_config(args.config)
    report = run_b1b_verification(
        app_config,
        args.questions,
        args.output_dir,
        expected_case_count=args.expected_cases,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
