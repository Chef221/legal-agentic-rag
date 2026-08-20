"""Unit tests for Phase B1B graphless equivalence verification tooling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from legal_agentic_rag.configuration import ApplicationConfig
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
from legal_agentic_rag.exceptions import DataValidationError, RetrievalError
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
    EXPECTED_CASE_COUNT,
    FINAL_TOP_K,
    S20_BRANCH_CANDIDATE_DEPTH,
    S20_HYBRID_OUTPUT_LIMIT,
    SCORE_ABS_TOLERANCE,
    CaseResult,
    EvaluatedHit,
    RelationshipCase,
    compare_hit_lists,
    load_relationship_cases,
    normalize_question_text,
    run_b1b_verification,
    sha256_bytes,
    sha256_file,
)


def _make_eval_hit(chunk_id: str, doc_id: str = "doc1", score: float = 0.9, rank: int = 1) -> EvaluatedHit:
    return EvaluatedHit(
        rank=rank,
        chunk_id=chunk_id,
        document_id=doc_id,
        score=score,
        strategy=RetrievalStrategy.HYBRID_RERANK.value,
    )


def test_normalize_question_text() -> None:
    raw = "  Điều 1   Luật  Doanh  nghiệp   \n\t"
    assert normalize_question_text(raw) == "Điều 1 Luật Doanh nghiệp"


def test_sha256_helpers(tmp_path: Path) -> None:
    data = b"hello world"
    assert sha256_bytes(data) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(data)
    assert sha256_file(file_path) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_load_relationship_cases_success(tmp_path: Path) -> None:
    questions_file = tmp_path / "questions.json"
    data = {
        "1": {"question": "Văn bản A sửa đổi bổ sung văn bản B như thế nào?"},
        "2": {"question": "Mức phạt tiền là bao nhiêu?"},
        "3": {"question": "Nghị định này thay thế thông tư nào?"},
    }
    questions_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    cases = load_relationship_cases(questions_file, expected_count=2)
    assert len(cases) == 2
    assert [c.question_id for c in cases] == ["1", "3"]
    assert all(c.detected_intent == QueryIntent.RELATIONSHIP.value for c in cases)


def test_load_relationship_cases_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="does not exist"):
        load_relationship_cases(tmp_path / "missing.json")


def test_load_relationship_cases_invalid_json_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "bad.json"
    file_path.write_text("not json", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Failed to parse"):
        load_relationship_cases(file_path)


def test_load_relationship_cases_count_mismatch_raises(tmp_path: Path) -> None:
    questions_file = tmp_path / "questions.json"
    data = {
        "1": {"question": "Văn bản A sửa đổi bổ sung văn bản B?"},
    }
    questions_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Expected exactly 2 relationship cases, found 1"):
        load_relationship_cases(questions_file, expected_count=2)


def test_compare_hit_lists_identical() -> None:
    hits_a = [_make_eval_hit("c1", "d1", 0.95, 1), _make_eval_hit("c2", "d2", 0.85, 2)]
    hits_b = [_make_eval_hit("c1", "d1", 0.95, 1), _make_eval_hit("c2", "d2", 0.85, 2)]
    matched, diffs = compare_hit_lists(hits_a, hits_b, tolerance=1e-6)
    assert matched is True
    assert diffs == [0.0, 0.0]


def test_compare_hit_lists_different_chunks() -> None:
    hits_a = [_make_eval_hit("c1", "d1", 0.95, 1)]
    hits_b = [_make_eval_hit("c2", "d1", 0.95, 1)]
    matched, _ = compare_hit_lists(hits_a, hits_b)
    assert matched is False


def test_compare_hit_lists_different_documents() -> None:
    hits_a = [_make_eval_hit("c1", "d1", 0.95, 1)]
    hits_b = [_make_eval_hit("c1", "d2", 0.95, 1)]
    matched, _ = compare_hit_lists(hits_a, hits_b)
    assert matched is False


def test_compare_hit_lists_within_tolerance() -> None:
    hits_a = [_make_eval_hit("c1", "d1", 0.9500001, 1)]
    hits_b = [_make_eval_hit("c1", "d1", 0.9500002, 1)]
    matched, diffs = compare_hit_lists(hits_a, hits_b, tolerance=1e-5)
    assert matched is True
    assert len(diffs) == 1
    assert diffs[0] < 1e-5


def test_compare_hit_lists_exceeding_tolerance() -> None:
    hits_a = [_make_eval_hit("c1", "d1", 0.95, 1)]
    hits_b = [_make_eval_hit("c1", "d1", 0.96, 1)]
    matched, diffs = compare_hit_lists(hits_a, hits_b, tolerance=1e-6)
    assert matched is False
    assert diffs == [pytest.approx(0.01)]


def test_case_result_dataclass() -> None:
    res = CaseResult(
        question_id="101",
        normalized_question="test question",
        s20_arm_hits=[_make_eval_hit("c1")],
        relationship_reranker_hits=[_make_eval_hit("c1")],
        h40_arm_hits=[_make_eval_hit("c1")],
        s20_match=True,
        s20_score_diffs=[0.0],
        warnings=[],
    )
    assert res.question_id == "101"
    assert res.s20_match is True
