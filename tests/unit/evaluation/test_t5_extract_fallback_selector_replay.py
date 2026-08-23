"""Comprehensive unit tests for T5-7C fallback selector replay.

Tests are synthetic, portable, and independent of developer-specific paths.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any
import zipfile

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
    render_competition_answer,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas.answering import AnswerResponse, Citation

import scripts.t5_extract_fallback_selector_replay as fsr
import scripts.t5_generator_contract_measurement as scm


# ==========================================================
# 1. TOKENIZATION & N-GRAM TESTS
# ==========================================================


def test_01_unicode_tokenization_and_casefold() -> None:
    """extract_tokens must split Unicode words and lowercase all tokens."""
    text = "Luật Dược: Quy định tại Điều 12, Khoản 3 (Nghị định 54/2017)!"
    tokens = fsr.extract_tokens(text)
    assert "luật" in tokens
    assert "dược" in tokens
    assert "nghị" in tokens
    assert "định" in tokens
    assert "54" in tokens
    assert "2017" in tokens
    assert all(t.islower() or t.isdigit() for t in tokens)


def test_02_contiguous_trigram_set_calculation() -> None:
    """extract_ngrams must return unique contiguous tuples of length n."""
    text = "quy định về quản lý thuốc"
    ngrams = fsr.extract_ngrams(text, n=3)
    assert len(ngrams) == 4
    assert ("quy", "định", "về") in ngrams
    assert ("định", "về", "quản") in ngrams
    assert ("về", "quản", "lý") in ngrams
    assert ("quản", "lý", "thuốc") in ngrams


def test_03_article_title_included_in_evidence_scoring_text() -> None:
    """Evidence article_title must participate in trigram coverage."""
    question = "cách ghi tên cao dược liệu trong tờ hướng dẫn"
    # Evidence text alone does not have "cách ghi tên" but article_title does
    ev_with_title = [
        {
            "evidence_id": "E1",
            "text": "Nội dung quy định chi tiết.",
            "article_title": "Cách ghi tên cao dược liệu",
            "metadata": {"retrieval_rank": 1},
        }
    ]
    ev_without_title = [
        {
            "evidence_id": "E1",
            "text": "Nội dung quy định chi tiết.",
            "article_title": None,
            "metadata": {"retrieval_rank": 1},
        }
    ]
    dec_with = fsr.select_reference_blind_fallback_evidence(
        question=question, selected_evidence=ev_with_title
    )
    dec_without = fsr.select_reference_blind_fallback_evidence(
        question=question, selected_evidence=ev_without_title
    )
    assert dec_with.e1_coverage > dec_without.e1_coverage
    assert dec_with.e1_coverage > 0.0
    assert dec_without.e1_coverage == 0.0


def test_04_empty_or_short_question_yields_zero_coverage() -> None:
    """Questions with fewer than 3 tokens yield 0.0 coverage without error."""
    ev = [{"evidence_id": "E1", "text": "Quy định chi tiết", "metadata": {"retrieval_rank": 1}}]
    dec_empty = fsr.select_reference_blind_fallback_evidence(
        question="", selected_evidence=ev
    )
    assert dec_empty.e1_coverage == 0.0
    assert dec_empty.switched is False

    dec_short = fsr.select_reference_blind_fallback_evidence(
        question="thuốc y", selected_evidence=ev
    )
    assert dec_short.e1_coverage == 0.0
    assert dec_short.switched is False


# ==========================================================
# 2. CANDIDATE SELECTION & TIE-BREAK TESTS
# ==========================================================


def test_05_exact_margin_switches_at_or_above_threshold() -> None:
    """Selector switches away from E1 when best coverage margin >= switch_margin."""
    # Question with 4 trigrams (6 tokens)
    question = "alpha beta gamma delta epsilon zeta"
    # E1 covers 1 trigram: coverage = 0.25
    # E2 covers 2 trigrams: coverage = 0.50 -> margin = 0.25 (>= 0.20)
    ev_list = [
        {"evidence_id": "E1", "text": "alpha beta gamma other words", "metadata": {"retrieval_rank": 1}},
        {"evidence_id": "E2", "text": "gamma delta epsilon zeta other words", "metadata": {"retrieval_rank": 2}},
    ]
    decision = fsr.select_reference_blind_fallback_evidence(
        question=question, selected_evidence=ev_list, switch_margin=0.20
    )
    assert decision.switched is True
    assert decision.selected_evidence_id == "E2"
    assert decision.e1_coverage == pytest.approx(0.25)
    assert decision.selected_coverage == pytest.approx(0.50)
    assert decision.coverage_margin == pytest.approx(0.25)


def test_06_margin_below_threshold_retains_e1() -> None:
    """Selector retains E1 when best coverage margin is strictly below switch_margin."""
    question = "alpha beta gamma delta epsilon zeta eta theta" # 6 trigrams
    # E1 covers 2 trigrams: cov = 2/6 = 0.3333
    # E2 covers 3 trigrams: cov = 3/6 = 0.5000 -> margin = 0.1667 (< 0.20)
    ev_list = [
        {"evidence_id": "E1", "text": "alpha beta gamma delta extra", "metadata": {"retrieval_rank": 1}},
        {"evidence_id": "E2", "text": "delta epsilon zeta eta extra", "metadata": {"retrieval_rank": 2}},
    ]
    decision = fsr.select_reference_blind_fallback_evidence(
        question=question, selected_evidence=ev_list, switch_margin=0.20
    )
    assert decision.switched is False
    assert decision.selected_evidence_id == "E1"
    assert decision.coverage_margin == 0.0


def test_07_equal_coverage_keeps_better_retrieval_rank() -> None:
    """When coverages are equal, evidence with lower retrieval_rank is preferred."""
    question = "alpha beta gamma delta epsilon"
    # E2 has retrieval_rank 2, E3 has retrieval_rank 5, both cover same trigrams
    ev_list = [
        {"evidence_id": "E1", "text": "unrelated tokens here", "metadata": {"retrieval_rank": 1}},
        {"evidence_id": "E2", "text": "alpha beta gamma delta epsilon", "metadata": {"retrieval_rank": 5}},
        {"evidence_id": "E3", "text": "alpha beta gamma delta epsilon", "metadata": {"retrieval_rank": 2}},
    ]
    decision = fsr.select_reference_blind_fallback_evidence(
        question=question, selected_evidence=ev_list
    )
    assert decision.switched is True
    assert decision.selected_evidence_id == "E3" # rank 2 beats rank 5


def test_08_exact_tie_resolves_deterministically_by_list_order() -> None:
    """When coverages and ranks are identical, lower list position breaks the tie."""
    question = "alpha beta gamma delta epsilon"
    ev_list = [
        {"evidence_id": "E1", "text": "unrelated tokens here", "metadata": {"retrieval_rank": 1}},
        {"evidence_id": "E2", "text": "alpha beta gamma delta epsilon", "metadata": {"retrieval_rank": 3}},
        {"evidence_id": "E3", "text": "alpha beta gamma delta epsilon", "metadata": {"retrieval_rank": 3}},
    ]
    decision = fsr.select_reference_blind_fallback_evidence(
        question=question, selected_evidence=ev_list
    )
    assert decision.switched is True
    assert decision.selected_evidence_id == "E2" # idx 1 beats idx 2


def test_09_e1_remains_when_best_is_e1() -> None:
    """When E1 has the highest coverage, it is retained with switched=False."""
    question = "alpha beta gamma delta epsilon"
    ev_list = [
        {"evidence_id": "E1", "text": "alpha beta gamma delta epsilon", "metadata": {"retrieval_rank": 1}},
        {"evidence_id": "E2", "text": "unrelated tokens here", "metadata": {"retrieval_rank": 2}},
    ]
    decision = fsr.select_reference_blind_fallback_evidence(
        question=question, selected_evidence=ev_list
    )
    assert decision.switched is False
    assert decision.selected_evidence_id == "E1"
    assert decision.e1_coverage == pytest.approx(1.0)
    assert decision.coverage_margin == 0.0


# ==========================================================
# 3. REFERENCE FIREWALL & API SAFETY TESTS
# ==========================================================


def test_10_selector_has_no_reference_answer_parameter() -> None:
    """Signature of select_reference_blind_fallback_evidence must NOT accept reference answers."""
    sig = inspect.signature(fsr.select_reference_blind_fallback_evidence)
    param_names = list(sig.parameters.keys())
    assert "reference_answers" not in param_names
    assert "reference" not in param_names
    assert "y_true" not in param_names
    assert "gold" not in param_names


def test_11_candidate_citation_identity_matches_selected_evidence() -> None:
    """Candidate fallback response creates exact Citation matching chosen evidence."""
    ev = {
        "evidence_id": "E3",
        "chunk_id": "chunk_abc123",
        "document_id": "doc_999",
        "text": "Nội dung pháp lý chính xác.",
    }
    resp = AnswerResponse(
        question="q",
        answer=f"[{ev['evidence_id']}] {ev['text']}",
        citations=[Citation(
            evidence_id=ev["evidence_id"],
            chunk_id=ev["chunk_id"],
            document_id=ev["document_id"],
        )],
        warnings=[],
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
    )
    assert resp.citations[0].evidence_id == "E3"
    assert resp.citations[0].chunk_id == "chunk_abc123"
    assert resp.citations[0].document_id == "doc_999"


def test_12_score_facing_rendering_removes_citation_markers() -> None:
    """render_competition_answer must remove [E#] markers before score evaluation."""
    resp = AnswerResponse(
        question="q",
        answer="[E3] Nội dung quy định tại Điều 5.",
        citations=[Citation(evidence_id="E3", chunk_id="c3", document_id="d3")],
        warnings=[],
        insufficient_evidence=False,
        retrieval_strategy="hybrid_rerank",
        trace_id="t1",
    )
    rendered = render_competition_answer(resp)
    assert "[E3]" not in rendered
    assert "Nội dung quy định tại Điều 5." in rendered


def test_13_replay_refuses_wrong_generation_archive_sha(tmp_path: Path) -> None:
    """Runner fails closed if generation archive SHA-256 does not match pinned authority."""
    fake_gen = tmp_path / "fake_gen.tar.gz"
    fake_gen.write_bytes(b"corrupted generation archive")
    runner = fsr.T57CFallbackSelectorReplayRunner(
        generation_archive=fake_gen,
        evidence_archive=tmp_path / "fake_ev.zip",
        scorer_archive=tmp_path / "fake_scorer.zip",
        output_path=tmp_path / "out.json",
    )
    with pytest.raises(ArtifactCompatibilityError, match="T5_7C_GENERATION_ARCHIVE_SHA_MISMATCH"):
        runner.verify_artifact_authorities()


def test_14_replay_refuses_wrong_scorer_archive_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner fails closed if official scorer archive SHA-256 does not match pinned authority."""
    fake_gen = tmp_path / "fake_gen.tar.gz"
    fake_gen.write_bytes(b"x")
    fake_ev = tmp_path / "fake_ev.zip"
    fake_ev.write_bytes(b"y")
    fake_scorer = tmp_path / "fake_scorer.zip"
    fake_scorer.write_bytes(b"corrupted scorer")

    runner = fsr.T57CFallbackSelectorReplayRunner(
        generation_archive=fake_gen,
        evidence_archive=fake_ev,
        scorer_archive=fake_scorer,
        output_path=tmp_path / "out.json",
    )

    def mock_hash(path: Path) -> str:
        if path == fake_gen.resolve():
            return fsr.EXPECTED_GENERATION_ARCHIVE_SHA256
        if path == fake_ev.resolve():
            return fsr.EXPECTED_FAST30_ARCHIVE_SHA256
        return "wrong_scorer_sha"

    monkeypatch.setattr(scm, "compute_file_sha256", mock_hash)
    with pytest.raises(ArtifactCompatibilityError, match="T5_7C_SCORER_ARCHIVE_SHA_MISMATCH"):
        runner.verify_artifact_authorities()


def test_15_empty_selected_evidence_raises_data_validation_error() -> None:
    """Passing an empty sequence to selector raises DataValidationError."""
    with pytest.raises(DataValidationError, match="selected_evidence sequence cannot be empty"):
        fsr.select_reference_blind_fallback_evidence(question="valid question", selected_evidence=[])


def test_16_result_schema_serialization_roundtrip() -> None:
    """T57CReplayResult serializes and deserializes deterministically with strict validation."""
    decision = fsr.SelectionDecision(
        selected_evidence_id="E3",
        e1_coverage=0.142857,
        selected_coverage=0.357143,
        coverage_margin=0.214286,
        switched=True,
        ngram_size=3,
        switch_margin=0.20,
    )
    result = fsr.T57CReplayResult(
        starting_authority="871ba6cea0d25abb27b38c845b51234f2a122e7c",
        generation_archive_sha256="gen_sha",
        fast30_archive_sha256="fast30_sha",
        official_scorer_archive_sha256="scorer_sha",
        official_scoring_py_sha256="scoring_py_sha",
        ngram_size=3,
        switch_margin=0.20,
        reference_blind_selection=True,
        selector_decisions={"89271": decision},
        switch_qids=["89271"],
        control={"rouge_l": 0.483133, "meteor": 0.404694},
        candidate={"rouge_l": 0.525617, "meteor": 0.443634},
        delta={"rouge_l": 0.042484, "meteor": 0.038940},
        per_question_scores={"89271": {"control_rouge": 0.356, "candidate_rouge": 0.637, "delta_rouge": 0.281, "control_meteor": 0.452, "candidate_meteor": 0.434, "delta_meteor": -0.018}},
        decision="PROMISING_FALLBACK_SELECTOR_REPLAY_CONFIRMED",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    serialized = result.model_dump_json(indent=2)
    deserialized = fsr.T57CReplayResult.model_validate(json.loads(serialized))
    assert deserialized.decision == "PROMISING_FALLBACK_SELECTOR_REPLAY_CONFIRMED"
    assert deserialized.selector_decisions["89271"].selected_evidence_id == "E3"
    assert deserialized.delta["rouge_l"] == pytest.approx(0.042484)
