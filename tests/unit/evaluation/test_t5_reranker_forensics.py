"""Unit tests for T5-5A Targeted Reranker Causal Investigation Tooling (Final Authority-Hardened)."""

import inspect
from pathlib import Path
import zipfile
import pytest
from scripts.t5_reranker_forensics import (
    EXPECTED_FAST30_ARCHIVE_SHA256,
    EXPECTED_Q134499_BEST_PRE,
    EXPECTED_Q60281_BEST_PRE,
    DeployableCandidateMetadata,
    DiagnosticOracleCandidate,
    RerankForensicClassification,
    RerankForensicPacket,
    apply_authority_bound_forensic_annotations,
    build_rerank_forensic_packet,
    compute_oracle_overlap_f1,
    evaluate_tune_reranker_policy_discovery,
    extract_candidate_metadata,
    load_fast30_rerank_forensics,
)


def test_retained_prerank_score_remains_retrieval_space():
    """Retained candidate retrieval_score remains in retrieval space (0.02), not reranker logit."""
    pre_cand = {"chunk_id": "c-1", "document_id": "doc-1", "score": 0.02, "rrf_score": 0.02, "text": "t"}
    post_hit = {"chunk_id": "c-1", "score": 5.9609, "text": "t"}
    meta = extract_candidate_metadata(pre_cand, pre_rank=1, post_hit=post_hit, post_rank=1)
    assert meta.retrieval_score == 0.02
    assert meta.reranker_score == 5.9609


def test_retained_reranker_score_comes_from_matched_post_hit():
    """Retained candidate reranker_score is taken directly from matched post hit."""
    pre_cand = {"chunk_id": "c-1", "score": 0.015, "text": "t"}
    post_hit = {"chunk_id": "c-1", "score": 6.28125, "text": "t"}
    meta = extract_candidate_metadata(pre_cand, pre_rank=5, post_hit=post_hit, post_rank=1)
    assert meta.reranker_score == 6.28125
    assert meta.post_rerank_rank == 1


def test_builder_separates_retrieval_and_reranker_scores_for_retained():
    """build_rerank_forensic_packet produces correct score separation for retained candidates."""
    record = {
        "question_id": "q-retained",
        "question": "Câu hỏi",
        "reference_answer": "Đáp án",
        "terminal_retrieval_hits": [{"chunk_id": "c-1", "score": 5.9609}],
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-1", "score": 5.9609, "text": "t"}],
                "pre_rerank_candidates": [{"chunk_id": "c-1", "score": 0.02, "rrf_score": 0.02, "text": "t"}],
            }
        ],
    }
    packet = build_rerank_forensic_packet(record, "Tune20")
    assert packet.candidates[0].retrieval_score == 0.02
    assert packet.candidates[0].reranker_score == 5.9609


def test_prerank_trace_reranker_score_not_trusted_as_current_event():
    """Stale reranker_score on pre-candidate is NOT used if candidate is dropped."""
    pre_cand = {
        "chunk_id": "c-dropped",
        "score": 0.0149,
        "rrf_score": 0.0149,
        "retrieval_trace": {"reranker_score": 99.9},
        "text": "t",
    }
    meta = extract_candidate_metadata(pre_cand, pre_rank=6, post_hit=None, post_rank=None)
    assert meta.retrieval_score == 0.0149
    assert meta.reranker_score is None


def test_dropped_current_reranker_score_remains_none():
    """Dropped candidate reranker_score is None."""
    pre_cand = {"chunk_id": "c-drop", "score": 0.016, "text": "t"}
    meta = extract_candidate_metadata(pre_cand, pre_rank=2, post_hit=None, post_rank=None)
    assert meta.reranker_score is None
    assert meta.post_rerank_rank is None


def test_dropped_relation_is_at_or_below_cutoff():
    """Dropped candidate relation is AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED."""
    record = {
        "question_id": "q-drop",
        "question": "Câu hỏi",
        "reference_answer": "Thủ tục",
        "terminal_retrieval_hits": [{"chunk_id": "c-post", "score": 4.9453}],
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-post", "score": 4.9453, "text": "t1"}],
                "pre_rerank_candidates": [
                    {"chunk_id": "c-post", "score": 0.02, "text": "t1"},
                    {"chunk_id": "c-drop", "score": 0.016, "text": "Thủ tục cấp phiếu lý lịch"},
                ],
            }
        ],
    }
    packet = build_rerank_forensic_packet(record, "Holdout10")
    assert packet.is_oracle_proxy_drop is True
    assert packet.dropped_score_relation == "AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED"
    assert packet.dropped_candidate_score is None
    assert packet.score_margin_to_cutoff is None


def test_equal_score_cutoff_represented_safely():
    """Equal score cutoff scenario handled without claiming strict lower logit."""
    record = {
        "question_id": "q-equal",
        "question": "Câu hỏi",
        "reference_answer": "Thủ tục",
        "terminal_retrieval_hits": [{"chunk_id": "c-1", "score": 5.0}],
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-1", "score": 5.0, "text": "t1"}],
                "pre_rerank_candidates": [
                    {"chunk_id": "c-1", "score": 0.02, "text": "t1"},
                    {"chunk_id": "c-2", "score": 0.015, "text": "Thủ tục cấp phiếu lý lịch"},
                ],
            }
        ],
    }
    packet = build_rerank_forensic_packet(record, "Holdout10")
    assert packet.lowest_retained_score == 5.0
    assert packet.dropped_candidate_score is None
    assert packet.dropped_score_relation == "AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED"


def test_oracle_proxy_drop_does_not_mean_semantic_loss():
    """Generic build_rerank_forensic_packet sets ORACLE_PROXY_RERANK_DROP, not semantic loss."""
    record = {
        "question_id": "q-gen",
        "question": "Câu hỏi",
        "reference_answer": "Thủ tục cấp phiếu",
        "terminal_retrieval_hits": [{"chunk_id": "c-post", "score": 5.0}],
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-post", "score": 5.0, "text": "Khác"}],
                "pre_rerank_candidates": [
                    {"chunk_id": "c-post", "score": 0.02, "text": "Khác"},
                    {"chunk_id": "c-drop", "score": 0.015, "text": "Thủ tục cấp phiếu lý lịch"},
                ],
            }
        ],
    }
    packet = build_rerank_forensic_packet(record, "Tune20")
    assert packet.is_oracle_proxy_drop is True
    assert packet.forensic_classification == RerankForensicClassification.ORACLE_PROXY_RERANK_DROP


def test_authority_bound_q134499_annotation_succeeds():
    """Test A: Exact authority + Holdout10 + Q134499 + expected chunk + proxy flag => SEMANTICALLY_PLAUSIBLE_RERANK_LOSS."""
    base_pkt = RerankForensicPacket(
        question_id="134499", question="q", reference_answer="a", split="Holdout10",
        mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=True,
        forensic_classification=RerankForensicClassification.ORACLE_PROXY_RERANK_DROP,
        best_pre_chunk_id=EXPECTED_Q134499_BEST_PRE, best_pre_f1=0.835, best_pre_rank=6,
        best_post_chunk_id="c-top", best_post_f1=0.643, post_top1_chunk_id="c-top1",
        post_top1_f1=0.578, f1_loss_gap=0.192, post_top1_score=5.96, lowest_retained_score=4.94,
        dropped_candidate_score=None, dropped_score_relation="AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED",
        score_margin_to_cutoff=None, top1_to_cutoff_margin=1.02, candidates=[], oracle_evals=[],
    )
    annotated = apply_authority_bound_forensic_annotations(
        base_pkt,
        actual_archive_sha256=EXPECTED_FAST30_ARCHIVE_SHA256,
    )
    assert annotated.forensic_classification == RerankForensicClassification.SEMANTICALLY_PLAUSIBLE_RERANK_LOSS


def test_authority_bound_q60281_annotation_succeeds():
    """Test B: Exact authority + Q60281 expected forensic identity => ORACLE_PROXY_FALSE_POSITIVE."""
    base_pkt = RerankForensicPacket(
        question_id="60281", question="q", reference_answer="a", split="Holdout10",
        mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=True,
        forensic_classification=RerankForensicClassification.ORACLE_PROXY_RERANK_DROP,
        best_pre_chunk_id=EXPECTED_Q60281_BEST_PRE, best_pre_f1=0.630, best_pre_rank=16,
        best_post_chunk_id="c-top", best_post_f1=0.543, post_top1_chunk_id="c-top1",
        post_top1_f1=0.226, f1_loss_gap=0.087, post_top1_score=6.28, lowest_retained_score=4.90,
        dropped_candidate_score=None, dropped_score_relation="AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED",
        score_margin_to_cutoff=None, top1_to_cutoff_margin=1.38, candidates=[], oracle_evals=[],
    )
    annotated = apply_authority_bound_forensic_annotations(
        base_pkt,
        actual_archive_sha256=EXPECTED_FAST30_ARCHIVE_SHA256,
    )
    assert annotated.forensic_classification == RerankForensicClassification.ORACLE_PROXY_FALSE_POSITIVE


def test_authority_mismatch_sha_omits_annotation():
    """Test C: Wrong actual archive SHA => semantic annotation NOT attached."""
    base_pkt = RerankForensicPacket(
        question_id="134499", question="q", reference_answer="a", split="Holdout10",
        mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=True,
        forensic_classification=RerankForensicClassification.ORACLE_PROXY_RERANK_DROP,
        best_pre_chunk_id=EXPECTED_Q134499_BEST_PRE, best_pre_f1=0.835, best_pre_rank=6,
        best_post_chunk_id="c-top", best_post_f1=0.643, post_top1_chunk_id="c-top1",
        post_top1_f1=0.578, f1_loss_gap=0.192, post_top1_score=5.96, lowest_retained_score=4.94,
        dropped_candidate_score=None, dropped_score_relation="AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED",
        score_margin_to_cutoff=None, top1_to_cutoff_margin=1.02, candidates=[], oracle_evals=[],
    )
    annotated = apply_authority_bound_forensic_annotations(
        base_pkt,
        actual_archive_sha256="0000000000000000000000000000000000000000000000000000000000000000",
    )
    assert annotated.forensic_classification == RerankForensicClassification.ORACLE_PROXY_RERANK_DROP


def test_authority_mismatch_chunk_id_omits_annotation():
    """Test D: Correct SHA but wrong best_pre_chunk_id => annotation NOT attached."""
    base_pkt = RerankForensicPacket(
        question_id="134499", question="q", reference_answer="a", split="Holdout10",
        mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=True,
        forensic_classification=RerankForensicClassification.ORACLE_PROXY_RERANK_DROP,
        best_pre_chunk_id="chunk_unexpected", best_pre_f1=0.835, best_pre_rank=6,
        best_post_chunk_id="c-top", best_post_f1=0.643, post_top1_chunk_id="c-top1",
        post_top1_f1=0.578, f1_loss_gap=0.192, post_top1_score=5.96, lowest_retained_score=4.94,
        dropped_candidate_score=None, dropped_score_relation="AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED",
        score_margin_to_cutoff=None, top1_to_cutoff_margin=1.02, candidates=[], oracle_evals=[],
    )
    annotated = apply_authority_bound_forensic_annotations(
        base_pkt,
        actual_archive_sha256=EXPECTED_FAST30_ARCHIVE_SHA256,
    )
    assert annotated.forensic_classification == RerankForensicClassification.ORACLE_PROXY_RERANK_DROP


def test_authority_mismatch_split_omits_annotation():
    """Test E: Correct SHA/chunk but wrong split => annotation NOT attached."""
    base_pkt = RerankForensicPacket(
        question_id="134499", question="q", reference_answer="a", split="Tune20",
        mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=True,
        forensic_classification=RerankForensicClassification.ORACLE_PROXY_RERANK_DROP,
        best_pre_chunk_id=EXPECTED_Q134499_BEST_PRE, best_pre_f1=0.835, best_pre_rank=6,
        best_post_chunk_id="c-top", best_post_f1=0.643, post_top1_chunk_id="c-top1",
        post_top1_f1=0.578, f1_loss_gap=0.192, post_top1_score=5.96, lowest_retained_score=4.94,
        dropped_candidate_score=None, dropped_score_relation="AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED",
        score_margin_to_cutoff=None, top1_to_cutoff_margin=1.02, candidates=[], oracle_evals=[],
    )
    annotated = apply_authority_bound_forensic_annotations(
        base_pkt,
        actual_archive_sha256=EXPECTED_FAST30_ARCHIVE_SHA256,
    )
    assert annotated.forensic_classification == RerankForensicClassification.ORACLE_PROXY_RERANK_DROP


def test_authority_mismatch_proxy_drop_flag_omits_annotation():
    """Test F: Correct SHA/chunk/split but is_oracle_proxy_drop=False => annotation NOT attached."""
    base_pkt = RerankForensicPacket(
        question_id="134499", question="q", reference_answer="a", split="Holdout10",
        mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=False,
        forensic_classification=RerankForensicClassification.NOT_ASSESSABLE,
        best_pre_chunk_id=EXPECTED_Q134499_BEST_PRE, best_pre_f1=0.835, best_pre_rank=6,
        best_post_chunk_id="c-top", best_post_f1=0.643, post_top1_chunk_id="c-top1",
        post_top1_f1=0.578, f1_loss_gap=0.192, post_top1_score=5.96, lowest_retained_score=4.94,
        dropped_candidate_score=None, dropped_score_relation="AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED",
        score_margin_to_cutoff=None, top1_to_cutoff_margin=1.02, candidates=[], oracle_evals=[],
    )
    annotated = apply_authority_bound_forensic_annotations(
        base_pkt,
        actual_archive_sha256=EXPECTED_FAST30_ARCHIVE_SHA256,
    )
    assert annotated.forensic_classification == RerankForensicClassification.NOT_ASSESSABLE


def test_caller_cannot_spoof_archive_authority(tmp_path: Path):
    """Test G: Prove load_fast30_rerank_forensics computes SHA from bytes and cannot be spoofed."""
    sig = inspect.signature(load_fast30_rerank_forensics)
    assert list(sig.parameters.keys()) == ["zip_path"], "Loader must only accept zip_path"
    
    # Create dummy zip with different bytes
    fake_zip = tmp_path / "fake.zip"
    with zipfile.ZipFile(fake_zip, "w") as z:
        z.writestr("diagnostics.jsonl", "")
        
    packets = load_fast30_rerank_forensics(fake_zip)
    assert len(packets) == 0


def test_semantic_annotation_without_oracle_proxy_triggers_further_investigation():
    """Semantic loss annotation even with is_oracle_proxy_drop=False causes helper to request investigation."""
    p_tune = RerankForensicPacket(
        question_id="q-1", question="q", reference_answer="a", split="Tune20",
        mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=False,
        forensic_classification=RerankForensicClassification.SEMANTICALLY_PLAUSIBLE_RERANK_LOSS,
        best_pre_chunk_id="c-1", best_pre_f1=0.35, best_pre_rank=1, best_post_chunk_id="c-2",
        best_post_f1=0.30, post_top1_chunk_id="c-2", post_top1_f1=0.30, f1_loss_gap=0.05,
        post_top1_score=5.0, lowest_retained_score=5.0, dropped_candidate_score=None,
        dropped_score_relation="AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED",
        score_margin_to_cutoff=None, top1_to_cutoff_margin=0.0, candidates=[], oracle_evals=[],
    )
    res = evaluate_tune_reranker_policy_discovery([p_tune])
    assert res["tune_semantically_plausible_losses"] == 1
    assert res["decision"] == "FURTHER_CAUSAL_INVESTIGATION_REQUIRED"


def test_zero_proxy_and_zero_semantic_yields_no_policy_justified():
    """Zero proxy drops and zero semantic losses yields NO_RERANK_POLICY_JUSTIFIED."""
    packets = [
        RerankForensicPacket(
            question_id=f"q-{i}", question="q", reference_answer="a", split="Tune20",
            mapping_status="UNIQUELY_MAPPED", is_oracle_proxy_drop=False,
            forensic_classification=RerankForensicClassification.NOT_ASSESSABLE,
            best_pre_chunk_id=f"c-{i}", best_pre_f1=0.5, best_pre_rank=1, best_post_chunk_id=f"c-{i}",
            best_post_f1=0.5, post_top1_chunk_id=f"c-{i}", post_top1_f1=0.5, f1_loss_gap=0.0,
            post_top1_score=5.0, lowest_retained_score=5.0, dropped_candidate_score=None,
            dropped_score_relation="NO_CANDIDATE_DROPPED",
            score_margin_to_cutoff=None, top1_to_cutoff_margin=0.0, candidates=[], oracle_evals=[],
        )
        for i in range(20)
    ]
    res = evaluate_tune_reranker_policy_discovery(packets)
    assert res["tune_oracle_proxy_drops"] == 0
    assert res["tune_semantically_plausible_losses"] == 0
    assert res["decision"] == "NO_RERANK_POLICY_JUSTIFIED"


def test_holdout_only_discovery_rejected():
    """Holdout-only discovery rejected."""
    p_holdout = build_rerank_forensic_packet(
        {"question_id": "q-21", "question": "Q", "reference_answer": "A"},
        "Holdout10",
    )
    with pytest.raises(ValueError, match="Holdout10 contains contaminated forensic seeds"):
        evaluate_tune_reranker_policy_discovery([p_holdout])


def test_mixed_split_discovery_rejected():
    """Mixed Tune/Holdout discovery rejected."""
    p_tune = build_rerank_forensic_packet(
        {"question_id": "q-1", "question": "Q", "reference_answer": "A"},
        "Tune20",
    )
    p_holdout = build_rerank_forensic_packet(
        {"question_id": "q-21", "question": "Q", "reference_answer": "A"},
        "Holdout10",
    )
    with pytest.raises(ValueError, match="Policy discovery requires exclusively Tune20 input"):
        evaluate_tune_reranker_policy_discovery([p_tune, p_holdout])


def test_duplicate_chunks_fail_closed():
    """Duplicate chunks fail closed to AMBIGUOUS."""
    record = {
        "question_id": "q-dup",
        "question": "Câu hỏi",
        "reference_answer": "Đáp án",
        "terminal_retrieval_hits": [{"chunk_id": "c-dup"}, {"chunk_id": "c-dup"}],
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-dup"}, {"chunk_id": "c-dup"}],
                "pre_rerank_candidates": [{"chunk_id": "c-dup", "text": "t"}],
            }
        ],
    }
    packet = build_rerank_forensic_packet(record, "Tune20")
    assert packet.mapping_status == "AMBIGUOUS_DUPLICATE_CHUNKS"
    assert packet.is_oracle_proxy_drop is False
