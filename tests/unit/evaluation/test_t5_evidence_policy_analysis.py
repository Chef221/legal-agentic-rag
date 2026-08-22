"""Unit tests for T5-4A Evidence Selection Policy Analysis tooling (Fix 2 Complete)."""

import pytest
from scripts.t5_evidence_policy_analysis import (
    DeployableHitFeatures,
    FailureLayer,
    OracleDiagnosticLabels,
    analyze_rerank_telemetry_event,
    build_question_census,
    classify_failure_layer_conservative,
    compute_deployable_lexical_overlap,
    compute_oracle_overlap_metrics,
    discover_policy_candidate_tune_only,
    extract_deployable_features,
)


def test_selected_top1_differs_from_rank1_but_reconciles():
    """Test 1: Selected top1 != rank1 but reconciles uniquely against terminal candidates."""
    record = {
        "question_id": "test-diff-01",
        "question": "Câu hỏi kiểm tra khác biệt rank",
        "reference_answer": "Nội dung đáp án chuẩn",
        "rouge_l_score": 0.35,
        "meteor_score": 0.25,
        "terminal_retrieval_hits": [
            {
                "chunk_id": "chunk-r1",
                "document_id": "doc-1",
                "rank": 1,
                "score": 6.0,
                "text": "Văn bản xếp hạng 1",
                "metadata": {"token_count": 100},
            },
            {
                "chunk_id": "chunk-r2",
                "document_id": "doc-2",
                "rank": 2,
                "score": 5.8,
                "text": "Văn bản xếp hạng 2 chọn làm top 1",
                "metadata": {"token_count": 100},
            },
        ],
        "selected_evidence": [
            {
                "chunk_id": "chunk-r2",
                "evidence_id": "E1",
                "document_id": "doc-2",
                "text": "Văn bản xếp hạng 2 chọn làm top 1",
                "metadata": {"evidence_selection": {"source_rank": 2, "selection_rank": 1}},
            },
        ],
    }
    census = build_question_census(record, "Tune20")
    assert census.analysis_valid is True
    assert census.actual_selected_top1_chunk_id == "chunk-r2"
    assert census.actual_selected_top1_source_rank == 2
    assert census.terminal_rank1_chunk_id == "chunk-r1"
    assert census.is_selected_top1_terminal_rank1 is False


def test_selected_top1_missing_from_terminal_hits_fails_closed():
    """Test 2: Selected top1 missing from terminal hits -> analysis_valid=False, AMBIGUOUS."""
    record = {
        "question_id": "test-missing-01",
        "question": "Câu hỏi",
        "reference_answer": "Đáp án",
        "terminal_retrieval_hits": [
            {"chunk_id": "chunk-t1", "document_id": "doc-1", "rank": 1, "text": "văn bản 1"},
        ],
        "selected_evidence": [
            {"chunk_id": "chunk-unreconciled-99", "evidence_id": "E1", "text": "văn bản lạ"},
        ],
    }
    census = build_question_census(record, "Tune20")
    assert census.analysis_valid is False
    assert "SELECTED_TOP1_NOT_RECONCILED_WITH_TERMINAL_HITS" in census.analysis_notes
    assert census.failure_layer == FailureLayer.AMBIGUOUS
    assert census.actual_selected_top1_source_rank is None


def test_selected_top1_duplicate_terminal_chunks_fails_closed():
    """Test 3: Duplicate terminal chunk IDs matching selected top1 fails closed to AMBIGUOUS."""
    record = {
        "question_id": "test-dup-01",
        "question": "Câu hỏi trùng lặp chunk",
        "reference_answer": "Đáp án chuẩn",
        "terminal_retrieval_hits": [
            {"chunk_id": "chunk-dup", "document_id": "doc-1", "rank": 1, "text": "văn bản 1"},
            {"chunk_id": "chunk-dup", "document_id": "doc-2", "rank": 2, "text": "văn bản 2 trùng chunk_id"},
        ],
        "selected_evidence": [
            {"chunk_id": "chunk-dup", "evidence_id": "E1", "text": "văn bản 1"},
        ],
    }
    census = build_question_census(record, "Tune20")
    assert census.analysis_valid is False
    assert "SELECTED_TOP1_AMBIGUOUS_DUPLICATE_TERMINAL_CHUNK_ID" in census.analysis_notes
    assert census.failure_layer == FailureLayer.AMBIGUOUS


def test_high_score_not_generation_miss():
    """Test 4: High score is NO_CLEAR_CAUSAL_OPPORTUNITY, not GENERATION_OR_DOWNSTREAM_MISS."""
    res = classify_failure_layer_conservative(
        analysis_valid=True,
        rouge_l=0.75,
        meteor=0.65,
        selected_top1_f1=0.85,
        oracle_best_f1=0.85,
        oracle_best_chunk_id="c-1",
        actual_selected_top1_chunk_id="c-1",
        f1_regret=0.0,
    )
    assert res == FailureLayer.NO_CLEAR_CAUSAL_OPPORTUNITY


def test_oracle_opportunity_not_confirmed_miss():
    """Test 5: Oracle opportunity produces SELECTION_OPPORTUNITY, not CONFIRMED_SELECTION_MISS."""
    res = classify_failure_layer_conservative(
        analysis_valid=True,
        rouge_l=0.35,
        meteor=0.25,
        selected_top1_f1=0.45,
        oracle_best_f1=0.75,
        oracle_best_chunk_id="c-best",
        actual_selected_top1_chunk_id="c-sel",
        f1_regret=0.30,
        has_proven_selection_miss=False,
    )
    assert res == FailureLayer.SELECTION_OPPORTUNITY


def test_confirmed_miss_requires_explicit_authority_flag():
    """Test 6: Confirmed miss requires explicit authority flag."""
    res = classify_failure_layer_conservative(
        analysis_valid=True,
        rouge_l=0.20,
        meteor=0.10,
        selected_top1_f1=0.20,
        oracle_best_f1=0.60,
        oracle_best_chunk_id="c-best",
        actual_selected_top1_chunk_id="c-sel",
        f1_regret=0.40,
        has_proven_selection_miss=True,
    )
    assert res == FailureLayer.CONFIRMED_SELECTION_MISS


def test_qid_54485_alone_cannot_trigger_confirmed_miss():
    """Test 7: Generic classifier knows nothing about Q54485 string unless flag is set."""
    res_without_flag = classify_failure_layer_conservative(
        analysis_valid=True,
        rouge_l=0.069,
        meteor=0.005,
        selected_top1_f1=0.198,
        oracle_best_f1=0.421,
        oracle_best_chunk_id="c-best",
        actual_selected_top1_chunk_id="c-sel",
        f1_regret=0.223,
        has_proven_selection_miss=False,
    )
    assert res_without_flag == FailureLayer.SELECTION_OPPORTUNITY
    assert res_without_flag != FailureLayer.CONFIRMED_SELECTION_MISS


def test_selection_opportunity_based_on_chunk_identity_not_rank():
    """Test 8: Selection opportunity based on chunk identity even if selected is rank 2 and best is rank 1."""
    res = classify_failure_layer_conservative(
        analysis_valid=True,
        rouge_l=0.35,
        meteor=0.25,
        selected_top1_f1=0.45,
        oracle_best_f1=0.85,
        oracle_best_chunk_id="chunk-rank1",
        actual_selected_top1_chunk_id="chunk-rank2",
        f1_regret=0.40,
    )
    assert res == FailureLayer.SELECTION_OPPORTUNITY


def test_deterministic_prerank_event_mapping():
    """Test 9: Deterministic pre/post rerank event mapping."""
    record = {
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-1"}, {"chunk_id": "c-2"}],
                "pre_rerank_candidates": [{"chunk_id": "c-1", "text": "a"}, {"chunk_id": "c-2", "text": "b"}],
            }
        ]
    }
    status, is_loss = analyze_rerank_telemetry_event(record, "ref", ["c-1", "c-2"])
    assert status == "UNIQUELY_MAPPED"
    assert is_loss is False


def test_ambiguous_rerank_event_cannot_become_rerank_loss():
    """Test 10: Ambiguous rerank event cannot become RERANK_LOSS."""
    record = {
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-other"}],
                "pre_rerank_candidates": [{"chunk_id": "c-other", "text": "c"}],
            }
        ]
    }
    status, is_loss = analyze_rerank_telemetry_event(record, "ref", ["c-1", "c-2"])
    assert status == "AMBIGUOUS_NOT_UNIQUELY_MAPPED"
    assert is_loss is False


def test_proven_prerank_candidate_removal_becomes_rerank_loss():
    """Test 11: Proven best-pre candidate removal with material gap becomes RERANK_LOSS."""
    record = {
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-weak", "text": "không liên quan"}],
                "pre_rerank_candidates": [
                    {"chunk_id": "c-weak", "text": "không liên quan"},
                    {"chunk_id": "c-strong", "text": "Căn cứ quy định chính xác về thủ tục đăng ký"},
                ],
            }
        ]
    }
    ref = "Quy định chính xác về thủ tục đăng ký"
    status, is_loss = analyze_rerank_telemetry_event(record, ref, ["c-weak"])
    assert status == "UNIQUELY_MAPPED"
    assert is_loss is True


def test_retained_prerank_candidate_not_rerank_loss():
    """Test 12: Retained best-pre candidate in post-rerank is NOT a RERANK_LOSS."""
    record = {
        "rerank_telemetry": [
            {
                "attempt_number": 1,
                "post_rerank_hits": [{"chunk_id": "c-strong", "text": "Căn cứ quy định chính xác về thủ tục"}],
                "pre_rerank_candidates": [
                    {"chunk_id": "c-strong", "text": "Căn cứ quy định chính xác về thủ tục"},
                    {"chunk_id": "c-weak", "text": "khác"},
                ],
            }
        ]
    }
    ref = "Quy định chính xác về thủ tục"
    status, is_loss = analyze_rerank_telemetry_event(record, ref, ["c-strong"])
    assert status == "UNIQUELY_MAPPED"
    assert is_loss is False


def test_budget_warning_alone_cannot_become_context_budget_loss():
    """Test 13: Budget warning alone cannot become CONTEXT_BUDGET_LOSS without proven candidate omission."""
    res = classify_failure_layer_conservative(
        analysis_valid=True,
        rouge_l=0.35,
        meteor=0.25,
        selected_top1_f1=0.60,
        oracle_best_f1=0.60,
        oracle_best_chunk_id="c-1",
        actual_selected_top1_chunk_id="c-1",
        f1_regret=0.0,
        has_proven_budget_loss=False,
    )
    assert res != FailureLayer.CONTEXT_BUDGET_LOSS
    assert res == FailureLayer.GENERATION_OR_DOWNSTREAM_MISS


def test_proven_budget_loss_becomes_context_budget_loss():
    """Test 14: Proven candidate-level budget omission can become CONTEXT_BUDGET_LOSS."""
    res = classify_failure_layer_conservative(
        analysis_valid=True,
        rouge_l=0.20,
        meteor=0.10,
        selected_top1_f1=0.20,
        oracle_best_f1=0.80,
        oracle_best_chunk_id="c-best",
        actual_selected_top1_chunk_id="c-sel",
        f1_regret=0.60,
        has_proven_budget_loss=True,
    )
    assert res == FailureLayer.CONTEXT_BUDGET_LOSS


def test_deployable_features_no_oracle_fields():
    """Test 15: Deployable feature object contains no oracle/reference fields."""
    hit = {
        "chunk_id": "c-1",
        "document_id": "d-1",
        "rank": 1,
        "score": 5.5,
        "bm25_rank": 2,
        "dense_rank": 3,
        "text": "Quy định về bảo hiểm",
        "metadata": {"token_count": 100},
    }
    features = extract_deployable_features(hit, "Bảo hiểm y tế?")
    assert isinstance(features, DeployableHitFeatures)
    assert not hasattr(features, "reference_answer")
    assert not hasattr(features, "reference_f1")
    assert not hasattr(features, "reference_recall")


def test_tune_holdout_ordering():
    """Test 16: Deterministic Tune/Holdout ordering."""
    dummy_records = [
        {"question_id": f"q-{i}", "question": f"Q {i}", "reference_answer": f"A {i}"}
        for i in range(30)
    ]
    census_list = [
        build_question_census(r, "Tune20" if i < 20 else "Holdout10")
        for i, r in enumerate(dummy_records)
    ]
    assert len([c for c in census_list if c.split == "Tune20"]) == 20
    assert len([c for c in census_list if c.split == "Holdout10"]) == 10
    assert census_list[0].question_id == "q-0"
    assert census_list[20].question_id == "q-20"


def test_tune_only_discovery_accepts_tune_only():
    """Test 17: Tune-only discovery accepts pure Tune20 input."""
    tune_census = [
        build_question_census(
            {"question_id": f"q-{i}", "question": f"Q {i}", "reference_answer": f"A {i}"},
            "Tune20",
        )
        for i in range(20)
    ]
    res = discover_policy_candidate_tune_only(tune_census)
    assert res["tune_count"] == 20
    assert res["candidate_found"] is False
    assert res["recommendation"] == "NO_SELECTION_POLICY_JUSTIFIED"


def test_holdout_only_discovery_rejected():
    """Test 18: Holdout-only discovery rejected with contamination message."""
    holdout_census = [
        build_question_census(
            {"question_id": f"q-{i}", "question": f"Q {i}", "reference_answer": f"A {i}"},
            "Holdout10",
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="Holdout10 is contaminated"):
        discover_policy_candidate_tune_only(holdout_census)


def test_mixed_tune_holdout_discovery_rejected():
    """Test 19: Mixed Tune+Holdout discovery rejected with contamination message."""
    mixed_census = [
        build_question_census(
            {"question_id": "q-1", "question": "Q 1", "reference_answer": "A 1"},
            "Tune20",
        ),
        build_question_census(
            {"question_id": "q-2", "question": "Q 2", "reference_answer": "A 2"},
            "Holdout10",
        ),
    ]
    with pytest.raises(ValueError, match="Holdout10 is contaminated"):
        discover_policy_candidate_tune_only(mixed_census)
