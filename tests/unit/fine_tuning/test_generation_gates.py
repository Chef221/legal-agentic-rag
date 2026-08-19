"""Unit tests for generation health safety gates, semantic preservation, and checkpoint selection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import pytest

from legal_agentic_rag.fine_tuning.generation_gates import (
    evaluate_checkpoint_health_gate,
    select_best_pilot_checkpoint,
)
from legal_agentic_rag.schemas import (
    CheckpointGateReport,
    CompetitionQuestion,
    QLoRACandidateConfig,
    ValProbeCaseResult,
)


def _make_case(
    qid: str,
    answer: str,
    token_count: int = 100,
    cap_without_eos: bool = False,
    eos_emitted: bool = True,
    repeat8: float = 0.0,
    dup_lines: float = 0.0,
    status: str = "success",
) -> ValProbeCaseResult:
    return ValProbeCaseResult(
        question_id=qid,
        question=f"Question {qid}?",
        generated_answer=answer,
        generated_token_count=token_count,
        reached_cap=cap_without_eos,
        eos_emitted=eos_emitted,
        cap_without_eos=cap_without_eos,
        repeat_8gram_ratio=repeat8,
        duplicate_line_ratio=dup_lines,
        status=status,  # type: ignore[arg-type]
        latency_ms=100.0,
        created_at=datetime.now(UTC),
    )


def test_safety_gate_exact_pass_case() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2")
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Ref {i}")
        for i in range(1, 21)
    ]

    base_cases = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]
    # Candidate slightly better/identical to base
    cand_cases = [_make_case(str(i), f"Ref {i}", token_count=105) for i in range(1, 21)]

    report = evaluate_checkpoint_health_gate(
        candidate_results=cand_cases,
        base_results=base_cases,
        references=questions,
        config=config,
        optimizer_step=50,
        val_loss=1.15,
    )

    assert report.safety_eligible is True
    assert len(report.safety_failure_reasons) == 0


def test_safety_gate_rejection_rules() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2")
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Ref {i}")
        for i in range(1, 21)
    ]

    base_cases = [
        _make_case(str(i), f"Ref {i}", token_count=100, cap_without_eos=False, eos_emitted=True)
        for i in range(1, 21)
    ]

    # 1. Failure: generation errors > 0
    cand_with_err = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]
    cand_with_err[0] = _make_case("1", "GENERATION_ERROR: timeout", status="error")
    rep_err = evaluate_checkpoint_health_gate(
        cand_with_err, base_cases, questions, config, optimizer_step=50, val_loss=1.15
    )
    assert rep_err.safety_eligible is False
    assert any("generation_error_count" in r for r in rep_err.safety_failure_reasons)

    # 2. Failure: excessive cap_without_eos (base=0, allowed <= 1, test with 3)
    cand_cap = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]
    for idx in range(3):
        cand_cap[idx] = _make_case(str(idx + 1), f"Ref {idx + 1}", cap_without_eos=True, eos_emitted=False)
    rep_cap = evaluate_checkpoint_health_gate(
        cand_cap, base_cases, questions, config, optimizer_step=50, val_loss=1.15
    )
    assert rep_cap.safety_eligible is False
    assert any("cap_without_eos_count" in r for r in rep_cap.safety_failure_reasons)

    # 3. Failure: excessive repetition (base=0, allowed <= 1, test with 3)
    cand_rep = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]
    for idx in range(3):
        cand_rep[idx] = _make_case(str(idx + 1), f"Ref {idx + 1}", repeat8=0.45)
    rep_rep = evaluate_checkpoint_health_gate(
        cand_rep, base_cases, questions, config, optimizer_step=50, val_loss=1.15
    )
    assert rep_rep.safety_eligible is False
    assert any("repeat8_high_count" in r for r in rep_rep.safety_failure_reasons)

    # 4. Failure: mean length > 1.35x base (base=100 -> >135 fails)
    cand_long = [_make_case(str(i), f"Ref {i}", token_count=160) for i in range(1, 21)]
    rep_long = evaluate_checkpoint_health_gate(
        cand_long, base_cases, questions, config, optimizer_step=50, val_loss=1.15
    )
    assert rep_long.safety_eligible is False
    assert any("mean_generated_tokens" in r for r in rep_long.safety_failure_reasons)


def test_semantic_gate_and_selection_ranking() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2")
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Dieu {i} quy dinh ro rang.")
        for i in range(1, 21)
    ]

    base_cases = [_make_case(str(i), f"Dieu {i} quy dinh chung.", token_count=100) for i in range(1, 21)]

    # Step 50: Eligible with positive semantic gain
    cand_50 = [_make_case(str(i), f"Dieu {i} quy dinh ro rang.", token_count=100) for i in range(1, 21)]
    rep_50 = evaluate_checkpoint_health_gate(
        cand_50, base_cases, questions, config, optimizer_step=50, val_loss=1.10
    )
    assert rep_50.checkpoint_eligible is True

    # Step 100: Eligible with even higher semantic gain
    cand_100 = [_make_case(str(i), f"Dieu {i} quy dinh ro rang.", token_count=98) for i in range(1, 21)]
    rep_100 = evaluate_checkpoint_health_gate(
        cand_100, base_cases, questions, config, optimizer_step=100, val_loss=1.05
    )
    assert rep_100.checkpoint_eligible is True

    # Step 150: Degenerated repetition loop -> Ineligible
    cand_150 = [_make_case(str(i), f"lap lai lap lai " * 20, token_count=300, repeat8=0.8) for i in range(1, 21)]
    rep_150 = evaluate_checkpoint_health_gate(
        cand_150, base_cases, questions, config, optimizer_step=150, val_loss=1.00
    )
    assert rep_150.checkpoint_eligible is False

    # Selection report ranking
    selection = select_best_pilot_checkpoint(
        gate_reports={50: rep_50, 100: rep_100, 150: rep_150},
        candidate_id="M50-C2",
        checkpoint_dirs={50: "ckpt-50", 100: "ckpt-100", 150: "ckpt-150"},
    )

    assert selection.status == "selected_pilot_checkpoint"
    assert selection.selected_checkpoint_step in [50, 100]
    assert 150 not in selection.ranked_steps


def test_selection_returns_no_promotable_when_zero_eligible() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2")
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Ref {i}")
        for i in range(1, 21)
    ]
    base_cases = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]

    # All checkpoints degenerated
    cand_bad = [_make_case(str(i), "lap lai " * 30, repeat8=0.9, token_count=400) for i in range(1, 21)]
    rep_50 = evaluate_checkpoint_health_gate(cand_bad, base_cases, questions, config, optimizer_step=50, val_loss=1.1)
    rep_100 = evaluate_checkpoint_health_gate(cand_bad, base_cases, questions, config, optimizer_step=100, val_loss=1.0)
    rep_150 = evaluate_checkpoint_health_gate(cand_bad, base_cases, questions, config, optimizer_step=150, val_loss=0.9)

    selection = select_best_pilot_checkpoint(
        gate_reports={50: rep_50, 100: rep_100, 150: rep_150},
        candidate_id="M50-C2",
    )

    assert selection.status == "no_promotable_checkpoint"
    assert selection.selected_checkpoint_step is None


def test_safety_gate_exact_boundary_conditions() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2", health_count_slack=1, max_mean_length_ratio=1.35)
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Ref {i}")
        for i in range(1, 21)
    ]
    # BASE has 5 cap_without_eos, 4 repeat8, 1 dup lines, 15 eos, mean=100, median=100
    base_cases = []
    for i in range(1, 21):
        base_cases.append(_make_case(
            qid=str(i),
            answer=f"Ref {i}",
            token_count=100,
            cap_without_eos=(i <= 5),
            eos_emitted=(i > 5),
            repeat8=0.30 if i <= 4 else 0.0,
            dup_lines=0.30 if i <= 1 else 0.0,
        ))

    # 1. Boundary PASS: cand has base + slack for all failure conditions
    # cap_without_eos = 5 + 1 = 6; repeat8 = 4 + 1 = 5; dup_lines = 1 + 1 = 2; eos = 15 - 1 = 14; mean = 135.0; median = 164.0
    cand_pass = []
    for i in range(1, 21):
        cand_pass.append(_make_case(
            qid=str(i),
            answer=f"Ref {i}",
            token_count=135,
            cap_without_eos=(i <= 6),
            eos_emitted=(i > 6),
            repeat8=0.30 if i <= 5 else 0.0,
            dup_lines=0.30 if i <= 2 else 0.0,
        ))
    rep_pass = evaluate_checkpoint_health_gate(cand_pass, base_cases, questions, config, optimizer_step=50, val_loss=1.0)
    assert rep_pass.safety_eligible is True

    # 2. Boundary FAIL: cap_without_eos = 5 + 2 = 7 (exceeds base + slack)
    cand_fail_cap = [_make_case(str(i), f"Ref {i}", token_count=100, cap_without_eos=(i <= 7), eos_emitted=(i > 7)) for i in range(1, 21)]
    rep_fail_cap = evaluate_checkpoint_health_gate(cand_fail_cap, base_cases, questions, config, optimizer_step=50, val_loss=1.0)
    assert rep_fail_cap.safety_eligible is False
    assert any("cap_without_eos_count" in r for r in rep_fail_cap.safety_failure_reasons)

    # 3. Boundary FAIL: repeat8 = 4 + 2 = 6 (exceeds base + slack)
    cand_fail_rep = [_make_case(str(i), f"Ref {i}", token_count=100, repeat8=0.30 if i <= 6 else 0.0) for i in range(1, 21)]
    rep_fail_rep = evaluate_checkpoint_health_gate(cand_fail_rep, base_cases, questions, config, optimizer_step=50, val_loss=1.0)
    assert rep_fail_rep.safety_eligible is False
    assert any("repeat8_high_count" in r for r in rep_fail_rep.safety_failure_reasons)

    # 4. Boundary FAIL: dup_lines = 1 + 2 = 3 (exceeds base + slack)
    cand_fail_dup = [_make_case(str(i), f"Ref {i}", token_count=100, dup_lines=0.30 if i <= 3 else 0.0) for i in range(1, 21)]
    rep_fail_dup = evaluate_checkpoint_health_gate(cand_fail_dup, base_cases, questions, config, optimizer_step=50, val_loss=1.0)
    assert rep_fail_dup.safety_eligible is False
    assert any("duplicate_line_high_count" in r for r in rep_fail_dup.safety_failure_reasons)

    # 5. Boundary FAIL: eos = 15 - 2 = 13 (below base - slack)
    cand_fail_eos = [_make_case(str(i), f"Ref {i}", token_count=100, eos_emitted=(i <= 13)) for i in range(1, 21)]
    rep_fail_eos = evaluate_checkpoint_health_gate(cand_fail_eos, base_cases, questions, config, optimizer_step=50, val_loss=1.0)
    assert rep_fail_eos.safety_eligible is False
    assert any("eos_emitted_count" in r for r in rep_fail_eos.safety_failure_reasons)

    # 6. Boundary FAIL: mean length = 135.1 > 100 * 1.35
    cand_fail_mean = [_make_case(str(i), f"Ref {i}", token_count=136) for i in range(1, 21)]
    rep_fail_mean = evaluate_checkpoint_health_gate(cand_fail_mean, base_cases, questions, config, optimizer_step=50, val_loss=1.0)
    assert rep_fail_mean.safety_eligible is False
    assert any("mean_generated_tokens" in r for r in rep_fail_mean.safety_failure_reasons)

    # 7. Boundary FAIL: median length > allowed_max (base median=100 -> max(135, 164) = 164)
    cand_fail_med = [_make_case(str(i), f"Ref {i}", token_count=165) for i in range(1, 21)]
    rep_fail_med = evaluate_checkpoint_health_gate(cand_fail_med, base_cases, questions, config, optimizer_step=50, val_loss=1.0)
    assert rep_fail_med.safety_eligible is False
    assert any("median_generated_tokens" in r for r in rep_fail_med.safety_failure_reasons)


def test_meteor_unavailable_semantic_gate_behavior() -> None:
    from legal_agentic_rag.exceptions import BackendInitializationError

    config = QLoRACandidateConfig(candidate_id="M50-C2")
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Dieu {i} quy dinh.")
        for i in range(1, 21)
    ]
    base_cases = [_make_case(str(i), f"Dieu {i} cu.", token_count=100) for i in range(1, 21)]
    cand_cases = [_make_case(str(i), f"Dieu {i} quy dinh.", token_count=100) for i in range(1, 21)]

    # Mock official_meteor_scorer raising BackendInitializationError
    def _raising_meteor_scorer(refs: list[list[str]], hyps: list[str]) -> float:
        raise BackendInitializationError("Mock WordNet unavailable")

    report = evaluate_checkpoint_health_gate(
        cand_cases,
        base_cases,
        questions,
        config,
        optimizer_step=50,
        val_loss=1.0,
        official_meteor_scorer=_raising_meteor_scorer,
    )

    assert report.meteor_available is False
    assert report.mean_meteor_delta is None
    assert report.combined_semantic_delta == report.mean_rouge_l_delta
    assert report.semantic_eligible is True
    assert report.checkpoint_eligible is True


def test_checkpoint_selection_tie_breaking_rules() -> None:
    from legal_agentic_rag.schemas import CheckpointGateReport

    def _stub_report(step: int, combined_delta: float, cap: int, rep8: int, val_loss: float) -> CheckpointGateReport:
        from datetime import UTC, datetime
        return CheckpointGateReport(
            created_at=datetime.now(UTC),
            code_version="0.50.5",
            candidate_id="M50-C2",
            optimizer_step=step,
            val_loss=val_loss,
            base_probe_sha256="base_sha",
            candidate_probe_sha256="cand_sha",
            candidate_eos_emitted_count=15,
            candidate_reached_cap_count=cap,
            candidate_cap_without_eos_count=cap,
            candidate_cap_without_eos_rate=cap / 20.0,
            candidate_repeat8_high_count=rep8,
            candidate_duplicate_line_high_count=0,
            candidate_mean_generated_token_count=100.0,
            candidate_median_generated_token_count=100.0,
            candidate_generation_error_count=0,
            base_eos_emitted_count=15,
            base_reached_cap_count=5,
            base_cap_without_eos_count=5,
            base_repeat8_high_count=4,
            base_duplicate_line_high_count=1,
            base_mean_generated_token_count=100.0,
            base_median_generated_token_count=100.0,
            mean_rouge_l_delta=combined_delta,
            median_rouge_l_delta=combined_delta,
            mean_meteor_delta=combined_delta,
            median_meteor_delta=combined_delta,
            meteor_available=True,
            combined_semantic_delta=combined_delta,
            safety_eligible=True,
            safety_failure_reasons=[],
            semantic_eligible=True,
            semantic_failure_reasons=[],
            checkpoint_eligible=True,
            warnings=[],
        )

    # Case 1: Higher semantic delta wins
    rep50 = _stub_report(step=50, combined_delta=0.03, cap=5, rep8=4, val_loss=1.1)
    rep100 = _stub_report(step=100, combined_delta=0.05, cap=5, rep8=4, val_loss=1.0)
    sel1 = select_best_pilot_checkpoint({50: rep50, 100: rep100})
    assert sel1.selected_checkpoint_step == 100

    # Case 2: Equal semantic delta -> lower cap_without_eos wins
    rep50 = _stub_report(step=50, combined_delta=0.04, cap=3, rep8=4, val_loss=1.1)
    rep100 = _stub_report(step=100, combined_delta=0.04, cap=5, rep8=4, val_loss=1.0)
    sel2 = select_best_pilot_checkpoint({50: rep50, 100: rep100})
    assert sel2.selected_checkpoint_step == 50

    # Case 3: Equal semantic delta and cap -> lower repeat8 wins
    rep50 = _stub_report(step=50, combined_delta=0.04, cap=3, rep8=4, val_loss=1.1)
    rep100 = _stub_report(step=100, combined_delta=0.04, cap=3, rep8=2, val_loss=1.0)
    sel3 = select_best_pilot_checkpoint({50: rep50, 100: rep100})
    assert sel3.selected_checkpoint_step == 100

    # Case 4: Equal semantic delta, cap, and repeat8 -> lower val_loss wins
    rep50 = _stub_report(step=50, combined_delta=0.04, cap=3, rep8=2, val_loss=1.05)
    rep100 = _stub_report(step=100, combined_delta=0.04, cap=3, rep8=2, val_loss=0.95)
    sel4 = select_best_pilot_checkpoint({50: rep50, 100: rep100})
    assert sel4.selected_checkpoint_step == 100


def test_semantic_gate_positive_rouge_meteor_passes() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2")
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Ref {i}")
        for i in range(1, 21)
    ]
    base_cases = [_make_case(str(i), f"Mismatched text {i}", token_count=100) for i in range(1, 21)]
    # Candidate matches reference exactly -> positive ROUGE gain
    cand_cases = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]

    rep = evaluate_checkpoint_health_gate(
        cand_cases, base_cases, questions, config, optimizer_step=50, val_loss=1.1
    )
    assert rep.semantic_eligible is True
    assert rep.mean_rouge_l_delta is not None and rep.mean_rouge_l_delta > 0.0


def test_semantic_gate_rejection_when_both_negative() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2")
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Ref {i}")
        for i in range(1, 21)
    ]
    base_cases = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]
    # Candidate produces worse lexical overlap than base
    cand_cases = [_make_case(str(i), f"Completely unrelated text {i}", token_count=100) for i in range(1, 21)]

    rep = evaluate_checkpoint_health_gate(
        cand_cases, base_cases, questions, config, optimizer_step=50, val_loss=1.1
    )
    assert rep.semantic_eligible is False
    assert any("positive semantic signal" in r or "tolerance" in r for r in rep.semantic_failure_reasons)


def test_semantic_gate_rejection_below_tolerance() -> None:
    config = QLoRACandidateConfig(candidate_id="M50-C2", semantic_regression_tolerance=-0.01)
    questions = [
        CompetitionQuestion(question_id=str(i), question=f"Q{i}?", reference_answer=f"Ref {i}")
        for i in range(1, 21)
    ]
    base_cases = [_make_case(str(i), f"Ref {i}", token_count=100) for i in range(1, 21)]
    cand_cases = [_make_case(str(i), f"Different {i}", token_count=100) for i in range(1, 21)]

    rep = evaluate_checkpoint_health_gate(
        cand_cases, base_cases, questions, config, optimizer_step=50, val_loss=1.1
    )
    assert rep.semantic_eligible is False
    assert rep.checkpoint_eligible is False


def test_selection_ineligible_cannot_be_selected() -> None:
    from legal_agentic_rag.schemas import CheckpointGateReport

    def _stub(step: int, eligible: bool) -> CheckpointGateReport:
        from datetime import UTC, datetime
        return CheckpointGateReport(
            created_at=datetime.now(UTC),
            code_version="0.50.5",
            candidate_id="M50-C2",
            optimizer_step=step,
            val_loss=1.0,
            base_probe_sha256="base_sha",
            candidate_probe_sha256="cand_sha",
            candidate_eos_emitted_count=15,
            candidate_reached_cap_count=5,
            candidate_cap_without_eos_count=5,
            candidate_cap_without_eos_rate=0.25,
            candidate_repeat8_high_count=4,
            candidate_duplicate_line_high_count=0,
            candidate_mean_generated_token_count=100.0,
            candidate_median_generated_token_count=100.0,
            candidate_generation_error_count=0,
            base_eos_emitted_count=15,
            base_reached_cap_count=5,
            base_cap_without_eos_count=5,
            base_repeat8_high_count=4,
            base_duplicate_line_high_count=1,
            base_mean_generated_token_count=100.0,
            base_median_generated_token_count=100.0,
            mean_rouge_l_delta=0.05 if eligible else -0.05,
            median_rouge_l_delta=0.05 if eligible else -0.05,
            meteor_available=False,
            combined_semantic_delta=0.05 if eligible else -0.05,
            safety_eligible=eligible,
            safety_failure_reasons=[] if eligible else ["failed safety"],
            semantic_eligible=eligible,
            semantic_failure_reasons=[] if eligible else ["failed semantics"],
            checkpoint_eligible=eligible,
            warnings=[],
        )

    # Only step 50 is eligible, steps 100 and 150 are ineligible
    rep50 = _stub(50, True)
    rep100 = _stub(100, False)
    rep150 = _stub(150, False)
    sel = select_best_pilot_checkpoint({50: rep50, 100: rep100, 150: rep150})
    assert sel.status == "selected_pilot_checkpoint"
    assert sel.selected_checkpoint_step == 50
    assert 100 not in sel.ranked_steps
    assert 150 not in sel.ranked_steps
