"""Generation health safety gates, semantic preservation evaluation, and checkpoint selection."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean, median
import time
from typing import Any, Callable

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.official_scoring import (
    score_official_compatible_answer,
)
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    DataValidationError,
)
from legal_agentic_rag.fine_tuning.val_probe import (
    compute_duplicate_line_ratio,
    compute_repeat_ngram_ratio,
)
from legal_agentic_rag.schemas import (
    CheckpointGateReport,
    CheckpointSelectionReport,
    CompetitionQuestion,
    QLoRACandidateConfig,
    ValProbeCaseResult,
)


def run_free_generation_probe(
    model: Any,
    tokenizer: Any,
    questions: list[CompetitionQuestion],
    *,
    system_prompt: str = "Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam.",
    max_new_tokens: int = 512,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ValProbeCaseResult]:
    """Execute greedy free generation on probe questions and compute health diagnostics."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    results: list[ValProbeCaseResult] = []
    total = len(questions)

    for idx, q in enumerate(questions, start=1):
        if progress_callback:
            progress_callback(idx, total)

        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)

        t0 = time.perf_counter()
        import torch

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            gen_tokens = outputs[0][input_ids.shape[-1] :].tolist()
            gen_token_count = len(gen_tokens)
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            reached_cap = gen_token_count >= max_new_tokens
            eos_emitted = (eos_id in gen_tokens) if eos_id is not None else False
            cap_without_eos = reached_cap and not eos_emitted

            rep8 = compute_repeat_ngram_ratio(gen_text, n=8)
            dup_lines = compute_duplicate_line_ratio(gen_text)

            case = ValProbeCaseResult(
                question_id=q.question_id,
                question=q.question,
                generated_answer=gen_text,
                generated_token_count=gen_token_count,
                reached_cap=reached_cap,
                eos_emitted=eos_emitted,
                cap_without_eos=cap_without_eos,
                repeat_8gram_ratio=rep8,
                duplicate_line_ratio=dup_lines,
                status="success",
                latency_ms=elapsed_ms,
                created_at=datetime.now(UTC),
            )
        except Exception as err:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            case = ValProbeCaseResult(
                question_id=q.question_id,
                question=q.question,
                generated_answer=f"GENERATION_ERROR: {err}",
                generated_token_count=0,
                reached_cap=False,
                eos_emitted=False,
                cap_without_eos=False,
                repeat_8gram_ratio=0.0,
                duplicate_line_ratio=0.0,
                status="error",
                latency_ms=elapsed_ms,
                created_at=datetime.now(UTC),
            )
        results.append(case)

    return results


def evaluate_checkpoint_health_gate(
    candidate_results: list[ValProbeCaseResult],
    base_results: list[ValProbeCaseResult],
    references: list[CompetitionQuestion],
    config: QLoRACandidateConfig,
    *,
    optimizer_step: int,
    val_loss: float,
    official_meteor_scorer: Callable[[list[list[str]], list[str]], float] | None = None,
) -> CheckpointGateReport:
    """Evaluate generation safety and semantic preservation against immutable BASE probe cache."""
    if len(candidate_results) != len(base_results):
        raise DataValidationError(
            f"Case count mismatch: candidate={len(candidate_results)}, base={len(base_results)}"
        )

    ref_map = {q.question_id: q.reference_answer for q in references if q.reference_answer is not None}
    base_map = {b.question_id: b for b in base_results}

    # 1. Candidate Generation Diagnostics
    candidate_error_count = sum(1 for c in candidate_results if c.status != "success")
    candidate_cap_without_eos = sum(1 for c in candidate_results if c.cap_without_eos)
    candidate_reached_cap = sum(1 for c in candidate_results if c.reached_cap)
    candidate_eos_count = sum(1 for c in candidate_results if c.eos_emitted)
    candidate_repeat8_high = sum(
        1 for c in candidate_results if c.repeat_8gram_ratio >= config.repetition_high_threshold
    )
    candidate_dup_lines_high = sum(
        1 for c in candidate_results if c.duplicate_line_ratio >= config.duplicate_line_high_threshold
    )
    candidate_lengths = [c.generated_token_count for c in candidate_results]
    cand_mean_len = float(fmean(candidate_lengths)) if candidate_lengths else 0.0
    cand_median_len = float(median(candidate_lengths)) if candidate_lengths else 0.0

    # 2. BASE Diagnostics
    base_cap_without_eos = sum(1 for b in base_results if b.cap_without_eos)
    base_reached_cap = sum(1 for b in base_results if b.reached_cap)
    base_eos_count = sum(1 for b in base_results if b.eos_emitted)
    base_repeat8_high = sum(
        1 for b in base_results if b.repeat_8gram_ratio >= config.repetition_high_threshold
    )
    base_dup_lines_high = sum(
        1 for b in base_results if b.duplicate_line_ratio >= config.duplicate_line_high_threshold
    )
    base_lengths = [b.generated_token_count for b in base_results]
    base_mean_len = float(fmean(base_lengths)) if base_lengths else 0.0
    base_median_len = float(median(base_lengths)) if base_lengths else 0.0

    # 3. Safety Gate Evaluation
    slack = config.health_count_slack
    max_length_ratio = config.max_mean_length_ratio
    safety_failures: list[str] = []

    if candidate_error_count > 0:
        safety_failures.append(f"generation_error_count ({candidate_error_count}) > 0")

    if candidate_cap_without_eos > base_cap_without_eos + slack:
        safety_failures.append(
            f"cap_without_eos_count ({candidate_cap_without_eos}) > BASE ({base_cap_without_eos}) + slack ({slack})"
        )

    if candidate_repeat8_high > base_repeat8_high + slack:
        safety_failures.append(
            f"repeat8_high_count ({candidate_repeat8_high}) > BASE ({base_repeat8_high}) + slack ({slack})"
        )

    if candidate_dup_lines_high > base_dup_lines_high + slack:
        safety_failures.append(
            f"duplicate_line_high_count ({candidate_dup_lines_high}) > BASE ({base_dup_lines_high}) + slack ({slack})"
        )

    if candidate_eos_count < base_eos_count - slack:
        safety_failures.append(
            f"eos_emitted_count ({candidate_eos_count}) < BASE ({base_eos_count}) - slack ({slack})"
        )

    if cand_mean_len > base_mean_len * max_length_ratio:
        safety_failures.append(
            f"mean_generated_tokens ({cand_mean_len:.1f}) > BASE ({base_mean_len:.1f}) * {max_length_ratio}"
        )

    allowed_max_median = max(base_median_len * max_length_ratio, base_median_len + 64.0)
    if cand_median_len > allowed_max_median:
        safety_failures.append(
            f"median_generated_tokens ({cand_median_len:.1f}) > allowed max ({allowed_max_median:.1f})"
        )

    safety_eligible = len(safety_failures) == 0

    # 4. Paired Semantic Metric Evaluation
    rouge_deltas: list[float] = []
    meteor_deltas: list[float] = []
    meteor_available = True

    for c in candidate_results:
        b = base_map.get(c.question_id)
        ref = ref_map.get(c.question_id)
        if not b or not ref:
            continue
        if c.status != "success" or b.status != "success":
            continue

        try:
            c_score = score_official_compatible_answer(
                prediction=c.generated_answer,
                reference=ref,
                meteor_scorer=official_meteor_scorer,
            )
            b_score = score_official_compatible_answer(
                prediction=b.generated_answer,
                reference=ref,
                meteor_scorer=official_meteor_scorer,
            )
            rouge_deltas.append(c_score.rouge_l - b_score.rouge_l)
            meteor_deltas.append(c_score.meteor - b_score.meteor)
        except BackendInitializationError:
            # WordNet/NLTK 3.7 unavailable locally — compute ROUGE-L only
            meteor_available = False
            from legal_agentic_rag.competition.uit_dsc_2026.official_scoring import _official_rouge_l

            c_rouge = _official_rouge_l(c.generated_answer, ref)
            b_rouge = _official_rouge_l(b.generated_answer, ref)
            rouge_deltas.append(c_rouge - b_rouge)

    mean_rouge_delta = float(fmean(rouge_deltas)) if rouge_deltas else 0.0
    med_rouge_delta = float(median(rouge_deltas)) if rouge_deltas else 0.0

    mean_meteor_delta: float | None = None
    med_meteor_delta: float | None = None
    if meteor_available and meteor_deltas:
        mean_meteor_delta = float(fmean(meteor_deltas))
        med_meteor_delta = float(median(meteor_deltas))
        combined_semantic = (mean_rouge_delta + mean_meteor_delta) / 2.0
    else:
        combined_semantic = mean_rouge_delta

    # 5. Semantic Gate Evaluation
    semantic_failures: list[str] = []
    reg_tol = config.semantic_regression_tolerance

    if mean_rouge_delta < reg_tol:
        semantic_failures.append(f"mean ROUGE-L delta ({mean_rouge_delta:+.4f}) < tolerance ({reg_tol})")

    if meteor_available and mean_meteor_delta is not None and mean_meteor_delta < reg_tol:
        semantic_failures.append(f"mean METEOR delta ({mean_meteor_delta:+.4f}) < tolerance ({reg_tol})")

    has_positive_signal = (mean_rouge_delta > 0.0) or (
        meteor_available and mean_meteor_delta is not None and mean_meteor_delta > 0.0
    )
    if not has_positive_signal:
        semantic_failures.append("No positive semantic signal: neither ROUGE-L nor METEOR mean delta > 0")

    semantic_eligible = len(semantic_failures) == 0
    checkpoint_eligible = safety_eligible and semantic_eligible

    cand_bytes = "".join(c.model_dump_json() for c in candidate_results).encode("utf-8")
    base_bytes = "".join(b.model_dump_json() for b in base_results).encode("utf-8")

    return CheckpointGateReport(
        created_at=datetime.now(UTC),
        code_version=__version__,
        candidate_id=config.candidate_id,
        optimizer_step=optimizer_step,
        val_loss=val_loss,
        base_probe_sha256=sha256(base_bytes).hexdigest(),
        candidate_probe_sha256=sha256(cand_bytes).hexdigest(),
        candidate_eos_emitted_count=candidate_eos_count,
        candidate_reached_cap_count=candidate_reached_cap,
        candidate_cap_without_eos_count=candidate_cap_without_eos,
        candidate_cap_without_eos_rate=float(candidate_cap_without_eos / max(len(candidate_results), 1)),
        candidate_repeat8_high_count=candidate_repeat8_high,
        candidate_duplicate_line_high_count=candidate_dup_lines_high,
        candidate_mean_generated_token_count=round(cand_mean_len, 2),
        candidate_median_generated_token_count=round(cand_median_len, 2),
        candidate_generation_error_count=candidate_error_count,
        base_eos_emitted_count=base_eos_count,
        base_reached_cap_count=base_reached_cap,
        base_cap_without_eos_count=base_cap_without_eos,
        base_repeat8_high_count=base_repeat8_high,
        base_duplicate_line_high_count=base_dup_lines_high,
        base_mean_generated_token_count=round(base_mean_len, 2),
        base_median_generated_token_count=round(base_median_len, 2),
        mean_rouge_l_delta=round(mean_rouge_delta, 5),
        median_rouge_l_delta=round(med_rouge_delta, 5),
        mean_meteor_delta=round(mean_meteor_delta, 5) if mean_meteor_delta is not None else None,
        median_meteor_delta=round(med_meteor_delta, 5) if med_meteor_delta is not None else None,
        meteor_available=meteor_available,
        combined_semantic_delta=round(combined_semantic, 5),
        safety_eligible=safety_eligible,
        safety_failure_reasons=safety_failures,
        semantic_eligible=semantic_eligible,
        semantic_failure_reasons=semantic_failures,
        checkpoint_eligible=checkpoint_eligible,
        warnings=[],
    )


def select_best_pilot_checkpoint(
    gate_reports: dict[int, CheckpointGateReport],
    candidate_id: str = "M50-C2",
    checkpoint_dirs: dict[int, str] | None = None,
) -> CheckpointSelectionReport:
    """Rank eligible checkpoints by semantic gain and generation health tie-breakers."""
    evaluated_steps = sorted(gate_reports.keys())
    eligible_steps = [step for step in evaluated_steps if gate_reports[step].checkpoint_eligible]

    dirs = checkpoint_dirs or {}

    if not eligible_steps:
        return CheckpointSelectionReport(
            created_at=datetime.now(UTC),
            code_version=__version__,
            candidate_id=candidate_id,
            status="no_promotable_checkpoint",
            selected_checkpoint_step=None,
            selected_checkpoint_dir=None,
            evaluated_steps=evaluated_steps,
            eligible_steps=[],
            ranked_steps=[],
            ranking_explanation=["Zero checkpoints met combined safety and semantic eligibility requirements."],
            gate_reports={str(k): v for k, v in gate_reports.items()},
            warnings=["No checkpoint passed generation health safety gates."],
        )

    # Rank eligible steps:
    # Primary: higher combined_semantic_delta
    # Tie-break 1: lower candidate_cap_without_eos_count
    # Tie-break 2: lower candidate_repeat8_high_count
    # Tie-break 3: lower val_loss
    def ranking_key(step: int) -> tuple[float, float, float, float]:
        rep = gate_reports[step]
        return (
            rep.combined_semantic_delta,
            -float(rep.candidate_cap_without_eos_count),
            -float(rep.candidate_repeat8_high_count),
            -float(rep.val_loss),
        )

    ranked = sorted(eligible_steps, key=ranking_key, reverse=True)
    best_step = ranked[0]
    best_dir = dirs.get(best_step)

    explanations: list[str] = [
        f"Selected step {best_step} as top candidate from eligible set {eligible_steps}."
    ]
    for r_idx, step in enumerate(ranked, start=1):
        rep = gate_reports[step]
        explanations.append(
            f"Rank {r_idx}: Step {step} — combined_semantic_delta={rep.combined_semantic_delta:+.4f}, "
            f"cap_without_eos={rep.candidate_cap_without_eos_count}, "
            f"repeat8_high={rep.candidate_repeat8_high_count}, val_loss={rep.val_loss:.4f}"
        )

    return CheckpointSelectionReport(
        created_at=datetime.now(UTC),
        code_version=__version__,
        candidate_id=candidate_id,
        status="selected_pilot_checkpoint",
        selected_checkpoint_step=best_step,
        selected_checkpoint_dir=best_dir,
        evaluated_steps=evaluated_steps,
        eligible_steps=eligible_steps,
        ranked_steps=ranked,
        ranking_explanation=explanations,
        gate_reports={str(k): v for k, v in gate_reports.items()},
        warnings=[],
    )
