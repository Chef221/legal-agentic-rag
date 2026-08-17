"""Paired evaluation and bootstrap statistics for direct QA screening."""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import fmean, median
from typing import Callable

import numpy as np

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.official_scoring import (
    score_official_compatible_answer,
)
from legal_agentic_rag.evaluation import score_text_answer
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas import (
    CompetitionQuestion,
    DirectQACaseResult,
    DirectQAPairedCaseScore,
    DirectQAPairedComparisonReport,
    PairedBootstrapInterval,
    PairedMetricSummary,
)


def compute_paired_bootstrap_ci(
    deltas: list[float],
    metric_name: str,
    *,
    resamples: int = 1000,
    seed: int = 2026,
) -> PairedBootstrapInterval:
    """Compute deterministic 95% bootstrap confidence interval for paired metric deltas."""
    if not deltas:
        raise DataValidationError("Cannot compute bootstrap interval on empty deltas")

    rng = np.random.default_rng(seed)
    n = len(deltas)
    delta_arr = np.array(deltas, dtype=np.float64)

    # Generate bootstrap samples
    boot_indices = rng.integers(0, n, size=(resamples, n))
    boot_means = np.mean(delta_arr[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    mean_val = float(np.mean(delta_arr))
    med_val = float(np.median(delta_arr))

    return PairedBootstrapInterval(
        metric_name=metric_name,
        mean_delta=mean_val,
        median_delta=med_val,
        ci_lower_95=ci_lower,
        ci_upper_95=ci_upper,
        resamples=resamples,
        seed=seed,
    )


class DirectQAPairedScorer:
    """Score and compare paired BASE vs TREATMENT outputs against official references."""

    def __init__(
        self,
        *,
        official_meteor_scorer: Callable[[list[list[str]], list[str]], float] | None = None,
        use_diagnostic_fallback: bool = False,
    ) -> None:
        self.official_meteor_scorer = official_meteor_scorer
        self.use_diagnostic_fallback = use_diagnostic_fallback

    def score_pair(self, prediction: str, reference: str) -> tuple[float, float]:
        """Compute METEOR and ROUGE-L for one prediction against official reference."""
        try:
            metrics = score_official_compatible_answer(
                prediction=prediction,
                reference=reference,
                meteor_scorer=self.official_meteor_scorer,
            )
            return metrics.meteor, metrics.rouge_l
        except Exception:
            if not self.use_diagnostic_fallback:
                raise
            diag = score_text_answer(prediction=prediction, reference=reference)
            return diag.meteor, diag.rouge_l

    def compare(
        self,
        base_results: list[DirectQACaseResult],
        treatment_results: list[DirectQACaseResult],
        references: list[CompetitionQuestion],
        *,
        bootstrap_resamples: int = 1000,
        bootstrap_seed: int = 2026,
        screen_holdout_sha256: str | None = None,
        base_results_sha256: str | None = None,
        treatment_results_sha256: str | None = None,
        training_manifest_sha256: str | None = None,
        adapter_config_sha256: str | None = None,
        adapter_weights_sha256: str | None = None,
        best_checkpoint_step: int | None = None,
    ) -> DirectQAPairedComparisonReport:
        """Produce a complete paired comparison report with bootstrap confidence intervals."""
        base_map = {r.question_id: r for r in base_results}
        treat_map = {r.question_id: r for r in treatment_results}
        ref_map = {q.question_id: q for q in references if q.reference_answer is not None}

        common_ids = [q.question_id for q in references if q.question_id in base_map and q.question_id in treat_map and q.reference_answer is not None]

        if len(common_ids) != len(references):
            raise DataValidationError(
                f"Question ID mismatch in paired screening: references={len(references)}, common={len(common_ids)}"
            )

        cases: list[DirectQAPairedCaseScore] = []
        base_meteors: list[float] = []
        treat_meteors: list[float] = []
        delta_meteors: list[float] = []

        base_rouges: list[float] = []
        treat_rouges: list[float] = []
        delta_rouges: list[float] = []

        base_lens: list[int] = []
        treat_lens: list[int] = []
        ref_lens: list[int] = []

        for qid in common_ids:
            b_res = base_map[qid]
            t_res = treat_map[qid]
            ref_ans = ref_map[qid].reference_answer
            assert ref_ans is not None

            # 1. Reject failed generations - do not score error placeholders
            if (
                b_res.status != "success"
                or t_res.status != "success"
                or b_res.generated_answer.startswith("GENERATION_ERROR:")
                or t_res.generated_answer.startswith("GENERATION_ERROR:")
            ):
                raise DataValidationError(
                    f"Generation error detected for question ID {qid} (BASE status: {b_res.status}, TREATMENT status: {t_res.status}). Paired evaluation rejected."
                )

            b_meteor, b_rouge = self.score_pair(b_res.generated_answer, ref_ans)
            t_meteor, t_rouge = self.score_pair(t_res.generated_answer, ref_ans)

            d_meteor = t_meteor - b_meteor
            d_rouge = t_rouge - b_rouge

            b_len = len(b_res.generated_answer)
            t_len = len(t_res.generated_answer)
            r_len = len(ref_ans)

            case = DirectQAPairedCaseScore(
                question_id=qid,
                base_meteor=b_meteor,
                treatment_meteor=t_meteor,
                delta_meteor=d_meteor,
                base_rouge_l=b_rouge,
                treatment_rouge_l=t_rouge,
                delta_rouge_l=d_rouge,
                base_answer_length=b_len,
                treatment_answer_length=t_len,
                reference_answer_length=r_len,
            )
            cases.append(case)

            base_meteors.append(b_meteor)
            treat_meteors.append(t_meteor)
            delta_meteors.append(d_meteor)

            base_rouges.append(b_rouge)
            treat_rouges.append(t_rouge)
            delta_rouges.append(d_rouge)

            base_lens.append(b_len)
            treat_lens.append(t_len)
            ref_lens.append(r_len)

        # Compute metric summaries
        meteor_ci = compute_paired_bootstrap_ci(
            delta_meteors, "meteor", resamples=bootstrap_resamples, seed=bootstrap_seed
        )
        rouge_ci = compute_paired_bootstrap_ci(
            delta_rouges, "rouge_l", resamples=bootstrap_resamples, seed=bootstrap_seed
        )

        meteor_summary = PairedMetricSummary(
            base_mean=fmean(base_meteors),
            treatment_mean=fmean(treat_meteors),
            mean_delta=fmean(delta_meteors),
            median_delta=median(delta_meteors),
            win_count=sum(1 for d in delta_meteors if d > 1e-6),
            tie_count=sum(1 for d in delta_meteors if abs(d) <= 1e-6),
            loss_count=sum(1 for d in delta_meteors if d < -1e-6),
            bootstrap_ci_95=meteor_ci,
        )

        rouge_summary = PairedMetricSummary(
            base_mean=fmean(base_rouges),
            treatment_mean=fmean(treat_rouges),
            mean_delta=fmean(delta_rouges),
            median_delta=median(delta_rouges),
            win_count=sum(1 for d in delta_rouges if d > 1e-6),
            tie_count=sum(1 for d in delta_rouges if abs(d) <= 1e-6),
            loss_count=sum(1 for d in delta_rouges if d < -1e-6),
            bootstrap_ci_95=rouge_ci,
        )

        length_summary = {
            "base_mean_characters": fmean(base_lens),
            "treatment_mean_characters": fmean(treat_lens),
            "reference_mean_characters": fmean(ref_lens),
            "mean_character_delta": fmean(treat_lens) - fmean(base_lens),
        }

        # Max token censorship inspection
        base_hit_max_count = sum(1 for r in base_results if r.hit_max_tokens)
        treatment_hit_max_count = sum(1 for r in treatment_results if r.hit_max_tokens)

        warnings: list[str] = []
        if base_hit_max_count > 0:
            warnings.append(f"base_generations_hit_max_tokens_count_{base_hit_max_count}")
        if treatment_hit_max_count > 0:
            warnings.append(f"treatment_generations_hit_max_tokens_count_{treatment_hit_max_count}")

        total_cases = len(cases)
        if total_cases > 0:
            if (base_hit_max_count / total_cases > 0.02) or (treatment_hit_max_count / total_cases > 0.02):
                warnings.append("screening_potentially_censored")

        sample_base = base_results[0]
        sample_treat = treatment_results[0]

        return DirectQAPairedComparisonReport(
            created_at=datetime.now(UTC),
            code_version=__version__,
            base_model_id=sample_base.model_id,
            base_model_revision=sample_base.model_revision,
            treatment_model_id=sample_treat.model_id,
            treatment_model_revision=sample_treat.model_revision,
            question_count=len(cases),
            screen_holdout_sha256=screen_holdout_sha256,
            base_results_sha256=base_results_sha256,
            treatment_results_sha256=treatment_results_sha256,
            training_manifest_sha256=training_manifest_sha256,
            adapter_config_sha256=adapter_config_sha256,
            adapter_weights_sha256=adapter_weights_sha256,
            best_checkpoint_step=best_checkpoint_step,
            base_hit_max_tokens_count=base_hit_max_count,
            treatment_hit_max_tokens_count=treatment_hit_max_count,
            meteor=meteor_summary,
            rouge_l=rouge_summary,
            length_summary=length_summary,
            cases=cases,
            warnings=warnings,
        )
