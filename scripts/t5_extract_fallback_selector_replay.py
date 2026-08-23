"""T5-7C Isolated Reference-Blind Extractive Fallback Selector Replay.

Replays reference-blind n-gram coverage evidence selection against closed Tune20 Control
artifacts and evaluates official ROUGE-L and METEOR metrics with exact official scorer authority.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import tarfile
import time
from typing import Any
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = _REPO_ROOT / "src"
for path_entry in (_REPO_ROOT, _REPO_SRC):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
    render_competition_answer,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas.answering import AnswerResponse, Citation

import scripts.t5_generator_contract_measurement as scm

logger = logging.getLogger("t5_extract_fallback_selector_replay")

# Pinned Artifact Authorities
EXPECTED_GENERATION_ARCHIVE_SHA256 = (
    "75d00bd42908387a94ccabb4eb76b27900bc6fcfbcaf516c74f08b4bf0c9af4e"
)
EXPECTED_FAST30_ARCHIVE_SHA256 = (
    "be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f"
)
EXPECTED_SCORER_ARCHIVE_SHA256 = scm.OFFICIAL_SCORER_ARCHIVE_SHA256
EXPECTED_SCORING_PY_SHA256 = scm.OFFICIAL_SCORING_PY_SHA256

DEFAULT_NGRAM_SIZE = 3
DEFAULT_SWITCH_MARGIN = 0.20

EXPECTED_SWITCH_SET = {
    "89271": "E3",
    "46497": "E2",
    "150207": "E4",
    "21011": "E3",
    "84363": "E4",
}

EXPECTED_DIAGNOSTIC_COVERAGES = {
    "89271": {"E1": 0.14285714285714285, "selected": 0.35714285714285715},
    "46497": {"E1": 0.36, "selected": 0.88},
    "150207": {"E1": 0.1875, "selected": 0.5625},
    "21011": {"E1": 0.391304347826087, "selected": 0.6086956521739131},
    "84363": {"E1": 0.21428571428571427, "selected": 0.5},
}

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def extract_tokens(text: str) -> list[str]:
    """Extract case-folded word tokens using Unicode word boundaries."""
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


def extract_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Compute contiguous word n-grams as a unique set."""
    toks = extract_tokens(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class SelectionDecision(BaseModel):
    """Immutable result of a reference-blind fallback evidence selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_evidence_id: str = Field(
        ..., description="Evidence ID chosen by the selector (e.g. 'E1', 'E3')"
    )
    e1_coverage: float = Field(
        ..., description="Trigram coverage score for E1"
    )
    selected_coverage: float = Field(
        ..., description="Trigram coverage score for the chosen evidence"
    )
    coverage_margin: float = Field(
        ..., description="Coverage difference: selected_coverage - e1_coverage"
    )
    switched: bool = Field(
        ..., description="True if selection switched away from E1"
    )
    ngram_size: int = Field(
        default=DEFAULT_NGRAM_SIZE, description="N-gram size used (default: 3)"
    )
    switch_margin: float = Field(
        default=DEFAULT_SWITCH_MARGIN,
        description="Minimum coverage margin required to switch (default: 0.20)",
    )


class T57CReplayResult(BaseModel):
    """Complete serialized result of T5-7C fallback selector replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "t5_7c_fallback_selector_replay_v1"
    starting_authority: str
    generation_archive_sha256: str
    fast30_archive_sha256: str
    official_scorer_archive_sha256: str
    official_scoring_py_sha256: str
    ngram_size: int
    switch_margin: float
    reference_blind_selection: bool = True
    selector_decisions: dict[str, SelectionDecision]
    switch_qids: list[str]
    control: dict[str, float]
    candidate: dict[str, float]
    delta: dict[str, float]
    per_question_scores: dict[str, dict[str, float]]
    decision: str
    created_at: str


def select_reference_blind_fallback_evidence(
    *,
    question: str,
    selected_evidence: Sequence[dict[str, Any] | Any],
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    switch_margin: float = DEFAULT_SWITCH_MARGIN,
) -> SelectionDecision:
    """Select the best evidence item using reference-blind question trigram coverage.

    Only question and selected_evidence participate. Reference answers must NEVER
    be passed to this function.

    Evidence scoring text = evidence.text + " " + (evidence.article_title or "")
    Coverage = len(question_ngrams & evidence_ngrams) / len(question_ngrams)

    Tie-break rules:
    a. higher trigram coverage
    b. lower original retrieval_rank
    c. lower selected-evidence list position (index)
    d. evidence_id lexical order

    Switch away from E1 ONLY IF:
    best.evidence_id != 'E1' AND best_coverage - e1_coverage >= switch_margin
    """
    if not selected_evidence:
        raise DataValidationError("selected_evidence sequence cannot be empty")

    q_ngrams = extract_ngrams(question, n=ngram_size)
    q_len = len(q_ngrams)

    e1_coverage = 0.0
    ev_candidates = []

    for idx, ev in enumerate(selected_evidence):
        if isinstance(ev, dict):
            ev_id = ev.get("evidence_id") or f"E{idx+1}"
            ev_text = ev.get("text", "")
            art_title = ev.get("article_title") or ""
            meta = ev.get("metadata") or {}
            retrieval_rank = meta.get("retrieval_rank")
            if retrieval_rank is None:
                retrieval_rank = ev.get("retrieval_rank", 999999)
        else:
            ev_id = getattr(ev, "evidence_id", f"E{idx+1}")
            ev_text = getattr(ev, "text", "")
            art_title = getattr(ev, "article_title", None) or ""
            meta = getattr(ev, "metadata", None) or {}
            if isinstance(meta, dict):
                retrieval_rank = meta.get("retrieval_rank")
            else:
                retrieval_rank = None
            if retrieval_rank is None:
                retrieval_rank = getattr(ev, "retrieval_rank", 999999)

        scoring_text = f"{ev_text} {art_title}".strip()
        ev_ngrams = extract_ngrams(scoring_text, n=ngram_size)
        cov = len(q_ngrams & ev_ngrams) / q_len if q_len > 0 else 0.0

        if ev_id == "E1":
            e1_coverage = cov

        ev_candidates.append((cov, retrieval_rank, idx, str(ev_id)))

    # Sort key:
    # 1. -coverage (descending)
    # 2. retrieval_rank (ascending)
    # 3. index in list (ascending)
    # 4. evidence_id (ascending lexical)
    ev_candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))

    best_cov, best_rank, best_idx, best_ev_id = ev_candidates[0]
    margin = best_cov - e1_coverage

    switched = (best_ev_id != "E1") and (margin >= switch_margin)
    final_ev_id = best_ev_id if switched else "E1"
    final_cov = best_cov if switched else e1_coverage
    final_margin = margin if switched else 0.0

    return SelectionDecision(
        selected_evidence_id=final_ev_id,
        e1_coverage=e1_coverage,
        selected_coverage=final_cov,
        coverage_margin=final_margin,
        switched=switched,
        ngram_size=ngram_size,
        switch_margin=switch_margin,
    )


class T57CFallbackSelectorReplayRunner:
    """Orchestrates verified offline replay of reference-blind fallback selection."""

    def __init__(
        self,
        *,
        generation_archive: Path,
        evidence_archive: Path,
        scorer_archive: Path,
        output_path: Path,
        ngram_size: int = DEFAULT_NGRAM_SIZE,
        switch_margin: float = DEFAULT_SWITCH_MARGIN,
    ) -> None:
        self.generation_archive = generation_archive.resolve()
        self.evidence_archive = evidence_archive.resolve()
        self.scorer_archive = scorer_archive.resolve()
        self.output_path = output_path.resolve()
        self.ngram_size = ngram_size
        self.switch_margin = switch_margin

    def verify_artifact_authorities(self) -> None:
        """Verify SHA-256 hashes of all required offline archives."""
        start_t = time.time()
        print(f"[ARTIFACT] 1/3 Verifying generation archive: {self.generation_archive.name}")
        gen_sha = scm.compute_file_sha256(self.generation_archive)
        if gen_sha != EXPECTED_GENERATION_ARCHIVE_SHA256:
            raise ArtifactCompatibilityError(
                f"T5_7C_GENERATION_ARCHIVE_SHA_MISMATCH: expected {EXPECTED_GENERATION_ARCHIVE_SHA256}, got {gen_sha}"
            )

        print(f"[ARTIFACT] 2/3 Verifying FAST30 evidence archive: {self.evidence_archive.name}")
        ev_sha = scm.compute_file_sha256(self.evidence_archive)
        if ev_sha != EXPECTED_FAST30_ARCHIVE_SHA256:
            raise ArtifactCompatibilityError(
                f"T5_7C_FAST30_ARCHIVE_SHA_MISMATCH: expected {EXPECTED_FAST30_ARCHIVE_SHA256}, got {ev_sha}"
            )

        print(f"[ARTIFACT] 3/3 Verifying official scorer archive: {self.scorer_archive.name}")
        scorer_sha = scm.compute_file_sha256(self.scorer_archive)
        if scorer_sha != EXPECTED_SCORER_ARCHIVE_SHA256:
            raise ArtifactCompatibilityError(
                f"T5_7C_SCORER_ARCHIVE_SHA_MISMATCH: expected {EXPECTED_SCORER_ARCHIVE_SHA256}, got {scorer_sha}"
            )

        elapsed = time.time() - start_t
        print(f"[ARTIFACT] Complete: all 3 archives verified in {elapsed:.2f}s")

    def run_replay(self) -> T57CReplayResult:
        """Execute reference-blind replay, verify switch set, and score officially."""
        total_start_t = time.time()

        # Step 1: Verify archives
        self.verify_artifact_authorities()

        # Step 2: Load FAST30 diagnostics
        print(f"[LOAD] Extracting diagnostics from {self.evidence_archive.name}")
        with ZipFile(self.evidence_archive, "r") as zf:
            diag_lines = [
                line
                for line in zf.read("diagnostics.jsonl").decode("utf-8").splitlines()
                if line.strip()
            ]
            fast30_diag: dict[str, dict[str, Any]] = {}
            for l in diag_lines:
                d = json.loads(l)
                fast30_diag[str(d["question_id"])] = d

        # Step 3: Load closed Control results from generation archive
        print(f"[LOAD] Extracting closed Control results from {self.generation_archive.name}")
        with tarfile.open(self.generation_archive, "r:gz") as tar:
            f = tar.extractfile("t5_6b_design_b/control/question_results.jsonl")
            if f is None:
                raise ArtifactCompatibilityError(
                    "control/question_results.jsonl not found in generation archive"
                )
            ctrl_lines = [
                line for line in f.read().decode("utf-8").splitlines() if line.strip()
            ]
            ctrl_results = [
                scm.QuestionMeasurementResult.model_validate(json.loads(line))
                for line in ctrl_lines
            ]

        # Step 4: Compute reference-blind selector decisions (REFERENCE FIREWALL)
        print(f"[SELECT] Computing reference-blind fallback selections across 20 Tune20 QIDs...")
        selector_decisions: dict[str, SelectionDecision] = {}
        switched_qids: list[str] = []
        candidate_responses: dict[str, AnswerResponse] = {}
        control_responses: dict[str, AnswerResponse] = {}

        for i, q_res in enumerate(ctrl_results):
            qid = q_res.question_id
            diag = fast30_diag.get(qid)
            if not diag:
                raise DataValidationError(f"Question ID {qid} missing from diagnostics")

            question_text = diag["question"]
            selected_evidence_list = diag["selected_evidence"]
            control_responses[qid] = q_res.response

            if q_res.final_generator_path == "MODEL_ERROR_FALLBACK":
                decision = select_reference_blind_fallback_evidence(
                    question=question_text,
                    selected_evidence=selected_evidence_list,
                    ngram_size=self.ngram_size,
                    switch_margin=self.switch_margin,
                )
                selector_decisions[qid] = decision

                if decision.switched:
                    switched_qids.append(qid)
                    # Find chosen evidence item
                    chosen_ev = next(
                        ev
                        for ev in selected_evidence_list
                        if ev["evidence_id"] == decision.selected_evidence_id
                    )
                    cand_resp = AnswerResponse(
                        question=question_text,
                        answer=f"[{chosen_ev['evidence_id']}] {chosen_ev['text']}",
                        citations=[
                            Citation(
                                evidence_id=chosen_ev["evidence_id"],
                                chunk_id=chosen_ev["chunk_id"],
                                document_id=chosen_ev["document_id"],
                            )
                        ],
                        warnings=[],
                        insufficient_evidence=False,
                        retrieval_strategy="hybrid_rerank",
                        trace_id=f"t5_7c_replay_{qid}",
                    )
                else:
                    # Retain E1
                    e1_ev = next(
                        ev
                        for ev in selected_evidence_list
                        if ev["evidence_id"] == "E1"
                    )
                    cand_resp = AnswerResponse(
                        question=question_text,
                        answer=f"[E1] {e1_ev['text']}",
                        citations=[
                            Citation(
                                evidence_id=e1_ev["evidence_id"],
                                chunk_id=e1_ev["chunk_id"],
                                document_id=e1_ev["document_id"],
                            )
                        ],
                        warnings=[],
                        insufficient_evidence=False,
                        retrieval_strategy="hybrid_rerank",
                        trace_id=f"t5_7c_replay_{qid}",
                    )
                candidate_responses[qid] = cand_resp
            else:
                # Non-fallback path: preserve exact Control response
                candidate_responses[qid] = q_res.response
                decision = SelectionDecision(
                    selected_evidence_id="E1",
                    e1_coverage=0.0,
                    selected_coverage=0.0,
                    coverage_margin=0.0,
                    switched=False,
                    ngram_size=self.ngram_size,
                    switch_margin=self.switch_margin,
                )
                selector_decisions[qid] = decision

            pct = ((i + 1) / len(ctrl_results)) * 100
            print(
                f"[SELECT] {i+1:02d}/{len(ctrl_results):02d} ({pct:3.0f}%) | QID={qid} | "
                f"switched={selector_decisions[qid].switched} | "
                f"ev={selector_decisions[qid].selected_evidence_id}"
            )

        # Step 5: Assert exact expected switch set & diagnostic coverages
        print(f"[ASSERT] Verifying exact switch set reproduction...")
        actual_switches = {
            qid: selector_decisions[qid].selected_evidence_id
            for qid in switched_qids
        }
        if actual_switches != EXPECTED_SWITCH_SET:
            raise ArtifactCompatibilityError(
                f"T5_7C_SELECTOR_REPRODUCTION_FAILED: expected switches {EXPECTED_SWITCH_SET}, got {actual_switches}"
            )

        for qid, exp_covs in EXPECTED_DIAGNOSTIC_COVERAGES.items():
            dec = selector_decisions[qid]
            if abs(dec.e1_coverage - exp_covs["E1"]) > 1e-9:
                raise ArtifactCompatibilityError(
                    f"T5_7C_SELECTOR_REPRODUCTION_FAILED: Q{qid} E1 cov expected {exp_covs['E1']}, got {dec.e1_coverage}"
                )
            if abs(dec.selected_coverage - exp_covs["selected"]) > 1e-9:
                raise ArtifactCompatibilityError(
                    f"T5_7C_SELECTOR_REPRODUCTION_FAILED: Q{qid} selected cov expected {exp_covs['selected']}, got {dec.selected_coverage}"
                )

        print(
            f"[ASSERT] Passed: exact switch set {list(actual_switches.keys())} "
            f"and diagnostic coverages verified."
        )

        # Step 6: Render predictions (strips [E#] competition markers)
        ctrl_preds = {
            qid: render_competition_answer(resp)
            for qid, resp in control_responses.items()
        }
        cand_preds = {
            qid: render_competition_answer(resp)
            for qid, resp in candidate_responses.items()
        }

        # Step 7: Load Tune20 reference answers AFTER selection is frozen (REFERENCE FIREWALL)
        print(f"[FIREWALL] Loading Tune20 reference answers for scoring...")
        with ZipFile(self.evidence_archive, "r") as zf:
            fast30_data = json.loads(
                zf.read("t5-1c-fast30-clean1.json").decode("utf-8")
            )
            reference_answers = {
                qid: fast30_data[qid]["answer"]
                for qid in scm.CANONICAL_TUNE20_ORDERED_QIDS
            }

        # Step 8: Score Control with exact official entrypoint
        print(f"[SCORE CONTROL] Executing official eval_qa on Control predictions...")
        t0 = time.time()
        ctrl_rouge, ctrl_meteor, ctrl_per_q, verified_scorer_sha = (
            scm.score_tune20_answers(
                predicted_answers=ctrl_preds,
                reference_answers=reference_answers,
                scorer_path=self.scorer_archive,
            )
        )
        print(
            f"[SCORE CONTROL] Complete in {time.time() - t0:.2f}s: "
            f"ROUGE-L={ctrl_rouge:.6f}, METEOR={ctrl_meteor:.6f}"
        )

        # Step 9: Score Candidate with exact official entrypoint
        print(f"[SCORE CANDIDATE] Executing official eval_qa on Candidate predictions...")
        t0 = time.time()
        cand_rouge, cand_meteor, cand_per_q, _ = scm.score_tune20_answers(
            predicted_answers=cand_preds,
            reference_answers=reference_answers,
            scorer_path=self.scorer_archive,
        )
        print(
            f"[SCORE CANDIDATE] Complete in {time.time() - t0:.2f}s: "
            f"ROUGE-L={cand_rouge:.6f}, METEOR={cand_meteor:.6f}"
        )

        # Step 10: Verify Control official reproduction
        expected_ctrl_rouge = 0.48313312484363263
        if abs(ctrl_rouge - expected_ctrl_rouge) > 1e-12:
            raise ArtifactCompatibilityError(
                f"T5_7C_CONTROL_REPRODUCTION_FAILED: expected ROUGE-L {expected_ctrl_rouge}, got {ctrl_rouge}"
            )

        # Step 11: Compute per-question deltas and evaluate decision rule
        per_q_combined: dict[str, dict[str, float]] = {}
        for qid in scm.CANONICAL_TUNE20_ORDERED_QIDS:
            c_m = ctrl_per_q[qid]
            k_m = cand_per_q[qid]
            per_q_combined[qid] = {
                "control_rouge": c_m["rouge_l"],
                "control_meteor": c_m["meteor"],
                "candidate_rouge": k_m["rouge_l"],
                "candidate_meteor": k_m["meteor"],
                "delta_rouge": k_m["rouge_l"] - c_m["rouge_l"],
                "delta_meteor": k_m["meteor"] - c_m["meteor"],
            }

        delta_rouge = cand_rouge - ctrl_rouge
        delta_meteor = cand_meteor - ctrl_meteor

        if cand_rouge > ctrl_rouge and cand_meteor >= ctrl_meteor:
            decision = "PROMISING_FALLBACK_SELECTOR_REPLAY_CONFIRMED"
        else:
            decision = "FALLBACK_SELECTOR_NOT_JUSTIFIED"

        result = T57CReplayResult(
            starting_authority="871ba6cea0d25abb27b38c845b51234f2a122e7c",
            generation_archive_sha256=scm.compute_file_sha256(self.generation_archive),
            fast30_archive_sha256=scm.compute_file_sha256(self.evidence_archive),
            official_scorer_archive_sha256=scm.compute_file_sha256(self.scorer_archive),
            official_scoring_py_sha256=scm.OFFICIAL_SCORING_PY_SHA256,
            ngram_size=self.ngram_size,
            switch_margin=self.switch_margin,
            reference_blind_selection=True,
            selector_decisions=selector_decisions,
            switch_qids=switched_qids,
            control={"rouge_l": ctrl_rouge, "meteor": ctrl_meteor},
            candidate={"rouge_l": cand_rouge, "meteor": cand_meteor},
            delta={"rouge_l": delta_rouge, "meteor": delta_meteor},
            per_question_scores=per_q_combined,
            decision=decision,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Write result artifact
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"[EXPORT] Saved replay result artifact to {self.output_path}")

        total_elapsed = time.time() - total_start_t
        print(f"[REPLAY] Finished full replay in {total_elapsed:.2f}s | Decision: {decision}")
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for T5-7C fallback selector replay."""
    parser = argparse.ArgumentParser(
        description="T5-7C Isolated Reference-Blind Fallback Selector Replay"
    )
    parser.add_argument(
        "--generation-archive",
        type=Path,
        default=Path("C:/Users/Nguyen/Downloads/t5_6b_design_b_generation_closed.tar.gz"),
        help="Path to closed T5-6B generation tar.gz archive",
    )
    parser.add_argument(
        "--evidence-archive",
        type=Path,
        default=Path("C:/Users/Nguyen/Downloads/t5-1c-fast30-clean1-evidence.zip"),
        help="Path to frozen FAST30 evidence zip archive",
    )
    parser.add_argument(
        "--scorer-archive",
        type=Path,
        default=Path("C:/Users/Nguyen/Downloads/Scoring-Program-Task-LegalQA.zip"),
        help="Path to official scorer zip archive",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("scratch/t5_7c_fallback_selector_replay_result.json"),
        help="Path to output result JSON artifact",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=DEFAULT_NGRAM_SIZE,
        help="Word n-gram size for coverage calculation (default: 3)",
    )
    parser.add_argument(
        "--switch-margin",
        type=float,
        default=DEFAULT_SWITCH_MARGIN,
        help="Coverage margin threshold to switch away from E1 (default: 0.20)",
    )
    return parser


def main() -> None:
    """CLI entrypoint for T5-7C replay runner."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    runner = T57CFallbackSelectorReplayRunner(
        generation_archive=args.generation_archive,
        evidence_archive=args.evidence_archive,
        scorer_archive=args.scorer_archive,
        output_path=args.output_path,
        ngram_size=args.ngram_size,
        switch_margin=args.switch_margin,
    )
    result = runner.run_replay()
    print(f"\nReplay Decision: {result.decision}")
    print(f"Control ROUGE-L:   {result.control['rouge_l']:.17f}")
    print(f"Control METEOR:    {result.control['meteor']:.17f}")
    print(f"Candidate ROUGE-L: {result.candidate['rouge_l']:.17f}")
    print(f"Candidate METEOR:  {result.candidate['meteor']:.17f}")
    print(f"Delta ROUGE-L:     {result.delta['rouge_l']:+.17f}")
    print(f"Delta METEOR:      {result.delta['meteor']:+.17f}")


if __name__ == "__main__":
    main()
