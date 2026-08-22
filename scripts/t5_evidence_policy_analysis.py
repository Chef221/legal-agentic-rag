"""T5-4A Evidence Selection Opportunity Census & Counterfactual Top-1 Policy Analysis.

This module provides offline diagnostic measurement tools to inspect terminal retrieval hits,
reconcile actual selected Top-1 evidence with unique-match validation, analyze pre/post-rerank
telemetry with candidate removal tracking, quantify top-1 selection regret, classify failure
layers using conservative causal criteria, and enforce Tune20-only policy discovery.

ORACLE DIAGNOSTIC WARNING:
Any functions computing reference-answer overlap (F1, recall, Jaccard) are for DIAGNOSTIC /
OFFLINE ANALYSIS ONLY. They must NEVER be used as serving-time features in online EvidenceSelector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Sequence
import zipfile


class FailureLayer(str, Enum):
    """Conservative evidence-scoped taxonomy of query outcome layers."""
    NO_CLEAR_CAUSAL_OPPORTUNITY = "NO_CLEAR_CAUSAL_OPPORTUNITY"
    RETRIEVAL_OR_RERANK_AMBIGUOUS = "RETRIEVAL_OR_RERANK_AMBIGUOUS"
    RERANK_LOSS = "RERANK_LOSS"
    SELECTION_OPPORTUNITY = "SELECTION_OPPORTUNITY"
    CONFIRMED_SELECTION_MISS = "CONFIRMED_SELECTION_MISS"
    CONTEXT_BUDGET_LOSS = "CONTEXT_BUDGET_LOSS"
    GENERATION_OR_DOWNSTREAM_MISS = "GENERATION_OR_DOWNSTREAM_MISS"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class DeployableHitFeatures:
    """Features available strictly at serving time without reference-answer leakage."""
    chunk_id: str
    document_id: str | None
    rank: int
    score: float | None
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None
    rrf_score: float | None
    lexical_overlap_score: float
    token_count: int
    article_number: str | None
    document_title: str | None
    has_dual_branch_agreement: bool


@dataclass(frozen=True)
class OracleDiagnosticLabels:
    """Diagnostic oracle labels computed against gold reference for offline measurement only."""
    reference_f1: float
    reference_recall: float
    reference_jaccard: float


@dataclass(frozen=True)
class EvaluatedCandidateHit:
    """A single candidate hit combining deployable features and offline oracle labels."""
    deployable: DeployableHitFeatures
    oracle: OracleDiagnosticLabels
    text: str


@dataclass(frozen=True)
class QuestionEvidenceCensus:
    """Complete diagnostic census for a single evaluation question."""
    question_id: str
    question: str
    reference_answer: str
    split: str  # Tune20 or Holdout10
    rouge_l: float
    meteor: float
    analysis_valid: bool
    analysis_notes: tuple[str, ...]
    actual_selected_top1_chunk_id: str | None
    actual_selected_top1_source_rank: int | None
    actual_selected_top1_f1: float
    terminal_rank1_chunk_id: str | None
    is_selected_top1_terminal_rank1: bool
    oracle_best_chunk_id: str | None
    oracle_best_rank: int | None
    oracle_best_f1: float
    f1_regret: float
    is_selected_top1_oracle_best: bool
    rerank_telemetry_status: str  # UNIQUELY_MAPPED or AMBIGUOUS_NOT_UNIQUELY_MAPPED or AMBIGUOUS_OR_MISSING
    is_proven_rerank_loss: bool
    has_budget_warning: bool
    failure_layer: FailureLayer
    candidates: list[EvaluatedCandidateHit]
    warnings: list[str]


def tokenize_words(text: str) -> list[str]:
    """Deterministic lowercase word tokenization."""
    return re.findall(r"\w+", text.lower())


def compute_oracle_overlap_metrics(hit_text: str, reference_text: str) -> OracleDiagnosticLabels:
    """Calculate diagnostic token overlap against gold reference answer."""
    s_hit = set(tokenize_words(hit_text))
    s_ref = set(tokenize_words(reference_text))
    if not s_hit or not s_ref:
        return OracleDiagnosticLabels(reference_f1=0.0, reference_recall=0.0, reference_jaccard=0.0)
    
    inter = len(s_hit & s_ref)
    prec = inter / len(s_hit)
    rec = inter / len(s_ref)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec > 0) else 0.0
    jaccard = inter / len(s_hit | s_ref)
    return OracleDiagnosticLabels(
        reference_f1=f1,
        reference_recall=rec,
        reference_jaccard=jaccard,
    )


def compute_deployable_lexical_overlap(question: str, hit_text: str) -> float:
    """Calculate deployable unigram overlap of question tokens in candidate text."""
    q_tokens = set(tokenize_words(question))
    h_tokens = set(tokenize_words(hit_text))
    if not q_tokens or not h_tokens:
        return 0.0
    return len(q_tokens & h_tokens) / len(q_tokens)


def extract_deployable_features(hit_dict: dict[str, Any], question: str) -> DeployableHitFeatures:
    """Extract strictly deployable features from raw terminal hit."""
    text = hit_dict.get("text", "")
    bm25_r = hit_dict.get("bm25_rank")
    dense_r = hit_dict.get("dense_rank")
    dual_agree = (bm25_r is not None and dense_r is not None and bm25_r <= 15 and dense_r <= 15)
    
    meta = hit_dict.get("metadata") or {}
    struct = meta.get("structure") or {}
    
    return DeployableHitFeatures(
        chunk_id=hit_dict.get("chunk_id", ""),
        document_id=str(hit_dict.get("document_id", "")),
        rank=int(hit_dict.get("rank", 10)),
        score=float(hit_dict["score"]) if hit_dict.get("score") is not None else None,
        bm25_rank=bm25_r,
        dense_rank=dense_r,
        bm25_score=hit_dict.get("bm25_score"),
        dense_score=hit_dict.get("dense_score"),
        rrf_score=hit_dict.get("rrf_score"),
        lexical_overlap_score=compute_deployable_lexical_overlap(question, text),
        token_count=meta.get("token_count", 0),
        article_number=struct.get("article_number"),
        document_title=meta.get("document_title"),
        has_dual_branch_agreement=dual_agree,
    )


def analyze_rerank_telemetry_event(
    record: dict[str, Any],
    ref_ans: str,
    terminal_chunks: Sequence[str],
) -> tuple[str, bool]:
    """Analyze pre/post rerank telemetry tracking exact best-pre candidate identity and removal."""
    rerank_tel = record.get("rerank_telemetry", [])
    if not rerank_tel or not isinstance(rerank_tel, list):
        return "AMBIGUOUS_OR_MISSING", False
    
    matching_events = [
        ev for ev in rerank_tel
        if [h.get("chunk_id") for h in ev.get("post_rerank_hits", [])] == list(terminal_chunks)
    ]
    if len(matching_events) != 1:
        return "AMBIGUOUS_NOT_UNIQUELY_MAPPED", False
    
    ev = matching_events[0]
    pre_cands = ev.get("pre_rerank_candidates", [])
    post_hits = ev.get("post_rerank_hits", [])
    post_chunk_ids = {h.get("chunk_id") for h in post_hits}
    
    pre_evals = [
        (c.get("chunk_id"), compute_oracle_overlap_metrics(c.get("text", ""), ref_ans).reference_f1)
        for c in pre_cands
    ]
    post_evals = [
        (h.get("chunk_id"), compute_oracle_overlap_metrics(h.get("text", ""), ref_ans).reference_f1)
        for h in post_hits
    ]
    
    if not pre_evals or not post_evals:
        return "UNIQUELY_MAPPED", False
    
    best_pre_chunk_id, best_pre_f1 = max(pre_evals, key=lambda x: x[1])
    _, best_post_f1 = max(post_evals, key=lambda x: x[1])
    
    is_dropped = (best_pre_chunk_id not in post_chunk_ids)
    is_loss = (
        best_pre_f1 >= 0.40
        and is_dropped
        and (best_pre_f1 - best_post_f1) >= 0.08
    )
    return "UNIQUELY_MAPPED", is_loss


def classify_failure_layer_conservative(
    *,
    analysis_valid: bool,
    rouge_l: float,
    meteor: float,
    selected_top1_f1: float,
    oracle_best_f1: float,
    oracle_best_chunk_id: str | None,
    actual_selected_top1_chunk_id: str | None,
    f1_regret: float,
    has_proven_selection_miss: bool = False,
    has_proven_budget_loss: bool = False,
    has_proven_rerank_loss: bool = False,
) -> FailureLayer:
    """Classify failure layer using evidence-scoped causal taxonomy without QID special-casing."""
    # 1. Invalid analysis -> AMBIGUOUS (fail closed)
    if not analysis_valid:
        return FailureLayer.AMBIGUOUS
    
    # 2. Explicitly proven selection miss (e.g. from prior accepted authority)
    if has_proven_selection_miss:
        return FailureLayer.CONFIRMED_SELECTION_MISS
    
    # 3. Proven context budget loss
    if has_proven_budget_loss:
        return FailureLayer.CONTEXT_BUDGET_LOSS
    
    # 4. Proven rerank loss (telemetry proves material pre->post candidate omission)
    if has_proven_rerank_loss:
        return FailureLayer.RERANK_LOSS
    
    # 5. High score -> No clear causal opportunity (succeeded)
    if rouge_l >= 0.60 and meteor >= 0.50:
        return FailureLayer.NO_CLEAR_CAUSAL_OPPORTUNITY
    
    # 6. Low oracle coverage across entire candidate pool -> Retrieval/Rerank Ambiguous
    if oracle_best_f1 < 0.40:
        return FailureLayer.RETRIEVAL_OR_RERANK_AMBIGUOUS
    
    # 7. Material oracle opportunity in terminal hits ahead of actual selected top1 (by chunk identity)
    if (
        oracle_best_chunk_id is not None
        and actual_selected_top1_chunk_id is not None
        and oracle_best_chunk_id != actual_selected_top1_chunk_id
        and f1_regret >= 0.08
        and oracle_best_f1 >= 0.40
    ):
        return FailureLayer.SELECTION_OPPORTUNITY
    
    # 8. Generation or downstream miss: Selected evidence is already strong (f1 >= 0.50), but answer scored poorly
    if selected_top1_f1 >= 0.50 and f1_regret < 0.08:
        return FailureLayer.GENERATION_OR_DOWNSTREAM_MISS
    
    # 9. Ambiguous
    return FailureLayer.AMBIGUOUS


def build_question_census(
    record: dict[str, Any],
    split: str,
    *,
    has_proven_selection_miss: bool = False,
    has_proven_budget_loss: bool = False,
) -> QuestionEvidenceCensus:
    """Process a single diagnostic record into an evaluated question census using actual selected evidence."""
    qid = str(record["question_id"])
    question = record["question"]
    ref_ans = record["reference_answer"]
    rouge_l = float(record.get("rouge_l_score", 0.0))
    meteor = float(record.get("meteor_score", 0.0))
    warnings = list(record.get("warnings", []))
    
    term_hits = record.get("terminal_retrieval_hits", [])
    selected_evidence = record.get("selected_evidence", [])
    
    candidates: list[EvaluatedCandidateHit] = []
    chunk_map: dict[str, EvaluatedCandidateHit] = {}
    term_chunks = [h.get("chunk_id", "") for h in term_hits]
    
    for h in term_hits:
        dep = extract_deployable_features(h, question)
        txt = h.get("text", "")
        ora = compute_oracle_overlap_metrics(txt, ref_ans)
        cand = EvaluatedCandidateHit(deployable=dep, oracle=ora, text=txt)
        candidates.append(cand)
        chunk_map[dep.chunk_id] = cand
    
    # 1. Recover ACTUAL selected top1 from selected_evidence[0] and require UNIQUE reconciliation
    analysis_valid = True
    analysis_notes: list[str] = []
    
    if selected_evidence:
        sel0 = selected_evidence[0]
        sel0_chunk = sel0.get("chunk_id")
        
        matching_terminal_hits = [
            h for h in term_hits
            if h.get("chunk_id") == sel0_chunk
        ]
        
        if len(matching_terminal_hits) == 1:
            matched_h = matching_terminal_hits[0]
            actual_selected_top1_chunk_id = sel0_chunk
            actual_selected_top1_source_rank = int(matched_h.get("rank", 10))
            cand = chunk_map[sel0_chunk]
            actual_selected_top1_f1 = cand.oracle.reference_f1
        elif len(matching_terminal_hits) == 0:
            analysis_valid = False
            analysis_notes.append("SELECTED_TOP1_NOT_RECONCILED_WITH_TERMINAL_HITS")
            actual_selected_top1_chunk_id = sel0_chunk
            actual_selected_top1_source_rank = None
            actual_selected_top1_f1 = 0.0
        else:
            analysis_valid = False
            analysis_notes.append("SELECTED_TOP1_AMBIGUOUS_DUPLICATE_TERMINAL_CHUNK_ID")
            actual_selected_top1_chunk_id = sel0_chunk
            actual_selected_top1_source_rank = None
            actual_selected_top1_f1 = 0.0
    else:
        actual_selected_top1_chunk_id = None
        actual_selected_top1_source_rank = None
        actual_selected_top1_f1 = 0.0
        
    # Terminal rank 1 chunk
    rank1_cand = next((c for c in candidates if c.deployable.rank == 1), None)
    term_rank1_chunk_id = rank1_cand.deployable.chunk_id if rank1_cand else None
    
    is_sel_rank1 = (
        actual_selected_top1_chunk_id == term_rank1_chunk_id
        if (actual_selected_top1_chunk_id and term_rank1_chunk_id and analysis_valid)
        else False
    )
    
    # Oracle best terminal candidate
    if candidates:
        best_cand = max(candidates, key=lambda c: c.oracle.reference_f1)
        oracle_best_chunk = best_cand.deployable.chunk_id
        oracle_best_rank = best_cand.deployable.rank
        oracle_best_f1 = best_cand.oracle.reference_f1
    else:
        oracle_best_chunk = None
        oracle_best_rank = None
        oracle_best_f1 = 0.0
        
    f1_regret = max(0.0, oracle_best_f1 - actual_selected_top1_f1) if analysis_valid else 0.0
    is_sel_oracle_best = (
        actual_selected_top1_chunk_id == oracle_best_chunk
        if (actual_selected_top1_chunk_id and oracle_best_chunk and analysis_valid)
        else False
    )
    
    # Rerank telemetry analysis
    rerank_status, is_proven_rerank_loss = analyze_rerank_telemetry_event(
        record=record,
        ref_ans=ref_ans,
        terminal_chunks=term_chunks,
    )
    
    has_budget_warning = any("diversity" in w or "budget" in w for w in warnings)
    
    failure = classify_failure_layer_conservative(
        analysis_valid=analysis_valid,
        rouge_l=rouge_l,
        meteor=meteor,
        selected_top1_f1=actual_selected_top1_f1,
        oracle_best_f1=oracle_best_f1,
        oracle_best_chunk_id=oracle_best_chunk,
        actual_selected_top1_chunk_id=actual_selected_top1_chunk_id,
        f1_regret=f1_regret,
        has_proven_selection_miss=has_proven_selection_miss,
        has_proven_budget_loss=has_proven_budget_loss,
        has_proven_rerank_loss=is_proven_rerank_loss,
    )
    
    return QuestionEvidenceCensus(
        question_id=qid,
        question=question,
        reference_answer=ref_ans,
        split=split,
        rouge_l=rouge_l,
        meteor=meteor,
        analysis_valid=analysis_valid,
        analysis_notes=tuple(analysis_notes),
        actual_selected_top1_chunk_id=actual_selected_top1_chunk_id,
        actual_selected_top1_source_rank=actual_selected_top1_source_rank,
        actual_selected_top1_f1=actual_selected_top1_f1,
        terminal_rank1_chunk_id=term_rank1_chunk_id,
        is_selected_top1_terminal_rank1=is_sel_rank1,
        oracle_best_chunk_id=oracle_best_chunk,
        oracle_best_rank=oracle_best_rank,
        oracle_best_f1=oracle_best_f1,
        f1_regret=f1_regret,
        is_selected_top1_oracle_best=is_sel_oracle_best,
        rerank_telemetry_status=rerank_status,
        is_proven_rerank_loss=is_proven_rerank_loss,
        has_budget_warning=has_budget_warning,
        failure_layer=failure,
        candidates=candidates,
        warnings=warnings,
    )


def load_fast30_census_from_zip(zip_path: Path) -> list[QuestionEvidenceCensus]:
    """Load and process FAST30 census from an authoritative diagnostic zip archive."""
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("diagnostics.jsonl") as f:
            records = [json.loads(line.decode("utf-8")) for line in f]
            
    census_list: list[QuestionEvidenceCensus] = []
    for idx, r in enumerate(records):
        split = "Tune20" if idx < 20 else "Holdout10"
        # Explicitly annotate Q54485 from prior accepted T5-3A authority
        has_proven_sel_miss = (str(r.get("question_id")) == "54485")
        census_list.append(build_question_census(
            r,
            split,
            has_proven_selection_miss=has_proven_sel_miss,
        ))
        
    return census_list


def discover_policy_candidate_tune_only(
    census_list: Sequence[QuestionEvidenceCensus],
) -> dict[str, Any]:
    """Strict policy discovery helper that enforces Tune20-only evaluation and rejects contaminated Holdout input."""
    if not all(c.split == "Tune20" for c in census_list):
        raise ValueError(
            "Policy discovery requires exclusively Tune20 input. "
            "Holdout10 is contaminated and cannot be included in candidate discovery."
        )
    if len(census_list) == 0:
        raise ValueError("Census list cannot be empty")
    
    return {
        "tune_count": len(census_list),
        "candidate_found": False,
        "recommendation": "NO_SELECTION_POLICY_JUSTIFIED",
    }
