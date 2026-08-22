"""T5-5A Targeted Reranker Causal Investigation Tooling.

This module provides offline diagnostic forensic tools to reconstruct pre/post rerank events,
track exact candidate removal with strict score-space separation (retrieval score vs cross-encoder
logit), inspect score margins to cutoff, analyze rendered legal context inputs, and enforce
strict separation between forensic seed cases and Tune20-only policy discovery.

ORACLE DIAGNOSTIC WARNING:
Any functions computing reference-answer overlap (F1, recall, Jaccard) are for DIAGNOSTIC /
OFFLINE ANALYSIS ONLY. They represent oracle proxy metrics and must NEVER be conflated with
proven semantic ground-truth quality or used as serving-time features in online reranking.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence
import zipfile

EXPECTED_FAST30_ARCHIVE_SHA256 = "be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f"
EXPECTED_Q134499_BEST_PRE = "chunk_6dbd79b888078e5047434fe0"
EXPECTED_Q60281_BEST_PRE = "chunk_c8d1589d4db4ce08c92e67cb"


class RerankForensicClassification(str, Enum):
    """Evidence-scoped taxonomy of cross-encoder candidate outcome layers."""
    ORACLE_PROXY_RERANK_DROP = "ORACLE_PROXY_RERANK_DROP"
    SEMANTICALLY_PLAUSIBLE_RERANK_LOSS = "SEMANTICALLY_PLAUSIBLE_RERANK_LOSS"
    ORACLE_PROXY_FALSE_POSITIVE = "ORACLE_PROXY_FALSE_POSITIVE"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


@dataclass(frozen=True)
class DeployableCandidateMetadata:
    """Candidate metadata accessible at serving time with explicit score semantics."""
    chunk_id: str
    document_id: str
    pre_rerank_rank: int
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float | None
    retrieval_score: float | None
    reranker_score: float | None
    post_rerank_rank: int | None
    text_char_count: int
    document_title: str | None
    document_number: str | None
    article_number: str | None
    effect_status: str | None
    has_dual_branch_agreement: bool


@dataclass(frozen=True)
class DiagnosticOracleCandidate:
    """Oracle reference overlap metrics for offline forensic analysis only."""
    chunk_id: str
    reference_f1: float
    reference_recall: float


@dataclass(frozen=True)
class RerankForensicPacket:
    """Detailed forensic reconstruction of a single query rerank event."""
    question_id: str
    question: str
    reference_answer: str
    split: str  # Tune20 or Holdout10 (Forensic Seed)
    mapping_status: str  # UNIQUELY_MAPPED or AMBIGUOUS
    is_oracle_proxy_drop: bool
    forensic_classification: RerankForensicClassification
    best_pre_chunk_id: str | None
    best_pre_f1: float
    best_pre_rank: int | None
    best_post_chunk_id: str | None
    best_post_f1: float
    post_top1_chunk_id: str | None
    post_top1_f1: float
    f1_loss_gap: float
    post_top1_score: float | None
    lowest_retained_score: float | None
    dropped_candidate_score: float | None
    dropped_score_relation: str
    score_margin_to_cutoff: float | None
    top1_to_cutoff_margin: float | None
    candidates: list[DeployableCandidateMetadata]
    oracle_evals: list[DiagnosticOracleCandidate]


def tokenize_words(text: str) -> list[str]:
    """Deterministic lowercase word tokenization."""
    return re.findall(r"\w+", text.lower())


def compute_oracle_overlap_f1(hit_text: str, reference_text: str) -> float:
    """Calculate token F1 overlap against gold reference answer."""
    s_hit = set(tokenize_words(hit_text))
    s_ref = set(tokenize_words(reference_text))
    if not s_hit or not s_ref:
        return 0.0
    inter = len(s_hit & s_ref)
    prec = inter / len(s_hit)
    rec = inter / len(s_ref)
    return (2 * prec * rec / (prec + rec)) if (prec + rec > 0) else 0.0


def extract_candidate_metadata(
    pre_candidate: dict[str, Any],
    *,
    pre_rank: int,
    post_hit: dict[str, Any] | None,
    post_rank: int | None,
) -> DeployableCandidateMetadata:
    """Extract deployable metadata with strict retrieval vs reranker score separation."""
    meta = pre_candidate.get("metadata") or {}
    struct = meta.get("structure") or {}
    bm25_r = pre_candidate.get("bm25_rank")
    dense_r = pre_candidate.get("dense_rank")
    dual_agree = (bm25_r is not None and dense_r is not None and bm25_r <= 15 and dense_r <= 15)
    
    # 1. Retrieval score derived ONLY from pre-candidate retrieval fields
    ret_score = pre_candidate.get("rrf_score")
    if ret_score is None:
        ret_score = pre_candidate.get("score")
        
    # 2. Reranker score derived ONLY from matched post_hit
    rerank_score = None
    if post_hit is not None and post_hit.get("score") is not None:
        rerank_score = float(post_hit["score"])
            
    return DeployableCandidateMetadata(
        chunk_id=str(pre_candidate.get("chunk_id", "")),
        document_id=str(pre_candidate.get("document_id", "")),
        pre_rerank_rank=pre_rank,
        bm25_rank=bm25_r,
        dense_rank=dense_r,
        rrf_score=pre_candidate.get("rrf_score"),
        retrieval_score=float(ret_score) if ret_score is not None else None,
        reranker_score=rerank_score,
        post_rerank_rank=post_rank,
        text_char_count=len(pre_candidate.get("text", "")),
        document_title=meta.get("document_title"),
        document_number=meta.get("document_number"),
        article_number=struct.get("article_number"),
        effect_status=meta.get("effect_status"),
        has_dual_branch_agreement=dual_agree,
    )


def build_rerank_forensic_packet(
    record: dict[str, Any],
    split: str,
    *,
    forensic_classification_override: RerankForensicClassification | None = None,
) -> RerankForensicPacket:
    """Reconstruct exact pre/post rerank event and compute score-margin metrics safely."""
    qid = str(record["question_id"])
    question = record["question"]
    ref_ans = record["reference_answer"]
    term_hits = record.get("terminal_retrieval_hits", [])
    term_chunks = [h.get("chunk_id", "") for h in term_hits]
    
    # Check for duplicate chunk IDs in terminal hits (fail-closed)
    if len(term_chunks) != len(set(term_chunks)):
        return RerankForensicPacket(
            question_id=qid,
            question=question,
            reference_answer=ref_ans,
            split=split,
            mapping_status="AMBIGUOUS_DUPLICATE_CHUNKS",
            is_oracle_proxy_drop=False,
            forensic_classification=RerankForensicClassification.AMBIGUOUS,
            best_pre_chunk_id=None,
            best_pre_f1=0.0,
            best_pre_rank=None,
            best_post_chunk_id=None,
            best_post_f1=0.0,
            post_top1_chunk_id=None,
            post_top1_f1=0.0,
            f1_loss_gap=0.0,
            post_top1_score=None,
            lowest_retained_score=None,
            dropped_candidate_score=None,
            dropped_score_relation="NOT_ASSESSABLE",
            score_margin_to_cutoff=None,
            top1_to_cutoff_margin=None,
            candidates=[],
            oracle_evals=[],
        )
    
    rerank_tel = record.get("rerank_telemetry", [])
    if not rerank_tel or not isinstance(rerank_tel, list):
        return RerankForensicPacket(
            question_id=qid,
            question=question,
            reference_answer=ref_ans,
            split=split,
            mapping_status="AMBIGUOUS_MISSING_TELEMETRY",
            is_oracle_proxy_drop=False,
            forensic_classification=RerankForensicClassification.AMBIGUOUS,
            best_pre_chunk_id=None,
            best_pre_f1=0.0,
            best_pre_rank=None,
            best_post_chunk_id=None,
            best_post_f1=0.0,
            post_top1_chunk_id=None,
            post_top1_f1=0.0,
            f1_loss_gap=0.0,
            post_top1_score=None,
            lowest_retained_score=None,
            dropped_candidate_score=None,
            dropped_score_relation="NOT_ASSESSABLE",
            score_margin_to_cutoff=None,
            top1_to_cutoff_margin=None,
            candidates=[],
            oracle_evals=[],
        )
        
    matching_events = [
        ev for ev in rerank_tel
        if [h.get("chunk_id") for h in ev.get("post_rerank_hits", [])] == term_chunks
    ]
    if len(matching_events) != 1:
        return RerankForensicPacket(
            question_id=qid,
            question=question,
            reference_answer=ref_ans,
            split=split,
            mapping_status="AMBIGUOUS_NOT_UNIQUELY_MAPPED",
            is_oracle_proxy_drop=False,
            forensic_classification=RerankForensicClassification.AMBIGUOUS,
            best_pre_chunk_id=None,
            best_pre_f1=0.0,
            best_pre_rank=None,
            best_post_chunk_id=None,
            best_post_f1=0.0,
            post_top1_chunk_id=None,
            post_top1_f1=0.0,
            f1_loss_gap=0.0,
            post_top1_score=None,
            lowest_retained_score=None,
            dropped_candidate_score=None,
            dropped_score_relation="NOT_ASSESSABLE",
            score_margin_to_cutoff=None,
            top1_to_cutoff_margin=None,
            candidates=[],
            oracle_evals=[],
        )
        
    ev = matching_events[0]
    pre_cands = ev.get("pre_rerank_candidates", [])
    post_hits = ev.get("post_rerank_hits", [])
    
    # Check duplicate chunk IDs in pre_cands (fail-closed)
    pre_chunk_list = [c.get("chunk_id", "") for c in pre_cands]
    if len(pre_chunk_list) != len(set(pre_chunk_list)):
        return RerankForensicPacket(
            question_id=qid,
            question=question,
            reference_answer=ref_ans,
            split=split,
            mapping_status="AMBIGUOUS_DUPLICATE_PRE_CHUNKS",
            is_oracle_proxy_drop=False,
            forensic_classification=RerankForensicClassification.AMBIGUOUS,
            best_pre_chunk_id=None,
            best_pre_f1=0.0,
            best_pre_rank=None,
            best_post_chunk_id=None,
            best_post_f1=0.0,
            post_top1_chunk_id=None,
            post_top1_f1=0.0,
            f1_loss_gap=0.0,
            post_top1_score=None,
            lowest_retained_score=None,
            dropped_candidate_score=None,
            dropped_score_relation="NOT_ASSESSABLE",
            score_margin_to_cutoff=None,
            top1_to_cutoff_margin=None,
            candidates=[],
            oracle_evals=[],
        )
    
    post_hit_map = {h.get("chunk_id"): h for h in post_hits}
    post_rank_map = {h.get("chunk_id"): idx for idx, h in enumerate(post_hits, 1)}
    
    candidates: list[DeployableCandidateMetadata] = []
    oracle_evals: list[DiagnosticOracleCandidate] = []
    
    for idx, c in enumerate(pre_cands, 1):
        cid = str(c.get("chunk_id", ""))
        post_h = post_hit_map.get(cid)
        post_r = post_rank_map.get(cid)
        meta_dep = extract_candidate_metadata(
            c,
            pre_rank=idx,
            post_hit=post_h,
            post_rank=post_r,
        )
        f1 = compute_oracle_overlap_f1(c.get("text", ""), ref_ans)
        candidates.append(meta_dep)
        oracle_evals.append(DiagnosticOracleCandidate(chunk_id=cid, reference_f1=f1, reference_recall=0.0))
        
    pre_eval_pairs = [(c.chunk_id, o.reference_f1, c.pre_rerank_rank) for c, o in zip(candidates, oracle_evals)]
    post_eval_pairs = [(h.get("chunk_id"), compute_oracle_overlap_f1(h.get("text", ""), ref_ans)) for h in post_hits]
    
    best_pre_chunk, best_pre_f1, best_pre_r = max(pre_eval_pairs, key=lambda x: x[1]) if pre_eval_pairs else (None, 0.0, None)
    best_post_chunk, best_post_f1 = max(post_eval_pairs, key=lambda x: x[1]) if post_eval_pairs else (None, 0.0)
    
    post_top1_chunk = post_hits[0].get("chunk_id") if post_hits else None
    post_top1_f1 = compute_oracle_overlap_f1(post_hits[0].get("text", ""), ref_ans) if post_hits else 0.0
    
    post_chunk_ids = set(post_rank_map.keys())
    is_dropped = (best_pre_chunk not in post_chunk_ids) if best_pre_chunk else False
    f1_gap = max(0.0, best_pre_f1 - best_post_f1)
    
    is_oracle_proxy = (
        best_pre_f1 >= 0.40
        and is_dropped
        and f1_gap >= 0.08
    )
    
    # Forensic classification
    classification = forensic_classification_override
    if classification is None:
        if is_oracle_proxy:
            classification = RerankForensicClassification.ORACLE_PROXY_RERANK_DROP
        else:
            classification = RerankForensicClassification.NOT_ASSESSABLE
            
    # Score margins & cutoff
    post_scores = [float(h.get("score")) for h in post_hits if h.get("score") is not None]
    lowest_retained = min(post_scores) if post_scores else None
    top1_score = float(post_hits[0].get("score")) if (post_hits and post_hits[0].get("score") is not None) else None
    
    top1_to_cutoff = (top1_score - lowest_retained) if (top1_score is not None and lowest_retained is not None) else None
    
    # Dropped candidate cross-encoder score
    dropped_score = None
    dropped_relation = "NO_CANDIDATE_DROPPED"
    if is_dropped:
        dropped_relation = "AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED"
        dropped_meta = next((c for c in candidates if c.chunk_id == best_pre_chunk), None)
        if dropped_meta and dropped_meta.reranker_score is not None:
            dropped_score = dropped_meta.reranker_score
            
    score_margin = None
    if dropped_score is not None and lowest_retained is not None:
        score_margin = lowest_retained - dropped_score
        
    return RerankForensicPacket(
        question_id=qid,
        question=question,
        reference_answer=ref_ans,
        split=split,
        mapping_status="UNIQUELY_MAPPED",
        is_oracle_proxy_drop=is_oracle_proxy,
        forensic_classification=classification,
        best_pre_chunk_id=best_pre_chunk,
        best_pre_f1=best_pre_f1,
        best_pre_rank=best_pre_r,
        best_post_chunk_id=best_post_chunk,
        best_post_f1=best_post_f1,
        post_top1_chunk_id=post_top1_chunk,
        post_top1_f1=post_top1_f1,
        f1_loss_gap=f1_gap,
        post_top1_score=top1_score,
        lowest_retained_score=lowest_retained,
        dropped_candidate_score=dropped_score,
        dropped_score_relation=dropped_relation,
        score_margin_to_cutoff=score_margin,
        top1_to_cutoff_margin=top1_to_cutoff,
        candidates=candidates,
        oracle_evals=oracle_evals,
    )


def apply_authority_bound_forensic_annotations(
    packet: RerankForensicPacket,
    *,
    actual_archive_sha256: str,
) -> RerankForensicPacket:
    """Attach semantic forensic annotations strictly when archive authority and identities match."""
    if actual_archive_sha256 != EXPECTED_FAST30_ARCHIVE_SHA256:
        return packet
    if packet.split != "Holdout10" or packet.mapping_status != "UNIQUELY_MAPPED":
        return packet
    if not packet.is_oracle_proxy_drop:
        return packet
        
    if packet.question_id == "134499" and packet.best_pre_chunk_id == EXPECTED_Q134499_BEST_PRE:
        return dataclasses.replace(
            packet,
            forensic_classification=RerankForensicClassification.SEMANTICALLY_PLAUSIBLE_RERANK_LOSS,
        )
    elif packet.question_id == "60281" and packet.best_pre_chunk_id == EXPECTED_Q60281_BEST_PRE:
        return dataclasses.replace(
            packet,
            forensic_classification=RerankForensicClassification.ORACLE_PROXY_FALSE_POSITIVE,
        )
    return packet


def load_fast30_rerank_forensics(
    zip_path: Path,
) -> list[RerankForensicPacket]:
    """Load and reconstruct rerank packets with authority-bound forensic semantic annotations."""
    # ALWAYS calculate actual archive SHA-256 directly from file bytes
    actual_archive_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("diagnostics.jsonl") as f:
            records = [json.loads(line.decode("utf-8")) for line in f]
            
    packets: list[RerankForensicPacket] = []
    for idx, r in enumerate(records):
        qid = str(r.get("question_id"))
        split = "Tune20" if idx < 20 else "Holdout10"
        
        # Build base telemetry packet first
        base_packet = build_rerank_forensic_packet(r, split)
        
        # Apply authority-bound forensic annotations via actual-byte hash
        annotated_packet = apply_authority_bound_forensic_annotations(
            base_packet,
            actual_archive_sha256=actual_archive_sha256,
        )
        packets.append(annotated_packet)
    return packets


def evaluate_tune_reranker_policy_discovery(
    packets: Sequence[RerankForensicPacket],
) -> dict[str, Any]:
    """Perform policy discovery strictly on Tune20 and reject any Holdout10 input."""
    if not all(p.split == "Tune20" for p in packets):
        raise ValueError(
            "Policy discovery requires exclusively Tune20 input. "
            "Holdout10 contains contaminated forensic seeds and cannot be used for policy discovery."
        )
    if len(packets) == 0:
        raise ValueError("Packets list cannot be empty")
        
    tune_proxy_drops = [p for p in packets if p.is_oracle_proxy_drop]
    tune_semantic_losses = [
        p for p in packets
        if p.forensic_classification == RerankForensicClassification.SEMANTICALLY_PLAUSIBLE_RERANK_LOSS
    ]
    
    if len(tune_proxy_drops) == 0 and len(tune_semantic_losses) == 0:
        decision = "NO_RERANK_POLICY_JUSTIFIED"
        rationale = (
            "Tune20 exhibits 0 oracle-proxy drops and 0 semantically plausible rerank-loss "
            "annotations. No deployable reranking policy can be causally justified without "
            "overfitting contaminated forensic seeds."
        )
    else:
        decision = "FURTHER_CAUSAL_INVESTIGATION_REQUIRED"
        rationale = (
            f"Tune20 exhibits {len(tune_proxy_drops)} oracle-proxy drops and "
            f"{len(tune_semantic_losses)} semantically plausible losses requiring deeper investigation."
        )
    
    return {
        "tune_count": len(packets),
        "tune_oracle_proxy_drops": len(tune_proxy_drops),
        "tune_semantically_plausible_losses": len(tune_semantic_losses),
        "decision": decision,
        "rationale": rationale,
    }
