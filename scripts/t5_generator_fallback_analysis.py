"""T5-6A Generator Contract and Fallback Efficiency Analysis Tooling.

This module provides offline diagnostic forensic tools to reconstruct generator execution paths,
analyze draft-rejection distributions, audit extractive fallback exact evidence identity,
derive canonical T5-6B generator input authorities, and enforce strict separation between
descriptive FAST30 census and Tune20-only policy discovery.

ORACLE DIAGNOSTIC WARNING:
Any functions computing reference-answer overlap (F1, recall, length ratio, ROUGE/METEOR) are for
DIAGNOSTIC / OFFLINE ANALYSIS ONLY. They represent oracle proxy metrics and must NEVER be used
as serving-time features in online answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import zipfile

from legal_agentic_rag.configuration.online import ClaimVerificationConfig, GenerationConfig

EXPECTED_FAST30_ARCHIVE_SHA256 = "be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f"
T5_6B_TUNE20_ORDERED_QIDS_SHA256 = "9cb88a00c2bcf9fbc0f24411de2f427d6a30f5da0f57feaaafb629f9fcd60b28"
T5_6B_FROZEN_GENERATOR_INPUT_SHA256 = "2fefbb03125f9927edf67c8bc8c165bdd856e1dd2eef0c737aefc7387a2cbbf2"
TUNE20_HISTORICAL_ROUGE_L = 0.4831331248436325
TUNE20_HISTORICAL_METEOR = 0.4046940181246421
PREREGISTERED_PRODUCTION_GENERATOR_BLOB_SHA = "1543eac766c0cf24ccb7904d8bfa2b802547e3c5"

T5_6B_CONTROL_GENERATION_CONFIG_SHA256 = "657ee87bdeac212857e9ec199c9fe34d6f7975ff5078c2371e1e6c2dba8738a7"
T5_6B_COMPACT_GENERATION_CONFIG_SHA256 = "810142a8ebacca5331ec13f1777be7edb6d4357b61a1c155d36751049b91bab2"
T5_6B_JSON_GENERATION_CONFIG_SHA256 = "8c930f08131b9cc9e07f1427d21b1d5e96c38431ca2d65f1e080abf04989596f"
T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256 = "fcb8cd2e65b74407be42a312f80624bb2be996e1a79d6a9228758d0893f23988"

CANONICAL_TUNE20_ORDERED_QIDS = [
    "89271", "39207", "31523", "116553", "113579", "83501", "102061", "94975",
    "56533", "17179", "89881", "140337", "36411", "46497", "58651", "150817",
    "150207", "21011", "84363", "102303"
]


class GeneratorPathClassification(str, Enum):
    """Taxonomy of final generator execution paths."""
    SEMANTIC_SYNTHESIS_SUCCESS = "SEMANTIC_SYNTHESIS_SUCCESS"
    STRUCTURED_OUTPUT_REJECTION_MODEL_FALLBACK = "STRUCTURED_OUTPUT_REJECTION_MODEL_FALLBACK"
    OTHER_MODEL_ERROR_FALLBACK = "OTHER_MODEL_ERROR_FALLBACK"
    GROUNDING_REPAIR_SUCCESS = "GROUNDING_REPAIR_SUCCESS"
    SUPPORTED_CLAIM_SALVAGE = "SUPPORTED_CLAIM_SALVAGE"
    GROUNDING_EXTRACTIVE_FALLBACK = "GROUNDING_EXTRACTIVE_FALLBACK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


class FallbackReconstructionStatus(str, Enum):
    """Exact evidence/citation identity verification status for fallback answers."""
    EXACT_IDENTITY_MATCH = "EXACT_IDENTITY_MATCH"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"
    NOT_FALLBACK = "NOT_FALLBACK"


@dataclass(frozen=True)
class GeneratorRejectionDetail:
    """Details of a single draft rejection event."""
    error_type: str
    structured_output_attempt: int


@dataclass(frozen=True)
class GeneratorForensicPacket:
    """Forensic reconstruction of a single question generator outcome."""
    question_id: str
    question: str
    reference_answer: str
    split: str  # Tune20 or Holdout10
    path_classification: GeneratorPathClassification
    warnings: list[str]
    rejections: list[GeneratorRejectionDetail]
    is_model_error_fallback: bool
    is_grounding_fallback: bool
    is_any_extractive_fallback: bool
    is_insufficient_evidence: bool
    selected_evidence_count: int
    selected_top1_chunk_id: str | None
    fallback_evidence_count: int
    fallback_reconstruction_status: FallbackReconstructionStatus
    public_answer_char_count: int
    reference_answer_char_count: int
    length_ratio: float
    matches_e1_verbatim: bool
    rouge_l: float
    meteor: float


def compute_tune20_ordered_qids_sha256(qids: Sequence[str]) -> str:
    """Compute canonical SHA-256 for ordered Tune20 question IDs."""
    payload_json = json.dumps(list(qids), separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def extract_frozen_generator_inputs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract deterministic generator input payloads from diagnostic records."""
    frozen_list = []
    for r in records:
        frozen_list.append({
            "question_id": str(r["question_id"]),
            "question": r["question"],
            "selected_evidence": r.get("selected_evidence", []),
        })
    return frozen_list


def compute_frozen_generator_input_sha256(frozen_inputs: Sequence[dict[str, Any]]) -> str:
    """Compute canonical SHA-256 for frozen generator input sequence."""
    payload_json = json.dumps(list(frozen_inputs), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def derive_t5_6b_generator_input_authority(zip_path: Path) -> tuple[str, str, str]:
    """Verify actual archive bytes, extract Tune20 generator inputs, and return canonical hashes.
    
    Returns:
        (actual_archive_sha256, tune20_ordered_qids_sha256, frozen_generator_input_sha256)
    """
    actual_archive_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if actual_archive_sha != EXPECTED_FAST30_ARCHIVE_SHA256:
        raise ValueError(
            f"Archive SHA mismatch: expected {EXPECTED_FAST30_ARCHIVE_SHA256}, got {actual_archive_sha}"
        )
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("diagnostics.jsonl") as f:
            records = [json.loads(line.decode("utf-8")) for line in f]
    tune20_records = records[:20]
    qids = [str(r["question_id"]) for r in tune20_records]
    if qids != CANONICAL_TUNE20_ORDERED_QIDS:
        raise ValueError(f"Tune20 QIDs mismatch: expected {CANONICAL_TUNE20_ORDERED_QIDS}, got {qids}")
        
    tune20_qids_hash = compute_tune20_ordered_qids_sha256(qids)
    frozen_inputs = extract_frozen_generator_inputs(tune20_records)
    frozen_input_hash = compute_frozen_generator_input_sha256(frozen_inputs)
    return actual_archive_sha, tune20_qids_hash, frozen_input_hash


def compute_model_config_canonical_sha256(model: Any) -> str:
    """Compute deterministic canonical SHA-256 for a Pydantic configuration model."""
    data = model.model_dump(mode="json")
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_preregistered_generation_config(prompt_schema_mode: str = "plain_text_markers") -> GenerationConfig:
    """Construct validated GenerationConfig for the preregistered T5-6B experiment."""
    return GenerationConfig(
        max_context_tokens=6144,
        max_evidence=10,
        timeout_seconds=360.0,
        backend="transformers",
        model_name="/kaggle/working/m49-generator-merged",
        model_revision="e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b",
        device="cuda",
        torch_dtype="float16",
        model_loader="image_text_to_text",
        local_files_only=True,
        max_input_tokens=8192,
        temperature=0.0,
        max_output_tokens=1536,
        repetition_penalty=1.08,
        no_repeat_ngram_size=8,
        max_structured_output_retries=1,
        max_model_error_retries=1,
        model_failure_policy="top_evidence",
        max_grounding_repair_retries=1,
        grounding_failure_policy="supported_claims_or_top_evidence",
        extractive_fallback_max_evidence=1,
        salvage_rendering="standalone",
        prompt_schema_mode=prompt_schema_mode,  # type: ignore[arg-type]
        answer_style="competition_reference",
    )


def get_preregistered_claim_verification_config() -> ClaimVerificationConfig:
    """Construct validated ClaimVerificationConfig for the preregistered T5-6B experiment."""
    return ClaimVerificationConfig(
        enabled=True,
        require_inline_citations=False,
        minimum_lexical_support=0.2,
        minimum_claim_tokens=2,
        require_numeric_match=True,
        require_negation_match=True,
        max_claims=60,
    )


def verify_fallback_identity_reconstruction(
    pub_ans: str,
    citations: Sequence[dict[str, Any]],
    selected_evidence: Sequence[dict[str, Any]],
    is_fallback: bool,
) -> tuple[FallbackReconstructionStatus, int, bool]:
    """Verify exact ordered evidence ID, chunk ID, and text correspondence for fallback answers."""
    if not is_fallback:
        return FallbackReconstructionStatus.NOT_FALLBACK, 0, False
    if not selected_evidence:
        return FallbackReconstructionStatus.IDENTITY_MISMATCH, 0, False
    if not citations:
        return FallbackReconstructionStatus.IDENTITY_MISMATCH, 0, False
        
    num_cits = len(citations)
    if num_cits > len(selected_evidence):
        return FallbackReconstructionStatus.IDENTITY_MISMATCH, num_cits, False
        
    # Check duplicate citations
    cit_ev_ids = [c.get("evidence_id") for c in citations]
    cit_chunk_ids = [c.get("chunk_id") for c in citations]
    if len(cit_ev_ids) != len(set(cit_ev_ids)) or len(cit_chunk_ids) != len(set(cit_chunk_ids)):
        return FallbackReconstructionStatus.IDENTITY_MISMATCH, num_cits, False
        
    expected_parts = []
    for idx, cit in enumerate(citations):
        ev_match = selected_evidence[idx]
        if cit.get("evidence_id") != ev_match.get("evidence_id"):
            return FallbackReconstructionStatus.IDENTITY_MISMATCH, num_cits, False
        if cit.get("chunk_id") != ev_match.get("chunk_id"):
            return FallbackReconstructionStatus.IDENTITY_MISMATCH, num_cits, False
        ev_id = ev_match.get("evidence_id", "")
        ev_text = ev_match.get("text", "").strip()
        expected_parts.append(f"[{ev_id}] {ev_text}")
        
    expected_answer = "\n\n".join(expected_parts)
    if pub_ans.strip() != expected_answer.strip():
        return FallbackReconstructionStatus.IDENTITY_MISMATCH, num_cits, False
        
    is_n1_e1 = (
        num_cits == 1
        and selected_evidence[0].get("evidence_id") == "E1"
        and citations[0].get("evidence_id") == "E1"
    )
    return FallbackReconstructionStatus.EXACT_IDENTITY_MATCH, num_cits, is_n1_e1


def classify_generator_path_from_telemetry(
    record: dict[str, Any],
) -> tuple[GeneratorPathClassification, bool, bool, bool, bool]:
    """Classify generator path strictly using persisted booleans as primary authority with warning cross-check."""
    warnings = list(record.get("warnings", []))
    raw_rejections = record.get("generator_draft_rejections", [])
    
    # Primary Authority: Persisted Booleans
    p_model_fb = bool(record.get("is_generator_model_error_fallback", False))
    p_grounding_fb = bool(record.get("is_grounding_extractive_fallback", False))
    p_any_fb = bool(record.get("is_any_extractive_fallback", False))
    p_insufficient = bool(record.get("is_insufficient_evidence", False))
    
    # Secondary Evidence: Warnings
    w_model_fb = "generator_model_error_fallback" in warnings
    w_grounding_fb = "extractive_fallback_applied" in warnings
    w_salvage = "supported_claim_salvage_applied" in warnings
    w_grounding_repair = "grounding_repair_attempted" in warnings
    w_insufficient = "insufficient_context" in warnings or "insufficient_evidence" in warnings
    
    # Check for invalid/fabricated generator warnings
    has_unknown_generator_warning = any(
        ("fallback" in w or "generator" in w or "grounding" in w or "salvage" in w)
        and w not in {
            "generator_model_error_fallback",
            "extractive_fallback_applied",
            "supported_claim_salvage_applied",
            "grounding_repair_attempted",
            "generator_model_error_retried",
            "grounding_repair_model_error",
            "grounding_repair_unresolved",
        }
        for w in warnings
    )
    if has_unknown_generator_warning:
        return GeneratorPathClassification.AMBIGUOUS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    
    # Consistency Checks (Fail-Closed on Telemetry Mismatch)
    if p_model_fb != w_model_fb:
        return GeneratorPathClassification.AMBIGUOUS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    if p_grounding_fb != w_grounding_fb:
        return GeneratorPathClassification.AMBIGUOUS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    if p_any_fb != (p_model_fb or p_grounding_fb):
        return GeneratorPathClassification.AMBIGUOUS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    if p_insufficient != w_insufficient:
        return GeneratorPathClassification.AMBIGUOUS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
        
    if p_model_fb and (p_grounding_fb or w_salvage):
        return GeneratorPathClassification.AMBIGUOUS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
        
    if p_insufficient:
        return GeneratorPathClassification.INSUFFICIENT_EVIDENCE, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    elif p_model_fb:
        if any(rej.get("error_type") == "structured_output_schema" for rej in raw_rejections):
            return GeneratorPathClassification.STRUCTURED_OUTPUT_REJECTION_MODEL_FALLBACK, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
        return GeneratorPathClassification.OTHER_MODEL_ERROR_FALLBACK, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    elif p_grounding_fb:
        return GeneratorPathClassification.GROUNDING_EXTRACTIVE_FALLBACK, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    elif w_salvage:
        return GeneratorPathClassification.SUPPORTED_CLAIM_SALVAGE, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    elif w_grounding_repair:
        return GeneratorPathClassification.GROUNDING_REPAIR_SUCCESS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient
    else:
        return GeneratorPathClassification.SEMANTIC_SYNTHESIS_SUCCESS, p_model_fb, p_grounding_fb, p_any_fb, p_insufficient


def build_generator_forensic_packet(
    record: dict[str, Any],
    split: str,
) -> GeneratorForensicPacket:
    """Reconstruct exact generator forensic packet from a diagnostic record."""
    qid = str(record["question_id"])
    question = record["question"]
    ref_ans = record["reference_answer"]
    warnings = list(record.get("warnings", []))
    raw_rejections = record.get("generator_draft_rejections", [])
    sel_ev = record.get("selected_evidence", [])
    pub_resp = record.get("public_response", {})
    pub_ans = pub_resp.get("answer", "")
    cits = pub_resp.get("citations", [])
    
    # Check for duplicate evidence IDs or chunk IDs in selected evidence (fail-closed)
    ev_ids = [e.get("evidence_id") for e in sel_ev]
    chunk_ids = [e.get("chunk_id") for e in sel_ev]
    if len(ev_ids) != len(set(ev_ids)) or len(chunk_ids) != len(set(chunk_ids)):
        return GeneratorForensicPacket(
            question_id=qid,
            question=question,
            reference_answer=ref_ans,
            split=split,
            path_classification=GeneratorPathClassification.AMBIGUOUS,
            warnings=warnings,
            rejections=[],
            is_model_error_fallback=False,
            is_grounding_fallback=False,
            is_any_extractive_fallback=False,
            is_insufficient_evidence=False,
            selected_evidence_count=len(sel_ev),
            selected_top1_chunk_id=None,
            fallback_evidence_count=0,
            fallback_reconstruction_status=FallbackReconstructionStatus.IDENTITY_MISMATCH,
            public_answer_char_count=len(pub_ans),
            reference_answer_char_count=len(ref_ans),
            length_ratio=0.0,
            matches_e1_verbatim=False,
            rouge_l=record.get("rouge_l_score", 0.0),
            meteor=record.get("meteor_score", 0.0),
        )
        
    path_cls, is_model_fb, is_grounding_fb, is_any_fb, is_insufficient = classify_generator_path_from_telemetry(record)
    
    rejections = [
        GeneratorRejectionDetail(
            error_type=str(r.get("error_type", "UNKNOWN")),
            structured_output_attempt=int(r.get("structured_output_attempt", 0)),
        )
        for r in raw_rejections
    ]
    
    top1_chunk = sel_ev[0].get("chunk_id") if sel_ev else None
    
    # Exact Fallback Identity Reconstruction
    recon_status, fallback_count, matches_e1 = verify_fallback_identity_reconstruction(
        pub_ans,
        cits,
        sel_ev,
        is_fallback=is_any_fb,
    )
    
    ref_len = len(ref_ans)
    pub_len = len(pub_ans)
    ratio = (pub_len / ref_len) if ref_len > 0 else 0.0
    
    return GeneratorForensicPacket(
        question_id=qid,
        question=question,
        reference_answer=ref_ans,
        split=split,
        path_classification=path_cls,
        warnings=warnings,
        rejections=rejections,
        is_model_error_fallback=is_model_fb,
        is_grounding_fallback=is_grounding_fb,
        is_any_extractive_fallback=is_any_fb,
        is_insufficient_evidence=is_insufficient,
        selected_evidence_count=len(sel_ev),
        selected_top1_chunk_id=top1_chunk,
        fallback_evidence_count=fallback_count,
        fallback_reconstruction_status=recon_status,
        public_answer_char_count=pub_len,
        reference_answer_char_count=ref_len,
        length_ratio=ratio,
        matches_e1_verbatim=matches_e1,
        rouge_l=record.get("rouge_l_score", 0.0),
        meteor=record.get("meteor_score", 0.0),
    )


def load_fast30_generator_forensics(zip_path: Path) -> list[GeneratorForensicPacket]:
    """Load and parse FAST30 generator forensic packets with byte-level archive validation."""
    actual_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_FAST30_ARCHIVE_SHA256:
        raise ValueError(
            f"Archive SHA mismatch: expected {EXPECTED_FAST30_ARCHIVE_SHA256}, got {actual_hash}"
        )
        
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("diagnostics.jsonl") as f:
            records = [json.loads(line.decode("utf-8")) for line in f]
            
    packets: list[GeneratorForensicPacket] = []
    for idx, r in enumerate(records):
        split = "Tune20" if idx < 20 else "Holdout10"
        packets.append(build_generator_forensic_packet(r, split))
    return packets


def evaluate_tune_fallback_counterfactuals(
    packets: Sequence[GeneratorForensicPacket],
) -> dict[str, Any]:
    """Perform fallback policy exploration strictly on Tune20 and reject Holdout10 input."""
    if not all(p.split == "Tune20" for p in packets):
        raise ValueError(
            "Policy discovery requires exclusively Tune20 input. "
            "Holdout10 is contaminated and cannot be used for policy exploration."
        )
    if len(packets) == 0:
        raise ValueError("Packets list cannot be empty")
        
    tune_count = len(packets)
    tune_fallbacks = sum(1 for p in packets if p.is_any_extractive_fallback)
    tune_model_fallbacks = sum(1 for p in packets if p.is_model_error_fallback)
    tune_contract_rejection_fallbacks = sum(
        1 for p in packets
        if p.path_classification == GeneratorPathClassification.STRUCTURED_OUTPUT_REJECTION_MODEL_FALLBACK
    )
    tune_grounding_fallbacks = sum(1 for p in packets if p.is_grounding_fallback)
    tune_ambiguous = sum(1 for p in packets if p.path_classification == GeneratorPathClassification.AMBIGUOUS)
    
    if tune_contract_rejection_fallbacks > 0:
        decision = "NEW_CONTROLLED_GENERATOR_MEASUREMENT_REQUIRED"
        rationale = (
            f"Diagnostic population exhibits {tune_contract_rejection_fallbacks}/{tune_count} "
            f"contract rejection fallbacks. Because rejected raw completions were not persisted, "
            f"a new controlled generator measurement is required to evaluate candidate output contracts."
        )
    else:
        decision = "NO_CONTRACT_MEASUREMENT_TRIGGER_FROM_THIS_POPULATION"
        rationale = (
            f"Diagnostic population exhibits 0/{tune_count} contract rejection fallbacks. "
            f"No controlled generator contract measurement is triggered from this population."
        )
        
    return {
        "tune_count": tune_count,
        "tune_fallbacks": tune_fallbacks,
        "tune_model_fallbacks": tune_model_fallbacks,
        "tune_contract_rejection_fallbacks": tune_contract_rejection_fallbacks,
        "tune_grounding_fallbacks": tune_grounding_fallbacks,
        "tune_ambiguous": tune_ambiguous,
        "decision": decision,
        "rationale": rationale,
    }
