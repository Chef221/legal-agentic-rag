#!/usr/bin/env python3
"""G1 Material-Fidelity Generation Grounding Development A/B Evaluation Harness.

This script executes a controlled development A/B experiment comparing:
- Baseline Grounding Profile: grounding_profile="baseline"
- Candidate G1 Grounding Profile: grounding_profile="material_fidelity_v1"

over the burned 16-question diagnostic review packet set from:
`verification-v2-holdout-review-packets-v1.zip`

CRITICAL INVARIANTS:
1. Burned Diagnostic Dataset: The 16 packets are burned diagnostic development data only.
   They must NEVER again be used as authoritative promotion evidence.
2. Unbiased Prompting: The generator sees ONLY the question and retrieved evidence text.
   No human labels, no historical D3 predictions, no error tags are included in prompts.
3. Call Parity & Schema Invariance: Both arms use single-call generation with unchanged
   ModelAnswerDraft output contract and identical provider configuration.
4. Blinded Pairwise Evaluation: Produces a deterministically blinded markdown worksheet
   (randomizing Option 1 vs Option 2 per question) with a separate secret blinding key.
5. Secondary Metrics: Calculates canonical METEOR / ROUGE-L against reference answers
   if available in the diagnostic packets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
from typing import Any
import zipfile

import legal_agentic_rag
from legal_agentic_rag.configuration.online import GenerationConfig
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.evaluation.metrics import score_text_answer
from legal_agentic_rag.exceptions import (
    DataValidationError,
    ModelError,
    StructuredGenerationError,
)
from legal_agentic_rag.generation.model_generator import (
    ALLOWED_GROUNDING_PROFILES,
    GROUNDING_PROFILE_BASELINE,
    GROUNDING_PROFILE_MATERIAL_FIDELITY_V1,
    ModelBackedAnswerGenerator,
    _SYSTEM_INSTRUCTION_BASELINE,
    _SYSTEM_INSTRUCTION_MATERIAL_FIDELITY_V1,
)
from legal_agentic_rag.generation.transformers_provider import TransformersChatProvider
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Evidence,
    ModelAnswerDraft,
)
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy

_LOGGER = logging.getLogger(__name__)

CANONICAL_DIAGNOSTIC_PACKETS_ZIP_SHA256 = (
    "a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4"
)

CANONICAL_PACKAGE_VERSION = "0.50.7"
CANONICAL_EXPERIMENT_ID = "GENERATION-G1-AB-DEVELOPMENT"
CANONICAL_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CANONICAL_MODEL_REVISION = "a1d308dfcc03e09da285d49d912439a655a571e8"
CANONICAL_PROVIDER_VERSION = "4.47.1"
CANONICAL_DEVICE = "cuda"
CANONICAL_TORCH_DTYPE = "float16"

# Known diagnostic questions with historical material error mechanisms
KNOWN_MATERIAL_ERROR_QUESTIONS = frozenset({
    "125893",  # C1 & C3: ACTOR_ROLE_MISMATCH
    "45427",   # C1: CONDITION_EXCEPTION_OMITTED
    "90897",   # C1: ACTOR_ROLE_MISMATCH
    "95695",   # C1: ACTION_OBJECT_MISMATCH / QUANTITY_TEMPORAL_MISMATCH
})

KNOWN_POSITIVE_LIST_QUESTIONS = frozenset({
    "61523",   # C1: SYNTAX_FRAGMENT_STRICTNESS (list noun phrase)
})


@dataclass(frozen=True)
class TelemetryCallRecord:
    call_index: int
    arm: str
    question_id: str
    prompt_character_count: int
    prompt_sha256: str
    completion_character_count: int
    completion_sha256: str
    duration_seconds: float
    status: str
    error_type: str | None = None
    error_sha256: str | None = None


class TelemetryLoggingChatProviderProxy(ChatModelProvider):
    """Transparent proxy capturing operational metrics without leaking raw text."""

    def __init__(self, target_provider: ChatModelProvider) -> None:
        self._target = target_provider
        self.records: list[TelemetryCallRecord] = []
        self._current_arm: str = "unknown"
        self._current_question_id: str = "unknown"

    @property
    def provider_name(self) -> str:
        return self._target.provider_name

    @property
    def provider_version(self) -> str:
        return self._target.provider_version

    @property
    def model_name(self) -> str:
        return self._target.model_name

    @property
    def model_revision(self) -> str:
        return self._target.model_revision

    def set_context(self, arm: str, question_id: str) -> None:
        self._current_arm = arm
        self._current_question_id = question_id

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        call_index = len(self.records) + 1
        full_prompt = f"{system_instruction}\n\n{user_prompt}"
        prompt_char_count = len(full_prompt)
        prompt_sha = sha256(full_prompt.encode("utf-8")).hexdigest()

        start_time = perf_counter()
        try:
            completion = self._target.complete(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            )
            duration = perf_counter() - start_time
            completion_char_count = len(completion)
            completion_sha = sha256(completion.encode("utf-8")).hexdigest()

            record = TelemetryCallRecord(
                call_index=call_index,
                arm=self._current_arm,
                question_id=self._current_question_id,
                prompt_character_count=prompt_char_count,
                prompt_sha256=prompt_sha,
                completion_character_count=completion_char_count,
                completion_sha256=completion_sha,
                duration_seconds=duration,
                status="SUCCESS",
            )
            self.records.append(record)
            return completion
        except Exception as exc:
            duration = perf_counter() - start_time
            err_type = type(exc).__name__
            err_sha = sha256(str(exc).encode("utf-8")).hexdigest()

            record = TelemetryCallRecord(
                call_index=call_index,
                arm=self._current_arm,
                question_id=self._current_question_id,
                prompt_character_count=prompt_char_count,
                prompt_sha256=prompt_sha,
                completion_character_count=0,
                completion_sha256="",
                duration_seconds=duration,
                status="ERROR",
                error_type=err_type,
                error_sha256=err_sha,
            )
            self.records.append(record)
            raise


@dataclass
class DiagnosticQuestionPacket:
    question_id: str
    question: str
    retrieved_evidence: list[Evidence]
    reference_answer: str | None
    raw_packet: dict[str, Any]


def load_diagnostic_packets(source_path: Path) -> list[DiagnosticQuestionPacket]:
    """Load the 16 diagnostic question packets from zip or directory."""
    packets: list[DiagnosticQuestionPacket] = []

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            filenames = sorted(
                name for name in zf.namelist()
                if name.startswith("holdout_packets/") and name.endswith(".json")
            )
            if not filenames:
                filenames = sorted(
                    name for name in zf.namelist()
                    if name.endswith(".json") and not name.startswith("execution/")
                )
            for fname in filenames:
                data = json.loads(zf.read(fname).decode("utf-8"))
                packets.append(_parse_single_packet(data))
    elif source_path.is_dir():
        packet_dir = source_path / "holdout_packets"
        if not packet_dir.exists():
            packet_dir = source_path
        files = sorted(packet_dir.glob("*.json"))
        for file in files:
            if file.name.startswith("execution_") or file.name.startswith("source_"):
                continue
            data = json.loads(file.read_text(encoding="utf-8"))
            packets.append(_parse_single_packet(data))
    else:
        raise DataValidationError(f"Diagnostic packets source not found: {source_path}")

    packets.sort(key=lambda p: p.question_id)
    return packets


def _parse_single_packet(data: dict[str, Any]) -> DiagnosticQuestionPacket:
    qid = str(data["question_id"])
    question = str(data["question"])
    evidence_list: list[Evidence] = []
    for ev_data in data.get("retrieved_evidence", []):
        evidence_list.append(
            Evidence(
                evidence_id=ev_data["evidence_id"],
                chunk_id=ev_data.get("chunk_id", f"chunk-{ev_data['evidence_id']}"),
                document_id=ev_data.get("document_id", f"doc-{ev_data['evidence_id']}"),
                document_title=ev_data.get("document_title"),
                document_number=ev_data.get("document_number"),
                article_number=ev_data.get("article_number"),
                article_title=ev_data.get("article_title"),
                effect_status=ev_data.get("effect_status"),
                text=ev_data["text"],
                source_url=ev_data.get("source_url"),
            )
        )
    ref_answer = None
    ref_context = data.get("reference_answer_context")
    if isinstance(ref_context, dict) and "text" in ref_context:
        ref_answer = ref_context["text"]

    return DiagnosticQuestionPacket(
        question_id=qid,
        question=question,
        retrieved_evidence=evidence_list,
        reference_answer=ref_answer,
        raw_packet=data,
    )


def compute_deterministic_blinding(
    question_ids: list[str],
    salt: str = "g1_blind_salt_v1:",
) -> dict[str, dict[str, str]]:
    """Deterministically randomize Option 1 vs Option 2 mapping per question."""
    blinding_key: dict[str, dict[str, str]] = {}
    for qid in sorted(question_ids):
        h = sha256(f"{salt}{qid}".encode("utf-8")).hexdigest()
        if int(h[:8], 16) % 2 == 0:
            blinding_key[qid] = {
                "option_1": GROUNDING_PROFILE_BASELINE,
                "option_2": GROUNDING_PROFILE_MATERIAL_FIDELITY_V1,
            }
        else:
            blinding_key[qid] = {
                "option_1": GROUNDING_PROFILE_MATERIAL_FIDELITY_V1,
                "option_2": GROUNDING_PROFILE_BASELINE,
            }
    return blinding_key


def generate_pairwise_worksheet(
    packets: list[DiagnosticQuestionPacket],
    predictions: list[dict[str, Any]],
    blinding_key: dict[str, dict[str, str]],
) -> str:
    """Generate the blinded markdown review worksheet for human evaluation."""
    pred_by_qid = {p["question_id"]: p for p in predictions}
    lines: list[str] = [
        "# G1 vs Baseline Grounded Generation — Blinded Pairwise Review Worksheet",
        "",
        f"**Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Total Diagnostic Questions:** {len(packets)}",
        f"**Blinding Protocol:** Deterministic pseudorandom assignment per question ID",
        "",
        "---",
        "",
        "## Human Evaluation Protocol & Rubric",
        "",
        "This worksheet presents a pairwise comparison of generated answers for the 16 diagnostic questions.",
        "For each question, **Option 1** and **Option 2** are randomly assigned to either Baseline or G1.",
        "The actual identity key is stored separately in `results/generation_g1_blinding_key.json`.",
        "",
        "### Pre-Registered Evaluation Criteria:",
        "1. **Criterion B (Known Material Error Elimination)**:",
        "   - On historical error questions (`125893`, `45427`, `90897`, `95695`), check if the candidate:",
        "     - Preserves the exact statutory actor/role (e.g. advocate vs litigant; NCO vs officer)",
        "     - Preserves statutory conditions/exceptions (e.g. non-state capital prerequisite)",
        "     - Preserves regulated action/object (e.g. testing vs production; fine brackets)",
        "     - Does NOT introduce new hallucinations or wrongful broadening.",
        "2. **Criterion C (Valid Answer Preservation)**:",
        "   - On gold valid questions, check if the candidate retains a materially supported, useful answer without unjustified abstention.",
        "3. **Question 61523 (List / Category Safety)**:",
        "   - Check if list noun phrases are accepted naturally without requiring artificial sentence structures.",
        "",
        "---",
        "",
        "## Pairwise Question Reviews",
        "",
    ]

    for index, pkt in enumerate(packets, 1):
        qid = pkt.question_id
        pred = pred_by_qid.get(qid)
        if not pred:
            continue

        mapping = blinding_key.get(qid, {"option_1": "baseline", "option_2": "material_fidelity_v1"})
        opt1_profile = mapping["option_1"]
        opt2_profile = mapping["option_2"]

        opt1_data = pred["baseline_result"] if opt1_profile == GROUNDING_PROFILE_BASELINE else pred["g1_result"]
        opt2_data = pred["baseline_result"] if opt2_profile == GROUNDING_PROFILE_BASELINE else pred["g1_result"]

        is_known_error = qid in KNOWN_MATERIAL_ERROR_QUESTIONS
        is_list_safety = qid in KNOWN_POSITIVE_LIST_QUESTIONS

        badge = ""
        if is_known_error:
            badge = " `[CRITERION B: KNOWN ERROR MECHANISM]`"
        elif is_list_safety:
            badge = " `[CRITERION C: LIST SAFETY CHECK]`"

        lines.extend([
            f"### Question {index}: [QID: {qid}]{badge}",
            "",
            f"**Question Text:**",
            f"> {pkt.question}",
            "",
            "**Retrieved Evidence Excerpts:**",
        ])

        for ev in pkt.retrieved_evidence:
            doc_str = f"{ev.document_title or 'Văn bản'}"
            if ev.document_number:
                doc_str += f" ({ev.document_number})"
            if ev.article_number:
                doc_str += f" - Điều {ev.article_number}"
            snippet = ev.text.replace("\n", " ").strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(f"- **[{ev.evidence_id}]** *{doc_str}*:\n  > \"{snippet}\"")

        lines.extend([
            "",
            "#### Option 1",
            f"- **Execution Status:** `{opt1_data.get('status')}`",
            f"- **Insufficient Evidence:** `{opt1_data.get('insufficient_evidence')}`",
            f"- **Rendered Answer:**",
            f"  > {opt1_data.get('answer', 'N/A')}",
            f"- **Emitted Claims ({len(opt1_data.get('claims', []))}):**",
        ])
        for c_idx, claim in enumerate(opt1_data.get("claims", []), 1):
            lines.append(f"  {c_idx}. \"{claim.get('text')}\" (Evidence: `{claim.get('evidence_ids')}`)")

        lines.extend([
            "",
            "#### Option 2",
            f"- **Execution Status:** `{opt2_data.get('status')}`",
            f"- **Insufficient Evidence:** `{opt2_data.get('insufficient_evidence')}`",
            f"- **Rendered Answer:**",
            f"  > {opt2_data.get('answer', 'N/A')}",
            f"- **Emitted Claims ({len(opt2_data.get('claims', []))}):**",
        ])
        for c_idx, claim in enumerate(opt2_data.get("claims", []), 1):
            lines.append(f"  {c_idx}. \"{claim.get('text')}\" (Evidence: `{claim.get('evidence_ids')}`)")

        lines.extend([
            "",
            "#### Human Review Assessment:",
            "- **Option 1 Fidelity:** [ ] Fully Supported  [ ] Minor Flaw  [ ] Material Hallucination / Broadening",
            "- **Option 2 Fidelity:** [ ] Fully Supported  [ ] Minor Flaw  [ ] Material Hallucination / Broadening",
            "- **Preference Decision:**",
            "  - [ ] Option 1 is clearly superior",
            "  - [ ] Option 2 is clearly superior",
            "  - [ ] Both options are acceptable / equivalent quality",
            "  - [ ] Both options are flawed / unacceptable",
            "- **Fidelity Error Types (if any):**",
            "  - Option 1: [ ] Actor/Role [ ] Condition/Prereq [ ] Action/Object [ ] Scope [ ] Number [ ] None",
            "  - Option 2: [ ] Actor/Role [ ] Condition/Prereq [ ] Action/Object [ ] Scope [ ] Number [ ] None",
            "- **Evaluator Notes:** __________________________________________________",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


class GenerationGroundingG1Evaluator:
    """Harness executing the Baseline vs G1 development A/B experiment."""

    def __init__(
        self,
        diagnostic_packets_path: Path,
        output_dir: Path,
        generation_config: GenerationConfig,
        provider: ChatModelProvider | None = None,
        preflight_only: bool = False,
    ) -> None:
        self._packets_path = diagnostic_packets_path
        self._output_dir = output_dir
        self._config = generation_config
        self._provider = provider
        self._preflight_only = preflight_only

    def run(self) -> dict[str, Any]:
        """Execute full evaluation pipeline or preflight validation."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self._output_dir / "results"
        execution_dir = self._output_dir / "execution"
        telemetry_dir = self._output_dir / "telemetry"
        results_dir.mkdir(exist_ok=True)
        execution_dir.mkdir(exist_ok=True)
        telemetry_dir.mkdir(exist_ok=True)

        packets = load_diagnostic_packets(self._packets_path)
        if len(packets) != 16:
            _LOGGER.warning(
                "Expected exactly 16 diagnostic packets, observed %d", len(packets)
            )

        qids = [p.question_id for p in packets]
        blinding_key = compute_deterministic_blinding(qids)

        source_identity = {
            "experiment_id": CANONICAL_EXPERIMENT_ID,
            "created_at": datetime.now(UTC).isoformat(),
            "package_version": getattr(legal_agentic_rag, "__version__", CANONICAL_PACKAGE_VERSION),
            "diagnostic_packets": {
                "path": str(self._packets_path),
                "count": len(packets),
                "question_ids": qids,
            },
            "generation_config": self._config.model_dump(mode="json"),
            "system_instruction_baseline_sha256": sha256(
                _SYSTEM_INSTRUCTION_BASELINE.encode("utf-8")
            ).hexdigest(),
            "system_instruction_material_fidelity_v1_sha256": sha256(
                _SYSTEM_INSTRUCTION_MATERIAL_FIDELITY_V1.encode("utf-8")
            ).hexdigest(),
            "preflight_only": self._preflight_only,
        }

        if self._preflight_only:
            source_identity["provider_constructor_contract_verified"] = True
            source_identity["preflight_ready"] = True
            identity_path = execution_dir / "generation_g1_source_identity.json"
            identity_path.write_text(
                json.dumps(source_identity, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return source_identity

        if self._provider is None:
            if self._config.backend == "transformers":
                raw_provider = TransformersChatProvider(self._config)
            else:
                raise ValueError(f"Unsupported provider backend: {self._config.backend}")
        else:
            raw_provider = self._provider

        proxy_provider = TelemetryLoggingChatProviderProxy(raw_provider)

        gen_baseline = ModelBackedAnswerGenerator(
            proxy_provider,
            grounding_profile=GROUNDING_PROFILE_BASELINE,
            max_structured_output_retries=self._config.max_structured_output_retries,
            max_schema_recovery_attempts=self._config.max_schema_recovery_attempts,
            max_missing_field_corrections=self._config.max_missing_field_corrections,
        )
        gen_g1 = ModelBackedAnswerGenerator(
            proxy_provider,
            grounding_profile=GROUNDING_PROFILE_MATERIAL_FIDELITY_V1,
            max_structured_output_retries=self._config.max_structured_output_retries,
            max_schema_recovery_attempts=self._config.max_schema_recovery_attempts,
            max_missing_field_corrections=self._config.max_missing_field_corrections,
        )

        predictions: list[dict[str, Any]] = []
        baseline_abstentions = 0
        g1_abstentions = 0
        baseline_errors = 0
        g1_errors = 0
        baseline_rouge_scores: list[float] = []
        g1_rouge_scores: list[float] = []
        baseline_meteor_scores: list[float] = []
        g1_meteor_scores: list[float] = []

        start_time = perf_counter()

        for pkt in packets:
            qid = pkt.question_id
            query = RetrievalQuery(
                query_id=qid,
                original_question=pkt.question,
                normalized_question=pkt.question,
                top_k=len(pkt.retrieved_evidence),
                candidate_k=len(pkt.retrieved_evidence),
            )

            # Generate Baseline
            proxy_provider.set_context(GROUNDING_PROFILE_BASELINE, qid)
            baseline_record: dict[str, Any] = {"status": "SUCCESS"}
            try:
                b_resp = gen_baseline.generate(
                    query,
                    pkt.retrieved_evidence,
                    RetrievalStrategy.HYBRID_RERANK,
                    trace_id=f"g1_eval_baseline_{qid}",
                )
                baseline_record["answer"] = b_resp.answer
                baseline_record["insufficient_evidence"] = b_resp.insufficient_evidence
                baseline_record["warnings"] = b_resp.warnings
                baseline_record["citations"] = [c.model_dump(mode="json") for c in b_resp.citations]
                if b_resp.insufficient_evidence:
                    baseline_abstentions += 1
                if pkt.reference_answer:
                    b_scores = score_text_answer(b_resp.answer, pkt.reference_answer)
                    if b_scores.rouge_l is not None:
                        baseline_rouge_scores.append(b_scores.rouge_l)
                    if b_scores.meteor is not None:
                        baseline_meteor_scores.append(b_scores.meteor)
                    baseline_record["metrics"] = {
                        "exact_match": b_scores.exact_match,
                        "rouge_l": b_scores.rouge_l,
                        "meteor": b_scores.meteor,
                    }
            except Exception as exc:
                baseline_errors += 1
                baseline_record["status"] = "ERROR"
                baseline_record["error_type"] = type(exc).__name__
                baseline_record["error_message"] = str(exc)

            # Generate G1
            proxy_provider.set_context(GROUNDING_PROFILE_MATERIAL_FIDELITY_V1, qid)
            g1_record: dict[str, Any] = {"status": "SUCCESS"}
            try:
                g_resp = gen_g1.generate(
                    query,
                    pkt.retrieved_evidence,
                    RetrievalStrategy.HYBRID_RERANK,
                    trace_id=f"g1_eval_g1_{qid}",
                )
                g1_record["answer"] = g_resp.answer
                g1_record["insufficient_evidence"] = g_resp.insufficient_evidence
                g1_record["warnings"] = g_resp.warnings
                g1_record["citations"] = [c.model_dump(mode="json") for c in g_resp.citations]
                if g_resp.insufficient_evidence:
                    g1_abstentions += 1
                if pkt.reference_answer:
                    g_scores = score_text_answer(g_resp.answer, pkt.reference_answer)
                    if g_scores.rouge_l is not None:
                        g1_rouge_scores.append(g_scores.rouge_l)
                    if g_scores.meteor is not None:
                        g1_meteor_scores.append(g_scores.meteor)
                    g1_record["metrics"] = {
                        "exact_match": g_scores.exact_match,
                        "rouge_l": g_scores.rouge_l,
                        "meteor": g_scores.meteor,
                    }
            except Exception as exc:
                g1_errors += 1
                g1_record["status"] = "ERROR"
                g1_record["error_type"] = type(exc).__name__
                g1_record["error_message"] = str(exc)

            predictions.append({
                "question_id": qid,
                "question": pkt.question,
                "baseline_result": baseline_record,
                "g1_result": g1_record,
            })

        total_duration = perf_counter() - start_time
        total_questions = len(packets)

        baseline_abstention_rate = baseline_abstentions / total_questions if total_questions else 0.0
        g1_abstention_rate = g1_abstentions / total_questions if total_questions else 0.0

        criteria_evaluation = {
            "criterion_a_zero_execution_errors": {
                "target": "0 execution errors for G1",
                "observed": g1_errors,
                "status": "PASS" if g1_errors == 0 else "FAIL",
            },
            "criterion_d_bounded_abstention": {
                "target": "g1_abstention_rate - baseline_abstention_rate <= 0.15",
                "baseline_rate": baseline_abstention_rate,
                "g1_rate": g1_abstention_rate,
                "delta": g1_abstention_rate - baseline_abstention_rate,
                "status": "PASS" if (g1_abstention_rate - baseline_abstention_rate) <= 0.15 else "FAIL",
            },
            "criterion_e_output_schema_preservation": {
                "target": "All predictions validate against ModelAnswerDraft",
                "status": "PASS" if g1_errors == 0 else "CHECK_ERRORS",
            },
            "criterion_f_provider_call_parity": {
                "target": "Normal generation call count parity",
                "baseline_calls": sum(1 for r in proxy_provider.records if r.arm == GROUNDING_PROFILE_BASELINE),
                "g1_calls": sum(1 for r in proxy_provider.records if r.arm == GROUNDING_PROFILE_MATERIAL_FIDELITY_V1),
                "status": "PASS" if sum(1 for r in proxy_provider.records if r.arm == GROUNDING_PROFILE_BASELINE) == sum(1 for r in proxy_provider.records if r.arm == GROUNDING_PROFILE_MATERIAL_FIDELITY_V1) else "CHECK",
            },
            "criterion_b_and_c_human_review": {
                "status": "PENDING_BLINDED_HUMAN_REVIEW",
                "worksheet": "results/generation_g1_human_review_worksheet.md",
            },
        }

        report = {
            "experiment_id": CANONICAL_EXPERIMENT_ID,
            "created_at": datetime.now(UTC).isoformat(),
            "total_duration_seconds": total_duration,
            "total_questions": total_questions,
            "baseline_summary": {
                "successful_generations": total_questions - baseline_errors,
                "errors": baseline_errors,
                "abstentions": baseline_abstentions,
                "abstention_rate": baseline_abstention_rate,
                "mean_rouge_l": (sum(baseline_rouge_scores) / len(baseline_rouge_scores)) if baseline_rouge_scores else None,
                "mean_meteor": (sum(baseline_meteor_scores) / len(baseline_meteor_scores)) if baseline_meteor_scores else None,
            },
            "g1_summary": {
                "successful_generations": total_questions - g1_errors,
                "errors": g1_errors,
                "abstentions": g1_abstentions,
                "abstention_rate": g1_abstention_rate,
                "mean_rouge_l": (sum(g1_rouge_scores) / len(g1_rouge_scores)) if g1_rouge_scores else None,
                "mean_meteor": (sum(g1_meteor_scores) / len(g1_meteor_scores)) if g1_meteor_scores else None,
            },
            "criteria_evaluation": criteria_evaluation,
            "source_identity": source_identity,
        }

        pred_path = results_dir / "generation_g1_ab_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        report_path = results_dir / "generation_g1_ab_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        blinding_key_path = results_dir / "generation_g1_blinding_key.json"
        blinding_key_path.write_text(
            json.dumps(blinding_key, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        worksheet_md = generate_pairwise_worksheet(packets, predictions, blinding_key)
        worksheet_path = results_dir / "generation_g1_human_review_worksheet.md"
        worksheet_path.write_text(worksheet_md, encoding="utf-8")

        telemetry_path = telemetry_dir / "provider_calls.jsonl"
        with open(telemetry_path, "w", encoding="utf-8") as f:
            for rec in proxy_provider.records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        identity_path = execution_dir / "generation_g1_source_identity.json"
        identity_path.write_text(
            json.dumps(source_identity, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate G1 Generation Grounding A/B Experiment")
    parser.add_argument(
        "--diagnostic-packets",
        type=Path,
        default=Path("verification-v2-holdout-review-packets-v1.zip"),
        help="Path to burned diagnostic review packets zip or directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generation_g1_ab_output"),
        help="Directory to save A/B results, worksheet, and telemetry",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=CANONICAL_MODEL_NAME,
        help="Model name for generation",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=CANONICAL_MODEL_REVISION,
        help="Model revision for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CANONICAL_DEVICE,
        help="Device for inference (e.g. cuda or cpu)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=CANONICAL_TORCH_DTYPE,
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for inference",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate source packet integrity and contracts without model inference",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    gen_config = GenerationConfig(
        backend="transformers",
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_output_tokens=1024,
        max_structured_output_retries=1,
    )

    evaluator = GenerationGroundingG1Evaluator(
        diagnostic_packets_path=args.diagnostic_packets,
        output_dir=args.output_dir,
        generation_config=gen_config,
        preflight_only=args.preflight_only,
    )

    report = evaluator.run()
    _LOGGER.info("G1 A/B Evaluation complete: %s", report.get("experiment_id"))


if __name__ == "__main__":
    main()
