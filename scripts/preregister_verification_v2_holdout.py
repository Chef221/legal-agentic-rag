#!/usr/bin/env python3
"""Fresh V2 Holdout Pre-Registration and Sealed Materialization.

This script deterministically selects and materializes pre-registered fresh holdout
candidates from frozen Phase-A historical answer_verified outputs:
- 16 PRIMARY holdout candidates (4 per stratum across 4 strata)
- 8 FRESH RESERVE holdout candidates (2 per stratum across 4 strata)

Strict Invariants:
- Selection algorithm does NOT use semantic verifiers, LLM judgment, or reference answers
- Selected candidates are unreviewed and holdout packets are sealed with review_status="sealed_unreviewed"
- Human labels are strictly null
- Selected QIDs are kept sealed and NOT printed in standard console summary output
- Zero live retrieval, reranking, generation, or model execution
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
import zipfile

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.schemas.answering import AnswerResponse, Evidence
from legal_agentic_rag.schemas.manifests import ArtifactType

_LOGGER = logging.getLogger(__name__)

CANONICAL_PHASE_A_ZIP_SHA256 = (
    "df05a401599c43a28e39136d72b225841b242d10a40dc5bc475b9be6ed86be8b"
)
CANONICAL_PHASE_A_ZIP_SIZE = 1036904
CANONICAL_PHASE_A_ZIP_FILENAME = "phase-a-current-system-census-final-evidence.zip"

CANONICAL_PHASE_A_RESULTS_SHA256 = (
    "7b1bf802c752e37cee7386c0b24f6e0ee5ea2f65056b22eaa9488d73161aaee6"
)
CANONICAL_DEVELOPMENT_SHA256 = (
    "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"
)

CANONICAL_SERVING_DATASET_NAME = "uit-dsc-2026-task2-selected-contexts"
CANONICAL_SERVING_DATASET_REVISION = (
    "sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e"
)
CANONICAL_SERVING_RECORD_COUNT = 330768

CANONICAL_SELECTION_SALT = "verification-v2-holdout-gen-v1:"
CANONICAL_SELECTION_ALGORITHM = "deterministic_sha256_stratified_v2"

# 4 Suspicious Forensic QIDs (Task B-FORENSIC-0 / 1A)
SUSPICIOUS_FORENSIC_QIDS = {"102047", "147239", "26541", "95861"}

# 16 Positive-Control PRIMARY QIDs (Task B-FORENSIC-1B / 1C)
POSITIVE_CONTROL_PRIMARY_QIDS = {
    "75171", "150131", "30405", "36801", "116877", "15181", "5967", "139413",
    "34351", "31883", "40489", "155139", "108497", "4031", "103983", "140693",
}

# 8 Positive-Control RESERVE QIDs (Task B-FORENSIC-1B)
POSITIVE_CONTROL_RESERVE_QIDS = {
    "27503", "31317", "33177", "85651", "112105", "112833", "130283", "137453",
}

# 22 Historical B1A Relationship QIDs
HISTORICAL_B1A_RELATIONSHIP_QIDS = {
    "102047", "107487", "110287", "111905", "113537", "122659",
    "125393", "133075", "134605", "147239", "147869", "150051",
    "26541", "29491", "29877", "39671", "45219", "47537",
    "48905", "64035", "95861", "99639",
}

# Full deduplicated contamination exclusion set (46 unique QIDs)
CANONICAL_CONTAMINATION_EXCLUSION_SET = sorted(
    SUSPICIOUS_FORENSIC_QIDS
    | POSITIVE_CONTROL_PRIMARY_QIDS
    | POSITIVE_CONTROL_RESERVE_QIDS
    | HISTORICAL_B1A_RELATIONSHIP_QIDS
)

_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*%?")
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_STOPWORDS = {"các", "cho", "của", "là", "một", "những", "theo", "trong", "tại", "và", "về"}
_NEGATION_TERMS = {"bãi", "cấm", "chưa", "hủy", "không", "ngoại", "trừ"}


def sha256_file(path: Path) -> str:
    """Compute deterministic SHA-256 hex digest for a file."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute deterministic SHA-256 hex digest for UTF-8 text."""
    return sha256(text.encode("utf-8")).hexdigest()


class HoldoutStratum(StrEnum):
    D_NEGATION_MODALITY = "D_NEGATION_MODALITY"
    C_NUMERIC = "C_NUMERIC"
    B_MULTI_CLAIM_CLEAN = "B_MULTI_CLAIM_CLEAN"
    A_SINGLE_CLAIM_CLEAN = "A_SINGLE_CLAIM_CLEAN"


STRATA_ORDER = [
    HoldoutStratum.D_NEGATION_MODALITY,
    HoldoutStratum.C_NUMERIC,
    HoldoutStratum.B_MULTI_CLAIM_CLEAN,
    HoldoutStratum.A_SINGLE_CLAIM_CLEAN,
]

STRATA_QUOTAS_PRIMARY = {
    HoldoutStratum.A_SINGLE_CLAIM_CLEAN: 4,
    HoldoutStratum.B_MULTI_CLAIM_CLEAN: 4,
    HoldoutStratum.C_NUMERIC: 4,
    HoldoutStratum.D_NEGATION_MODALITY: 4,
}

STRATA_QUOTAS_RESERVE = {
    HoldoutStratum.A_SINGLE_CLAIM_CLEAN: 2,
    HoldoutStratum.B_MULTI_CLAIM_CLEAN: 2,
    HoldoutStratum.C_NUMERIC: 2,
    HoldoutStratum.D_NEGATION_MODALITY: 2,
}


def has_numeric_tokens(text: str) -> bool:
    """Check if text contains numeric patterns."""
    return bool(_NUMBER_PATTERN.findall(text.casefold()))


def has_negation_tokens(text: str) -> bool:
    """Check if text contains negation terms matching verifier tokenizer."""
    tokens = {t.casefold() for t in _TOKEN_PATTERN.findall(text) if t.casefold() not in _STOPWORDS}
    return bool(tokens & _NEGATION_TERMS)


@dataclass(frozen=True)
class SelectedHoldoutCandidate:
    question_id: str
    stratum: str
    selection_key: str
    pool_type: str  # "primary" or "reserve"
    claim_count: int
    historical_stop_reason: str


class V2HoldoutPreRegistrar:
    """Validate Phase-A source, perform sealed stratified selection, and materialize holdout packets."""

    def __init__(
        self,
        *,
        phase_a_evidence_path: Path,
        serving_root: Path,
        development_path: Path,
        output_dir: Path,
        excluded_qids: Sequence[str] | set[str] | None = None,
        selection_salt: str = CANONICAL_SELECTION_SALT,
    ) -> None:
        self._phase_a_evidence_path = phase_a_evidence_path.resolve()
        self._serving_root = serving_root.resolve()
        self._development_path = development_path.resolve()
        self._output_dir = output_dir.resolve()
        self._excluded_qids = (
            sorted(set(excluded_qids))
            if excluded_qids is not None
            else list(CANONICAL_CONTAMINATION_EXCLUSION_SET)
        )
        self._selection_salt = selection_salt

    def run(self) -> dict[str, Any]:
        """Execute holdout discovery, selection, and primary packet materialization."""
        # 1. Validate Development JSON
        dev_questions, dev_sha = self._load_and_validate_development(self._development_path)

        # 2. Validate and Load Phase-A Evidence Archive
        (
            bundle_dir,
            cleanup_dir,
            source_kind,
            archive_filename,
            archive_sha,
            results_path,
        ) = self._resolve_phase_a_evidence(self._phase_a_evidence_path)

        try:
            # 3. Validate Raw Results and Historical Invariants
            records_map, results_sha = self._load_and_validate_phase_a_records(bundle_dir, results_path)

            # 4. Construct Eligible Pool and Perform Stratified Selection
            eligible_records = self._filter_eligible_pool(records_map)
            stratum_assignments = self._stratify_records(eligible_records)
            primary_candidates, reserve_candidates = self._sample_strata(stratum_assignments)

            # 5. Load Legal Chunks for PRIMARY candidates
            primary_qids = [c.question_id for c in primary_candidates]
            chunks_dir = self._find_legal_chunks_dir(self._serving_root)
            chunk_manifest, chunks_by_id = self._load_needed_chunks(
                chunks_dir=chunks_dir,
                primary_qids=primary_qids,
                records_map=records_map,
            )

            # 6. Reconstruct Evidence & Replay Verifier for all PRIMARY candidates
            verifier = RuleBasedCitationVerifier()
            primary_packets: dict[str, dict[str, Any]] = {}
            primary_arm_summaries: list[dict[str, Any]] = []

            for candidate in primary_candidates:
                qid = candidate.question_id
                r = records_map[qid]
                dev_entry = dev_questions[qid]
                q_text = dev_entry.get("question", "")
                ref_ans = dev_entry.get("answer", "")

                packet, summary = self._process_primary_case(
                    candidate=candidate,
                    record=r,
                    question_text=q_text,
                    reference_answer=ref_ans,
                    chunks_by_id=chunks_by_id,
                    chunk_manifest=chunk_manifest,
                    verifier=verifier,
                    dev_sha=dev_sha,
                    archive_filename=archive_filename,
                    archive_sha=archive_sha,
                    results_sha=results_sha,
                    source_kind=source_kind,
                )
                primary_packets[qid] = packet
                primary_arm_summaries.append(summary)

            # Check primary verification invariants
            all_chunks_lookup_pass = all(s["selected_chunk_lookup_pass"] for s in primary_arm_summaries)
            all_source_mapping_pass = all(s["source_mapping_pass"] for s in primary_arm_summaries)
            all_metadata_pass = all(s["metadata_crosscheck_pass"] for s in primary_arm_summaries)
            all_replay_pass = all(s["rule_verifier_replay_pass"] for s in primary_arm_summaries)

            if not (
                all_chunks_lookup_pass
                and all_source_mapping_pass
                and all_metadata_pass
                and all_replay_pass
            ):
                verdict = "INVALID_V2_HOLDOUT_PROVENANCE"
            else:
                verdict = "V2_HOLDOUT_PRE_REGISTERED"

            # 7. Build Full Selection Report & Content-Free Commitment Report
            full_selection_report = self._build_full_selection_report(
                verdict=verdict,
                source_kind=source_kind,
                archive_filename=archive_filename,
                archive_sha=archive_sha,
                results_sha=results_sha,
                dev_sha=dev_sha,
                chunk_manifest=chunk_manifest,
                eligible_count=len(eligible_records),
                stratum_assignments=stratum_assignments,
                primary_candidates=primary_candidates,
                reserve_candidates=reserve_candidates,
                primary_summaries=primary_arm_summaries,
            )

            # 8. Write Materialized Outputs
            written_paths = self._write_outputs(
                primary_packets=primary_packets,
                full_selection_report=full_selection_report,
                primary_candidates=primary_candidates,
                reserve_candidates=reserve_candidates,
                source_kind=source_kind,
                archive_filename=archive_filename,
                archive_sha=archive_sha,
                results_sha=results_sha,
                dev_sha=dev_sha,
            )

            # 9. Build Content-Free Public Commitment Report (No QIDs)
            public_commitment_report = self._build_public_commitment_report(
                verdict=verdict,
                source_kind=source_kind,
                archive_filename=archive_filename,
                archive_sha=archive_sha,
                results_sha=results_sha,
                dev_sha=dev_sha,
                chunk_manifest=chunk_manifest,
                eligible_count=len(eligible_records),
                stratum_assignments=stratum_assignments,
                primary_candidates=primary_candidates,
                reserve_candidates=reserve_candidates,
                primary_summaries=primary_arm_summaries,
                selection_artifact_path=written_paths["selection_artifact"],
            )

            # Write public commitment report
            (self._output_dir / "results" / "verification-v2-holdout-public-commitment-v1.json").write_text(
                json.dumps(public_commitment_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            return public_commitment_report

        finally:
            if cleanup_dir is not None and cleanup_dir.exists():
                import shutil
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def _load_and_validate_development(self, path: Path) -> tuple[dict[str, Any], str]:
        """Validate canonical development.json file and hash."""
        if not path.exists():
            raise DataValidationError(f"development.json does not exist: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != CANONICAL_DEVELOPMENT_SHA256:
            raise DataValidationError(
                f"development.json SHA mismatch: expected {CANONICAL_DEVELOPMENT_SHA256}, got {actual_sha}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise DataValidationError("development.json root must be a mapping")
        if len(data) != 991:
            raise DataValidationError(f"development.json expected 991 records, got {len(data)}")
        return dict(data), actual_sha

    def _resolve_phase_a_evidence(
        self, path: Path
    ) -> tuple[Path, Path | None, str, str, str | None, Path]:
        """Verify and resolve Phase-A evidence ZIP or extracted directory fail-closed."""
        if not path.exists():
            raise DataValidationError(f"Phase-A evidence path does not exist: {path}")

        if path.is_file() and path.suffix.lower() == ".zip":
            actual_sha = sha256_file(path)
            if actual_sha != CANONICAL_PHASE_A_ZIP_SHA256:
                raise DataValidationError(
                    f"Phase-A ZIP SHA mismatch: expected {CANONICAL_PHASE_A_ZIP_SHA256}, got {actual_sha}"
                )
            temp_unpack = Path(tempfile.mkdtemp(prefix="phase_a_v2_holdout_unpack_"))
            with zipfile.ZipFile(path, "r") as z:
                names = set(z.namelist())
                req_member = "phase-a-current-system-census-batch/results.jsonl"
                req_manifest = "phase-a-current-system-census-batch/manifest.json"
                if req_member not in names or req_manifest not in names:
                    raise DataValidationError("Phase-A evidence missing required batch results or manifest")
                z.extractall(temp_unpack)

            results_file = temp_unpack / req_member
            return (
                temp_unpack,
                temp_unpack,
                "canonical_zip",
                path.name,
                actual_sha,
                results_file,
            )

        if path.is_dir():
            req_member = path / "phase-a-current-system-census-batch" / "results.jsonl"
            req_manifest = path / "phase-a-current-system-census-batch" / "manifest.json"
            if not req_member.is_file() or not req_manifest.is_file():
                raise DataValidationError("Phase-A extracted directory missing required batch results or manifest")

            return (
                path,
                None,
                "canonical_extracted_bundle",
                path.name,
                None,
                req_member,
            )

        raise DataValidationError(f"Invalid Phase-A evidence path: {path}")

    def _load_and_validate_phase_a_records(
        self, bundle_dir: Path, results_path: Path
    ) -> tuple[dict[str, dict[str, Any]], str]:
        """Validate Phase-A manifest and 991 records against historical invariants."""
        manifest_path = bundle_dir / "phase-a-current-system-census-batch" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        actual_results_sha = sha256_file(results_path)
        if actual_results_sha != CANONICAL_PHASE_A_RESULTS_SHA256:
            raise DataValidationError(
                f"Phase-A results.jsonl SHA mismatch: expected {CANONICAL_PHASE_A_RESULTS_SHA256}, got {actual_results_sha}"
            )
        if manifest.get("records_sha256") != actual_results_sha:
            raise DataValidationError("Phase-A manifest records_sha256 does not match results.jsonl")
        if manifest.get("record_count") != 991:
            raise DataValidationError(
                f"Phase-A manifest record_count mismatch: expected 991, got {manifest.get('record_count')}"
            )
        if manifest.get("question_source_sha256") != CANONICAL_DEVELOPMENT_SHA256:
            raise DataValidationError("Phase-A manifest question_source_sha256 does not match development.json")

        lines = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != 991:
            raise DataValidationError(f"Phase-A results lines count mismatch: expected 991, got {len(lines)}")

        records_map: dict[str, dict[str, Any]] = {}
        stop_reasons = Counter()

        for r in lines:
            qid = str(r.get("question_id", "")).strip()
            if not qid:
                raise DataValidationError("Phase-A record contains blank question_id")
            if qid in records_map:
                raise DataValidationError(f"Phase-A contains duplicate question_id: '{qid}'")
            records_map[qid] = r

            agent_meta = r.get("response", {}).get("metadata", {}).get("agent", {})
            sr = agent_meta.get("stop_reason", "unknown")
            stop_reasons[sr] += 1

        # Assert historical stop-reason counts
        expected_counts = {
            "answer_verified": 806,
            "generation_failed": 177,
            "citation_verification_failed": 7,
            "max_retry_reached": 1,
        }
        for sr, exp_count in expected_counts.items():
            if stop_reasons[sr] != exp_count:
                raise DataValidationError(
                    f"Phase-A stop reason '{sr}' count mismatch: expected {exp_count}, got {stop_reasons[sr]}"
                )

        return records_map, actual_results_sha

    def _filter_eligible_pool(
        self, records_map: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter records to eligible answer_verified pool excluding contamination set."""
        excluded_set = set(self._excluded_qids)
        eligible: list[dict[str, Any]] = []

        for qid, r in records_map.items():
            # Exclude contamination QIDs
            if qid in excluded_set:
                continue

            resp = r.get("response", {})
            meta = resp.get("metadata", {})
            agent_meta = meta.get("agent", {})
            stop_reason = agent_meta.get("stop_reason")

            if stop_reason != "answer_verified":
                continue

            cv = meta.get("citation_verification", {})
            if not cv.get("is_valid"):
                continue

            sel_ev = meta.get("selected_evidence", [])
            if not sel_ev:
                continue

            ctx = meta.get("context", {})
            if not ctx.get("selection_trace"):
                continue

            claims = cv.get("claim_verifications", [])
            if not claims:
                continue

            eligible.append(r)

        return eligible

    def _stratify_records(
        self, eligible_records: list[dict[str, Any]]
    ) -> dict[HoldoutStratum, list[dict[str, Any]]]:
        """Classify records into exactly one of 4 deterministic strata using fixed precedence."""
        stratum_assignments: dict[HoldoutStratum, list[dict[str, Any]]] = defaultdict(list)

        for r in eligible_records:
            meta = r.get("response", {}).get("metadata", {})
            cv = meta.get("citation_verification", {})
            claims = cv.get("claim_verifications", [])

            # Check D: Negation
            is_negation = False
            for c in claims:
                ctext = c.get("claim_text", "")
                if has_negation_tokens(ctext) and c.get("negation_match") is True:
                    is_negation = True
                    break

            # Check C: Numeric
            is_numeric = False
            for c in claims:
                ctext = c.get("claim_text", "")
                if has_numeric_tokens(ctext) and c.get("numeric_match") is True:
                    is_numeric = True
                    break

            # Check B & A
            claim_count = len(claims)
            has_num_mismatch = any(c.get("numeric_match") is False for c in claims)
            has_neg_mismatch = any(c.get("negation_match") is False for c in claims)
            is_clean = (not has_num_mismatch) and (not has_neg_mismatch)

            # Precedence: D -> C -> B -> A
            if is_negation:
                stratum_assignments[HoldoutStratum.D_NEGATION_MODALITY].append(r)
            elif is_numeric:
                stratum_assignments[HoldoutStratum.C_NUMERIC].append(r)
            elif claim_count >= 2 and is_clean:
                stratum_assignments[HoldoutStratum.B_MULTI_CLAIM_CLEAN].append(r)
            elif claim_count == 1 and is_clean:
                stratum_assignments[HoldoutStratum.A_SINGLE_CLAIM_CLEAN].append(r)
            else:
                raise DataValidationError(f"Record {r.get('question_id')} could not be assigned to any stratum")

        return stratum_assignments

    def _sample_strata(
        self, stratum_assignments: dict[HoldoutStratum, list[dict[str, Any]]]
    ) -> tuple[list[SelectedHoldoutCandidate], list[SelectedHoldoutCandidate]]:
        """Deterministic sampling within each stratum using SHA-256 selection key."""
        primary_candidates: list[SelectedHoldoutCandidate] = []
        reserve_candidates: list[SelectedHoldoutCandidate] = []

        for stratum in STRATA_ORDER:
            records = stratum_assignments.get(stratum, [])
            primary_quota = STRATA_QUOTAS_PRIMARY[stratum]
            reserve_quota = STRATA_QUOTAS_RESERVE[stratum]
            total_required = primary_quota + reserve_quota

            if len(records) < total_required:
                raise DataValidationError(
                    f"Stratum '{stratum.value}' has only {len(records)} records, "
                    f"insufficient for required {total_required} (quota: {primary_quota} primary + {reserve_quota} reserve)"
                )

            # Compute selection key per record and sort deterministically
            keyed_records = []
            for r in records:
                qid = r["question_id"]
                key = sha256_text(f"{self._selection_salt}{qid}")
                keyed_records.append((key, qid, r))

            keyed_records.sort(key=lambda item: (item[0], item[1]))

            # Primary slice
            for key, qid, r in keyed_records[:primary_quota]:
                claims = r["response"]["metadata"]["citation_verification"]["claim_verifications"]
                primary_candidates.append(
                    SelectedHoldoutCandidate(
                        question_id=qid,
                        stratum=stratum.value,
                        selection_key=key,
                        pool_type="primary",
                        claim_count=len(claims),
                        historical_stop_reason=r["response"]["metadata"]["agent"]["stop_reason"],
                    )
                )

            # Reserve slice
            for key, qid, r in keyed_records[primary_quota:total_required]:
                claims = r["response"]["metadata"]["citation_verification"]["claim_verifications"]
                reserve_candidates.append(
                    SelectedHoldoutCandidate(
                        question_id=qid,
                        stratum=stratum.value,
                        selection_key=key,
                        pool_type="reserve",
                        claim_count=len(claims),
                        historical_stop_reason=r["response"]["metadata"]["agent"]["stop_reason"],
                    )
                )

        if len(primary_candidates) != 16:
            raise DataValidationError(f"Expected 16 primary candidates, got {len(primary_candidates)}")
        if len(reserve_candidates) != 8:
            raise DataValidationError(f"Expected 8 reserve candidates, got {len(reserve_candidates)}")

        return primary_candidates, reserve_candidates

    def _find_legal_chunks_dir(self, serving_root: Path) -> Path:
        """Find and validate legal_chunks artifact directory within serving root."""
        if not serving_root.exists():
            raise ArtifactCompatibilityError(f"Serving root does not exist: {serving_root}")

        if (serving_root / "legal_chunks").is_dir():
            candidate = serving_root / "legal_chunks"
        elif serving_root.name == "legal_chunks" and serving_root.is_dir():
            candidate = serving_root
        else:
            candidates = list(serving_root.glob("**/legal_chunks"))
            if not candidates:
                raise ArtifactCompatibilityError(
                    f"No legal_chunks directory found under serving root: {serving_root}"
                )
            candidate = candidates[0]

        manifest_file = candidate / "manifest.json"
        records_file = candidate / "records.jsonl"
        if not manifest_file.is_file() or not records_file.is_file():
            raise ArtifactCompatibilityError(
                f"legal_chunks directory missing manifest.json or records.jsonl at {candidate}"
            )
        return candidate

    def _load_needed_chunks(
        self,
        *,
        chunks_dir: Path,
        primary_qids: Sequence[str],
        records_map: dict[str, dict[str, Any]],
    ) -> tuple[Any, dict[str, dict[str, Any]]]:
        """Validate serving payload in-protocol and load only chunks needed by primary candidates."""
        chunk_manifest = load_artifact_manifest(
            chunks_dir,
            expected_type=ArtifactType.LEGAL_CHUNKS,
            verify_payload=True,
        )

        if chunk_manifest.dataset_name != CANONICAL_SERVING_DATASET_NAME:
            raise ArtifactCompatibilityError(
                f"Serving dataset_name mismatch: expected '{CANONICAL_SERVING_DATASET_NAME}', "
                f"got '{chunk_manifest.dataset_name}'"
            )
        if chunk_manifest.dataset_revision != CANONICAL_SERVING_DATASET_REVISION:
            raise ArtifactCompatibilityError(
                f"Serving dataset_revision mismatch: expected '{CANONICAL_SERVING_DATASET_REVISION}', "
                f"got '{chunk_manifest.dataset_revision}'"
            )
        if chunk_manifest.record_count != CANONICAL_SERVING_RECORD_COUNT:
            raise ArtifactCompatibilityError(
                f"Serving record_count mismatch: expected {CANONICAL_SERVING_RECORD_COUNT}, "
                f"got {chunk_manifest.record_count}"
            )

        needed_cids: set[str] = set()
        for qid in primary_qids:
            r = records_map[qid]
            sel_ev = r.get("response", {}).get("metadata", {}).get("selected_evidence", [])
            for item in sel_ev:
                if isinstance(item, dict) and "chunk_id" in item:
                    needed_cids.add(item["chunk_id"])

        chunks_by_id: dict[str, dict[str, Any]] = {}
        records_file = chunks_dir / "records.jsonl"
        with records_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("chunk_id")
                if cid in needed_cids:
                    chunks_by_id[cid] = row
                    if len(chunks_by_id) == len(needed_cids):
                        break

        missing_cids = needed_cids - set(chunks_by_id.keys())
        if missing_cids:
            raise DataValidationError(
                f"legal_chunks artifact missing {len(missing_cids)} required chunk IDs: {sorted(missing_cids)}"
            )

        return chunk_manifest, chunks_by_id

    def _process_primary_case(
        self,
        *,
        candidate: SelectedHoldoutCandidate,
        record: dict[str, Any],
        question_text: str,
        reference_answer: str,
        chunks_by_id: dict[str, dict[str, Any]],
        chunk_manifest: Any,
        verifier: RuleBasedCitationVerifier,
        dev_sha: str,
        archive_filename: str,
        archive_sha: str | None,
        results_sha: str,
        source_kind: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reconstruct evidence and replay verifier for one PRIMARY holdout candidate."""
        qid = candidate.question_id
        resp_dict = record.get("response", {})
        meta = resp_dict.get("metadata", {})
        agent_meta = meta.get("agent", {})
        hist_cv = meta.get("citation_verification", {})
        stop_reason = str(agent_meta.get("stop_reason", "unknown"))

        # Context selection trace
        context_meta = meta.get("context", {})
        selection_trace = context_meta.get("selection_trace", [])

        # Cross-check selected_evidence against selection_trace
        sel_ev_records = meta.get("selected_evidence", [])
        selected_trace_entries = [
            t for t in selection_trace if isinstance(t, dict) and t.get("selected") is True
        ]
        selected_trace_entries.sort(
            key=lambda t: t.get("selection_rank") if isinstance(t.get("selection_rank"), int) else 0
        )

        source_mapping_pass = True
        if len(sel_ev_records) != len(selected_trace_entries):
            source_mapping_pass = False
        else:
            for idx, (ev_item, trace_item) in enumerate(zip(sel_ev_records, selected_trace_entries, strict=True)):
                expected_eid = f"E{idx + 1}"
                if (
                    ev_item.get("evidence_id") != expected_eid
                    or ev_item.get("chunk_id") != trace_item.get("chunk_id")
                ):
                    source_mapping_pass = False
                    break

        # Reconstruct Evidence
        evidence_list: list[Evidence] = []
        lookup_pass = True
        metadata_crosscheck_pass = True

        for item in sel_ev_records:
            eid = item.get("evidence_id")
            cid = item.get("chunk_id")
            if not eid or not cid or cid not in chunks_by_id:
                lookup_pass = False
                continue

            raw_chunk = chunks_by_id[cid]
            structure = raw_chunk.get("structure") or {}
            chunk_meta = raw_chunk.get("metadata") or {}

            ev = Evidence(
                evidence_id=eid,
                chunk_id=cid,
                document_id=raw_chunk["document_id"],
                text=raw_chunk["text"],
                article_number=structure.get("article_number"),
                article_title=structure.get("article_title"),
                document_title=raw_chunk.get("document_title") or chunk_meta.get("document_title"),
                document_number=raw_chunk.get("document_number") or chunk_meta.get("document_number"),
                document_type=raw_chunk.get("document_type") or chunk_meta.get("document_type"),
                effective_date=raw_chunk.get("effective_date") or chunk_meta.get("effective_date"),
                expiry_date=raw_chunk.get("expiry_date") or chunk_meta.get("expiry_date"),
                effect_status=raw_chunk.get("effect_status") or chunk_meta.get("effect_status"),
                source_url=raw_chunk.get("source_url") or chunk_meta.get("source_url"),
                metadata=chunk_meta,
            )
            evidence_list.append(ev)

        # Cross-check citation metadata
        resp_citations = resp_dict.get("citations", [])
        ev_by_id = {ev.evidence_id: ev for ev in evidence_list}
        for cit in resp_citations:
            ev = ev_by_id.get(cit.get("evidence_id"))
            if ev is None:
                metadata_crosscheck_pass = False
                break
            if (
                cit.get("chunk_id") != ev.chunk_id
                or cit.get("document_id") != ev.document_id
                or cit.get("document_title") != ev.document_title
                or cit.get("document_number") != ev.document_number
                or cit.get("article_number") != ev.article_number
                or cit.get("source_url") != ev.source_url
            ):
                metadata_crosscheck_pass = False
                break

        # Replay Verifier
        replay_pass = False
        try:
            resp_obj = AnswerResponse.model_validate(resp_dict)
            replay_res = verifier.verify(resp_obj, evidence_list)

            hist_is_valid = hist_cv.get("is_valid")
            hist_valid_cits = [c.get("evidence_id") for c in hist_cv.get("valid_citations", [])]
            hist_invalid_cits = [c.get("evidence_id") for c in hist_cv.get("invalid_citations", [])]
            hist_errors = hist_cv.get("errors", [])
            hist_warnings = hist_cv.get("warnings", [])

            replay_is_valid = replay_res.is_valid
            replay_valid_cits = [c.evidence_id for c in replay_res.valid_citations]
            replay_invalid_cits = [c.evidence_id for c in replay_res.invalid_citations]
            replay_errors = replay_res.errors
            replay_warnings = replay_res.warnings

            hist_claims = hist_cv.get("claim_verifications", [])
            replay_claims = [c.model_dump(mode="json") for c in replay_res.claim_verifications]

            claims_match = len(hist_claims) == len(replay_claims)
            if claims_match:
                for hc, rc in zip(hist_claims, replay_claims, strict=True):
                    if (
                        hc.get("claim_id") != rc.get("claim_id")
                        or hc.get("claim_text") != rc.get("claim_text")
                        or hc.get("evidence_ids") != rc.get("evidence_ids")
                        or hc.get("status") != rc.get("status")
                        or hc.get("numeric_match") != rc.get("numeric_match")
                        or hc.get("negation_match") != rc.get("negation_match")
                        or hc.get("errors") != rc.get("errors")
                        or abs(float(hc.get("lexical_support_score", 0.0)) - float(rc.get("lexical_support_score", 0.0))) > 1e-9
                    ):
                        claims_match = False
                        break

            replay_pass = (
                hist_is_valid == replay_is_valid
                and hist_valid_cits == replay_valid_cits
                and hist_invalid_cits == replay_invalid_cits
                and set(hist_errors) == set(replay_errors)
                and set(hist_warnings) == set(replay_warnings)
                and claims_match
            )
            replay_dict = {
                "replay_applicable": True,
                "replay_matches_historical": replay_pass,
                "replay_result": replay_res.model_dump(mode="json"),
            }
        except Exception as exc:
            _LOGGER.error("Replay exception on holdout candidate %s: %s", qid, exc)
            replay_pass = False
            replay_dict = {
                "replay_applicable": True,
                "replay_matches_historical": False,
                "replay_error": str(exc),
            }

        packet = {
            "schema_version": "1.0",
            "question_id": qid,
            "holdout_metadata": {
                "stratum": candidate.stratum,
                "selection_key": candidate.selection_key,
                "pool_type": candidate.pool_type,
                "sampling_algorithm": CANONICAL_SELECTION_ALGORITHM,
                "selection_salt": self._selection_salt,
            },
            "source_identity": {
                "source_kind": source_kind,
                "archive_filename": archive_filename,
                "archive_sha256_observed": archive_sha,
                "phase_a_results_sha256": results_sha,
                "canonical_development_sha256": dev_sha,
                "development_filename": self._development_path.name,
                "serving_artifact_identity": {
                    "artifact_type": chunk_manifest.artifact_type.value,
                    "dataset_name": chunk_manifest.dataset_name,
                    "dataset_revision": chunk_manifest.dataset_revision,
                    "code_version": chunk_manifest.code_version,
                    "record_count": chunk_manifest.record_count,
                    "payload_integrity_verified": True,
                    "payload_sha256": chunk_manifest.metadata.get("payload_sha256"),
                },
            },
            "question": question_text,
            "reference_answer_context": {
                "text": reference_answer,
                "ground_truth_status": (
                    "human_review_context_only_not_claim_entailment_ground_truth"
                ),
            },
            "historical_arm": {
                "historical_response": {
                    "question": resp_dict.get("question"),
                    "answer": resp_dict.get("answer"),
                    "insufficient_evidence": resp_dict.get("insufficient_evidence"),
                    "retrieval_strategy": resp_dict.get("retrieval_strategy"),
                    "citations": resp_dict.get("citations", []),
                    "warnings": resp_dict.get("warnings", []),
                    "trace_id": resp_dict.get("trace_id"),
                },
                "agent_outcome": agent_meta,
                "selected_evidence": [ev.model_dump(mode="json") for ev in evidence_list],
                "context_selection_trace": selection_trace,
                "historical_verification": hist_cv,
                "rule_verifier_replay": replay_dict,
            },
            "human_forensic_review": {
                "review_status": "sealed_unreviewed",
                "claim_labels": None,
                "reviewer_notes": None,
                "root_cause_classification": None,
            },
        }

        summary = {
            "question_id": qid,
            "stratum": candidate.stratum,
            "selection_key": candidate.selection_key,
            "historical_stop_reason": stop_reason,
            "selected_evidence_count": len(evidence_list),
            "selected_chunk_lookup_pass": lookup_pass and len(evidence_list) == len(sel_ev_records),
            "source_mapping_pass": source_mapping_pass,
            "metadata_crosscheck_pass": metadata_crosscheck_pass,
            "rule_verifier_replay_pass": replay_pass,
        }

        return packet, summary

    def _build_full_selection_report(
        self,
        *,
        verdict: str,
        source_kind: str,
        archive_filename: str,
        archive_sha: str | None,
        results_sha: str,
        dev_sha: str,
        chunk_manifest: Any,
        eligible_count: int,
        stratum_assignments: dict[HoldoutStratum, list[dict[str, Any]]],
        primary_candidates: list[SelectedHoldoutCandidate],
        reserve_candidates: list[SelectedHoldoutCandidate],
        primary_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build external full holdout selection commitment report."""
        exclusion_json = json.dumps(self._excluded_qids, ensure_ascii=False, sort_keys=True)
        exclusion_sha = sha256_text(exclusion_json)

        return {
            "schema_version": "1.0",
            "artifact_type": "verification_v2_holdout_selection",
            "selection_version": "v1",
            "verdict": verdict,
            "source_identity": {
                "source_kind": source_kind,
                "archive_filename": archive_filename,
                "archive_sha256_observed": archive_sha,
                "expected_archive_sha256": CANONICAL_PHASE_A_ZIP_SHA256,
                "results_sha256_observed": results_sha,
                "expected_results_sha256": CANONICAL_PHASE_A_RESULTS_SHA256,
                "development_filename": self._development_path.name,
                "development_sha256": dev_sha,
                "serving_artifact_identity": {
                    "artifact_type": chunk_manifest.artifact_type.value,
                    "dataset_name": chunk_manifest.dataset_name,
                    "dataset_revision": chunk_manifest.dataset_revision,
                    "code_version": chunk_manifest.code_version,
                    "record_count": chunk_manifest.record_count,
                    "payload_integrity_verified": True,
                    "payload_sha256": chunk_manifest.metadata.get("payload_sha256"),
                },
            },
            "candidate_sampling": {
                "sampling_algorithm": CANONICAL_SELECTION_ALGORITHM,
                "selection_salt": self._selection_salt,
                "contamination_exclusion_count": len(self._excluded_qids),
                "contamination_exclusion_set_sha256": exclusion_sha,
                "contamination_exclusion_set": self._excluded_qids,
                "eligible_answer_verified_pool_size": eligible_count,
                "strata_distribution_before_sampling": {
                    s.value: len(stratum_assignments.get(s, [])) for s in STRATA_ORDER
                },
                "primary_count": len(primary_candidates),
                "reserve_count": len(reserve_candidates),
            },
            "primary_candidates": [asdict(c) for c in primary_candidates],
            "reserve_candidates": [asdict(c) for c in reserve_candidates],
            "primary_validation": {
                "total_primary_materialized": len(primary_summaries),
                "selected_chunk_lookup_pass_count": sum(1 for s in primary_summaries if s["selected_chunk_lookup_pass"]),
                "source_mapping_pass_count": sum(1 for s in primary_summaries if s["source_mapping_pass"]),
                "metadata_crosscheck_pass_count": sum(1 for s in primary_summaries if s["metadata_crosscheck_pass"]),
                "rule_verifier_replay_pass_count": sum(1 for s in primary_summaries if s["rule_verifier_replay_pass"]),
                "zero_model_reruns_invariant": True,
                "human_labels_populated": False,
                "holdout_sealed": True,
            },
            "per_case_primary": primary_summaries,
        }

    def _build_public_commitment_report(
        self,
        *,
        verdict: str,
        source_kind: str,
        archive_filename: str,
        archive_sha: str | None,
        results_sha: str,
        dev_sha: str,
        chunk_manifest: Any,
        eligible_count: int,
        stratum_assignments: dict[HoldoutStratum, list[dict[str, Any]]],
        primary_candidates: list[SelectedHoldoutCandidate],
        reserve_candidates: list[SelectedHoldoutCandidate],
        primary_summaries: list[dict[str, Any]],
        selection_artifact_path: Path,
    ) -> dict[str, Any]:
        """Build content-free public commitment report with zero QIDs."""
        exclusion_json = json.dumps(self._excluded_qids, ensure_ascii=False, sort_keys=True)
        exclusion_sha = sha256_text(exclusion_json)

        sel_artifact_sha = sha256_file(selection_artifact_path)
        sel_artifact_size = selection_artifact_path.stat().st_size

        primary_strata_counts = Counter(c.stratum for c in primary_candidates)
        reserve_strata_counts = Counter(c.stratum for c in reserve_candidates)

        return {
            "schema_version": "1.0",
            "artifact_type": "verification_v2_holdout_public_commitment",
            "commitment_version": "v1",
            "verdict": verdict,
            "source_identity": {
                "source_kind": source_kind,
                "archive_filename": archive_filename,
                "archive_sha256_observed": archive_sha,
                "expected_archive_sha256": CANONICAL_PHASE_A_ZIP_SHA256,
                "results_sha256_observed": results_sha,
                "expected_results_sha256": CANONICAL_PHASE_A_RESULTS_SHA256,
                "development_filename": self._development_path.name,
                "development_sha256": dev_sha,
                "serving_artifact_identity": {
                    "artifact_type": chunk_manifest.artifact_type.value,
                    "dataset_name": chunk_manifest.dataset_name,
                    "dataset_revision": chunk_manifest.dataset_revision,
                    "code_version": chunk_manifest.code_version,
                    "record_count": chunk_manifest.record_count,
                    "payload_integrity_verified": True,
                    "payload_sha256": chunk_manifest.metadata.get("payload_sha256"),
                },
            },
            "selection_commitment": {
                "selection_artifact_filename": selection_artifact_path.name,
                "selection_artifact_sha256": sel_artifact_sha,
                "selection_artifact_size_bytes": sel_artifact_size,
                "sampling_algorithm": CANONICAL_SELECTION_ALGORITHM,
                "selection_salt": self._selection_salt,
                "contamination_exclusion_count": len(self._excluded_qids),
                "contamination_exclusion_set_sha256": exclusion_sha,
                "eligible_answer_verified_pool_size": eligible_count,
                "strata_distribution_before_sampling": {
                    s.value: len(stratum_assignments.get(s, [])) for s in STRATA_ORDER
                },
                "primary_count": len(primary_candidates),
                "reserve_count": len(reserve_candidates),
                "stratum_primary_counts": {s.value: primary_strata_counts[s.value] for s in STRATA_ORDER},
                "stratum_reserve_counts": {s.value: reserve_strata_counts[s.value] for s in STRATA_ORDER},
            },
            "primary_validation": {
                "total_primary_materialized": len(primary_summaries),
                "selected_chunk_lookup_pass_count": sum(1 for s in primary_summaries if s["selected_chunk_lookup_pass"]),
                "source_mapping_pass_count": sum(1 for s in primary_summaries if s["source_mapping_pass"]),
                "metadata_crosscheck_pass_count": sum(1 for s in primary_summaries if s["metadata_crosscheck_pass"]),
                "rule_verifier_replay_pass_count": sum(1 for s in primary_summaries if s["rule_verifier_replay_pass"]),
                "zero_model_reruns_invariant": True,
                "human_labels_populated": False,
                "holdout_sealed": True,
            },
        }

    def _write_outputs(
        self,
        *,
        primary_packets: dict[str, dict[str, Any]],
        full_selection_report: dict[str, Any],
        primary_candidates: list[SelectedHoldoutCandidate],
        reserve_candidates: list[SelectedHoldoutCandidate],
        source_kind: str,
        archive_filename: str,
        archive_sha: str | None,
        results_sha: str,
        dev_sha: str,
    ) -> dict[str, Path]:
        """Write selection outputs, packets, and report to output directory."""
        exec_dir = self._output_dir / "execution"
        results_dir = self._output_dir / "results"
        packets_dir = self._output_dir / "holdout_packets"

        exec_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        packets_dir.mkdir(parents=True, exist_ok=True)

        identity = {
            "source_kind": source_kind,
            "archive_filename": archive_filename,
            "archive_sha256_observed": archive_sha,
            "results_sha256_observed": results_sha,
            "canonical_development_sha256": dev_sha,
            "development_filename": self._development_path.name,
            "primary_candidate_count": len(primary_candidates),
            "reserve_candidate_count": len(reserve_candidates),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (exec_dir / "holdout_source_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (results_dir / "holdout_selection_commitment.json").write_text(
            json.dumps(full_selection_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        selection_artifact_path = results_dir / "verification-v2-holdout-selection-v1.json"
        selection_artifact_path.write_text(
            json.dumps(full_selection_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (results_dir / "primary_holdout_identity.json").write_text(
            json.dumps([asdict(c) for c in primary_candidates], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (results_dir / "fresh_reserve_identity.json").write_text(
            json.dumps([asdict(c) for c in reserve_candidates], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for qid, packet in primary_packets.items():
            (packets_dir / f"{qid}.json").write_text(
                json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        return {
            "selection_artifact": selection_artifact_path,
        }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Pre-register and materialize sealed fresh V2 holdout candidates from frozen Phase-A evidence."
    )
    parser.add_argument(
        "--phase-a-evidence",
        type=Path,
        required=True,
        help="Path to phase-a-current-system-census-final-evidence.zip or extracted directory.",
    )
    parser.add_argument(
        "--serving-root",
        type=Path,
        required=True,
        help="Path to canonical serving artifact root (e.g. artifacts/uit-dsc-2026-task2-v0400).",
    )
    parser.add_argument(
        "--development",
        type=Path,
        required=True,
        help="Path to canonical development.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for V2 holdout selection artifacts.",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Optional output ZIP path for packaging sealed review packets.",
    )

    args = parser.parse_args()

    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=args.phase_a_evidence,
        serving_root=args.serving_root,
        development_path=args.development,
        output_dir=args.output_dir,
    )

    public_commitment = registrar.run()

    # Content-free console summary (ZERO QIDs printed)
    print(json.dumps(public_commitment, ensure_ascii=False, indent=2))

    if args.package_zip:
        args.package_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.package_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(args.output_dir):
                for f in sorted(files):
                    fp = Path(root) / f
                    arcname = fp.relative_to(args.output_dir).as_posix()
                    z.write(fp, arcname=arcname)
        zip_sha = sha256_file(args.package_zip)
        zip_size = args.package_zip.stat().st_size
        print(f"\nSealed review ZIP written to: {args.package_zip.name}")
        print(f"ZIP SHA-256: {zip_sha}")
        print(f"ZIP Size (bytes): {zip_size}")


if __name__ == "__main__":
    main()
