#!/usr/bin/env python3
"""Materialize paired forensic source packets from frozen B1A historical records.

This script implements Priority B Forensic Source Materialization (B-FORENSIC-0).
It operates strictly in READ-ONLY replay mode:
- Zero retrieval reruns
- Zero generation reruns
- Zero model-backed semantic verification
- Zero auto-generation of legal correctness labels
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from legal_agentic_rag import __version__
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
from legal_agentic_rag.runtime.artifact_store import load_artifact_manifest
from legal_agentic_rag.schemas.answering import AnswerResponse, Evidence
from legal_agentic_rag.schemas.manifests import ArtifactType

_LOGGER = logging.getLogger(__name__)

CANONICAL_B1A_ZIP_SHA256 = (
    "b88ccce928b4cecfc9239d490c59f91405834c5a3199f917d757c5735b1d6631"
)
CANONICAL_DEVELOPMENT_SHA256 = (
    "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"
)
CANONICAL_MATERIALIZED_QUESTIONS_SHA256 = (
    "f5d681c447a2bb964de90298207af0363c76b3546bfa027603d7fa98322a3ce3"
)
CANONICAL_BASE_RESULTS_SHA256 = (
    "c72dc38f37945831095c71ba4b0f24f328de2e01dc4f7ecac6e527b997dc3cac"
)
CANONICAL_CANDIDATE_RESULTS_SHA256 = (
    "420d2297d529c3d8c246de499111840d4a4a98b010bd95e42639b7ba9f3fb6ad"
)

CANONICAL_SERVING_DATASET_NAME = "uit-dsc-2026-task2-selected-contexts"
CANONICAL_SERVING_DATASET_REVISION = (
    "sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e"
)
CANONICAL_SERVING_RECORD_COUNT = 330768

CANONICAL_TARGET_IDS = ["102047", "147239", "26541", "95861"]
CANONICAL_ARM_NAMES = ["BASE", "CANDIDATE"]

CANONICAL_B1A_MEMBER_HASHES: dict[str, str] = {
    "configs/phase-b1a-graph-routing-cases.json": "b1efe824f320d9323af462869fd8842ef8544fa14d5f81ae35decca99e1ee99f",
    "evidence/materialized_questions_identity.json": "abad62cb31dc24bc40213ada580f8b464bfe2f98d1340d3820fd10de338ebcd3",
    "configs/base_runtime_config.json": "03a32009c0dc9a68ac93538710ec741b7a7e68a8ef9e160116ecc6bcb76d64fc",
    "configs/candidate_runtime_config.json": "27a490b947336a2f3aa0c34e9f7a19494be28193f10e5934c5909930fea7f99a",
    "results/phase_b1a_paired_report.json": "ed7b5129539a4f31b4f8b9153ef8060d75d58cebee1d7932b346d63b0dc1e0e7",
    "results/phase_b1a_decision_report.json": "6fce0e2daf6af1a50fdc1ed41bba271b99c7f6844e73edd1f3f52babdd365c5e",
    "base_batch/manifest.json": "72cb07ae18d9539357e693b0bd4385565980b9d22b5cac054cb7d3a0a0012406",
    "candidate_batch/manifest.json": "c64baf71ff13ccb2d864e84e875b06bdeb204dd636a45ad25ba6b3bed2499908",
    "base_batch/results.jsonl": "c72dc38f37945831095c71ba4b0f24f328de2e01dc4f7ecac6e527b997dc3cac",
    "candidate_batch/results.jsonl": "420d2297d529c3d8c246de499111840d4a4a98b010bd95e42639b7ba9f3fb6ad",
    "base_batch/batch_state.json": "af15d1676144570ea75c9183119ea6c6890aed062554db516c0a3ddb642cb159",
    "candidate_batch/batch_state.json": "11054fe5a9568da70f4e3f694185b846a8097eb90e37233825db1a02635ae219",
}

REQUIRED_B1A_MEMBERS = list(CANONICAL_B1A_MEMBER_HASHES.keys())


def sha256_file(path: Path) -> str:
    """Compute deterministic SHA-256 hex digest for a file."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest for bytes."""
    return sha256(data).hexdigest()


@dataclass(frozen=True)
class ArmValidationSummary:
    question_id: str
    arm: str
    historical_stop_reason: str
    historical_verifier_present: bool
    replay_applicable: bool
    selected_evidence_count: int
    selected_chunk_lookup_pass: bool
    source_mapping_pass: bool
    metadata_crosscheck_pass: bool
    rule_verifier_replay_pass: bool
    replay_reason: str | None = None


class ForensicSourceMaterializer:
    """Validate frozen B1A evidence and materialize paired forensic packets."""

    def __init__(
        self,
        *,
        b1a_evidence_path: Path,
        serving_root: Path,
        development_path: Path,
        output_dir: Path,
        target_ids: Sequence[str] | None = None,
    ) -> None:
        self._b1a_evidence_path = b1a_evidence_path.resolve()
        self._serving_root = serving_root.resolve()
        self._development_path = development_path.resolve()
        self._output_dir = output_dir.resolve()
        self._target_ids = list(target_ids or CANONICAL_TARGET_IDS)

        # Validate unique non-empty target IDs
        if not self._target_ids:
            raise DataValidationError("Target IDs list cannot be empty")
        if len(self._target_ids) != len(set(self._target_ids)):
            raise DataValidationError(f"Target IDs contain duplicates: {self._target_ids}")

    def run(self) -> dict[str, Any]:
        """Execute full materialization pipeline and produce forensic report."""
        # 1. Validate Development JSON
        dev_questions, dev_sha = self._load_and_validate_development(self._development_path)

        # 2. Validate and Load B1A Evidence Archive/Bundle
        (
            b1a_bundle_dir,
            b1a_cleanup_dir,
            source_kind,
            archive_filename,
            archive_sha,
            member_hashes,
        ) = self._resolve_b1a_evidence(self._b1a_evidence_path)

        try:
            # 3. Validate B1A Manifests and Load Historical Records
            base_manifest, base_records = self._load_arm_batch(
                b1a_bundle_dir, "base_batch", CANONICAL_BASE_RESULTS_SHA256
            )
            cand_manifest, cand_records = self._load_arm_batch(
                b1a_bundle_dir, "candidate_batch", CANONICAL_CANDIDATE_RESULTS_SHA256
            )

            # Validate materialized questions identity
            self._validate_materialized_identity(b1a_bundle_dir)

            # 4. Validate Serving Artifacts with in-protocol payload check and Load Legal Chunks Index
            chunks_dir = self._find_legal_chunks_dir(self._serving_root)
            chunk_manifest, chunks_by_id = self._load_needed_chunks(
                chunks_dir=chunks_dir,
                target_ids=self._target_ids,
                base_records=base_records,
                cand_records=cand_records,
            )

            # 5. Process Target Questions & Replay Verifier
            arm_summaries: list[ArmValidationSummary] = []
            materialized_packets: dict[str, dict[str, Any]] = {}

            verifier = RuleBasedCitationVerifier()

            for qid in self._target_ids:
                if qid not in base_records:
                    raise DataValidationError(f"Target question ID '{qid}' missing from BASE batch")
                if qid not in cand_records:
                    raise DataValidationError(f"Target question ID '{qid}' missing from CANDIDATE batch")
                if qid not in dev_questions:
                    raise DataValidationError(f"Target question ID '{qid}' missing from development.json")

                q_entry = dev_questions[qid]
                canonical_question_text = q_entry.get("question", "")
                reference_answer_text = q_entry.get("answer", "")

                base_rec = base_records[qid]
                cand_rec = cand_records[qid]

                # Process BASE arm
                base_packet_arm, base_summary = self._process_arm(
                    qid=qid,
                    arm_name="BASE",
                    record=base_rec,
                    chunks_by_id=chunks_by_id,
                    verifier=verifier,
                )
                arm_summaries.append(base_summary)

                # Process CANDIDATE arm
                cand_packet_arm, cand_summary = self._process_arm(
                    qid=qid,
                    arm_name="CANDIDATE",
                    record=cand_rec,
                    chunks_by_id=chunks_by_id,
                    verifier=verifier,
                )
                arm_summaries.append(cand_summary)

                # Build Paired Packet with Transport-Neutral Provenance
                packet = {
                    "schema_version": "1.0",
                    "question_id": qid,
                    "source_identity": {
                        "source_kind": source_kind,
                        "archive_filename": archive_filename,
                        "archive_sha256_observed": archive_sha,
                        "canonical_zip_sha256_expected": CANONICAL_B1A_ZIP_SHA256,
                        "base_results_sha256": CANONICAL_BASE_RESULTS_SHA256,
                        "candidate_results_sha256": CANONICAL_CANDIDATE_RESULTS_SHA256,
                        "code_version": base_manifest.get("code_version", "0.50.6"),
                        "materialized_question_source_sha256": CANONICAL_MATERIALIZED_QUESTIONS_SHA256,
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
                    "question": canonical_question_text,
                    "reference_answer_context": {
                        "text": reference_answer_text,
                        "ground_truth_status": (
                            "human_review_context_only_not_claim_entailment_ground_truth"
                        ),
                    },
                    "arms": {
                        "BASE": base_packet_arm,
                        "CANDIDATE": cand_packet_arm,
                    },
                    "human_forensic_review": {
                        "review_status": "unreviewed",
                        "base_claim_labels": None,
                        "candidate_claim_labels": None,
                        "cross_arm_notes": None,
                        "root_cause_classification": None,
                    },
                }
                materialized_packets[qid] = packet

            # 6. Check Overall Verdict Invariants
            all_chunks_lookup_pass = all(s.selected_chunk_lookup_pass for s in arm_summaries)
            all_source_mapping_pass = all(s.source_mapping_pass for s in arm_summaries)
            all_metadata_pass = all(s.metadata_crosscheck_pass for s in arm_summaries)
            all_applicable_replay_pass = all(
                s.rule_verifier_replay_pass
                for s in arm_summaries
                if s.replay_applicable
            )

            if not (
                all_chunks_lookup_pass
                and all_source_mapping_pass
                and all_metadata_pass
                and all_applicable_replay_pass
            ):
                verdict = "INVALID_FORENSIC_PROVENANCE"
            else:
                verdict = "FORENSIC_SOURCE_READY"

            # 7. Write Materialized Outputs
            report = self._build_report(
                verdict=verdict,
                source_kind=source_kind,
                archive_filename=archive_filename,
                archive_sha=archive_sha,
                dev_sha=dev_sha,
                chunk_manifest=chunk_manifest,
                base_manifest=base_manifest,
                cand_manifest=cand_manifest,
                arm_summaries=arm_summaries,
                member_hashes=member_hashes,
            )

            self._write_outputs(
                materialized_packets=materialized_packets,
                report=report,
                source_kind=source_kind,
                archive_filename=archive_filename,
                archive_sha=archive_sha,
                dev_sha=dev_sha,
            )

            return report

        finally:
            if b1a_cleanup_dir is not None and b1a_cleanup_dir.exists():
                import shutil
                shutil.rmtree(b1a_cleanup_dir, ignore_errors=True)

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

    def _resolve_b1a_evidence(
        self, path: Path
    ) -> tuple[Path, Path | None, str, str, str | None, dict[str, str]]:
        """Verify and resolve B1A evidence ZIP or extracted directory fail-closed."""
        if not path.exists():
            raise DataValidationError(f"B1A evidence path does not exist: {path}")

        if path.is_file() and path.suffix.lower() == ".zip":
            actual_sha = sha256_file(path)
            if actual_sha != CANONICAL_B1A_ZIP_SHA256:
                raise DataValidationError(
                    f"B1A ZIP SHA mismatch: expected {CANONICAL_B1A_ZIP_SHA256}, got {actual_sha}"
                )
            temp_unpack = Path(tempfile.mkdtemp(prefix="b1a_evidence_unpack_"))
            with zipfile.ZipFile(path, "r") as z:
                names = z.namelist()
                for req in REQUIRED_B1A_MEMBERS:
                    if req not in names:
                        raise DataValidationError(f"B1A evidence missing required member '{req}'")
                z.extractall(temp_unpack)

            # Verify all member hashes
            member_hashes: dict[str, str] = {}
            for req, expected_hash in CANONICAL_B1A_MEMBER_HASHES.items():
                member_file = temp_unpack / req
                m_sha = sha256_file(member_file)
                if m_sha != expected_hash:
                    raise DataValidationError(
                        f"B1A ZIP unpacked member '{req}' SHA mismatch: expected {expected_hash}, got {m_sha}"
                    )
                member_hashes[req] = m_sha

            return (
                temp_unpack,
                temp_unpack,
                "canonical_zip",
                path.name,
                actual_sha,
                member_hashes,
            )

        if path.is_dir():
            member_hashes = {}
            for req, expected_hash in CANONICAL_B1A_MEMBER_HASHES.items():
                member_file = path / req
                if not member_file.is_file():
                    raise DataValidationError(f"B1A extracted directory missing required member '{req}'")
                m_sha = sha256_file(member_file)
                if m_sha != expected_hash:
                    raise DataValidationError(
                        f"B1A extracted directory member '{req}' SHA mismatch: expected {expected_hash}, got {m_sha}"
                    )
                member_hashes[req] = m_sha

            return (
                path,
                None,
                "canonical_extracted_bundle",
                path.name,
                None,
                member_hashes,
            )

        raise DataValidationError(f"Invalid B1A evidence path: {path}")

    def _validate_materialized_identity(self, bundle_dir: Path) -> None:
        """Validate materialized questions identity against canonical standards."""
        ident_path = bundle_dir / "evidence" / "materialized_questions_identity.json"
        ident = json.loads(ident_path.read_text(encoding="utf-8"))

        if ident.get("source_question_sha256") != CANONICAL_DEVELOPMENT_SHA256:
            raise DataValidationError(
                f"materialized_questions_identity source_question_sha256 mismatch: "
                f"expected {CANONICAL_DEVELOPMENT_SHA256}, got {ident.get('source_question_sha256')}"
            )
        if ident.get("materialized_case_count") != 22:
            raise DataValidationError(
                f"materialized_case_count mismatch: expected 22, got {ident.get('materialized_case_count')}"
            )
        if ident.get("materialized_case_sha256") != CANONICAL_MATERIALIZED_QUESTIONS_SHA256:
            raise DataValidationError(
                f"materialized_case_sha256 mismatch: expected {CANONICAL_MATERIALIZED_QUESTIONS_SHA256}, "
                f"got {ident.get('materialized_case_sha256')}"
            )

    def _load_arm_batch(
        self,
        bundle_dir: Path,
        arm_dir_name: str,
        expected_results_sha: str,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Load manifest and results.jsonl for one arm, validating full fail-closed gates."""
        manifest_path = bundle_dir / arm_dir_name / "manifest.json"
        results_path = bundle_dir / arm_dir_name / "results.jsonl"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_results_sha = sha256_file(results_path)

        if actual_results_sha != expected_results_sha:
            raise DataValidationError(
                f"{arm_dir_name} results.jsonl SHA mismatch: expected {expected_results_sha}, got {actual_results_sha}"
            )
        if manifest.get("records_sha256") != actual_results_sha:
            raise DataValidationError(
                f"{arm_dir_name} manifest records_sha256 mismatch: "
                f"expected {manifest.get('records_sha256')}, got {actual_results_sha}"
            )
        if manifest.get("record_count") != 22:
            raise DataValidationError(
                f"{arm_dir_name} manifest record_count mismatch: expected 22, got {manifest.get('record_count')}"
            )
        if manifest.get("code_version") != "0.50.6":
            raise DataValidationError(
                f"{arm_dir_name} manifest code_version mismatch: expected '0.50.6', got {manifest.get('code_version')}"
            )
        if manifest.get("question_source_sha256") != CANONICAL_MATERIALIZED_QUESTIONS_SHA256:
            raise DataValidationError(
                f"{arm_dir_name} manifest question_source_sha256 mismatch: "
                f"expected {CANONICAL_MATERIALIZED_QUESTIONS_SHA256}, got {manifest.get('question_source_sha256')}"
            )

        records_raw = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records_raw) != 22:
            raise DataValidationError(
                f"{arm_dir_name} results line count mismatch: expected 22, got {len(records_raw)}"
            )

        record_map = {}
        for r in records_raw:
            qid = str(r.get("question_id", "")).strip()
            if not qid:
                raise DataValidationError(f"{arm_dir_name} contains record with blank question_id")
            if qid in record_map:
                raise DataValidationError(f"{arm_dir_name} contains duplicate question_id: '{qid}'")
            record_map[qid] = r

        return manifest, record_map

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
        target_ids: Sequence[str],
        base_records: dict[str, dict[str, Any]],
        cand_records: dict[str, dict[str, Any]],
    ) -> tuple[Any, dict[str, dict[str, Any]]]:
        """Validate legal_chunks payload in-protocol and load only the chunk records needed by target cases."""
        chunk_manifest = load_artifact_manifest(
            chunks_dir,
            expected_type=ArtifactType.LEGAL_CHUNKS,
            verify_payload=True,
        )

        # Gate canonical serving identity
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
        for qid in target_ids:
            for rec_dict in [base_records.get(qid), cand_records.get(qid)]:
                if not rec_dict:
                    continue
                meta = rec_dict.get("response", {}).get("metadata", {})
                sel_ev = meta.get("selected_evidence", [])
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

    def _process_arm(
        self,
        *,
        qid: str,
        arm_name: str,
        record: dict[str, Any],
        chunks_by_id: dict[str, dict[str, Any]],
        verifier: RuleBasedCitationVerifier,
    ) -> tuple[dict[str, Any], ArmValidationSummary]:
        """Reconstruct evidence, cross-check selection trace, validate metadata, and replay verifier."""
        resp_dict = record.get("response", {})
        meta = resp_dict.get("metadata", {})
        agent_meta = meta.get("agent", {})
        hist_cv = meta.get("citation_verification")
        stop_reason = str(agent_meta.get("stop_reason", "unknown"))

        # Context selection trace is located at response.metadata.context.selection_trace
        context_meta = meta.get("context", {})
        selection_trace = context_meta.get("selection_trace", [])

        # FIX 4: Cross-check selected_evidence against selection_trace
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

        # Replay Rule-Based Verifier if applicable
        replay_applicable = hist_cv is not None and stop_reason == "answer_verified"
        replay_pass = False
        replay_dict: dict[str, Any]

        if not replay_applicable:
            replay_dict = {
                "replay_applicable": False,
                "reason": "historical_verifier_not_reached",
            }
        else:
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

                # Check claim status and details
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
                _LOGGER.error("Replay exception on case %s arm %s: %s", qid, arm_name, exc)
                replay_pass = False
                replay_dict = {
                    "replay_applicable": True,
                    "replay_matches_historical": False,
                    "replay_error": str(exc),
                }

        packet_arm = {
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
        }

        summary = ArmValidationSummary(
            question_id=qid,
            arm=arm_name,
            historical_stop_reason=stop_reason,
            historical_verifier_present=hist_cv is not None,
            replay_applicable=replay_applicable,
            selected_evidence_count=len(evidence_list),
            selected_chunk_lookup_pass=lookup_pass and len(evidence_list) == len(sel_ev_records),
            source_mapping_pass=source_mapping_pass,
            metadata_crosscheck_pass=metadata_crosscheck_pass,
            rule_verifier_replay_pass=replay_pass if replay_applicable else True,
            replay_reason=None if replay_applicable else "historical_verifier_not_reached",
        )

        return packet_arm, summary

    def _build_report(
        self,
        *,
        verdict: str,
        source_kind: str,
        archive_filename: str,
        archive_sha: str | None,
        dev_sha: str,
        chunk_manifest: Any,
        base_manifest: dict[str, Any],
        cand_manifest: dict[str, Any],
        arm_summaries: list[ArmValidationSummary],
        member_hashes: dict[str, str],
    ) -> dict[str, Any]:
        """Construct the standardized forensic source report without local machine paths."""
        replay_applicable_summaries = [s for s in arm_summaries if s.replay_applicable]
        replay_pass_count = sum(1 for s in replay_applicable_summaries if s.rule_verifier_replay_pass)

        return {
            "schema_version": "1.0",
            "verdict": verdict,
            "source_archive_identity": {
                "source_kind": source_kind,
                "archive_filename": archive_filename,
                "archive_sha256_observed": archive_sha,
                "canonical_zip_sha256_expected": CANONICAL_B1A_ZIP_SHA256,
                "canonical_members_verified": len(member_hashes) == len(REQUIRED_B1A_MEMBERS),
                "member_hashes": member_hashes,
            },
            "base_results_identity": {
                "manifest_records_sha256": base_manifest.get("records_sha256"),
                "expected_records_sha256": CANONICAL_BASE_RESULTS_SHA256,
                "record_count": base_manifest.get("record_count"),
                "code_version": base_manifest.get("code_version"),
                "question_source_sha256": base_manifest.get("question_source_sha256"),
            },
            "candidate_results_identity": {
                "manifest_records_sha256": cand_manifest.get("records_sha256"),
                "expected_records_sha256": CANONICAL_CANDIDATE_RESULTS_SHA256,
                "record_count": cand_manifest.get("record_count"),
                "code_version": cand_manifest.get("code_version"),
                "question_source_sha256": cand_manifest.get("question_source_sha256"),
            },
            "canonical_development_identity": {
                "filename": self._development_path.name,
                "sha256": dev_sha,
                "expected_sha256": CANONICAL_DEVELOPMENT_SHA256,
            },
            "serving_artifact_identity": {
                "artifact_type": chunk_manifest.artifact_type.value,
                "dataset_name": chunk_manifest.dataset_name,
                "dataset_revision": chunk_manifest.dataset_revision,
                "code_version": chunk_manifest.code_version,
                "record_count": chunk_manifest.record_count,
                "payload_integrity_verified": True,
                "payload_sha256": chunk_manifest.metadata.get("payload_sha256"),
            },
            "target_question_count": len(self._target_ids),
            "historical_arm_count": len(arm_summaries),
            "per_arm": [asdict(s) for s in arm_summaries],
            "aggregate": {
                "base_records_valid": 22,
                "candidate_records_valid": 22,
                "targets_present_base": len(self._target_ids),
                "targets_present_candidate": len(self._target_ids),
                "selected_chunk_lookup_pass_count": sum(1 for s in arm_summaries if s.selected_chunk_lookup_pass),
                "selected_chunk_lookup_total": len(arm_summaries),
                "source_mapping_pass_count": sum(1 for s in arm_summaries if s.source_mapping_pass),
                "metadata_crosscheck_pass_count": sum(1 for s in arm_summaries if s.metadata_crosscheck_pass),
                "replay_applicable_count": len(replay_applicable_summaries),
                "replay_pass_count": replay_pass_count,
                "zero_rerun_invariant": True,
                "human_labels_populated": False,
            },
        }

    def _write_outputs(
        self,
        *,
        materialized_packets: dict[str, dict[str, Any]],
        report: dict[str, Any],
        source_kind: str,
        archive_filename: str,
        archive_sha: str | None,
        dev_sha: str,
    ) -> None:
        """Write forensic packets and reports to the designated output directory without machine paths."""
        exec_dir = self._output_dir / "execution"
        results_dir = self._output_dir / "results"
        packets_dir = self._output_dir / "forensic_packets"

        exec_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        packets_dir.mkdir(parents=True, exist_ok=True)

        identity = {
            "source_kind": source_kind,
            "archive_filename": archive_filename,
            "archive_sha256_observed": archive_sha,
            "canonical_zip_sha256_expected": CANONICAL_B1A_ZIP_SHA256,
            "base_results_sha256": CANONICAL_BASE_RESULTS_SHA256,
            "candidate_results_sha256": CANONICAL_CANDIDATE_RESULTS_SHA256,
            "canonical_development_sha256": dev_sha,
            "development_filename": self._development_path.name,
            "target_question_ids": self._target_ids,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (exec_dir / "forensic_source_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (results_dir / "forensic_source_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for qid, packet in materialized_packets.items():
            (packets_dir / f"{qid}.json").write_text(
                json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize paired forensic source packets from frozen B1A records."
    )
    parser.add_argument(
        "--b1a-evidence",
        type=Path,
        required=True,
        help="Path to phase-b1a-graph-routing-ablation-evidence.zip or extracted directory.",
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
        help="Output directory for generated forensic packets and reports.",
    )
    parser.add_argument(
        "--target-ids",
        type=str,
        default="102047,147239,26541,95861",
        help="Comma-separated list of target question IDs.",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Optional output ZIP file path for packaging materialized packets and report.",
    )

    args = parser.parse_args()

    targets = [qid.strip() for qid in args.target_ids.split(",") if qid.strip()]

    materializer = ForensicSourceMaterializer(
        b1a_evidence_path=args.b1a_evidence,
        serving_root=args.serving_root,
        development_path=args.development,
        output_dir=args.output_dir,
        target_ids=targets,
    )

    report = materializer.run()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.package_zip:
        args.package_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.package_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(args.output_dir):
                for f in sorted(files):
                    fp = Path(root) / f
                    arcname = fp.relative_to(args.output_dir).as_posix()
                    z.write(fp, arcname=arcname)
        print(f"\nPackaged evidence ZIP written to: {args.package_zip}")


if __name__ == "__main__":
    main()
