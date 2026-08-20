#!/usr/bin/env python3
"""Freeze human-approved positive-control labels v1 (B-FORENSIC-1C).

This script ingests the reviewed positive-control review package
`verification-positive-control-review-packets-v1.zip`, binds human-approved
claim entailment and error mode tags to exact UTF-8 claim text SHA-256 digests,
validates exact aggregate distributions, and generates the immutable external
overlay artifact `verification-positive-control-human-labels-v1.json`.

Invariants:
- All labels are strictly INTERNAL EVALUATION-ONLY
- Labels are NOT organizer ground truth or retrieval relevance labels
- Labels are NOT training or fine-tuning data
- Zero live retrieval, reranking, generation, or model execution
- Output is 100% deterministic
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

from legal_agentic_rag.exceptions import DataValidationError

_LOGGER = logging.getLogger(__name__)

CANONICAL_REVIEW_ZIP_SHA256 = (
    "cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4"
)
CANONICAL_REVIEW_ZIP_SIZE = 110095
CANONICAL_REVIEW_ZIP_FILENAME = "verification-positive-control-review-packets-v1.zip"

CANONICAL_PHASE_A_ZIP_SHA256 = (
    "df05a401599c43a28e39136d72b225841b242d10a40dc5bc475b9be6ed86be8b"
)
CANONICAL_PHASE_A_RESULTS_SHA256 = (
    "7b1bf802c752e37cee7386c0b24f6e0ee5ea2f65056b22eaa9488d73161aaee6"
)

CANONICAL_APPROVAL_KIND = "explicit_user_human_approval"
CANONICAL_APPROVAL_DATE = "2026-08-20"
CANONICAL_REVIEWER_ID = "human_reviewer_1"
CANONICAL_APPROVAL_STATEMENT = (
    "These labels are internal human forensic annotations over frozen, "
    "train-derived development outputs and their exact supplied frozen evidence. "
    "They are not official UIT DSC ground truth, retrieval relevance labels, "
    "public/private test annotations, or training labels."
)


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


class HumanEntailmentLabel(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class LegalErrorTag(StrEnum):
    NONE = "NONE"
    CONDITION_INVERTED = "CONDITION_INVERTED"
    CONDITION_OMITTED = "CONDITION_OMITTED"
    ACTOR_ROLE_INVERTED = "ACTOR_ROLE_INVERTED"
    SCOPE_OVERGENERALIZED = "SCOPE_OVERGENERALIZED"
    WRONG_DOCUMENT = "WRONG_DOCUMENT"
    WRONG_ARTICLE = "WRONG_ARTICLE"
    TEMPORAL_MISAPPLICATION = "TEMPORAL_MISAPPLICATION"
    QUANTITY_ERROR = "QUANTITY_ERROR"
    PROCEDURE_MISORDERED = "PROCEDURE_MISORDERED"
    OTHER = "OTHER"


@dataclass(frozen=True)
class ApprovedClaimReview:
    claim_id: str
    entailment_label: HumanEntailmentLabel
    error_tags: tuple[LegalErrorTag, ...]
    diagnostic_note: str | None = None


# Canonical Human-Approved Labels Matrix for the 16 PRIMARY Positive-Control Candidates
APPROVED_POSITIVE_CONTROLS_MATRIX: dict[str, dict[str, ApprovedClaimReview]] = {
    # Stratum A: SINGLE-CLAIM CLEAN
    "75171": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.SCOPE_OVERGENERALIZED, LegalErrorTag.WRONG_ARTICLE),
        ),
    },
    "150131": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.CONTRADICTED,
            error_tags=(LegalErrorTag.ACTOR_ROLE_INVERTED, LegalErrorTag.WRONG_DOCUMENT),
        ),
    },
    "30405": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.SCOPE_OVERGENERALIZED, LegalErrorTag.WRONG_DOCUMENT),
        ),
    },
    "36801": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
    },
    # Stratum B: MULTI-CLAIM CLEAN
    "116877": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C2": ApprovedClaimReview(
            claim_id="C2",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C3": ApprovedClaimReview(
            claim_id="C3",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
    },
    "15181": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C2": ApprovedClaimReview(
            claim_id="C2",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C3": ApprovedClaimReview(
            claim_id="C3",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
    },
    "5967": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.SCOPE_OVERGENERALIZED, LegalErrorTag.OTHER),
            diagnostic_note="rank-level mismatch",
        ),
        "C2": ApprovedClaimReview(
            claim_id="C2",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.SCOPE_OVERGENERALIZED, LegalErrorTag.OTHER),
            diagnostic_note="rank-level mismatch",
        ),
        "C3": ApprovedClaimReview(
            claim_id="C3",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.SCOPE_OVERGENERALIZED, LegalErrorTag.OTHER),
            diagnostic_note="rank-level mismatch",
        ),
    },
    "139413": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C2": ApprovedClaimReview(
            claim_id="C2",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C3": ApprovedClaimReview(
            claim_id="C3",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
    },
    # Stratum C: NUMERIC
    "34351": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C2": ApprovedClaimReview(
            claim_id="C2",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.OTHER,),
            diagnostic_note="anesthesia specificity unsupported by exact cited evidence",
        ),
    },
    "31883": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
    },
    "40489": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.OTHER,),
            diagnostic_note="legal-instrument type mismatch",
        ),
    },
    "155139": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.QUANTITY_ERROR, LegalErrorTag.WRONG_ARTICLE),
        ),
    },
    # Stratum D: NEGATION / MODALITY STRESS
    "108497": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.INSUFFICIENT,
            error_tags=(LegalErrorTag.SCOPE_OVERGENERALIZED, LegalErrorTag.WRONG_DOCUMENT),
        ),
    },
    "4031": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
    },
    "103983": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C2": ApprovedClaimReview(
            claim_id="C2",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
        "C3": ApprovedClaimReview(
            claim_id="C3",
            entailment_label=HumanEntailmentLabel.CONTRADICTED,
            error_tags=(LegalErrorTag.CONDITION_OMITTED, LegalErrorTag.QUANTITY_ERROR, LegalErrorTag.WRONG_ARTICLE),
        ),
    },
    "140693": {
        "C1": ApprovedClaimReview(
            claim_id="C1",
            entailment_label=HumanEntailmentLabel.SUPPORTED,
            error_tags=(LegalErrorTag.NONE,),
        ),
    },
}


class PositiveControlLabelFreezer:
    """Validate review package, bind human labels to exact claim texts, and build overlay."""

    def __init__(
        self,
        *,
        review_packets_path: Path,
        output_json_path: Path,
        output_zip_path: Path | None = None,
    ) -> None:
        self._review_packets_path = review_packets_path.resolve()
        self._output_json_path = output_json_path.resolve()
        self._output_zip_path = output_zip_path.resolve() if output_zip_path else None

    def run(self) -> dict[str, Any]:
        """Execute validation and generate frozen positive-control human label artifact."""
        # 1. Validate Review Packets Source
        (
            bundle_dir,
            cleanup_dir,
            archive_filename,
            archive_sha,
            archive_size,
        ) = self._resolve_review_packets_source(self._review_packets_path)

        try:
            # 2. Ingest Packets and Validate Pre-Registration Contracts
            packets_by_qid = self._load_and_validate_packets(bundle_dir)

            # 3. Construct Bound Human Label Overlay
            labels_overlay = self._build_labels_overlay(
                packets_by_qid=packets_by_qid,
                archive_filename=archive_filename,
                archive_sha=archive_sha,
                archive_size=archive_size,
            )

            # 4. Write Canonical JSON Artifact
            self._write_json_artifact(labels_overlay, self._output_json_path)

            # 5. Optional Transport Packaging
            if self._output_zip_path:
                self._write_transport_zip(
                    json_path=self._output_json_path,
                    zip_path=self._output_zip_path,
                )

            return labels_overlay

        finally:
            if cleanup_dir is not None and cleanup_dir.exists():
                import shutil
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def _resolve_review_packets_source(
        self, path: Path
    ) -> tuple[Path, Path | None, str, str, int]:
        """Verify review packets archive SHA-256, size, and members fail-closed."""
        if not path.exists():
            raise DataValidationError(f"Review packets path does not exist: {path}")

        if path.is_file() and path.suffix.lower() == ".zip":
            actual_sha = sha256_file(path)
            actual_size = path.stat().st_size
            if actual_sha != CANONICAL_REVIEW_ZIP_SHA256:
                raise DataValidationError(
                    f"Review packets ZIP SHA mismatch: expected {CANONICAL_REVIEW_ZIP_SHA256}, got {actual_sha}"
                )
            if actual_size != CANONICAL_REVIEW_ZIP_SIZE:
                raise DataValidationError(
                    f"Review packets ZIP size mismatch: expected {CANONICAL_REVIEW_ZIP_SIZE}, got {actual_size}"
                )

            temp_unpack = Path(tempfile.mkdtemp(prefix="control_review_packets_unpack_"))
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(temp_unpack)

            return temp_unpack, temp_unpack, path.name, actual_sha, actual_size

        if path.is_dir():
            exec_file = path / "execution" / "control_source_identity.json"
            if not exec_file.is_file():
                raise DataValidationError("Review directory missing execution/control_source_identity.json")
            return path, None, path.name, CANONICAL_REVIEW_ZIP_SHA256, CANONICAL_REVIEW_ZIP_SIZE

        raise DataValidationError(f"Invalid review packets path: {path}")

    def _load_and_validate_packets(self, bundle_dir: Path) -> dict[str, dict[str, Any]]:
        """Load all 16 PRIMARY packets and cross-check IDs and historical fields."""
        packets_dir = bundle_dir / "positive_control_packets"
        if not packets_dir.is_dir():
            raise DataValidationError("Review bundle missing positive_control_packets/ directory")

        expected_qids = set(APPROVED_POSITIVE_CONTROLS_MATRIX.keys())
        packet_files = list(packets_dir.glob("*.json"))

        if len(packet_files) != 16:
            raise DataValidationError(
                f"Expected exactly 16 primary packet JSON files, got {len(packet_files)}"
            )

        packets_by_qid: dict[str, dict[str, Any]] = {}
        for pf in packet_files:
            qid = pf.stem
            if qid not in expected_qids:
                raise DataValidationError(f"Unexpected packet question_id '{qid}' in review bundle")
            data = json.loads(pf.read_text(encoding="utf-8"))
            if data.get("question_id") != qid:
                raise DataValidationError(f"Packet filename {pf.name} does not match question_id {data.get('question_id')}")
            packets_by_qid[qid] = data

        missing = expected_qids - set(packets_by_qid.keys())
        if missing:
            raise DataValidationError(f"Missing required primary packets: {sorted(missing)}")

        return packets_by_qid

    def _build_labels_overlay(
        self,
        *,
        packets_by_qid: dict[str, dict[str, Any]],
        archive_filename: str,
        archive_sha: str,
        archive_size: int,
    ) -> dict[str, Any]:
        """Construct deterministic human-approved positive-control label overlay."""
        questions_payload: dict[str, Any] = {}
        label_counter = Counter()
        total_claims_count = 0

        # Deterministic QID order
        for qid in sorted(APPROVED_POSITIVE_CONTROLS_MATRIX.keys()):
            packet = packets_by_qid[qid]
            approved_claims = APPROVED_POSITIVE_CONTROLS_MATRIX[qid]

            hist_arm = packet.get("historical_arm", {})
            hist_cv = hist_arm.get("historical_verification", {})
            raw_claims = hist_cv.get("claim_verifications", [])
            raw_claims_by_id = {c["claim_id"]: c for c in raw_claims}

            control_meta = packet.get("control_metadata", {})
            stratum = control_meta.get("stratum")
            selection_key = control_meta.get("selection_key")
            pool_type = control_meta.get("pool_type")
            hist_stop_reason = hist_arm.get("agent_outcome", {}).get("stop_reason")

            if pool_type != "primary":
                raise DataValidationError(f"Packet {qid} has pool_type '{pool_type}', expected 'primary'")

            if set(approved_claims.keys()) != set(raw_claims_by_id.keys()):
                raise DataValidationError(
                    f"Claim IDs mismatch for QID {qid}: approved={sorted(approved_claims.keys())}, "
                    f"raw={sorted(raw_claims_by_id.keys())}"
                )

            frozen_claims: dict[str, Any] = {}
            for cid in sorted(approved_claims.keys()):
                appr = approved_claims[cid]
                raw_c = raw_claims_by_id[cid]
                ctext = raw_c.get("claim_text", "")
                c_sha = sha256_text(ctext)

                label_counter[appr.entailment_label.value] += 1
                total_claims_count += 1

                claim_dict: dict[str, Any] = {
                    "claim_id": cid,
                    "claim_text_sha256": c_sha,
                    "claim_text": ctext,
                    "entailment_label": appr.entailment_label.value,
                    "error_tags": [tag.value for tag in appr.error_tags],
                    "diagnostic_metadata": {
                        "status": "diagnostic_not_ground_truth",
                        "historical_rule_status": raw_c.get("status"),
                        "historical_evidence_ids": raw_c.get("evidence_ids", []),
                    },
                }
                if appr.diagnostic_note:
                    claim_dict["diagnostic_note"] = appr.diagnostic_note

                frozen_claims[cid] = claim_dict

            questions_payload[qid] = {
                "question_id": qid,
                "stratum": stratum,
                "selection_key": selection_key,
                "pool_type": pool_type,
                "historical_stop_reason": hist_stop_reason,
                "claim_review_applicable": True,
                "claims": frozen_claims,
            }

        # Validate aggregate distribution invariants
        if total_claims_count != 27:
            raise DataValidationError(f"Expected 27 total labeled claims, got {total_claims_count}")
        if label_counter[HumanEntailmentLabel.SUPPORTED.value] != 16:
            raise DataValidationError(
                f"Expected 16 SUPPORTED claims, got {label_counter[HumanEntailmentLabel.SUPPORTED.value]}"
            )
        if label_counter[HumanEntailmentLabel.CONTRADICTED.value] != 2:
            raise DataValidationError(
                f"Expected 2 CONTRADICTED claims, got {label_counter[HumanEntailmentLabel.CONTRADICTED.value]}"
            )
        if label_counter[HumanEntailmentLabel.INSUFFICIENT.value] != 9:
            raise DataValidationError(
                f"Expected 9 INSUFFICIENT claims, got {label_counter[HumanEntailmentLabel.INSUFFICIENT.value]}"
            )

        return {
            "schema_version": "1.0",
            "artifact_type": "verification_positive_control_human_labels",
            "label_version": "v1",
            "verdict": "POSITIVE_CONTROL_HUMAN_LABELS_FROZEN",
            "source_review_package": {
                "filename": archive_filename,
                "sha256": archive_sha,
                "size_bytes": archive_size,
            },
            "source_selection_identity": {
                "phase_a_evidence_zip_sha256": CANONICAL_PHASE_A_ZIP_SHA256,
                "phase_a_results_sha256": CANONICAL_PHASE_A_RESULTS_SHA256,
                "sampling_algorithm": "deterministic_sha256_stratified_v1",
                "selection_salt_prefix": "verification-positive-control-v1|",
                "primary_candidate_count": 16,
                "reserve_candidate_count": 8,
            },
            "approval": {
                "approval_kind": CANONICAL_APPROVAL_KIND,
                "approval_date": CANONICAL_APPROVAL_DATE,
                "reviewer_id": CANONICAL_REVIEWER_ID,
                "organizer_ground_truth": False,
                "legal_expert_credential_asserted": False,
                "approval_statement": CANONICAL_APPROVAL_STATEMENT,
            },
            "usage_policy": {
                "allowed_initial_uses": [
                    "verification_correctness_evaluation",
                    "forensic_analysis",
                ],
                "prohibited_initial_uses": [
                    "training",
                    "fine_tuning",
                    "retrieval_relevance_supervision",
                    "public_test_annotation",
                    "private_test_annotation",
                    "manual_submission_correction",
                ],
            },
            "questions": questions_payload,
            "aggregate": {
                "question_count": len(questions_payload),
                "historical_arm_count": len(questions_payload),
                "labeled_claim_count": total_claims_count,
                "supported": label_counter[HumanEntailmentLabel.SUPPORTED.value],
                "contradicted": label_counter[HumanEntailmentLabel.CONTRADICTED.value],
                "insufficient": label_counter[HumanEntailmentLabel.INSUFFICIENT.value],
            },
            "combined_benchmark_summary": {
                "suspicious_forensic_claims": 11,
                "suspicious_forensic_supported": 2,
                "suspicious_forensic_contradicted": 5,
                "suspicious_forensic_insufficient": 4,
                "positive_control_claims": 27,
                "positive_control_supported": 16,
                "positive_control_contradicted": 2,
                "positive_control_insufficient": 9,
                "total_combined_benchmark_claims": 38,
                "total_combined_supported": 18,
                "total_combined_contradicted": 7,
                "total_combined_insufficient": 13,
                "prevalence_disclaimer": (
                    "This combined benchmark is an intentionally stratified evaluation set "
                    "containing deliberate forensic failure cases. It does NOT estimate "
                    "production system error prevalence."
                ),
            },
        }

    def _write_json_artifact(self, data: dict[str, Any], path: Path) -> None:
        """Write deterministic formatted UTF-8 JSON artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        path.write_text(serialized, encoding="utf-8")

    def _write_transport_zip(self, *, json_path: Path, zip_path: Path) -> None:
        """Package external transport ZIP containing human label overlay and manifest."""
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema_version": "1.0",
            "artifact_type": "verification_positive_control_human_labels_package",
            "label_file": json_path.name,
            "label_file_sha256": sha256_file(json_path),
            "approval_date": CANONICAL_APPROVAL_DATE,
            "reviewer_id": CANONICAL_REVIEWER_ID,
        }

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(json_path, arcname=json_path.name)
            z.writestr(
                "control_human_label_identity.json",
                json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Freeze human-approved positive-control labels v1."
    )
    parser.add_argument(
        "--review-packets",
        type=Path,
        required=True,
        help="Path to verification-positive-control-review-packets-v1.zip or unpacked directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to output external verification-positive-control-human-labels-v1.json.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help="Optional path to output external transport verification-positive-control-human-labels-v1.zip.",
    )

    args = parser.parse_args()

    freezer = PositiveControlLabelFreezer(
        review_packets_path=args.review_packets,
        output_json_path=args.output_json,
        output_zip_path=args.output_zip,
    )

    overlay = freezer.run()
    print(json.dumps(overlay, ensure_ascii=False, indent=2))
    print(f"\nFrozen positive-control human label JSON written to: {args.output_json}")
    if args.output_zip:
        print(f"Packaged transport ZIP written to: {args.output_zip}")


if __name__ == "__main__":
    main()
