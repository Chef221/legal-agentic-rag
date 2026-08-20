#!/usr/bin/env python3
"""Freeze human-approved forensic labels for verification-correctness audit (B-FORENSIC-1A).

This script creates an immutable, content-bound overlay of human-approved
claim entailment labels and error tags over frozen B-FORENSIC-0 review packets.

Invariants:
- Read-only forensic evaluation labels
- Not organizer ground truth
- Not retrieval relevance supervision
- Not training or fine-tuning data
- Zero retrieval, generation, or verifier execution
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from typing import Any
import zipfile

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError

_LOGGER = logging.getLogger(__name__)

CANONICAL_REVIEW_ZIP_SHA256 = (
    "996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a"
)
CANONICAL_REVIEW_ZIP_SIZE = 42826
CANONICAL_REVIEW_ZIP_FILENAME = "verification-forensic-review-packets.zip"

CANONICAL_TARGET_IDS = ["102047", "147239", "26541", "95861"]
CANONICAL_ARM_NAMES = ["BASE", "CANDIDATE"]


class ClaimEntailmentLabel(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class ForensicErrorTag(StrEnum):
    CONDITION_OMITTED = "CONDITION_OMITTED"
    CONDITION_INVERTED = "CONDITION_INVERTED"
    EXCEPTION_IGNORED = "EXCEPTION_IGNORED"
    ACTOR_ROLE_INVERTED = "ACTOR_ROLE_INVERTED"
    NEGATION_INVERTED = "NEGATION_INVERTED"
    QUANTITY_ERROR = "QUANTITY_ERROR"
    SCOPE_OVERGENERALIZED = "SCOPE_OVERGENERALIZED"
    WRONG_DOCUMENT = "WRONG_DOCUMENT"
    WRONG_ARTICLE = "WRONG_ARTICLE"
    TEMPORAL_APPLICABILITY_ERROR = "TEMPORAL_APPLICABILITY_ERROR"
    OTHER = "OTHER"
    NONE = "NONE"


@dataclass(frozen=True)
class ApprovedClaimSpec:
    label: ClaimEntailmentLabel
    error_tags: tuple[ForensicErrorTag, ...]


# Strict authoritative human specification
APPROVED_HUMAN_SPEC: dict[str, dict[str, dict[str, Any]]] = {
    "102047": {
        "BASE": {
            "stop_reason": "answer_verified",
            "claim_review_applicable": True,
            "claims": {
                "C1": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.CONTRADICTED,
                    error_tags=(
                        ForensicErrorTag.CONDITION_INVERTED,
                        ForensicErrorTag.SCOPE_OVERGENERALIZED,
                    ),
                )
            },
        },
        "CANDIDATE": {
            "stop_reason": "answer_verified",
            "claim_review_applicable": True,
            "claims": {
                "C1": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.CONTRADICTED,
                    error_tags=(
                        ForensicErrorTag.CONDITION_OMITTED,
                        ForensicErrorTag.SCOPE_OVERGENERALIZED,
                    ),
                )
            },
        },
    },
    "147239": {
        "BASE": {
            "stop_reason": "generation_failed",
            "claim_review_applicable": False,
            "reason": "historical_generation_failed_no_verified_claim",
            "claims": {},
        },
        "CANDIDATE": {
            "stop_reason": "answer_verified",
            "claim_review_applicable": True,
            "claims": {
                "C1": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.SUPPORTED,
                    error_tags=(ForensicErrorTag.NONE,),
                ),
                "C2": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.CONTRADICTED,
                    error_tags=(ForensicErrorTag.ACTOR_ROLE_INVERTED,),
                ),
            },
        },
    },
    "26541": {
        "BASE": {
            "stop_reason": "answer_verified",
            "claim_review_applicable": True,
            "claims": {
                "C1": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.INSUFFICIENT,
                    error_tags=(
                        ForensicErrorTag.WRONG_DOCUMENT,
                        ForensicErrorTag.WRONG_ARTICLE,
                    ),
                )
            },
        },
        "CANDIDATE": {
            "stop_reason": "generation_failed",
            "claim_review_applicable": False,
            "reason": "historical_generation_failed_no_verified_claim",
            "claims": {},
        },
    },
    "95861": {
        "BASE": {
            "stop_reason": "answer_verified",
            "claim_review_applicable": True,
            "claims": {
                "C1": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.CONTRADICTED,
                    error_tags=(
                        ForensicErrorTag.ACTOR_ROLE_INVERTED,
                        ForensicErrorTag.WRONG_DOCUMENT,
                    ),
                ),
                "C2": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.INSUFFICIENT,
                    error_tags=(ForensicErrorTag.WRONG_DOCUMENT,),
                ),
                "C3": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.INSUFFICIENT,
                    error_tags=(ForensicErrorTag.WRONG_DOCUMENT,),
                ),
            },
        },
        "CANDIDATE": {
            "stop_reason": "answer_verified",
            "claim_review_applicable": True,
            "claims": {
                "C1": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.CONTRADICTED,
                    error_tags=(
                        ForensicErrorTag.ACTOR_ROLE_INVERTED,
                        ForensicErrorTag.WRONG_DOCUMENT,
                    ),
                ),
                "C2": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.INSUFFICIENT,
                    error_tags=(
                        ForensicErrorTag.ACTOR_ROLE_INVERTED,
                        ForensicErrorTag.WRONG_DOCUMENT,
                    ),
                ),
                "C3": ApprovedClaimSpec(
                    label=ClaimEntailmentLabel.SUPPORTED,
                    error_tags=(ForensicErrorTag.NONE,),
                ),
            },
        },
    },
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of a UTF-8 string."""
    return sha256(text.encode("utf-8")).hexdigest()


class ForensicLabelFreezer:
    """Validate review packets and generate immutable human forensic labels overlay."""

    def __init__(
        self,
        *,
        review_packets_path: Path,
        output_path: Path,
        approved_spec: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._review_packets_path = review_packets_path.resolve()
        self._output_path = output_path.resolve()
        self._approved_spec = approved_spec or APPROVED_HUMAN_SPEC

    def run(self) -> dict[str, Any]:
        """Execute validation and generate frozen human labels artifact."""
        if not self._review_packets_path.exists():
            raise DataValidationError(
                f"Review packets archive does not exist: {self._review_packets_path}"
            )

        actual_zip_sha = sha256_file(self._review_packets_path)
        if actual_zip_sha != CANONICAL_REVIEW_ZIP_SHA256:
            raise DataValidationError(
                f"Review packets ZIP SHA mismatch: expected {CANONICAL_REVIEW_ZIP_SHA256}, got {actual_zip_sha}"
            )

        with zipfile.ZipFile(self._review_packets_path, "r") as z:
            names = set(z.namelist())

            # Verify required members
            for qid in CANONICAL_TARGET_IDS:
                pkt_name = f"forensic_packets/{qid}.json"
                if pkt_name not in names:
                    raise DataValidationError(f"Missing forensic packet '{pkt_name}' in review archive")

            # Load and validate all 4 forensic packets
            packets: dict[str, dict[str, Any]] = {}
            for qid in CANONICAL_TARGET_IDS:
                raw_bytes = z.read(f"forensic_packets/{qid}.json")
                pkt_data = json.loads(raw_bytes.decode("utf-8"))
                if pkt_data.get("question_id") != qid:
                    raise DataValidationError(
                        f"Packet question_id mismatch: expected '{qid}', got '{pkt_data.get('question_id')}'"
                    )
                packets[qid] = pkt_data

        # Build human labels structure
        questions_output: dict[str, Any] = {}
        labeled_claim_count = 0
        supported_count = 0
        contradicted_count = 0
        insufficient_count = 0
        generation_failed_unlabeled_arms = 0

        for qid, arm_specs in self._approved_spec.items():
            if qid not in packets:
                raise DataValidationError(f"Question '{qid}' in approved spec not found in review packets")
            pkt = packets[qid]
            pkt_arms = pkt.get("arms", {})

            q_arms_output: dict[str, Any] = {}

            for arm_name in CANONICAL_ARM_NAMES:
                if arm_name not in arm_specs:
                    raise DataValidationError(f"Arm '{arm_name}' missing from approved spec for question '{qid}'")
                if arm_name not in pkt_arms:
                    raise DataValidationError(f"Arm '{arm_name}' missing from packet for question '{qid}'")

                spec = arm_specs[arm_name]
                arm_data = pkt_arms[arm_name]
                actual_stop_reason = arm_data.get("agent_outcome", {}).get("stop_reason")
                expected_stop_reason = spec.get("stop_reason")

                if actual_stop_reason != expected_stop_reason:
                    raise DataValidationError(
                        f"Stop reason mismatch for {qid} {arm_name}: "
                        f"expected '{expected_stop_reason}', got '{actual_stop_reason}'"
                    )

                if not spec.get("claim_review_applicable"):
                    generation_failed_unlabeled_arms += 1
                    # Ensure no claims in spec
                    if spec.get("claims"):
                        raise DataValidationError(
                            f"Generation-failed arm {qid} {arm_name} cannot receive approved claims"
                        )
                    q_arms_output[arm_name] = {
                        "historical_stop_reason": actual_stop_reason,
                        "claim_review_applicable": False,
                        "reason": spec.get("reason", "historical_generation_failed_no_verified_claim"),
                        "claims": {},
                    }
                    continue

                # Extract claims from replay result or historical verification
                replay_claims = (
                    arm_data.get("rule_verifier_replay", {})
                    .get("replay_result", {})
                    .get("claim_verifications", [])
                )
                if not replay_claims:
                    replay_claims = (
                        arm_data.get("historical_verification", {})
                        .get("claim_verifications", [])
                    )

                claims_by_id = {c["claim_id"]: c for c in replay_claims if "claim_id" in c}
                approved_claims_spec: dict[str, ApprovedClaimSpec] = spec.get("claims", {})

                # Check all approved claim IDs exist in packet
                for cid, c_spec in approved_claims_spec.items():
                    if cid not in claims_by_id:
                        raise DataValidationError(
                            f"Approved claim '{cid}' for {qid} {arm_name} not found in packet claims: {list(claims_by_id.keys())}"
                        )

                # Check packet has no unapproved claims
                if set(claims_by_id.keys()) != set(approved_claims_spec.keys()):
                    raise DataValidationError(
                        f"Claim ID mismatch for {qid} {arm_name}: "
                        f"packet has {sorted(claims_by_id.keys())}, approved spec has {sorted(approved_claims_spec.keys())}"
                    )

                arm_claims_output: dict[str, Any] = {}
                for cid, c_spec in approved_claims_spec.items():
                    claim_obj = claims_by_id[cid]
                    claim_text = str(claim_obj.get("claim_text", ""))
                    if not claim_text:
                        raise DataValidationError(f"Claim '{cid}' in {qid} {arm_name} has empty claim_text")

                    c_text_sha = sha256_text(claim_text)
                    label_str = c_spec.label.value
                    error_tags_list = [t.value for t in c_spec.error_tags]

                    labeled_claim_count += 1
                    if c_spec.label == ClaimEntailmentLabel.SUPPORTED:
                        supported_count += 1
                    elif c_spec.label == ClaimEntailmentLabel.CONTRADICTED:
                        contradicted_count += 1
                    elif c_spec.label == ClaimEntailmentLabel.INSUFFICIENT:
                        insufficient_count += 1

                    arm_claims_output[cid] = {
                        "claim_id": cid,
                        "claim_text_sha256": c_text_sha,
                        "claim_text": claim_text,
                        "entailment_label": label_str,
                        "error_tags": error_tags_list,
                        "diagnostic_metadata": {
                            "status": "diagnostic_not_ground_truth",
                            "historical_rule_status": claim_obj.get("status"),
                            "historical_evidence_ids": claim_obj.get("evidence_ids", []),
                        },
                    }

                q_arms_output[arm_name] = {
                    "historical_stop_reason": actual_stop_reason,
                    "claim_review_applicable": True,
                    "claims": arm_claims_output,
                }

            questions_output[qid] = {
                "question_id": qid,
                "arms": q_arms_output,
            }

        # Assert strict expected aggregate counts
        if (
            labeled_claim_count != 11
            or supported_count != 2
            or contradicted_count != 5
            or insufficient_count != 4
            or generation_failed_unlabeled_arms != 2
        ):
            raise DataValidationError(
                f"Aggregate label counts mismatch: total={labeled_claim_count} (expected 11), "
                f"supported={supported_count} (2), contradicted={contradicted_count} (5), "
                f"insufficient={insufficient_count} (4), gen_failed_arms={generation_failed_unlabeled_arms} (2)"
            )

        artifact = {
            "schema_version": "1.0",
            "artifact_type": "verification_human_forensic_labels",
            "label_version": "v1",
            "verdict": "HUMAN_FORENSIC_LABELS_FROZEN",
            "source_review_package": {
                "filename": CANONICAL_REVIEW_ZIP_FILENAME,
                "sha256": actual_zip_sha,
                "size_bytes": self._review_packets_path.stat().st_size,
            },
            "approval": {
                "approval_kind": "explicit_user_human_approval",
                "approval_date": "2026-08-20",
                "reviewer_id": "human_reviewer_1",
                "organizer_ground_truth": False,
                "legal_expert_credential_asserted": False,
                "approval_statement": (
                    "These labels are internal human forensic annotations over frozen "
                    "train-derived development outputs and supplied frozen evidence. "
                    "They are not official UIT DSC relevance or legal-answer ground-truth labels."
                ),
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
            "questions": questions_output,
            "aggregate": {
                "question_count": len(questions_output),
                "historical_arm_count": len(questions_output) * 2,
                "labeled_claim_count": labeled_claim_count,
                "supported": supported_count,
                "contradicted": contradicted_count,
                "insufficient": insufficient_count,
                "generation_failed_unlabeled_arms": generation_failed_unlabeled_arms,
            },
        }

        # Write output JSON
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        return artifact


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Freeze human-approved verification forensic labels overlay."
    )
    parser.add_argument(
        "--review-packets",
        type=Path,
        required=True,
        help="Path to verification-forensic-review-packets.zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for verification-human-forensic-labels-v1.json.",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Optional output ZIP path for packaging the frozen label artifact.",
    )

    args = parser.parse_args()

    freezer = ForensicLabelFreezer(
        review_packets_path=args.review_packets,
        output_path=args.output,
    )

    artifact = freezer.run()
    print(json.dumps(artifact, ensure_ascii=False, indent=2))

    if args.package_zip:
        args.package_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.package_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(args.output, arcname=args.output.name)
            # Add identity metadata
            ident = {
                "artifact_type": "verification_human_forensic_labels",
                "label_version": "v1",
                "label_json_sha256": sha256_file(args.output),
                "label_json_filename": args.output.name,
                "source_review_zip_sha256": CANONICAL_REVIEW_ZIP_SHA256,
                "created_at": datetime.now(UTC).isoformat(),
            }
            z.writestr("human_label_identity.json", json.dumps(ident, indent=2) + "\n")
        print(f"\nPackaged human label ZIP written to: {args.package_zip}")


if __name__ == "__main__":
    main()
