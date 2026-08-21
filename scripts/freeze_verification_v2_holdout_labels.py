#!/usr/bin/env python3
"""Freeze human-reviewed gold labels for V2-D3 Fresh Holdout Benchmark (Phase H-LABEL).

This script binds human-reviewed claim entailment labels to exact holdout review
packets and selection artifacts, validates complete set equality across all claims,
and generates an immutable gold-label artifact alongside a content-free commitment record.

Key Invariants:
- Human Gold Labels only: strictly internal evaluation labels, not organizer ground truth.
- Content-Bound: Binds each claim label to UTF-8 claim_text SHA-256 and source packets SHA-256.
- Fail-Closed: Rejects missing, extra, duplicate, or invalid claim labels.
- Permitted Labels: Exactly 'SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT'.
- Content-Free Commitment: Produces a commitment record with artifact checksums, counts,
  and review status, with zero question text, claim text, or question IDs.
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
from typing import Any
import zipfile

from legal_agentic_rag.exceptions import DataValidationError

_LOGGER = logging.getLogger(__name__)

CANONICAL_HOLDOUT_SELECTION_SHA256 = (
    "08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b"
)
CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256 = (
    "a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4"
)
CANONICAL_REVIEW_PROTOCOL_VERSION = "1.0"
GOVERNANCE_STATUS_FROZEN_PENDING_REVIEW = "FROZEN_PENDING_EXTERNAL_REVIEW"
GOVERNANCE_STATUS_EXTERNALLY_REVIEWED = "EXTERNALLY_REVIEWED_FOR_H_EXEC"


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object pairs hook that raises DataValidationError on duplicate object keys."""
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for k, v in pairs:
        if k in seen:
            raise DataValidationError(f"Duplicate JSON key detected in review input: '{k}'")
        seen.add(k)
        result[k] = v
    return result


class HoldoutEntailmentLabel(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class PacketClaimSpec:
    question_id: str
    arm_id: str
    claim_id: str
    claim_text: str
    claim_text_sha256: str
    stratum: str


@dataclass(frozen=True)
class ReviewedClaimSpec:
    question_id: str
    arm_id: str
    claim_id: str
    entailment_label: HoldoutEntailmentLabel
    claim_text_sha256: str | None = None
    error_tags: tuple[str, ...] = ()
    diagnostic_note: str | None = None


def extract_packet_claims(packets_path: Path) -> tuple[dict[tuple[str, str, str], PacketClaimSpec], int, int]:
    """Extract all claims from holdout review packets ZIP without exposing sensitive content."""
    packet_claims: dict[tuple[str, str, str], PacketClaimSpec] = {}
    question_count = 0
    arm_count = 0

    with zipfile.ZipFile(packets_path, "r") as zf:
        json_members = [
            m for m in zf.namelist()
            if m.endswith(".json")
            and not m.startswith("__MACOSX")
            and ("holdout_packets/" in m or "packets/" in m)
        ]
        if not json_members:
            json_members = [
                m for m in zf.namelist()
                if m.endswith(".json") and not m.startswith("__MACOSX") and "/" not in m
            ]
        for member in sorted(json_members):
            question_count += 1
            pkt = json.loads(zf.read(member).decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
            qid = str(pkt.get("question_id") or Path(member).stem)
            stratum = pkt.get("stratum") or pkt.get("holdout_metadata", {}).get("stratum", "UNKNOWN")

            pkt_arms = pkt.get("arms") or pkt.get("historical_arms", {})
            if not pkt_arms and "historical_arm" in pkt:
                pkt_arms = {"PRIMARY": pkt["historical_arm"]}
            for arm_id, arm_data in pkt_arms.items():
                arm_count += 1
                raw_claims = (
                    arm_data.get("claims")
                    or arm_data.get("historical_verification", {}).get("claim_verifications", [])
                )
                for rc in raw_claims:
                    cid = rc.get("claim_id", "")
                    ctext = rc.get("claim_text", "")
                    if not cid:
                        raise DataValidationError(f"Missing claim_id in packet {member} arm {arm_id}")
                    
                    key = (qid, str(arm_id), str(cid))
                    if key in packet_claims:
                        raise DataValidationError(f"Duplicate claim key {key} in review packets archive")

                    packet_claims[key] = PacketClaimSpec(
                        question_id=qid,
                        arm_id=str(arm_id),
                        claim_id=str(cid),
                        claim_text=ctext,
                        claim_text_sha256=sha256_text(ctext),
                        stratum=stratum,
                    )

    return packet_claims, question_count, arm_count


def parse_human_reviewed_input(input_path: Path) -> dict[tuple[str, str, str], ReviewedClaimSpec]:
    """Parse raw human review input file into a validated mapping with strict duplicate rejection."""
    raw_text = input_path.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text, object_pairs_hook=_reject_duplicate_json_keys)
    reviewed_claims: dict[tuple[str, str, str], ReviewedClaimSpec] = {}

    if "questions" in raw_data:
        questions_dict = raw_data["questions"]
        if not isinstance(questions_dict, dict):
            raise DataValidationError("Review input 'questions' field must be a JSON object")
        for qid, q_data in questions_dict.items():
            arms_dict = q_data.get("arms", {})
            if not isinstance(arms_dict, dict):
                raise DataValidationError(f"Arms for question {qid} must be a JSON object")
            for arm_id, arm_data in arms_dict.items():
                claims_dict = arm_data.get("claims", {})
                if isinstance(claims_dict, dict):
                    for cid, c_data in claims_dict.items():
                        key = (str(qid), str(arm_id), str(cid))
                        if key in reviewed_claims:
                            raise DataValidationError(
                                f"HOLD_OUT_LABEL_DUPLICATE: Duplicate review item for claim key ({qid}, {arm_id}, {cid})"
                            )
                        reviewed_claims[key] = _parse_single_claim_review(qid, arm_id, cid, c_data)
                elif isinstance(claims_dict, list):
                    for c_data in claims_dict:
                        cid = c_data.get("claim_id")
                        if not cid:
                            raise DataValidationError(f"Missing claim_id in review item for question {qid}")
                        key = (str(qid), str(arm_id), str(cid))
                        if key in reviewed_claims:
                            raise DataValidationError(
                                f"HOLD_OUT_LABEL_DUPLICATE: Duplicate review item for claim key ({qid}, {arm_id}, {cid})"
                            )
                        reviewed_claims[key] = _parse_single_claim_review(qid, arm_id, cid, c_data)
                else:
                    raise DataValidationError(f"Claims for question {qid} arm {arm_id} must be object or list")
    elif isinstance(raw_data, list):
        for item in raw_data:
            qid = str(item.get("question_id", ""))
            arm_id = str(item.get("arm_id", ""))
            cid = str(item.get("claim_id", ""))
            if not qid or not arm_id or not cid:
                raise DataValidationError(f"Invalid review list item: missing question_id, arm_id, or claim_id: {item}")
            key = (qid, arm_id, cid)
            if key in reviewed_claims:
                raise DataValidationError(
                    f"HOLD_OUT_LABEL_DUPLICATE: Duplicate review item for claim key ({qid}, {arm_id}, {cid})"
                )
            reviewed_claims[key] = _parse_single_claim_review(qid, arm_id, cid, item)
    else:
        raise DataValidationError(f"Unrecognized review input format in {input_path}")

    return reviewed_claims


def _parse_single_claim_review(qid: Any, arm_id: Any, cid: Any, data: dict[str, Any]) -> ReviewedClaimSpec:
    lbl_raw = data.get("entailment_label")
    if not lbl_raw:
        raise DataValidationError(f"Missing entailment_label for claim ({qid}, {arm_id}, {cid})")
    
    lbl_str = str(lbl_raw).strip().upper()
    if lbl_str not in HoldoutEntailmentLabel.__members__:
        raise DataValidationError(
            f"Invalid entailment_label '{lbl_raw}' for claim ({qid}, {arm_id}, {cid}). "
            f"Must be one of {list(HoldoutEntailmentLabel.__members__.keys())}"
        )
    
    error_tags = tuple(str(t) for t in data.get("error_tags", []))
    diag_note = data.get("diagnostic_note")
    claim_sha = data.get("claim_text_sha256")

    return ReviewedClaimSpec(
        question_id=str(qid),
        arm_id=str(arm_id),
        claim_id=str(cid),
        entailment_label=HoldoutEntailmentLabel(lbl_str),
        claim_text_sha256=claim_sha,
        error_tags=error_tags,
        diagnostic_note=diag_note,
    )


def freeze_holdout_labels(
    *,
    holdout_packets_path: Path,
    holdout_selection_path: Path,
    reviewed_input_path: Path,
    output_labels_path: Path,
    commitment_output_path: Path | None = None,
    bypass_source_checksums: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze human gold labels and generate immutable artifact and commitment."""
    if not holdout_packets_path.is_file():
        raise DataValidationError(f"Holdout packets archive missing: {holdout_packets_path}")
    if not holdout_selection_path.is_file():
        raise DataValidationError(f"Holdout selection file missing: {holdout_selection_path}")
    if not reviewed_input_path.is_file():
        raise DataValidationError(f"Reviewed input file missing: {reviewed_input_path}")

    packets_sha = sha256_file(holdout_packets_path)
    selection_sha = sha256_file(holdout_selection_path)

    if not bypass_source_checksums:
        if packets_sha != CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256:
            raise DataValidationError(
                f"Holdout review packets SHA mismatch: expected {CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256}, got {packets_sha}"
            )
        if selection_sha != CANONICAL_HOLDOUT_SELECTION_SHA256:
            raise DataValidationError(
                f"Holdout selection SHA mismatch: expected {CANONICAL_HOLDOUT_SELECTION_SHA256}, got {selection_sha}"
            )

    # 1. Extract packet claims
    packet_claims, question_count, arm_count = extract_packet_claims(holdout_packets_path)
    
    # 2. Parse human review input
    reviewed_claims = parse_human_reviewed_input(reviewed_input_path)

    # 3. Assert Exact Set Equality
    packet_keys = set(packet_claims.keys())
    reviewed_keys = set(reviewed_claims.keys())

    missing_in_review = packet_keys - reviewed_keys
    extra_in_review = reviewed_keys - packet_keys

    if missing_in_review:
        raise DataValidationError(
            f"HOLD_OUT_LABEL_FREEZE_ERROR: {len(missing_in_review)} packet claims missing in human review input."
        )
    if extra_in_review:
        raise DataValidationError(
            f"HOLD_OUT_LABEL_FREEZE_ERROR: {len(extra_in_review)} extra claims in human review input not present in packets."
        )

    # 4. Validate and Bind Claim Text SHA
    structured_questions: dict[str, dict[str, Any]] = {}
    class_counts: Counter[str] = Counter()

    for key, pkt_claim in sorted(packet_claims.items()):
        rev = reviewed_claims[key]
        qid, arm_id, cid = key

        # Verify claim text SHA if present in review input
        if rev.claim_text_sha256 and rev.claim_text_sha256 != pkt_claim.claim_text_sha256:
            raise DataValidationError(
                f"HOLD_OUT_LABEL_FREEZE_ERROR: Claim text SHA mismatch for claim key ({qid}, {arm_id}, {cid}). "
                f"Packet SHA: {pkt_claim.claim_text_sha256}, Review SHA: {rev.claim_text_sha256}"
            )

        class_counts[rev.entailment_label.value] += 1

        if qid not in structured_questions:
            structured_questions[qid] = {"arms": {}}
        if arm_id not in structured_questions[qid]["arms"]:
            structured_questions[qid]["arms"][arm_id] = {"claims": {}}

        claim_entry: dict[str, Any] = {
            "claim_text_sha256": pkt_claim.claim_text_sha256,
            "entailment_label": rev.entailment_label.value,
            "error_tags": list(rev.error_tags),
        }
        if rev.diagnostic_note:
            claim_entry["diagnostic_note"] = rev.diagnostic_note

        structured_questions[qid]["arms"][arm_id]["claims"][cid] = claim_entry

    total_claims = len(packet_claims)
    now_iso = datetime.now(UTC).isoformat()

    # 5. Build authoritative labels artifact payload
    labels_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "verification_v2_holdout_reviewed_labels",
        "review_status": "frozen_human_reviewed",
        "review_protocol_version": CANONICAL_REVIEW_PROTOCOL_VERSION,
        "created_at": now_iso,
        "holdout_packets_sha256": packets_sha,
        "holdout_selection_sha256": selection_sha,
        "total_questions": question_count,
        "total_arms": arm_count,
        "total_claims": total_claims,
        "class_counts": {
            HoldoutEntailmentLabel.SUPPORTED.value: class_counts[HoldoutEntailmentLabel.SUPPORTED.value],
            HoldoutEntailmentLabel.CONTRADICTED.value: class_counts[HoldoutEntailmentLabel.CONTRADICTED.value],
            HoldoutEntailmentLabel.INSUFFICIENT.value: class_counts[HoldoutEntailmentLabel.INSUFFICIENT.value],
        },
        "questions": structured_questions,
    }

    # Write labels file
    output_labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_json_str = json.dumps(labels_payload, indent=2, ensure_ascii=False) + "\n"
    output_labels_path.write_text(labels_json_str, encoding="utf-8")
    
    labels_sha = sha256_file(output_labels_path)
    labels_size = output_labels_path.stat().st_size

    # 6. Build content-free commitment payload
    # Note: Initial governance status is strictly FROZEN_PENDING_EXTERNAL_REVIEW.
    # The label freeze script MUST NOT self-authorize Phase H-EXEC.
    commitment_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "verification_v2_holdout_label_commitment",
        "artifact_filename": output_labels_path.name,
        "labels_sha256": labels_sha,
        "labels_size_bytes": labels_size,
        "total_questions": question_count,
        "total_arms": arm_count,
        "total_claims": total_claims,
        "class_counts": {
            HoldoutEntailmentLabel.SUPPORTED.value: class_counts[HoldoutEntailmentLabel.SUPPORTED.value],
            HoldoutEntailmentLabel.CONTRADICTED.value: class_counts[HoldoutEntailmentLabel.CONTRADICTED.value],
            HoldoutEntailmentLabel.INSUFFICIENT.value: class_counts[HoldoutEntailmentLabel.INSUFFICIENT.value],
        },
        "review_status": "frozen_human_reviewed",
        "holdout_packets_sha256": packets_sha,
        "holdout_selection_sha256": selection_sha,
        "review_timestamp": now_iso,
        "reviewer_governance_status": GOVERNANCE_STATUS_FROZEN_PENDING_REVIEW,
    }

    if commitment_output_path:
        commitment_output_path.parent.mkdir(parents=True, exist_ok=True)
        commitment_output_path.write_text(
            json.dumps(commitment_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    _LOGGER.info(
        "Successfully froze %d holdout claim labels into %s (SHA: %s)",
        total_claims,
        output_labels_path,
        labels_sha,
    )
    return labels_payload, commitment_payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for holdout label freezing."""
    parser = argparse.ArgumentParser(
        description="Freeze human-reviewed gold labels for V2-D3 Fresh Holdout Benchmark.",
    )
    parser.add_argument(
        "--holdout-packets",
        type=Path,
        required=True,
        help="Path to holdout review packets ZIP archive.",
    )
    parser.add_argument(
        "--holdout-selection",
        type=Path,
        required=True,
        help="Path to holdout selection JSON commitment file.",
    )
    parser.add_argument(
        "--reviewed-input",
        type=Path,
        required=True,
        help="Path to human-reviewed labels input JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for frozen verification-v2-holdout-reviewed-labels-v1.json.",
    )
    parser.add_argument(
        "--commitment-output",
        type=Path,
        default=None,
        help="Optional destination path for content-free label commitment JSON.",
    )
    parser.add_argument(
        "--bypass-source-checksums",
        action="store_true",
        help="Bypass canonical source checksum verification (synthetic testing only).",
    )
    return parser


def main() -> int:
    """Main CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        freeze_holdout_labels(
            holdout_packets_path=args.holdout_packets,
            holdout_selection_path=args.holdout_selection,
            reviewed_input_path=args.reviewed_input,
            output_labels_path=args.output,
            commitment_output_path=args.commitment_output,
            bypass_source_checksums=args.bypass_source_checksums,
        )
        return 0
    except DataValidationError as exc:
        _LOGGER.error("Data validation error during label freeze: %s", exc)
        return 1
    except Exception as exc:
        _LOGGER.exception("Unexpected error during label freeze: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
