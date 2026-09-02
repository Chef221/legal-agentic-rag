"""Legal reference extraction and deterministic resolution for M54 Preprocessing V2."""

from __future__ import annotations

import collections
import re
from typing import Any

from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    LegalProvisionV2,
    LegalReferenceResolutionV2,
    LegalReferenceTargetV2,
    LegalReferenceV2,
    TextSpanV2,
)

# Reference Extraction Patterns
_AMENDS_PATTERN = re.compile(
    r"(?:sửa đổi|bổ sung|sua doi|bo sung)(?:,?\s*(?:sửa đổi|bổ sung|sua doi|bo sung))*\s+(?:một số điều của|khoản\s+\d+|điều\s+\d+)?\s*(?:Luật|Nghị định|Thông tư|Quyết định|Pháp lệnh)\s+(?:số|so)?\s*([0-9]+(?:/[0-9]+)?/[A-ZĐ0-9_\-]+(?:\.[A-ZĐ0-9_\-]+)*|[A-ZĐ0-9_\-]+/[A-ZĐ0-9_\-]+)",
    re.IGNORECASE,
)

_REPEALS_PATTERN = re.compile(
    r"(?:bãi bỏ|hủy bỏ|bai bo|huy bo)\s+(?:toàn bộ|một phần|khoản\s+\d+|điều\s+\d+)?\s*(?:Luật|Nghị định|Thông tư|Quyết định|Pháp lệnh)\s+(?:số|so)?\s*([0-9]+(?:/[0-9]+)?/[A-ZĐ0-9_\-]+(?:\.[A-ZĐ0-9_\-]+)*|[A-ZĐ0-9_\-]+/[A-ZĐ0-9_\-]+)",
    re.IGNORECASE,
)

_REPLACES_PATTERN = re.compile(
    r"(?:thay thế|thay the)\s+(?:Luật|Nghị định|Thông tư|Quyết định|Pháp lệnh)\s+(?:số|so)?\s*([0-9]+(?:/[0-9]+)?/[A-ZĐ0-9_\-]+(?:\.[A-ZĐ0-9_\-]+)*|[A-ZĐ0-9_\-]+/[A-ZĐ0-9_\-]+)",
    re.IGNORECASE,
)

_CITES_PATTERN = re.compile(
    r"(?:căn cứ|theo quy định tại|áp dụng|chiếu theo|can cu)\s+(?:khoản\s+\d+|điều\s+\d+)?\s*(?:Luật|Nghị định|Thông tư|Quyết định|Pháp lệnh)?\s*(?:số|so)?\s*([0-9]+(?:/[0-9]+)?/[A-ZĐ0-9_\-]+(?:\.[A-ZĐ0-9_\-]+)*|[A-ZĐ0-9_\-]+/[A-ZĐ0-9_\-]+)",
    re.IGNORECASE,
)

_EFFECTIVITY_PATTERN = re.compile(
    r"(?:có hiệu lực|co hieu luc|hiệu lực thi hành|hieu luc thi hanh)\s+(?:kể từ|từ)?\s*ngày\s+([0-9]{1,2})\s+tháng\s+([0-9]{1,2})\s+năm\s+([0-9]{4})",
    re.IGNORECASE,
)


def extract_and_resolve_references_v2(
    documents: list[dict[str, Any] | CanonicalDocumentV2],
    confirmed_index: dict[str, list[str]] | None = None,
    ambiguous_candidate_index: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    # Build indexes if not provided
    if confirmed_index is None:
        confirmed_index = collections.defaultdict(list)
        ambiguous_candidate_index = collections.defaultdict(list)
        for doc in documents:
            if isinstance(doc, CanonicalDocumentV2):
                did = doc.document_id
                d_ident = doc.identity.model_dump()
            else:
                did = doc["document_id"]
                d_ident = doc["identity"]

            d_status = d_ident.get("status")
            if d_status in ("EXPLICIT", "DERIVED_FROM_NAME") and d_ident.get("document_number"):
                norm_num = d_ident["document_number"].strip().upper()
                confirmed_index[norm_num].append(did)
            elif d_status == "AMBIGUOUS":
                for cand in d_ident.get("candidate_document_numbers") or []:
                    norm_cand = cand.strip().upper()
                    ambiguous_candidate_index[norm_cand].append(did)

    def resolve_target_number(target_norm: str) -> tuple[str, str | None, list[str]]:
        confirmed = confirmed_index.get(target_norm, [])
        ambiguous = ambiguous_candidate_index.get(target_norm, [])

        if len(confirmed) == 1 and len(ambiguous) == 0:
            return "RESOLVED_UNIQUE", confirmed[0], []
        elif len(confirmed) > 1 or (len(confirmed) >= 1 and len(ambiguous) >= 1) or (len(confirmed) == 0 and len(ambiguous) >= 1):
            all_cands = sorted(list(set(confirmed + ambiguous)))
            return "RESOLVED_AMBIGUOUS", None, all_cands
        else:
            return "UNRESOLVED", None, []

    all_references: list[dict[str, Any]] = []

    for d in documents:
        if isinstance(d, CanonicalDocumentV2):
            doc_id = d.document_id
            auth_text = d.authority_text
        else:
            doc_id = d["document_id"]
            auth_text = d["authority_text"]

        ref_idx = 0

        # Structural relation matchers
        for pattern, rel_type in (
            (_AMENDS_PATTERN, "AMENDS"),
            (_REPEALS_PATTERN, "REPEALS"),
            (_REPLACES_PATTERN, "REPLACES"),
            (_CITES_PATTERN, "CITES"),
        ):
            for m in pattern.finditer(auth_text):
                ref_idx += 1
                raw_num = m.group(1).strip(" /.,:;")
                norm_num = raw_num.upper()

                status, target_id, candidates = resolve_target_number(norm_num)

                all_references.append({
                    "schema_version": "m54-preprocessing-v2.1",
                    "reference_id": f"{doc_id}::ref:{ref_idx}",
                    "source_document_id": doc_id,
                    "source_provision_id": None,
                    "source_span": {"start": m.start(), "end": m.end()},
                    "evidence_text": m.group(0),
                    "relation_type": rel_type,
                    "target": {
                        "document_number_raw": raw_num,
                        "document_number_normalized": norm_num,
                        "article_label": None,
                        "clause_label": None,
                        "point_label": None,
                    },
                    "resolution": {
                        "status": status,
                        "target_document_id": target_id,
                        "candidate_document_ids": candidates,
                    },
                    "extraction_rule": f"{rel_type}_REGEX_V1",
                })

        # Effectivity matcher
        for m in _EFFECTIVITY_PATTERN.finditer(auth_text):
            ref_idx += 1
            all_references.append({
                "schema_version": "m54-preprocessing-v2.1",
                "reference_id": f"{doc_id}::ref:{ref_idx}",
                "source_document_id": doc_id,
                "source_provision_id": None,
                "source_span": {"start": m.start(), "end": m.end()},
                "evidence_text": m.group(0),
                "relation_type": "EFFECTIVITY",
                "target": {
                    "document_number_raw": None,
                    "document_number_normalized": None,
                    "article_label": None,
                    "clause_label": None,
                    "point_label": None,
                },
                "resolution": {
                    "status": "UNRESOLVED",
                    "target_document_id": None,
                    "candidate_document_ids": [],
                },
                "extraction_rule": "EFFECTIVITY_REGEX_V1",
            })

    return all_references


def extract_legal_references(
    documents: list[CanonicalDocumentV2] | list[dict[str, Any]],
    provisions: list[LegalProvisionV2] | list[dict[str, Any]] | None = None,
) -> list[LegalReferenceV2]:
    """Typed wrapper returning Pydantic model instances."""
    ref_dicts = extract_and_resolve_references_v2(documents)
    return [
        LegalReferenceV2(
            schema_version=r["schema_version"],
            reference_id=r["reference_id"],
            source_document_id=r["source_document_id"],
            source_provision_id=r["source_provision_id"],
            source_span=TextSpanV2(**r["source_span"]),
            evidence_text=r["evidence_text"],
            relation_type=r["relation_type"],
            target=LegalReferenceTargetV2(**r["target"]),
            resolution=LegalReferenceResolutionV2(**r["resolution"]),
            extraction_rule=r["extraction_rule"],
        )
        for r in ref_dicts
    ]
