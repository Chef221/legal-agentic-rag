"""Retrieval unit materialization for M54 Preprocessing V2."""

from __future__ import annotations

import collections
from typing import Any

from legal_agentic_rag.offline.chunking.tokenizer import UnicodeWordTokenizer
from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    HeadingPathItemV2,
    LegalProvisionV2,
    PreprocessingV2SegmentationProfile,
    RetrievalUnitDocumentIdentityV2,
    RetrievalUnitHierarchyV2,
    RetrievalUnitV2,
    TextSpanV2,
)


def build_retrieval_text(
    doc_title: str | None,
    doc_number: str | None,
    instrument_type: str | None,
    heading_path: list[dict[str, Any]] | list[HeadingPathItemV2],
    article_label: str | None,
    clause_label: str | None,
    point_label: str | None,
    authority_text: str,
) -> str:
    header_lines = []

    if doc_title:
        header_lines.append(f"Văn bản: {doc_title}")
    # Omit Số ký hiệu if doc_number is null (Residual Blocker B3)
    if doc_number:
        header_lines.append(f"Số ký hiệu: {doc_number}")
    if instrument_type and not (doc_title and instrument_type.lower() in doc_title.lower()):
        header_lines.append(f"Loại văn bản: {instrument_type}")

    for h in heading_path:
        h_type = h.type if isinstance(h, HeadingPathItemV2) else h.get("type")
        h_label = h.label if isinstance(h, HeadingPathItemV2) else h.get("label")
        h_title = h.title if isinstance(h, HeadingPathItemV2) else h.get("title")
        t_str = f" {h_title}" if h_title else ""
        if h_type == "APPENDIX":
            header_lines.append(f"Phụ lục {h_label}{t_str}")
        elif h_type == "PART":
            header_lines.append(f"Phần {h_label}{t_str}")
        elif h_type == "CHAPTER":
            header_lines.append(f"Chương {h_label}{t_str}")
        elif h_type == "SECTION":
            header_lines.append(f"Mục {h_label}{t_str}")
        elif h_type == "SUBSECTION":
            header_lines.append(f"Tiểu mục {h_label}{t_str}")

    if article_label:
        header_lines.append(f"Điều {article_label}")
    if clause_label:
        header_lines.append(f"Khoản {clause_label}")
    if point_label:
        header_lines.append(f"Điểm {point_label}")

    if header_lines:
        prefix = "\n".join(header_lines) + "\n---\n"
    else:
        prefix = ""

    return prefix + authority_text


def materialize_retrieval_units_v2(
    documents: list[dict[str, Any] | CanonicalDocumentV2],
    provisions: list[dict[str, Any] | LegalProvisionV2],
    max_tokens: int | PreprocessingV2SegmentationProfile = 512,
) -> list[dict[str, Any]]:
    max_tok_val = max_tokens.max_tokens if isinstance(max_tokens, PreprocessingV2SegmentationProfile) else max_tokens

    # 1. Load document identity metadata map
    doc_identity_map = {}
    for d in documents:
        if isinstance(d, CanonicalDocumentV2):
            did = d.document_id
            d_ident = d.identity.model_dump()
        else:
            did = d["document_id"]
            d_ident = d["identity"]

        d_status = d_ident.get("status")
        d_docnum = d_ident.get("document_number") if d_status in ("EXPLICIT", "DERIVED_FROM_NAME") else None
        doc_identity_map[did] = {
            "title": d_ident.get("title"),
            "document_number": d_docnum,
            "instrument_type": d_ident.get("instrument_type"),
            "status": d_status,
        }

    # 2. Group provisions by document and analyze hierarchy tree
    provisions_by_doc = collections.defaultdict(list)
    for p in provisions:
        if isinstance(p, LegalProvisionV2):
            p_dict = p.model_dump()
        else:
            p_dict = p
        provisions_by_doc[p_dict["document_id"]].append(p_dict)

    tokenizer = UnicodeWordTokenizer()
    retrieval_units: list[dict[str, Any]] = []

    for doc_id, doc_provs in provisions_by_doc.items():
        doc_ident = doc_identity_map.get(doc_id, {})
        d_title = doc_ident.get("title")
        d_num = doc_ident.get("document_number")
        d_type = doc_ident.get("instrument_type")

        # Build parent -> children map for this document
        children_by_parent = collections.defaultdict(list)
        for p in doc_provs:
            if p.get("parent_provision_id"):
                children_by_parent[p["parent_provision_id"]].append(p)

        # Process each provision
        for p in doc_provs:
            p_id = p["provision_id"]
            p_type = p["provision_type"]
            p_auth = p["authority_text"]
            p_start = p["authority_span"]["start"]

            children = children_by_parent.get(p_id, [])

            if children:
                # Parent provision with children (Article containing Clauses, or Clause containing Points)
                # D1 Check: Does the parent have unique preamble content before its first child?
                first_child = min(children, key=lambda c: c["authority_span"]["start"])
                first_child_start = first_child["authority_span"]["start"]

                rel_preamble_end = first_child_start - p_start
                if rel_preamble_end > 0:
                    preamble_text = p_auth[:rel_preamble_end]
                    if preamble_text.strip():
                        strategy = f"{p_type}_PREAMBLE"
                        ret_text = build_retrieval_text(
                            d_title, d_num, d_type,
                            p["heading_path"],
                            p["article_label"], p["clause_label"], p["point_label"],
                            preamble_text,
                        )
                        token_count_auth = tokenizer.count(preamble_text)
                        token_count_ret = tokenizer.count(ret_text)

                        ru_obj = {
                            "schema_version": "m54-preprocessing-v2.1",
                            "retrieval_unit_id": f"{p_id}::preamble",
                            "document_id": doc_id,
                            "provision_id": p_id,
                            "segment_index": 1,
                            "segment_count": 1,
                            "authority_span_in_provision": {"start": 0, "end": len(preamble_text)},
                            "authority_text": preamble_text,
                            "retrieval_text": ret_text,
                            "document_identity": {
                                "document_number": d_num,
                                "title": d_title,
                            },
                            "hierarchy": {
                                "article_label": p["article_label"],
                                "clause_label": p["clause_label"],
                                "point_label": p["point_label"],
                                "heading_path": p["heading_path"],
                            },
                            "strategy": strategy,
                            "token_count_authority": token_count_auth,
                            "token_count_retrieval": token_count_ret,
                            "quality_flags": list(p.get("quality_flags") or []) + ["PARENT_PREAMBLE_UNIT"],
                        }
                        retrieval_units.append(ru_obj)
                continue

            # Leaf provision: materialize whole provision or controlled segments
            token_count_auth = tokenizer.count(p_auth)

            if token_count_auth <= max_tok_val:
                # Whole provision
                strategy = "WHOLE_PROVISION" if p_type != "DOCUMENT_FALLBACK" else "DOCUMENT_FALLBACK"
                ret_text = build_retrieval_text(
                    d_title, d_num, d_type,
                    p["heading_path"],
                    p["article_label"], p["clause_label"], p["point_label"],
                    p_auth,
                )
                token_count_ret = tokenizer.count(ret_text)

                ru_obj = {
                    "schema_version": "m54-preprocessing-v2.1",
                    "retrieval_unit_id": p_id,
                    "document_id": doc_id,
                    "provision_id": p_id,
                    "segment_index": 1,
                    "segment_count": 1,
                    "authority_span_in_provision": {"start": 0, "end": len(p_auth)},
                    "authority_text": p_auth,
                    "retrieval_text": ret_text,
                    "document_identity": {
                        "document_number": d_num,
                        "title": d_title,
                    },
                    "hierarchy": {
                        "article_label": p["article_label"],
                        "clause_label": p["clause_label"],
                        "point_label": p["point_label"],
                        "heading_path": p["heading_path"],
                    },
                    "strategy": strategy,
                    "token_count_authority": token_count_auth,
                    "token_count_retrieval": token_count_ret,
                    "quality_flags": list(p.get("quality_flags") or []),
                }
                retrieval_units.append(ru_obj)
            else:
                # Oversized provision -> Exact Controlled Segmentation (D2)
                seg_texts = tokenizer.split(p_auth, max_tokens=max_tok_val, overlap_tokens=0)
                n_segs = len(seg_texts)
                char_cursor = 0
                for seg_idx, s_text in enumerate(seg_texts, start=1):
                    s_find = p_auth.find(s_text, char_cursor)
                    if s_find >= 0:
                        s_start = s_find
                        s_end = s_start + len(s_text)
                        char_cursor = s_end
                    else:
                        s_start = char_cursor
                        s_end = s_start + len(s_text)
                        char_cursor = s_end

                    assert p_auth[s_start:s_end] == s_text, f"Segment slice mismatch in {p_id} seg {seg_idx}"

                    seg_ru_id = f"{p_id}::seg:{seg_idx}of{n_segs}"
                    ret_text = build_retrieval_text(
                        d_title, d_num, d_type,
                        p["heading_path"],
                        p["article_label"], p["clause_label"], p["point_label"],
                        s_text,
                    )

                    token_auth = tokenizer.count(s_text)
                    token_ret = tokenizer.count(ret_text)

                    ru_obj = {
                        "schema_version": "m54-preprocessing-v2.1",
                        "retrieval_unit_id": seg_ru_id,
                        "document_id": doc_id,
                        "provision_id": p_id,
                        "segment_index": seg_idx,
                        "segment_count": n_segs,
                        "authority_span_in_provision": {"start": s_start, "end": s_end},
                        "authority_text": s_text,
                        "retrieval_text": ret_text,
                        "document_identity": {
                            "document_number": d_num,
                            "title": d_title,
                        },
                        "hierarchy": {
                            "article_label": p["article_label"],
                            "clause_label": p["clause_label"],
                            "point_label": p["point_label"],
                            "heading_path": p["heading_path"],
                        },
                        "strategy": "CONTROLLED_SEGMENT",
                        "token_count_authority": token_auth,
                        "token_count_retrieval": token_ret,
                        "quality_flags": list(p.get("quality_flags") or []) + ["OVERSIZED_PROVISION_SEGMENT"],
                    }
                    retrieval_units.append(ru_obj)

    return retrieval_units


def materialize_retrieval_units(
    documents: list[CanonicalDocumentV2] | list[dict[str, Any]],
    provisions: list[LegalProvisionV2] | list[dict[str, Any]],
    segmentation_profile: PreprocessingV2SegmentationProfile | int = 512,
) -> list[RetrievalUnitV2]:
    """Typed wrapper returning Pydantic model instances."""
    ru_dicts = materialize_retrieval_units_v2(documents, provisions, segmentation_profile)
    return [
        RetrievalUnitV2(
            schema_version=r["schema_version"],
            retrieval_unit_id=r["retrieval_unit_id"],
            document_id=r["document_id"],
            provision_id=r["provision_id"],
            segment_index=r["segment_index"],
            segment_count=r["segment_count"],
            authority_span_in_provision=TextSpanV2(**r["authority_span_in_provision"]),
            authority_text=r["authority_text"],
            retrieval_text=r["retrieval_text"],
            document_identity=RetrievalUnitDocumentIdentityV2(**r["document_identity"]),
            hierarchy=RetrievalUnitHierarchyV2(
                article_label=r["hierarchy"]["article_label"],
                clause_label=r["hierarchy"]["clause_label"],
                point_label=r["hierarchy"]["point_label"],
                heading_path=[
                    HeadingPathItemV2(type=h["type"], label=h["label"], title=h.get("title"))
                    for h in r["hierarchy"]["heading_path"]
                ],
            ),
            strategy=r["strategy"],
            token_count_authority=r["token_count_authority"],
            token_count_retrieval=r["token_count_retrieval"],
            quality_flags=r["quality_flags"],
        )
        for r in ru_dicts
    ]
