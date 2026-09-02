"""Parser module for M54 Preprocessing V2 hierarchy extraction (Exact Audited M45 Grammar)."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import re
from typing import Any

from legal_agentic_rag.schemas.preprocessing_v2 import (
    HeadingPathItemV2,
    LegalProvisionV2,
    TextSpanV2,
    UnrecognizedMarkerV2,
)

# Exact Audited M45 Marker Grammar
_NUMBER = r"(?:\d+[A-ZĐ]?|[IVXLCDM]+)"
_CLAUSE_NUMBER = r"\d+[A-ZĐ]?"
_DELIMITER = r"[.:\-–—]"

_PART_PATTERN = re.compile(
    rf"^PHẦN(?:\s+THỨ)?\s+(?P<number>[^\s.:\-–—]+)\s*(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_CHAPTER_PATTERN = re.compile(
    rf"^CHƯƠNG\s+(?P<number>{_NUMBER})\s*(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_SECTION_PATTERN = re.compile(
    rf"^MỤC\s+(?P<number>{_NUMBER})\s*(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_SUBSECTION_PATTERN = re.compile(
    rf"^TIỂU\s+MỤC\s+(?P<number>{_NUMBER})\s*(?:(?:{_DELIMITER})\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_ARTICLE_PATTERN = re.compile(
    rf"^ĐIỀU\s+(?P<number>{_NUMBER})(?!\w)\s*(?:(?:{_DELIMITER})\s*)?(?P<title>.*)$",
    re.IGNORECASE,
)
_EXPLICIT_CLAUSE_PATTERN = re.compile(
    rf"^KHOẢN\s+(?P<number>{_CLAUSE_NUMBER})\s*(?:(?:{_DELIMITER})\s*)?(?P<body>.*)$",
    re.IGNORECASE,
)
_CLAUSE_PATTERN = re.compile(
    rf"^(?P<number>{_CLAUSE_NUMBER})(?P<delimiter>[.)])\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_EXPLICIT_POINT_PATTERN = re.compile(
    rf"^ĐIỂM\s+(?P<number>[A-ZĐ])\s*(?:(?:[.):\-–—])\s*)?(?P<body>.*)$",
    re.IGNORECASE,
)
_POINT_PATTERN = re.compile(
    r"^(?P<number>[A-ZĐ])[.)]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_APPENDIX_PATTERN = re.compile(
    rf"^PHỤ\s+LỤC(?:\s+(?P<number>{_NUMBER}))?\s*(?:(?:{_DELIMITER})\s*)?(?P<title>.*)$",
    re.IGNORECASE,
)
_POTENTIAL_MARKER_PATTERN = re.compile(
    r"^(?:PHẦN|CHƯƠNG|TIỂU\s+MỤC|MỤC|ĐIỀU|KHOẢN|ĐIỂM|PHỤ\s+LỤC)\s+(?:\d|[IVXLCDM]+\b|[A-ZĐ][.):])",
    re.IGNORECASE,
)
_BARE_MARKER_PATTERN = re.compile(
    r"^(?:PHẦN|CHƯƠNG|TIỂU\s+MỤC|MỤC|ĐIỀU|KHOẢN|ĐIỂM|PHỤ\s+LỤC)$",
    re.IGNORECASE,
)

MAX_TITLE_CHARS = 200
MAX_TITLE_WORDS = 30


def is_table_row(line: str) -> bool:
    return " | " in line


def looks_like_title(line: str) -> bool:
    if not line or len(line) > MAX_TITLE_CHARS:
        return False
    if len(line.split()) > MAX_TITLE_WORDS:
        return False
    if is_table_row(line):
        return False
    if _BARE_MARKER_PATTERN.fullmatch(line):
        return False
    if line.endswith((".", ";", ":")):
        return False
    if any(
        pattern.match(line)
        for pattern in (
            _PART_PATTERN, _CHAPTER_PATTERN, _SUBSECTION_PATTERN,
            _SECTION_PATTERN, _ARTICLE_PATTERN, _APPENDIX_PATTERN,
            _EXPLICIT_CLAUSE_PATTERN, _CLAUSE_PATTERN,
            _EXPLICIT_POINT_PATTERN, _POINT_PATTERN,
        )
    ):
        return False
    return True


def is_valid_implicit_clause(match: re.Match[str]) -> bool:
    body = match.group("body").strip()
    if not body:
        return False
    return not (match.group("delimiter") == "." and body[0].isdigit())


@dataclass
class MarkerToken:
    block_type: str   # PART, CHAPTER, SECTION, SUBSECTION, ARTICLE, CLAUSE, POINT, APPENDIX
    level: int        # 1: PART/APPENDIX, 2: CHAPTER, 3: SECTION, 4: SUBSECTION, 5: ARTICLE, 6: CLAUSE, 7: POINT
    number: str | None
    label: str
    title: str | None
    rule_id: str
    line_start_idx: int
    line_count: int
    start_char: int
    header_end_char: int


def parse_document_structure_v2(doc_id: str, authority_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Parse document structure returning exact dictionary outputs matching shadow build."""
    if not authority_text.strip():
        # Empty document
        return [], [], False

    # Extract non-empty lines and calculate exact character spans in authority_text
    lines: list[str] = []
    line_spans: list[tuple[int, int]] = []

    cursor = 0
    for raw_line in authority_text.splitlines():
        s_line = raw_line.strip()
        if not s_line:
            continue
        idx = authority_text.find(s_line, cursor)
        if idx >= 0:
            start_pos = idx
            end_pos = idx + len(s_line)
            cursor = end_pos
        else:
            start_pos = cursor
            end_pos = cursor + len(s_line)
            cursor = end_pos
        lines.append(s_line)
        line_spans.append((start_pos, end_pos))

    unrecognized_markers: list[dict[str, Any]] = []
    markers: list[MarkerToken] = []

    active_article_present = False

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        start_char = line_spans[line_idx][0]
        end_char = line_spans[line_idx][1]

        if is_table_row(line):
            line_idx += 1
            continue

        # Check bare marker joining
        if _BARE_MARKER_PATTERN.fullmatch(line) and line_idx + 1 < len(lines):
            test_joined = f"{line} {lines[line_idx + 1]}"
            m_found = False
            for pat, btype, lvl, pfx, rid in (
                (_PART_PATTERN, "PART", 1, "Phần", "M45_PART_PATTERN"),
                (_CHAPTER_PATTERN, "CHAPTER", 2, "Chương", "M45_CHAPTER_PATTERN"),
                (_SUBSECTION_PATTERN, "SUBSECTION", 4, "Tiểu mục", "M45_SUBSECTION_PATTERN"),
                (_SECTION_PATTERN, "SECTION", 3, "Mục", "M45_SECTION_PATTERN"),
                (_ARTICLE_PATTERN, "ARTICLE", 5, "Điều", "M45_ARTICLE_PATTERN"),
                (_APPENDIX_PATTERN, "APPENDIX", 1, "Phụ lục", "M45_APPENDIX_PATTERN"),
            ):
                m_match = pat.match(test_joined)
                if m_match:
                    num = m_match.groupdict().get("number")
                    num_clean = num.strip() if num else None
                    ttl = m_match.groupdict().get("title")
                    ttl_clean = ttl.strip() if ttl else None
                    lbl = pfx if num_clean is None else f"{pfx} {num_clean}"

                    header_end_char = line_spans[line_idx + 1][1]
                    markers.append(MarkerToken(
                        block_type=btype, level=lvl, number=num_clean, label=lbl,
                        title=ttl_clean, rule_id=rid, line_start_idx=line_idx, line_count=2,
                        start_char=start_char, header_end_char=header_end_char
                    ))
                    if btype == "ARTICLE":
                        active_article_present = True
                    elif lvl <= 4:
                        active_article_present = False
                    m_found = True
                    line_idx += 2
                    break
            if m_found:
                continue

        # Regular classification
        matched_marker: MarkerToken | None = None

        # 1. Structural headings (level 1-5)
        for pat, btype, lvl, pfx, rid in (
            (_PART_PATTERN, "PART", 1, "Phần", "M45_PART_PATTERN"),
            (_CHAPTER_PATTERN, "CHAPTER", 2, "Chương", "M45_CHAPTER_PATTERN"),
            (_SUBSECTION_PATTERN, "SUBSECTION", 4, "Tiểu mục", "M45_SUBSECTION_PATTERN"),
            (_SECTION_PATTERN, "SECTION", 3, "Mục", "M45_SECTION_PATTERN"),
            (_ARTICLE_PATTERN, "ARTICLE", 5, "Điều", "M45_ARTICLE_PATTERN"),
            (_APPENDIX_PATTERN, "APPENDIX", 1, "Phụ lục", "M45_APPENDIX_PATTERN"),
        ):
            m_match = pat.match(line)
            if m_match:
                num = m_match.groupdict().get("number")
                num_clean = num.strip() if num else None
                ttl = m_match.groupdict().get("title")
                ttl_clean = ttl.strip() if ttl else None
                lbl = pfx if num_clean is None else f"{pfx} {num_clean}"

                curr_line_count = 1
                hdr_end = end_char
                if ttl_clean is None and line_idx + 1 < len(lines) and looks_like_title(lines[line_idx + 1]):
                    ttl_clean = lines[line_idx + 1].strip()
                    curr_line_count = 2
                    hdr_end = line_spans[line_idx + 1][1]

                matched_marker = MarkerToken(
                    block_type=btype, level=lvl, number=num_clean, label=lbl,
                    title=ttl_clean, rule_id=rid, line_start_idx=line_idx, line_count=curr_line_count,
                    start_char=start_char, header_end_char=hdr_end
                )
                if btype == "ARTICLE":
                    active_article_present = True
                elif lvl <= 4:
                    active_article_present = False
                break

        # 2. Provisions under active Article (level 6: Clause, level 7: Point)
        if matched_marker is None and active_article_present:
            # Explicit Clause
            m_exp_cl = _EXPLICIT_CLAUSE_PATTERN.match(line)
            if m_exp_cl:
                num = m_exp_cl.groupdict().get("number")
                num_clean = num.strip() if num else None
                matched_marker = MarkerToken(
                    block_type="CLAUSE", level=6, number=num_clean, label=f"Khoản {num_clean}",
                    title=None, rule_id="M45_EXPLICIT_CLAUSE_PATTERN", line_start_idx=line_idx, line_count=1,
                    start_char=start_char, header_end_char=end_char
                )
            else:
                # Implicit Clause
                m_imp_cl = _CLAUSE_PATTERN.match(line)
                if m_imp_cl and is_valid_implicit_clause(m_imp_cl):
                    num = m_imp_cl.groupdict().get("number")
                    num_clean = num.strip() if num else None
                    matched_marker = MarkerToken(
                        block_type="CLAUSE", level=6, number=num_clean, label=f"Khoản {num_clean}",
                        title=None, rule_id="M45_IMPLICIT_CLAUSE_PATTERN", line_start_idx=line_idx, line_count=1,
                        start_char=start_char, header_end_char=end_char
                    )
                else:
                    # Explicit Point
                    m_exp_pt = _EXPLICIT_POINT_PATTERN.match(line)
                    if m_exp_pt:
                        num = m_exp_pt.groupdict().get("number")
                        num_clean = num.strip() if num else None
                        matched_marker = MarkerToken(
                            block_type="POINT", level=7, number=num_clean, label=f"Điểm {num_clean}",
                            title=None, rule_id="M45_EXPLICIT_POINT_PATTERN", line_start_idx=line_idx, line_count=1,
                            start_char=start_char, header_end_char=end_char
                        )
                    else:
                        # Implicit Point
                        m_imp_pt = _POINT_PATTERN.match(line)
                        if m_imp_pt:
                            num = m_imp_pt.groupdict().get("number")
                            num_clean = num.strip() if num else None
                            matched_marker = MarkerToken(
                                block_type="POINT", level=7, number=num_clean, label=f"Điểm {num_clean}",
                                title=None, rule_id="M45_IMPLICIT_POINT_PATTERN", line_start_idx=line_idx, line_count=1,
                                start_char=start_char, header_end_char=end_char
                            )

        if matched_marker is not None:
            markers.append(matched_marker)
            line_idx += matched_marker.line_count
        else:
            if _POTENTIAL_MARKER_PATTERN.match(line):
                unrecognized_markers.append({
                    "document_id": doc_id,
                    "line_index": line_idx,
                    "line_text": line,
                    "character_span": {"start": start_char, "end": end_char}
                })
            line_idx += 1

    has_structure = any(m.block_type == "ARTICLE" for m in markers)

    if not has_structure:
        # Fallback Document: single provision spanning entire authority_text
        fb_provision = {
            "schema_version": "m54-preprocessing-v2.1",
            "provision_id": f"{doc_id}::doc_fallback",
            "document_id": doc_id,
            "canonical_path": "doc_fallback",
            "parent_provision_id": None,
            "provision_type": "DOCUMENT_FALLBACK",
            "article_label": None,
            "clause_label": None,
            "point_label": None,
            "heading_path": [],
            "authority_span": {"start": 0, "end": len(authority_text)},
            "header_span": {"start": 0, "end": 0},
            "raw_marker": None,
            "parse_status": "CONTROLLED_FALLBACK",
            "parse_rule": "DOCUMENT_FALLBACK",
            "authority_text": authority_text,
            "quality_flags": ["DOCUMENT_FALLBACK"]
        }
        return [fb_provision], unrecognized_markers, False

    # Build hierarchy provisions with exact Level-Based Spans (Blocker C & F)
    doc_len = len(authority_text)
    total_markers = len(markers)

    # Pre-calculate end_char for every marker based on hierarchy level
    marker_end_chars: list[int] = []
    for i, m in enumerate(markers):
        end_c = doc_len
        for j in range(i + 1, total_markers):
            mj = markers[j]
            if mj.level <= m.level:
                end_c = mj.start_char
                break
        marker_end_chars.append(end_c)

    provisions: list[dict[str, Any]] = []

    active_headings: dict[str, dict[str, Any] | None] = {
        "PART": None,
        "CHAPTER": None,
        "SECTION": None,
        "SUBSECTION": None,
        "APPENDIX": None
    }

    active_article_id: str | None = None
    active_article_label: str | None = None
    active_clause_id: str | None = None
    active_clause_label: str | None = None

    path_collision_counts: dict[str, int] = collections.defaultdict(int)

    for i, m in enumerate(markers):
        m_type = m.block_type
        m_num = m.number
        m_lbl = m.label
        m_ttl = m.title
        s_char = m.start_char
        e_char = marker_end_chars[i]
        hdr_end_char = m.header_end_char
        rule_id = m.rule_id

        # Track Top-Level Headings obeying Step 2 Appendix Reset Semantics (Blocker F)
        if m_type in ("PART", "CHAPTER", "SECTION", "SUBSECTION", "APPENDIX"):
            h_obj = {"type": m_type, "label": m_num or m_lbl, "title": m_ttl}
            if m_type == "APPENDIX":
                # Reset previous Part/Chapter/Section/Subsection
                active_headings["PART"] = None
                active_headings["CHAPTER"] = None
                active_headings["SECTION"] = None
                active_headings["SUBSECTION"] = None
                active_headings["APPENDIX"] = h_obj
            elif m_type == "PART":
                # Reset previous Appendix/Chapter/Section/Subsection
                active_headings["APPENDIX"] = None
                active_headings["PART"] = h_obj
                active_headings["CHAPTER"] = None
                active_headings["SECTION"] = None
                active_headings["SUBSECTION"] = None
            elif m_type == "CHAPTER":
                active_headings["CHAPTER"] = h_obj
                active_headings["SECTION"] = None
                active_headings["SUBSECTION"] = None
            elif m_type == "SECTION":
                active_headings["SECTION"] = h_obj
                active_headings["SUBSECTION"] = None
            elif m_type == "SUBSECTION":
                active_headings["SUBSECTION"] = h_obj
            continue

        # Active heading path snapshot (Blocker F)
        if active_headings["APPENDIX"] is not None:
            curr_heading_path = [h for h in [
                active_headings["APPENDIX"], active_headings["CHAPTER"],
                active_headings["SECTION"], active_headings["SUBSECTION"]
            ] if h is not None]
        else:
            curr_heading_path = [h for h in [
                active_headings["PART"], active_headings["CHAPTER"],
                active_headings["SECTION"], active_headings["SUBSECTION"]
            ] if h is not None]

        p_flags = []

        if m_type == "ARTICLE":
            art_label = m_num or "unlabeled"
            canonical_path = f"art:{art_label}"
            active_article_label = art_label
            active_clause_id = None
            active_clause_label = None

            path_collision_counts[canonical_path] += 1
            seq = path_collision_counts[canonical_path]
            prov_id_suffix = canonical_path if seq == 1 else f"{canonical_path}~{seq}"
            if seq > 1:
                p_flags.append("DUPLICATE_CANONICAL_PATH")

            prov_id = f"{doc_id}::{prov_id_suffix}"
            active_article_id = prov_id

            p_obj = {
                "schema_version": "m54-preprocessing-v2.1",
                "provision_id": prov_id,
                "document_id": doc_id,
                "canonical_path": canonical_path,
                "parent_provision_id": None,
                "provision_type": "ARTICLE",
                "article_label": art_label,
                "clause_label": None,
                "point_label": None,
                "heading_path": curr_heading_path,
                "authority_span": {"start": s_char, "end": e_char},
                "header_span": {"start": s_char, "end": hdr_end_char},
                "raw_marker": m_lbl,
                "parse_status": "EXPLICIT",
                "parse_rule": rule_id,
                "authority_text": authority_text[s_char:e_char],
                "quality_flags": p_flags
            }
            provisions.append(p_obj)

        elif m_type == "CLAUSE":
            cl_label = m_num or "unlabeled"
            canonical_path = f"art:{active_article_label}::cl:{cl_label}"
            active_clause_label = cl_label

            path_collision_counts[canonical_path] += 1
            seq = path_collision_counts[canonical_path]
            prov_id_suffix = canonical_path if seq == 1 else f"{canonical_path}~{seq}"
            if seq > 1:
                p_flags.append("DUPLICATE_CANONICAL_PATH")

            prov_id = f"{doc_id}::{prov_id_suffix}"
            active_clause_id = prov_id

            p_obj = {
                "schema_version": "m54-preprocessing-v2.1",
                "provision_id": prov_id,
                "document_id": doc_id,
                "canonical_path": canonical_path,
                "parent_provision_id": active_article_id,
                "provision_type": "CLAUSE",
                "article_label": active_article_label,
                "clause_label": cl_label,
                "point_label": None,
                "heading_path": curr_heading_path,
                "authority_span": {"start": s_char, "end": e_char},
                "header_span": {"start": s_char, "end": hdr_end_char},
                "raw_marker": m_lbl,
                "parse_status": "EXPLICIT",
                "parse_rule": rule_id,
                "authority_text": authority_text[s_char:e_char],
                "quality_flags": p_flags
            }
            provisions.append(p_obj)

        elif m_type == "POINT":
            pt_label = m_num or "unlabeled"
            cl_part = f"::cl:{active_clause_label}" if active_clause_label else ""
            canonical_path = f"art:{active_article_label}{cl_part}::pt:{pt_label}"

            path_collision_counts[canonical_path] += 1
            seq = path_collision_counts[canonical_path]
            prov_id_suffix = canonical_path if seq == 1 else f"{canonical_path}~{seq}"
            if seq > 1:
                p_flags.append("DUPLICATE_CANONICAL_PATH")

            prov_id = f"{doc_id}::{prov_id_suffix}"
            parent_id = active_clause_id or active_article_id

            p_obj = {
                "schema_version": "m54-preprocessing-v2.1",
                "provision_id": prov_id,
                "document_id": doc_id,
                "canonical_path": canonical_path,
                "parent_provision_id": parent_id,
                "provision_type": "POINT",
                "article_label": active_article_label,
                "clause_label": active_clause_label,
                "point_label": pt_label,
                "heading_path": curr_heading_path,
                "authority_span": {"start": s_char, "end": e_char},
                "header_span": {"start": s_char, "end": hdr_end_char},
                "raw_marker": m_lbl,
                "parse_status": "EXPLICIT",
                "parse_rule": rule_id,
                "authority_text": authority_text[s_char:e_char],
                "quality_flags": p_flags
            }
            provisions.append(p_obj)

    # Residual Blocker G2: Preserve actual uncovered text as deterministic fallback spans
    # Compute gaps in [0, doc_len]
    covered_spans = [p["authority_span"] for p in provisions]
    # Build union of covered intervals
    intervals = sorted([(s["start"], s["end"]) for s in covered_spans], key=lambda x: x[0])
    merged_intervals: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged_intervals or merged_intervals[-1][1] < start:
            merged_intervals.append((start, end))
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))

    gap_intervals: list[tuple[int, int]] = []
    prev_end = 0
    for s_int, e_int in merged_intervals:
        if s_int > prev_end:
            gap_intervals.append((prev_end, s_int))
        prev_end = max(prev_end, e_int)
    if prev_end < doc_len:
        gap_intervals.append((prev_end, doc_len))

    gap_fb_idx = 0
    for g_start, g_end in gap_intervals:
        gap_text = authority_text[g_start:g_end]
        if gap_text.strip():  # Contains non-whitespace text
            gap_fb_idx += 1
            fb_id = f"{doc_id}::doc_fallback:{gap_fb_idx}"
            fb_obj = {
                "schema_version": "m54-preprocessing-v2.1",
                "provision_id": fb_id,
                "document_id": doc_id,
                "canonical_path": f"doc_fallback:{gap_fb_idx}",
                "parent_provision_id": None,
                "provision_type": "DOCUMENT_FALLBACK",
                "article_label": None,
                "clause_label": None,
                "point_label": None,
                "heading_path": [],
                "authority_span": {"start": g_start, "end": g_end},
                "header_span": {"start": g_start, "end": g_start},
                "raw_marker": None,
                "parse_status": "CONTROLLED_FALLBACK",
                "parse_rule": "ORDINARY_BODY_FALLBACK",
                "authority_text": gap_text,
                "quality_flags": ["ORDINARY_BODY_FALLBACK"]
            }
            provisions.append(fb_obj)

    # Sort provisions by authority_span.start
    provisions.sort(key=lambda p: (p["authority_span"]["start"], p["authority_span"]["end"]))

    return provisions, unrecognized_markers, has_structure


def parse_provisions_from_document(
    doc_id: str,
    authority_text: str,
) -> tuple[list[LegalProvisionV2], list[UnrecognizedMarkerV2]]:
    """Typed wrapper returning Pydantic model instances."""
    prov_dicts, unrec_dicts, _ = parse_document_structure_v2(doc_id, authority_text)

    p_models = [
        LegalProvisionV2(
            schema_version=d.get("schema_version", "m54-preprocessing-v2.1"),
            provision_id=d["provision_id"],
            document_id=d["document_id"],
            canonical_path=d.get("canonical_path", ""),
            parent_provision_id=d.get("parent_provision_id"),
            provision_type=d["provision_type"],
            article_label=d.get("article_label"),
            clause_label=d.get("clause_label"),
            point_label=d.get("point_label"),
            heading_path=[
                HeadingPathItemV2(type=h["type"], label=h["label"], title=h.get("title"))
                for h in d.get("heading_path", [])
            ],
            authority_span=TextSpanV2(**d["authority_span"]),
            header_span=TextSpanV2(**d.get("header_span", {"start": 0, "end": 0})),
            raw_marker=d.get("raw_marker"),
            parse_status=d["parse_status"],
            parse_rule=d["parse_rule"],
            authority_text=d["authority_text"],
            quality_flags=d.get("quality_flags", []),
        )
        for d in prov_dicts
    ]

    u_models = [
        UnrecognizedMarkerV2(
            document_id=d["document_id"],
            line_index=d["line_index"],
            character_span=TextSpanV2(**d["character_span"]),
            line_text=d["line_text"],
        )
        for d in unrec_dicts
    ]

    return p_models, u_models
