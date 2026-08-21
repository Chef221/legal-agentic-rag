#!/usr/bin/env python3
"""DATA-CENTRIC PHASE D0: Deep Official Data Census & Legal Retrieval-Unit Audit.

Read-only, model-free, deterministic audit of:
1. Official raw corpus (selected-contexts.zip)
2. Canonical processed artifacts (v0400 serving root)
3. Legal structure parser coverage & unresolved cases
4. Legal chunking units, boundary risks, and search_text construction
5. Official train Q&A (train.json) style, taxonomy, and legal reference signals
6. High-confidence train -> legal source linkability
7. Data-loss lineage matrix and bottleneck ranking
8. Pre-registration of exactly ONE data-centric D1 experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import html
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import unicodedata
import zipfile
from typing import Any, Literal

from legal_agentic_rag import __version__
from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.competition.uit_dsc_2026.passage_cleaner import (
    UitDsc2026PassageCleaner,
    _TVPL_PRO_NOTICE,
    _KNOWN_MARKUP,
)
from legal_agentic_rag.offline.parsing.structure_parser import LegalStructureParser
from legal_agentic_rag.schemas.legal_documents import LegalDocument, LegalBlockType
from legal_agentic_rag.schemas.competition import CompetitionQuestion, CompetitionContext

_LOGGER = logging.getLogger("audit_official_data_d0")

# --- Percentile Utilities ---

def compute_percentiles(values: list[int | float]) -> dict[str, float]:
    """Compute deterministic distribution percentiles."""
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n
    var_val = sum((x - mean_val) ** 2 for x in sorted_vals) / n if n > 1 else 0.0
    std_val = math.sqrt(var_val)

    def _p(p: float) -> float:
        if n == 1:
            return float(sorted_vals[0])
        idx = p * (n - 1)
        low = int(math.floor(idx))
        high = int(math.ceil(idx))
        if low == high:
            return float(sorted_vals[low])
        weight = idx - low
        return float(sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight)

    return {
        "count": n,
        "min": float(sorted_vals[0]),
        "p25": _p(0.25),
        "p50": _p(0.50),
        "p75": _p(0.75),
        "p90": _p(0.90),
        "p95": _p(0.95),
        "p99": _p(0.99),
        "max": float(sorted_vals[-1]),
        "mean": float(mean_val),
        "std": float(std_val),
    }


def compute_file_sha256(path: Path) -> str:
    """Stream file SHA-256 calculation."""
    h = sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def normalize_text_for_dup(text: str) -> str:
    """Normalize text for non-semantic duplicate detection."""
    norm = unicodedata.normalize("NFC", text)
    norm = norm.replace("\r\n", "\n").replace("\r", "\n")
    norm = re.sub(r"[ \t]+", " ", norm)
    norm = re.sub(r"\n\s+", "\n", norm)
    return norm.strip().casefold()


# --- Section 2 & 3: Source & Artifact Discovery ---

def audit_sources_and_artifacts(
    train_path: Path,
    public_path: Path,
    contexts_path: Path,
    serving_root: Path,
) -> dict[str, Any]:
    """Audit source files and serving artifact manifests."""
    loader = UitDsc2026DataLoader()
    
    # 1. Raw Sources
    sources_info: dict[str, Any] = {}
    for name, p in [
        ("official_train", train_path),
        ("official_public", public_path),
        ("official_selected_contexts", contexts_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required official source not found: {p}")
        size = p.stat().st_size
        sha = compute_file_sha256(p)
        info: dict[str, Any] = {
            "path": str(p),
            "size_bytes": size,
            "sha256": sha,
        }
        if p.suffix.casefold() == ".json":
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            info["record_count"] = len(data)
            first_key = next(iter(data.keys())) if data else None
            info["schema_keys"] = list(data[first_key].keys()) if first_key else []
        elif p.suffix.casefold() == ".zip":
            ctx_id = loader.inspect_context_source(p)
            info["record_count"] = ctx_id.member_count
            info["canonical_revision"] = ctx_id.revision
            with zipfile.ZipFile(p) as z:
                members = [m.filename for m in z.infolist() if not m.is_dir()]
                info["zip_member_count"] = len(members)
                info["sample_members"] = members[:5]
        sources_info[name] = info

    # 2. Canonical Serving Artifacts
    artifacts_info: dict[str, Any] = {
        "serving_root": str(serving_root),
        "manifests": {},
    }
    dataset_manifest_path = serving_root / "dataset_manifest.json"
    if dataset_manifest_path.exists():
        with dataset_manifest_path.open("r", encoding="utf-8") as f:
            artifacts_info["dataset_manifest"] = json.load(f)
            
    for sub in [
        "cleaned_documents",
        "normalized_documents",
        "legal_blocks",
        "legal_chunks",
        "bm25",
        "vector",
        "vector_serving",
        "relationships",
        "graph",
    ]:
        sub_dir = serving_root / sub
        manifest_p = sub_dir / "manifest.json"
        if manifest_p.exists():
            with manifest_p.open("r", encoding="utf-8") as f:
                man = json.load(f)
            artifacts_info["manifests"][sub] = man

    return {
        "sources": sources_info,
        "artifacts": artifacts_info,
    }


# --- Section 5 & 6: Raw Corpus & Duplication Census ---

_HTML_TAG_REGEX = re.compile(r"<[^>]+>")
_TABLE_PATTERN = re.compile(r"(?:\|.*\||\+[-+]+\+|\bBảng\s+\d+|\bBiểu\s+số\s+\d+|\t{2,})", re.IGNORECASE)

def audit_raw_corpus(contexts_path: Path) -> dict[str, Any]:
    """Audit the raw official context records."""
    loader = UitDsc2026DataLoader()
    contexts = list(loader.iter_contexts(contexts_path))
    
    total_records = len(contexts)
    non_empty_records = 0
    empty_records = 0
    with_title_records = 0
    with_url_records = 0
    missing_title_ids: list[str] = []
    
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    
    newline_counts = {"crlf": 0, "lf": 0, "cr": 0, "mixed": 0, "none": 0}
    unicode_nfc_diff_count = 0
    html_remnants_count = 0
    tvpl_boilerplate_count = 0
    table_pattern_count = 0
    
    exact_duplicates: dict[str, list[str]] = defaultdict(list)
    norm_duplicates: dict[str, list[str]] = defaultdict(list)
    name_duplicates: dict[str, list[str]] = defaultdict(list)
    url_duplicates: dict[str, list[str]] = defaultdict(list)
    
    doc_sizes: list[dict[str, Any]] = []

    for ctx in contexts:
        passage = ctx.passage
        p_len = len(passage)
        has_content = bool(passage.strip())
        
        if has_content:
            non_empty_records += 1
            char_lengths.append(p_len)
            tokens = passage.split()
            word_lengths.append(len(tokens))
            doc_sizes.append({
                "context_id": ctx.context_id,
                "title": ctx.title,
                "char_length": p_len,
                "word_length": len(tokens),
            })
        else:
            empty_records += 1
            
        if ctx.title and ctx.title.strip():
            with_title_records += 1
            name_duplicates[ctx.title.strip().casefold()].append(ctx.context_id)
        else:
            missing_title_ids.append(ctx.context_id)
            
        if ctx.source_url and ctx.source_url.strip():
            with_url_records += 1
            url_duplicates[ctx.source_url.strip().casefold()].append(ctx.context_id)
            
        # Duplication
        if has_content:
            exact_duplicates[passage].append(ctx.context_id)
            norm_text = normalize_text_for_dup(passage)
            norm_duplicates[norm_text].append(ctx.context_id)
            
        # Newline style
        has_crlf = "\r\n" in passage
        has_pure_lf = "\n" in passage.replace("\r\n", "")
        has_pure_cr = "\r" in passage.replace("\r\n", "")
        if has_crlf and (has_pure_lf or has_pure_cr):
            newline_counts["mixed"] += 1
        elif has_crlf:
            newline_counts["crlf"] += 1
        elif has_pure_lf:
            newline_counts["lf"] += 1
        elif has_pure_cr:
            newline_counts["cr"] += 1
        else:
            newline_counts["none"] += 1
            
        # Unicode normalization
        if unicodedata.normalize("NFC", passage) != passage:
            unicode_nfc_diff_count += 1
            
        # HTML remnants
        if _KNOWN_MARKUP.search(passage) or _HTML_TAG_REGEX.search(passage):
            html_remnants_count += 1
            
        # TVPL Boilerplate
        if _TVPL_PRO_NOTICE in passage:
            tvpl_boilerplate_count += 1
            
        # Table-like patterns
        if _TABLE_PATTERN.search(passage):
            table_pattern_count += 1

    # Top 20 largest documents
    doc_sizes.sort(key=lambda x: x["char_length"], reverse=True)
    top_20_largest = doc_sizes[:20]

    # Duplicate clusters
    exact_clusters = {k: v for k, v in exact_duplicates.items() if len(v) > 1}
    norm_clusters = {k: v for k, v in norm_duplicates.items() if len(v) > 1}
    name_clusters = {k: v for k, v in name_duplicates.items() if len(v) > 1}
    url_clusters = {k: v for k, v in url_duplicates.items() if len(v) > 1}

    exact_dup_records = sum(len(v) for v in exact_clusters.values())
    norm_dup_records = sum(len(v) for v in norm_clusters.values())

    return {
        "total_records": total_records,
        "non_empty_records": non_empty_records,
        "empty_records": empty_records,
        "with_title_records": with_title_records,
        "without_title_records": total_records - with_title_records,
        "with_url_records": with_url_records,
        "char_length_distribution": compute_percentiles(char_lengths),
        "word_length_distribution": compute_percentiles(word_lengths),
        "total_characters": sum(char_lengths),
        "newline_counts": newline_counts,
        "unicode_nfc_diff_count": unicode_nfc_diff_count,
        "html_remnants_count": html_remnants_count,
        "tvpl_boilerplate_count": tvpl_boilerplate_count,
        "table_pattern_count": table_pattern_count,
        "top_20_largest_documents": top_20_largest,
        "duplication": {
            "exact_duplicate_clusters": len(exact_clusters),
            "exact_duplicate_records": exact_dup_records,
            "exact_duplicate_percentage": (exact_dup_records / total_records) * 100 if total_records else 0,
            "largest_exact_clusters": [
                {"cluster_size": len(v), "sample_ids": v[:5]}
                for v in sorted(exact_clusters.values(), key=len, reverse=True)[:5]
            ],
            "norm_duplicate_clusters": len(norm_clusters),
            "norm_duplicate_records": norm_dup_records,
            "norm_duplicate_percentage": (norm_dup_records / total_records) * 100 if total_records else 0,
            "largest_norm_clusters": [
                {"cluster_size": len(v), "sample_ids": v[:5]}
                for v in sorted(norm_clusters.values(), key=len, reverse=True)[:5]
            ],
            "duplicate_title_clusters": len(name_clusters),
            "duplicate_title_records": sum(len(v) for v in name_clusters.values()),
            "duplicate_url_clusters": len(url_clusters),
            "duplicate_url_records": sum(len(v) for v in url_clusters.values()),
        },
    }


# --- Section 7 & 8: Legal Structure Markers & Parser Coverage ---

_RE_MARKERS = {
    "PHAN": re.compile(r"^PHẦN\b", re.IGNORECASE | re.MULTILINE),
    "CHUONG": re.compile(r"^CHƯƠNG\b", re.IGNORECASE | re.MULTILINE),
    "MUC": re.compile(r"^MỤC\b", re.IGNORECASE | re.MULTILINE),
    "TIEU_MUC": re.compile(r"^TIỂU\s+MỤC\b", re.IGNORECASE | re.MULTILINE),
    "DIEU": re.compile(r"^ĐIỀU\s+\d+", re.IGNORECASE | re.MULTILINE),
    "KHOAN_NUMBERED": re.compile(r"^\d+[A-ZĐ]?\.\s+", re.MULTILINE),
    "KHOAN_EXPLICIT": re.compile(r"^KHOẢN\s+\d+", re.IGNORECASE | re.MULTILINE),
    "DIEM_LETTERED": re.compile(r"^[a-zđ]\)\s+", re.IGNORECASE | re.MULTILINE),
    "DIEM_EXPLICIT": re.compile(r"^ĐIỂM\s+[a-zđ]", re.IGNORECASE | re.MULTILINE),
    "PHU_LUC": re.compile(r"^PHỤ\s+LỤC\b", re.IGNORECASE | re.MULTILINE),
}

def audit_legal_markers_and_parser(
    contexts_path: Path,
    cleaner: UitDsc2026PassageCleaner,
    parser: LegalStructureParser,
) -> dict[str, Any]:
    """Audit raw legal markers and parser coverage on canonical corpus."""
    loader = UitDsc2026DataLoader()
    contexts = list(loader.iter_contexts(contexts_path))
    
    marker_presence: dict[str, int] = defaultdict(int)
    marker_and_parsed_counts = {
        "dieu_marker_and_article_parsed": 0,
        "dieu_marker_present_article_not_parsed": 0,
        "article_parsed_dieu_marker_absent": 0,
    }
    
    docs_parsed_article_ge_1 = 0
    docs_zero_structure = 0
    article_counts: list[int] = []
    clause_counts: list[int] = []
    point_counts: list[int] = []
    issue_type_counts: Counter[str] = Counter()
    
    unresolved_marker_samples: list[dict[str, Any]] = []

    for ctx in contexts:
        if not ctx.passage.strip():
            docs_zero_structure += 1
            continue
            
        cleaned = cleaner.clean(ctx.passage)
        clean_text = cleaned.text
        
        # Raw marker detection on cleaned text
        has_markers: dict[str, bool] = {}
        for m_name, regex in _RE_MARKERS.items():
            found = bool(regex.search(clean_text))
            has_markers[m_name] = found
            if found:
                marker_presence[m_name] += 1
                
        # Parse structure
        doc = LegalDocument(
            document_id=ctx.context_id,
            title=ctx.title,
            source_url=ctx.source_url,
            clean_text=clean_text,
            has_content=bool(clean_text.strip()),
            source_dataset="uit-dsc-2026-task2-selected-contexts",
        )
        parsed_res = parser.parse_document(doc)
        
        n_articles = sum(1 for b in parsed_res.blocks if b.block_type == LegalBlockType.ARTICLE)
        n_clauses = sum(1 for b in parsed_res.blocks if b.block_type == LegalBlockType.CLAUSE)
        n_points = sum(1 for b in parsed_res.blocks if b.block_type == LegalBlockType.POINT)
        
        article_counts.append(n_articles)
        clause_counts.append(n_clauses)
        point_counts.append(n_points)
        
        if n_articles >= 1:
            docs_parsed_article_ge_1 += 1
        else:
            docs_zero_structure += 1
            
        for issue in parsed_res.issues:
            issue_type_counts[issue.issue_type] += 1
            
        # Agreement for DIEU
        has_dieu_marker = has_markers["DIEU"]
        has_article_parsed = n_articles > 0
        
        if has_dieu_marker and has_article_parsed:
            marker_and_parsed_counts["dieu_marker_and_article_parsed"] += 1
        elif has_dieu_marker and not has_article_parsed:
            marker_and_parsed_counts["dieu_marker_present_article_not_parsed"] += 1
            if len(unresolved_marker_samples) < 30:
                match = _RE_MARKERS["DIEU"].search(clean_text)
                snippet = clean_text[max(0, match.start() - 50): min(len(clean_text), match.end() + 150)] if match else ""
                unresolved_marker_samples.append({
                    "context_id": ctx.context_id,
                    "title": ctx.title,
                    "marker_type": "DIEU",
                    "snippet": snippet.strip(),
                })
        elif not has_dieu_marker and has_article_parsed:
            marker_and_parsed_counts["article_parsed_dieu_marker_absent"] += 1

    return {
        "raw_marker_presence_counts": dict(marker_presence),
        "dieu_marker_agreement": marker_and_parsed_counts,
        "docs_parsed_article_ge_1": docs_parsed_article_ge_1,
        "docs_zero_structure": docs_zero_structure,
        "article_count_distribution": compute_percentiles(article_counts),
        "clause_count_distribution": compute_percentiles(clause_counts),
        "point_count_distribution": compute_percentiles(point_counts),
        "parser_issue_counts": dict(issue_type_counts),
        "unresolved_marker_samples": unresolved_marker_samples,
    }


# --- Section 9, 10, 11, 12, 13, 14: Legal Chunks, Boundary Risks, Metadata & Search Text ---

_MATERIAL_CONNECTORS = [
    r"nếu\b",
    r"khi\b",
    r"trường\s+hợp\b",
    r"trừ\s+trường\s+hợp\b",
    r"trừ\s+khi\b",
    r"với\s+điều\s+kiện\b",
    r"được\b[^\n.]{1,40}\bkhi\b",
    r"chỉ\b[^\n.]{1,40}\bkhi\b",
    r"theo\s+quy\s+định\s+tại\b",
    r"quy\s+định\s+tại\s+khoản\b",
    r"quy\s+định\s+tại\s+điểm\b",
    r"quy\s+định\s+tại\s+điều\b",
    r"bao\s+gồm\s*(?:các\s+trường\s+hợp\s+sau)?\s*:",
    r"như\s+sau\s*:",
]
_RE_MATERIAL_CONNECTORS = re.compile("|".join(_MATERIAL_CONNECTORS), re.IGNORECASE)

def audit_legal_chunks(serving_root: Path) -> dict[str, Any]:
    """Audit canonical legal_chunks artifact records.jsonl."""
    chunks_path = serving_root / "legal_chunks" / "records.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"legal_chunks records not found: {chunks_path}")
        
    total_chunks = 0
    doc_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    
    char_lengths: list[int] = []
    search_text_lengths: list[int] = []
    token_counts: list[int] = []
    
    strategy_counts: Counter[str] = Counter()
    multi_article_chunks = 0
    multi_clause_chunks = 0
    multi_point_chunks = 0
    
    # search_text coverage
    st_title_coverage = 0
    st_number_coverage = 0
    st_type_coverage = 0
    st_hierarchy_coverage = 0
    st_article_coverage = 0
    st_clause_coverage = 0
    st_point_coverage = 0
    
    # Metadata coverage
    metadata_fields = [
        "document_title",
        "document_number",
        "document_type",
        "issuing_authority",
        "issuance_date",
        "effective_date",
        "expiry_date",
        "effect_status",
        "legal_field",
        "chapter",
        "section",
        "article",
        "clause",
        "point",
        "source_url",
    ]
    meta_non_null: Counter[str] = Counter()
    meta_distinct: dict[str, set[str]] = defaultdict(set)
    
    exact_chunk_duplicates: Counter[str] = Counter()
    norm_chunk_duplicates: Counter[str] = Counter()
    exact_st_duplicates: Counter[str] = Counter()
    chunk_text_to_docs: dict[str, set[str]] = defaultdict(set)
    
    # Sample search_text headers
    sample_st_headers: dict[str, list[dict[str, str]]] = defaultdict(list)

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_chunks += 1
            
            doc_id = rec.get("document_id", "")
            doc_chunks[doc_id].append(rec)
            
            text = rec.get("text", "")
            st_text = rec.get("search_text", "")
            tok_count = rec.get("token_count", 0)
            
            char_lengths.append(len(text))
            search_text_lengths.append(len(st_text))
            token_counts.append(tok_count)
            
            meta = rec.get("metadata", {})
            strat = meta.get("chunk_strategy", "unknown")
            strategy_counts[strat] += 1
            
            struct = rec.get("structure", {})
            clauses = struct.get("clause_numbers", [])
            points = struct.get("point_numbers", [])
            art_num = struct.get("article_number")
            
            if len(clauses) > 1:
                multi_clause_chunks += 1
            if len(points) > 1:
                multi_point_chunks += 1
                
            # Search text coverage
            if "Văn bản:" in st_text:
                st_title_coverage += 1
            if "Số ký hiệu:" in st_text:
                st_number_coverage += 1
            if "Loại văn bản:" in st_text:
                st_type_coverage += 1
            if any(k in st_text for k in ["Phần", "Chương", "Mục", "Tiểu mục"]):
                st_hierarchy_coverage += 1
            if "Điều " in st_text:
                st_article_coverage += 1
            if "Khoản:" in st_text:
                st_clause_coverage += 1
            if "Điểm:" in st_text:
                st_point_coverage += 1
                
            # Sample headers
            if len(sample_st_headers[strat]) < 3:
                header = st_text.split("Nội dung:")[0].strip() if "Nội dung:" in st_text else st_text[:120]
                sample_st_headers[strat].append({
                    "chunk_id": rec.get("chunk_id", ""),
                    "header": header,
                })
                
            # Metadata stats
            for field_name in metadata_fields:
                val = rec.get(field_name)
                if val is None:
                    val = struct.get(field_name)
                if val is not None and str(val).strip():
                    meta_non_null[field_name] += 1
                    meta_distinct[field_name].add(str(val))
                    
            # Duplication
            exact_chunk_duplicates[text] += 1
            norm_chunk_duplicates[normalize_text_for_dup(text)] += 1
            exact_st_duplicates[st_text] += 1
            chunk_text_to_docs[text].add(doc_id)

    # Chunks per document
    chunks_per_doc = [len(v) for v in doc_chunks.values()]
    
    # Boundary Risk Audit
    boundary_risk_counts: Counter[str] = Counter()
    boundary_risk_samples: list[dict[str, Any]] = []
    total_adjacent_pairs = 0
    
    for doc_id, c_list in doc_chunks.items():
        if len(c_list) < 2:
            continue
        c_list.sort(key=lambda x: x.get("chunk_index", 0))
        for i in range(len(c_list) - 1):
            total_adjacent_pairs += 1
            c1 = c_list[i]
            c2 = c_list[i + 1]
            t1 = c1.get("text", "")
            t2 = c2.get("text", "")
            
            tail_t1 = t1[-100:] if len(t1) >= 100 else t1
            head_t2 = t2[:100] if len(t2) >= 100 else t2
            
            risks: list[str] = []
            if re.search(r"(?:nếu|khi|trường hợp|trừ|với điều kiện)\s*$", tail_t1, re.IGNORECASE):
                risks.append("CONDITION_OPEN_AT_LEFT_BOUNDARY")
            if re.search(r"(?:theo quy định tại|quy định tại khoản|quy định tại điểm|tại khoản|tại điểm)\s*$", tail_t1, re.IGNORECASE):
                risks.append("CROSS_REFERENCE_SPLIT")
            if re.search(r"(?:bao gồm|như sau|các trường hợp sau)\s*:\s*$", tail_t1, re.IGNORECASE):
                risks.append("LIST_HEADER_SPLIT_FROM_ITEMS")
                
            for r_name in risks:
                boundary_risk_counts[r_name] += 1
                if len(boundary_risk_samples) < 20:
                    boundary_risk_samples.append({
                        "document_id": doc_id,
                        "risk_type": r_name,
                        "c1_chunk_id": c1.get("chunk_id"),
                        "c2_chunk_id": c2.get("chunk_id"),
                        "c1_tail": tail_t1.strip()[-60:],
                        "c2_head": head_t2.strip()[:60],
                    })

    # Duplication summary
    exact_dup_chunk_count = sum(cnt for cnt in exact_chunk_duplicates.values() if cnt > 1)
    norm_dup_chunk_count = sum(cnt for cnt in norm_chunk_duplicates.values() if cnt > 1)
    cross_doc_dup_count = sum(1 for docs in chunk_text_to_docs.values() if len(docs) > 1)

    return {
        "total_chunks": total_chunks,
        "unique_source_documents": len(doc_chunks),
        "chunks_per_document_distribution": compute_percentiles(chunks_per_doc),
        "char_length_distribution": compute_percentiles(char_lengths),
        "search_text_length_distribution": compute_percentiles(search_text_lengths),
        "token_count_distribution": compute_percentiles(token_counts),
        "strategy_distribution": {
            k: {"count": v, "percentage": (v / total_chunks) * 100}
            for k, v in strategy_counts.items()
        },
        "multi_span_chunks": {
            "multi_clause_chunks": multi_clause_chunks,
            "multi_point_chunks": multi_point_chunks,
        },
        "search_text_coverage": {
            "title_coverage_count": st_title_coverage,
            "title_coverage_pct": (st_title_coverage / total_chunks) * 100,
            "number_coverage_count": st_number_coverage,
            "number_coverage_pct": (st_number_coverage / total_chunks) * 100,
            "type_coverage_count": st_type_coverage,
            "type_coverage_pct": (st_type_coverage / total_chunks) * 100,
            "hierarchy_coverage_count": st_hierarchy_coverage,
            "hierarchy_coverage_pct": (st_hierarchy_coverage / total_chunks) * 100,
            "article_coverage_count": st_article_coverage,
            "article_coverage_pct": (st_article_coverage / total_chunks) * 100,
            "clause_coverage_count": st_clause_coverage,
            "clause_coverage_pct": (st_clause_coverage / total_chunks) * 100,
            "point_coverage_count": st_point_coverage,
            "point_coverage_pct": (st_point_coverage / total_chunks) * 100,
        },
        "sample_search_text_headers": dict(sample_st_headers),
        "metadata_coverage": {
            field_name: {
                "non_null_count": meta_non_null[field_name],
                "coverage_percentage": (meta_non_null[field_name] / total_chunks) * 100,
                "distinct_count": len(meta_distinct[field_name]),
            }
            for field_name in metadata_fields
        },
        "boundary_risk": {
            "total_adjacent_pairs_evaluated": total_adjacent_pairs,
            "risk_counts": dict(boundary_risk_counts),
            "risk_percentages": {
                k: (v / total_adjacent_pairs) * 100 if total_adjacent_pairs else 0
                for k, v in boundary_risk_counts.items()
            },
            "samples": boundary_risk_samples,
        },
        "chunk_duplication": {
            "exact_duplicate_chunks": exact_dup_chunk_count,
            "exact_duplicate_pct": (exact_dup_chunk_count / total_chunks) * 100,
            "norm_duplicate_chunks": norm_dup_chunk_count,
            "norm_duplicate_pct": (norm_dup_chunk_count / total_chunks) * 100,
            "cross_document_duplicate_chunks": cross_doc_dup_count,
        },
    }


# --- Section 15, 16, 17, 18: Train Q&A Deep Census & Linkability ---

_TAXONOMY_RULES = {
    "binary_yes_no": re.compile(r"^(?:Có\b|Không\b|Được\b|Không được\b|Phải\b|Không phải\b|Đúng\b|Sai\b)", re.IGNORECASE),
    "definition": re.compile(r"(?:là gì|được hiểu là|khái niệm|định nghĩa|quy định tại khoản \d+ điều \d+ về khái niệm)", re.IGNORECASE),
    "procedure_steps": re.compile(r"(?:thủ tục|trình tự|các bước|hồ sơ|quy trình|tiến hành)", re.IGNORECASE),
    "rights_obligations": re.compile(r"(?:quyền|nghĩa vụ|trách nhiệm|bổn phận|được phép)", re.IGNORECASE),
    "authority_competence": re.compile(r"(?:thẩm quyền|cơ quan nào|ai có quyền|chủ tịch|bộ trưởng|ủy ban)", re.IGNORECASE),
    "penalty_sanction": re.compile(r"(?:xử phạt|phạt tiền|tước quyền|hình phạt|tịch thu|phạt cảnh cáo|truy cứu)", re.IGNORECASE),
    "condition_eligibility": re.compile(r"(?:điều kiện|tiêu chuẩn|đối tượng|yêu cầu|nguyên tắc)", re.IGNORECASE),
    "deadline_temporal": re.compile(r"(?:thời hạn|thời gian|ngày|tháng|năm|bao lâu|kể từ ngày)", re.IGNORECASE),
    "numeric_money_quantity": re.compile(r"(?:mức phạt|tỷ lệ|phần trăm|đồng|triệu đồng|tỷ đồng|\d+\s*%)", re.IGNORECASE),
    "legal_citation_lookup": re.compile(r"(?:theo quy định tại|căn cứ vào|quy định tại điều|luật số|nghị định số)", re.IGNORECASE),
}

_RE_DOC_NUMBER = re.compile(
    r"\b\d+/\d+/(?:NĐ-CP|TT-B\w+|QĐ-UBND|TT-BTC|TT-BYT|TT-BGDĐT|TT-BCA|TT-BLĐTBXH|TT-BCT|TT-BNV|TT-BTNMT|TT-BTP|TT-NHNN|NQ-CP|NQ-HĐND|VBHN-B\w+)\b",
    re.IGNORECASE,
)
_RE_DOC_TYPE_NAME = re.compile(
    r"\b(?:Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Pháp lệnh|Nghị quyết)\s+(?:số\s+)?[\w\s\d/–\-]{3,40}",
    re.IGNORECASE,
)
_RE_ARTICLE_REF = re.compile(r"\bĐiều\s+(\d+[A-ZĐ]?)\b", re.IGNORECASE)
_RE_CLAUSE_REF = re.compile(r"\bKhoản\s+(\d+[A-ZĐ]?)\b", re.IGNORECASE)
_RE_POINT_REF = re.compile(r"\bĐiểm\s+([a-zđ])\b", re.IGNORECASE)

def _clean_slug(s: str | None) -> str:
    if not s:
        return ""
    norm = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", norm.lower())


_RE_DOC_NUM_GENERIC = re.compile(
    r"\b\d+/(?:[0-9]{4}/)?[A-ZĐ0-9\-]+(?:/[A-ZĐ0-9\-]+)?\b",
    re.IGNORECASE,
)

def audit_train_qa_and_linkability(
    train_path: Path,
    public_path: Path,
    contexts_path: Path,
    serving_root: Path,
) -> dict[str, Any]:
    """Deep audit of train Q&A, legal references, and high-confidence source linkability."""
    loader = UitDsc2026DataLoader()
    train_questions = loader.load_questions(train_path, require_reference_answers=True)
    public_questions = loader.load_questions(public_path, require_reference_answers=False)
    
    total_qa = len(train_questions)
    q_char_lengths: list[int] = []
    q_word_lengths: list[int] = []
    a_char_lengths: list[int] = []
    a_word_lengths: list[int] = []
    ratios: list[float] = []
    
    taxonomy_counts: Counter[str] = Counter()
    prose_style_counts = {
        "single_short_sentence": 0,
        "multi_sentence_prose": 0,
        "multi_paragraph": 0,
        "enumerated_list": 0,
        "bullet_list": 0,
    }
    
    # Legal reference signal counts
    ref_signal_counts = {
        "q_has_article": 0,
        "q_has_clause": 0,
        "q_has_point": 0,
        "q_has_doc_number": 0,
        "q_has_doc_type_name": 0,
        "a_has_article": 0,
        "a_has_clause": 0,
        "a_has_point": 0,
        "a_has_doc_number": 0,
        "a_has_doc_type_name": 0,
        "both_have_article": 0,
        "either_has_article": 0,
    }
    
    # Duplication & Family
    exact_q_clusters: dict[str, list[str]] = defaultdict(list)
    norm_q_clusters: dict[str, list[str]] = defaultdict(list)
    norm_q_to_answers: dict[str, set[str]] = defaultdict(set)
    a_clusters: dict[str, list[str]] = defaultdict(list)
    
    # Load canonical contexts for linkability
    contexts = list(loader.iter_contexts(contexts_path))
    doc_slugs: dict[str, str] = {}
    for c in contexts:
        if c.title:
            doc_slugs[c.context_id] = _clean_slug(c.title)

    # Linkability counters
    unambiguous_doc_links: list[dict[str, Any]] = []
    unambiguous_article_links: list[dict[str, Any]] = []
    ambiguous_links = 0
    unresolved_anchors = 0
    answers_with_explicit_anchors = 0

    for q in train_questions:
        q_text = q.question
        a_text = q.reference_answer or ""
        
        q_char_lengths.append(len(q_text))
        q_tokens = q_text.split()
        q_word_lengths.append(len(q_tokens))
        
        a_char_lengths.append(len(a_text))
        a_tokens = a_text.split()
        a_word_lengths.append(len(a_tokens))
        
        ratio = len(a_text) / len(q_text) if len(q_text) > 0 else 0.0
        ratios.append(ratio)
        
        # Taxonomy
        for tax_name, pattern in _TAXONOMY_RULES.items():
            if pattern.search(q_text) or pattern.search(a_text):
                taxonomy_counts[tax_name] += 1
                
        # Prose style
        paragraphs = [p for p in a_text.split("\n") if p.strip()]
        if len(paragraphs) > 1:
            prose_style_counts["multi_paragraph"] += 1
        if re.search(r"(?:^\s*\d+\.|\b\d+\.\s+|^\s*[a-zđ]\))", a_text, re.MULTILINE):
            prose_style_counts["enumerated_list"] += 1
        if re.search(r"^\s*[-*•]\s+", a_text, re.MULTILINE):
            prose_style_counts["bullet_list"] += 1
        if len(paragraphs) == 1 and len(a_text) < 120 and ("." not in a_text[:-1]):
            prose_style_counts["single_short_sentence"] += 1
        elif len(paragraphs) == 1:
            prose_style_counts["multi_sentence_prose"] += 1
            
        # Reference signals
        q_art = bool(_RE_ARTICLE_REF.search(q_text))
        q_cl = bool(_RE_CLAUSE_REF.search(q_text))
        q_pt = bool(_RE_POINT_REF.search(q_text))
        q_dn = bool(_RE_DOC_NUMBER.search(q_text))
        q_dtn = bool(_RE_DOC_TYPE_NAME.search(q_text))
        
        a_art = bool(_RE_ARTICLE_REF.search(a_text))
        a_cl = bool(_RE_CLAUSE_REF.search(a_text))
        a_pt = bool(_RE_POINT_REF.search(a_text))
        a_dn = bool(_RE_DOC_NUMBER.search(a_text))
        a_dtn = bool(_RE_DOC_TYPE_NAME.search(a_text))
        
        if q_art: ref_signal_counts["q_has_article"] += 1
        if q_cl: ref_signal_counts["q_has_clause"] += 1
        if q_pt: ref_signal_counts["q_has_point"] += 1
        if q_dn: ref_signal_counts["q_has_doc_number"] += 1
        if q_dtn: ref_signal_counts["q_has_doc_type_name"] += 1
        
        if a_art: ref_signal_counts["a_has_article"] += 1
        if a_cl: ref_signal_counts["a_has_clause"] += 1
        if a_pt: ref_signal_counts["a_has_point"] += 1
        if a_dn: ref_signal_counts["a_has_doc_number"] += 1
        if a_dtn: ref_signal_counts["a_has_doc_type_name"] += 1
        
        if q_art and a_art: ref_signal_counts["both_have_article"] += 1
        if q_art or a_art: ref_signal_counts["either_has_article"] += 1
        
        # Duplicate questions
        exact_q_clusters[q_text].append(q.question_id)
        norm_q = normalize_text_for_dup(q_text)
        norm_q_clusters[norm_q].append(q.question_id)
        norm_q_to_answers[norm_q].add(normalize_text_for_dup(a_text))
        a_clusters[a_text].append(q.question_id)
        
        # Linkability: Extract explicit document and article citations from reference answer
        doc_nums = _RE_DOC_NUMBER.findall(a_text) or _RE_DOC_NUM_GENERIC.findall(a_text)
        art_nums = _RE_ARTICLE_REF.findall(a_text)
        
        has_anchor = bool(doc_nums or (a_dtn and art_nums))
        if has_anchor:
            answers_with_explicit_anchors += 1
            
            matched_docs: set[str] = set()
            # Match doc numbers using normalized alphanumeric slug
            for dn in doc_nums:
                dn_slug = _clean_slug(dn)
                if len(dn_slug) < 4:
                    continue
                for cid, title_slug in doc_slugs.items():
                    if dn_slug in title_slug:
                        matched_docs.add(cid)
                        
            if len(matched_docs) == 1:
                doc_id = next(iter(matched_docs))
                unambiguous_doc_links.append({
                    "question_id": q.question_id,
                    "target_document_id": doc_id,
                    "doc_numbers": doc_nums,
                    "article_numbers": art_nums,
                })
                if len(art_nums) == 1:
                    unambiguous_article_links.append({
                        "question_id": q.question_id,
                        "target_document_id": doc_id,
                        "target_article_number": art_nums[0],
                    })
            elif len(matched_docs) > 1:
                ambiguous_links += 1
            else:
                unresolved_anchors += 1

    # Overlap with Public & Warmup
    public_exact_q = {q.question for q in public_questions}
    public_norm_q = {normalize_text_for_dup(q.question) for q in public_questions}
    
    train_exact_set = {q.question for q in train_questions}
    train_norm_set = {normalize_text_for_dup(q.question) for q in train_questions}
    
    public_exact_overlap = train_exact_set.intersection(public_exact_q)
    public_norm_overlap = train_norm_set.intersection(public_norm_q)
    
    # Inconsistent question families (same question, different answer)
    inconsistent_families = {q: ans for q, ans in norm_q_to_answers.items() if len(ans) > 1}

    return {
        "total_train_questions": total_qa,
        "question_char_distribution": compute_percentiles(q_char_lengths),
        "question_word_distribution": compute_percentiles(q_word_lengths),
        "answer_char_distribution": compute_percentiles(a_char_lengths),
        "answer_word_distribution": compute_percentiles(a_word_lengths),
        "answer_to_question_ratio_distribution": compute_percentiles(ratios),
        "taxonomy_distribution": {
            k: {"count": v, "percentage": (v / total_qa) * 100}
            for k, v in taxonomy_counts.items()
        },
        "prose_style_distribution": {
            k: {"count": v, "percentage": (v / total_qa) * 100}
            for k, v in prose_style_counts.items()
        },
        "legal_reference_signal_counts": ref_signal_counts,
        "duplication": {
            "exact_duplicate_questions": sum(len(v) for v in exact_q_clusters.values() if len(v) > 1),
            "norm_duplicate_questions": sum(len(v) for v in norm_q_clusters.values() if len(v) > 1),
            "duplicate_answers": sum(len(v) for v in a_clusters.values() if len(v) > 1),
            "inconsistent_question_families": len(inconsistent_families),
            "train_public_exact_overlap_count": len(public_exact_overlap),
            "train_public_norm_overlap_count": len(public_norm_overlap),
        },
        "linkability": {
            "answers_with_explicit_anchors": answers_with_explicit_anchors,
            "unambiguous_doc_links_count": len(unambiguous_doc_links),
            "unambiguous_doc_links_pct": (len(unambiguous_doc_links) / total_qa) * 100,
            "unambiguous_article_links_count": len(unambiguous_article_links),
            "unambiguous_article_links_pct": (len(unambiguous_article_links) / total_qa) * 100,
            "ambiguous_links_count": ambiguous_links,
            "unresolved_anchors_count": unresolved_anchors,
            "sample_unambiguous_links": unambiguous_doc_links[:10],
            "all_unambiguous_links": unambiguous_doc_links,
        },
    }


# --- Section 19: Optional Retrieval Proxy Audit ---

def audit_retrieval_proxy(
    serving_root: Path,
    unambiguous_links: list[dict[str, Any]],
    train_questions: list[CompetitionQuestion],
) -> dict[str, Any]:
    """Run read-only BM25 retrieval against SQLite index to measure proxy document/article recall."""
    import sqlite3
    
    bm25_db_path = serving_root / "bm25" / "index.sqlite3"
    if not bm25_db_path.exists():
        return {"status": "SKIPPED", "reason": f"BM25 index not found at {bm25_db_path}"}
        
    if not unambiguous_links:
        return {"status": "SKIPPED", "reason": "No unambiguous linkages available for proxy audit"}
        
    # Map qid -> question
    q_map = {q.question_id: q.question for q in train_questions}
    
    conn = sqlite3.connect(str(bm25_db_path))
    cursor = conn.cursor()
    
    # Detect table name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('bm25_documents', 'chunks_fts')")
    table_row = cursor.fetchone()
    if not table_row:
        conn.close()
        return {"status": "SKIPPED", "reason": "No valid BM25 FTS5 table found in database"}
    table_name = table_row[0]
    
    doc_recall_at_1 = 0
    doc_recall_at_5 = 0
    doc_recall_at_10 = 0
    doc_recall_at_20 = 0
    
    evaluated_count = 0
    
    # We take up to 200 unambiguous links for fast, deterministic evaluation
    eval_sample = unambiguous_links[:200]
    
    for item in eval_sample:
        qid = item["question_id"]
        target_doc = item["target_document_id"]
        q_text = q_map.get(qid, "")
        if not q_text:
            continue
            
        # Clean query for FTS5: alphanumeric and whitespace only
        clean_q = re.sub(r"[^\w\s]", " ", q_text)
        tokens = [f'"{t}"' for t in clean_q.split() if len(t) > 1][:15]
        if not tokens:
            continue
            
        match_query = " OR ".join(tokens)
        try:
            cursor.execute(
                f"SELECT chunk_id, document_id, bm25({table_name}) as score "
                f"FROM {table_name} WHERE {table_name} MATCH ? ORDER BY score LIMIT 20",
                (match_query,),
            )
            rows = cursor.fetchall()
        except Exception:
            continue
            
        retrieved_docs = [r[1] for r in rows]
        evaluated_count += 1
        
        if target_doc in retrieved_docs[:1]:
            doc_recall_at_1 += 1
        if target_doc in retrieved_docs[:5]:
            doc_recall_at_5 += 1
        if target_doc in retrieved_docs[:10]:
            doc_recall_at_10 += 1
        if target_doc in retrieved_docs[:20]:
            doc_recall_at_20 += 1
            
    conn.close()
    
    return {
        "status": "COMPLETED",
        "evaluated_links_count": evaluated_count,
        "document_recall_at_1": (doc_recall_at_1 / evaluated_count) * 100 if evaluated_count else 0,
        "document_recall_at_5": (doc_recall_at_5 / evaluated_count) * 100 if evaluated_count else 0,
        "document_recall_at_10": (doc_recall_at_10 / evaluated_count) * 100 if evaluated_count else 0,
        "document_recall_at_20": (doc_recall_at_20 / evaluated_count) * 100 if evaluated_count else 0,
    }


# --- Main CLI & Orchestration ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Official Data Census & Legal Retrieval-Unit Audit (Phase D0)")
    parser.add_argument("--train-path", type=Path, default=Path(r"C:\Users\Nguyen\Downloads\train.json"))
    parser.add_argument("--public-path", type=Path, default=Path(r"C:\Users\Nguyen\Downloads\public-official.json"))
    parser.add_argument("--contexts-path", type=Path, default=Path(r"C:\Users\Nguyen\Downloads\selected-contexts.zip"))
    parser.add_argument("--serving-root", type=Path, default=Path("artifacts/uit-dsc-2026-task2-v0400"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/data_d0_audit"))
    parser.add_argument("--evidence-zip", type=Path, default=Path(r"C:\Users\Nguyen\Downloads\data-d0-official-data-audit-evidence.zip"))
    parser.add_argument("--run-retrieval-proxy", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    
    _LOGGER.info("Starting Phase D0 Deep Official Data Audit...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sources & Artifacts
    _LOGGER.info("Auditing source identities and serving manifests...")
    sources_and_artifacts = audit_sources_and_artifacts(
        args.train_path,
        args.public_path,
        args.contexts_path,
        args.serving_root,
    )
    with (args.output_dir / "source_identity.json").open("w", encoding="utf-8") as f:
        json.dump(sources_and_artifacts, f, indent=2, ensure_ascii=False)

    # 2. Raw Corpus Census & Duplication
    _LOGGER.info("Auditing raw official context records...")
    raw_corpus = audit_raw_corpus(args.contexts_path)
    with (args.output_dir / "raw_corpus_census.json").open("w", encoding="utf-8") as f:
        json.dump(raw_corpus, f, indent=2, ensure_ascii=False)

    # 3. Legal Structure Markers & Parser Coverage
    _LOGGER.info("Auditing legal structure markers & parser coverage...")
    cleaner = UitDsc2026PassageCleaner()
    parser = LegalStructureParser()
    parser_audit = audit_legal_markers_and_parser(args.contexts_path, cleaner, parser)
    with (args.output_dir / "parser_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(parser_audit, f, indent=2, ensure_ascii=False)

    # 4. Legal Chunks, Boundary Risks, Search Text & Metadata
    _LOGGER.info("Auditing legal chunks, boundary risks, and search_text...")
    chunks_audit = audit_legal_chunks(args.serving_root)
    with (args.output_dir / "chunk_census.json").open("w", encoding="utf-8") as f:
        json.dump(chunks_audit, f, indent=2, ensure_ascii=False)

    # 5. Train Q&A Deep Census & Linkability
    _LOGGER.info("Auditing official train Q&A and legal source linkability...")
    train_audit = audit_train_qa_and_linkability(
        args.train_path,
        args.public_path,
        args.contexts_path,
        args.serving_root,
    )
    with (args.output_dir / "train_qa_census.json").open("w", encoding="utf-8") as f:
        json.dump(train_audit, f, indent=2, ensure_ascii=False)

    # 6. Retrieval Proxy
    retrieval_proxy = {"status": "SKIPPED"}
    if args.run_retrieval_proxy:
        _LOGGER.info("Running read-only retrieval proxy evaluation...")
        loader = UitDsc2026DataLoader()
        train_q_list = loader.load_questions(args.train_path, require_reference_answers=True)
        retrieval_proxy = audit_retrieval_proxy(
            args.serving_root,
            train_audit["linkability"]["all_unambiguous_links"],
            train_q_list,
        )
        with (args.output_dir / "retrieval_proxy_report.json").open("w", encoding="utf-8") as f:
            json.dump(retrieval_proxy, f, indent=2, ensure_ascii=False)

    # 7. Summary Report
    summary = {
        "phase": "D0",
        "timestamp": datetime.now(UTC).isoformat(),
        "sources": sources_and_artifacts["sources"],
        "raw_corpus_headline": {
            "total_records": raw_corpus["total_records"],
            "non_empty": raw_corpus["non_empty_records"],
            "empty": raw_corpus["empty_records"],
            "with_title": raw_corpus["with_title_records"],
            "without_title": raw_corpus["without_title_records"],
            "exact_duplicate_records": raw_corpus["duplication"]["exact_duplicate_records"],
            "mean_char_length": raw_corpus["char_length_distribution"]["mean"],
        },
        "parser_headline": {
            "docs_parsed_article_ge_1": parser_audit["docs_parsed_article_ge_1"],
            "docs_zero_structure": parser_audit["docs_zero_structure"],
            "dieu_marker_agreement": parser_audit["dieu_marker_agreement"],
        },
        "chunks_headline": {
            "total_chunks": chunks_audit["total_chunks"],
            "unique_source_docs": chunks_audit["unique_source_documents"],
            "mean_chunk_tokens": chunks_audit["token_count_distribution"]["mean"],
            "token_fallback_pct": chunks_audit["strategy_distribution"].get("token_fallback", {}).get("percentage", 0),
            "boundary_risks_count": sum(chunks_audit["boundary_risk"]["risk_counts"].values()),
        },
        "train_headline": {
            "total_qa": train_audit["total_train_questions"],
            "unambiguous_doc_links": train_audit["linkability"]["unambiguous_doc_links_count"],
            "unambiguous_doc_links_pct": train_audit["linkability"]["unambiguous_doc_links_pct"],
            "unambiguous_article_links": train_audit["linkability"]["unambiguous_article_links_count"],
            "train_public_exact_overlap": train_audit["duplication"]["train_public_exact_overlap_count"],
        },
        "retrieval_proxy": retrieval_proxy,
    }
    with (args.output_dir / "d0_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 8. Evidence Packaging
    _LOGGER.info("Packaging D0 audit evidence into %s...", args.evidence_zip)
    args.evidence_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.evidence_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for json_file in args.output_dir.glob("*.json"):
            z.write(json_file, arcname=json_file.name)
            
    zip_size = args.evidence_zip.stat().st_size
    zip_sha = compute_file_sha256(args.evidence_zip)
    _LOGGER.info("Evidence ZIP successfully created: %s (%d bytes, SHA: %s)", args.evidence_zip, zip_size, zip_sha)
    print(f"D0_AUDIT_COMPLETE: evidence_zip={args.evidence_zip} sha256={zip_sha} size={zip_size}")


if __name__ == "__main__":
    main()
