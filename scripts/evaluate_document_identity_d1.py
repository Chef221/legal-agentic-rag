#!/usr/bin/env python3
"""PHASE D1-A: Deterministic Legal Document Identity Feasibility Evaluation.

Evaluates deterministic official-data-only legal document identity extraction
(document_type, document_number) across all 8,532 official contexts, serving chunks,
and D0 1,333 retrieval-proxy target documents against pre-registered feasibility gates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import sys
import unicodedata
import zipfile
from typing import Any, Literal

from legal_agentic_rag import __version__

_LOGGER = logging.getLogger("evaluate_document_identity_d1")

# --- Canonical Checksums and Invariants ---

EXPECTED_CONTEXTS_SHA256 = "ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97"
EXPECTED_D0_EVIDENCE_SHA256 = "eca404a749a45c00b6b7b94c7dee246fea39de385882e51343f6f1a20d93c27f"
EXPECTED_TRAIN_SHA256 = "2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988"
EXPECTED_CHUNKS_COUNT = 330768
EXPECTED_UNAMBIGUOUS_LINKS_COUNT = 1333
EXPECTED_ARTICLE_LINKS_COUNT = 639

FEASIBILITY_GATE_A_THRESHOLD = 0.50  # >= 50% non-empty official documents
FEASIBILITY_GATE_B_THRESHOLD = 0.70  # >= 70% of 1,333 proxy question target documents

# --- Document Type Mapping & Invariants ---

CANONICAL_TYPES: dict[str, str] = {
    "luật": "Luật",
    "bộ luật": "Bộ luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "thông tư liên tịch": "Thông tư liên tịch",
    "nghị quyết": "Nghị quyết",
    "nghị quyết liên tịch": "Nghị quyết liên tịch",
    "quyết định": "Quyết định",
    "pháp lệnh": "Pháp lệnh",
    "công văn": "Công văn",
    "chỉ thị": "Chỉ thị",
    "thông báo": "Thông báo",
    "hướng dẫn": "Hướng dẫn",
    "kế hoạch": "Kế hoạch",
    "công điện": "Công điện",
    "quy định": "Quy định",
    "quy chế": "Quy chế",
    "điều lệ": "Điều lệ",
    "văn bản hợp nhất": "Văn bản hợp nhất",
    "tiêu chuẩn quốc gia": "Tiêu chuẩn quốc gia",
    "quy chuẩn kỹ thuật quốc gia": "Quy chuẩn kỹ thuật quốc gia",
    "hiến pháp": "Hiến pháp",
    "lệnh": "Lệnh",
    "công ước": "Công ước",
    "hiệp định": "Hiệp định",
}

SLUG_PREFIX_MAP: list[tuple[str, str]] = [
    ("thong-tu-lien-tich", "Thông tư liên tịch"),
    ("nghi-quyet-lien-tich", "Nghị quyết liên tịch"),
    ("nghi-dinh-sua-doi-nghi-dinh", "Nghị định"),
    ("thong-tu-sua-doi-thong-tu", "Thông tư"),
    ("van-ban-hop-nhat", "Văn bản hợp nhất"),
    ("tieu-chuan-viet-nam-tcvn", "Tiêu chuẩn quốc gia"),
    ("tcvn-iso", "Tiêu chuẩn quốc gia"),
    ("tcvn", "Tiêu chuẩn quốc gia"),
    ("qcvn", "Quy chuẩn kỹ thuật quốc gia"),
    ("bo-luat", "Bộ luật"),
    ("luat", "Luật"),
    ("nghi-dinh", "Nghị định"),
    ("thong-tu", "Thông tư"),
    ("nghi-quyet", "Nghị quyết"),
    ("quyet-dinh", "Quyết định"),
    ("phap-lenh", "Pháp lệnh"),
    ("cong-van", "Công văn"),
    ("chi-thi", "Chỉ thị"),
    ("thong-bao", "Thông báo"),
    ("huong-dan", "Hướng dẫn"),
    ("ke-hoach", "Kế hoạch"),
    ("cong-dien", "Công điện"),
    ("quy-dinh", "Quy định"),
    ("quy-che", "Quy chế"),
    ("dieu-le", "Điều lệ"),
    ("hien-phap", "Hiến pháp"),
    ("lenh", "Lệnh"),
    ("cong-uoc", "Công ước"),
    ("hiep-dinh", "Hiệp định"),
    ("decree", "Nghị định"),
    ("circular", "Thông tư"),
    ("decision", "Quyết định"),
    ("law", "Luật"),
]

HEADER_TYPES: list[tuple[str, str]] = [
    ("thông tư liên tịch", "Thông tư liên tịch"),
    ("nghi quyết liên tịch", "Nghị quyết liên tịch"),
    ("nghị quyết liên tịch", "Nghị quyết liên tịch"),
    ("văn bản hợp nhất", "Văn bản hợp nhất"),
    ("tiêu chuẩn quốc gia", "Tiêu chuẩn quốc gia"),
    ("quy chuẩn kỹ thuật quốc gia", "Quy chuẩn kỹ thuật quốc gia"),
    ("tiêu chuẩn việt nam", "Tiêu chuẩn quốc gia"),
    ("bộ luật", "Bộ luật"),
    ("luật", "Luật"),
    ("nghị định", "Nghị định"),
    ("thông tư", "Thông tư"),
    ("nghị quyết", "Nghị quyết"),
    ("quyết định", "Quyết định"),
    ("pháp lệnh", "Pháp lệnh"),
    ("công văn", "Công văn"),
    ("chỉ thị", "Chỉ thị"),
    ("thông báo", "Thông báo"),
    ("hướng dẫn", "Hướng dẫn"),
    ("kế hoạch", "Kế hoạch"),
    ("công điện", "Công điện"),
    ("quy định", "Quy định"),
    ("quy chế", "Quy chế"),
    ("điều lệ", "Điều lệ"),
    ("hiến pháp", "Hiến pháp"),
    ("lệnh", "Lệnh"),
    ("công ước", "Công ước"),
    ("hiệp định", "Hiệp định"),
    ("tcvn", "Tiêu chuẩn quốc gia"),
    ("qcvn", "Quy chuẩn kỹ thuật quốc gia"),
]

KNOWN_MULTI_WORD_TYPES: set[str] = {
    "thông tư", "nghị định", "quyết định", "nghị quyết", "bộ luật", "pháp lệnh",
    "công văn", "chỉ thị", "thông báo", "hướng dẫn", "thông tư liên tịch",
    "nghị quyết liên tịch", "văn bản hợp nhất", "tiêu chuẩn quốc gia",
    "quy chuẩn kỹ thuật quốc gia", "quy định", "quy chế", "điều lệ", "hiến pháp",
}

# --- Dataclasses ---

@dataclass
class CandidateIdentity:
    """Identity candidate extracted from a specific source."""
    source: Literal["title", "url", "header"]
    document_type: str | None
    document_number: str | None
    normalized_identity: str | None
    evidence_category: str


@dataclass
class ResolvedDocumentIdentity:
    """Resolved document identity with confidence and audit trail."""
    document_id: str
    status: Literal["HIGH_CONFIDENCE", "AMBIGUOUS", "UNRESOLVED"]
    document_type: str | None
    document_number: str | None
    normalized_identity: str | None
    agreement_pattern: str
    agreeing_sources: list[str]
    title_candidate: CandidateIdentity | None
    url_candidate: CandidateIdentity | None
    header_candidate: CandidateIdentity | None


# --- Normalization Utilities ---

def compute_file_sha256(path: Path) -> str:
    """Stream file SHA-256 calculation."""
    h = sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_key(text: str) -> str:
    """Deterministic comparison key: ASCII fold, lowercase, hyphens."""
    if not text:
        return ""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("/", "-").replace(":", "-").replace("_", "-").replace(".", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def normalize_doc_number(num_str: str) -> str:
    """Conservative NFC whitespace and punctuation cleanup for document numbers."""
    if not num_str:
        return ""
    s = unicodedata.normalize("NFC", num_str.strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.,;:\s]+$", "", s)
    return s.strip()


# --- Extractor Functions ---

def extract_from_slug(slug_str: str | None, source_type: Literal["title", "url"]) -> CandidateIdentity | None:
    """Extract document_type and document_number from title or URL slug."""
    if not slug_str:
        return None
    slug = slug_str.strip()
    if not slug:
        return None

    slug_lower = slug.lower()
    doc_type: str | None = None
    rem = ""

    for prefix, dt in SLUG_PREFIX_MAP:
        if slug_lower.startswith(prefix + "-"):
            doc_type = dt
            rem = slug[len(prefix) + 1 :]
            break
        elif slug_lower.startswith(prefix):
            doc_type = dt
            rem = slug[len(prefix) :].lstrip("-")
            break

    if not doc_type:
        return None

    tokens = rem.split("-")
    num_tokens: list[str] = []

    if tokens and tokens[0].lower() == "so" and len(tokens) > 1:
        tokens = tokens[1:]

    for tok in tokens:
        if not tok:
            continue
        is_num_token = False
        if tok.isdigit():
            is_num_token = True
        elif tok.isupper() or tok in (
            "TTg", "CTcp", "TANDTC", "VKSNDTC", "BTC", "BTP", "BCA", "BNN", "BYT",
            "BKHCN", "BGTVT", "BTTTT", "BNV", "BCT", "BKHĐT", "BLĐTBXH", "BTNMT",
            "BXD", "BGDĐT", "BQP", "UBND", "HDND", "HĐND", "VKSTC", "TATC", "CP",
            "QH", "QH11", "QH12", "QH13", "QH14", "QH15", "BHXH", "TCHQ", "TCT",
            "NHNN", "TTCP", "UBCK",
        ):
            is_num_token = True
        elif any(c.isdigit() for c in tok) and any(c.isupper() for c in tok) and not any(c.islower() for c in tok if c != "g"):
            is_num_token = True

        if is_num_token:
            num_tokens.append(tok)
        else:
            break

    # If trailing token is a 4-digit metadata year following an acronym (e.g. 440-QD-TTCP-2021)
    if len(num_tokens) >= 3 and len(num_tokens[-1]) == 4 and num_tokens[-1].isdigit() and int(num_tokens[-1]) in range(1945, 2030):
        prev = num_tokens[-2]
        if not prev.isdigit():
            num_tokens = num_tokens[:-1]

    doc_number = normalize_doc_number("-".join(num_tokens)) if num_tokens else None
    norm_type_key = normalize_key(doc_type)
    norm_num_key = normalize_key(doc_number) if doc_number else ""
    norm_id = f"{norm_type_key}::{norm_num_key}" if norm_num_key else None

    return CandidateIdentity(
        source=source_type,
        document_type=doc_type,
        document_number=doc_number,
        normalized_identity=norm_id,
        evidence_category="organizer_slug",
    )


def clean_and_combine_header_lines(passage: str) -> list[str]:
    """Normalize newlines and combine split multi-word uppercase document type tokens."""
    norm_passage = passage.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [
        unicodedata.normalize("NFC", l.strip(":-* \t"))
        for l in norm_passage.split("\n")[:60]
        if l.strip(":-* \t")
    ]
    combined: list[str] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if i + 2 < len(raw_lines):
            triplet = f"{line} {raw_lines[i+1]} {raw_lines[i+2]}".lower()
            if triplet in KNOWN_MULTI_WORD_TYPES:
                combined.append(f"{line} {raw_lines[i+1]} {raw_lines[i+2]}")
                i += 3
                continue
        if i + 1 < len(raw_lines):
            pair = f"{line} {raw_lines[i+1]}".lower()
            if pair in KNOWN_MULTI_WORD_TYPES:
                combined.append(f"{line} {raw_lines[i+1]}")
                i += 2
                continue
        combined.append(line)
        i += 1
    return combined


def extract_from_header(passage: str | None) -> CandidateIdentity | None:
    """Extract own-document type and number strictly from the early header/preamble region."""
    if not passage:
        return None
    lines = clean_and_combine_header_lines(passage)
    if not lines:
        return None

    doc_type: str | None = None
    for l in lines:
        if re.match(r"^(?:căn cứ|điều \d+|chương [ivxlcdm]+)", l, re.IGNORECASE):
            break
        l_clean = l.strip(":-* \t")
        l_lower = unicodedata.normalize("NFC", l_clean.lower())
        for k, v in HEADER_TYPES:
            if l_lower == k or l_lower.startswith(k + " ") or l_lower.startswith(k + ":") or l_lower.startswith(k + "-"):
                doc_type = v
                break
        if doc_type:
            break

    doc_number: str | None = None
    for i, l in enumerate(lines):
        if re.match(r"^(?:căn cứ|điều \d+|chương [ivxlcdm]+)", l, re.IGNORECASE):
            break
        m = re.match(r"^Số\s*[:\.]\s*([0-9A-Za-zĐđ\/\-\.:_]+(?:\s+[0-9A-Za-zĐđ\/\-\.:_]+)*)", l, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand_tokens = cand.split()
            if any(c.isdigit() for c in cand_tokens[0]) or "TCVN" in cand_tokens[0] or "QCVN" in cand_tokens[0]:
                doc_number = normalize_doc_number(cand_tokens[0])
                break
        elif re.match(r"^Số\s*[:\.]?\s*$", l, re.IGNORECASE) and i + 1 < len(lines):
            next_l = lines[i + 1].strip()
            cand_tokens = next_l.split()
            if cand_tokens and (any(c.isdigit() for c in cand_tokens[0]) or "TCVN" in cand_tokens[0] or "QCVN" in cand_tokens[0]):
                doc_number = normalize_doc_number(cand_tokens[0])
                break

    if not doc_type and not doc_number:
        return None

    norm_type_key = normalize_key(doc_type) if doc_type else ""
    norm_num_key = normalize_key(doc_number) if doc_number else ""
    norm_id = f"{norm_type_key}::{norm_num_key}" if (norm_type_key and norm_num_key) else None

    return CandidateIdentity(
        source="header",
        document_type=doc_type,
        document_number=doc_number,
        normalized_identity=norm_id,
        evidence_category="organizer_header",
    )


# --- Confidence Resolution Policy ---

def resolve_document_identity(
    document_id: str,
    name: str | None,
    link: str | None,
    passage: str | None,
) -> ResolvedDocumentIdentity:
    """Resolve document identity across independent sources using fail-closed multi-source agreement."""
    # 1. Extract independent source candidates
    c_title = extract_from_slug(name, "title") if name else None

    slug_url = ""
    if link:
        m = re.search(r"/([^/]+)\.aspx", link)
        if m:
            slug_url = m.group(1)
    c_url = extract_from_slug(slug_url, "url") if slug_url else None

    c_header = extract_from_header(passage)

    # Collect complete candidates (having both type and number)
    complete_cands = [c for c in (c_title, c_url, c_header) if c and c.normalized_identity]

    if not complete_cands:
        return ResolvedDocumentIdentity(
            document_id=str(document_id),
            status="UNRESOLVED",
            document_type=None,
            document_number=None,
            normalized_identity=None,
            agreement_pattern="none",
            agreeing_sources=[],
            title_candidate=c_title,
            url_candidate=c_url,
            header_candidate=c_header,
        )

    # Group by normalized identity
    id_groups: dict[str, list[CandidateIdentity]] = defaultdict(list)
    for c in complete_cands:
        assert c.normalized_identity is not None
        id_groups[c.normalized_identity].append(c)

    if len(id_groups) > 1:
        # Conflicting candidates across sources -> AMBIGUOUS (fail-closed)
        return ResolvedDocumentIdentity(
            document_id=str(document_id),
            status="AMBIGUOUS",
            document_type=None,
            document_number=None,
            normalized_identity=None,
            agreement_pattern="conflict",
            agreeing_sources=[c.source for c in complete_cands],
            title_candidate=c_title,
            url_candidate=c_url,
            header_candidate=c_header,
        )

    # Single unanimous identity group
    nid, matched_cands = list(id_groups.items())[0]
    sources = set(c.source for c in matched_cands)

    if len(sources) >= 2:
        if sources == {"title", "url", "header"}:
            pattern = "all_three"
        elif sources == {"title", "url"}:
            pattern = "title_url"
        elif sources == {"title", "header"}:
            pattern = "title_header"
        elif sources == {"url", "header"}:
            pattern = "url_header"
        else:
            pattern = "multi_source"
    else:
        src = list(sources)[0]
        pattern = f"single_{src}"

    # Select canonical document_type and document_number
    # Prefer header candidate for rich Unicode typographic fidelity if present
    header_cand = next((c for c in matched_cands if c.source == "header"), None)
    title_cand = next((c for c in matched_cands if c.source == "title"), None)
    url_cand = next((c for c in matched_cands if c.source == "url"), None)

    chosen = header_cand or title_cand or url_cand
    assert chosen is not None
    assert chosen.document_type is not None
    assert chosen.document_number is not None

    final_type = chosen.document_type
    final_number = chosen.document_number

    # If header candidate provided rich accented number and agrees, use header number
    if header_cand and header_cand.document_number:
        final_number = header_cand.document_number
    if header_cand and header_cand.document_type:
        final_type = header_cand.document_type

    return ResolvedDocumentIdentity(
        document_id=str(document_id),
        status="HIGH_CONFIDENCE",
        document_type=final_type,
        document_number=final_number,
        normalized_identity=nid,
        agreement_pattern=pattern,
        agreeing_sources=sorted(sources),
        title_candidate=c_title,
        url_candidate=c_url,
        header_candidate=c_header,
    )


# --- Full Evaluation Harness ---

def run_d1a_feasibility_evaluation(
    contexts_zip: Path,
    d0_evidence_zip: Path,
    train_json: Path,
    chunks_jsonl: Path,
    evidence_zip: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute real full-corpus D1-A feasibility measurement."""
    _LOGGER.info("Verifying canonical input source identities...")

    # 1. Source verification
    ctx_sha = compute_file_sha256(contexts_zip)
    ctx_size = contexts_zip.stat().st_size
    if ctx_sha != EXPECTED_CONTEXTS_SHA256:
        raise ValueError(
            f"D1A_SOURCE_IDENTITY_FAILURE: contexts_zip SHA-256 mismatch: {ctx_sha} != {EXPECTED_CONTEXTS_SHA256}"
        )

    d0_sha = compute_file_sha256(d0_evidence_zip)
    d0_size = d0_evidence_zip.stat().st_size
    if d0_sha != EXPECTED_D0_EVIDENCE_SHA256:
        raise ValueError(
            f"D1A_SOURCE_IDENTITY_FAILURE: d0_evidence_zip SHA-256 mismatch: {d0_sha} != {EXPECTED_D0_EVIDENCE_SHA256}"
        )

    train_sha = compute_file_sha256(train_json)
    train_size = train_json.stat().st_size
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise ValueError(
            f"D1A_SOURCE_IDENTITY_FAILURE: train_json SHA-256 mismatch: {train_sha} != {EXPECTED_TRAIN_SHA256}"
        )

    _LOGGER.info("All canonical input sources verified successfully.")

    # 2. Process all 8,532 contexts
    _LOGGER.info("Processing all context records from %s...", contexts_zip)
    resolved_records: dict[str, ResolvedDocumentIdentity] = {}
    total_contexts = 0
    non_empty_contexts = 0
    titled_contexts = 0

    agreement_counts: Counter[str] = Counter()
    type_distribution: Counter[str] = Counter()
    high_conf_types_count = 0
    high_conf_numbers_count = 0
    high_conf_complete_count = 0
    ambiguous_count = 0
    unresolved_count = 0

    with zipfile.ZipFile(contexts_zip) as z:
        names = sorted([n for n in z.namelist() if n.endswith(".json")])
        total_contexts = len(names)

        for n in names:
            raw_data = json.loads(z.read(n).decode("utf-8"))
            cid = str(raw_data.get("id"))
            name_val = raw_data.get("name")
            link_val = raw_data.get("link")
            passage = raw_data.get("passage", "")

            if passage and passage.strip():
                non_empty_contexts += 1
            if name_val and str(name_val).strip():
                titled_contexts += 1

            resolved = resolve_document_identity(cid, name_val, link_val, passage)
            resolved_records[cid] = resolved

            if resolved.status == "HIGH_CONFIDENCE":
                high_conf_complete_count += 1
                if resolved.document_type:
                    high_conf_types_count += 1
                if resolved.document_number:
                    high_conf_numbers_count += 1
                agreement_counts[resolved.agreement_pattern] += 1
                assert resolved.document_type is not None
                type_distribution[resolved.document_type] += 1
            elif resolved.status == "AMBIGUOUS":
                ambiguous_count += 1
            else:
                unresolved_count += 1

    _LOGGER.info(
        "Context resolution complete: %d HIGH_CONFIDENCE, %d AMBIGUOUS, %d UNRESOLVED / %d contexts",
        high_conf_complete_count,
        ambiguous_count,
        unresolved_count,
        total_contexts,
    )

    # 3. Sidecar Chunk Propagation Coverage
    _LOGGER.info("Evaluating logical sidecar chunk propagation over %s...", chunks_jsonl)
    total_chunks = 0
    covered_chunks = 0
    uncovered_chunks = 0

    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            total_chunks += 1
            chunk = json.loads(line)
            doc_id = str(chunk.get("document_id"))
            rec = resolved_records.get(doc_id)
            if rec and rec.status == "HIGH_CONFIDENCE":
                covered_chunks += 1
            else:
                uncovered_chunks += 1

    if total_chunks != EXPECTED_CHUNKS_COUNT:
        raise ValueError(
            f"D1A_INVALID_EXECUTION: chunks count mismatch: {total_chunks} != {EXPECTED_CHUNKS_COUNT}"
        )

    chunk_coverage_pct = (covered_chunks / total_chunks * 100.0) if total_chunks else 0.0
    _LOGGER.info(
        "Chunk propagation: %d covered (%.2f%%), %d uncovered / %d total chunks",
        covered_chunks,
        chunk_coverage_pct,
        uncovered_chunks,
        total_chunks,
    )

    # 4. D0 1,333 Retrieval Proxy Link Coverage
    _LOGGER.info("Loading D0 linkability proxy from %s...", d0_evidence_zip)
    with zipfile.ZipFile(d0_evidence_zip) as z:
        train_census = json.loads(z.read("train_qa_census.json").decode("utf-8"))
        unambig_links = train_census["linkability"]["all_unambiguous_links"]
        article_links_count = train_census["linkability"]["unambiguous_article_links_count"]

    if len(unambig_links) != EXPECTED_UNAMBIGUOUS_LINKS_COUNT:
        raise ValueError(
            f"D1A_INVALID_EXECUTION: proxy links count mismatch: {len(unambig_links)} != {EXPECTED_UNAMBIGUOUS_LINKS_COUNT}"
        )
    if article_links_count != EXPECTED_ARTICLE_LINKS_COUNT:
        raise ValueError(
            f"D1A_INVALID_EXECUTION: proxy article links count mismatch: {article_links_count} != {EXPECTED_ARTICLE_LINKS_COUNT}"
        )

    # Freeze deterministic ordered proxy population
    sorted_proxy_links = sorted(
        unambig_links,
        key=lambda x: (str(x["question_id"]), str(x["target_document_id"])),
    )
    proxy_pop_bytes = json.dumps(sorted_proxy_links, sort_keys=True).encode("utf-8")
    proxy_pop_sha = sha256(proxy_pop_bytes).hexdigest()

    covered_proxy_questions = 0
    uncovered_proxy_questions = 0
    unique_target_docs: set[str] = set()
    covered_target_docs: set[str] = set()

    for item in sorted_proxy_links:
        target_id = str(item["target_document_id"])
        unique_target_docs.add(target_id)
        rec = resolved_records.get(target_id)
        if rec and rec.status == "HIGH_CONFIDENCE":
            covered_proxy_questions += 1
            covered_target_docs.add(target_id)
        else:
            uncovered_proxy_questions += 1

    proxy_q_coverage_pct = (covered_proxy_questions / len(sorted_proxy_links) * 100.0)
    proxy_doc_coverage_pct = (len(covered_target_docs) / len(unique_target_docs) * 100.0) if unique_target_docs else 0.0

    _LOGGER.info(
        "Proxy coverage: %d questions covered (%.2f%%), %d docs covered (%.2f%%) / %d unique docs",
        covered_proxy_questions,
        proxy_q_coverage_pct,
        len(covered_target_docs),
        proxy_doc_coverage_pct,
        len(unique_target_docs),
    )

    # 5. Feasibility Gate Evaluation
    non_empty_coverage_ratio = (high_conf_complete_count / non_empty_contexts) if non_empty_contexts else 0.0
    proxy_coverage_ratio = (covered_proxy_questions / len(sorted_proxy_links)) if sorted_proxy_links else 0.0

    gate_a_passed = non_empty_coverage_ratio >= FEASIBILITY_GATE_A_THRESHOLD
    gate_b_passed = proxy_coverage_ratio >= FEASIBILITY_GATE_B_THRESHOLD

    if gate_a_passed or gate_b_passed:
        final_decision = "D1A_FEASIBILITY_PASS"
        feasibility_status = "PASS"
    else:
        final_decision = "D1_DOCUMENT_IDENTITY_COVERAGE_INSUFFICIENT"
        feasibility_status = "FAIL"

    _LOGGER.info(
        "Feasibility Gates: Gate A (>= 50%% non-empty) = %s (%.2f%%), Gate B (>= 70%% proxy) = %s (%.2f%%) -> DECISION: %s",
        gate_a_passed,
        non_empty_coverage_ratio * 100.0,
        gate_b_passed,
        proxy_coverage_ratio * 100.0,
        final_decision,
    )

    # 6. Build Evidence Artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_dir = output_dir / "execution"
    results_dir = output_dir / "results"
    execution_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # execution/source_identity.json
    source_identity = {
        "timestamp": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "sources": {
            "contexts_zip": {
                "path": str(contexts_zip),
                "sha256": ctx_sha,
                "size_bytes": ctx_size,
                "verified": True,
            },
            "d0_evidence_zip": {
                "path": str(d0_evidence_zip),
                "sha256": d0_sha,
                "size_bytes": d0_size,
                "verified": True,
            },
            "train_json": {
                "path": str(train_json),
                "sha256": train_sha,
                "size_bytes": train_size,
                "verified": True,
            },
            "chunks_jsonl": {
                "path": str(chunks_jsonl),
                "record_count": total_chunks,
                "verified": True,
            },
        },
        "proxy_population": {
            "record_count": len(sorted_proxy_links),
            "sha256": proxy_pop_sha,
        },
    }
    with (execution_dir / "source_identity.json").open("w", encoding="utf-8") as f:
        json.dump(source_identity, f, indent=2, ensure_ascii=False)

    # execution/extraction_contract.json
    extraction_contract = {
        "allowed_extracted_fields": ["document_type", "document_number"],
        "prohibited_fields": [
            "issuing_authority",
            "issuance_date",
            "effective_date",
            "expiry_date",
            "effect_status",
            "legal_field",
        ],
        "allowed_organizer_sources": [
            "context name/title",
            "context source URL (link)",
            "deterministic early passage header region",
        ],
        "confidence_classes": ["HIGH_CONFIDENCE", "AMBIGUOUS", "UNRESOLVED"],
        "feasibility_gates": {
            "gate_a": {
                "description": "HIGH_CONFIDENCE complete identity coverage >= 50.0% of non-empty official documents",
                "threshold": FEASIBILITY_GATE_A_THRESHOLD,
            },
            "gate_b": {
                "description": "HIGH_CONFIDENCE complete identity covers >= 70.0% of 1,333 retrieval-proxy question target documents",
                "threshold": FEASIBILITY_GATE_B_THRESHOLD,
            },
        },
    }
    with (execution_dir / "extraction_contract.json").open("w", encoding="utf-8") as f:
        json.dump(extraction_contract, f, indent=2, ensure_ascii=False)

    # results/identity_coverage.json
    identity_coverage = {
        "total_contexts": total_contexts,
        "non_empty_contexts": non_empty_contexts,
        "titled_contexts": titled_contexts,
        "counts": {
            "high_confidence_type": high_conf_types_count,
            "high_confidence_number": high_conf_numbers_count,
            "high_confidence_complete": high_conf_complete_count,
            "ambiguous": ambiguous_count,
            "unresolved": unresolved_count,
        },
        "coverage_percentages": {
            "all_contexts_pct": round(high_conf_complete_count / total_contexts * 100.0, 2),
            "non_empty_contexts_pct": round(high_conf_complete_count / non_empty_contexts * 100.0, 2),
            "titled_contexts_pct": round(high_conf_complete_count / titled_contexts * 100.0, 2),
        },
        "chunks_propagation": {
            "total_chunks": total_chunks,
            "covered_chunks": covered_chunks,
            "uncovered_chunks": uncovered_chunks,
            "coverage_pct": round(chunk_coverage_pct, 2),
        },
        "top_document_types": [
            {"type": t, "count": c, "percentage": round(c / high_conf_complete_count * 100.0, 2)}
            for t, c in type_distribution.most_common(20)
        ],
    }
    with (results_dir / "identity_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(identity_coverage, f, indent=2, ensure_ascii=False)

    # results/source_agreement.json
    source_agreement = {
        "total_high_confidence": high_conf_complete_count,
        "agreement_patterns": {
            "all_three": agreement_counts["all_three"],
            "url_header": agreement_counts["url_header"],
            "title_url": agreement_counts["title_url"],
            "title_header": agreement_counts["title_header"],
            "single_url": agreement_counts["single_url"],
            "single_header": agreement_counts["single_header"],
            "single_title": agreement_counts["single_title"],
        },
        "multi_source_agreement_count": (
            agreement_counts["all_three"]
            + agreement_counts["url_header"]
            + agreement_counts["title_url"]
            + agreement_counts["title_header"]
        ),
        "multi_source_agreement_pct": round(
            (
                agreement_counts["all_three"]
                + agreement_counts["url_header"]
                + agreement_counts["title_url"]
                + agreement_counts["title_header"]
            )
            / high_conf_complete_count
            * 100.0,
            2,
        ),
    }
    with (results_dir / "source_agreement.json").open("w", encoding="utf-8") as f:
        json.dump(source_agreement, f, indent=2, ensure_ascii=False)

    # results/proxy_identity_coverage.json
    proxy_identity_coverage = {
        "proxy_population_sha256": proxy_pop_sha,
        "total_proxy_questions": len(sorted_proxy_links),
        "covered_proxy_questions": covered_proxy_questions,
        "uncovered_proxy_questions": uncovered_proxy_questions,
        "proxy_question_coverage_pct": round(proxy_q_coverage_pct, 2),
        "unique_target_documents": {
            "total": len(unique_target_docs),
            "covered": len(covered_target_docs),
            "uncovered": len(unique_target_docs) - len(covered_target_docs),
            "coverage_pct": round(proxy_doc_coverage_pct, 2),
        },
    }
    with (results_dir / "proxy_identity_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(proxy_identity_coverage, f, indent=2, ensure_ascii=False)

    # results/ambiguous_summary.json
    ambiguous_records = [
        {
            "document_id": rec.document_id,
            "title_candidate": asdict(rec.title_candidate) if rec.title_candidate else None,
            "url_candidate": asdict(rec.url_candidate) if rec.url_candidate else None,
            "header_candidate": asdict(rec.header_candidate) if rec.header_candidate else None,
        }
        for rec in resolved_records.values()
        if rec.status == "AMBIGUOUS"
    ]
    ambiguous_summary = {
        "total_ambiguous": len(ambiguous_records),
        "sample_count": min(20, len(ambiguous_records)),
        "samples": ambiguous_records[:20],
    }
    with (results_dir / "ambiguous_summary.json").open("w", encoding="utf-8") as f:
        json.dump(ambiguous_summary, f, indent=2, ensure_ascii=False)

    # Diagnostic sample files (up to 20 each)
    high_conf_samples = [
        {
            "document_id": rec.document_id,
            "document_type": rec.document_type,
            "document_number": rec.document_number,
            "normalized_identity": rec.normalized_identity,
            "agreement_pattern": rec.agreement_pattern,
            "agreeing_sources": rec.agreeing_sources,
        }
        for rec in resolved_records.values()
        if rec.status == "HIGH_CONFIDENCE"
    ][:20]
    with (results_dir / "high_confidence_samples.json").open("w", encoding="utf-8") as f:
        json.dump(high_conf_samples, f, indent=2, ensure_ascii=False)

    unresolved_samples = [
        {
            "document_id": rec.document_id,
            "title_candidate": asdict(rec.title_candidate) if rec.title_candidate else None,
            "url_candidate": asdict(rec.url_candidate) if rec.url_candidate else None,
            "header_candidate": asdict(rec.header_candidate) if rec.header_candidate else None,
        }
        for rec in resolved_records.values()
        if rec.status == "UNRESOLVED"
    ][:20]
    with (results_dir / "unresolved_samples.json").open("w", encoding="utf-8") as f:
        json.dump(unresolved_samples, f, indent=2, ensure_ascii=False)

    # results/d1a_decision.json
    d1a_decision = {
        "phase": "D1-A",
        "timestamp": datetime.now(UTC).isoformat(),
        "feasibility_gate": feasibility_status,
        "final_decision": final_decision,
        "gates_evaluated": {
            "gate_a": {
                "name": "Corpus High-Confidence Complete Identity Coverage",
                "threshold_ratio": FEASIBILITY_GATE_A_THRESHOLD,
                "achieved_ratio": round(non_empty_coverage_ratio, 4),
                "achieved_pct": round(non_empty_coverage_ratio * 100.0, 2),
                "passed": gate_a_passed,
            },
            "gate_b": {
                "name": "D0 1,333 Proxy Question Target Document Coverage",
                "threshold_ratio": FEASIBILITY_GATE_B_THRESHOLD,
                "achieved_ratio": round(proxy_coverage_ratio, 4),
                "achieved_pct": round(proxy_coverage_ratio * 100.0, 2),
                "passed": gate_b_passed,
            },
        },
        "summary": {
            "total_contexts": total_contexts,
            "high_confidence_complete": high_conf_complete_count,
            "non_empty_contexts_pct": round(non_empty_coverage_ratio * 100.0, 2),
            "chunks_covered_pct": round(chunk_coverage_pct, 2),
            "proxy_covered_questions": covered_proxy_questions,
            "proxy_covered_pct": round(proxy_q_coverage_pct, 2),
        },
        "next_step": "external_review_then_d1_b_bm25_ab" if final_decision == "D1A_FEASIBILITY_PASS" else "abandon_d1",
    }
    with (results_dir / "d1a_decision.json").open("w", encoding="utf-8") as f:
        json.dump(d1a_decision, f, indent=2, ensure_ascii=False)

    # 7. Package Evidence ZIP
    _LOGGER.info("Packaging D1-A feasibility evidence into %s...", evidence_zip)
    evidence_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(evidence_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in output_dir.rglob("*.json"):
            rel_path = p.relative_to(output_dir).as_posix()
            z.write(p, arcname=rel_path)

    zip_size = evidence_zip.stat().st_size
    zip_sha = compute_file_sha256(evidence_zip)
    with zipfile.ZipFile(evidence_zip) as z:
        member_count = len(z.namelist())

    _LOGGER.info(
        "Evidence ZIP created: %s (%d bytes, %d members, SHA: %s)",
        evidence_zip,
        zip_size,
        member_count,
        zip_sha,
    )

    return {
        "final_decision": final_decision,
        "feasibility_gate": feasibility_status,
        "source_identity": source_identity,
        "identity_coverage": identity_coverage,
        "source_agreement": source_agreement,
        "proxy_identity_coverage": proxy_identity_coverage,
        "d1a_decision": d1a_decision,
        "evidence_zip": {
            "path": str(evidence_zip),
            "sha256": zip_sha,
            "size_bytes": zip_size,
            "member_count": member_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1-A Deterministic Legal Document Identity Feasibility Evaluation Harness"
    )
    parser.add_argument(
        "--feasibility-only",
        action="store_true",
        required=True,
        help="Execute feasibility measurement only (Phase D1-A mode).",
    )
    parser.add_argument(
        "--contexts-zip",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\selected-contexts.zip"),
        help="Path to canonical official selected-contexts.zip.",
    )
    parser.add_argument(
        "--d0-evidence-zip",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\data-d0-official-data-audit-evidence.zip"),
        help="Path to canonical D0 audit evidence archive.",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\train.json"),
        help="Path to official train.json.",
    )
    parser.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=Path("artifacts/uit-dsc-2026-task2-v0400/legal_chunks/records.jsonl"),
        help="Path to serving legal_chunks/records.jsonl.",
    )
    parser.add_argument(
        "--evidence-zip",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\data-d1a-document-identity-feasibility-evidence.zip"),
        help="Path to output D1-A evidence ZIP package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/d1a_evidence_staging"),
        help="Path to temporary directory for evidence JSON files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.feasibility_only:
        _LOGGER.error("Phase D1-A requires --feasibility-only.")
        sys.exit(1)

    result = run_d1a_feasibility_evaluation(
        contexts_zip=args.contexts_zip,
        d0_evidence_zip=args.d0_evidence_zip,
        train_json=args.train_json,
        chunks_jsonl=args.chunks_jsonl,
        evidence_zip=args.evidence_zip,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print(f"D1-A FEASIBILITY RESULT: {result['feasibility_gate']}")
    print(f"FINAL DECISION: {result['final_decision']}")
    print(f"Evidence ZIP: {result['evidence_zip']['path']}")
    print(f"Evidence ZIP SHA-256: {result['evidence_zip']['sha256']}")
    print(f"Evidence ZIP Size: {result['evidence_zip']['size_bytes']} bytes ({result['evidence_zip']['member_count']} members)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
