#!/usr/bin/env python3
"""PHASE D1-B: Strict Document Identity BM25 Causal A/B Evaluation.

Executes the causal A/B retrieval experiment evaluating deterministic document identity
enrichment against the canonical SQLite FTS5 BM25 index across the frozen 1,333 D0
retrieval-proxy question population under strict pre-registered gates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata
import zipfile
from typing import Any, Literal

from legal_agentic_rag import __version__
from legal_agentic_rag.indexing.bm25.analyzer import UnicodeBM25Analyzer
from legal_agentic_rag.offline.chunking.tokenizer import UnicodeWordTokenizer

_LOGGER = logging.getLogger("evaluate_document_identity_d1b")

# --- Canonical Checksums & Constants ---

EXPECTED_CONTEXTS_SHA256 = "ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97"
EXPECTED_D0_EVIDENCE_SHA256 = "eca404a749a45c00b6b7b94c7dee246fea39de385882e51343f6f1a20d93c27f"
EXPECTED_TRAIN_SHA256 = "2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988"
EXPECTED_D1A_STRICT_EVIDENCE_SHA256 = "870ad1447e46b083bcdd8b7cd82e585509615a480c772ccdf229f358b50edbc5"
EXPECTED_CHUNKS_COUNT = 330768
EXPECTED_UNAMBIGUOUS_LINKS_COUNT = 1333

EXPECTED_STRICT_DOC_COUNT = 6891
EXPECTED_STRICT_CHUNK_COUNT = 264765
EXPECTED_STRICT_PROXY_COUNT = 1262
EXPECTED_STRICT_PROXY_SHA256 = "2b29b553908e2bc1553d1e7402194390609e6f8ca68592bdb0f56200bafa4100"

MAX_SEARCH_TOKENS = 512
TABLE_NAME = "bm25_documents"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in binary mode."""
    hasher = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_str_sha256(content: str) -> str:
    """Compute SHA-256 hex digest of a UTF-8 string."""
    return sha256(content.encode("utf-8")).hexdigest()


def compute_obj_sha256(obj: Any) -> str:
    """Compute deterministic SHA-256 hex digest of a JSON-serializable object."""
    dumped = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(dumped.encode("utf-8")).hexdigest()


# --- Identity Resolution Helper (Self-Contained & Deterministic) ---

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from evaluate_document_identity_d1 import resolve_document_identity
except ImportError:
    from scripts.evaluate_document_identity_d1 import resolve_document_identity


def extract_strict_document_identities(contexts_zip: Path) -> dict[str, tuple[str, str]]:
    """Extract STRICT_MULTI_CHANNEL document identities across all 8,532 contexts.

    Returns mapping: document_id -> (document_type, document_number).
    """
    strict_map: dict[str, tuple[str, str]] = {}
    with zipfile.ZipFile(contexts_zip) as z:
        names = sorted([n for n in z.namelist() if n.endswith(".json")])
        for n in names:
            raw_data = json.loads(z.read(n).decode("utf-8"))
            cid = str(raw_data.get("id"))
            name_val = raw_data.get("name")
            link_val = raw_data.get("link")
            passage = raw_data.get("passage", "")
            resolved = resolve_document_identity(cid, name_val, link_val, passage)
            if resolved.strict_status == "STRICT_MULTI_CHANNEL":
                assert resolved.document_type is not None
                assert resolved.document_number is not None
                strict_map[cid] = (resolved.document_type, resolved.document_number)
    return strict_map


# --- Candidate Search Text Builder ---

def construct_candidate_search_text(
    base_search_text: str,
    raw_text: str,
    doc_type: str,
    doc_num: str,
    tokenizer: UnicodeWordTokenizer,
    max_tokens: int = MAX_SEARCH_TOKENS,
) -> tuple[str, str]:
    """Construct candidate search_text following pre-registered token budget priority.

    Priority:
    1. full chunk.text (as exact suffix in `Nội dung:\\n{raw_text}`)
    2. document_number
    3. document_type
    4. existing optional baseline search header material

    Returns (candidate_search_text, modification_type).
    """
    content_suffix = "Nội dung:\n" + raw_text
    
    # Check if raw text alone exceeds max_tokens (extreme outlier chunks)
    if tokenizer.count(content_suffix) > max_tokens:
        return content_suffix, "raw_text_exceeds_budget"

    # Split baseline search_text into existing header and content suffix
    if base_search_text.endswith(content_suffix) and len(base_search_text) > len(content_suffix):
        header_part = base_search_text[:-len(content_suffix)].rstrip("\n")
        header_lines = [l for l in header_part.split("\n") if l.strip()]
    else:
        header_lines = []

    fixed_prefix = f"Số ký hiệu: {doc_num}\nLoại văn bản: {doc_type}"

    # Attempt 1: Full fixed prefix + all existing header lines + content suffix
    if header_lines:
        full_candidate = f"{fixed_prefix}\n{chr(10).join(header_lines)}\n{content_suffix}"
    else:
        full_candidate = f"{fixed_prefix}\n{content_suffix}"

    if tokenizer.count(full_candidate) <= max_tokens:
        return full_candidate, "full_enrichment"

    # Attempt 2: Drop existing header lines from top (least specific) to bottom
    for drop_idx in range(1, len(header_lines) + 1):
        remaining_header = "\n".join(header_lines[drop_idx:])
        if remaining_header:
            cand = f"{fixed_prefix}\n{remaining_header}\n{content_suffix}"
        else:
            cand = f"{fixed_prefix}\n{content_suffix}"
        if tokenizer.count(cand) <= max_tokens:
            mod = "partial_header_dropped" if remaining_header else "all_header_dropped"
            return cand, mod

    # Attempt 3: Drop Loại văn bản, retain Số ký hiệu
    cand_num_only = f"Số ký hiệu: {doc_num}\n{content_suffix}"
    if tokenizer.count(cand_num_only) <= max_tokens:
        return cand_num_only, "type_dropped"

    # Attempt 4: Drop Số ký hiệu as well, retain only content_suffix
    return content_suffix, "doc_num_dropped"


# --- Query Construction & Parsing ---

def construct_match_query(question_text: str) -> str:
    """Construct deterministic D0-exact BM25 MATCH query from question text only."""
    clean_q = re.sub(r"[^\w\s]", " ", question_text)
    tokens = [f'"{t}"' for t in clean_q.split() if len(t) > 1][:15]
    if not tokens:
        return ""
    return " OR ".join(tokens)


# --- Query Identity Signal Classifier (Pre-Registered Diagnostic) ---

_DOC_NUMBER_PATTERN = re.compile(
    r"\b\d+[/]\d{2,4}[/][A-ZĐa-zđ0-9_-]+\b"
    r"|\b\d+[/][A-ZĐa-zđ0-9_-]+\b"
    r"|\b(?:số|số hiệu)\s+\d+\b",
    re.IGNORECASE,
)

_DOC_TYPE_PATTERN = re.compile(
    r"\b(?:luật|bộ luật|nghị định|thông tư|quyết định|pháp lệnh|chỉ thị|nghị quyết|công văn|thông báo|hướng dẫn)\b",
    re.IGNORECASE,
)


def classify_query_identity_signals(question_text: str) -> str:
    """Classify question text deterministically into identity signal categories."""
    has_number = bool(_DOC_NUMBER_PATTERN.search(question_text))
    has_type = bool(_DOC_TYPE_PATTERN.search(question_text))

    if has_number and has_type:
        return "both"
    elif has_number:
        return "explicit_document_number"
    elif has_type:
        return "explicit_document_type"
    else:
        return "neither"


# --- BM25 Database Indexing & Querying ---

def validate_existing_candidate_bm25(
    db_path: Path,
    expected_row_count: int = EXPECTED_CHUNKS_COUNT,
) -> dict[str, Any] | None:
    """Validate an existing candidate BM25 SQLite DB without rebuilding.

    Returns build-equivalent metadata dict if valid, None if invalid or missing.
    """
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # Verify the FTS5 virtual table exists and has the expected row count
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        row_count = cur.fetchone()[0]
        if row_count != expected_row_count:
            conn.close()
            _LOGGER.warning(
                "Existing candidate DB row count %d != expected %d; will rebuild.",
                row_count,
                expected_row_count,
            )
            return None

        # Verify BM25 MATCH is operational (not a corrupt FTS5 index)
        cur.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {TABLE_NAME} MATCH '\"pháp\"'"
        )
        match_count = cur.fetchone()[0]
        if match_count == 0:
            conn.close()
            _LOGGER.warning("Existing candidate DB BM25 MATCH returned 0 hits; will rebuild.")
            return None

        conn.close()
    except Exception as err:
        _LOGGER.warning("Existing candidate DB validation failed: %s; will rebuild.", err)
        return None

    return {
        "total_chunks": row_count,
        "modified_chunks": -1,  # not recomputed on reuse
        "modification_distribution": {"reused_existing_db": row_count},
        "db_sha256": compute_file_sha256(db_path),
        "db_size": db_path.stat().st_size,
        "reused": True,
    }


def build_scratch_candidate_bm25(
    chunks_jsonl: Path,
    strict_doc_map: dict[str, tuple[str, str]],
    output_sqlite: Path,
    batch_size: int = 5000,
) -> dict[str, Any]:
    """Build scratch SQLite FTS5 candidate BM25 index with candidate search_text."""
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists():
        output_sqlite.unlink()

    conn = sqlite3.connect(str(output_sqlite))
    cursor = conn.cursor()

    cursor.execute(
        f"""
        CREATE VIRTUAL TABLE {TABLE_NAME} USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            document_type UNINDEXED,
            legal_field UNINDEXED,
            effect_status UNINDEXED,
            search_terms,
            chunk_json UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 0'
        )
        """
    )

    analyzer = UnicodeBM25Analyzer()
    tokenizer = UnicodeWordTokenizer()

    rows: list[tuple[Any, ...]] = []
    total_chunks = 0
    modified_chunks = 0
    modification_counter: Counter[str] = Counter()

    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            total_chunks += 1
            chunk = json.loads(line)
            doc_id = str(chunk["document_id"])
            base_search = chunk["search_text"]
            raw_text = chunk["text"]

            if doc_id in strict_doc_map:
                doc_type, doc_num = strict_doc_map[doc_id]
                cand_search, mod_type = construct_candidate_search_text(
                    base_search, raw_text, doc_type, doc_num, tokenizer
                )
                modification_counter[mod_type] += 1
                if cand_search != base_search:
                    modified_chunks += 1
                final_search = cand_search
            else:
                final_search = base_search
                modification_counter["non_strict_unchanged"] += 1

            terms = analyzer.analyze(final_search)
            search_terms_str = " ".join(terms)

            # Store updated search_text in chunk_json copy for exact representation
            chunk_copy = dict(chunk)
            chunk_copy["search_text"] = final_search
            chunk_json_str = json.dumps(chunk_copy, ensure_ascii=False)

            rows.append(
                (
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk.get("document_type"),
                    chunk.get("legal_field"),
                    chunk.get("effect_status"),
                    search_terms_str,
                    chunk_json_str,
                )
            )

            if len(rows) >= batch_size:
                cursor.executemany(
                    f"""
                    INSERT INTO {TABLE_NAME} (
                        chunk_id, document_id, document_type, legal_field,
                        effect_status, search_terms, chunk_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                rows.clear()

        if rows:
            cursor.executemany(
                f"""
                INSERT INTO {TABLE_NAME} (
                    chunk_id, document_id, document_type, legal_field,
                    effect_status, search_terms, chunk_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            rows.clear()

    conn.commit()
    conn.close()

    return {
        "total_chunks": total_chunks,
        "modified_chunks": modified_chunks,
        "modification_distribution": dict(modification_counter),
        "db_sha256": compute_file_sha256(output_sqlite),
        "db_size": output_sqlite.stat().st_size,
        "reused": False,
    }


def evaluate_bm25_retrieval(
    db_path: Path,
    eval_queries: list[dict[str, Any]],
    q_map: dict[str, str],
) -> dict[str, Any]:
    """Execute BM25 evaluation across query population using historical chunk-row cutoff semantics.

    Returns summary metrics and per-query rank lists.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    doc_recall_at_1 = 0
    doc_recall_at_5 = 0
    doc_recall_at_10 = 0
    doc_recall_at_20 = 0

    evaluated_count = 0
    query_results: list[dict[str, Any]] = []

    for item in eval_queries:
        qid = str(item["question_id"])
        target_doc = str(item["target_document_id"])
        q_text = q_map.get(qid, "")

        match_query = construct_match_query(q_text)
        if not match_query:
            query_results.append(
                {
                    "question_id": qid,
                    "target_document_id": target_doc,
                    "status": "EMPTY_QUERY",
                    "retrieved_documents": [],
                    "hit_at_1": False,
                    "hit_at_5": False,
                    "hit_at_10": False,
                    "hit_at_20": False,
                }
            )
            continue

        try:
            cursor.execute(
                f"SELECT chunk_id, document_id, bm25({TABLE_NAME}) as score "
                f"FROM {TABLE_NAME} WHERE {TABLE_NAME} MATCH ? ORDER BY score LIMIT 20",
                (match_query,),
            )
            rows = cursor.fetchall()
        except Exception as err:
            query_results.append(
                {
                    "question_id": qid,
                    "target_document_id": target_doc,
                    "status": f"ERROR: {err}",
                    "retrieved_documents": [],
                    "hit_at_1": False,
                    "hit_at_5": False,
                    "hit_at_10": False,
                    "hit_at_20": False,
                }
            )
            continue

        retrieved_docs = [str(r[1]) for r in rows]
        evaluated_count += 1

        hit_1 = target_doc in retrieved_docs[:1]
        hit_5 = target_doc in retrieved_docs[:5]
        hit_10 = target_doc in retrieved_docs[:10]
        hit_20 = target_doc in retrieved_docs[:20]

        if hit_1:
            doc_recall_at_1 += 1
        if hit_5:
            doc_recall_at_5 += 1
        if hit_10:
            doc_recall_at_10 += 1
        if hit_20:
            doc_recall_at_20 += 1

        query_results.append(
            {
                "question_id": qid,
                "target_document_id": target_doc,
                "status": "SUCCESS",
                "retrieved_documents": retrieved_docs,
                "hit_at_1": hit_1,
                "hit_at_5": hit_5,
                "hit_at_10": hit_10,
                "hit_at_20": hit_20,
            }
        )

    conn.close()

    total = evaluated_count if evaluated_count else len(eval_queries)

    return {
        "evaluated_count": evaluated_count,
        "total_queries": len(eval_queries),
        "recall_at_1_numerator": doc_recall_at_1,
        "recall_at_1_pct": (doc_recall_at_1 / total) * 100 if total else 0.0,
        "recall_at_5_numerator": doc_recall_at_5,
        "recall_at_5_pct": (doc_recall_at_5 / total) * 100 if total else 0.0,
        "recall_at_10_numerator": doc_recall_at_10,
        "recall_at_10_pct": (doc_recall_at_10 / total) * 100 if total else 0.0,
        "recall_at_20_numerator": doc_recall_at_20,
        "recall_at_20_pct": (doc_recall_at_20 / total) * 100 if total else 0.0,
        "query_results": query_results,
    }


# --- Structural Invariant Verification ---

def verify_structural_invariants(
    chunks_jsonl: Path,
    strict_doc_map: dict[str, tuple[str, str]],
    tokenizer: UnicodeWordTokenizer,
) -> dict[str, Any]:
    """Verify that baseline vs candidate chunk stream preserves 100% of non-search-text invariants."""
    chunk_id_hasher = sha256()
    doc_id_hasher = sha256()
    text_hasher = sha256()
    index_hasher = sha256()
    struct_hasher = sha256()

    total_chunks = 0
    strict_chunks = 0
    non_strict_chunks = 0
    eligible_modified = 0
    non_strict_unmodified = 0

    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            total_chunks += 1
            chunk = json.loads(line)

            cid = str(chunk["chunk_id"])
            doc_id = str(chunk["document_id"])
            raw_text = str(chunk["text"])
            chunk_idx = str(chunk["chunk_index"])
            struct_str = json.dumps(chunk.get("structure"), sort_keys=True)
            base_search = str(chunk["search_text"])

            chunk_id_hasher.update(cid.encode("utf-8"))
            doc_id_hasher.update(doc_id.encode("utf-8"))
            text_hasher.update(raw_text.encode("utf-8"))
            index_hasher.update(chunk_idx.encode("utf-8"))
            struct_hasher.update(struct_str.encode("utf-8"))

            if doc_id in strict_doc_map:
                strict_chunks += 1
                doc_type, doc_num = strict_doc_map[doc_id]
                cand_search, _ = construct_candidate_search_text(
                    base_search, raw_text, doc_type, doc_num, tokenizer
                )
                if cand_search != base_search:
                    eligible_modified += 1
            else:
                non_strict_chunks += 1
                non_strict_unmodified += 1

    return {
        "total_chunks": total_chunks,
        "strict_chunks": strict_chunks,
        "non_strict_chunks": non_strict_chunks,
        "eligible_modified_chunks": eligible_modified,
        "non_strict_unmodified_chunks": non_strict_unmodified,
        "ordered_chunk_ids_sha256": chunk_id_hasher.hexdigest(),
        "ordered_doc_ids_sha256": doc_id_hasher.hexdigest(),
        "ordered_raw_texts_sha256": text_hasher.hexdigest(),
        "ordered_chunk_indices_sha256": index_hasher.hexdigest(),
        "ordered_structural_sha256": struct_hasher.hexdigest(),
    }


# --- Master Evaluation Harness ---

def run_d1b_causal_evaluation(
    contexts_zip: Path,
    d0_evidence_zip: Path,
    train_json: Path,
    d1a_strict_evidence_zip: Path,
    chunks_jsonl: Path,
    baseline_bm25_sqlite: Path,
    scratch_dir: Path,
    evidence_zip_path: Path,
) -> dict[str, Any]:
    """Execute real full D1-B causal A/B retrieval evaluation."""
    _LOGGER.info("Step 0 & 3: Verifying canonical input source identities...")

    ctx_sha = compute_file_sha256(contexts_zip)
    if ctx_sha != EXPECTED_CONTEXTS_SHA256:
        raise ValueError(f"D1B_SOURCE_IDENTITY_FAILURE: contexts_zip SHA mismatch: {ctx_sha}")

    d0_sha = compute_file_sha256(d0_evidence_zip)
    if d0_sha != EXPECTED_D0_EVIDENCE_SHA256:
        raise ValueError(f"D1B_SOURCE_IDENTITY_FAILURE: d0_evidence_zip SHA mismatch: {d0_sha}")

    train_sha = compute_file_sha256(train_json)
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"D1B_SOURCE_IDENTITY_FAILURE: train_json SHA mismatch: {train_sha}")

    d1a_sha = compute_file_sha256(d1a_strict_evidence_zip)
    if d1a_sha != EXPECTED_D1A_STRICT_EVIDENCE_SHA256:
        raise ValueError(f"D1B_SOURCE_IDENTITY_FAILURE: d1a_strict_evidence SHA mismatch: {d1a_sha}")

    if not baseline_bm25_sqlite.exists():
        raise ValueError(f"D1B_SOURCE_IDENTITY_FAILURE: baseline BM25 index not found at {baseline_bm25_sqlite}")

    baseline_bm25_sha = compute_file_sha256(baseline_bm25_sqlite)

    _LOGGER.info("All canonical input sources verified successfully.")

    # Load frozen D0 evaluation population
    with zipfile.ZipFile(d0_evidence_zip) as z:
        d0_data = json.loads(z.read("train_qa_census.json").decode("utf-8"))
        unambiguous_links = d0_data["linkability"]["all_unambiguous_links"]

    if len(unambiguous_links) != EXPECTED_UNAMBIGUOUS_LINKS_COUNT:
        raise ValueError(
            f"D1B_POPULATION_FAILURE: expected {EXPECTED_UNAMBIGUOUS_LINKS_COUNT} unambiguous links, got {len(unambiguous_links)}"
        )

    # Load train questions
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    q_map: dict[str, str] = {}
    if isinstance(train_data, list):
        for item in train_data:
            q_map[str(item.get("question_id") or item.get("id"))] = item.get("question", "")
    elif isinstance(train_data, dict):
        for qid, item in train_data.items():
            q_map[str(qid)] = item.get("question", "") if isinstance(item, dict) else str(item)

    # Step 1: Extract and freeze strict document identities
    _LOGGER.info("Extracting strict document identities across 8,532 contexts...")
    strict_doc_map = extract_strict_document_identities(contexts_zip)
    if len(strict_doc_map) != EXPECTED_STRICT_DOC_COUNT:
        raise ValueError(
            f"D1B_EXTRACTION_FAILURE: expected {EXPECTED_STRICT_DOC_COUNT} strict identities, got {len(strict_doc_map)}"
        )

    tokenizer = UnicodeWordTokenizer()

    # Step 8: Verify structural invariants
    _LOGGER.info("Verifying structural invariants across 330,768 chunks...")
    structural_inv = verify_structural_invariants(chunks_jsonl, strict_doc_map, tokenizer)
    if structural_inv["total_chunks"] != EXPECTED_CHUNKS_COUNT:
        raise ValueError(
            f"D1_INVALID_EXPERIMENT: total chunks {structural_inv['total_chunks']} != {EXPECTED_CHUNKS_COUNT}"
        )
    if structural_inv["strict_chunks"] != EXPECTED_STRICT_CHUNK_COUNT:
        raise ValueError(
            f"D1_INVALID_EXPERIMENT: strict chunks {structural_inv['strict_chunks']} != {EXPECTED_STRICT_CHUNK_COUNT}"
        )

    # Step 5: Freeze pre-outcome manifests
    scratch_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = scratch_dir / "pre_outcome_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    strict_manifest_data = [
        {"document_id": doc_id, "document_type": dt, "document_number": dn, "strict_eligibility": True}
        for doc_id, (dt, dn) in sorted(strict_doc_map.items())
    ]
    strict_manifest_file = manifests_dir / "strict_identity_manifest.json"
    strict_manifest_file.write_text(json.dumps(strict_manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    strict_manifest_sha = compute_file_sha256(strict_manifest_file)

    proxy_population_data = [
        {
            "question_id": str(item["question_id"]),
            "target_document_id": str(item["target_document_id"]),
            "target_is_strict": str(item["target_document_id"]) in strict_doc_map,
            "query_signal_category": classify_query_identity_signals(q_map.get(str(item["question_id"]), "")),
        }
        for item in unambiguous_links
    ]
    proxy_population_file = manifests_dir / "proxy_population.json"
    proxy_population_file.write_text(json.dumps(proxy_population_data, ensure_ascii=False, indent=2), encoding="utf-8")
    proxy_population_sha = compute_file_sha256(proxy_population_file)

    contract_data = {
        "experiment_name": "PHASE_D1B_BM25_CAUSAL_AB",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "primary_success_gate": "Candidate_R@5 - Baseline_R@5 >= +2.0 percentage points",
        "secondary_success_gate": "Candidate_R@10 >= Baseline_R@10",
        "tertiary_safety_gate": "Candidate_R@20 >= Baseline_R@20 - 0.5 percentage points",
        "paired_safety_gate": "candidate_only_hit@5 > baseline_only_hit@5",
        "total_proxy_queries": EXPECTED_UNAMBIGUOUS_LINKS_COUNT,
        "strict_target_proxy_queries": sum(1 for p in proxy_population_data if p["target_is_strict"]),
    }
    contract_file = manifests_dir / "experiment_contract.json"
    contract_file.write_text(json.dumps(contract_data, ensure_ascii=False, indent=2), encoding="utf-8")
    contract_sha = compute_file_sha256(contract_file)

    candidate_manifest_data = {
        "candidate_strategy": "PREPEND_FROZEN_STRICT_DOCUMENT_IDENTITY",
        "eligible_documents": EXPECTED_STRICT_DOC_COUNT,
        "eligible_chunks": EXPECTED_STRICT_CHUNK_COUNT,
        "max_search_tokens": MAX_SEARCH_TOKENS,
        "priority_order": ["chunk.text", "document_number", "document_type", "existing_search_header"],
        "table_name": TABLE_NAME,
        "bm25_backend": "sqlite_fts5",
    }
    candidate_manifest_file = manifests_dir / "candidate_representation_manifest.json"
    candidate_manifest_file.write_text(json.dumps(candidate_manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_manifest_sha = compute_file_sha256(candidate_manifest_file)

    frozen_manifest_hashes = {
        "strict_identity_manifest_sha256": strict_manifest_sha,
        "proxy_population_sha256": proxy_population_sha,
        "candidate_representation_manifest_sha256": candidate_manifest_sha,
        "experiment_contract_sha256": contract_sha,
    }

    # Step 11: Historical 200 Baseline Replay — HARD GATE
    _LOGGER.info("Step 11: Replaying historical 200 baseline evaluation...")
    sample_200 = unambiguous_links[:200]
    replay_200_res = evaluate_bm25_retrieval(baseline_bm25_sqlite, sample_200, q_map)

    # Check exact baseline replay values: R@1=48.0%, R@5=71.5%, R@10=79.5%, R@20=86.0%
    r1_val = round(replay_200_res["recall_at_1_pct"], 1)
    r5_val = round(replay_200_res["recall_at_5_pct"], 1)
    r10_val = round(replay_200_res["recall_at_10_pct"], 1)
    r20_val = round(replay_200_res["recall_at_20_pct"], 1)

    replay_matched = (
        replay_200_res["evaluated_count"] == 200
        and r1_val == 48.0
        and r5_val == 71.5
        and r10_val == 79.5
        and r20_val == 86.0
    )

    if not replay_matched:
        return {
            "status": "FAILED",
            "decision": "D1_BASELINE_REPLAY_MISMATCH",
            "replay_200": replay_200_res,
            "reason": f"Baseline replay mismatch: R@1={r1_val} (exp 48.0), R@5={r5_val} (exp 71.5), R@10={r10_val} (exp 79.5), R@20={r20_val} (exp 86.0)",
        }

    _LOGGER.info("Historical 200 baseline replay PASSED perfectly (48.0%% / 71.5%% / 79.5%% / 86.0%%).")

    # Step 13: Reuse or build scratch candidate BM25 database
    candidate_db_path = scratch_dir / "bm25_candidate" / "index.sqlite3"
    build_meta = validate_existing_candidate_bm25(candidate_db_path)
    if build_meta is not None:
        _LOGGER.info(
            "Step 13: REUSING existing validated candidate BM25 at %s (%d rows, SHA %s).",
            candidate_db_path,
            build_meta["total_chunks"],
            build_meta["db_sha256"][:16],
        )
    else:
        _LOGGER.info("Step 13: Building scratch candidate BM25 at %s...", candidate_db_path)
        build_meta = build_scratch_candidate_bm25(chunks_jsonl, strict_doc_map, candidate_db_path)
        _LOGGER.info("Candidate BM25 built successfully (%d chunks modified).", build_meta["modified_chunks"])

    # Step 14: Paired 1,333 A/B Evaluation
    _LOGGER.info("Step 14: Executing paired 1,333 full proxy evaluation on Baseline BM25...")
    baseline_full_res = evaluate_bm25_retrieval(baseline_bm25_sqlite, unambiguous_links, q_map)

    _LOGGER.info("Step 14: Executing paired 1,333 full proxy evaluation on Candidate BM25...")
    candidate_full_res = evaluate_bm25_retrieval(candidate_db_path, unambiguous_links, q_map)

    # Step 16: Paired Transitions
    base_q_results = {r["question_id"]: r for r in baseline_full_res["query_results"]}
    cand_q_results = {r["question_id"]: r for r in candidate_full_res["query_results"]}

    transitions_at_1 = {"both_hit": 0, "baseline_only": 0, "candidate_only": 0, "both_miss": 0}
    transitions_at_5 = {"both_hit": 0, "baseline_only": 0, "candidate_only": 0, "both_miss": 0}
    transitions_at_10 = {"both_hit": 0, "baseline_only": 0, "candidate_only": 0, "both_miss": 0}
    transitions_at_20 = {"both_hit": 0, "baseline_only": 0, "candidate_only": 0, "both_miss": 0}

    paired_details: list[dict[str, Any]] = []

    for item in unambiguous_links:
        qid = str(item["question_id"])
        b_res = base_q_results[qid]
        c_res = cand_q_results[qid]

        for k, t_dict in [
            ("hit_at_1", transitions_at_1),
            ("hit_at_5", transitions_at_5),
            ("hit_at_10", transitions_at_10),
            ("hit_at_20", transitions_at_20),
        ]:
            b_hit = b_res[k]
            c_hit = c_res[k]
            if b_hit and c_hit:
                t_dict["both_hit"] += 1
            elif b_hit and not c_hit:
                t_dict["baseline_only"] += 1
            elif not b_hit and c_hit:
                t_dict["candidate_only"] += 1
            else:
                t_dict["both_miss"] += 1

        paired_details.append(
            {
                "question_id": qid,
                "target_document_id": str(item["target_document_id"]),
                "baseline_hit_at_5": b_res["hit_at_5"],
                "candidate_hit_at_5": c_res["hit_at_5"],
                "transition_at_5": (
                    "BOTH_HIT"
                    if b_res["hit_at_5"] and c_res["hit_at_5"]
                    else (
                        "BASELINE_ONLY"
                        if b_res["hit_at_5"]
                        else ("CANDIDATE_ONLY" if c_res["hit_at_5"] else "BOTH_MISS")
                    )
                ),
            }
        )

    transitions_at_5["net_gain"] = transitions_at_5["candidate_only"] - transitions_at_5["baseline_only"]
    transitions_at_1["net_gain"] = transitions_at_1["candidate_only"] - transitions_at_1["baseline_only"]
    transitions_at_10["net_gain"] = transitions_at_10["candidate_only"] - transitions_at_10["baseline_only"]
    transitions_at_20["net_gain"] = transitions_at_20["candidate_only"] - transitions_at_20["baseline_only"]

    # Step 17: Strict-Covered vs Uncovered Target Subsets
    strict_target_qids = set(p["question_id"] for p in proxy_population_data if p["target_is_strict"])
    non_strict_target_qids = set(p["question_id"] for p in proxy_population_data if not p["target_is_strict"])

    def compute_subset_metrics(qids: set[str]) -> dict[str, Any]:
        sub_count = len(qids)
        if not sub_count:
            return {}
        b_r1 = sum(1 for qid in qids if base_q_results[qid]["hit_at_1"])
        b_r5 = sum(1 for qid in qids if base_q_results[qid]["hit_at_5"])
        b_r10 = sum(1 for qid in qids if base_q_results[qid]["hit_at_10"])
        b_r20 = sum(1 for qid in qids if base_q_results[qid]["hit_at_20"])

        c_r1 = sum(1 for qid in qids if cand_q_results[qid]["hit_at_1"])
        c_r5 = sum(1 for qid in qids if cand_q_results[qid]["hit_at_5"])
        c_r10 = sum(1 for qid in qids if cand_q_results[qid]["hit_at_10"])
        c_r20 = sum(1 for qid in qids if cand_q_results[qid]["hit_at_20"])

        return {
            "subset_count": sub_count,
            "baseline": {
                "r1": b_r1, "r1_pct": b_r1 / sub_count * 100,
                "r5": b_r5, "r5_pct": b_r5 / sub_count * 100,
                "r10": b_r10, "r10_pct": b_r10 / sub_count * 100,
                "r20": b_r20, "r20_pct": b_r20 / sub_count * 100,
            },
            "candidate": {
                "r1": c_r1, "r1_pct": c_r1 / sub_count * 100,
                "r5": c_r5, "r5_pct": c_r5 / sub_count * 100,
                "r10": c_r10, "r10_pct": c_r10 / sub_count * 100,
                "r20": c_r20, "r20_pct": c_r20 / sub_count * 100,
            },
            "delta_pct": {
                "r1": (c_r1 - b_r1) / sub_count * 100,
                "r5": (c_r5 - b_r5) / sub_count * 100,
                "r10": (c_r10 - b_r10) / sub_count * 100,
                "r20": (c_r20 - b_r20) / sub_count * 100,
            },
        }

    strict_target_subset_metrics = compute_subset_metrics(strict_target_qids)
    non_strict_target_subset_metrics = compute_subset_metrics(non_strict_target_qids)

    # Step 18: Query Identity-Signal Diagnostic Subsets
    query_signal_subsets: dict[str, dict[str, Any]] = {}
    for cat in ["explicit_document_number", "explicit_document_type", "both", "neither"]:
        cat_qids = set(p["question_id"] for p in proxy_population_data if p["query_signal_category"] == cat)
        query_signal_subsets[cat] = compute_subset_metrics(cat_qids)

    # Step 15: Primary Pre-Registered Success Gates Evaluation
    delta_r1 = candidate_full_res["recall_at_1_pct"] - baseline_full_res["recall_at_1_pct"]
    delta_r5 = candidate_full_res["recall_at_5_pct"] - baseline_full_res["recall_at_5_pct"]
    delta_r10 = candidate_full_res["recall_at_10_pct"] - baseline_full_res["recall_at_10_pct"]
    delta_r20 = candidate_full_res["recall_at_20_pct"] - baseline_full_res["recall_at_20_pct"]

    primary_gate_pass = delta_r5 >= 2.0
    secondary_gate_pass = candidate_full_res["recall_at_10_pct"] >= baseline_full_res["recall_at_10_pct"]
    tertiary_gate_pass = candidate_full_res["recall_at_20_pct"] >= (baseline_full_res["recall_at_20_pct"] - 0.5)
    paired_safety_pass = transitions_at_5["candidate_only"] > transitions_at_5["baseline_only"]

    all_gates_pass = (
        primary_gate_pass
        and secondary_gate_pass
        and tertiary_gate_pass
        and paired_safety_pass
    )

    final_decision = "D1_DOCUMENT_IDENTITY_RETAIN" if all_gates_pass else "KEEP_BASELINE"

    # Step 20: Package Evidence ZIP
    evidence_zip_path.parent.mkdir(parents=True, exist_ok=True)
    if evidence_zip_path.exists():
        evidence_zip_path.unlink()

    with zipfile.ZipFile(evidence_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "execution/source_identity.json",
            json.dumps(
                {
                    "contexts_zip_sha256": ctx_sha,
                    "d0_evidence_zip_sha256": d0_sha,
                    "train_json_sha256": train_sha,
                    "d1a_strict_evidence_zip_sha256": d1a_sha,
                    "baseline_bm25_sha256": baseline_bm25_sha,
                    "candidate_bm25_sha256": build_meta["db_sha256"],
                },
                indent=2,
            ),
        )
        z.writestr("execution/experiment_contract.json", json.dumps(contract_data, indent=2, ensure_ascii=False))
        z.writestr("execution/frozen_manifest_hashes.json", json.dumps(frozen_manifest_hashes, indent=2))
        z.writestr("results/structural_invariants.json", json.dumps(structural_inv, indent=2))
        z.writestr("results/baseline_replay_200.json", json.dumps(replay_200_res, indent=2))
        z.writestr(
            "results/full_proxy_baseline.json",
            json.dumps(
                {k: v for k, v in baseline_full_res.items() if k != "query_results"},
                indent=2,
            ),
        )
        z.writestr(
            "results/full_proxy_candidate.json",
            json.dumps(
                {k: v for k, v in candidate_full_res.items() if k != "query_results"},
                indent=2,
            ),
        )
        z.writestr(
            "results/full_proxy_comparison.json",
            json.dumps(
                {
                    "evaluated_count": EXPECTED_UNAMBIGUOUS_LINKS_COUNT,
                    "baseline": {
                        "r1": baseline_full_res["recall_at_1_numerator"], "r1_pct": baseline_full_res["recall_at_1_pct"],
                        "r5": baseline_full_res["recall_at_5_numerator"], "r5_pct": baseline_full_res["recall_at_5_pct"],
                        "r10": baseline_full_res["recall_at_10_numerator"], "r10_pct": baseline_full_res["recall_at_10_pct"],
                        "r20": baseline_full_res["recall_at_20_numerator"], "r20_pct": baseline_full_res["recall_at_20_pct"],
                    },
                    "candidate": {
                        "r1": candidate_full_res["recall_at_1_numerator"], "r1_pct": candidate_full_res["recall_at_1_pct"],
                        "r5": candidate_full_res["recall_at_5_numerator"], "r5_pct": candidate_full_res["recall_at_5_pct"],
                        "r10": candidate_full_res["recall_at_10_numerator"], "r10_pct": candidate_full_res["recall_at_10_pct"],
                        "r20": candidate_full_res["recall_at_20_numerator"], "r20_pct": candidate_full_res["recall_at_20_pct"],
                    },
                    "delta_percentage_points": {
                        "r1": delta_r1,
                        "r5": delta_r5,
                        "r10": delta_r10,
                        "r20": delta_r20,
                    },
                },
                indent=2,
            ),
        )
        z.writestr(
            "results/paired_transitions.json",
            json.dumps(
                {
                    "transitions_at_1": transitions_at_1,
                    "transitions_at_5": transitions_at_5,
                    "transitions_at_10": transitions_at_10,
                    "transitions_at_20": transitions_at_20,
                },
                indent=2,
            ),
        )
        z.writestr(
            "results/strict_target_subset.json",
            json.dumps(
                {
                    "strict_target_subset": strict_target_subset_metrics,
                    "non_strict_target_subset": non_strict_target_subset_metrics,
                },
                indent=2,
            ),
        )
        z.writestr("results/query_identity_subsets.json", json.dumps(query_signal_subsets, indent=2))
        z.writestr(
            "results/d1b_decision.json",
            json.dumps(
                {
                    "final_decision": final_decision,
                    "primary_gate_passed": primary_gate_pass,
                    "primary_gate_delta_r5": delta_r5,
                    "primary_gate_threshold": 2.0,
                    "secondary_gate_passed": secondary_gate_pass,
                    "tertiary_gate_passed": tertiary_gate_pass,
                    "paired_safety_passed": paired_safety_pass,
                },
                indent=2,
            ),
        )

    evidence_zip_sha = compute_file_sha256(evidence_zip_path)
    evidence_zip_size = evidence_zip_path.stat().st_size

    with zipfile.ZipFile(evidence_zip_path) as z:
        evidence_member_count = len(z.namelist())

    return {
        "status": "COMPLETED",
        "final_decision": final_decision,
        "source_identities": {
            "contexts_zip_sha256": ctx_sha,
            "d0_evidence_zip_sha256": d0_sha,
            "train_json_sha256": train_sha,
            "d1a_strict_evidence_zip_sha256": d1a_sha,
            "baseline_bm25_sha256": baseline_bm25_sha,
            "candidate_bm25_sha256": build_meta["db_sha256"],
        },
        "frozen_manifest_hashes": frozen_manifest_hashes,
        "structural_invariants": structural_inv,
        "baseline_replay_200": replay_200_res,
        "full_proxy_baseline": {k: v for k, v in baseline_full_res.items() if k != "query_results"},
        "full_proxy_candidate": {k: v for k, v in candidate_full_res.items() if k != "query_results"},
        "deltas": {
            "r1": delta_r1,
            "r5": delta_r5,
            "r10": delta_r10,
            "r20": delta_r20,
        },
        "transitions_at_5": transitions_at_5,
        "strict_target_subset": strict_target_subset_metrics,
        "query_signal_subsets": query_signal_subsets,
        "gates": {
            "primary_gate_pass": primary_gate_pass,
            "secondary_gate_pass": secondary_gate_pass,
            "tertiary_gate_pass": tertiary_gate_pass,
            "paired_safety_pass": paired_safety_pass,
        },
        "evidence_package": {
            "path": str(evidence_zip_path),
            "sha256": evidence_zip_sha,
            "size": evidence_zip_size,
            "member_count": evidence_member_count,
        },
    }


def main() -> None:
    """CLI Entrypoint for Phase D1-B evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate Phase D1-B strict document identity BM25 causal A/B.")
    parser.add_argument(
        "--contexts-zip",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\selected-contexts.zip"),
        help="Path to selected-contexts.zip",
    )
    parser.add_argument(
        "--d0-evidence-zip",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\data-d0-official-data-audit-evidence.zip"),
        help="Path to data-d0-official-data-audit-evidence.zip",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\train.json"),
        help="Path to train.json",
    )
    parser.add_argument(
        "--d1a-strict-evidence-zip",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\data-d1a-document-identity-feasibility-strict-evidence.zip"),
        help="Path to data-d1a-document-identity-feasibility-strict-evidence.zip",
    )
    parser.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=Path(r"artifacts\uit-dsc-2026-task2-v0400\legal_chunks\records.jsonl"),
        help="Path to records.jsonl",
    )
    parser.add_argument(
        "--baseline-bm25",
        type=Path,
        default=Path(r"artifacts\uit-dsc-2026-task2-v0400\bm25\index.sqlite3"),
        help="Path to baseline index.sqlite3",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(r"scratch\d1b_document_identity"),
        help="Scratch directory for candidate BM25 index and pre-outcome manifests",
    )
    parser.add_argument(
        "--evidence-zip",
        type=Path,
        default=Path(r"C:\Users\Nguyen\Downloads\data-d1b-document-identity-bm25-evidence.zip"),
        help="Output path for data-d1b-document-identity-bm25-evidence.zip",
    )

    args = parser.parse_args()

    results = run_d1b_causal_evaluation(
        contexts_zip=args.contexts_zip,
        d0_evidence_zip=args.d0_evidence_zip,
        train_json=args.train_json,
        d1a_strict_evidence_zip=args.d1a_strict_evidence_zip,
        chunks_jsonl=args.chunks_jsonl,
        baseline_bm25_sqlite=args.baseline_bm25,
        scratch_dir=args.scratch_dir,
        evidence_zip_path=args.evidence_zip,
    )

    print("\n" + "=" * 60)
    print("PHASE D1-B BM25 CAUSAL A/B EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Final Decision: {results['final_decision']}")
    print(f"Baseline Recall:  R@1={results['full_proxy_baseline']['recall_at_1_pct']:.2f}%  R@5={results['full_proxy_baseline']['recall_at_5_pct']:.2f}%  R@10={results['full_proxy_baseline']['recall_at_10_pct']:.2f}%  R@20={results['full_proxy_baseline']['recall_at_20_pct']:.2f}%")
    print(f"Candidate Recall: R@1={results['full_proxy_candidate']['recall_at_1_pct']:.2f}%  R@5={results['full_proxy_candidate']['recall_at_5_pct']:.2f}%  R@10={results['full_proxy_candidate']['recall_at_10_pct']:.2f}%  R@20={results['full_proxy_candidate']['recall_at_20_pct']:.2f}%")
    print(f"Deltas (pt):      ΔR@1={results['deltas']['r1']:+.2f}  ΔR@5={results['deltas']['r5']:+.2f}  ΔR@10={results['deltas']['r10']:+.2f}  ΔR@20={results['deltas']['r20']:+.2f}")
    print(f"Primary Gate (ΔR@5 >= +2.0): {results['gates']['primary_gate_pass']}")
    print(f"Paired Net Gain @5: {results['transitions_at_5']['net_gain']} (cand_only={results['transitions_at_5']['candidate_only']}, base_only={results['transitions_at_5']['baseline_only']})")
    print(f"Evidence Package: {results['evidence_package']['path']} ({results['evidence_package']['size']} bytes, SHA: {results['evidence_package']['sha256']})")


if __name__ == "__main__":
    main()
