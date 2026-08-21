# 34. Phase D0 — Deep Official Dataset Census & Retrieval Unit Audit

---

## 1. Executive Summary & Audit Authority

Phase D0 conducts an exhaustive, byte-level census and structural audit of the canonical official UIT Data Science Challenge 2026 Task 2 dataset and current serving artifacts (`artifacts/uit-dsc-2026-task2-v0400`).

- **Audit Date**: 2026-08-21
- **Parent / Starting Authority Commit**: `469dd45834f6ef2406198e3368669459bebeb264`
- **Audit Tool**: `scripts/audit_official_data_d0.py`
- **Evidence Archive**: `C:\Users\Nguyen\Downloads\data-d0-official-data-audit-evidence.zip`
- **Evidence Archive SHA-256**: `eca404a749a45c00b6b7b94c7dee246fea39de385882e51343f6f1a20d93c27f` (40,549 bytes)

---

## 2. Official Source Identities & Checksum Verification

| Dataset Role | File Path | File Size (Bytes) | SHA-256 Checksum | Record Count | Schema Keys |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Official Train** | `C:\Users\Nguyen\Downloads\train.json` | 16,078,892 | `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988` | 7,000 | `["question", "answer"]` |
| **Official Public** | `C:\Users\Nguyen\Downloads\public-official.json` | 185,622 | `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5` | 1,000 | `["question", "answer"]` |
| **Official Contexts Archive** | `C:\Users\Nguyen\Downloads\selected-contexts.zip` | 97,276,888 | `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97` | 8,532 | `["id", "name", "link", "passage"]` |

- **Context Canonical Revision (Content-Level)**: `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e` (verified across all 8,532 JSON member contents).
- **Context Archive SHA-256 (Byte-Level ZIP Identity)**: `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97` (97,276,888 bytes, 8,532 member files).
- **Public Label Status**: In `public-official.json`, all 1,000 records have `"answer": null`.

---

## 3. Raw Corpus Census & Distribution

Across the 8,532 raw context documents:
- **Total Records**: 8,532
- **Non-Empty Passages**: 8,512 (99.77%)
- **Empty Passages (`passage == ""` / whitespace)**: 20 (0.23%) (e.g., `id: 8645`, `id: 288347`)
- **Records with Non-Null `name` (Title)**: 7,407 (86.81%)
- **Records with Null/Missing `name`**: 1,125 (13.19%)
- **Records with `link` (Source URL)**: 8,532 (100.0%)
- **Total Raw Characters**: 353,691,230 characters
- **Exact Duplicate Document Clusters**: 4 clusters (9 records total)
- **Top 3 Largest Documents**:
  1. `context_68843`: 5,983,358 chars (~1.24M words / 1,236,787 words), title: `null`
  2. `context_4644`: 3,015,870 chars (~571k words), title: `null`
  3. `context_42223`: 1,097,089 chars (~237k words), title: `Thong-tu-200-2014-TT-BTC-huong-dan-Che-do-ke-toan-Doanh-nghiep-263599`

### Character & Word Length Percentiles (Non-Empty Documents, N=8,512)

| Metric | Min | P25 | P50 (Median) | P75 | P90 | P95 | P99 | Max | Mean | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Characters** | 390 | 11,937 | 23,215 | 45,848 | 90,214 | 135,737 | 285,268 | 5,983,358 | 41,552.1 | 91,348.4 |
| **Words** | 67 | 2,487 | 4,828 | 9,402 | 18,418 | 27,532 | 59,249 | 1,242,409 | 8,555.6 | 18,720.9 |

---

## 4. Legal Structure Markers & Parser Coverage Audit

Conservative marker regex analysis across all 8,512 non-empty documents:
- `KHOAN_NUMBERED` (`^\s*\d+\.`): 8,174 documents (96.03%)
- `DIEU` (`^\s*Điều\s+\d+`): 7,331 documents (86.13%)
- `DIEM_LETTERED` (`^\s*[a-zđ]\)`): 6,848 documents (80.45%)
- `CHUONG` (`^\s*Chương\s+[IVXLCDM\d]+`): 5,116 documents (60.10%)
- `PHU_LUC` (`^\s*Phụ\s+lục`): 2,828 documents (33.22%)
- `MUC` (`^\s*Mục\s+\d+`): 2,720 documents (31.96%)
- `DIEM_EXPLICIT` (`^\s*Điểm\s+[a-zđ]`): 2,438 documents (28.64%)
- `KHOAN_EXPLICIT` (`^\s*Khoản\s+\d+`): 2,302 documents (27.04%)
- `PHAN` (`^\s*Phần\s+thứ`): 2,137 documents (25.11%)
- `TIEU_MUC` (`^\s*Tiểu\s+mục`): 48 documents (0.56%)

### Parser Agreement & Diagnostic Coverage
- **Documents with >= 1 Parsed Article**: 7,332 documents (86.14%)
- **Documents with 0 Recognized Structure**: 1,200 documents (13.86%) (including 20 empty documents + 1,180 unstructured/announcement documents)
- **`Điều` Marker vs Parser Agreement**:
  - `DIEU` marker present & article parsed: 7,331 documents (100.0% precision on explicit markers)
  - `DIEU` marker present but article NOT parsed: 0 documents (0 false negatives on clean marker starts)
  - Article parsed with `DIEU` marker absent: 1 document (variant typography)
- **Document Structure Count Distributions (N=8,512)**:
  - **Articles per doc**: Median 12.0, Mean 19.27, P95 59.0, Max 838.0
  - **Clauses per doc**: Median 33.0, Mean 59.19, P95 202.5, Max 1,425.0
  - **Points per doc**: Median 20.0, Mean 49.85, P95 191.0, Max 2,938.0

---

## 5. Serving Chunks Census & Boundary Risk Audit

Current production serving chunks in `artifacts/uit-dsc-2026-task2-v0400/legal_chunks/records.jsonl`:
- **Total Chunks**: 330,768 chunks across 8,512 unique source documents
- **Chunks per Document**: Min 1, P25 13, P50 25, P75 44, P95 120, P99 241, Max 3,061, Mean 38.86

### Chunk Strategy Distribution
| Strategy | Chunk Count | Percentage |
| :--- | :--- | :--- |
| `article` | 130,748 | 39.53% |
| `token_fallback` | 108,009 | 32.65% |
| `clause_group` | 81,122 | 24.53% |
| `standalone_block` | 10,889 | 3.29% |
| **Total** | **330,768** | **100.0%** |

### Chunk Token Count Percentiles
- **Min**: 5 tokens
- **P25**: 152 tokens
- **P50 (Median)**: 298 tokens
- **P75**: 447 tokens
- **P90 / P95 / P99 / Max**: 448 tokens (hard chunking ceiling)
- **Mean**: 284.72 tokens (Std Dev: 143.82)

### Boundary Risk Findings (N=322,256 adjacent pairs evaluated)
- **Cross-Reference Split (`CROSS_REFERENCE_SPLIT`)**: 971 pairs (0.30%)
  - *Manifestation*: A statutory condition references `theo quy định tại` at the end of Chunk A, while the target `khoản 3 Điều này...` is placed at the start of Chunk B.
- **List Header Split From Items (`LIST_HEADER_SPLIT_FROM_ITEMS`)**: 649 pairs (0.20%)
  - *Manifestation*: An introductory clause ending with `:` is separated from the indented itemized provisions `a), b), c)`.
- **Condition Open at Left Boundary (`CONDITION_OPEN_AT_LEFT_BOUNDARY`)**: 436 pairs (0.14%)
  - *Manifestation*: Chunk starts mid-sentence with `người sử dụng lao động phải...` without the governing conditional premise (`Trường hợp người lao động...`).
- **Total Boundary Risk Pairs**: 2,056 pairs.

### Search Text Context & Header Findings
- **Title Context (`Văn bản:`)**: Present in 271,355 chunks (82.04%)
- **Article Header (`Điều X:`)**: Present in 264,770 chunks (80.05%)
- **Hierarchy Header (`Chương/Mục`)**: Present in 223,994 chunks (67.72%)
- **Clause Numbers (`Khoản:`)**: Present in 207,391 chunks (62.70%)
- **D0 Search-Header Interpretation Correction (Verified 2026-08-21)**:
  - An initial D0 sampling artifact in `scripts/audit_official_data_d0.py` inspected only the first 3 `token_fallback` chunks encountered in corpus order (which happened to come from unstructured documents `context_0` and `context_1` lacking titles and structure), leading to an incorrect generalization that all 108,009 fallback chunks lacked headers.
  - Independent full-corpus verification over all 108,009 `token_fallback` chunks established the exact empirical distribution:
    * **Total `token_fallback` Chunks**: 108,009 (32.65% of corpus)
    * **Non-Empty Baseline Headers**: 94,679 (87.66%) — `LegalChunker._search_text()` had already successfully prepended article, clause, point, chapter, or title headers into these chunks.
    * **Empty Baseline Headers**: 13,330 (12.34%)
    * **With $\ge 1$ Usable Lineage Parent Field**: 94,790 (87.76%)
    * **With No Usable Parent Field** (unstructured documents): 13,219 (12.24%)
    * **Budget-Constrained Chunks** (had usable parent field but text length reached 448 tokens ceiling and header could not fit within 512 tokens): 111 (0.10%)

---

## 6. Official Metadata Population Deficiencies

| Metadata Field | Non-Null Chunks | Coverage Pct | Distinct Values |
| :--- | :--- | :--- | :--- |
| `source_url` | 330,768 | 100.0% | 8,512 |
| `document_title` | 300,047 | 90.71% | 7,407 |
| `chapter` | 188,366 | 56.95% | 77 |
| `section` | 59,689 | 18.05% | 40 |
| `document_number` | 0 | **0.0%** | 0 |
| `document_type` | 0 | **0.0%** | 0 |
| `issuing_authority` | 0 | **0.0%** | 0 |
| `issuance_date` | 0 | **0.0%** | 0 |
| `effective_date` | 0 | **0.0%** | 0 |
| `expiry_date` | 0 | **0.0%** | 0 |
| `effect_status` | 0 | **0.0%** | 0 |
| `legal_field` | 0 | **0.0%** | 0 |

*Root Cause*: Official raw contexts provide only `id`, `name` (slug), `link`, and `passage`. No explicit metadata dictionary was provided by BTC. The current ingestion pipeline intentionally uses `infer_missing_legal_metadata = False`, leaving document numbers and document types unextracted from the passage header, URL, or document title.

---

## 7. Official Train Q&A Census & Linkability

Across the 7,000 answer-labeled train records:
- **Total Questions**: 7,000
- **Question Length**: Mean 88.1 chars / 19.7 words (Median: 85 chars / 19 words, Min: 12 chars, Max: 272 chars)
- **Answer Length**: Mean 1,575.7 chars / 347.4 words (Median: 1,410 chars / 312 words, Min: 129 chars, Max: 10,755 chars)
- **Answer-to-Question Ratio**: Median 16.51×, Mean 19.89×, P95 44.87×, Max 207.33×

### Question Taxonomy Breakdown
- `deadline_temporal` (Thời hạn, thời gian): 4,236 (60.51%)
- `rights_obligations` (Quyền và nghĩa vụ): 4,012 (57.31%)
- `condition_eligibility` (Điều kiện, tiêu chuẩn): 3,644 (52.06%)
- `legal_citation_lookup` (Căn cứ, quy định tại): 3,068 (43.83%)
- `numeric_money_quantity` (Mức phạt, tỷ lệ, tiền): 2,818 (40.26%)
- `authority_competence` (Thẩm quyền, cơ quan): 2,727 (38.96%)
- `procedure_steps` (Trình tự, thủ tục): 2,305 (32.93%)
- `penalty_sanction` (Xử phạt, vi phạm): 635 (9.07%)
- `definition` (Khái niệm, định nghĩa): 500 (7.14%)
- `binary_yes_no` (Có được phép, hay không): 222 (3.17%)

### Prose Style Breakdown
- `multi_paragraph` (>= 2 đoạn văn): 6,961 (99.44%)
- `enumerated_list` (Đánh số 1, 2, 3... hoặc a, b, c): 4,954 (70.77%)
- `bullet_list` (Gạch đầu dòng `-`, `*`, `•`): 3,009 (42.99%)
- `multi_sentence_prose` (Đoạn đơn nhiều câu): 39 (0.56%)
- `single_short_sentence`: 0 (0.0%)

### Legal Reference Signals & Linkability
- **Answers with Explicit Legal Anchors**: 6,396 / 7,000 (91.37%)
- **Unambiguous Question -> Document Links Discovered**: 1,333 questions (19.04%)
- **Unambiguous Question -> Article Links Discovered**: 639 questions (9.13%)
- **Ambiguous Multi-Document Matches**: 372 questions (5.31%)
- **Unresolved Anchors (Citations not in context corpus)**: 3,453 questions (49.33%)
- **Exact Overlap Between Train & Public**: 1 identical question (`train.json` has answer, `public-official.json` has `null`).

---

## 8. High-Confidence Official-Data Retrieval Proxy Evaluation

Evaluated against the serving SQLite FTS5 index (`bm25_documents`, 330,768 rows) across the historical 200-sample of unambiguous question-to-document links:
- **Evaluation Role**: **HIGH-CONFIDENCE OFFICIAL-DATA RETRIEVAL PROXY** (not official relevance ground truth).
- **Evaluated Link Count**: 200 unambiguous QA pairs (subset of the 1,333 discovered links; not all 1,333 were evaluated in D0).
- **Document Recall @ 1**: 48.0%
- **Document Recall @ 5**: 71.5%
- **Document Recall @ 10**: 79.5%
- **Document Recall @ 20**: 86.0%

---

## 9. Next Experiment: Selection of Exactly One D1 Candidate

### Cancelled Candidate: D1 — Parent-Context Enriched Token-Fallback Search Representation
- **Status**: `CANCELLED_AFTER_PREMISE_FALSIFICATION`
- **Reason**: The original premise that all 108,009 `token_fallback` chunks lacked parent headers was disproven (94,679 already possessed headers; only 111 were budget-constrained).

### Selected Candidate: D1 — Deterministic Legal Document Identity Enrichment

- **Single Causal Variable**:
  Deterministically extract `document_type` and `document_number` from official context content (title, source URL, and early passage header) using multi-source agreement, and enrich candidate `search_text` with canonical `Loại văn bản:` and `Số ký hiệu:` lines where high confidence is achieved.
- **Strict Invariants Preserved**:
  - Exact total chunk count unchanged (330,768).
  - Exact chunk IDs unchanged.
  - Exact chunk boundaries / token offsets unchanged.
  - Exact raw chunk `text` unchanged.
  - Max search tokens ceiling ($512$) and complete chunk text suffix strictly preserved.
  - Zero external data or LLM inference.
  - Production ingestion and chunker untouched during experiment.

### Pre-Registered D1 Measurement Protocol & Gates
- **Feasibility Gate**: High-confidence complete identity (type + number) coverage $\ge 50\%$ of non-empty documents or $\ge 70\%$ of 1,333 proxy target documents.
- **Structural Invariants Gate**: 100% pass on 330,768 chunk count, IDs, boundaries, raw text invariance, and 512 token ceiling.
- **Retrieval Performance Gate**:
  - **Primary**: Full 1,333 BM25 document Recall@5 improves by $\ge +2.0$ absolute percentage points over baseline.
  - **Secondary**: Recall@10 must not regress.
  - **Tertiary**: Recall@20 must not regress by $> 0.5$ absolute percentage points.

---

## 10. Architectural Backlog (Future Hypotheses)

1. **Backlog Item 1 — Hierarchical Legal Chunking (Clause / Point Resegmentation)**:
   Re-architect the offline chunker to parse long articles into structured clauses/points instead of token-based sliding windows.
2. **Backlog Item 2 — Online Adjacent Chunk Window Expansion & Parent Stitching**:
   Implement dynamic boundary expansion during context building for adjacent chunks possessing high boundary risk tags (list headers, condition truncations).
3. **Backlog Item 3 — Secondary Legal Metadata Extraction (Authority, Dates, Effect Status)**:
   Extract `issuing_authority`, `issuance_date`, `effective_date`, and `effect_status` once basic document identity is proven.
