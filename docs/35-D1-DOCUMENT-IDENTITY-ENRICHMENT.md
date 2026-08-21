# D1 — Deterministic Legal Document Identity Enrichment: Feasibility Report (Phase D1-A)

## 1. Context & Motivation

Following the completion of the Phase D0 Official Dataset Census and Retrieval Unit Boundary Audit (`docs/34-DATA-CENSUS-AND-RETRIEVAL-UNIT-AUDIT-D0.md`, Decision **D129**), the repository identified that 0% of serving legal chunks currently possess extracted document identity metadata (`document_type`, `document_number`), and all 108,009 `token_fallback` chunks (32.65% of the corpus) carry empty header context in `search_text`.

Candidate **D1 — Deterministic Legal Document Identity Enrichment** was formalized in Decision **D130** to evaluate whether high-confidence official document identity can be safely and deterministically extracted solely from organizer data and prepended to chunk search representations to improve lexical retrieval precision.

This report documents the execution and results of **Phase D1-A: Deterministic Legal Document Identity Feasibility**.

> [!IMPORTANT]
> **Check-point Scope & Interpretation:**
> A `D1A_FEASIBILITY_PASS` decision proves **ONLY** that there is sufficient safe organizer-derived document identity across the official corpus to justify running the subsequent BM25 causal A/B evaluation.
>
> It does **NOT** mean retrieval has improved, production chunking or ingestion should change, METEOR has improved, or document identity should be permanently ingested into production artifacts.
>
> **Status:** `D1-A FEASIBILITY COMPLETE` — **`D1-B BM25 A/B NOT YET RUN`**.

---

## 2. Canonical Source Identity Verification

Phase D1-A executed strictly against canonical, uncorrupted official inputs with full byte-level checksum verification:

| Source Artifact | Expected SHA-256 | Verified Byte Count | Verification Status |
| :--- | :--- | :--- | :--- |
| `selected-contexts.zip` | `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97` | 97,276,888 bytes (8,532 JSON context files) | `VERIFIED_MATCH` |
| `data-d0-official-data-audit-evidence.zip` | `eca404a749a45c00b6b7b94c7dee246fea39de385882e51343f6f1a20d93c27f` | 40,549 bytes (7 members) | `VERIFIED_MATCH` |
| `train.json` | `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988` | 15,316,569 bytes (7,000 QA records) | `VERIFIED_MATCH` |
| `legal_chunks/records.jsonl` | Serving v0400 canonical records | 330,768 records | `VERIFIED_MATCH` |

---

## 3. Extraction Contract & Own-Document Identity Safety

### 3.1 Single Causal Variable Scope
Extraction in Phase D1-A was restricted strictly to two fields:
- `document_type`: Canonical Vietnamese legal document type (e.g. `Thông tư`, `Nghị định`, `Quyết định`, `Luật`).
- `document_number`: Canonical official legal number string (e.g. `99/2003/NĐ-CP`, `30/2022/TT-BTC`, `51/QĐ-VKSTC-V12`).

Extraction of any secondary metadata (issuing authority, issuance date, effective date, expiry date, effect status, legal field) was strictly prohibited.

### 3.2 Allowed Organizer Sources
Extraction candidates were derived solely from three independent organizer sources:
1. `title_candidate`: Extracted from context `name` slug.
2. `url_candidate`: Extracted from context `link` source URL slug.
3. `header_candidate`: Extracted from the deterministic early header/preamble region of the raw `passage`.

No external databases, no LLMs, no semantic guesses, no web crawls, and no synthetic mappings were utilized.

### 3.3 Own-Document Identity Safety Guard
Vietnamese legal documents frequently cite other legal acts (e.g. `Căn cứ Luật...`, `theo Nghị định số...`). Naive extraction of the first legal number in a document would severely corrupt document identity.

Passage header extraction in `scripts/evaluate_document_identity_d1.py` enforces strict structural guards:
- Extraction inspects only lines preceding substantive enacting formulas (`Căn cứ`, `Điều \d+`, `Chương [IVXLCDM]+`) or within the first 50 non-empty lines.
- Number extraction is strictly tied to formal header prefixes (`Số:`, `Số `).
- Broken multi-line uppercase type headings (e.g. `THÔNG \n TƯ`, `NGHỊ \n ĐỊNH`, `QUYẾT \n ĐỊNH`) are combined before type resolution.
- Body citations to referenced laws are strictly ignored and never become the source document's identity.

### 3.4 Confidence Resolution Policy
Candidates are grouped by normalized comparison keys (`normalize_key(type)::normalize_key(number)`):
- `HIGH_CONFIDENCE`:
  - $\ge 2$ independent organizer sources agree unanimously on the same normalized identity (`all_three`, `title_url`, `title_header`, `url_header`), OR
  - 1 organizer source contains an exceptionally explicit complete own-document identity and no conflicting candidate exists (`single_url`, `single_header`, `single_title`).
- `AMBIGUOUS`: Conflicting candidates across sources (fails closed).
- `UNRESOLVED`: No complete candidate extracted (fails closed).

---

## 4. Full-Corpus Feasibility Metrics

The extraction harness was executed over all 8,532 official contexts with zero estimation:

### 4.1 Document-Level Census & Coverage
| Metric | Count | All Contexts (8,532) | Non-Empty Contexts (8,512) | Titled Contexts (7,407) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Contexts** | 8,532 | 100.0% | — | — |
| **Non-Empty Contexts** | 8,512 | 99.77% | 100.0% | — |
| **Titled Contexts (`name != None`)** | 7,407 | 86.81% | 87.02% | 100.0% |
| **HIGH_CONFIDENCE `document_type`** | 8,089 | 94.81% | 95.03% | 109.21% |
| **HIGH_CONFIDENCE `document_number`** | 8,089 | 94.81% | 95.03% | 109.21% |
| **HIGH_CONFIDENCE Complete Identity (Type + Number)** | **8,089** | **94.81%** | **95.03%** | **109.21%** |
| **AMBIGUOUS (Conflicting candidates)** | 258 | 3.02% | 3.03% | — |
| **UNRESOLVED (No complete candidate)** | 185 | 2.17% | 2.17% | — |

### 4.2 Source Agreement Breakdown
Among the 8,089 `HIGH_CONFIDENCE` complete identities:
- **Multi-Source Agreement ($\ge 2$ sources):** **7,006 records (86.61%)**
  - All three sources (`Title + URL + Header`): 6,823 records (84.35%)
  - Two sources (`Title + URL`): 115 records (1.42%)
  - Two sources (`URL + Header`): 68 records (0.84%)
  - Two sources (`Title + Header`): 0 records (0.00%)
- **Single Explicit Source (No conflict):** **1,083 records (13.39%)**
  - Single URL (`single_url`): 901 records (11.14%)
  - Single Header (`single_header`): 182 records (2.25%)
  - Single Title (`single_title`): 0 records (0.00%)

### 4.3 Top Document Types in High-Confidence Population
1. `Quyết định`: 2,627 (32.48%)
2. `Thông tư`: 2,523 (31.19%)
3. `Nghị định`: 1,131 (13.98%)
4. `Công văn`: 504 (6.23%)
5. `Nghị quyết`: 285 (3.52%)
6. `Tiêu chuẩn quốc gia` (TCVN): 269 (3.33%)
7. `Thông tư liên tịch`: 179 (2.21%)
8. `Luật`: 136 (1.68%)
9. `Quy chuẩn kỹ thuật quốc gia` (QCVN): 126 (1.56%)
10. `Hướng dẫn`: 63 (0.78%)

---

## 5. Logical Sidecar Chunk Propagation Coverage

Evaluating propagation across existing serving chunks without modifying `records.jsonl`:
- **Total Serving Chunks:** 330,768
- **Chunks with HIGH_CONFIDENCE Complete Document Identity:** **304,504 chunks (92.06%)**
- **Chunks without Complete Identity:** 26,264 chunks (7.94%)

---

## 6. D0 1,333 Retrieval-Proxy Target Identity Coverage

The D0 gold retrieval proxy population was frozen deterministically (`(question_id, target_document_id)`):
- **Frozen Proxy Population SHA-256:** `41f13e0bbe8cf91ef12f1cf1bfd325c6bfab6c354c7d874ab12aaf9b2e1f4f9c`
- **Total Proxy Questions:** 1,333
- **Proxy Questions with Covered Target Document Identity:** **1,299 questions (97.45%)**
- **Proxy Questions Uncovered:** 34 questions (2.55%)
- **Unique Proxy Target Documents:** 823 total
  - Covered target documents: **796 documents (96.72%)**
  - Uncovered target documents: 27 documents (3.28%)

---

## 7. Pre-Registered Feasibility Gates & Final Decision

| Feasibility Gate | Pre-Registered Threshold | Achieved Metric | Status |
| :--- | :--- | :--- | :--- |
| **Gate A: Corpus Non-Empty Coverage** | $\ge 50.0\%$ | **95.03%** (8,089 / 8,512) | **`PASSED`** |
| **Gate B: 1,333 Retrieval Proxy Coverage** | $\ge 70.0\%$ | **97.45%** (1,299 / 1,333) | **`PASSED`** |

### Final Checkpoint Decision:
$$\mathbf{D1A\_FEASIBILITY\_PASS}$$

---

## 8. Feasibility Evidence Archive

The content-safe diagnostic evidence package was generated and verified:
- **Archive Path:** `C:\Users\Nguyen\Downloads\data-d1a-document-identity-feasibility-evidence.zip`
- **SHA-256:** `8c3b3a1f2a74257f3b265e657a1f38975da67777deb8dafa016be029a0d772f3`
- **Size:** 6,311 bytes
- **Member Count:** 9 JSON files
  - `execution/source_identity.json`
  - `execution/extraction_contract.json`
  - `results/identity_coverage.json`
  - `results/source_agreement.json`
  - `results/proxy_identity_coverage.json`
  - `results/ambiguous_summary.json`
  - `results/d1a_decision.json`
  - `results/high_confidence_samples.json` (20 samples)
  - `results/unresolved_samples.json` (20 samples)

---

## 9. Next Actions & Pipeline Status

1. **Phase D1-A Checkpoint Committed:** Harness `scripts/evaluate_document_identity_d1.py`, unit tests `tests/unit/evaluation/test_document_identity_d1.py`, specification `docs/35-D1-DOCUMENT-IDENTITY-ENRICHMENT.md`, and evidence archive.
2. **Production Invariants Preserved:** Zero modification to production adapters, chunkers, chunk artifacts, BM25 indices, vector indices, reranker, generator, G1, or verifiers.
3. **Next Step:** External review of Phase D1-A feasibility evidence, followed by authorization to proceed to **Phase D1-B: BM25 Causal A/B Evaluation**.
