# D1 — Deterministic Legal Document Identity Enrichment: Feasibility Report (Phase D1-A Strict Hardened)

## 1. Context & Motivation

Following the completion of the Phase D0 Official Dataset Census and Retrieval Unit Boundary Audit (`docs/34-DATA-CENSUS-AND-RETRIEVAL-UNIT-AUDIT-D0.md`, Decision **D129**), the repository identified that 0% of serving legal chunks currently possess extracted document identity metadata (`document_type`, `document_number`), and all 108,009 `token_fallback` chunks (32.65% of the corpus) carry empty header context in `search_text`.

Candidate **D1 — Deterministic Legal Document Identity Enrichment** was formalized in Decision **D130** to evaluate whether high-confidence official document identity can be safely and deterministically extracted solely from organizer data and prepended to chunk search representations to improve lexical retrieval precision.

This report documents the execution and results of **Phase D1-A: Deterministic Legal Document Identity Feasibility**, including the pre-D1-B **Strict Evidence Hardening Checkpoint**.

> [!IMPORTANT]
> **Check-point Scope & Interpretation:**
> A `D1A_STRICT_FEASIBILITY_PASS` decision proves **ONLY** that there is sufficient safe organizer-derived document identity across the official corpus to justify running the subsequent BM25 causal A/B evaluation.
>
> It does **NOT** mean retrieval has improved, production chunking or ingestion should change, METEOR has improved, or document identity should be permanently ingested into production artifacts.
>
> **Status:** `D1-A STRICT FEASIBILITY COMPLETE` — **`D1-B BM25 A/B NOT YET RUN`**.

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

## 3. Extraction Contract & Strict Multi-Channel Policy

### 3.1 Single Causal Variable Scope
Extraction in Phase D1-A was restricted strictly to two fields:
- `document_type`: Canonical Vietnamese legal document type (e.g. `Thông tư`, `Nghị định`, `Quyết định`, `Luật`).
- `document_number`: Canonical official legal number string (e.g. `99/2003/NĐ-CP`, `30/2022/TT-BTC`, `51/QĐ-VKSTC-V12`).

Extraction of any secondary metadata (issuing authority, issuance date, effective date, expiry date, effect status, legal field) was strictly prohibited.

### 3.2 Allowed Organizer Sources
Extraction candidates were derived solely from three organizer sources:
1. `title_candidate`: Extracted from context `name` slug.
2. `url_candidate`: Extracted from context `link` source URL slug.
3. `header_candidate`: Extracted from the deterministic early header/preamble region of the raw `passage`.

No external databases, no LLMs, no semantic guesses, no web crawls, and no synthetic mappings were utilized.

### 3.3 Own-Document Identity Safety Guard
Vietnamese legal documents frequently cite other legal acts in enacting preambles (e.g. `Căn cứ Luật...`, `theo Nghị định số...`). Passage header extraction in `scripts/evaluate_document_identity_d1.py` enforces strict structural guards:
- Extraction inspects only lines preceding substantive enacting formulas (`Căn cứ`, `Điều \d+`, `Chương [IVXLCDM]+`) or within the first 50 non-empty lines.
- Number extraction is strictly tied to formal header prefixes (`Số:`, `Số `).
- Broken multi-line uppercase type headings (e.g. `THÔNG \n TƯ`, `NGHỊ \n ĐỊNH`, `QUYẾT \n ĐỊNH`) are combined before type resolution.
- Body citations to referenced laws are strictly ignored.

### 3.4 Strict Multi-Channel vs Provisional Single-Source Policy
Following external review and prior to observing any BM25 retrieval outcome, the confidence policy was hardened:
- **Scientific Independence Principle:** Title/Name slug and URL slug belong to the single correlated **SLUG CHANNEL**.
- **`STRICT_MULTI_CHANNEL_IDENTITY` (D1-B Primary Candidate Population):**
  - Requires agreement between Passage Header and at least one slug source:
    - `all_three` (`Title + URL + Header`), OR
    - `url_header` (`URL + Header`), OR
    - `title_header` (`Title + Header`).
  - Only this strict population is eligible for Phase D1-B candidate enrichment.
- **`PROVISIONAL_SINGLE_SOURCE` (Diagnostic Only):**
  - Identities derived from `title_url_only` (slug channel only), `single_url`, `single_header`, or `single_title`.
  - Must **NOT** be used in D1-B candidate search text representations.
- **`AMBIGUOUS` (Fail Closed):** Conflicting candidates across sources.
- **`UNRESOLVED` (Fail Closed):** No complete candidate extracted.

---

## 4. Full-Corpus Feasibility Metrics

The extraction harness was executed over all 8,532 official contexts with zero estimation:

### 4.1 Document-Level Census & Coverage
| Metric | Count | All Contexts (8,532) | Non-Empty Contexts (8,512) | Titled Contexts (7,407) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Contexts** | 8,532 | 100.0% | — | — |
| **Non-Empty Contexts** | 8,512 | 99.77% | 100.0% | — |
| **Titled Contexts (`name != None`)** | 7,407 | 86.81% | 87.02% | 100.0% |
| **`STRICT_MULTI_CHANNEL_IDENTITY` (Complete Type + Number)** | **6,891** | **80.77%** | **80.96%** | **92.12%** (6,823/7,407) |
| **`PROVISIONAL_SINGLE_SOURCE` (Diagnostic Only)** | 1,198 | 14.04% | 14.07% | — |
| **Original Diagnostic Total (`HIGH_CONFIDENCE`)** | 8,089 | 94.81% | 95.03% | 95.67% (7,086/7,407) |
| **`AMBIGUOUS` (Conflicting candidates)** | 258 | 3.02% | 3.03% | — |
| **`UNRESOLVED` (No complete candidate)** | 185 | 2.17% | 2.17% | — |

*Note on Titled Contexts Coverage:* The titled-context percentage correctly computes the subset-specific numerator (strict titled identities / 7,407 titled contexts = 92.12%; diagnostic titled identities / 7,407 titled contexts = 95.67%). No reported percentage exceeds 100%.

### 4.2 Agreement Pattern Breakdown
Among all 8,532 contexts:
- **`STRICT_MULTI_CHANNEL`:** **6,891 records (80.77%)**
  - All three sources (`all_three`): 6,823 records (79.97%)
  - URL + Header (`url_header`): 68 records (0.80%)
  - Title + Header (`title_header`): 0 records (0.00%)
- **`PROVISIONAL_SINGLE_SOURCE`:** **1,198 records (14.04%)**
  - Title + URL only (`title_url_only`): 115 records (1.35%)
  - Single URL (`single_url`): 901 records (10.56%)
  - Single Header (`single_header`): 182 records (2.13%)
  - Single Title (`single_title`): 0 records (0.00%)
- **`AMBIGUOUS` (Conflict):** 258 records (3.02%)
- **`UNRESOLVED` (None):** 185 records (2.17%)

### 4.3 Top Document Types in Strict Multi-Channel Population
1. `Quyết định`: 2,425 (35.19%)
2. `Thông tư`: 2,347 (34.06%)
3. `Nghị định`: 1,070 (15.53%)
4. `Công văn`: 370 (5.37%)
5. `Nghị quyết`: 239 (3.47%)
6. `Tiêu chuẩn quốc gia` (TCVN): 181 (2.63%)
7. `Thông tư liên tịch`: 173 (2.51%)
8. `Quy chuẩn kỹ thuật quốc gia` (QCVN): 39 (0.57%)
9. `Luật`: 29 (0.42%)
10. `Hướng dẫn`: 18 (0.26%)

---

## 5. Logical Sidecar Chunk Propagation Coverage

Evaluating propagation across existing serving chunks without modifying `records.jsonl`:
- **Total Serving Chunks:** 330,768
- **Chunks with `STRICT_MULTI_CHANNEL` Document Identity:** **264,765 chunks (80.05%)**
- **Chunks with `PROVISIONAL_SINGLE_SOURCE` Document Identity:** 39,739 chunks (12.01%)
- **Chunks Uncovered (`AMBIGUOUS` / `UNRESOLVED`):** 26,264 chunks (7.94%)

---

## 6. Strict D0 1,333 Retrieval-Proxy Target Identity Coverage

The D0 gold retrieval proxy population was evaluated against `STRICT_MULTI_CHANNEL_IDENTITY`:
- **Strict Proxy Population SHA-256:** `2b29b553908e2bc1553d1e7402194390609e6f8ca68592bdb0f56200bafa4100`
- **Total Proxy Questions:** 1,333
- **Proxy Questions with Strict Target Document Identity:** **1,262 questions (94.67%)**
- **Proxy Questions Uncovered:** 71 questions (5.33%)
- **Unique Proxy Target Documents:** 823 total
  - Covered target documents: **770 documents (93.56%)**
  - Uncovered target documents: 53 documents (6.44%)

---

## 7. Pre-Registered Feasibility Gates & Final Decision

Evaluating pre-registered feasibility gates using the **Strict Primary Population**:

| Feasibility Gate | Pre-Registered Threshold | Strict Achieved Metric | Status |
| :--- | :--- | :--- | :--- |
| **Gate A: Corpus Non-Empty Coverage** | $\ge 50.0\%$ | **80.96%** (6,891 / 8,512) | **`PASSED`** |
| **Gate B: 1,333 Retrieval Proxy Coverage** | $\ge 70.0\%$ | **94.67%** (1,262 / 1,333) | **`PASSED`** |

### Final Checkpoint Decision:
$$\mathbf{D1A\_STRICT\_FEASIBILITY\_PASS}$$

---

## 8. Feasibility Evidence Archives

Both the original diagnostic archive and the hardened strict evidence package are preserved and checksum-verified:

### 8.1 Strict Feasibility Evidence Package
- **Archive Path:** `C:\Users\Nguyen\Downloads\data-d1a-document-identity-feasibility-strict-evidence.zip`
- **SHA-256:** `870ad1447e46b083bcdd8b7cd82e585509615a480c772ccdf229f358b50edbc5`
- **Size:** 6,448 bytes
- **Member Count:** 8 JSON files
  - `execution/source_identity.json`
  - `execution/strict_identity_policy.json`
  - `results/strict_identity_coverage.json`
  - `results/strict_proxy_coverage.json`
  - `results/d1a_strict_decision.json`
  - `results/strict_samples.json` (20 samples)
  - `results/provisional_samples.json` (20 samples)
  - `results/ambiguous_summary.json`

### 8.2 Original Diagnostic Feasibility Evidence Archive
- **Archive Path:** `C:\Users\Nguyen\Downloads\data-d1a-document-identity-feasibility-evidence.zip`
- **SHA-256:** `8c3b3a1f2a74257f3b265e657a1f38975da67777deb8dafa016be029a0d772f3`
- **Size:** 6,311 bytes (9 members)

---

## 9. Next Actions & Pipeline Status

1. **Phase D1-A Strict Checkpoint Committed:** Harness `scripts/evaluate_document_identity_d1.py`, unit tests `tests/unit/evaluation/test_document_identity_d1.py`, specification `docs/35-D1-DOCUMENT-IDENTITY-ENRICHMENT.md`, and strict evidence archive.
2. **Production Invariants Preserved:** Zero modification to production adapters, chunkers, chunk artifacts, BM25 indices, vector indices, reranker, generator, G1, or verifiers.
3. **Primary Population Frozen for D1-B:** D1-B BM25 causal A/B evaluation must use **ONLY** the 6,891 `STRICT_MULTI_CHANNEL_IDENTITY` documents.
4. **Next Step:** Authorize execution of **Phase D1-B: BM25 Causal A/B Evaluation**.
