# M49.1-JINA35 — Public-1000 Recovery, Hotfix V2, and Final Submission

## 1. Executive Summary

Milestone **M49.1-JINA35** upgraded the legal evidence reranking component of the UIT DSC 2026 Task 2 RAG system from `Qwen/Qwen3-Reranker-0.6B` to `jinaai/jina-reranker-v3.5` (revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`, 596,836,352 parameters) under the competition parameter limit (< 4B total stack).

Following initial offline Clean100 validation and dual-T4 mechanical integration, the system executed full **Public-1000** inference on Kaggle Dual-T4 hardware, resolved two critical competition-boundary root causes through non-semantic hotfixes (Hotfix V1/V2), and completed the full 1,000-question evaluation via an isolated Kaggle execution transport artifact.

### Official Codabench Benchmark Result

| Candidate / Milestone | Reranker Backend | ROUGE-L | METEOR | Status |
|---|---|---:|---:|---|
| **M48 Control** | `Qwen3-Reranker-0.6B` | 0.3631401334440235 | 0.2685876695455311 | Retained Control |
| **M49.1 Baseline** | `Qwen3-Reranker-0.6B` | 0.473653736 | 0.382772249 | Prior Milestone Baseline |
| **M49.1-JINA35 (Final)** | `jina-reranker-v3.5` | **0.496260842** | **0.406858976** | **Latest Evaluated Benchmark Authority** |
| **Absolute Delta vs M49.1** | — | **+0.022607106** | **+0.024086727** | — |
| **Absolute Delta vs M48** | — | **+0.133120708556** | **+0.138271306455** | — |

The submission achieved full official validity (**1,000 / 1,000 valid answers**, 0 blank, 0 invalid objects, 0 malformed records) under strict scoring contract compliance.

---

## 2. Starting State and Objectives

The system inherited:
- Base commit: `10681c8c05008432cd1c7170cd3917f4317c1c69`
- Canonical corpus: `selected-contexts.zip` revision `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e` (8,532 contexts)
- M45 offline indexes: SQLite FTS5 BM25 + Qwen3-Embedding-0.6B dense vector DB
- M49 merged generator: Qwen3.5-2B (SFT on official training QA pairs)
- Reranker candidate: `jinaai/jina-reranker-v3.5`

### Exact Total Active Learned Parameter Inventory

*(Proven from `docs/artifacts/m491-jina35-parameter-budget-authority.json` and unit test assertions)*

| Model Role | Model Name / Identifier | Parameters | Provenance |
|---|---|---:|---|
| Dense Embedding | `Qwen/Qwen3-Embedding-0.6B` | 595,776,512 | `docs/artifacts/m491-jina35-parameter-budget-authority.json` (Safetensors header) |
| Listwise Reranker | `jinaai/jina-reranker-v3.5` | 596,836,352 | `docs/artifacts/m491-jina35-parameter-budget-authority.json` (Safetensors header) |
| Generator (Merged) | `Qwen/Qwen3.5-2B` (M49 SFT) | 2,213,241,664 | `docs/artifacts/m491-jina35-parameter-budget-authority.json` (Instantiated model numel) |
| **Total Active Learned Stack** | — | **3,405,854,528** | `< 4,000,000,000` competition cap |
| **Remaining Parameter Headroom** | — | **+594,145,472** | **+14.85% headroom** |

---

## 3. Gate-C Runtime and Cache Authority Stabilization

Early physical execution on Kaggle Dual-T4 hardware encountered infrastructure and runtime environment discrepancies:

1. **HuggingFace Import-Order / Re-entry Environment Defect**:
   - `HF_HOME` and `HF_HUB_CACHE` environment variables set after initial package import in the parent kernel caused cache misses.
   - *Fix / Classification:* `KAGGLE_IMPORT_ORDER_OR_REENTRY_ENV_DEFECT`. Subprocess workers set environment variables as line 1 before any library import.
2. **Jina Cache Staging Missing `refs/main`**:
   - The pre-staged Kaggle dataset contained the exact commit hash `e8a93f33...` but lacked the Git reference pointer `refs/main`.
   - *Fix / Classification:* `PROVEN__SELECTIVE_STAGING_MISSING_JINA_REFS_MAIN`. Resolved reference pointer in staged cache.
3. **Dual-Worker Spawn Preflight**:
   - Validated independent OS child processes, isolated `CUDA_VISIBLE_DEVICES="0"` and `"1"`, correct environment constants, and local offline cache resolution.
4. **Physical Gate C Attempt #3**:
   - Successfully passed real dual-GPU smoke execution across 10 test questions (5 per GPU).

---

## 4. First Long Public-1000 Run and Quarantine

A long Dual-T4 Public-1000 execution progressed structurally to:
- Structural progress: **814 / 1000**
- Semantic audit: **39 invalid records detected**

### Reason for Quarantine
The 814 checkpoint was **QUARANTINED** (`ee7b8c21f1f1bea7ca2ed137ebf3a824db74d19ad2e31df19bab58b0fcd0d56e`) and explicitly **NOT** promoted as resumable production authority. In accordance with strict competition engineering standards, structural completion alone never upgrades checkpoint authority.

---

## 5. Salvage to Frozen Production Authority (775 / 1000)

The quarantined 814 checkpoint underwent rigorous semantic sanitization:
- **Worker 0:** 398 structural records -> **380 valid records** (18 invalid purged)
- **Worker 1:** 416 structural records -> **395 valid records** (21 invalid purged)
- **Total Invalid Purged:** 39 records
- **Final Salvaged Authority:** **775 / 1000 valid records**
- **Remaining Pending:** **225 / 1000 records**

### Checkpoint Lineage
- **Quarantined 814 Checkpoint SHA256:** `ee7b8c21f1f1bea7ca2ed137ebf3a824db74d19ad2e31df19bab58b0fcd0d56e`
- **Authoritative Sanitized 775 Checkpoint SHA256:** `b9fbfff4173e7704831b921cadfffed1d8e1a26c4386cc0534a604dba0a1611e`

This 775 checkpoint was frozen as the authoritative starting base for subsequent execution.

---

## 6. Failure Forensic and Root Cause Analysis

Investigation of the 39 invalid records proved two distinct root causes:

### 6.1 Root Cause #1 — Raw Question Whitespace and Identity Loss (38 / 39 rows)
- **Forensic Discovery:** 38 of the 39 contract failures occurred on official questions containing trailing or surrounding raw whitespace.
- **Population Audit:** Exactly 54 questions in Public-1000 contained raw surrounding whitespace (38 already executed and failed; 16 pending and at risk).
- **Mechanism:** The baseline `AnswerResponse` schema validator in `answering.py` applied `.strip()` normalization to all input text fields, including `question`. This caused the output field `question` to lose byte-exact identity with the official prompt, triggering output contract rejection at the competition boundary.
- **Classification:** Schema normalization defect on identity-bearing field.

### 6.2 Root Cause #2 — Explicit Document Identity Matching on QID 17789 (1 / 39 rows)
- **Question:** `Nghị định 26/2023/NĐ-CP được áp dụng từ ngày nào?`
- **Mechanism:** Retrieval hit metadata lacked `document_number`. The target document existed in corpus (`document_id: 210540`, title: `Nghi-dinh-26-2023-ND-CP-Bieu-thue-xuat-khau-Bieu-thue-nhap-khau-uu-dai-548616`, URL: `.../26/2023/ND-CP...`). However, the reference normalizer treated `NĐ` and `ND` as distinct, preventing metadata match and leading to `insufficient_evidence=true`.
- **Classification:** Strict metadata reference normalizer gap in the presence of unpopulated `document_number`.

---

## 7. Hotfix V1 — Raw Question Identity Preservation

Hotfix V1 modified `src/legal_agentic_rag/schemas/answering.py` within the isolated Kaggle execution bundle:
- **Baseline SHA256:** `2ece483d97128263cbe5ef245f78a6caed71230357d1a10f2b953394f204e98b`
- **Patched V1 SHA256:** `72c912761c00ee9d8def4f526fac611ce128a69cb554d2d5041ff171652fbc88`

### Logic Changes
1. Separated question validation from answer and trace ID normalization.
2. `question` validator rejects empty or whitespace-only inputs but preserves exact raw string bytes and trailing whitespace.
3. `answer` and `trace_id` normalization remained strictly unchanged.

### Static Verification
- Baseline schema raw-question mismatches on Public-1000: **54 / 1000**
- Patched schema raw-question mismatches on Public-1000: **0 / 1000**
- *Classification:* `PROVEN_STATIC_FIX__ANSWERRESPONSE_QUESTION_NOW_PRESERVES_RAW_IDENTITY__54_OF_54_WHITESPACE_MISMATCHES_ELIMINATED__ZERO_NONQUESTION_SEMANTIC_RELAXATION`.

---

## 8. Hotfix V2 — Conservative Explicit Document Identity Fallback

Hotfix V2 modified `src/legal_agentic_rag/generation/evidence_selector.py` (preserving Hotfix V1 `answering.py`):
- **Baseline/V1 Selector SHA256:** `cc7d2cdb804ee0b4cf17ae58786bafd6e253b64ef1998f2adf76a278b3bb0b44`
- **Patched V2 Selector SHA256:** `f35e337e18ef58ba45f9a1583ff7a8bde47ef8934596f0b91966c7d100c4c3b9`

### Logic Changes
- When a query contains an explicit legal document reference, `document_number` is missing in candidate metadata, and normal document matching fails:
  1. Applies conservative Vietnamese `Đ` / `ND` identity equivalence within the narrow document reference normalizer (`26/2023/NĐ-CP` <-> `26/2023/nd-cp`).
  2. Requires **dual-source anchoring**: BOTH `document_title` AND `source_url` basename must match the canonicalized document reference.
  3. Fails closed (no match) if document family is unsupported, title or URL is missing, only one source matches, or ambiguity exists (e.g., `03/2020/NQ-HĐND` mapping to multiple IDs).
  4. Never fabricates or assigns missing metadata fields.

---

## 9. Physical Microtest Proofs and Generator Fallback Forensic

1. **Whitespace Microtest**: Physically confirmed 0 contract errors on raw-whitespace prompts.
2. **QID 17789 Physical Microtest**:
   - Status: `answer_verified`
   - `success = true`, `insufficient_evidence = false`
   - Zero `generator:contract_mismatch`, zero `explicit_reference_not_selected`, zero `max_retry_reached`.
3. **Generator Fallback Forensic**:
   - QID 17789 emitted `generator_model_error_fallback` and logged `model_answer_draft_rejected`.
   - *Forensic Finding:* When structured generation is rejected and bounded retries are exhausted, the pipeline falls back to the deterministic top-evidence extractor. The resulting response still undergoes strict citation verification and schema validation. Emitting `generator_model_error_fallback` is a designed recovery path, not a fatal failure.

---

## 10. Execution Authority for Remaining 225 Questions

- **Frozen Base:** 775 / 1000
- **Pending Batch:** 225 questions
  - 54 raw-whitespace questions (38 retried + 16 new)
  - 1 separate QID 17789
  - 170 remaining unexecuted questions
- **Dual-Worker Preflight:** Verified in both workers: V2 selector SHA, V1 answering SHA, CUDA process isolation, HF cache constants, and Jina offline model resolution.
- **Classification:** `FULL PENDING-225 EXECUTION AUTHORIZED`.

---

## 11. Production Orchestration Multiprocessing Fix

### The Spawn Driver Defect
On initial resume launch, worker processes failed at startup before inference:
- `multiprocessing` with `"spawn"` context re-imports the main driver module.
- The module contained top-level assertions (such as verifying output directory did not yet exist).
- The parent process created the directory during checkpoint restore, causing children to fail top-level assertions upon re-import.
- *Durable Progress:* 0 new rows generated; original 775 rows preserved intact.

### The Orchestration Repair
- Encapsulated production logic inside `production_main()` protected by `if __name__ == "__main__":`.
- Added `multiprocessing.freeze_support()`.
- Added explicit spawn-only preflight test.
- Enforced fail-closed coordinator exit on `DUAL_GPU_WORKER_FAILURE_CHECKPOINT_READY`.

---

## 12. Final Public-1000 Execution

The repaired Dual-T4 production run resumed from the frozen 775 checkpoint without re-executing valid rows:
- **Starting Authority:** 775 / 1000
- **New Pending Executed:** 225 / 225
- **Worker 0 Completed:** 500 / 500
- **Worker 1 Completed:** 500 / 500
- **Global Completed:** **1,000 / 1,000**
- **Strict Invalid Rows:** **0**
- **Frozen Prefix Preserved:** True (W0 prefix preserved, W1 prefix preserved)

### Authoritative Output Checksum
- **Final Combined Checkpoint SHA256:** `3e69f6d5f9f99fa997893f2fd9d263dd7f6cba932adef46e484e79da23724eec`
- **Worker 0 Results SHA256:** `a1c0eee44f65d126cc34adde5a440681d1575bd4cd24118511838154892e61ea`
- **Worker 1 Results SHA256:** `234768d33d556547050b1ed371c22f4abc25440d6c5e23bc98e818db7590d29c`
- **Classification:** `PROVEN__HOTFIX_V2_PUBLIC1000_COMPLETE__FROZEN_775_PREFIX_PRESERVED__225_PENDING_EXECUTED__1000_OF_1000_STRICT_VALID`

---

## 13. Submission Packaging and Audit

Submission packaging was executed via the frozen implementation `DualPublic1000SessionRunner.package_final_submission()`:
- Merged worker outputs into canonical official QID order (1 to 1000).
- Generated deterministic one-member `submission.zip` containing UTF-8 `submission.json`.
- Validated via the frozen official submission loader.

### Final Submission Audit Metrics
- Official QIDs: 1,000 / 1,000
- Blank answers: 0
- Invalid objects: 0
- Canonical order: PASS
- Exact answer match: PASS
- Checkpoint & worker result immutability: PASS

---

## 14. Authoritative Artifact Lineage and Checksums

| Artifact Description | Filename | SHA256 Checksum |
|---|---|---|
| **Final Codabench Submission ZIP** | `submission.zip` | `f11af3c9a4571ff8e8997716b39484bcf69f636b54af7c815ba44756ac2d9200` |
| **Official Submission JSON** | `submission.json` | `417a34c7ab785b7c90741dafec0031dddb7f57f2e71417d973e5973d7eb35618` |
| **Authoritative Combined Checkpoint** | `public1000_dual_gpu_checkpoint_latest.zip` | `3e69f6d5f9f99fa997893f2fd9d263dd7f6cba932adef46e484e79da23724eec` |
| **Final Submission Audit Report** | `m491_FINAL_SUBMISSION_AUDIT.json` | `24d743d8c8a6d5c886c6d4ddff422abd4264552fb2a8f63a444db9b67bad8bb6` |
| **Public-1000 Authority Manifest** | `m491_FINAL_PUBLIC1000_AUTHORITY_1000.json` | `9f6a68a23eef049d058e481acff8659cabbf39c0e16b752ed9d596c59331c0da` |
| **Public-1000 Authority Bundle** | `m491_FINAL_PUBLIC1000_AUTHORITY_1000_3e69f6d5.zip` | `792ceb1e8cc9ae7185ac1a8b28fc47a372d524856d4098fcee5fc0e578b5e19e` |
| **Archival Handoff Bundle** | `m491_FINAL_SUBMISSION_AND_AUTHORITY_HANDOFF.zip` | `3647123bd9e47a737cc5849bb317f83eaceb8d056e3619f6901f881419a4ad0d` |

*(Note: `submission.zip` is the sole Codabench upload artifact. Authority and handoff bundles are internal archival records.)*

---

## 15. Key Engineering Lessons and Design Decisions

1. **Raw Identity Preservation on Competition Boundaries**:
   Validation logic on input fields must never apply mutative normalization (e.g. `.strip()`) to identity-bearing fields.
2. **Checkpoint Authority Discipline**:
   Structural completion is insufficient for promotion. Intermediate checkpoints with invalid records must be quarantined and sanitized before resuming.
3. **Immutability of Frozen Progress**:
   Resuming execution must strictly preserve previously validated record prefixes and never rerun valid questions.
4. **Dual-Source Anchored Fallback**:
   Missing metadata must not be fabricated. Fallback identity matching is valid only when dual independent sources (title and URL) strictly agree.
5. **Interpretation of Recovery Warnings**:
   Generator fallback (`generator_model_error_fallback`) is a designed graceful recovery mechanism. When citation and schema validation pass, the result is valid.
6. **Multiprocessing Spawn Invariants**:
   Modules executed via `spawn` must be import-safe. Top-level assertions or side effects must be guarded behind `if __name__ == "__main__":`.

---

## 16. Closed Work / Do-Not-Repeat

The following investigations, fixes, and experiments are **CLOSED**:

- Blind retries of physical Gate C infrastructure tests.
- Re-evaluation of the quarantined 814 checkpoint.
- Re-investigation of the 39 raw-whitespace and QID 17789 failures.
- Re-running Hotfix V1 or Hotfix V2 static/physical regression proofs.
- Re-running the completed 1,000-question Public evaluation.
- Modifying submission packaging format or loader validation.

**Policy:** Closed work must not be rerun merely to verify or reproduce it. A future Public-1000 run is allowed only under a NEW explicitly stated hypothesis/objective with newly established execution authority.

---

## 17. Repository Source Code Reconciliation

There is a clear architectural boundary between:
- **Repository Branch State:** Git branch `m491/jina35-production-integration` at commit `3d713c8` (which established full OS-multiprocessing runner isolation and 516 passing tests).
- **Evaluated Kaggle Artifact:** The isolated execution transport bundle (`792ceb1e...`) which executed on Kaggle hardware containing the patched Hotfix V1 `answering.py` (SHA256 `72c91276...`) and Hotfix V2 `evidence_selector.py` (SHA256 `f35e337e...`).

Integrating these proven semantic fixes into the git branch is designated as a separate future repository task. That task must:
1. Diff the current git branch against the Kaggle-proven V1/V2 changes;
2. Integrate the minimal semantic changes without blindly replacing whole files;
3. Add regression unit tests for raw question identity preservation;
4. Add regression unit tests for conservative explicit-document identity fallback;
5. Run the full repository test suite;
6. Only then consider the source fixes integrated into the branch.
