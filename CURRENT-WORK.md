# Current Work — UIT DSC 2026 LegalQA

**Status:** `CLOSED` / `COMPLETE` (Evaluation & Submission Phase)
**Latest Evaluated Benchmark Authority:** `M49.1-JINA35`
**Updated:** 2026-08-25

---

## 1. Executive Status

Milestone **M49.1-JINA35** Public-1000 execution, validation, submission packaging, and official Codabench scoring are **CLOSED and COMPLETE**.

There is **ZERO** pending Public-1000 execution work.

### Authoritative Final Benchmark Result

- **Codabench ROUGE-L:** `0.496260842`
- **Codabench METEOR:** `0.406858976`
- **Official Submission Status:** `1000 / 1000` Valid Answers (0 blank, 0 invalid objects, strict-valid loader contract verified)

### Milestone Benchmark Comparison

| Candidate / Milestone | Reranker Backend | ROUGE-L | METEOR | Status |
|---|---|---:|---:|---|
| **M48 Control** | `Qwen3-Reranker-0.6B` | 0.3631401334440235 | 0.2685876695455311 | Retained Control |
| **M49.1 Baseline** | `Qwen3-Reranker-0.6B` | 0.473653736 | 0.382772249 | Prior Milestone Baseline |
| **M49.1-JINA35 (Final)** | `jina-reranker-v3.5` | **0.496260842** | **0.406858976** | **Latest Evaluated Benchmark Authority** |
| **Absolute Delta vs M49.1** | — | **+0.022607106** | **+0.024086727** | — |
| **Absolute Delta vs M48** | — | **+0.133120708556** | **+0.138271306455** | — |

---

## 2. Final Evaluated Production Authority Hashes

- **Authoritative Combined Checkpoint SHA256:**
  `3e69f6d5f9f99fa997893f2fd9d263dd7f6cba932adef46e484e79da23724eec`
- **Final Official submission.zip SHA256:**
  `f11af3c9a4571ff8e8997716b39484bcf69f636b54af7c815ba44756ac2d9200`
- **Final submission.json SHA256:**
  `417a34c7ab785b7c90741dafec0031dddb7f57f2e71417d973e5973d7eb35618`
- **Submission Audit Report SHA256:**
  `24d743d8c8a6d5c886c6d4ddff422abd4264552fb2a8f63a444db9b67bad8bb6`
- **Public-1000 Authority Manifest SHA256:**
  `9f6a68a23eef049d058e481acff8659cabbf39c0e16b752ed9d596c59331c0da`
- **Public-1000 Authority Bundle SHA256:**
  `792ceb1e8cc9ae7185ac1a8b28fc47a372d524856d4098fcee5fc0e578b5e19e`
- **Archival Handoff Bundle SHA256:**
  `3647123bd9e47a737cc5849bb317f83eaceb8d056e3619f6901f881419a4ad0d`

---

## 3. Operational Policy & Next Repository Task

1. **Closed Work Policy:**
   M49.1-JINA35 Public-1000 execution is **CLOSED**. Closed work must not be rerun merely to verify or reproduce it. A future Public-1000 run is allowed only under a NEW explicitly stated hypothesis/objective with newly established execution authority.
2. **Next Repository Task — Source Code Reconciliation:**
   The next task is reconciling the Kaggle-proven Hotfix V1/V2 semantic changes against the local git branch (`m491/jina35-production-integration` at commit `3d713c8`):
   - Diff current git branch against the Kaggle-proven V1/V2 changes;
   - Integrate minimal semantic changes (Hotfix V1 raw question identity preservation in `answering.py` and Hotfix V2 dual-source anchored fallback in `evidence_selector.py`);
   - Add/update regression unit tests;
   - Run the full pytest suite.
   - Do **NOT** rerun benchmark inference.
3. **Detailed Documentation:**
   Consult [`docs/21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md`](docs/21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md) for the complete causal history, root cause analysis, and audit trail.
