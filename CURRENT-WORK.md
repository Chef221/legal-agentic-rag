# Current Work — UIT DSC 2026 LegalQA

**Canonical Git Branch:** `main`
**Architectural Lineage Status:** `M49→M53 CLOSED (ARCHITECTURAL RESET)`
**Closure Authority:** [`M49_M53_ARCHITECTURAL_RESET_HANDOFF_2026-09-02.md`](M49_M53_ARCHITECTURAL_RESET_HANDOFF_2026-09-02.md)
**Next Authorized Engineering Work:** `M54.0 — Raw Legal Corpus Forensics + Preprocessing V2 Design`
**Updated:** 2026-09-02

---

## 1. Architectural Lineage Closure & M54 Reset

The **M49→M53 architectural lineage is formally CLOSED**.

- **Closure Authority**: [`M49_M53_ARCHITECTURAL_RESET_HANDOFF_2026-09-02.md`](M49_M53_ARCHITECTURAL_RESET_HANDOFF_2026-09-02.md).
- **Status of M49/M50/M52/M53**: All existing artifacts, models, checkpoints, traces, and metrics remain **baselines and forensics only**.
- **Prohibitions**:
  - Do NOT rerun M52 full SFT.
  - Do NOT rescue M53.1 / M53.2B / M53.3.
  - Do NOT blindly rerun O2 projector training.
  - Do NOT burn `INTERNAL_TEST`, `Public-1000`, or `Holdout` splits.
- **Next Authorized Work**:
  - **`M54.0 — Raw Legal Corpus Forensics + Preprocessing V2 Design`**.
  - **Constraint**: No full-corpus re-embedding or new model training is authorized until M54.0 corpus forensics and design are complete.

---

## 2. Canonical Baseline (M49.1-JINA35 Historical Control)

Milestone **M49.1-JINA35** remains the historical benchmark baseline on `main`.

- **History-Preserving Main Adoption Merge:** `6ebc0e5bde118e8c83e810251557a2f66c69a0d8`
- **Pre-Promotion GitHub Main Preserved At:** `archive/main-before-m491-canonical-e1a7916` (`e1a79162c394411ab45349353678be4278dcce71`).
- **Canonical Promotion Branch Preserved At:** `promotion/m491-jina35-canonical` (`0640a4289a098b1946c5121de90446e0d4dd09d5`).

### Authoritative Final Benchmark Result

- **Codabench ROUGE-L:** `0.496260842`
- **Codabench METEOR:** `0.406858976`
- **Official Submission Status:** `1000 / 1000` Valid Answers (0 blank, 0 invalid objects, strict-valid loader contract verified)

### Milestone Benchmark Comparison

| Candidate / Milestone | Reranker Backend | ROUGE-L | METEOR | Status |
|---|---|---:|---:|---|
| **M48 Control** | `Qwen3-Reranker-0.6B` | 0.3631401334440235 | 0.2685876695455311 | Retained Control |
| **M49.1 Baseline** | `Qwen3-Reranker-0.6B` | 0.473653736 | 0.382772249 | Prior Milestone Baseline |
| **M49.1-JINA35 (Final)** | `jina-reranker-v3.5` | **0.496260842** | **0.406858976** | **Historical Baseline** |
| **Absolute Delta vs M49.1** | — | **+0.022607106** | **+0.024086727** | — |
| **Absolute Delta vs M48** | — | **+0.133120708556** | **+0.138271306455** | — |

---

## 3. Final Evaluated Production Authority Hashes

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

## 4. Operational Policy & Governance

1. **Closed Lineage Policy:**
   M49→M53 lineage is **CLOSED**. Artifacts are preserved strictly for baseline comparison and forensics.
2. **Next Authorized Work:**
   **M54.0 — Raw Legal Corpus Forensics + Preprocessing V2 Design**. No training or re-embedding prior to M54.0 completion.
3. **Detailed Documentation:**
   Consult [`M49_M53_ARCHITECTURAL_RESET_HANDOFF_2026-09-02.md`](M49_M53_ARCHITECTURAL_RESET_HANDOFF_2026-09-02.md) and [`docs/21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md`](docs/21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md).
