# M49.1 Reranker Research Story

## 1. Where We Started

The project inherited an authoritative production baseline:
- **Repository Baseline:** `lkey07/legal-agentic-rag`
- **Commit:** `10681c8c05008432cd1c7170cd3917f4317c1c69`
- **Retained Public M49.1 Benchmark:**
  - **ROUGE-L:** `0.473653736`
  - **METEOR:** `0.382772249`

M49.1 established significant architectural gains over M48 by deploying a coherent stack:
- Embedding: `Qwen/Qwen3-Embedding-0.6B` (`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`)
- Control Reranker: `Qwen/Qwen3-Reranker-0.6B` (`e61197ed45024b0ed8a2d74b80b4d909f1255473`)
- Generator: `Qwen/Qwen2.5-3B-Instruct`
- Candidate Pool: $K=40$
- Evidence Selector: Title-strict & article-bounded filtering

Because M49.1 demonstrated state-of-the-art grounded generation performance across official evaluation protocols, it serves as the **immutable control baseline** for all subsequent research.

---

## 2. Early Improvement Attempts

Following the establishment of M49.1, extensive empirical cycles explored multiple avenues to boost end-to-end extraction and reranking accuracy.

### Summary of Diagnostic Cycles

| Cycle | Question | Hypothesis | Experiment | Observed Result | Interpretation | Decision |
|---|---|---|---|---|---|---|
| **T5-1** | Can prompt structure improve extraction? | Schema formatting unlocks better chunk selection | Generator/parser prompt contract tuning | Minor noise on Dev20 | Generation formatting is not the primary bottleneck | Keep M49 generator contract frozen |
| **T5-2** | Does heuristic evidence selection help? | Aggressive deduplication removes distracting passages | EvidenceSelector heuristic thresholds | ROUGE-L degradation on validation sets | EvidenceSelector already operates near optimal selectivity | Keep EvidenceSelector frozen |
| **T5-3** | Can metadata scalar calibration boost ranking? | Adding legal field / doc type bonus improves top-1 | Metadata score weighting & scalar fusion | Marginal gain on train, regression on fresh splits | Hardcoded weights overfit training distribution | Reject scalar fusion |
| **T5-4** | Can LoRA fine-tuning specialize the reranker? | LoRA on Qwen reranker adapts to Vietnamese legal queries | Reranker LoRA fine-tuning (Dev20) | High Dev20 gain (+0.04 ROUGE-L) | Appeared promising on development set | Advance to fresh validation |
| **T5-5** | Does LoRA generalize to unseen questions? | Dev20 LoRA gains hold on unexposed data | Raw adapter evaluation on Fresh80 | Severe regression on Fresh80 (-0.035 ROUGE-L) | Heavy distribution shift / adapter overfitting | **CLOSED** (Adapter rejected) |
| **T5-6** | Can hybrid priors guard against reranker errors? | Blending BM25/Dense RRF priors prevents catastrophic reranker drops | Fresh80-derived hybrid-prior guard | Positive on Fresh80 | Overfitted to Fresh80 failure modes | Advance to Fresh100 validation |
| **T5-7** | Does hybrid-prior guard hold on fresh queries? | Guard generalizes across unseen distributions | Fresh100 guard validation | Guard failed on Fresh100 | Guard was an artifact of Fresh80 exposure | **CLOSED** (Heuristic guard rejected) |

**Key Takeaway:** DEV SUCCESS $
eq$ CLEAN VALIDATION SUCCESS. Failed and exposed populations (Dev20, Fresh80, Fresh100) are permanently consumed and must never be retuned.

---

## 3. The Strategic Reset

Rather than continuing to iterate on post-hoc heuristic patches, the team halted candidate testing to execute a foundational upper-bound forensic:

> *"Where is the largest real recoverable score headroom in the M49.1 retrieval pipeline?"*

### Exact M49.1 Error Decomposition

On the benchmark evaluation pool:
- **First-stage Top-1:** `0.463578859`
- **Reranker / Actual Selected E1:** `0.519984606`
- **Selected E1–E10 Oracle (best candidate in selected):** `0.613574290`
- **Reranked Top-10 Oracle (best candidate in reranked top-10):** `0.617464668`
- **First-stage Pool Oracle (best candidate in $K=40$ pool):** `0.636813024`

### Headroom & Loss Breakdown
- **Rerank Truncation Loss ($K=40 	o 10$):** `0.019348355` (16.56% of headroom)
- **EvidenceSelector Pruning Loss ($10 	o 	ext{selected}$):** `0.003890378` (3.33% of headroom)
- **Selected Top-1 Ranking Loss (E1 vs Selected Oracle):** `0.093589684` (**80.11% of headroom**)
- **Total Within-Pool Recoverable Headroom:** `0.116828417`

### Structural Breakdown of Selected Top-1 Regret
When the selected top-1 was suboptimal compared to the oracle candidate within the pool:
1. **DIFFERENT_DOCUMENT:** 26 cases (**61.90%** of selected top-1 regret)
2. **SAME_DOCUMENT_DIFFERENT_ARTICLE_OR_SECTION:** 10 cases (**23.51%** of selected top-1 regret)
3. **SAME_DOCUMENT_SAME_ARTICLE_DIFFERENT_CHUNK:** 9 cases (**14.60%** of selected top-1 regret)

**Interpretation:** The dominant bottleneck in M49.1 was **cross-encoder document discrimination** (ranking accuracy), NOT EvidenceSelector pruning or generation schema. Research shifted decisively from selector/generator adjustments to **evaluating high-capacity rerankers**.

---

## 4. Why Jina v3.5

- **Model Identifier:** `jinaai/jina-reranker-v3.5`
- **Revision:** `e8a93f33f0b22108f8c2364f8484ce3422552fbc`
- **Approved Competition Model:** YES (Open weights, permissive license, officially registrable)
- **Parameter Count:** `596,836,352` (0.597B)
- **Candidate Stack Total:** $600	ext{M (Embedding)} + 596.8	ext{M (Jina)} + 2.115	ext{B (Generator)} = \mathbf{3,311,750,784} < 4,000,000,000$

### Architectural Advantage
Unlike standard pointwise/pairwise cross-encoders (which score query-document pairs independently), Jina v3.5 employs a native **listwise cross-attention architecture** supporting context lengths up to 12,288 tokens. This enables the model to observe all $K=40$ candidates simultaneously and evaluate relative legal relevance directly across candidate boundaries.

---

## 5. Exposed Fresh80 Probe

A diagnostic probe on the Fresh80 population compared Qwen control against Jina v3.5:

- **ROUGE-L:**
  - Control: `0.519984606`
  - Jina v3.5: `0.554557080`
  - Delta: `+0.034572474`
  - 95% Bootstrap CI: `[+0.0081676, +0.0627994]`
  - $P(\Delta > 0)$: `0.9964`
- **METEOR:**
  - Control: `0.418761459`
  - Jina v3.5: `0.460491716`
  - Delta: `+0.041730258`
  - 95% Bootstrap CI: `[+0.0152179, +0.0713287]`
  - $P(\Delta > 0)$: `0.9998`

While highly statistically significant, Fresh80 had prior research exposure. Therefore, this result served as strong diagnostic evidence to authorize a formal, unexposed validation study.

---

## 6. Clean100 Design & Two-Phase Reference Firewall

To prevent any data contamination or post-hoc bias, a strict two-phase protocol was constructed:

```
[Phase 1: Zero-Reference Inference]
Clean100 Questions Only (No Reference Answers)
       │
       ▼
10681c8 Detached First-Stage Retrieval (Shared Candidate Pools, K=40)
       ├──► Control (Qwen3-Reranker-0.6B) ──► EvidenceSelector ──► Control Evidence
       └──► Candidate (Jina-Reranker-v3.5) ──► EvidenceSelector ──► Jina Evidence
       │
       ▼
Cryptographic Manifest Frozen (SHA-256 Checksums Logged)

──────────────────────── FIREWALL ────────────────────────

[Phase 2: Scored Evaluation]
Input: Frozen Phase 1 Evidence + train.json References + Official Scorer
Output: Non-negotiable metrics & paired bootstrap statistics
```

- **Population Size:** $n = 100$ unexposed legal questions.
- **Rules:** Zero tuning, zero threshold adjustments, zero prompt tweaks, zero post-hoc exclusions.

---

## 7. Engineering Journey & Harness Lessons

During harness development, several pre-inference runtime issues were discovered and resolved:

| Event | Root Cause | Candidate Evidence Affected? | Reference Exposed? | Valid Run Produced? | Engineering Lesson |
|---|---|---|---|---|---|
| **E1: AutoModel Class** | `AutoModelForSequenceClassification` failed on Jina v3.5 | NO (Pre-inference fail) | NO | NO | Custom architectures require `AutoModel` with `trust_remote_code=True`. |
| **E2: Native Signature** | Passed `max_length` to `.rerank()`, raising `TypeError` | NO (Pre-inference fail) | NO | NO | Jina sets cap via `_tokenizer.model_max_length = 12288`, not `.rerank(max_length=...)`. |
| **E3: HF Cache Layout** | Kaggle cache lacked `refs/main` pointer | NO (Pre-inference fail) | NO | NO | Snapshot cache directory requires direct target mapping or local pointer. |
| **E4: Baseline Config Symbol** | `AppConfig` imported instead of `ApplicationConfig` | NO (Pre-inference fail) | NO | NO | Baseline commit `10681c8` defines `ApplicationConfig(BaseModel)`. |
| **E5: Authority Zip Structure** | `m491_control_source_10681c8.tar.gz` omitted `scripts/` | NO (Pre-inference fail) | NO | NO | Detached source packaging must encapsulate `src/` root cleanly. |
| **E6: Scorer Metric API** | Scorer proxy expected `score_meteor` alongside ROUGE-L | NO (Pre-scoring fail) | NO | NO | Exact BTC official scoring contract requires whitespace NLTK METEOR and ASCII ROUGE-L. |

All issues were mechanical harness bugs resolved prior to unblinding reference data.

---

## 8. Clean100 Frozen Validation Results

### Exact Phase-1 Cryptographic Authorities
- Shared Candidate Pools: `45a9bd9716f14c7a5a72c54bd82f5ee17a822caa56a26a6a3998f8234e899bb0`
- Control Reranked: `d775bd99c2820bc9874247d1e69eb2b439a47ec6c799e172d2a55cb189a0efc9`
- Jina Reranked: `eaafc39d9e3a5e5b11949d5546fea1b7b4da058cf56d99d463a1b2e642e337c9`
- Control Selected: `7cf9fb780ff7685f342e46a265ee0e3343213d301d580b942415322205b4ea5a`
- Jina Selected: `cb53ea5adc297296c8a01c93b7dcc811f9db6bcf0f0a88310b7d14a2a0406690`
- Phase-1 Manifest: `2f733ac8a2d1d5ca94c8f18844226865f598b21f4a109959daf9bef4ea3992c3`

### Exact Phase-2 Scored Benchmark ($n=100$)

| Metric | Control (Qwen3-0.6B) | Candidate (Jina-v3.5) | Delta ($\Delta$) | 95% Bootstrap CI | $P(\Delta > 0)$ | Gate Status |
|---|---|---|---|---|---|---|
| **ROUGE-L** | `0.486886682` | `0.524012464` | `+0.037125782` | `[+0.0053798, +0.0697749]` | `0.9884` | **PASS** |
| **METEOR** | `0.369941800` | `0.409442224` | `+0.039500424` | `[+0.0085795, +0.0712050]` | `0.9939` | **PASS** |

### Pairwise Breakdown
- **Pairwise Wins:** 30
- **Pairwise Losses:** 16
- **Pairwise Ties:** 54
- **Top-1 Rank Changes:** 46
- **Large Wins ($\Delta > +0.20$):** 13
- **Catastrophic Regressions ($\Delta < -0.20$):** 4
- **Oracle Captures:** 19
- **Oracle Losses:** 5
- **First-Stage Pool Oracle:** `0.624540299`

All three pre-registered gates passed with high statistical significance ($P > 0.98$).

---

## 9. Validation Evidence Authority

- **Complete Evidence Archive:** `m491_jina35_clean100_validation_evidence.zip`
  - **SHA-256:** `f8ee98878b8cf6d6268879957e4ca4b95f3d21e7bdec9c474f4c71a03d8b6f39`
- **Phase-2 Recovery Summary Hash:** `3533eb31f364c0aff584bf89a6d429a6446c8d74a19d753a9bbdf9ddeb943275`
- **Per-QID Metric Record Hash:** `93bae7d8e114b23d5b6d84b3ac68536445f6dc424c429e6044948c93c731f4a8`

*(Zero raw reference answers are committed to repository tracking).*

---

## 10. What We Learned

1. **Ranking is the Primary Driver:** The M49.1 pipeline was not fundamentally limited by evidence filtering heuristics, but by cross-encoder document ranking accuracy within the top-40 retrieved candidate pool.
2. **Listwise Attention Recovers Headroom:** Jina v3.5 successfully unlocked $+0.037$ ROUGE-L and $+0.039$ METEOR over the pointwise Qwen baseline across completely unexposed data.
3. **Clean Validation Protocol is Essential:** Maintaining a strict two-phase firewall ensured all conclusions are reproducible and untainted by overfitting.
4. **Clean100 is Now Consumed:** Clean100 is an evaluation benchmark and must never be used for training, prompt optimization, or heuristic guard tuning.

---

## 11. Decision

```text
RESEARCH_VALIDATION           = PASS
JINA_PRODUCTION_INTEGRATION   = AUTHORIZED
PRODUCTION_PROMOTION          = PENDING_MECHANICAL_INTEGRATION_GATE
M49_1_CONTROL                 = IMMUTABLE
CLEAN100_ROLE                 = CONSUMED_VALIDATION_SET_DO_NOT_TUNE
```

---

## 12. Next Chapter: Mechanical Integration & T4 Smoke

Phase A completes the integration of `JinaNativeReranker` as a selectable backend, preserving legacy control. Phase B will execute:
1. **Mechanical Parity Gate (Gate A):** Verify production `JinaNativeReranker` produces identical outputs to frozen Clean100 Phase-1 files.
2. **T4 Runtime Smoke (Gate B):** Verify stable coexistence and VRAM headroom under Tesla T4 GPU execution.


---

## 13. Phase A.1 — Pre-GPU Contract Hardening & Quality Gate Audit

Following the successful exact-baseline rebuild on root commit `10681c8c05008432cd1c7170cd3917f4317c1c69`, pre-GPU strategic source review rejected the initial Phase A mechanical runner and configuration draft due to several integration-engineering defects:

1. **Config Isolation Drift Caught & Fixed:**
   - The initial candidate configuration draft had drifted outside `online.reranker` (retrieval strategy, generation model identifier, token budgets).
   - In Phase A.1, `configs/uit-dsc-2026-task2-m491-jina35.example.json` was programmatically regenerated from exact `10681c8` control `configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json`.
   - Comprehensive recursive AST/JSON tests proved exactly **zero** differing paths outside `online.reranker.*`.
2. **Phase-1 JSONL Schema Realignment:**
   - Gate A was realigned from single JSON objects to the authoritative line-by-line JSONL contract (`clean100_shared_candidate_pools.jsonl` SHA `45a9bd97...`, `clean100_jina_reranked.jsonl` SHA `eaafc39d...`).
   - `RetrievalHit.model_validate()` is used to reconstruct candidates byte-identically.
   - Result mapping was verified against exact V4 runner authority `scripts/t5_jina35_clean100_runner.py` (SHA `ac46aae...`).
3. **Runtime API & Telemetry Contract Hardened:**
   - Gate B was corrected to invoke `runtime.answer(query: RetrievalQuery)` rather than string/dict wrappers.
   - `JinaNativeReranker` was wrapped in explicit `with torch.no_grad():`, with fail-closed checks on context cap (`native_context_cap=12288`), tokenizer presence, and device-dtype contracts (CUDA float16, CPU float32).
   - Completion telemetry persists `actual_parameter_device` and `actual_parameter_count` without sensitive query/document text.
4. **Research Integrity:**
   - Candidate model evidence was NOT affected.
   - Zero GPU compute was consumed.
   - Zero Clean100 reference answers were opened.
   - All engineering defects were caught and corrected before any GPU execution could occur.


---

## 14. Phase A.2 — Final Pre-GPU Mechanical Closure & Test Restoration

A second pre-GPU strategic review identified final mechanical harness and telemetry defects prior to GPU authorization:

1. **Gate A Full-K Parity Harness Correction:**
   - The initial Gate A parity call requested `top_k=10`, causing the production wrapper to truncate ranked candidates to 10 items, which mathematically prevented full K=40 mechanical parity evaluation against the 40-item frozen Clean100 authority.
   - Corrected Gate A to explicitly request `top_k=len(candidate_hits)` (40 candidates), validating full K=40 ranking sequence, derived top-10, derived top-1, and chunk-ID aligned scores.
   - Added a positive synthetic 100-QID test proving full 40-candidate parity passes cleanly.
2. **Gate B `AgentRunResult` Telemetry Contract Fixed:**
   - In M49.1 (`10681c8`), `OnlineRuntime.answer(query)` returns an `AgentRunResult` containing `.response` (`AnswerResponse`), `.state` (`AgentState`), and `.stop_reason`.
   - Gate B was corrected from direct/`hasattr` access to proper object extraction, recording real operational metrics (`answer_length`, `selected_evidence_count`, `stop_reason`, `insufficient_evidence`, `retrieval_strategy`, `retry_count`, `warning_count`, `latency_seconds`, `vram_mb`).
3. **Full Test Suite & Coverage Restoration:**
   - Restored all 10 original Phase A tests in `tests/unit/reranking/test_jina_native.py` and expanded with Phase A.1/A.2 contract tests to a total of 17 tests.
   - Added background heartbeat thread (15s interval) and strict fail-closed JSONL authority validation.
   - Total workspace pytest suite reached **464 passed** (0 failed, 1 skipped).
4. **Research Integrity:**
   - Zero GPU compute was consumed locally.
   - Zero Clean100 reference answers were opened.
   - All harness bugs were resolved before packaging for Kaggle execution.


---

## 15. Phase A.3 — Immutable Pre-GPU Checkpoint & Execution Authority Freeze

Prior to any GPU execution on Kaggle, the engineering authority was frozen into an immutable checkpoint:

1. **Static Test Suite & Coverage Verification:**
   - 465 passed, 1 skipped across full pytest suite (100% pass rate).
   - 36 targeted tests verified covering all JinaNativeReranker contracts, configuration isolation, Gate A/B CLI, BackgroundHeartbeat, and strict fail-closed validation.
2. **Exact Competition Parameter Budget Authority:**
   - Documented in `docs/artifacts/m491-jina35-parameter-budget-authority.json`:
     - Dense Embedding (`Qwen/Qwen3-Embedding-0.6B` @ `97b0c61...`): 596,000,000 params.
     - Candidate Reranker (`jinaai/jina-reranker-v3.5` @ `e8a93f3...`): 596,836,352 params.
     - Generator (`/kaggle/working/m49-generator-merged`): 2,118,914,432 params.
     - Semantic Verifier (Disabled): 0 params.
     - **Exact Total Candidate Stack:** **3,311,750,784** parameters ($< 4,000,000,000$ limit, 17.21% headroom).
3. **Pre-GPU Local Code Authority Commit:**
   - Created local commit `90de9a9d813df87432bc9183f8edebd4ed1f0b24` on branch `m491/jina35-production-integration`.
   - Tracked control config (`configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json`) is confirmed UNCHANGED from root control `10681c8c05008432cd1c7170cd3917f4317c1c69` (Git object SHA: `a38bc642f0e4bf006d624ccb1f56721775c5d9aa4a4b24cf82abe5ed52046be6`).
4. **Reproducible Kaggle Execution Authority Bundle:**
   - Packaged deterministic ZIP `m491_jina35_production_gate_v1.zip` (SHA `be32b11284fd627750d0afa17723e625522d1cf5c26dac5f58715e128d8ca711`, 151 members).
   - Contains zero training data, zero gold reference answers, and zero scoring programs.
5. **Current Operational Status:**
   - `GATE_A_HARNESS_VERIFIED = PASS`
   - `GATE_B_HARNESS_VERIFIED = PASS`
   - `REAL_GATE_A_FROZEN_CLEAN100 = NOT_RUN`
   - `REAL_GATE_B_T4 = NOT_RUN`
   - `PRODUCTION_PROMOTION = PENDING`


---

## 16. Phase A.3.1 — Final Authority Reconciliation & Parameter Provenance

Following external strategic review of Phase A.3 artifacts:
1. **Two-Commit Lineage Reconciliation:**
   - `M491_JINA35_EXECUTION_CODE_AUTHORITY`: Commit `90de9a9d813df87432bc9183f8edebd4ed1f0b24` (frozen pre-GPU execution package).
   - `M491_JINA35_ORCHESTRATION_DOC_AUTHORITY`: Commit `e37c357a59f6f12447bb3dbe930da38d4461bedd` (added Kaggle cells and updated ledger).
   - Verified that zero executable source, configuration, or runner scripts changed between Commit A and Commit B.
   - Preserved `m491_jina35_production_gate_v1.zip` (`be32b11284fd627750d0afa17723e625522d1cf5c26dac5f58715e128d8ca711`) without needless rebuild.
2. **Parameter-Budget Provenance Rigor:**
   - Distinguished historical registered substitution total ($3,311,750,784$ params) from independently proven safetensors component tensor numels:
     - `Qwen/Qwen3-Embedding-0.6B`: 595,776,512 params (proven from 310 safetensors tensor shapes).
     - `jinaai/jina-reranker-v3.5`: 596,836,352 params (proven from 312 safetensors tensor shapes).
     - M49 Merged Generator: dynamic weights produced on Kaggle, exact numel status `PENDING_GATE_B_PREFLIGHT`.
   - Gate B preflight verifies live initialized parameter counts before running test questions.
