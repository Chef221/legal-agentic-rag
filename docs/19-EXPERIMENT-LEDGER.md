# 19. Experiment & Optimization Ledger

This document is the permanent, chronological engineering and evaluation ledger
for all optimization, experimental, architectural, and runtime changes from the
M49.1 takeover baseline onward.

Every future optimization phase must record a complete entry following the
standard schema before the phase is considered complete.

---

## Standard Entry Schema

Every entry must document:
- **Phase ID & Title**
- **Status & Date**
- **Objective & Hypothesis**
- **Authority**: Baseline SHA/tag, Candidate SHA, Branch, Parent commit, Artifact identities, Model/Generator identities, Scorer identity, Evaluation population.
- **Intended Change vs Actual Change**: Exact production and test files modified.
- **Validation & Numeric Results**: Exact numbers, paired deltas, causal telemetry.
- **Invalid / Discarded Runs**: Forensic failure analysis, root causes, and methodological lessons.
- **Decision & Rationale**: `ACCEPTED`, `REJECTED`, `INCONCLUSIVE`, or `SUPERSEDED`.
- **Production Impact & Known Risks**: Serving behavioral shifts, technical debt, limitations.
- **Next Step**: Concrete planned follow-up work.

---

## Historical Backfill Policy

1. The full Standard Entry Schema is mandatory for all experiments and phases from T5-3C onward.
2. Entries for M49.1, T4, and early T5 phases predating the creation of this permanent ledger are retrospective backfills.
3. For retrospective entries, the entry body together with the **Historical Backfill Completeness Matrix** below satisfies the Standard Entry Schema.
4. Every required field in retrospective entries must either:
   - be populated from verifiable repository / experiment authority; or
   - explicitly state `NOT RECOVERED FROM CURRENT AUTHORITY`; or
   - state `NOT APPLICABLE` where genuinely inapplicable.
5. Never infer missing commands, file lists, dates, metrics, artifact identities, or causal evidence merely to make a historical entry look complete.
6. A missing historical field is an uncertainty to preserve, not a reason to invent data.

---

## Historical Backfill Completeness Matrix

This matrix is part of the authoritative retrospective record and formally accounts for every Standard Entry Schema category across all retrospective entries predating T5-3C:

| Entry ID & Phase | Authority & Lineage | Scope & Population | Changes & Validation | Results & Telemetry | Invalid Runs | Decision & Future Step |
|---|---|---|---|---|---|---|
| **1. M49.1 Takeover Baseline** | Takeover Commit: `10681c8...`<br>Parent: `7466388d...`<br>Branch: `NOT RECOVERED FROM CURRENT AUTHORITY` (Ancestor of `takeover/m491-graphless-narrow20`)<br>Artifacts/Scorer: Verified | Official Public-1000 benchmark (1,000 Qs) | Retained M48-M49.1 pipeline (`10681c8...`); validation commands `NOT RECOVERED FROM CURRENT AUTHORITY` | ROUGE-L: `0.473654`<br>METEOR: `0.382772`<br>Submission ZIP: `fe226ea...`<br>996 verified / 4 insufficient | Retrospective runs prior to takeover: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `ACCEPTED TAKEOVER BASELINE`<br>Rationale: D096<br>Risks: M45 DB dependency<br>Next: T4 Graph Retirement |
| **2. T4 Graph Retirement** | Lineage: `10681c8` $\to$ `b8458ac` $\to$ `aeede78` $\to$ `f61603b` $\to$ `1773ad4` $\to$ `87e71eb`<br>Tag: `t4-graph-retirement-closed`<br>Branch: `takeover/m491-graphless-narrow20` | 7 Dev-200 relationship queries (`89271`, `54485`, `91585`, `60811`, `47573`, `98963`, `75965`) | Retired online graph execution; modified retrieval/agent/serving modules; verified clean Kaggle startup without `graph/` | 7/7 identical answers<br>7/7 identical warnings<br>ROUGE/METEOR delta: `0.000000000`<br>Archives: `1abdea9...`, `1e5f1c2...` | Historical failures: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `ACCEPTED` (D097-D101)<br>Rationale: 0 regression, simpler serving<br>Risks: Offline graph compatibility only<br>Next: T5 Baseline Decomposition |
| **3. T5-1A Diagnostic Harness** | Commit: `fa3b902...`<br>Parent: `87e71eb...`<br>Branch: `t5/baseline-error-decomposition`<br>Files: `scripts/t5_diagnostic_runner.py`, `tests/.../test_t5_diagnostic_harness.py` | Tooling Development (`NOT APPLICABLE` to QA population) | Added diagnostic runner; 0 production files modified; full tests (457 passed, 1 skipped) | `NOT APPLICABLE` (Tooling harness only) | Implementation failures: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `ACCEPTED AS MEASUREMENT TOOLING`<br>Rationale: Deterministic observability<br>Risks: Observer lifecycle debt<br>Next: FAST30 GPU replay |
| **4. T5 FAST30 Invalid OOM Run** | Head: `fa3b902...`<br>Parent: `87e71eb...`<br>Branch: `t5/baseline-error-decomposition` | FAST30 Dev-200 population | Execution aborted due to GPU OOM | 29/30 backend error, 30/30 insufficient evidence; GPU0 memory leak forensic | Entire run: `DISCARDED` | Decision: `DISCARDED`<br>Rationale: Environmental GPU leak<br>Lesson: Single clean runtime per session<br>Next: Clean FAST30 replay |
| **5. T5-1C FAST30 Clean Baseline** | Head: `fa3b902...`<br>Ordered-ID Hash: `11fcb46...`<br>Branch: `t5/baseline-error-decomposition` | First 30 Dev-200 questions (FAST30) | Observation only (0 code modifications) | ROUGE-L: `0.449561`<br>METEOR: `0.360600`<br>28/30 model fallback<br>56 draft rejections<br>Reranker overlap: DOWN 9, SAME 21, UP 0 | Historical run failures: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `ACCEPTED AS DIAGNOSTIC EVIDENCE`<br>Rationale: Baseline telemetry<br>Risks: Generator fallback $\neq$ quality failure<br>Next: T5-2A offline exploration |
| **6. T5-2A Evidence Policy Investigation** | Baseline: `fa3b902...`<br>Branch: `t5/baseline-error-decomposition` | FAST30 Tune (first 20) / Holdout (last 10) | Simulation of heuristic weights and reranker bypass (0 code modifications) | Exact simulation values: `NOT RECOVERED FROM CURRENT AUTHORITY`; pre-rerank bypass scored worse; oracle non-deployable | Intermediate simulation runs: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `REJECTED` naive heuristic tuning and global bypass<br>Rationale: Unstable generalization<br>Next: T5-3A strict own-document recovery |
| **7. T5-3A Strict Own-Document Recovery** | Candidate: `b1ffa5d...`<br>Parent: `fa3b902...`<br>Branch: `t5/baseline-error-decomposition` | Explicit-document queries | Changed `evidence_selector.py` and `test_evidence_selector.py`; 20 unit tests passed; 474 full suite passed | Unit test metrics verified; live inference deferred to targeted A/B | Implementation failures: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `ACCEPTED CANDIDATE`<br>Rationale: Solves document false negatives<br>Risks: Scoped to explicit queries<br>Next: Dev-200 census & causal A/B |
| **8. T5-3B Explicit Document Census** | Baseline: `b1ffa5d...`<br>Branch: `t5/baseline-error-decomposition` | Complete frozen Dev-200 (200 Qs) | Census scanning only (0 code modifications) | Found exactly 2/200 explicit queries (1.00%): Q54485 (`17/2023/QĐ-TTg`) and Q158985 (`42/2022/TT-BTC`) | `NOT APPLICABLE` | Decision: `COMPLETE CENSUS`<br>Rationale: Defines 100% evaluation scope<br>Next: Q54485 and Q158985 A/B |
| **9. T5-3A Q54485 Causal A/B** | Baseline: `fa3b902...`<br>Candidate: `b1ffa5d...`<br>Target: Q54485 (Doc 301729) | Single query (QID 54485) | Evaluated parent vs candidate on Kaggle GPU | Baseline: ROUGE 0.068966 / METEOR 0.005058<br>Candidate: ROUGE 0.363955 / METEOR 0.282805<br>Delta: ROUGE +0.295 / METEOR +0.278<br>Retries: $2 \to 0$, verified | Failures: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `PASS`<br>Rationale: Proven causal fix; identical retrieval hashes<br>Next: Q158985 evaluation |
| **10. T5-3A Invalid Q158985 Run** | Candidate HEAD `b1ffa5d...` accidentally run as baseline | Single query (QID 158985) | Control run executed with candidate source | Evaluated candidate twice; detected via loaded class symbol inspection | Entire run: `DISCARDED` | Decision: `DISCARDED`<br>Rationale: Wrong source authority executed<br>Lesson: Verify HEAD + loaded symbols<br>Next: Clean parent A/B |
| **11. T5-3A Q158985 Exact Causal A/B** | Parent: `fa3b902...`<br>Candidate: `b1ffa5d...`<br>Target: Q158985 | Single query (QID 158985) | Evaluated verified parent vs candidate on Kaggle GPU | Parent: ROUGE 0.070485 / METEOR 0.011678<br>Candidate: ROUGE 0.190604 / METEOR 0.087424<br>Delta: ROUGE +0.120 / METEOR +0.076<br>Retries: $2 \to 1$, BM25 recovery | Failures: `NOT RECOVERED FROM CURRENT AUTHORITY` | Decision: `PASS`<br>Rationale: Proven causal fix on 2nd query; identical retrieval hashes<br>Next: Final decision summary |
| **12. T5-3A Final Decision Summary** | Candidate: `b1ffa5d...`<br>Rollback: `fa3b902...` | 2/2 explicit queries in Dev-200 (100% coverage) | No additional code changes in the final-decision step; candidate `b1ffa5d` retains its targeted `evidence_selector.py` production change. | Q54485 (+0.295 ROUGE, +0.278 METEOR)<br>Q158985 (+0.120 ROUGE, +0.076 METEOR)<br>Zero retrieval regressions | `NOT APPLICABLE` | Decision: `ACCEPTED TARGETED CANDIDATE`<br>Rationale: 2/2 validated; no Public-1000 claim yet<br>Next: T5-3C Governance & Ledger |

---
## 1. M49.1 — Takeover Baseline Authority

- **Status:** ACCEPTED TAKEOVER BASELINE
- **Takeover Commit Date:** 2026-08-21 (from git commit `10681c8c05008432cd1c7170cd3917f4317c1c69`)
- **Historical Public-1000 Benchmark Execution Date:** NOT RECOVERED FROM CURRENT AUTHORITY (retained benchmark output documented in `docs/18-M491-PUBLIC-RESULT.md`)
- **Objective:** Establish the frozen, reproducible starting point for all post-takeover engineering.
- **Hypothesis:** M49.1 provides the highest validated baseline quality on official competition data.

### Authority
- **Takeover Source Commit:** `10681c8c05008432cd1c7170cd3917f4317c1c69` ("Retain M48-M49.1 pipeline and add project handoff")
- **Parent Commit:** `7466388d01f5af246ed25670482c7a079c86084b`
- **Historical Branch at Commit Creation:** NOT RECOVERED FROM CURRENT AUTHORITY (Verifiable ancestor of the later `takeover/m491-graphless-narrow20` lineage)
- **Baseline Lineage Note:** Commit `10681c8` represents the authoritative repository source state that contained the retained M49.1 baseline before T4 production modifications. The later graphless branch state (`takeover/m491-graphless-narrow20`) was created subsequently during T4 and was not the original source of the Public-1000 benchmark submission.
- **M45 Corpus Archive SHA-256:** `7e78ad60ff2982592a9471eb8704fce44042add0496268fade3f32db1823ea7a`
- **Official train.json SHA-256:** `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988`
- **Official Scorer (NLTK METEOR / ASCII ROUGE-L) SHA-256:** `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`
- **M49 Generator Merged Tree SHA-256:** `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b`
- **Frozen Dev-200 Ordered-ID SHA-256:** `694825b5961a90a284ad0364ac4f31a1a85f446519c92274a784c8e2be9a48ad`
- **Public-1000 Question SHA-256:** `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5`
- **Evaluation Population:** Official Public-1000 dataset (1,000 questions)

### Intended Change vs Actual Change
- **Intended Change:** Retain verified M48-M49.1 pipeline, clean up obsolete M46/M47 executables, and establish handoff documentation.
- **Actual Change:** Documented in commit `10681c8c05008432cd1c7170cd3917f4317c1c69`.

### Validation & Numeric Results
From `docs/18-M491-PUBLIC-RESULT.md`:

| Metric | M48 Control | M49.1 Takeover Baseline | Delta |
|---|---:|---:|---:|
| **ROUGE-L** | 0.363140133 | **0.473653736** | +0.110513603 (+30.4%) |
| **METEOR** | 0.268587670 | **0.382772249** | +0.114184579 (+42.5%) |

- **Validated Submission ZIP SHA-256:** `fe226ea3d56d2d11623910ab3f52f05463c0fd88f8e13ca66568fbc877a911d0`
- **Causal Telemetry:** 996/1000 `answer_verified`, 4/1000 `insufficient_evidence=True`.

### Invalid / Discarded Runs
- Retrospective exploratory runs predating M49.1 takeover: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** ACCEPTED TAKEOVER BASELINE
- **Rationale:** Established in `docs/08-DESIGN-DECISIONS.md` D096. Provides the highest verified public benchmark score under organizer-compliant constraints.

### Production Impact & Known Risks
- **Production Impact:** Retains M49.1 serving pipeline as the active baseline.
- **Known Risks:** Dependency on external M45 offline indexes and GPU memory requirements during generation.

### Next Step
- Phase T4 (Graph retirement and serving simplification).

---

## 2. T4 — Online Graph Execution Retirement

- **Status:** ACCEPTED & CLOSED
- **Date:** 2026-08-21 to 2026-08-22 (Commit timestamps: `2026-08-21 23:12:56` to `2026-08-22 00:31:30`)
- **Objective:** Eliminate complex online graph traversals and memory overhead while preserving exact serving answer quality.
- **Hypothesis:** Graph retrieval adds runtime complexity without contributing differentiated evidence over narrow HYBRID_RERANK.

### Authority & Lineage
- **Pre-T4 Takeover Source:** `10681c8c05008432cd1c7170cd3917f4317c1c69`
- **T4-PROD-A Candidate Commit:** `b8458acfe1fa3ef318c9052ec5c5364acda36630` ("Replace relationship graph routing with narrow reranking")
- **T4-PROD-B1 Commit:** `aeede789184e1bf588bbe16835e5f2c76d889a4c` ("Decouple graph runtime from M49.1 serving")
- **T4-PROD-B2 Commit:** `f61603b0f82d669ca6364fa1d50036663a0c3934` ("Make graph artifacts optional for graphless runtime")
- **T4-PROD-B3 Final Implementation Source:** `1773ad4c0ab95147f18d13036e69ba9b46341cbf` ("Retire online graph execution")
- **T4 Close Checkpoint Commit:** `87e71eb7661eb9cda1e63f4f0af16ef4613dadfb` ("Close T4 graph retirement checkpoint")
- **Tag:** `t4-graph-retirement-closed`
- **Branch:** `takeover/m491-graphless-narrow20`
- **Artifact Identities:** M45 offline indexes with optional graph artifacts.
- **Model / Generator Identities:** Pretrained embedding/reranker + fine-tuned M49 generator.
- **Scorer Identity:** Official scorer (`4fac9142...`)
- **Evaluation Population:** Exactly 7 `QueryIntent.RELATIONSHIP` questions in frozen Dev-200 (7 / 200, 3.5%): `89271`, `54485`, `91585`, `60811`, `47573`, `98963`, `75965`.

### Intended Change vs Actual Change
- **Intended Change:** Route relationship queries to `HYBRID_RERANK` narrow20; decouple graph retriever/tool; make graph artifacts optional; fully retire online graph execution.
- **Actual Change:** Progressively implemented across commits `b8458ac`, `aeede78`, `f61603b`, `1773ad4`, and closed at `87e71eb`.

### Validation & Numeric Results
- **Answer Identity:** 7/7 (100%) exact identical strings between GRAPH baseline and graphless narrow20 candidate.
- **Warning Profiles:** 7/7 identical warning profiles.
- **ROUGE-L Delta:** `0.000000000` (0 regressions).
- **METEOR Delta:** `0.000000000` (0 regressions).
- **Live Online State:** Reduced to 3 runtime manifests (`legal_chunks`, `bm25_index`, `vector_index`), 4 retrieval strategies (`BM25`, `DENSE`, `HYBRID`, `HYBRID_RERANK`), and 7 runtime tools.
- **Evidence Archives:** `t4-graphless-narrow20-evidence.zip` (`1abdea95...`), `t4-final-graph-retirement-evidence.zip` (`1e5f1c2c...`).

### Invalid / Discarded Runs
- Historical exploratory runs: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** ACCEPTED (Decisions D097, D098, D099, D100, D101).
- **Rationale:** Simplifies production architecture, eliminates online graph dependencies, and preserves exact behavioral equivalence.

### Production Impact & Known Risks
- **Production Impact:** Online graph execution retired; offline graph index builders and historical schema compatibility retained.
- **Known Risks:** Offline graph artifacts retained for historical compatibility but not validated at runtime startup.

### Next Step
- Phase T5 (Baseline error decomposition).

---

## 3. T5-1A — Baseline Diagnostic Harness

- **Status:** ACCEPTED AS MEASUREMENT TOOLING
- **Commit Date:** 2026-08-22 09:31:18 +0700
- **Objective:** Build a transparent, deterministic telemetry and diagnostic runner to observe pre/post rerank candidates, branch retrieval hits, evidence selection, context grading, generator draft rejections, and per-question official metrics.
- **Hypothesis:** Comprehensive intermediate observability guides targeted optimizations without modifying production code.

### Authority
- **Committed SHA:** `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
- **Parent Baseline:** `87e71eb7661eb9cda1e63f4f0af16ef4613dadfb` (`t4-graph-retirement-closed`)
- **Branch:** `t5/baseline-error-decomposition`
- **Committed Files:** `scripts/t5_diagnostic_runner.py`, `tests/unit/evaluation/test_t5_diagnostic_harness.py` (0 production files modified).
- **Artifact / Model / Scorer Identities:** Standard M49.1 serving stack and official scorer.
- **Evaluation Population:** NOT APPLICABLE (Tooling development).

### Intended Change vs Actual Change
- **Intended Change:** Create standalone diagnostic runner with fail-closed telemetry and durable JSONL persistence.
- **Actual Change:** Created `scripts/t5_diagnostic_runner.py` and unit test suite `tests/unit/evaluation/test_t5_diagnostic_harness.py`.

### Validation & Numeric Results
- Strict 40-hex SHA validation, pre-inference gold answer/scorer validation gates.
- Syntax-vs-schema JSONL recovery with durable `fsync` and atomic writes.
- Report-before-manifest completion marker ordering.
- Focused test suite: 7 passed in 0.44s. Full suite: 457 passed, 1 skipped.

### Invalid / Discarded Runs
- Implementation failure logs: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** ACCEPTED AS MEASUREMENT TOOLING
- **Rationale:** Provides deterministic intermediate stage observability for baseline error decomposition.

### Production Impact & Known Risks
- **Production Impact:** Zero production source modifications.
- **Known Risks / Technical Debt:** Observer lifecycle defect when multiple `T5Dev200DiagnosticRunner` instances are executed sequentially in the same process (`T5-H1`).

### Next Step
- Run FAST30 diagnostic baseline on Kaggle GPU.

---

## 4. T5 FAST30 — Invalid OOM Run

- **Status:** INVALID / DISCARDED
- **Date:** 2026-08-22
- **Objective:** Initial 30-question diagnostic replay on Kaggle GPU.
- **Hypothesis:** Diagnostic runner would execute smoothly on Kaggle GPU.

### Authority
- **Git HEAD:** `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
- **Evaluation Population:** First 30 questions of frozen Dev-200.

### Intended Change vs Actual Change
- **Intended Change:** Execute diagnostic replay.
- **Actual Change:** Execution aborted due to GPU initialization failure.

### Validation & Numeric Results
- **Outcome:** 29/30 questions failed with `generator:backend_initialization_error` and 30/30 defaulted to `insufficient_evidence=True`.
- **Forensic Analysis:** GPU memory inspection revealed GPU0 remained nearly fully allocated from a prior uncollected runtime, triggering `torch.OutOfMemoryError`.

### Invalid / Discarded Runs
- Entire run DISCARDED. Metrics must never be used as baseline evidence.

### Decision & Rationale
- **Decision:** DISCARDED
- **Rationale:** Environmental GPU memory leak from prior uncollected session, not a pipeline algorithm failure.

### Production Impact & Known Risks
- **Lesson Learned:** Every GPU evaluation session must instantiate exactly one clean runtime, cleanly manage memory lifecycles, and assert GPU availability before starting inference.

### Next Step
- Replay clean FAST30 diagnostic baseline.

---

## 5. T5-1C — FAST30 Clean Diagnostic Baseline (Clean1)

- **Status:** VALID DIAGNOSTIC EVIDENCE
- **Date:** 2026-08-22
- **Objective:** Execute clean deterministic baseline diagnostics over the first 30 questions of frozen Dev-200.
- **Hypothesis:** Intermediate telemetry reveals specific failure modes (retrieval vs context grading vs generation).

### Authority
- **Git HEAD:** `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
- **Branch:** `t5/baseline-error-decomposition`
- **Evaluation Population:** First 30 questions of frozen Dev-200 (FAST30).
- **Ordered-ID SHA-256:** `11fcb465f9194cbaeee5d6fe5ed127da38e1bb66969654a4cdf984c25b3c8417`

### Intended Change vs Actual Change
- **Intended Change:** Baseline observation only (0 code modifications).
- **Actual Change:** None.

### Validation & Numeric Results
- **Backend Initialization Failures:** 0 / 30 (100% operational success).
- **ROUGE-L:** `0.44956125747562414`
- **METEOR:** `0.36059965374134695`
- **Generator Telemetry:** 28/30 questions experienced generator model-error fallback; 28/30 had `structured_output_schema` rejections; 56 structured draft rejections total.
- **Reranker Reference Overlap Proxy:** Reranking moved gold-bearing evidence DOWN in 9/30 queries, kept it SAME in 21/30 queries, and moved it UP in 0/30 queries.

### Invalid / Discarded Runs
- Historical run failures: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** ACCEPTED AS DIAGNOSTIC EVIDENCE
- **Rationale:** Provides verified, reproducible baseline telemetry for FAST30.
- *Caution:* Generator fallback does not necessarily indicate a quality failure (fallback to top-1 evidence often yields strong groundings).
- *Caution:* Reranker proxy drops do not justify globally disabling reranking.

### Production Impact & Known Risks
- Zero code modifications.

### Next Step
- Offline evidence policy investigation (T5-2A).

---

## 6. T5-2A — Offline Evidence Policy Investigation

- **Status:** INCONCLUSIVE (HYPOTHESES REJECTED)
- **Date:** 2026-08-22
- **Objective:** Evaluate whether simple heuristic tuning of lexical overlap weights or bypassing the reranker improves evidence selection.
- **Hypothesis:** Adjusting lexical weights or prioritizing pre-rerank evidence could improve answer grounding.

### Authority
- **Evaluation Population:** Deterministic FAST30 diagnostic population partitioned into Tune (first 20 questions) and Holdout (last 10 questions).
- **Exact Aggregate Numeric Values:** NOT RECOVERED FROM CURRENT AUTHORITY (Exploratory simulation).

### Intended Change vs Actual Change
- **Intended Change:** Offline simulation of heuristic evidence selection policies.
- **Actual Change:** None (Offline analysis only).

### Validation & Findings
- Direct use of pre-rerank/top-1 evidence as a global replacement produced a worse FAST30 aggregate than the current baseline.
- Naive lexical/rank selection policies did not produce stable gains across the deterministic FAST30 Tune20/Holdout10 split.
- Reference-aware oracle selection showed large theoretical headroom, but oracle policies rely on gold reference information, are non-deployable, and must never be represented as candidate scores.

### Invalid / Discarded Runs
- Intermediate exploratory simulation runs: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** REJECTED naive heuristic tuning and global reranker bypass.
- **Rationale:** Unstable generalization across Tune20/Holdout10; production policy must rely only on deployable, causal signals.

### Production Impact & Known Risks
- Zero code modifications.

### Next Step
- Target explicit proven root causes (T5-3A strict own-document recovery).

---
## 7. T5-3A — Strict Own-Document Reference Recovery

- **Status:** ACCEPTED CANDIDATE
- **Commit Date:** 2026-08-22 14:03:16 +0700
- **Objective:** Eliminate explicit-document false negatives caused by missing `hit.metadata["document_number"]` when the query contains an explicit document reference.
- **Hypothesis:** If a query contains an explicit legal document number (e.g. `17/2023/QĐ-TTg`) and serving metadata lacks `document_number`, strictly recovering the own-document identity from a recognized leading title slug will prevent false context-grader rejections, bounded retries reaching max retry limit, and resulting insufficient-evidence responses.

### Authority
- **Candidate Commit SHA:** `b1ffa5d59cf8d7506176a8a1ecfa35c034fa8543`
- **Parent Baseline SHA:** `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
- **Branch:** `t5/baseline-error-decomposition`
- **Changed Production File:** `src/legal_agentic_rag/generation/evidence_selector.py` (1 file)
- **Changed Test File:** `tests/unit/generation/test_evidence_selector.py` (1 file)

### Safety & Invariant Contract
1. **Metadata Precedence:** Non-empty string `hit.metadata["document_number"]` is authoritative. Mismatches return `False` immediately; title fallback is never evaluated to override metadata.
2. **Malformed Metadata Fails Closed:** Present non-string values (e.g. `int`, `list`) fail closed (`False`) and never invoke title fallback.
3. **Recognized Slug Prefix Required:** The normalized title must begin strictly with one of 22 recognized legal-document prefixes (e.g. `quyet-dinh`, `thong-tu`, `nghi-dinh`, `luat`, `van-ban-hop-nhat`, etc.).
4. **Immediate Adjacency:** The own-document number must appear immediately after the recognized prefix (with an optional `"so"` token). Any descriptive prose between prefix and number (e.g. `Huong-dan-thuc-hien-...`, `Quyet-dinh-ve-viec-...`) returns `False`.
5. **No Later Number Matching:** Numbers appearing later in the title (e.g. amended/cited laws such as `31-2007-QD-TTg`) return `False`.
6. **No Body Scanning:** Chunk body text is never scanned for this fallback.
7. **No Side Effects:** Retrieval ranking, reranking weights, generator models, and context token budgets remain completely unchanged.

### Validation & Numeric Results
- **Focused Test Suite:** 20 passed in 0.39s.
- **Full Test Suite:** 474 passed, 1 skipped.
- **Compileall / Diff-Check:** Clean (0 errors).

### Invalid / Discarded Runs
- Implementation failure logs: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** ACCEPTED CANDIDATE
- **Rationale:** Restores correct document-identity matching on explicit-document queries while strictly preserving negative safety on adversarial non-matching documents.

### Production Impact & Known Risks
- **Production Impact:** Activates title slug fallback only when `document_number` metadata is absent and query has explicit legal references.
- **Known Risks:** Narrowly scoped to explicit-document queries; does not alter general text retrieval ranking.

### Next Step
- Frozen Dev-200 census and causal A/B validation.

---

## 8. T5-3B — Frozen Dev-200 Explicit Document Census

- **Status:** COMPLETE CENSUS
- **Date:** 2026-08-22
- **Objective:** Identify the complete population of explicit-document queries in frozen Dev-200.
- **Hypothesis:** Explicit-document queries form a small, well-defined subset of frozen Dev-200.

### Authority
- **Evaluation Population:** Complete frozen Dev-200 (200 questions).
- **Result:** Exactly 2 out of 200 questions (1.00%) contain explicit document numbers:
  1. **QID 54485:** `17/2023/QĐ-TTg`
  2. **QID 158985:** `42/2022/TT-BTC`
- **Implication:** Causal A/B validation on QID 54485 and QID 158985 covers 100% of the explicit-document population in frozen Dev-200.

### Intended Change vs Actual Change
- Population census only (0 code modifications).

### Decision & Rationale
- **Decision:** COMPLETE CENSUS
- **Rationale:** Identifies exact evaluation scope for T5-3A validation.

### Next Step
- Targeted causal A/B on Q54485 and Q158985.

---

## 9. T5-3A — Q54485 Causal A/B Validation

- **Status:** PASS (CAUSAL VALIDATION CONFIRMED)
- **Date:** 2026-08-22
- **Target Question:** QID 54485 (`Quyết định 17/2023/QĐ-TTg`)
- **Document ID:** 301729 (`Quyet-dinh-17-2023-QD-TTg-sua-doi-Quyet-dinh-31-2007-QD-TTg-...`)

### Causal Evidence & Metric Results
- **Pre-Rerank Retrieval Identity Hash:** Identical between baseline and candidate.
- **Post-Rerank Retrieval Identity Hash:** Identical between baseline and candidate.

| Metric / Behavior | Parent Baseline (`fa3b902`) | Candidate (`b1ffa5d`) | Delta |
|---|---:|---:|---:|
| **ROUGE-L** | 0.068965517 | **0.363954506** | **+0.294988988** |
| **METEOR** | 0.005058169 | **0.282804763** | **+0.277746594** |
| **Stop Reason** | `max_retry_reached` | **`answer_verified`** | Fixed |
| **Retry Count** | 2 | **0** | -2 retries |
| **Insufficient Evidence** | `True` | **`False`** | Fixed |
| **Document Match** | `False` | **`True`** | Fixed |
| **Context Coverage** | `False` | **`True`** | Fixed |

### Invalid / Discarded Runs
- Intermediate run logs: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** PASS
- **Rationale:** Substantial metric gains with 0 retrieval regressions; identical intermediate retrieval hashes prove causal link to document identity recovery.

### Next Step
- Evaluate Q158985.

---

## 10. T5-3A — Invalid Q158985 Run

- **Status:** INVALID / DISCARDED
- **Date:** 2026-08-22
- **Outcome:** A baseline control run was started but executed against candidate commit `b1ffa5d` instead of parent `fa3b902`.
- **Forensic Gate:** Inspection of loaded module symbols revealed `_match_own_document_title` was present.
- **Lesson Learned:** Verification gates must assert both `git rev-parse HEAD` and inspect loaded class attributes before starting A/B measurements.
- **Decision:** DISCARDED

### Next Step
- Re-run Q158985 baseline with verified parent source.

---

## 11. T5-3A — Q158985 Exact Causal A/B Validation

- **Status:** PASS (CAUSAL VALIDATION CONFIRMED)
- **Date:** 2026-08-22
- **Target Question:** QID 158985 (`Thông tư 42/2022/TT-BTC`)
- **Verified Parent Source:** `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
- **Verified Candidate Source:** `b1ffa5d59cf8d7506176a8a1ecfa35c034fa8543`

### Causal Evidence & Metric Results
- **Attempt 1 (`HYBRID_RERANK`) Hash:** `969cc87237ff09d235cd3c4e665ee88d7fcfc1200677a490744939ba6fe1ba83` (Identical).
- **Attempt 2 (`BM25`) Hash:** `889441fb6add1be10c7fd2f8d8c393b802f79082ee8429bcea2640c740e35184` (Identical).

| Metric / Behavior | Parent Baseline (`fa3b902`) | Candidate (`b1ffa5d`) | Delta |
|---|---:|---:|---:|
| **ROUGE-L** | 0.070484581 | **0.190604027** | **+0.120119445** |
| **METEOR** | 0.011678450 | **0.087424060** | **+0.075745610** |
| **Stop Reason** | `max_retry_reached` | **`answer_verified`** | Fixed |
| **Retry Count** | 2 | **1** | -1 retry |
| **Selected Strategy** | `hybrid` | **`bm25`** | Clean recovery |
| **Insufficient Evidence** | `True` | **`False`** | Fixed |
| **BM25 Document Coverage** | `False` | **`True`** | Fixed |
| **BM25 Context Sufficiency** | `False` | **`True`** | Fixed |

### Invalid / Discarded Runs
- Intermediate run logs: NOT RECOVERED FROM CURRENT AUTHORITY.

### Decision & Rationale
- **Decision:** PASS
- **Rationale:** Demonstrates causal recovery on second explicit-document question; completes validation across 100% of explicit-document population in Dev-200.

### Next Step
- Final decision summary for T5-3A.

---

## 12. T5-3A Final Decision Summary

- **Status:** **ACCEPTED TARGETED CANDIDATE**
- **Candidate Commit:** `b1ffa5d59cf8d7506176a8a1ecfa35c034fa8543`
- **Rollback Authority:** `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
- **Decision Rationale:**
  1. Complete causal verification on 2/2 explicit-document queries in frozen Dev-200 (Q54485 and Q158985).
  2. Large quality improvements with 0 retrieval regressions:
     - Q54485: ROUGE +0.295, METEOR +0.278.
     - Q158985: ROUGE +0.120, METEOR +0.076.
  3. Identical intermediate retrieval hashes prove improvement stems strictly from fixing document-identity false negatives at evidence selection and context grading.
  4. Unit test suite expanded to 20 tests covering all adversarial prefix/prose/boundary/malformed cases; full test suite passes 474/475 (1 skipped).
- **Scope & Limitations:**
  - *No Leaderboard-Wide Claim Yet:* T5-3A is accepted as a targeted candidate. Full Dev-200 and Public-1000 benchmark numbers will be measured after completing baseline decomposition milestones.
- **Next Planned Step:** `T5-3C` (Acceptance documentation and permanent ledger governance).

---

## 13. T5-3C — Accept T5-3A and Establish Permanent Experiment Ledger

- **Status:** ACCEPTED & CLOSED
- **Date:** 2026-08-22
- **Objective:** Formally record T5-3A Strict Own-Document Reference Recovery as an accepted targeted candidate and establish a permanent repository-level experiment/change ledger for all post-takeover engineering.
- **Hypothesis:** A durable chronological ledger and strict governance rules in AGENTS.md will prevent historical drift, undocumented experiments, and unsubstantiated metric claims across future agent turns.

### Authority
- **Baseline / Starting Authority:** `b1ffa5d59cf8d7506176a8a1ecfa35c034fa8543`
- **Candidate Commit SHA:** `ca00c133b70905dc0bcff8cb469727264b25a995` (Acceptance / Closure Checkpoint)
- **Candidate Parent Commit:** `b1ffa5d59cf8d7506176a8a1ecfa35c034fa8543`
- **Branch:** `t5/baseline-error-decomposition`
- **Artifact Identities:** NOT APPLICABLE (Documentation/Governance only)
- **Model / Generator Identities:** NOT APPLICABLE
- **Scorer Identity:** NOT APPLICABLE
- **Evaluation Population:** NOT APPLICABLE (0 model inference runs)

### Intended Change vs Actual Change
- **Intended Production Change:** None (Documentation and governance only).
- **Actual Production Change:** None (`src/`, `tests/`, `configs/`, `scripts/`, `notebooks/` completely unmodified).
- **Exact Repository Files Modified:**
  - `AGENTS.md` (Mandatory experiment ledger governance rule)
  - `docs/00-START-HERE.md` (Pointer to experiment ledger)
  - `docs/08-DESIGN-DECISIONS.md` (Decision D102)
  - `docs/09-IMPLEMENTATION-PLAN.md` (Milestones 55 and 56)
  - `docs/19-EXPERIMENT-LEDGER.md` (New ledger creation, backfill, and completeness matrix)

### Validation & Numeric Results
- **Validation Commands:** `git diff --check`, `git diff --name-status`, `python -m pytest -q` (474 passed, 1 skipped), automated string verification across changed documents.
- **Numeric Metric Results:** NOT APPLICABLE (No inference executed).
- **Causal Evidence:** NOT APPLICABLE.

### Invalid / Discarded Runs
- None in this phase.

### Decision & Rationale
- **Decision:** ACCEPTED
- **Rationale:** External review verified all historical authority corrections, retrospective backfill completeness governance, zero production/code modifications, and clean test/diff validation gates.

### Production Impact & Known Risks
- **Production Impact:** Zero serving or runtime behavior changes.
- **Known Risks:** Potential historical documentation drift if retrospective authorities are reconstructed without strict provenance. Mitigated by explicit Historical Backfill Policy and Completeness Matrix.
- **Rollback Authority:** Discard uncommitted documentation changes and return to `b1ffa5d59cf8d7506176a8a1ecfa35c034fa8543`.

### Next Step
- Commit and push the T5-3C documentation/governance checkpoint on `t5/baseline-error-decomposition`, then begin T5-H1 only from that committed checkpoint.

---

## 14. T5-H1 — Repair Diagnostic Observer Lifecycle

- **Status:** ACCEPTED & CLOSED
- **Date:** 2026-08-22
- **Objective:** Repair observer installation/restoration ownership so sequential diagnostic runners on one shared runtime cannot leak telemetry contexts.
- **Hypothesis:** Explicit instrumentation lease ownership with object-identity restoration ensures sequential diagnostic runs on shared runtimes remain isolated without leaking observer wrappers or handlers.

### Authority
- **Baseline / Starting Authority:** `ca00c133b70905dc0bcff8cb469727264b25a995`
- **Implementation / Acceptance Commit SHA:** `4c4b40472c14ea75fa95ea3a129f1077ac40fc33`
- **Branch:** `t5/baseline-error-decomposition`
- **Parent Commit:** `ca00c133b70905dc0bcff8cb469727264b25a995`
- **Artifact Identities:** NOT APPLICABLE (Measurement tooling only)
- **Model / Generator Identities:** NOT APPLICABLE
- **Scorer Identity:** NOT APPLICABLE
- **Evaluation Population:** NOT APPLICABLE (Tooling regression suite)

### Intended Change vs Actual Change
- **Intended Production Change:** None (Measurement harness tooling only).
- **Actual Production Change:** None (`src/` completely unmodified).
- **Exact Files Modified:**
  - `scripts/t5_diagnostic_runner.py` (Implemented `DiagnosticInstrumentationLease` and reversible lifecycle management)
  - `tests/unit/evaluation/test_t5_diagnostic_harness.py` (Added 8 mandatory regression tests)
  - `docs/19-EXPERIMENT-LEDGER.md` (Backfilled T5-3C checkpoint SHA and added T5-H1 entry)
  - `docs/09-IMPLEMENTATION-PLAN.md` (Updated Milestone 56 status)

### Validation & Numeric Results
- **Validation Commands:**
  - `python -m pytest tests/unit/evaluation/test_t5_diagnostic_harness.py -q` (15 passed)
  - `python -m pytest -q` (482 passed, 1 skipped)
  - `python -m compileall scripts/t5_diagnostic_runner.py`
  - `git diff --check`
- **Mandatory Lifecycle Invariants Verified:**
  1. *Exact Object Identity Restoration:* `hybrid_rerank._reranker`, `hybrid._bm25`, and `hybrid._dense` are restored to the exact original instances (`is`) upon `close()`.
  2. *Sequential Runner Telemetry Isolation:* Runner B on the same runtime records telemetry only to its own context; runner A's context retains its exact rerank, BM25 branch, and dense branch events completely isolated and untouched after runner A closes.
  3. *No Nested Wrappers:* Observer nesting is strictly prevented across sequential runner lifecycles.
  4. *Logger Lifecycle:* Logging handlers are cleanly attached on init and detached on `close()`.
  5. *Idempotent Close:* Calling `close()` multiple times is safe and maintains original object identities.
  6. *Context Manager Exception Safety:* Exceptions raised inside `with` blocks trigger full cleanup in `__exit__`.
  7. *Overlapping Runners Fail Closed With Zero Mutation:* Attempting concurrent runners on the same runtime raises `BackendInitializationError` while leaving exact runtime observer identities (`is`), lease state, and logger handlers completely unchanged.
  8. *Minimal Mock Compatibility:* Services without full `FixedRetriever` internals remain fully compatible.

### Invalid / Discarded Runs
- NOT APPLICABLE (Deterministic unit test suite).

### Decision & Rationale
- **Decision:** ACCEPTED
- **Rationale:** External review verified exact runtime-object restoration, sequential rerank/BM25/dense telemetry isolation, zero-mutation overlapping-runner rejection, logger cleanup, idempotent and exception-safe lifecycle behavior, and zero production `src/**` changes.

### Production Impact & Known Risks
- **Production Impact:** Zero serving or runtime behavior changes.
- **Known Limitations:** Single-session concurrency on a single `ServingService` runtime is unsupported by design (existing runner must be closed before instantiating a new runner).
- **Rollback Authority:** `ca00c133b70905dc0bcff8cb469727264b25a995`.

### Next Step
- T5-H1 implementation is accepted and closed at `4c4b40472c14ea75fa95ea3a129f1077ac40fc33`. After this documentation-closure checkpoint is committed and remote-verified, begin T5-4 from that new checkpoint.


---

## 15. T5-4A — Evidence Selection Opportunity Census and Counterfactual Top-1 Policy Analysis

- **Status:** ACCEPTED & CLOSED
- **Date:** 2026-08-22
- **Objective:** Determine whether meaningful quality improvement remains available specifically at the evidence selection / top-1 stage using only deployable serving-time signals.
- **Hypothesis:** Systematic census of conservative causal failure layers, actual selected Top-1 evidence reconciliation with unique-match validation, and pre/post-rerank telemetry analysis tracking candidate removal will establish whether deployable signals can reliably promote oracle-better candidates without heuristic regressions.

### Authority & Artifact Scope
- **Starting / Rollback Authority:** 985bf4bb03a588880413c57fd09deb74b87a8294
- **Analysis / Acceptance Commit SHA:** 2b17c8ecbaeb818f30efd014f59febfcd28f00c7
- **Branch:** t5/baseline-error-decomposition
- **Parent Commit:** 985bf4bb03a588880413c57fd09deb74b87a8294
- **Telemetry Scope:** HISTORICAL T4 FAST30 BASELINE TELEMETRY (t5-1c-fast30-clean1-evidence.zip, SHA-256: be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f).
- **Execution Identity:**
  - production_baseline_source_sha: 87e71eb7661eb9cda1e63f4f0af16ef4613dadfb (Pre-T5-3A baseline)
  - `measurement_harness_source_sha`: `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
  - application_config_hash: ca1f0aa45a22df7aa9293e42df94473c059b4480b8c03d6ef942c21e9f3da261
  - official_scorer_sha256: 4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891
  - ordered_question_ids_sha256: 11fcb465f9194cbaeee5d6fe5ed127da38e1bb66969654a4cdf984c25b3c8417
- **Evaluation Population:** FAST30 (Tune20: first 20 QIDs, Holdout10: last 10 QIDs).

### Intended Change vs Actual Change
- **Intended Production Change:** None (Diagnostic analysis and tooling only).
- **Actual Production Change:** None (src/** completely unmodified; EvidenceSelector remains unchanged).
- **Exact Files Created / Modified:**
  - scripts/t5_evidence_policy_analysis.py (Created census, unique-reconciliation fail-closed validation, candidate-removal rerank telemetry analysis, and Tune-only policy discovery tooling)
  - tests/unit/evaluation/test_t5_evidence_policy_analysis.py (Created 19 unit tests covering all causal failure layers, reconciliation fail-closed rules, and partition gates)
  - docs/19-EXPERIMENT-LEDGER.md (Recorded accepted and closed T5-4A census entry)
  - docs/09-IMPLEMENTATION-PLAN.md (Updated Milestone 56 status to accepted)

### Methodology & Telemetry Auditing
- **Top-1 Resolution & Unique Reconciliation:** ACTUAL selected_evidence[0] is extracted and uniquely reconciled with terminal_retrieval_hits via chunk_id and source_rank (with fail-closed validation: un-reconciled or duplicate records become analysis_valid = False and AMBIGUOUS).
  - Unique Reconciliation Valid: 30 / 30 (100.0%).
  - Reconciliation Invalid: 0 / 30 (0.0%).
  - selected_evidence[0] == terminal_retrieval_hits[0]: 30 / 30 (100.0%) in historical baseline.
- **Rerank Telemetry Analysis (Tracking Candidate Removal):**
  - Uniquely Mapped Rerank Events: 29 / 30 (all 20 Tune20, 9 Holdout10 where post_rerank_hits == terminal_retrieval_hits).
  - Ambiguous Rerank Events: 1 / 30 (Q54485, where final execution path used strategy: bm25 directly without terminal reranking).
  - Confirmed RERANK_LOSS: 2 / 30 (Q134499, Q60281 in Holdout10, where an oracle-useful candidate with F1 >= 0.40 was dropped post-rerank with F1 gap >= 0.08).
  - Tune20 Confirmed RERANK_LOSS: 0 / 20 (0.0%).
  - Causal Rule: Requires unique pre/post event mapping, best pre candidate F1 >= 0.40, exact best-pre chunk identity absent from post-rerank hits, and best-pre F1 minus best-post F1 >= 0.08.
- **Context & Diversity Budget Warning Analysis:**
  - Records with Budget/Diversity Warnings: 13 / 30 (89271, 102061, 94975, 56533, 89881, 140337, 36411, 46497, 150817, 21011, 134499, 122809, 138771).
  - Confirmed CONTEXT_BUDGET_LOSS: 0 / 30 (persisted telemetry logs aggregate omission count but does not identify individual candidate IDs).
  - CONTEXT_BUDGET_CAUSALITY_NOT_ASSESSABLE_FROM_ARTIFACT: 13 / 30.

### Conservative Causal Failure-Layer Census Across FAST30 (n=30)
- **NO_CLEAR_CAUSAL_OPPORTUNITY:** 6 / 30 (20.0%)
  - *Tune20 (5):* 39207, 31523, 17179, 89881, 58651
  - *Holdout10 (1):* 66625
- **GENERATION_OR_DOWNSTREAM_MISS:** 7 / 30 (23.3%)
  - *Tune20 (6):* 116553, 83501, 56533, 140337, 36411, 150817
  - *Holdout10 (1):* 52453
- **SELECTION_OPPORTUNITY:** 12 / 30 (40.0%)
  - *Tune20 (8):* 89271, 113579, 102061, 94975, 46497, 150207, 21011, 84363
  - *Holdout10 (4):* 129571, 122809, 138771, 16851
- **CONFIRMED_SELECTION_MISS:** 1 / 30 (3.3%)
  - *Holdout10 (1):* 54485 (Historical baseline document false-negative, imported from prior accepted T5-3A authority; not proven by FAST30 oracle overlap alone).
- **RERANK_LOSS:** 2 / 30 (6.7%)
  - *Holdout10 (2):* 134499, 60281
- **AMBIGUOUS / RETRIEVAL_OR_RERANK_AMBIGUOUS:** 2 / 30 (6.7%)
  - *Tune20 (1):* 102303
  - *Holdout10 (1):* 3359
- **CONTEXT_BUDGET_LOSS:** 0 / 30 (0.0% confirmed)

### Historical Controls Evidence Scope
- **Q54485:** Present in FAST30 (Holdout10). In historical baseline 87e71eb7661eb9cda1e63f4f0af16ef4613dadfb, Q54485 suffered a confirmed selection miss due to document slug mismatch. This failure mode was causally resolved and accepted in candidate T5-3A (b1ffa5d59cf8d7506176a8a1ecfa35c034fa8543).
- **Q158985:** Absent from FAST30 (part of the larger Dev-200 census); its T5-3A recovery stands on prior accepted T5-3A authority.

### Invalid / Discarded Runs
- **T5-4A Exploratory Holdout-Peeking Run:**
  - During initial exploratory parameter analysis, the simplified lexical-weight sweep (weight range 0.0 to 5.0) was evaluated on both Tune20 and Holdout10.
  - This violated single-use holdout discipline.
  - All T5-4A Holdout10 policy-comparison results are DISCARDED as validation evidence.
  - Holdout records are retained strictly for descriptive historical census, not for candidate selection or generalization claims.
  - Candidate policy discovery is restricted strictly to Tune20.
  - **NO GENERALIZATION CLAIM is made from T5-4A.**

### Decision & Rationale
- **Decision:** NO_SELECTION_POLICY_JUSTIFIED
- **Rationale:** No candidate policy survived Tune20 discovery without unacceptable regressions. Holdout10 is contaminated and provides no generalization evidence. No production EvidenceSelector change is justified. The only valid, causally proven selection policy is T5-3A (Strict Own-Document Reference Recovery), which is already accepted and committed. EvidenceSelector remains unmutated; future work should investigate:
  1. targeted reranker behavior
  2. generator contract / fallback efficiency

### Production Impact & Known Risks
- **Production Impact:** Zero serving or runtime behavior changes (EvidenceSelector remains unchanged; src/** unchanged).
- **Known Limitations:** Oracle overlap metrics are diagnostic proxies only and are never exposed as candidate features.
- **Rollback Authority:** 985bf4bb03a588880413c57fd09deb74b87a8294.

### Next Step
- T5-4A is ACCEPTED & CLOSED at 2b17c8ecbaeb818f30efd014f59febfcd28f00c7. Downstream investigation will proceed only from the remote-verified checkpoint.

## 16. T5-5A — Targeted Reranker Causal Investigation

- **Status:** ACCEPTED — EXTERNAL REVIEW PASSED
- **Date:** 2026-08-22
- **Objective:** Investigate historical FAST30 reranker cases flagged by oracle/reference-overlap diagnostics (Q134499, Q60281), determine whether the flags correspond to genuine semantic reranker concerns or proxy false positives, and evaluate whether any clean Tune20 evidence justifies a deployable intervention.
- **Hypothesis:** Reconstructing exact pre/post rerank events, score margins, input representations, and query representations for historical forensic seed cases will establish whether cross-encoder candidate omissions stem from a fixable deployable failure mode or model preference limitations that cannot be addressed without fine-tuning.

### Authority & Artifact Scope
- **Starting / Rollback Authority:** `024e6b5e7481d7dc3e1a4878e158f5f32c0f3080`
- **Analysis / Acceptance Commit SHA:** `e810bfadf0c3a7e80f0d70d43e84e3258c842c63`
- **Branch:** `t5/baseline-error-decomposition`
- **Parent Commit:** `024e6b5e7481d7dc3e1a4878e158f5f32c0f3080`
- **Telemetry Scope:** HISTORICAL T4 FAST30 BASELINE TELEMETRY (`t5-1c-fast30-clean1-evidence.zip`, SHA-256: `be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f`).
- **Execution Identity:**
  - `production_baseline_source_sha`: `87e71eb7661eb9cda1e63f4f0af16ef4613dadfb`
  - `measurement_harness_source_sha`: `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
  - `application_config_hash`: `ca1f0aa45a22df7aa9293e42df94473c059b4480b8c03d6ef942c21e9f3da261`
  - `official_scorer_sha256`: `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`
  - `ordered_question_ids_sha256`: `11fcb465f9194cbaeee5d6fe5ed127da38e1bb66969654a4cdf984c25b3c8417`
- **Evaluation Population:** FAST30 (Tune20: first 20 QIDs, Holdout10: last 10 QIDs).

### Intended Change vs Actual Change
- **Intended Production Change:** None (Forensic investigation only).
- **Actual Production Change:** None (`src/**` completely unmodified; `CrossEncoderReranker` and `RerankingRetriever` remain unchanged).
- **Exact Files Created / Modified:**
  - `scripts/t5_reranker_forensics.py` (Created reranker forensic reconstruction, score space separation, cutoff analysis, and Tune-only policy gate tooling with byte-level authority binding)
  - `tests/unit/evaluation/test_t5_reranker_forensics.py` (Created unit tests covering score separation, cutoff bounds, authority-bound annotations, and partition gates)
  - `docs/19-EXPERIMENT-LEDGER.md` (Added Entry 16 investigation report)
  - `docs/09-IMPLEMENTATION-PLAN.md` (Updated Milestone 56 planned work status)

### Current Reranker Contract vs Historical Execution
- **Current Default Configuration (`RerankerConfig`):**
  - `model_name`: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
  - `model_revision`: `1427fd652930e4ba29e8149678df786c240d8825`
  - `batch_size`: 8
  - `max_length`: 512
  - `max_candidates`: 100
  - `relationship_candidate_k`: None
  - `input_mode`: `legal_context`
  - `prompt_name`: None, `instruction`: None
  - `device`: cpu, `torch_dtype`: float32
- **Historical FAST30 Candidate Pool:** Seed cases evaluated an actual pre-rerank candidate pool of 40 candidates (rather than 100).
- **Query Text:** `query.rewritten_question or query.normalized_question`.
- **Candidate Text:** `build_legal_rerank_text(hit)` prepending unified legal metadata headers and structural hierarchy followed by `Nội dung:` and chunk text.
- **Scoring Semantics & Tie-Breaking:** Raw unbounded cross-encoder logits sorted deterministically by (1) descending score, (2) original candidate `RetrievalHit.rank`, (3) `chunk_id`.

### Score Space Separation & Measurement Limitations
- **Strict Score Separation:** Pre-rerank candidate scores belong to the hybrid RRF retrieval space; post-rerank scores belong to cross-encoder logit space. Tooling strictly enforces this boundary and never computes arithmetic differences between distinct score spaces.
- **Dropped Candidate Logits:** Cross-encoder scores for candidates truncated by top-k were not persisted into historical telemetry. For dropped candidates, exact cross-encoder scores and exact dropped-to-cutoff margins are NOT RECOVERED FROM CURRENT ARTIFACT. The deterministic relationship is recorded as the bounded invariant: dropped score <= cutoff score (due to score/rank/chunk-id tie-breaking), recorded as AT_OR_BELOW_CUTOFF_EXACT_VALUE_NOT_PERSISTED.
- **Authority Binding:** Semantic forensic annotations are bound to the SHA-256 computed directly from actual historical ZIP bytes, plus split/QID/chunk/proxy identity.

### Forensic Case 1 — Q134499
- **Question:** Cấp Phiếu lý lịch tư pháp cho người nước ngoài được pháp luật quy định ra sao?
- **Classification:** SEMANTICALLY_PLAUSIBLE_RERANK_LOSS (Forensic seed candidate; not a validated production claim).
- **Candidate Pool:** 40 pre-rerank candidates.
- **Dropped Candidate:** `chunk_6dbd79b888078e5047434fe0`
  - *Document ID:* `301774` (`Luat-ly-lich-tu-phap-2009-28-2009-QH12-90615`)
  - *Structure:* Điều 45 (Thủ tục yêu cầu cấp Phiếu lý lịch tư pháp số 1)
  - *Source Ranks:* Pre-rank #6, BM25 rank 40, Dense rank 8, RRF score 0.0161
  - *Diagnostic Reference Overlap:* Reference F1 = 0.835
  - *Outcome:* Dropped outside top 10 post-rerank hits. Current-event score not persisted; deterministic rank outcome implies score <= 4.9453, subject to score/rank/chunk-id tie-breaking.
- **Post-Rerank Retained Hits:**
  - *Post #1 (Selected Top-1):* `chunk_7f2df393108cd40cc4646f1d` (doc `301774`, Điều 44: Thẩm quyền cấp Phiếu, score 5.9609, F1 = 0.578)
  - *Post #5 (Best Retained Post):* `chunk_6c08d873b47adaa5f940adbe` (doc `301774`, Điều 46: Thủ tục cấp Phiếu số 2, score 5.0859, F1 = 0.643)
  - *Cutoff (Post #10):* `chunk_41937c0ba97b2165490c50e2` (score 4.9453)
- **Score Margin Summary:** Top-1 to Cutoff Margin = +1.0156 (5.9609 - 4.9453). Exact dropped-to-cutoff margin is unrecovered from artifact.
- **Causal Mechanism:** Điều 44 received a higher retained reranker score and contains an explicit foreigner/jurisdiction phrase, while the dropped Điều 45 chunk is procedure-focused. This ranking pattern is consistent with model preference for the explicit jurisdiction wording. The causal contribution of that wording is not established.
- **Query Representation Evidence:** Persisted query_analysis: intent = general, document_numbers = [], article_numbers = [], clause_numbers = [], point_numbers = []. normalized_question and rewritten_question are not persisted in T5QuestionDiagnosticRecord telemetry (record.get(...) returns None). Telemetry limitation: QUERY_REPRESENTATION_CAUSALITY_NOT_ASSESSABLE_FROM_ARTIFACT.
- **Offline Input Rendering:** build_legal_rerank_text(hit) reconstruction succeeded for Điều 45 (1338 chars, Nội dung index 139) and Điều 44 (1171 chars, Nội dung index 143). Key passages and structural metadata are present early in the rendered text. Limitation: STRUCTURAL_TRUNCATION_RISK_NOT_ASSESSABLE_OFFLINE.

### Forensic Case 2 — Q60281
- **Question:** Mức hưởng bảo hiểm y tế hiện nay của người tham gia bảo hiểm y tế như thế nào?
- **Classification:** ORACLE_PROXY_FALSE_POSITIVE (Forensic assessment; NOT a verified reranker quality loss).
- **Candidate Pool:** 40 pre-rerank candidates.
- **Dropped Candidate with High Oracle F1:** `chunk_c8d1589d4db4ce08c92e67cb`
  - *Document ID:* `155934` (`Quyet-dinh-824-QD-BYT-2023-bo-sung-danh-muc-ma-quan-ly-chi-phi-kham-chua-benh-bao-hiem-y-te-554944`)
  - *Structure:* Phụ lục 2, Điều 6
  - *Source Ranks:* Pre-rank #16, BM25 rank 7, Dense rank None, RRF score 0.0149
  - *Diagnostic Reference Overlap:* Reference F1 = 0.630
  - *Outcome:* Dropped outside top 10 post-rerank hits. Current-event score not persisted; deterministic rank outcome implies score <= 4.90625, subject to score/rank/chunk-id tie-breaking.
- **Post-Rerank Retained Hits:**
  - *Post #1 (Selected Top-1):* `chunk_0a863645e2e3da97e40ac2ce` (doc `211473`, Điều 6, score 6.28125, F1 = 0.226)
  - *Post #2:* `chunk_649e1c065f4d1e22055627eb` (doc `243812`, Điều 22, score 5.8906, F1 = 0.483)
  - *Post #3 (Best Retained Post):* `chunk_175acd29e59d9a4697ffdc0b` (doc `227964`, Điều 12, score 5.6875, F1 = 0.543)
  - *Cutoff (Post #10):* `chunk_a8df7352a7be6e709a341e97` (score 4.90625)
- **Score Margin Summary:** Top-1 to Cutoff Margin = +1.3750 (6.28125 - 4.90625). Exact dropped-to-cutoff margin is unrecovered from artifact.
- **Causal Mechanism:** Forensic review judges the high-F1 billing-table candidate to be an oracle-overlap false positive and the retained general legal sources to be more directly relevant to the substantive query.
- **Query Representation Evidence:** Persisted query_analysis: intent = general, document_numbers = [], article_numbers = [], clause_numbers = [], point_numbers = []. normalized_question and rewritten_question are not persisted in telemetry. QUERY_REPRESENTATION_CAUSALITY_NOT_ASSESSABLE_FROM_ARTIFACT.
- **Offline Input Rendering:** build_legal_rerank_text(hit) reconstruction succeeded for Doc 155934 Điều 6 (1586 chars, Nội dung index 181) and Doc 211473 Điều 6 (354 chars, Nội dung index 173). Key passages are present. Limitation: STRUCTURAL_TRUNCATION_RISK_NOT_ASSESSABLE_OFFLINE.

### FAST30 Descriptive Rerank Census (n=30)
- **Uniquely Mapped Rerank Events:** 29 / 30 (96.7%).
- **Ambiguous Events:** 1 / 30 (`Q54485`, BM25 strategy route).
- **Oracle Best-Pre Candidate Survives Post-Rerank:** 21 / 29 (72.4%).
- **Oracle Best-Pre Candidate Dropped:** 8 / 29 (27.6%).
  - *Tune20 Dropped Cases (5):* `Q113579` (gap +0.050), `Q150817` (gap +0.027), `Q150207` (gap +0.038), `Q84363` (gap +0.078), `Q102303` (gap +0.075). (All 5 have F1 gap < 0.08).
  - *Holdout10 Dropped Cases (3):* `Q134499` (gap +0.192), `Q60281` (gap +0.087), `Q129571` (gap +0.024).
- **Oracle Proxy Drops Across FAST30:** 2 / 29 (6.9%) — `Q134499`, `Q60281`.
- **Semantically Plausible Losses:** 1 / 29 (3.4%) — `Q134499` (in contaminated Holdout10).
- **Oracle Proxy False Positives:** 1 / 29 (3.4%) — `Q60281`.

### Tune20 Policy Discovery Findings
- **Confirmed Tune20 Oracle Proxy Drops:** **0 / 20 (0.0%)**.
- **Semantically Plausible Tune20 Rerank Losses:** **0 / 20 (0.0%)**.
- **Discovery Result:** No clean training signal or causal evidence exists in Tune20 to justify a serving-time reranking heuristic or cutoff override.

### Counterfactual Candidate Rules Considered & Rejected
1. *Rule 1 — Near-Cutoff Preservation via Dual-Branch Agreement:*
   - Preserve pre-rerank candidates ranked in top 10 of both BM25 and Dense if rerank score is within epsilon of cutoff.
   - *Rejection Reason:* Tune20 has 0 oracle-proxy drops and 0 semantically plausible rerank-loss annotations under the current forensic taxonomy; injecting near-cutoff candidates risks displacing higher-quality reranked candidates on clean queries.
2. *Rule 2 — Structural Legal-Reference Match Retention:*
   - Retain candidates matching explicit legal citation numbers from query understanding.
   - *Rejection Reason:* Q134499 and Q60281 were general queries with empty legal references; this rule would not have fired on either forensic seed and would add dead serving complexity.
3. *Rule 3 — Global RRF / Cross-Encoder Score Blending:*
   - Blend normalized cross-encoder logits with RRF retrieval rank scores.
   - *Rejection Reason:* Not pursued. T5-2A already showed adjacent global reranker-bypass and naive rank/lexical heuristics were unstable; no causal evidence justifies introducing a new global score-blending policy in T5-5A.

### Decision & Rationale
- **Decision:** NO_RERANK_POLICY_JUSTIFIED
- **Rationale:** No clean Tune20 evidence supports a deployable reranker intervention. Historical Holdout10 forensic seeds are contaminated and cannot validate a policy. Q60281 additionally demonstrates that reference-overlap oracle labels can produce semantic false positives. Production reranker behavior remains unchanged. Further error decomposition will investigate downstream generator output contract alignment and extractive fallback efficiency.

### Production Impact & Known Risks
- **Production Impact:** Zero serving or runtime behavior changes (`CrossEncoderReranker` and `RerankingRetriever` remain unchanged; `src/**` untouched).
- **Rollback Authority:** `024e6b5e7481d7dc3e1a4878e158f5f32c0f3080`.

### Next Step
- Proceed to Milestone 57 (Generator Contract / Fallback Efficiency Investigation).

## 17. T5-6A — Generator Contract and Fallback Efficiency Investigation

- **Status:** ACCEPTED — EXTERNAL REVIEW PASSED
- **Date:** 2026-08-22
- **Objective:** Investigate whether historical generator contract failures (56 draft rejections across 28/30 questions in FAST30) stem from output schema rejection vs underlying model incapacity, verify whether extractive fallback identities match selected evidence exactly, audit token efficiency, and preregister a controlled output contract measurement (T5-6B) on Tune20 under Design B.
- **Hypothesis:** Historical M49.1 generator fallback is dominated by prompt/schema contract rejection (`plain_text_markers` unparseable without citation/evidence markers) rather than model failure; a schema-aligned output contract will reduce draft rejections, eliminate wasteful retries, and recover genuine model-generated answers without degrading official retrieval quality.

### Authority & Artifact Scope
- **Starting / Rollback Authority:** `e810bfadf0c3a7e80f0d70d43e84e3258c842c63`
- **Analysis / Acceptance Commit SHA:** `5a19d18bd43a7aa865d420ded64dc886718e0c75`
- **Branch:** `t5/baseline-error-decomposition`
- **Parent Commit:** `e810bfadf0c3a7e80f0d70d43e84e3258c842c63`
- **Historical FAST30 Telemetry Scope:** `t5-1c-fast30-clean1-evidence.zip` (SHA-256: `be2c7a3b17232e4f568d1bc0be98e41c6a3fc1307d3576c68d499acff039a04f`).
- **Tripartite Configuration Authority Distinction:**
  1. **Historical FAST30 Execution Identity (Persisted):**
     - `production_baseline_source_sha`: `87e71eb7661eb9cda1e63f4f0af16ef4613dadfb`
     - `measurement_harness_source_sha`: `fa3b902da1041ac9d2b35cbe61a47351bccf10eb`
     - `application_config_hash`: `ca1f0aa45a22df7aa9293e42df94473c059b4480b8c03d6ef942c21e9f3da261`
     - `m49_generator_tree_sha256`: `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b`
     - `m45_archive_sha256`: `7e78ad60ff2982592a9471eb8704fce44042add0496268fade3f32db1823ea7a`
     - `official_scorer_sha256`: `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`
  2. **M49.1 Repository Base Config Authority:**
     - Verified documented settings: `float16`, `image_text_to_text`, 8192 input, 1536 output, `repetition_penalty=1.08`, `no_repeat_ngram_size=8`, `salvage_rendering="standalone"`, `ClaimVerificationConfig(require_inline_citations=False, min_lexical=0.2, max_claims=60)`.
  3. **Exact Historical Resolved Full ApplicationConfig:**
     - **`FAST30_EXECUTION_CONFIG_HASH_NOT_REPRODUCED`** (Accepted limitation: Expected `ca1f0aa45a22df7aa9293e42df94473c059b4480b8c03d6ef942c21e9f3da261`, Reconstructed `a62ba27e2ff5d02987bd68d0611226fed71f8b6275443a0764d692605b80431b`).
- **Population Separation & Quarantined Partitions:**
  - `T5_6B_TUNE20_ORDERED_QIDS_SHA256`: `9cb88a00c2bcf9fbc0f24411de2f427d6a30f5da0f57feaaafb629f9fcd60b28`
  - `T5_6B_FROZEN_GENERATOR_INPUT_SHA256`: `2fefbb03125f9927edf67c8bc8c165bdd856e1dd2eef0c737aefc7387a2cbbf2`
  - Holdout10 is contaminated and quarantined; zero evaluation permitted in T5-6B.
- **Model / Generator Authority:**
  - `TRAINING_MANIFEST_MODEL_REVISION`: Base `15852e8c16360a2fea060d615a32b45270f8a8fc` (upstream pretrained Qwen revision)
  - `TRAINING_MANIFEST_GENERATOR_TREE_SHA256`: `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b`
  - `RESOLVED_GENERATION_CONFIG_MODEL_REVISION`: `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b` (explicitly bound by `m491_kaggle_candidate_dev.py` via `generation["model_revision"] = manifest["merged_model_sha256"]`).
  - `EXPECTED_M49_GENERATOR_TREE_SHA256`: `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b`
  - `TOKENIZER_AUTHORITY_COVERED_BY_MODEL_TREE_SHA256`: Tokenizer config and vocabulary are packaged within the immutable merged tree at `/kaggle/working/m49-generator-merged`.
  - `PREREGISTERED_PRODUCTION_GENERATOR_BLOB_SHA`: `1543eac766c0cf24ccb7904d8bfa2b802547e3c5` (verified byte-identical between `87e71eb7661eb9cda1e63f4f0af16ef4613dadfb` and `e810bfadf0c3a7e80f0d70d43e84e3258c842c63`).

### Complete Descriptive Forensic Census (FAST30 Population, n=30)
1. **Fallback & Generation Outcomes:**
   - 28 / 30 questions experienced `generator_model_error_fallback` directly associated with `structured_output_schema` rejection events.
   - 1 / 30 questions (`Q83501`) succeeded via `supported_claim_salvage_applied` (ROUGE-L 0.6326, METEOR 0.4367).
   - 1 / 30 questions (`Q54485`) terminated with `insufficient_evidence=true` (ROUGE-L 0.0690, METEOR 0.0051).
   - 0 / 30 questions completed end-to-end unassisted semantic synthesis without warning flags.
2. **Draft Rejection Census:**
   - Exactly 56 structured draft rejection events recorded across the 28 fallback questions (28 attempt 1 rejections, 28 attempt 2 rejections).
   - All 56 rejections exhibited error type `structured_output_schema`.
3. **Exact Fallback Evidence Identity Verification:**
   - 28 / 28 fallback questions verified as `EXACT_IDENTITY_MATCH` with N=1, identical to top-selected evidence `[E1]`.
   - Length ratio of fallback answers against reference answers: mean 1.868 +/- 1.623 (range [0.260, 6.786]).
4. **Historical Descriptive Reference Metrics (Descriptive Reference Only):**
   - FAST30 (n=30): ROUGE-L = 0.44956125747562414, METEOR = 0.36059965374134695
   - Tune20 (n=20): ROUGE-L = 0.4831331248436325, METEOR = 0.4046940181246421
   - Holdout10 (n=10): ROUGE-L = 0.3824175227396073, METEOR = 0.2724109249747567

### Root Cause Diagnosis & Evidence Boundary
- **PROVEN:**
  - 28 / 30 questions reached `generator_model_error_fallback` associated with `structured_output_schema` rejection.
  - 56 completed generation drafts reached parsing and were rejected across 28 questions (2 attempts per question).
  - Historical prompt contract was `prompt_schema_mode="plain_text_markers"`.
  - Raw rejected completion text was not persisted in historical telemetry.
- **PLAUSIBLE HYPOTHESIS:**
  - The model's natural output format may have been mismatched with the `[E#]` evidence-marker contract.
- **NOT PROVEN:**
  - That all 56 rejected outputs were marker-free prose.
  - The exact malformed shape or token sequence of rejected drafts.
  - That an alternative output contract will necessarily eliminate rejections without quality degradation.
- **Role of T5-6B:** T5-6B exists as a controlled experiment to discriminate this hypothesis on frozen generator inputs.

### T5-6B Preregistered Experimental Specification (Design B — Same-Run Controlled Output-Contract Measurement)
1. **Shared Base Configuration Authority (M49.1-Repository-Derived T5-6B Control Config):**
   - **GenerationConfig:**
     - `backend`: "transformers"
     - `model_name`: "/kaggle/working/m49-generator-merged"
     - `model_revision`: "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b"
     - `device`: "cuda"
     - `torch_dtype`: "float16"
     - `model_loader`: "image_text_to_text"
     - `local_files_only`: True
     - `max_input_tokens`: 8192
     - `temperature`: 0.0
     - `max_output_tokens`: 1536
     - `repetition_penalty`: 1.08
     - `no_repeat_ngram_size`: 8
     - `max_structured_output_retries`: 1
     - `max_model_error_retries`: 1
     - `model_failure_policy`: "top_evidence"
     - `max_grounding_repair_retries`: 1
     - `grounding_failure_policy`: "supported_claims_or_top_evidence"
     - `extractive_fallback_max_evidence`: 1
     - `salvage_rendering`: "standalone"
     - `answer_style`: "competition_reference"
   - **ClaimVerificationConfig:**
     - `enabled`: True
     - `require_inline_citations`: False
     - `minimum_lexical_support`: 0.2
     - `minimum_claim_tokens`: 2
     - `require_numeric_match`: True
     - `require_negation_match`: True
     - `max_claims`: 60
2. **Output Contract Candidates (Differing ONLY in `prompt_schema_mode`):**
   - `CONTROL`: `prompt_schema_mode="plain_text_markers"` (`T5_6B_CONTROL_GENERATION_CONFIG_SHA256`: `657ee87bdeac212857e9ec199c9fe34d6f7975ff5078c2371e1e6c2dba8738a7`)
   - `CANDIDATE_COMPACT`: `prompt_schema_mode="compact_example"` (`T5_6B_COMPACT_GENERATION_CONFIG_SHA256`: `810142a8ebacca5331ec13f1777be7edb6d4357b61a1c155d36751049b91bab2`)
   - `CANDIDATE_JSON`: `prompt_schema_mode="json_schema"` (`T5_6B_JSON_GENERATION_CONFIG_SHA256`: `8c930f08131b9cc9e07f1427d21b1d5e96c38431ca2d65f1e080abf04989596f`)
   - `CLAIM_CONFIG`: `T5_6B_CLAIM_VERIFICATION_CONFIG_SHA256`: `fcb8cd2e65b74407be42a312f80624bb2be996e1a79d6a9228758d0893f23988`
3. **Execution Invariants & Control Authority (Design B):**
   - **Environment Gate:** Runner must verify single GPU `cuda` with `float16` capability. Zero runtime substitution permitted.
   - **Frozen Input Gate:** Runner verifies `T5_6B_FROZEN_GENERATOR_INPUT_SHA256` (`2fefbb03125f9927edf67c8bc8c165bdd856e1dd2eef0c737aefc7387a2cbbf2`) before calling the model.
   - **Same-Run Control Authority:** Same-run `plain_text_markers` executed under the exact base config is the authoritative comparator for all candidate metrics. If control execution fails infrastructure or authority gates, emit `T5_6B_CONTROL_EXECUTION_FAILED` and STOP.
4. **Candidate Advancement Gate (6 Cumulative Conditions vs Same-Run Control):**
   1. Parser Acceptance Rate >= 80.0% on Tune20
   2. Official Tune20 METEOR >= Same-Run `plain_text_markers` Control METEOR
   3. Official Tune20 ROUGE-L >= Same-Run `plain_text_markers` Control ROUGE-L
   4. Citation identity validity == 100.0% for all non-abstaining responses
   5. Contract-rejection fallback count <= Same-Run Control fallback count
   6. Insufficient-evidence count <= Same-Run Control count
5. **Tie-Break Precedence:**
   - 1. Highest Official METEOR -> 2. Highest Official ROUGE-L -> 3. Highest Parser Acceptance -> 4. Lexicographical Name.
6. **Per-Call Telemetry Schema (`t5_generator_call_telemetry_v1`):**
   - `question_id`, `candidate_contract`, `call_stage`, `call_index`, `provider_attempt_index`, `provider_call_success`, `provider_error_type`, `raw_completion_text`, `parse_result`, `rejection_error_type`, `structured_output_attempt`, `parsed_cited_evidence_ids`, `grounding_verification_pass`, `grounding_claim_count`, `grounding_supported_claim_count`, `final_generator_path`, `fallback_reason`, `input_token_count`, `output_token_count`.
7. **Manifest Schema (`t5_generator_measurement_manifest_v1`):**
   - `candidate_contract`, `repository_base_sha`, `measurement_source_sha`, `production_generator_blob_sha`, `model_artifact_sha256`, `generation_config_sha256`, `claim_verification_config_sha256`, `frozen_generator_input_sha256`, `tune20_ordered_qids_sha256`, `official_scorer_sha256`, `raw_completion_artifact_sha256`, `record_count`, all metric outputs.

### Findings & Decision
- **Finding:** 28/30 historical FAST30 questions ended in model-error fallback associated with output-contract rejection (`structured_output_schema` rejection).
- **Decision:** **`NEW_CONTROLLED_GENERATOR_MEASUREMENT_REQUIRED`**
- **Follow-up / Next Milestone:** Execute T5-6B controlled generator measurement under the preregistered Design B protocol upon external review acceptance.

## 18. T5-6B-PREP - Design-B Measurement Runner and Telemetry Implementation (Fix 5.1: Official Scorer Authority and Hardened Authority Closure)
- **Status:** TOOLING COMPLETE - PENDING FINAL EXTERNAL REVIEW (NO INFERENCE RUN)
- **Objective:** Construct, harden, and unit-test a deterministic, frozen-input measurement harness and telemetry framework for executing the preregistered T5-6B Design-B controlled output-contract experiment across the three candidate arms (control: plain_text_markers, compact: compact_example, json_schema: json_schema) without loading weights or executing model inference during the prep phase.
- **Fix 5 / 5.1 Hardening Accomplishments:**
  1. **Exact Official Scoring Entrypoint Execution:** In `score_tune20_answers()`, macro ROUGE-L and METEOR metrics are computed by directly calling the verified official entrypoint `eval_qa(y_pred, y_true)` inside `scoring.py` from the pinned scorer archive (`OFFICIAL_SCORER_ARCHIVE_SHA256`). Payloads strictly match `{qid: {"answer": pred}}` and `{qid: ref}` for all 20 Tune20 QIDs. Returned official macro values are the ONLY metrics participating in candidate advancement. Per-question scores are evaluated through the exact same official entrypoint on single-QID dicts.
  2. **Exact Prediction and Reference QID Set Validation:** Required `set(predicted_answers.keys()) == set(CANONICAL_TUNE20_ORDERED_QIDS)` and `set(reference_answers.keys()) == set(CANONICAL_TUNE20_ORDERED_QIDS)`. Missing, extra, or mismatched QIDs fail closed immediately with `DataValidationError`.
  3. **Unit Test Scorer Isolation:** Removed machine-specific paths and skip decorators from committed unit tests. All committed tests execute against a synthetic `eval_qa` fixture verifying entrypoint delegation, payload structure, sentinel returns, and QID set boundaries.
  4. **Transparent Token Provider Architecture:** Removed `complete()` override entirely from `ObservableTransformersChatProvider`. Installed `_GenerateProxy` inside `_load_runtime()`, preserving exact `complete` method identity from `TransformersChatProvider`. Output object identity and exception handling preserved with pre-call input token recording.
  5. **Production Prompt Sentinels Restored:** Pinned exact Vietnamese production sentinel substrings from `ModelBackedAnswerGenerator._correction_prompt` (`STRUCTURED_RETRY_SENTINEL = "OUTPUT TRƯỚC KHÔNG HỢP LỆ. Hãy tạo lại từ đầu."`, `GROUNDING_REPAIR_SENTINEL = "BẢN NHÁP TRƯỚC KHÔNG QUA KIỂM TRA GROUNDING:"`). `classify_prompt_call_stage` raises `DataValidationError` (ambiguous) if both sentinels are present simultaneously.
  6. **Deterministic Logical Call / Provider Attempt State Machine:** State machine in `MeasurementProviderObserver.complete()` tracks prompt key `(system_prompt_sha256, user_prompt_sha256, call_stage)`. Repeated prompt (e.g. ModelError retry) preserves `call_index` and increments `provider_attempt_index`. New prompt or stage increments `call_index` and resets `provider_attempt_index = 1`.
  7. **Logger Rejection Context Binding:** `ModelGeneratorRejectionObserver` binds rejections to `(question_id, candidate_contract, logical_call_index)`. Provider observer calls `set_active_logical_call` before each provider execution, ensuring rejections during draft parsing are attributed to the correct logical call rather than retry attempt index.
  8. **Deterministic Query Reconstruction:** `reconstruct_query` applies exact `ServingService.create_query` normalization (whitespace stripping, NFC Unicode normalization of collapsed whitespace) and sets `query_id = f"t5-6b:{question_id}"`.
  9. **Exact Parser Acceptance Definition:** `parser_accepted` defined as `any(c.parse_result == "ACCEPTED" for c in calls if c.provider_call_success)`. Initial rejection followed by successful structured retry is correctly counted as parser accepted.
  10. **Preregistered Contract Rejection Metrics:** `had_contract_rejection` checks `rejection_error_type == "structured_output_schema"`. `contract_rejection_fallback_count` counts questions ending in `MODEL_ERROR_FALLBACK` with `had_contract_rejection == True`. `total_structured_output_rejections` counts schema rejection calls strictly.
  11. **Strict Citation Identity Validation:** `evaluate_citation_identity_validity` validates that every citation's `evidence_id` exists in supplied evidence, `chunk_id` matches the evidence item's `chunk_id`, no duplicate `(evidence_id, chunk_id)` pairs exist, and no `[E#]` markers in answer text reference unknown evidence IDs.
  12. **Model Tree SHA Algorithm Authority:** Restored exact historical model-tree SHA-256 algorithm from `notebooks/m491_kaggle_candidate_dev.py` at commit `10681c8` (sorted `rglob("*")`, 8-byte big-endian length prefix, raw UTF-8 relative path, raw 32-byte file SHA).
  13. **Resume Arm-Order Fail-Closed Gates:** Enforced strict execution arm ordering (`control` -> `compact` -> `json_schema`) on resume; later arms with completed QIDs when earlier arms are incomplete fail closed.
  14. **Non-Blocking Execution Exclusivity:** Enforced `_RUNNER_LOCK.acquire(blocking=False)` and `_LEASE_LOCK.acquire(blocking=False)` to reject overlapping runner or logger leases immediately without deadlocking.
  15. **Test Collection Evidence & Zero Regression:** Base authority `5a19d18` test collection confirmed at 558 tests. Fix 5.1 test collection is 633 tests (558 base + 75 measurement tests: 632 passed, 1 skipped). Zero tests deleted or regressed.
- **Verification Evidence:**
  - Generation preflight: `python scripts/t5_generator_contract_measurement.py --archive C:/Users/Nguyen/Downloads/t5-1c-fast30-clean1-evidence.zip --preflight-only` -> `preflight_status: SUCCESS` (20 Tune20 records validated, 169 evidence items; `scorer_gate = NOT_APPLICABLE_TO_GENERATION_PHASE`).
  - Separate PREP scorer-authority validation: PASS (`eval_qa` verified on 8 golden vectors with exact archive & member SHA-256 matches).
  - T5-6B Measurement Suite: `pytest tests/unit/evaluation/test_t5_generator_contract_measurement.py` -> 75 / 75 passed.
  - Focused Evaluation Suites: `pytest tests/unit/evaluation/test_t5_generator_fallback_analysis.py tests/unit/evaluation/test_t5_reranker_forensics.py tests/unit/evaluation/test_t5_evidence_policy_analysis.py tests/unit/evaluation/test_t5_generator_contract_measurement.py` -> 150 / 150 passed.
  - Full Repo Test Suite: `pytest` -> 632 passed, 1 skipped (0 failures).
