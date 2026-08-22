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

- **Status:** ACCEPTED — EXTERNAL REVIEW PASSED
- **Date:** 2026-08-22
- **Objective:** Repair observer installation/restoration ownership so sequential diagnostic runners on one shared runtime cannot leak telemetry contexts.
- **Hypothesis:** Explicit instrumentation lease ownership with object-identity restoration ensures sequential diagnostic runs on shared runtimes remain isolated without leaking observer wrappers or handlers.

### Authority
- **Baseline / Starting Authority:** `ca00c133b70905dc0bcff8cb469727264b25a995`
- **Candidate Commit SHA:** PENDING / UNASSIGNED WHILE WORKING TREE IS UNCOMMITTED
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
- Commit and push the accepted T5-H1 measurement-harness checkpoint on `t5/baseline-error-decomposition`. Begin T5-4 only from that committed checkpoint after remote verification.
