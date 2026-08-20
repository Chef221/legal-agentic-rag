# CURRENT-WORK.md

## 1. Purpose

This file is the short-lived working-memory handoff for coding agents working on
`legal-agentic-rag`.

It exists to answer:

- What is the current development frontier?
- What is the stable baseline?
- What was changed recently?
- What must be validated next?
- What must not be changed accidentally?
- What repository facts are currently uncertain or inconsistent?

This file is **not** the highest-level source of truth.

The persistent source-of-truth order remains:

1. `AGENTS.md`
2. authoritative files under `docs/`
3. approved source code and tests
4. explicitly confirmed new user decisions

When this file conflicts with an authoritative contract, the authoritative
contract wins. When documentation conflicts with the current implementation,
the agent must report the discrepancy instead of silently choosing one.

---

## 2. Handoff Snapshot

Repository:

- `Chef221/legal-agentic-rag`
- default branch: `main`
- live `origin/main` MUST be verified at the start of every agent session; `CURRENT-WORK.md` does not pin a live HEAD because this document is itself versioned.

Stable provenance & artifact identities:

- current package version: `0.50.7`
- Phase B1B reviewed execution commit: `38a6feec8867a41454c453cce9c54b162801579e`
- Phase B1B canonical evidence archive SHA-256: `f392cc650699ecc562cb43ea0ea7f6e965455a36a621843ec6a882172913c9c3`
- Phase B1A.2 frozen baseline results SHA-256: `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a`
- Phase B1A.2 execution commit: `9265f3dadcf1ef0170f0abe618519da1657fc55e`

---

## 3. Project Mission

The project builds an Agentic Retrieval-Augmented Generation system for
Vietnamese legal question answering for UIT Data Science Challenge 2026,
Task 2.

The system must retrieve relevant legal evidence, rank/select grounded context,
generate a Vietnamese answer, verify claims/citations, and produce competition
output while preserving reproducibility and compliance.

The repository is not an exploratory notebook. It is a controlled competition
system with explicit contracts, artifact lineage, regression gates, and
documented design decisions.

---

## 4. Competition and Data Invariants

Unless the user explicitly approves a policy change and the authoritative
documentation is updated, preserve all of the following:

- active data policy is `competition_only`;
- use only official competition data for the competition pipeline;
- do not introduce an external corpus;
- do not create or use synthetic competition QA, answers, evidence, hard
  negatives, or training examples;
- raw UIT DSC schemas stay behind the competition adapter boundary;
- official artifacts must preserve validated lineage/checksums;
- do not silently reuse vector/index artifacts after changing corpus,
  preprocessing, chunking, embedding identity, or another lineage-defining
  input;
- no external model API may become part of the competition system;
- model identity and approval/compliance constraints must remain traceable;
- total model-parameter constraints documented by the repository must be
  respected;
- submission/scorer contracts must not be changed from assumptions or metric
  names;
- public/private competition behavior must remain reproducible from commit,
  config, model identity/revision, corpus revision, and artifact lineage.

Before any competition-facing change, read:

- `docs/11-COMPETITION-COMPLIANCE.md`
- `docs/13-UIT-DSC-2026-DATA-CONTRACT.md`
- `docs/15-OFFICIAL-SCORING-CONTRACT.md`

---

## 5. Stable Quality-Control Baseline

The quality-control comparison baseline remains:

- baseline: **M43.1** (public batch reference)
- baseline commit: `96e6d5a5c77ff19761d234a933e8684a44efb3bd`
- generator: `Qwen/Qwen2.5-3B-Instruct` (pretrained)
- public batch: 1,000 / 1,000 IDs completed
- public METEOR: `0.07862292376534387`
- public ROUGE-L: `0.16735433212043324`
- 425 abstentions
- 384 citation-verification failures
- 33 generator model errors

The frozen operational reliability baseline remains:

- baseline: **M49.6** (commit `9b0cd0b1d40fb01bb62d4841f7728af2264f3957`)
- generator: `Qwen/Qwen2.5-3B-Instruct` (pretrained) with bounded missing-field recovery
- targeted 2-ID gate: 0 errors, 2/2 missing-field corrections succeeded
- immutable 50-question smoke: 0 model errors, 46/50 verified answers, 4 abstentions

Do **not** confuse:

- the stable comparison baseline (`M43.1` / `M49.6`);
- the current repository/package version (`0.50.5`);
- experimental candidate models (e.g. M50-C1, M50-C2).

No fine-tuned generator model has been promoted to production.

---

## 6. M43.1 As-Built Online Baseline

The M43.1 online path is the comparison reference.

Conceptually:

```text
question
  -> query understanding
  -> BM25 retrieval
  -> dense E5 retrieval
  -> RRF fusion
  -> evidence/context selection
  -> Qwen2.5-3B generation
  -> claim/citation verification
  -> internal AnswerResponse
  -> competition rendering
```

Important:

- reranker code exists in the repository but is not part of the proven M43.1
  online baseline;
- graph retrieval code exists but graph expansion is not part of the proven
  M43.1 online baseline;
- implementation existence is not evidence of metric benefit;
- do not enable dormant components merely because they appear architecturally
  attractive.

Before modifying a module, follow the module-specific reading path in
`docs/00-START-HERE.md`.

---

## 7. Completed Milestone: M50 Official-Data Fine-Tuning (CLOSED)

Milestone 50 was the official-data LegalQA generator fine-tuning experiment.
The experiment has been completed and **officially closed with all fine-tuned candidates rejected**.

### 7.1 M50-C1 Execution & Rejection
- **Recipe**: Aggressive QLoRA ($r=8, \alpha=16, \text{LR}=5\times 10^{-5}$, targets $\{q, k, v, o\}$, 3.68M params, 282 steps);
- **Outcome**: Teacher-forced validation loss converged (1.09828), but free autoregressive generation suffered catastrophic collapse (14/20 cap without EOS, 18/20 high repetition loops);
- **Status**: **REJECTED** before SCREEN.

### 7.2 M50-C2 Pilot Execution & Step 100 Selection
- **Recipe**: Conservative QLoRA ($r=4, \alpha=8, \text{LR}=10^{-5}$, targets $\{q, v\}$, 921,600 params, 150 steps, Qwen ChatML EOS canonicalization);
- **Pilot Gates (VAL20)**: Steps 50, 100, and 150 all passed safety (20/20 EOS, 0 cap-no-EOS) and semantic gates;
- **Ranking**: `[100, 150, 50]`;
- **Selected Checkpoint**: **Step 100** (combined delta $+0.01182$).

### 7.3 M50-C2 Step 100 Final SCREEN617 Evaluation & Rejection
- **Evaluation**: 617/617 holdout questions generated (0 errors);
- **Generation Health**: Regressed across all 3 key safety indicators relative to BASE:
  - Cap without EOS: 40/617 (6.48%) vs BASE 30/617 (4.86%) — **FAIL**
  - High Repetition (repeat8 $\ge 0.25$): 56/617 (9.08%) vs BASE 39/617 (6.32%) — **FAIL**
  - Terminal EOS rate: 577/617 (93.52%) vs BASE 587/617 (95.14%) — **FAIL**
- **Semantic Point Estimates**:
  - METEOR mean delta: **-0.002803** (95% CI `[-0.006577, +0.000909]`, W/T/L 291/15/311) — **FAIL** ($\le 0$)
  - ROUGE-L mean delta: **-0.002594** (95% CI `[-0.007145, +0.001844]`) — within tolerance ($\ge -0.01$) but directionally negative
  - Combined mean delta: **-0.002698** (95% CI `[-0.006544, +0.001103]`) — **FAIL** ($\le 0$)
- **Semantic Gate Decision**: `semantic_signal_pass = false`, `semantic_strict_pass = false` (failed primary METEOR and combined delta criteria)
- **Final Decision**: **FAIL**
- **Candidate Status**: **REJECTED**
- **Production Promotion**: **NO**

### 7.4 Holdout Consumption Status
- **`screen_holdout.json` IS CONSUMED**: Evaluated once as the formal holdout for M50-C2. It must not be treated as an untouched holdout for future adaptive tuning, cherry-picking, or candidate iterations.

---

## 8. Frozen Reliability Baseline: M49.6 GPU Validation Results

The M49.6 reliability candidate (`9b0cd0b1d40fb01bb62d4841f7728af2264f3957`, version `0.49.6`) successfully passed both acceptance gates on Kaggle Tesla T4:

1. **Targeted 2-ID Gate (`139655`, `25945`)**:
   - `records = 2`
   - `generator_model_errors = 0`
   - `missing_field_correction_attempted = 2`
   - `missing_field_correction_succeeded = 2`
   - `stop_reason: answer_verified = 2`

2. **Immutable 50-Question Smoke Gate**:
   - `records = 50`
   - `stop_reason_counts: answer_verified = 46, insufficient_evidence = 4`
   - `generator_model_errors = 0`
   - `retrieval_model_errors = 0`
   - `citation_verification_failed = 0`
   - `missing_field_correction: attempted = 2, succeeded = 2, failed = 0`
   - `numeric_salvage = 2/2`, `supported_claim_salvage = 1/1`

M49.6 remains the frozen fallback baseline for production answering. It was not replaced or superseded by M50.

---

## 9. Immediate Predecessors

- **M49.5**: bounded local structural terminal schema recovery;
- **M49.6**: bounded terminal model-correction attempt for missing-required-field failures;
- **M50**: official-data QLoRA fine-tuning infrastructure, pilot gates, and holdout evaluation (closed; candidates rejected).

---

## 10. Recent Remote History Relevant to the Handoff

```text
560d059 Close M50 after C2 holdout rejection
4444825 Fix Qwen EOS canonicalization for M50 SFT
dcf47f5 Add generation-safe M50-C2 pilot infrastructure
690c352 Fix M50 candidate config integration
dc4f427 Fix M50 training runtime correctness
aa4d110 Add official-data QLoRA experiment infrastructure
9b0cd0b Add bounded missing-field model correction
be190a5 Add bounded terminal schema recovery
4b4d8eb Add structured claim salvage diagnostics
0f4ed75 Add deterministic numeric claim salvage
885c123 Add numeric-only grounded repair
e36aa63 Improve bounded structured generation recovery
d64bf34 Use claim-linked model output for citations
1712d67 Add batch quality gates and context diversity ablation
8cdec61 Improve citation coverage diagnostics
189e125 Fix Kaggle batch command variable expansion
0b4000a Stream durable competition batch progress
```

---

## 11. Current Known Failure Themes

The system is operational end-to-end, but quality remains weak relative to the
competition metric.

Known/important failure themes include:

- excessive abstention;
- citation-verification failure;
- generator model/structured-output failure;
- retrieval/context selection not always matching the answer form rewarded by
  the references/metrics;
- LoRA fine-tuning on QA pairs did not outperform the pretrained base model on unseen questions;
- the risk of treating intermediate validation loss or small-sample probes as proof of holdout generalization.

---

## 12. Development Protocol

Before editing:

1. Read `AGENTS.md`.
2. Read `docs/00-START-HERE.md`.
3. Read this file.
4. Read the authoritative docs for the subsystem.
5. Inspect the current implementation.
6. Inspect the relevant tests.
7. Inspect recent related git history.
8. State the hypothesis being tested.
9. Define the smallest change that can test the hypothesis.
10. Define validation and regression gates before implementation.

During implementation:

- keep the experiment scope narrow;
- avoid opportunistic architectural rewrites;
- do not change unrelated configs;
- do not silently modify baseline profiles;
- preserve default-off behavior for experimental features unless promotion has
  been explicitly approved;
- preserve deterministic/reproducible behavior where required;
- preserve telemetry sanitization and privacy boundaries.

After implementation:

1. run targeted unit/integration tests;
2. run the experiment-specific gate;
3. run the required regression gate;
4. report exactly what changed;
5. report exactly what was intentionally not changed;
6. report regressions and uncertainty;
7. update authoritative docs if persistent behavior/architecture/contracts
   changed;
8. update this file if the active frontier or next action changed.

---

## 13. Documentation vs Implementation Policy

Documentation is authoritative for contracts and accepted decisions, but rapid
experimentation can create stale status text.

Therefore:

- never assume every status/version sentence in docs is current;
- never ignore documentation contracts because a status line is stale;
- distinguish:
  - contract;
  - design decision;
  - historical experiment report;
  - current implementation state;
- reconcile important discrepancies explicitly.

---

## 14. Local-State Safety

Before modifying anything:

```bash
git status --short
git diff --stat
git diff
```

Do not commit:

- competition datasets;
- model weights;
- generated large artifacts;
- secrets/tokens;
- local machine paths;
- private competition outputs that repository policy excludes.

---

## 15. Current Next Action

The immediate next action is:

1. **Phase A Census CLOSED**: The authoritative 991-question census and forensic architecture findings are documented in [`docs/22-PHASE-A-CLOSURE.md`](file:///c:/legal-agentic-rag/docs/22-PHASE-A-CLOSURE.md).
2. **Phase B1A.2 Graph Equivalence Experiment COMPLETE**: Run archive `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a` confirmed `GRAPH_REDUNDANCY_PROVEN` with 22/22 seed match, 22/22 top-8 match, and score diffs within $10^{-6}$.
3. **Phase B1B Structural Graph Removal VERIFIED & CLOSED**: Kaggle post-change equivalence verification (reviewed commit `38a6feec8867a41454c453cce9c54b162801579e`, evidence archive `phase-b1b-graphless-equivalence-evidence.zip` SHA-256 `f392cc650699ecc562cb43ea0ea7f6e965455a36a621843ec6a882172913c9c3`, size `14157 bytes`) yielded mechanical verdict **`B1B_EQUIVALENCE_PASS`** (`b1b_verified = true`). Confirmed 22/22 exact matches, 22/22 score tolerance passes ($\le 10^{-6}$), 22/22 branch depth 40, candidate query, fusion limit, final top-k, and route plan passes, with 0 retrieval model errors. Competition graph is officially removed from runtime and build ([`docs/25-PHASE-B1B-GRAPH-REMOVAL.md`](file:///c:/legal-agentic-rag/docs/25-PHASE-B1B-GRAPH-REMOVAL.md)).
4. **Immediate Post-B1B Research Priority**: Candidate-pool / reranker audit (S20 vs H40).
   - In Phase B1A.2, S20 vs H40 changed top-8 evidence in 17/22 relationship cases.
   - Research question: Does reranking the fused top 40 introduce distractors compared with reranking the fused top 20, or does the larger pool recover better evidence?
   - Invariant: H40 remains unpromoted. Do NOT claim H40 is superior. A separate controlled experiment is required.
5. **Subsequent Planned Research Frontiers (Sequenced Order)**:
   - Priority B: Verification-correctness audit (known failure modes: legal-condition inversion, actor/role inversion, wrong-source/wrong-document answers; key cases `102047`, `147239`, `26541`, `95861`).
   - Priority C: Retrieval-miss analysis (gold source misses on `26541`, `95861`).
   - Priority D: Generation/fine-tuning later (deferred until retrieval/reranker/verifier architecture is better understood).

---

## 16. Status of Historical Questions

- **What happened in Phase B1B?** Resolved: Phase B1B post-change verification passed cleanly with `B1B_EQUIVALENCE_PASS`. Exact S20 retrieval behavior preserved, zero-edge competition graph removed, 3 online artifacts active.
- **What happened in Phase B1A.2?** Resolved: Graph traversal proved redundant on UIT DSC competition corpus (0 edges, 0 steps). S20 matches G perfectly across all 22 cases. H40 diverges on 17/22 cases and is preserved as a separate second route.
- **What happened in Phase-A Census?** Resolved: 991/991 benchmark questions completed. 806 answer_verified (81.33%), 177 generation_failed, 7 citation_verification_failed, 10 generator model errors. All 22 relationship queries routed to GRAPH_SEARCH and terminated on Attempt 1.
- **What happened in M50-C1?** Resolved: C1 trained cleanly to step 282 (val loss 1.09828) and showed lexical gain on ROUGE-L, but collapsed in free-generation health (70% cap reached, 90% repetition loops). Conclusively rejected.
- **What happened in M50-C2?** Resolved: C2 reduced capacity ($r=4$, LR $10^{-5}$, targets $\{q, v\}$, 921K params) and canonicalized EOS. Step 100 failed SCREEN617 holdout evaluation across generation health and semantic criteria. Conclusively rejected.
- **Is `screen_holdout.json` still untouched?** Resolved: `screen_holdout.json` has been consumed by the one-shot M50-C2 evaluation and is no longer an untouched holdout for future adaptive candidates.

---

## 17. Update Rules for This File

Update `docs/CURRENT-WORK.md` when:

- the active milestone/frontier changes;
- an experiment gate passes/fails and changes the next action;
- a new blocking issue is discovered;
- a hypothesis is rejected and should not be repeated;
- important local-only state must be summarized for agent handoff;
- the stable comparison baseline changes after explicit promotion.

Do not turn this file into a permanent architecture document.

Move long-lived decisions into the appropriate authoritative documentation and
keep this file focused on the current frontier.

---

## 18. Handoff Summary

```text
Stable comparison baseline:
    M49.6 production RAG pipeline (pretrained Qwen2.5-3B-Instruct)

Current repository state:
    0.50.7 (Phase B1B Structural Competition Graph Removal Closed — B1B_EQUIVALENCE_PASS)

Active development frontier:
    Candidate-pool / reranker audit (S20 vs H40)

Next action:
    Design a controlled candidate-pool / reranker audit comparing S20 vs H40
```
