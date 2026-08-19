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

Current remote `main` snapshot:

- HEAD: `4444825178e07f5acff3b7a5b2296e06729f0816`
- commit message: `Fix Qwen EOS canonicalization for M50 SFT`
- commit date: 2026-08-18 UTC

Current source/package version:

- `0.50.5` (M50 Official-Data Fine-Tuning Closure & Holdout Rejection)

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
  - METEOR mean delta: **-0.002803** (95% CI `[-0.006577, +0.000909]`, W/T/L 291/15/311) — **FAIL**
  - ROUGE-L mean delta: **-0.002594** (95% CI `[-0.007145, +0.001844]`) — **FAIL**
  - Combined mean delta: **-0.002698** (95% CI `[-0.006544, +0.001103]`) — **FAIL**
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

1. **Review M50 Closure & Holdout Rejection Documentation**: Ensure all team members and future agents understand the empirical findings from C1 and C2.
2. **Post-M50 Strategic Alignment**: Do not start M51 or C3 without an explicit user decision. Formulate new hypotheses for retrieval, context selection, or prompt optimization based on established competition data.
3. **Preserve Production Reliability**: Keep pretrained `Qwen/Qwen2.5-3B-Instruct` within the frozen M49.6 reliability pipeline as the active generator.

---

## 16. Status of Historical Questions

- **What happened in M50-C1?** Resolved: C1 trained cleanly to step 282 (val loss 1.09828) and showed lexical gain on ROUGE-L, but collapsed in free-generation health (70% cap reached, 90% repetition loops). Conclusively rejected.
- **What happened in M50-C2?** Resolved: C2 reduced capacity ($r=4$, LR $10^{-5}$, targets $\{q, v\}$, 921K params) and canonicalized EOS. All checkpoints (50, 100, 150) passed VAL20 gates. Step 100 was selected, but on full SCREEN617 evaluation, it regressed in generation health (6.48% cap-no-EOS, 9.08% repetition) and achieved negative point estimates on METEOR (-0.002803) and ROUGE-L (-0.002594). Conclusively rejected.
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
    0.50.5 (M50 official-data fine-tuning closed; C1 and C2 rejected; SCREEN617 consumed; M49.6 frozen fallback preserved)

Active development frontier:
    Post-M50 strategic review and milestone planning

Next action:
    review M50 closure documentation -> align on next hypothesis/direction
```
