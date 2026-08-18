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

- HEAD: `690c35215160cc11319d5ada5701de4fec8fb82f`
- commit message: `Fix M50 candidate config integration`
- commit date: 2026-08-16 UTC

Current source/package version:

- `0.50.3` (M50-C2 Conservative QLoRA Pilot Infrastructure)

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

- baseline: **M43.1**
- baseline commit: `96e6d5a5c77ff19761d234a933e8684a44efb3bd`
- generator: `Qwen/Qwen2.5-3B-Instruct`
- public batch: 1,000 / 1,000 IDs completed
- public METEOR: `0.07862292376534387`
- public ROUGE-L: `0.16735433212043324`
- 425 abstentions
- 384 citation-verification failures
- 33 generator model errors

Do **not** confuse:

- the stable comparison baseline (`M43.1`);
- the current repository/package version (`0.49.5`);
- the current experiment frontier (`M49.5` at this handoff).

A newer implementation is not automatically a better baseline. Promotion
requires evidence through the documented evaluation protocol.

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

## 7. Current Development Frontier: M50 (Official-Data QLoRA Fine-Tuning)

The active development frontier is **M50 — Official-Data LegalQA Generator Fine-Tuning**.

Background and status:

- M49.6 completed both targeted 2-ID and immutable 50-smoke gates with zero errors, establishing the frozen reliability baseline;
- M50 Phase 0 & 0.5 established competition compliance, three-way split design (`sft_train.json` ~4,500, `sft_val.json` ~500, `screen_holdout.json` ~617), and sequence length $L=1536$ truncation ceiling;
- `development.json` (991 records) is preserved as the frozen historical development benchmark (strictly excluded from gradient updates and intermediate selection);
- `quarantined.json` (392 records) remains permanently excluded;
- M50 Phase 1 infrastructure is implemented locally: deterministic splitter, answer-only dataset with `-100` prompt loss masking, dynamic collator, QLoRA Candidate-1 trainer with parameter preflight, cached BASE direct-QA screening runner, and paired bootstrap scoring;
- Local unit tests (513 passed) and pre-commit checks verified; Kaggle GPU training execution is pending.

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

M49.6 is the frozen fallback baseline. It must not be redesigned or weakened.

---

## 9. Immediate Predecessors: M49.5 and M49.6

- **M49.5**: bounded local structural terminal schema recovery (shape normalization, single claim wrapping, duplicate stripping, excess complete claim dropping);
- **M49.6**: bounded terminal model-correction attempt specifically for unrecoverable missing-required-field failures;
- **M50.1**: official-data QLoRA fine-tuning infrastructure and staged direct-QA semantic screening ladder.

Each experiment remains independently measurable with its own closed telemetry.

Recent commits show the work progressed through:

- bounded structured-generation recovery;
- claim-linked model output for citations;
- numeric-only grounded repair;
- deterministic numeric-claim salvage;
- structured claim-salvage diagnostics;
- bounded terminal schema recovery.

Do not reimplement these ideas from scratch before inspecting the current code
and tests.

---

## 10. Recent Remote History Relevant to the Handoff

Recent `main` commits at the time this file was prepared:

```text
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
efbc544 Pin low-memory startup runbook commit
edc5a35 Stabilize low-memory Qwen startup
f217ceb Prepare end-to-end reranker answer ablation
7383376 m44.3
f8d1f68 Add leakage-safe evaluation and reranker diagnostics
7466388 Document M43 baseline and team improvement plan
96e6d5a Handle generator abstention in agent workflow
```

When taking over an active task, inspect the full diff of the relevant recent
commits instead of relying only on these commit titles.

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
- the risk of improving one local symptom while regressing another subsystem;
- the risk of treating implemented optional components as proven improvements;
- documentation/status drift during rapid milestone iteration.

Any proposed change must identify which failure mode it is intended to affect
and how the effect will be measured.

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

At this handoff, one known example is the source/package version being `0.49.5`
while some overview text still reports `0.49.3`.

Therefore:

- never assume every status/version sentence in docs is current;
- never ignore documentation contracts because a status line is stale;
- distinguish:
  - contract;
  - design decision;
  - historical experiment report;
  - current implementation state;
- reconcile important discrepancies explicitly.

Do not perform broad documentation cleanup during an unrelated experiment.
Fix only verified inconsistencies, preferably in a dedicated change unless the
documentation update is required by the code change.

---
## 14. Local-State Safety

This handoff file describes the repository state after resolving the Codex session handoff:

- Codex previously started M49.6 but reached its usage limit;
- partial M49.6 source edits were intentionally undone by the user;
- the working tree content matches clean M49.5 (`be190a5`);
- the 12 files shown as modified in `git status` are stat-cache / CRLF artifacts from the edit/undo cycle (their normalized hashes match HEAD);
- no uncommitted substantive M49.6 changes exist in the workspace.

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

1. **Review M50-C2 local infrastructure**:
   - EOS-preserving SFT encoding with ChatML loss masking (`-100`);
   - 20-case deterministic VAL probe extractor from `sft_val.json` (salted SHA-256) with strict holdout isolation;
   - Baseline cache generation and validation with SHA256 verification;
   - Candidate 2 pilot recipe ($r=4, \alpha=8, \text{LR}=10^{-5}$, target $\{q\_proj, v\_proj\}$, 921,600 trainable parameters);
   - Checkpoint safety and semantic preservation gates at steps [50, 100, 150];
   - Multi-checkpoint selection ranking;
   - Complete pilot archive packaging (`m50-c2-pilot-complete.zip`) with checksum manifest.
2. **Execute M50-C2 Pilot on Kaggle GPU**: Run 5-cell deterministic runbook per `docs/20-M50-C1-POSTMORTEM-AND-C2-PILOT.md`.
3. **Inspect Pilot Output**: Check `checkpoint-selection-report.json` and diagnostic metrics before considering any stage 2/3 promotion.

---

## 16. Status of Historical Questions

- **What happened in M50-C1?** Resolved: C1 trained cleanly to step 282 (val loss 1.09828) and showed positive ROUGE-L lexical gain (+0.04153), but collapsed in free-generation health (70% cap reached, 90% repetition loops). Rejected for production.
- **Why is Candidate 2 bounded to 150 steps?** Resolved: To test intermediate checkpoints at steps 50, 100, and 150, evaluate whether conservative rank ($r=4$) and lower LR ($10^{-5}$) prevent degeneration earlier in training.
- **Why is `screen_holdout.json` untouched?** Resolved: Strict holdout rule guarantees `screen_holdout.json` is never contaminated by intermediate probing or checkpoint selection.

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
    M49.6 production RAG pipeline

Current repository state:
    0.50.3 (M50-C2 conservative QLoRA pilot infrastructure implemented and locally verified)

Active development frontier:
    M50-C2 conservative QLoRA pilot execution on Kaggle GPU

Next action:
    review M50-C2 local infrastructure -> run Kaggle pilot (Cells 1-5) -> review checkpoint selection report
`````
