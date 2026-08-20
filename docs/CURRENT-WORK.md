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
4. **Stage R1 Candidate-Pool / Reranker Mechanics Audit CLOSED — PASS**:
   - Reviewed Kaggle execution commit: `9a5b708c2769425dbd65731feb8ede96975b5b46`.
   - Canonical evidence archive: `candidate-pool-reranker-audit-evidence.zip` (SHA-256 `ce9b239b808c3d7b0e575ce1c1683db243bbea909f0e6d9c306df21cb2899860`, size `56,706 bytes`).
   - Mechanical verdict: **`CANDIDATE_POOL_AUDIT_PASS`** (`audit_verified = true`, `h40_promotion_authorized = false`).
   - Confirmed 22/22 seed prefix passes, 22/22 shared S20 sequence passes, 22/22 legacy S20 frozen score passes ($\le 10^{-6}$), 22/22 H40 frozen score passes ($\le 10^{-6}$), 22/22 branch depth passes, 0 retrieval model errors.
   - Characterized candidate-pool mechanics: 5 identical top-8 cases, 17 changed top-8 cases, 35 total tail entrants across 17 cases, 20 document-level churn events.
   - Shared S20 maximum numerical delta of `1.621e-05` confirmed as batch-shape inference artifact; observational legacy S20 probe reproduced all 22 cases within $10^{-6}$.
   - Runtime config evidence self-identity verified matching on disk and inside ZIP.
   - Invariant: H40 remains unpromoted. Production routing unchanged (S20 in Attempt 1, H40 in Attempt 2).
5. **Active Research Frontier — Priority B (Verification-Correctness Audit) — BENCHMARK COMPLETED & CLOSED**:
   - **Task B-FORENSIC-0 COMPLETE — FORENSIC_SOURCE_READY**: Four-question paired forensic source packets (`102047`, `147239`, `26541`, `95861`) materialized from frozen B1A historical evidence (`phase-b1a-graph-routing-ablation-evidence.zip` SHA-256 `b88ccce928b4cecfc9239d490c59f91405834c5a3199f917d757c5735b1d6631`). Verified 100% chunk lookup pass from canonical `uit-dsc-2026-task2-v0400/legal_chunks` (330,768 records), 100% citation cross-check pass, and 100% (6/6) verifier replay pass across all applicable arms. See [`docs/27-VERIFICATION-CORRECTNESS-FORENSIC-AUDIT.md`](file:///c:/legal-agentic-rag/docs/27-VERIFICATION-CORRECTNESS-FORENSIC-AUDIT.md).
   - **Task B-FORENSIC-1A COMPLETE — HUMAN_FORENSIC_LABELS_FROZEN**: Human-approved claim entailment and granular error mode labels frozen into immutable overlay artifact `verification-human-forensic-labels-v1.json` (SHA-256 `bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733`) and `verification-human-forensic-labels-v1.zip` (SHA-256 `25c23b80fb94a59976ccd821944355ff80aa60c7360e32a3dd8dea19dae12cbb`). Verified 4 questions, 8 historical arms, 11 labeled claims (2 SUPPORTED, 5 CONTRADICTED, 4 INSUFFICIENT), 2 generation-failed unlabeled arms. Scope is strictly internal evaluation-only.
   - **Task B-FORENSIC-1B COMPLETE — POSITIVE_CONTROL_SOURCE_READY**: 16 PRIMARY (4 per stratum) and 8 RESERVE (2 per stratum) positive-control candidates deterministically sampled from 788 eligible Phase-A `answer_verified` records across 4 pre-registered strata (`D_NEGATION_MODALITY`, `C_NUMERIC`, `B_MULTI_CLAIM_CLEAN`, `A_SINGLE_CLAIM_CLEAN`). All 16 PRIMARY evidence packets materialized with 100% chunk lookups, 100% selection trace cross-checks, and 100% (16/16) rule verifier replay matches. Human labels strictly initialized as `unreviewed` with null values. Zero model/retrieval/generation executions. Packaged review archive: `verification-positive-control-review-packets-v1.zip` (SHA-256 `cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4`, size 110,095 bytes). See [`docs/27-VERIFICATION-CORRECTNESS-FORENSIC-AUDIT.md`](file:///c:/legal-agentic-rag/docs/27-VERIFICATION-CORRECTNESS-FORENSIC-AUDIT.md).
   - **Task B-FORENSIC-1C COMPLETE — POSITIVE_CONTROL_HUMAN_LABELS_FROZEN**: Human-approved positive-control claim entailment and error mode labels frozen into immutable content-bound overlay artifact `verification-positive-control-human-labels-v1.json` (SHA-256 `60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373`) and `verification-positive-control-human-labels-v1.zip` (SHA-256 `bf8efc191c74786af76b1580247d02989254fa2ee65063bfaccaea0035a4ecf2`). Verified 16 questions, 16 historical arms, 27 labeled claims (16 SUPPORTED, 2 CONTRADICTED, 9 INSUFFICIENT). Combined benchmark established at 38 claims (18 SUPPORTED, 7 CONTRADICTED, 13 INSUFFICIENT). Scope is strictly internal evaluation-only. See [`docs/27-VERIFICATION-CORRECTNESS-FORENSIC-AUDIT.md`](file:///c:/legal-agentic-rag/docs/27-VERIFICATION-CORRECTNESS-FORENSIC-AUDIT.md).
   - **Task B-BENCHMARK-EXECUTION COMPLETE — VERIFIER_BENCHMARK_PASS (V1 NOT PROMOTED)**:
     - Canonical Kaggle execution commit: `d3aac626400cbe31ed0ed5ad109762fcb78d737d`.
     - Canonical evidence archive: `verification-semantic-benchmark-evidence.zip` (SHA-256 `bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6`, size `17,290` bytes, `8` members).
     - Execution health: 0 model errors, 2 structured output retries, 38/38 deterministic stable claims across 2 passes.
     - Mechanical execution verdict: `VERIFIER_BENCHMARK_PASS`.
     - Empirical outcome: V1 improved net claim correctness by $+5$ claims (60.53% vs V0 47.37%) and invalid answer catch from 0.0% to 46.67%, but failed on 65% of negative claims ($13/20$), regressed on 2/18 supported claims, and exhibited 0% catch on condition omissions/inversions, wrong articles, and quantity errors.
     - Formal decision: `V1_EXISTING_SEMANTIC_VERIFIER_NOT_PROMOTED`. `semantic_verifier_promotion_authorized = false` remains in effect.
     - 38-claim dataset lifecycle: Permanently burned as DEVELOPMENT DATA (`verification_benchmark_v1_role = "development_after_first_evaluation"`). Future V2 promotion strictly requires a fresh, uncompromised holdout.
     - Invariant: Positive-control reserve cases (`27503`, `31317`, `33177`, `85651`, `112105`, `112833`, `130283`, `137453`) cannot be repurposed as a secret promotion holdout.
   - **Task B-HOLDOUT-SEALED COMPLETE — V2_HOLDOUT_PRE_REGISTERED**:
     - Fresh, independent holdout pre-registered from frozen Phase-A census records (`phase-a-current-system-census-final-evidence.zip` SHA-256 `df05a401599c43a28e39136d72b225841b242d10a40dc5bc475b9be6ed86be8b`).
     - Applied 46-QID contamination exclusion set (SHA-256 `eefdd8967c39324bc7e88a8451ef8fb9241f765af1e68a0199db9ba33af01fda`) across 772 eligible records.
     - Deterministically sampled 16 PRIMARY (4/4/4/4) + 8 FRESH RESERVE (2/2/2/2) cases under pre-registered salt `verification-v2-holdout-gen-v1:` (`deterministic_sha256_stratified_v2`).
     - Materialized 16 PRIMARY packets with `review_status = "sealed_unreviewed"` and null claim labels.
     - Confirmed 16/16 chunk lookups, 16/16 trace mappings, 16/16 metadata cross-checks, and 16/16 exact V0 RuleBasedCitationVerifier replay matches.
     - Selection commitment artifact: `verification-v2-holdout-selection-v1.json` (SHA-256 `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b`, size `16,788` bytes).
     - Sealed review package: `verification-v2-holdout-review-packets-v1.zip` (SHA-256 `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4`, size `108,532` bytes).
     - Invariants: Holdout blindness strictly enforced (zero selected QIDs exposed in tracked docs or console summary); review packets remain sealed until candidate V2 system is frozen.
   - **Next Action in Priority B**: Develop V2 verifier hypotheses using ONLY the 38-claim development benchmark while the fresh holdout remains sealed.

---

## 16. Status of Historical Questions

- **What happened in Task B-HOLDOUT-SEALED?** Resolved: Pre-registered fresh holdout from frozen Phase-A census with 46-QID contamination exclusion set (772 eligible records). Selected 16 PRIMARY (4/4/4/4) + 8 FRESH RESERVE (2/2/2/2) under salt `verification-v2-holdout-gen-v1:`. All 16 primary packets materialized with 100% chunk lookups, 100% trace mapping, and 16/16 V0 verifier replay match. Review status: `sealed_unreviewed` (claim labels null). Artifacts: `verification-v2-holdout-selection-v1.json` (SHA-256 `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b`) and `verification-v2-holdout-review-packets-v1.zip` (SHA-256 `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4`). Verdict: `V2_HOLDOUT_PRE_REGISTERED`.
- **What happened in the V0 vs V1 Semantic Verifier Benchmark Execution?** Resolved: Executed controlled offline benchmark on Kaggle Tesla T4 (`d3aac626400cbe31ed0ed5ad109762fcb78d737d`). Evidence archived in `verification-semantic-benchmark-evidence.zip` (SHA-256 `bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6`). Mechanical verdict: `VERIFIER_BENCHMARK_PASS`. V1 improved net correctness by +5 claims and caught 7/15 invalid answers, but failed on 13/20 negative claims and regressed on 2 supported claims. Formal decision: `V1_EXISTING_SEMANTIC_VERIFIER_NOT_PROMOTED`. 38 claims burned as development data; fresh holdout required before future V2 promotion.
- **What happened in the Benchmark Harness Implementation?** Resolved: Built `scripts/evaluate_verification_semantic_benchmark.py` and `tests/unit/evaluation/test_verification_semantic_benchmark.py` covering fail-closed source verification, exact V0 replay on 22 historical arms, 2-pass deterministic stability evaluation, paired delta analysis, false accept/reject rates, and error tag diagnostics. Pre-flight verification passed cleanly on canonical external files. Verdict: `VERIFIER_BENCHMARK_READY`.
- **What happened in Task B-FORENSIC-1C?** Resolved: Human-approved positive-control labels frozen into immutable content-bound overlay artifact `verification-positive-control-human-labels-v1.json` (27 claims: 16 SUPPORTED, 2 CONTRADICTED, 9 INSUFFICIENT). Composite 38-claim benchmark established (18 SUPPORTED, 7 CONTRADICTED, 13 INSUFFICIENT). Verdict: `POSITIVE_CONTROL_HUMAN_LABELS_FROZEN`.
- **What happened in Task B-FORENSIC-1B?** Resolved: Deterministically sampled 16 PRIMARY and 8 RESERVE positive-control candidates from 788 eligible Phase-A `answer_verified` outputs across 4 strata with 100% chunk lookups, 100% trace mapping, and 16/16 rule verifier replay match. External review package `verification-positive-control-review-packets-v1.zip` created with status `unreviewed`. Verdict: `POSITIVE_CONTROL_SOURCE_READY`.
- **What happened in Task B-FORENSIC-1A?** Resolved: Human-approved forensic labels frozen into immutable content-bound overlay artifact `verification-human-forensic-labels-v1.json` (11 claims: 2 SUPPORTED, 5 CONTRADICTED, 4 INSUFFICIENT, 2 generation-failed unlabeled arms). Verdict: `HUMAN_FORENSIC_LABELS_FROZEN`.
- **What happened in Task B-FORENSIC-0?** Resolved: Materialized 4 paired forensic packets (`102047`, `147239`, `26541`, `95861`) across 8 historical arms from canonical B1A frozen evidence with 0 retrieval reruns, 0 generation reruns, 0 auto-labeled semantic truth, and 100% rule verifier replay match. Verdict: `FORENSIC_SOURCE_READY`.
- **What happened in Stage R1?** Resolved: Stage R1 candidate-pool audit passed cleanly with `CANDIDATE_POOL_AUDIT_PASS`. Proved candidate-pool depth is behaviorally material (35 tail entrants, 20 document-level churn events across 17/22 relationship cases). H40 remains unpromoted in Attempt 2.
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
    0.50.7 (Fresh V2 Holdout Pre-Registered & Sealed — V2_HOLDOUT_PRE_REGISTERED; 38 Claims = Development Data; Model-backed Verifier Disabled)

Active development frontier:
    Priority B — Verification-correctness audit (V2 Verifier Hypothesis Development)

Next action:
    Develop V2 verifier hypotheses using ONLY the 38-claim development benchmark while the fresh holdout remains sealed
```
