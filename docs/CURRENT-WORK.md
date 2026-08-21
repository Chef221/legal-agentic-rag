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
   - **Task B-V2-IMPLEMENTATION COMPLETE — V2-D1 IMPLEMENTED**:
     - Implemented `StructuredSemanticCitationVerifier` in `src/legal_agentic_rag/generation/structured_semantic_verifier.py`.
     - Multi-dimensional structured audit across 6 material legal dimensions (`ACTOR_ROLE`, `ACTION_OBJECT`, `CONDITION_EXCEPTION`, `QUANTITY_TEMPORAL`, `NEGATION_MODALITY`, `SOURCE_ARTICLE_SCOPE`) + `EVIDENCE_COVERAGE`.
     - Deterministic code derivation: `CONFLICT` -> `CONTRADICTED`, incomplete coverage / `INSUFFICIENT` -> `INSUFFICIENT`, all-match -> `SUPPORTED`.
     - Pinned exact Qwen2.5-3B-Instruct model identity (`a1d308dfcc03e09da285d49d912439a655a571e8`, `cuda`, `float16`, `temp=0.0`).
     - Development benchmark harness built in `scripts/evaluate_verification_v2_development.py` with fail-closed SHA verification across all 5 sources and baseline V1 comparison.
     - Preflight verification passed on canonical development sources (`V2_DEVELOPMENT_BENCHMARK_READY`).
     - 33 comprehensive unit tests passing across verifier and harness.
     - Production isolation preserved: `RuleBasedCitationVerifier` active in production; semantic verifier disabled. Fresh holdout remains strictly sealed.
   - **Task B-V2-D3.1 BENCHMARK COMPLETE — KEEP_D3**:
     - Executed canonical development benchmark for V2-D3.1 on Kaggle (`1383bf379a01c3f7456e3c41ba3be42846ceee2c`).
     - Evidence archived in `verification-v2-d31-development-evidence.zip` (SHA-256 `e14f9656a13a04b8e545d88a5dca13653fa317166ff530f45e4b13124f864041`).
     - Verdict: `V2_D31_DEVELOPMENT_BENCHMARK_PASS`, Decision: `KEEP_D3`, `promotion_authorized = false`.
     - Behavioral Finding: D3.1 operated as a high-recall / low-precision contradiction detector (caught 6/7 contradictions, but emitted 14 false contradiction positives, damaging supported retention from 17/18 down to 12/18 and regressing 6/7 D3 fixes).
   - **Task B-V2-D3.2 BENCHMARK COMPLETE & FORMALLY CLOSED — KEEP_D3**:
     - Executed canonical development benchmark for V2-D3.2 on Kaggle (`e5db78f0796c53e973fc63f9dd98df6c95f43f6e`).
     - Evidence archived in `verification-v2-d32-development-evidence.zip` (SHA-256 `bf44b9d77172d4f1823b62c02abae9e462bfbb9fdc5c650ba87e192e4928878f`, size `28,738` bytes, 13 members).
     - Verdict: `V2_D32_DEVELOPMENT_BENCHMARK_PASS`, Decision: `KEEP_D3`, `d32_supersedes_d3 = false`, `promotion_authorized = false`.
     - Mechanical reconciliation: 152 calls (76 D3 base + 76 strict-conflict), 0 errors, 0 retries, 38/38 stable claims, base drift: false (Pass 1: 38/38, Pass 2: 38/38).
     - Findings: Strict conflict overlay produced 0 overrides, preserving 100% of D3 gains (7/7) with 0 false overrides, but catching 0 contradictions. Net delta vs D3 = 0.
     - Formal decision: `KEEP_D3`. V2-D3.2 closed.
   - **Task B-V2-D3-FREEZE & PRE-H-LABEL INTEGRITY HARDENING COMPLETE**:
     - V2-D3 officially frozen as selected champion V2 development candidate (`Qwen/Qwen2.5-3B-Instruct` rev `a1d308df...`, Impl SHA `a6e8bca1...`, System Prompt SHA `546cd8bd...`, Schema SHA `3591144a...`).
     - Development benchmark (38 claims) is permanently CLOSED for candidate tuning. There is NO D3.3.
     - Pre-H-LABEL integrity hardening completed (D124):
       - Duplicate JSON key hook and fail-closed duplicate review item detection in `freeze_verification_v2_holdout_labels.py`;
       - 2-state governance lifecycle (`FROZEN_PENDING_EXTERNAL_REVIEW` -> `EXTERNALLY_REVIEWED_FOR_H_EXEC`);
       - Mandatory label commitment enforcement in canonical H-EXEC (blocking raw `--holdout-labels-sha256`);
       - Top-level label metadata and per-claim `claim_text_sha256` matching against review packets;
       - Exact prediction set equality ($\text{ExpectedKeys} \equiv \text{Pass1Keys} \equiv \text{Pass2Keys}$) and fail-closed stability;
       - Content-safe exception telemetry (`error_type`, `error_sha256`, `error_message_length`) eliminating raw exception strings and secret leakage;
       - Provider call reconciliation gate ($\text{calls} == 2 \times N_{\text{claims}} + \text{retries}$) with exact system instruction SHA;
       - Hardened canonical provenance validation in `_validate_canonical_provenance()`;
       - Cell H1 pinned runtime package assertions and Cell H6 independent evidence recomputation & verification assertions;
     - Holdout evaluation protocol pre-registered in `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md` with immutable rate gates (`supp_ret >= 0.88`, `neg_catch >= 0.50`, `val_ans_ret >= 0.80`, `full_ans_acc >= 0.60`, `claim_bin_acc >= 0.70`).
     - 38 comprehensive unit tests passing across holdout evaluation and label freeze test suites (`test_verification_v2_d3_holdout.py`, `test_freeze_verification_v2_holdout_labels.py`).
   - **Task B-H-LABEL & EXTERNAL H-EXEC AUTHORIZATION COMPLETE**:
     - Completed neutral human forensic review across all 31 claims of the 16 primary holdout packets (`verification-v2-holdout-review-packets-v1.zip` SHA-256 `a7a59175...`, `verification-v2-holdout-selection-v1.json` SHA-256 `08c480f6...`) with zero model predictions.
     - Authoritative human gold labels frozen into immutable artifact `verification-v2-holdout-reviewed-labels-v1.json` (SHA-256 `85d348dbb7da1567398836b96156a9d08fcfe181b676c5ecd593535ec8904215`, size `9,383` bytes, 31 claims: 24 SUPPORTED, 1 CONTRADICTED, 6 INSUFFICIENT).
     - Historical pending commitment recorded: `verification-v2-d3-holdout-label-commitment.json` (SHA-256 `c7755e37e394e80484f73c52ee6965c34c65917c38fa83b1dc453bbb466bcf86`, size `823` bytes, status `FROZEN_PENDING_EXTERNAL_REVIEW`).
     - External chain-of-custody review passed (D125, decision: `APPROVED`).
     - Approved commitment tracked in repository at `configs/verification-v2-d3-holdout-label-commitment.json` (SHA-256 `5cc7f58ed52c43d091bce98d5296ab68cded981495736a479644aefc1428b6dc`, size `1,060` bytes, `reviewer_governance_status: "EXTERNALLY_REVIEWED_FOR_H_EXEC"`).
     - Candidate V2-D3 and promotion rate gates remain strictly frozen.
   - **Task B-H-EXEC ATTEMPT 0 INVALIDATION & PRE-INFERENCE HARNESS REPAIR (D126, M51.12)**:
     - Canonical H-EXEC Attempt 0 on commit `21b7ffcf10d4621b0fdcbf18dcd565e4d5186699` failed synchronously during provider initialization (`TypeError: TransformersChatProvider.__init__() got an unexpected keyword argument 'model_name'`) before provider instantiation, model weight loading, or Pass 1 start.
     - Classified as `H_EXEC_ATTEMPT_0_INVALID_PRE_INFERENCE_HARNESS_FAILURE` (0 provider calls, 0 D3 predictions, 0 holdout metrics; zero holdout scientific results consumed).
     - Repaired `scripts/evaluate_verification_v2_d3_holdout.py` by porting the proven provider construction architecture from `scripts/evaluate_verification_v2_d3_development.py` via `SemanticVerificationConfig.as_generation_config()`, with fail-closed GenerationConfig invariant assertions.
     - Added model-free provider-constructor smoke verification to the `--preflight-only` gate (recording `provider_constructor_contract_verified: True`) and added unit regression tests with monkeypatched `_load_runtime` guards.
     - Candidate V2-D3 implementation (`a6e8bca1...`), prompt (`546cd8bd...`), schema (`3591144a...`), frozen labels (`85d348db...`), tracked commitment (`5cc7f58ed5...`), and promotion rate gates remain strictly frozen and byte-identical.
     - Exactly ONE recovery execution authorized as **H-EXEC Recovery Attempt 1** on a fresh Kaggle GPU session following external review of the corrected execution-authority commit.
   - **Task B-H-EXEC CLOSURE & POSTMORTEM FAILURE ANALYSIS (D127, M51.13)**:
     - Executed canonical one-shot Phase H-EXEC evaluation on Kaggle GPU on reviewed authority commit `77561aa7c4b242e12d011a84a21f3a262a17a0f8`. Evidence archived in `verification-v2-d3-holdout-evidence.zip` (SHA-256 `9e2b38d4189f9c68901051a07b999845c660ec6ab4b4fa1e6ec69d3088fe6a5d`, size `10,463` bytes, 9 members).
     - Final verdict: `V2_D3_HOLDOUT_EXECUTION_FAILURE`, decision: `REJECT_V2_D3_PROMOTION`, `promotion_recommended: false`, `promotion_authorized: false`.
     - Scientific rate gates: Supported Retention Rate = 22/23 = 95.65% (PASS), Negative Catch Rate = 2/7 = 28.57% (**FAILED** vs 50.0% minimum), Valid Answer Retention = 8/10 = 80.00% (PASS), Full Answer Accuracy = 10/16 = 62.50% (PASS), Claim Binary Accuracy = 24/30 = 80.00% (PASS).
     - Operational telemetry: 61/62 provider calls succeeded, 0 retries, 1 call failure (`103383:PRIMARY:C1`) on Call 1 due to transient cold-start runtime loading (54.55s), 30/30 stable claims. Operational error isolated from semantic failure (eliminating runtime error does not fix failed negative catch).
     - Candidate V2-D3 development track is **PERMANENTLY CLOSED**. Production semantic verifier remains **DISABLED**. NO D3.3. Holdout claims (31) are **BURNED** and converted to diagnostic development data only.
     - Forensic failure analysis completed in `docs/32-V2-D3-HOLDOUT-CLOSURE-AND-POSTMORTEM.md`: classified 5 False Accepts (3x `ACTOR_ROLE_MISMATCH`, 1x `CONDITION_EXCEPTION_OMITTED`, 1x `ACTION_OBJECT_MISMATCH`/`QUANTITY_TEMPORAL_MISMATCH`), 1 False Reject (`SYNTAX_FRAGMENT_STRICTNESS`), and 2 True Negatives.
     - Root-cause ranking: #1 Lack of explicit legal dimension decomposition, #2 Lexical entailment bias, #3 Context-blind fragment evaluation.
     - Recommended V3 architecture: Option C (Structured Dimension Decomposition with 3 boolean predicates and deterministic aggregation).
     - Strategic ROI recommendation: Shift primary engineering focus to Generation Grounding & Prompt Optimization (Task 2 metric leverage) and Retrieval / Reranking depth tuning.
   - **Task GENERATION G1 — MATERIAL-FIDELITY GROUNDING (D128, M52.1)**:
     - Implemented candidate grounding profile `material_fidelity_v1` in `ModelBackedAnswerGenerator` (`src/legal_agentic_rag/generation/model_generator.py`) alongside default `baseline`.
     - Added Vietnamese prompt instructions enforcing: Actor/Role preservation, Action/Object preservation, Conditions/Exceptions preservation, Legal Scope preservation, Numeric/Temporal verbatim copying, Full Material Coverage, and List/Noun-phrase acceptance.
     - Preserved strict invariants: `grounding_profile="baseline"` remains production default, `ModelAnswerDraft` schema strictly preserved, call count parity preserved (1 provider call).
     - Created `scripts/evaluate_generation_grounding_g1.py` executing Baseline vs G1 A/B experiment across the 16 burned diagnostic review packets with content-safe telemetry and blinded pairwise worksheet generation (`results/generation_g1_human_review_worksheet.md`).
     - Pre-registered development success criteria (Criteria A–F: 0 errors, $\ge 4/5$ known material errors fixed, $\ge 9/10$ valid answers preserved, $\Delta_{\text{abstain}} \le 15\%$, 0 schema regressions, call parity).
     - Published specification in `docs/33-GENERATION-GROUNDING-G1-DEVELOPMENT.md`.

---

## 16. Status of Historical Questions

- **What happened in Generation Grounding G1 Development?** Resolved: Implemented candidate `material_fidelity_v1` in `ModelBackedAnswerGenerator`, added `GenerationConfig.grounding_profile`, created A/B evaluation harness `scripts/evaluate_generation_grounding_g1.py` for burned 16-question diagnostic review set, implemented deterministic pairwise blinding (`results/generation_g1_human_review_worksheet.md`), pre-registered success criteria (Criteria A–F), added 20 unit tests across generator and evaluator, and documented specification in `docs/33-GENERATION-GROUNDING-G1-DEVELOPMENT.md` (D128, M52.1).
- **What happened in Phase H-EXEC Holdout Evaluation and Closure?** Resolved: Executed canonical one-shot Phase H-EXEC evaluation on Kaggle GPU on commit `77561aa7c4b242e12d011a84a21f3a262a17a0f8` (`verification-v2-d3-holdout-evidence.zip` SHA-256 `9e2b38d4189f9c68901051a07b999845c660ec6ab4b4fa1e6ec69d3088fe6a5d`). Final verdict: `V2_D3_HOLDOUT_EXECUTION_FAILURE`, decision: `REJECT_V2_D3_PROMOTION`. Supported retention passed at 95.65% (22/23), valid answer retention passed at 80.0% (8/10), full answer accuracy passed at 62.5% (10/16), claim binary accuracy passed at 80.0% (24/30). Negative catch rate failed significantly at 28.57% (2/7 vs 50.0% gate). 61/62 provider calls succeeded, with 1 cold-start failure on Call 1 (`103383:PRIMARY:C1`). V2-D3 permanently closed, production verifier remains disabled, NO D3.3, 31 holdout claims burned as diagnostic data. Postmortem failure analysis published in `docs/32-V2-D3-HOLDOUT-CLOSURE-AND-POSTMORTEM.md`. Recommended next architecture: Option C (Structured Dimension Decomposition). (D127, M51.13).
- **What happened in Phase H-EXEC Attempt 0 and Harness Correction?** Resolved: Canonical H-EXEC Attempt 0 on commit `21b7ffcf10d4621b0fdcbf18dcd565e4d5186699` encountered a mechanical pre-inference harness defect during provider initialization (`TypeError: TransformersChatProvider.__init__() got an unexpected keyword argument 'model_name'`). Synchronous failure occurred before provider construction, weight loading, or Pass 1 start (0 provider calls, 0 D3 predictions, 0 holdout metrics produced). Formally classified as `H_EXEC_ATTEMPT_0_INVALID_PRE_INFERENCE_HARNESS_FAILURE` with zero scientific holdout consumption. Repaired `scripts/evaluate_verification_v2_d3_holdout.py` to construct `TransformersChatProvider` via `SemanticVerificationConfig.as_generation_config()`. Added model-free constructor smoke verification to `--preflight-only` and unit regression tests with runtime loading guards. Candidate V2-D3, prompt, schema, frozen labels, commitment, and rate gates remain 100% immutable. Authorized exactly ONE recovery run as **H-EXEC Recovery Attempt 1** on a fresh Kaggle session (D126, M51.12).
- **What happened in Phase H-LABEL and H-EXEC Authorization?** Resolved: Phase H-LABEL human review completed across all 16 primary review packets (31 claims: 24 SUPPORTED, 1 CONTRADICTED, 6 INSUFFICIENT) with zero model predictions. Labels frozen into immutable artifact `verification-v2-holdout-reviewed-labels-v1.json` (SHA-256 `85d348dbb7da1567398836b96156a9d08fcfe181b676c5ecd593535ec8904215`). Pending commitment `c7755e37...` externally reviewed and approved. Approved commitment tracked in repository at `configs/verification-v2-d3-holdout-label-commitment.json` (SHA-256 `5cc7f58ed52c43d091bce98d5296ab68cded981495736a479644aefc1428b6dc`). Goverance status set to `EXTERNALLY_REVIEWED_FOR_H_EXEC`. Candidate V2-D3 and rate gates remain frozen.
- **What happened in Pre-H-LABEL Integrity Hardening?** Resolved: Completed final pre-holdout hardening pass (D124). Added duplicate-key-aware JSON loader and fail-closed duplicate review checks to `scripts/freeze_verification_v2_holdout_labels.py`. Changed commitment status initialization to `FROZEN_PENDING_EXTERNAL_REVIEW`. Required `--label-commitment` with `EXTERNALLY_REVIEWED_FOR_H_EXEC` for canonical H-EXEC. Validated label artifact metadata and per-claim `claim_text_sha256`. Enforced exact prediction-set equality and eliminated fail-open stability bugs. Replaced raw exception strings with content-safe telemetry (`error_type`, `error_sha256`, `error_message_length`). Added provider call reconciliation gate. Hardened `_validate_canonical_provenance()` with fail-closed checks on all execution parameters. Updated Runbook Cell H1 with runtime package assertions and Cell H6 with independent evidence recomputation assertions.
- **What happened in V2-D3 Holdout Governance Hardening?** Resolved: Hardened holdout evaluation governance into two distinct irreversible phases (Phase H-LABEL for human gold freezing and Phase H-EXEC for Kaggle execution). Created `scripts/freeze_verification_v2_holdout_labels.py` and hardened `scripts/evaluate_verification_v2_d3_holdout.py` with exact packet-label claim set equality, non-vacuous coverage denominator gates, zero-denominator None semantics, pinned Kaggle environment, and single model loading. Tested exclusively on synthetic fixtures with zero real holdout exposure.
- **What happened in V2-D3.2 Execution?** Resolved: V2-D3.2 completed canonical Kaggle execution (`e5db78f0796c53e973fc63f9dd98df6c95f43f6e`, evidence `verification-v2-d32-development-evidence.zip` SHA-256 `bf44b9d77172d4f1823b62c02abae9e462bfbb9fdc5c650ba87e192e4928878f`). Reconciled 152 calls, 0 errors, 0 retries, 38/38 stable claims, 0 base drift. Overcall was 0 (0 false overrides, 7/7 D3 gains preserved), but caught 0 contradictions (net delta 0 vs D3). Formal decision: `KEEP_D3`. V2-D3.2 formally closed.
- **What is the status of V2 Candidate Selection?** Resolved: V2-D3 is officially selected and frozen as the exclusive V2 candidate. Development iterations are permanently closed. There is NO D3.3.
- **What is the status of the Fresh Holdout Protocol?** Resolved: Pre-registered in `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md` and `scripts/evaluate_verification_v2_d3_holdout.py` with pre-registered rate gates and two-pass stability requirements. Fresh holdout data remains strictly sealed and unreviewed.
- **What happened in V2-D3.1 Execution?** Resolved: V2-D3.1 passed mechanical benchmarks (`V2_D31_DEVELOPMENT_BENCHMARK_PASS`), but caused severe regressions on supported retention (12/18 vs D3's 17/18) and regressed 6/7 D3 gains due to contradiction overcalling (precision 30%, recall 85.71%). Formal decision: `KEEP_D3`. Evidence archived in `verification-v2-d31-development-evidence.zip` (SHA-256 `e14f9656a13a04b8e545d88a5dca13653fa317166ff530f45e4b13124f864041`).
- **What happened in Task B-V2-IMPLEMENTATION?** Resolved: Implemented experimental candidate `StructuredSemanticCitationVerifier` (multi-dimensional audit + deterministic derivation) and offline development benchmark harness `scripts/evaluate_verification_v2_development.py`. Preflight validation passed on canonical sources (`V2_DEVELOPMENT_BENCHMARK_READY`). 33 unit tests pass. Status: `V2-D1 IMPLEMENTED — REAL DEVELOPMENT MODEL EXECUTION PENDING EXTERNAL REVIEW`.
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
    0.50.7 (G1 Material-Fidelity Grounding Implemented; Production Generator Default Remains Baseline; D128 Accepted; M52.1 Completed; Blinded A/B Development Harness Ready)

Active development frontier:
    Generation Grounding G1 Development A/B Evaluation / Diagnostic Verification

Next action:
    External review of G1 candidate implementation before executing canonical one-run development A/B evaluation on Kaggle GPU.
```
