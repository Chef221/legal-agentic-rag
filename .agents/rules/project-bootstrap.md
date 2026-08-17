# Legal Agentic RAG — Mandatory Project Bootstrap

You are a coding agent working inside the `legal-agentic-rag` repository.

This rule is project-wide. Treat it as mandatory before architecture analysis,
implementation, configuration changes, experiments, dependency changes, test
changes, or persistent documentation changes.

## 1. Mandatory Context Bootstrap

Before substantive work, read and obey:

- @../../AGENTS.md
- @../../docs/00-START-HERE.md
- @../../docs/CURRENT-WORK.md

Do not treat the current chat as the sole source of project truth.

If the user asks a narrow question that can be answered without repository
modification, inspect only the additional context necessary. If the task can
change code, configuration, tests, artifacts, experiment behavior, or
documentation, complete the repository bootstrap first.

## 2. Source-of-Truth Hierarchy

Use this hierarchy:

1. `AGENTS.md`
2. authoritative documentation in `docs/`
3. approved source code and tests
4. explicitly confirmed new user decisions
5. short-lived context in `docs/CURRENT-WORK.md`
6. current conversation

`CURRENT-WORK.md` is working memory, not a replacement for architecture,
competition, schema, or design-decision documents.

If sources disagree:

- identify the discrepancy;
- distinguish stale status text from a true contract conflict;
- inspect implementation/tests/history;
- do not silently choose the interpretation that is easiest to implement;
- ask the user only when the repository cannot resolve a decision that
  materially changes the task.

## 3. Always Inspect Local Git State Before Editing

Before the first edit in a new task/session, inspect:

```bash
git status --short
git branch --show-current
git log --oneline -20
```

When uncommitted changes exist, inspect the relevant diff.

Do not:

- overwrite unknown local changes;
- reset, clean, stash, revert, or discard user/previous-agent work without
  explicit approval;
- assume remote `main` equals the local working tree;
- bundle unrelated local work into the current task.

## 4. Reconstruct Before Redesigning

Before modifying a subsystem:

1. read the relevant documentation referenced by `docs/00-START-HERE.md`;
2. inspect the actual implementation;
3. inspect the relevant tests;
4. inspect recent commits affecting the subsystem;
5. identify the current behavior and intended contract;
6. identify the hypothesis behind the requested change;
7. identify the smallest change that can test that hypothesis;
8. define validation before editing.

Never redesign an existing subsystem from first principles without first
understanding why the current design exists.

Do not replace documented architecture with a fashionable alternative merely
because it appears cleaner.

## 5. Competition Safety Is a Hard Boundary

Preserve all active competition rules and repository compliance constraints.

In particular, unless the user explicitly approves a documented policy change:

- keep the competition pipeline official-data-only / `competition_only`;
- do not introduce external corpus data;
- do not create/use synthetic competition QA, answers, evidence, negatives, or
  training examples;
- do not add model APIs to the competition system;
- keep raw competition schemas behind adapters;
- preserve approved model/compliance constraints;
- preserve scorer/submission contracts;
- preserve artifact lineage, checksums, reproducibility metadata, and
  fail-closed validation;
- do not reuse stale indexes/vectors after lineage-defining inputs change.

Before competition-facing changes, read the relevant compliance, data-contract,
and scoring-contract documentation.

## 6. Baseline and Experiment Discipline

The current repository may contain code newer than the stable comparison
baseline.

Do not confuse:

- stable quality-control baseline;
- current package version;
- active experimental milestone;
- locally enabled candidate configuration.

At the current handoff, `docs/CURRENT-WORK.md` records the exact distinction.

An implemented feature is not automatically a proven improvement.

Examples of components that may exist without baseline promotion include
reranking, graph retrieval, recovery logic, verifier variants, or other
candidate paths.

Do not enable or promote them based on architectural preference alone.

Every promotion must be supported by the repository's evaluation protocol.

## 7. One Hypothesis per Change

Prefer a narrow, interpretable experiment.

Before editing, be able to state:

- the failure mode;
- the hypothesis;
- the affected boundary;
- the expected measurable effect;
- the validation gate;
- the regression risks.

Avoid combining unrelated changes such as retrieval, context selection,
generation, verification, and scoring behavior into a single experiment.

If multiple changes are necessary for correctness, explain why they form one
atomic hypothesis.

## 8. Preserve Default-Off Experimental Behavior

Experimental features that are documented as candidate-only or default-off must
remain default-off unless explicit promotion has been approved.

Do not silently change baseline/default profiles to make an experiment easier to
run.

Do not increase retries, loops, retrieval breadth, model calls, or fallback
behavior beyond documented bounds without an explicit, measured experiment.

Bounded/fail-closed behavior is intentional.

## 9. Generator and Verifier Safety

When working on structured generation, claim salvage, citation verification, or
schema recovery:

- preserve exact legal claim text unless the experiment explicitly and safely
  authorizes content changes;
- do not invent evidence identities;
- do not alter numbers/negation/conditions to force verification;
- do not infer missing semantic values during a structural recovery;
- do not leak raw rejected model completions or sensitive legal text into
  telemetry when the existing boundary intentionally stores closed codes only;
- recovered/salvaged answers must pass the normal verification path;
- verification failure must remain visible/fail-closed according to the
  documented policy.

For M49.x work, read the milestone/design-decision documentation before touching
generation or verifier code.

## 10. Retrieval and Artifact Safety

When modifying parsing, chunking, embeddings, indexing, retrieval, fusion,
reranking, graph retrieval, or context selection:

- preserve artifact identity/lineage rules;
- never combine incomparable raw scores casually;
- do not reuse vector/index artifacts whose upstream identity changed;
- do not enable reranker/graph behavior merely because implementations exist;
- validate changes on the documented leakage-safe evaluation path;
- measure downstream answer impact when a retrieval change is intended for
  end-to-end promotion.

## 11. Tests Before Broad Experiments

Use the narrowest useful validation first.

Typical order:

1. unit tests for the changed logic;
2. targeted integration/regression tests;
3. milestone-specific targeted gate;
4. immutable/fixed smoke set;
5. larger evaluation/batch only when the documented earlier gates pass.

Do not start expensive GPU/large-batch work when a smaller required gate has not
been reviewed.

Do not automatically continue to the next experiment stage merely because a
command completed successfully.

A completed run is not an accepted result until its report has been reviewed.

## 12. Baseline and Frontier Guardrail

The frozen validated reliability baseline is **M49.6** (commit
`9b0cd0b1d40fb01bb62d4841f7728af2264f3957`, version `0.49.6`), which completed
both targeted and immutable 50-smoke gates with zero model errors.

The active experimental frontier is **M50 — Official-Data LegalQA Generator
Fine-Tuning**.

Candidate 1 (`M50-C1`) infrastructure is implemented locally; GPU training and
direct-QA semantic screening on `screen_holdout.json` are pending. `M50-C1` has
**not** yet been trained or proven successful.

Do not start the 991-question historical development benchmark until Candidate 1
passes both direct screening and the immutable 50-smoke.


## 13. Documentation Responsibilities

Persistent changes to:

- architecture;
- schemas/contracts;
- competition behavior;
- artifact lineage;
- pipeline semantics;
- default profiles;
- experiment acceptance protocol;
- long-lived design decisions

must update the appropriate authoritative documentation.

Do not use `CURRENT-WORK.md` as the only record of a permanent architectural
decision.

Conversely, do not perform broad documentation cleanup during an unrelated
code task merely because stale wording exists. Report verified discrepancies
and keep scope controlled.

## 14. Do Not Invent Repository Facts

When uncertain:

- inspect the file;
- inspect the test;
- inspect recent git history;
- inspect config/profile definitions;
- inspect generated report metadata if available.

Never fabricate:

- experiment results;
- GPU run outcomes;
- leaderboard scores;
- model approval state;
- artifact hashes;
- local file presence;
- command success;
- test success;
- the reason for an undocumented change.

State uncertainty explicitly when evidence is absent.

## 15. Required Pre-Edit Report

Before substantive edits, provide a concise task reconstruction containing:

- task objective;
- relevant repository contracts/docs;
- current implementation path;
- relevant tests;
- related recent history;
- hypothesis;
- proposed minimal change;
- validation plan;
- important invariants/risks.

For a trivial local correctness fix, this can be short. For architecture,
competition, experiment, or model behavior changes, it must be explicit.

Do not spend the entire response theorizing if the repository already provides a
clear implementation path. Inspect first, then act.

## 16. Required Post-Edit Report

After implementation, report:

- files changed;
- behavior changed;
- behavior intentionally unchanged;
- tests/commands actually run;
- exact pass/fail status;
- experiment reports produced, if any;
- regressions;
- remaining uncertainty;
- documentation updated;
- recommended next action.

Never claim a test, benchmark, or GPU run was performed if it was not actually
performed.

## 17. Scope and Safety Defaults

Unless the user asks otherwise:

- do not modify unrelated files;
- do not introduce new dependencies for convenience;
- do not rewrite working modules merely for style;
- do not remove compatibility behavior without evidence it is obsolete;
- do not broaden public interfaces unnecessarily;
- do not weaken validation to make tests pass;
- do not disable tests/gates to promote an experiment;
- do not commit secrets, datasets, model weights, large generated artifacts, or
  local paths prohibited by repository policy;
- do not perform destructive git operations on unknown work.

## 18. Context Maintenance

At the end of a meaningful milestone or experiment result, determine whether
`docs/CURRENT-WORK.md` needs an update.

Update it when the current frontier, next action, blocker, accepted/rejected
hypothesis, or handoff-critical local state changes.

Keep it concise enough that a new agent can reconstruct the current frontier
without reading old chat transcripts.

The repository—not one model's conversation history—must remain the durable
project memory.
