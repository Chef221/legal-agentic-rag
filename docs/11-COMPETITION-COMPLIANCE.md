# Competition compliance and reproducibility

## 1. Source-of-truth policy

This document summarizes organizer rules already captured by the project. It does
not replace the organizer's original rules, email, Form or registration evidence.
Whenever BTC publishes an update, archive the exact source and review this file,
`AGENTS.md`, the data contract and scoring contract before changing the pipeline.

## 2. Data rules

- Use only official BTC data.
- Do not manually label or import external legal/QA data.
- Do not create synthetic QA, answers, evidence, hard negatives or training
  examples, even from official data.
- `warmup.json` and `train.json` provide answer-level supervision, not retrieval
  relevance labels.
- Public answers cannot be used for training or question-ID-specific rules.
- Every raw input goes through the UIT DSC adapter.
- Every persisted artifact records dataset identity, revision, config hash and code
  version.
- Official data, artifacts and submissions are not committed to Git.

## 3. Model and API rules

- No model API or intermediate AI product may be used.
- Models must be downloaded, run and controlled directly by the team.
- Exact model name, URL, immutable revision, license, role and parameter count must
  be recorded.
- Registration/approval evidence must be retained outside Git when it contains
  private team information.
- A fine-tuned revision may require a registration update; do not infer approval
  from approval of its base model.
- Changing the embedding model/revision requires re-embedding and rebuilding the
  vector index.

### Current retained inventory

| Role | Exact model/revision | Approx. parameters | Registration evidence |
|---|---|---:|---|
| Embedding | `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 0.6B | must be verified against team/BTC records |
| Reranker | `Qwen/Qwen3-Reranker-0.6B@e61197ed45024b0ed8a2d74b80b4d909f1255473` | 0.6B | must be verified against team/BTC records |
| Generator base | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` | 2.266B | must be verified against team/BTC records |
| M49 merged generator | local fine-tuned revision `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b` | same generator architecture | confirm whether a new registration entry is required |

Approximate active total: 3.466B parameters. BM25, RRF, deterministic parsing and
rule-based verification have no learned parameters. No model-based semantic
verifier is enabled in retained competition configs.

### Parameter-budget gate

- Combined original parameters of every active model must be below 4B.
- Count embedding, reranker, generator, model-based grader/verifier and any helper
  model.
- Quantization, pruning storage or LoRA does not reduce the declared original
  parameter count.
- Unknown parameter evidence or total `>= 4_000_000_000` blocks an official run.

## 4. Fine-tuning rules

M49 uses only the 5,617-record train partition of the frozen normalized-question
group split. The 678 dev and 705 holdout records do not enter the optimizer. Public
questions do not enter training. QLoRA is a training-memory technique; final model
inventory is still the original Qwen3.5-2B architecture.

Training artifacts must record official train hash, split identity, seed, base
revision, dependency versions, LoRA config, checkpoint identity and merged-tree
hash.

## 5. Submission governance

- Use the audited formatter; never submit internal `results.jsonl`.
- `submission.zip` contains only UTF-8 `submission.json`.
- Root shape is `question_id -> {"answer": string}`.
- Validate exact ID set/order, non-empty answers and archive membership.
- Record source, batch, config, code and final ZIP hashes in the private ledger.
- Respect the organizer's current daily submission limit.
- Do not alter predictions to work around scorer infrastructure errors.

Official scorer source SHA-256:

```text
4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891
```

The inspected scorer uses whitespace-tokenized NLTK METEOR, vendored
ASCII-tokenized ROUGE-L and arithmetic macro mean. See
`docs/15-OFFICIAL-SCORING-CONTRACT.md`.

## 6. Reproducibility package

A release candidate must include or reference:

- Git commit and source license;
- exact config and hash;
- official input hashes and corpus revision;
- artifact manifests/checksums;
- model inventory and registration evidence;
- environment (`pip freeze`, OS, CUDA, driver, GPU, seeds);
- exact execution commands;
- result/submission hashes and quality report;
- Data Statement, Model Card and submission ledger.

Templates:

- `docs/templates/DATA-STATEMENT.md`;
- `docs/templates/MODEL-CARD.md`;
- `docs/templates/PRIVATE-SUBMISSION-CHECKLIST.md`;
- `docs/templates/SUBMISSION-LEDGER.csv`.

The current Dockerfile is a CPU/non-root reproducibility scaffold, not a final GPU
competition image. Large data/model/artifact state must be mounted or restored by
an organizer-approved process, never committed or embedded silently.

## 7. Open organizer confirmations

1. Must the M49 merged fine-tuned revision be registered separately from the base?
2. What are the exact final GPU, RAM, disk, timeout and Internet constraints?
3. What exact model-weight mounting/download procedure will reproduction use?
4. Where and when are Model Card/Data Statement submitted?
5. Will private scoring retain the inspected scorer/dependency identity?

These remain fail-closed questions. The repository cannot turn missing organizer
evidence into an implicit approval.
