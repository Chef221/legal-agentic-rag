# Project handoff — UIT DSC 2026 LegalQA

Updated: 2026-08-25

This file is the operational handoff for the next developer. Read `AGENTS.md`
first. The repository has been cleaned to retain M48, M49, M49.1 and the M49.1-JINA35
canonical baseline plus the M45 offline foundation required by all of them. M46 and
M47 remain only as historical measurements in design documents; their runnable
notebooks, configs and outputs have been removed.

## 1. Current answer in one minute

- Current canonical baseline: **M49.1-JINA35** (reconciled and verified in repository source).
- Official Codabench score: **METEOR 0.406858976**, **ROUGE-L 0.496260842** (1000/1000 valid answers).
- Prior baseline: **M49.1** (Qwen3-Reranker), METEOR 0.382772249, ROUGE-L 0.473653736.
- Retained control: **M48**, METEOR 0.2685876695, ROUGE-L 0.3631401334.
- Active reranker: `jinaai/jina-reranker-v3.5` (596,836,352 parameters), total stack 3,405,854,528 parameters (< 4B cap; `docs/artifacts/m491-jina35-parameter-budget-authority.json`).
- Retrieval uses the immutable M45 DB/index built from official `selected-contexts.zip`.
- **M49.1-JINA35 Public-1000 execution is CLOSED.** Closed work must not be rerun merely to verify or reproduce it. A future Public-1000 run is allowed only under a NEW explicitly stated hypothesis/objective with newly established execution authority.
- Postmortem & artifact checksums: [`docs/21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md`](docs/21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md).
- Status tracker: [`CURRENT-WORK.md`](CURRENT-WORK.md).
- **Repository source reconciliation:** **COMPLETE** (Hotfix V1 raw-identity preservation and Hotfix V2 dual-source anchored fallback are integrated and verified in source and tests).
- **Next candidate direction:** Future work (such as M50) requires explicit user approval and a defined experiment plan.

## 2. Retained candidate lineage

```text
Official selected-contexts.zip
  -> M45 normalization/chunking
  -> BM25 + Qwen3 embedding vector index + graph metadata
  -> immutable validated M45 artifact

Official train.json
  -> leakage-safe grouped split: 5,617 train / 678 dev / 705 holdout
  -> M49 QLoRA SFT of Qwen3.5-2B for one epoch
  -> merged local M49 generator

Question
  -> M45 hybrid retrieval + Jina v3.5 reranking
  -> M48 context selection and bounded recovery policy
  -> M49.1 / M49.1-JINA35 plain-text generation/repetition policy using M49 weights
  -> deterministic verification/rendering
  -> answer-only submission.json (1000/1000 valid)
```

M48 is retained as the last non-fine-tuned control. M49 retains the training
workflow and weights identity. M49.1-JINA35 is the active canonical baseline.
