# Project handoff — UIT DSC 2026 LegalQA

Updated: 2026-08-21

This file is the operational handoff for the next developer. Read `AGENTS.md`
first. The repository has been cleaned to retain M48, M49 and M49.1 plus the M45
offline foundation required by all three. M46 and M47 remain only as historical
measurements in design documents; their runnable notebooks, configs and outputs
have been removed.

## Current Baseline Authority

- **Branch:** `takeover/m491-graphless-narrow20`
- **T4 Implementation Source Checkpoint:** `1773ad4c0ab95147f18d13036e69ba9b46341cbf`
- **Status:** **T4 CLOSED**
- **Serving Architecture:** 4 live retrieval strategies (`BM25`, `DENSE`, `HYBRID`, `HYBRID_RERANK`), 4 retrieval tools (`BM25_SEARCH`, `DENSE_SEARCH`, `HYBRID_SEARCH`, `RERANK_SEARCH`), 7 total runtime tools, 3 active online manifests (`legal_chunks`, `bm25_index`, `vector_index`), no online graph execution.
- **Relationship Behavior:** candidate pool 20 (`relationship_candidate_k = 20`), global `candidate_k = 40`, `top_k = 10`.
- **Final Structural Evidence SHA-256:** `1e5f1c2cc20a0a11be99d20bbc928679a7605f99d3ab49841e2159d67865e348`
- **Behavioral Evidence SHA-256:** `1abdea95697ce5273da9c6b7ac8553bccc41589b9149253783131f86f1694731`
- **M45 Authority SHA-256:** `7e78ad60ff2982592a9471eb8704fce44042add0496268fade3f32db1823ea7a`
- **Next:** No active implementation phase. Future improvement work intentionally deferred.

## 1. Current answer in one minute

- Current best measured candidate: **M49.1**.
- Public-compatible score: **METEOR 0.382772249**, **ROUGE-L 0.473653736**.
- Retained control: **M48**, METEOR 0.2685876695 and ROUGE-L 0.3631401334.
- M49 is the official-only SFT generator used by M49.1; it is not a separate
  recommended public runtime.
- Retrieval still uses the immutable M45 DB/index built from the official
  `selected-contexts.zip` only.
- Do not rebuild the DB unless the corpus, embedding model, chunking or vector
  configuration changes.
- Do not commit official data, indexes, model weights, batch outputs or
  submissions.

The detailed M49.1 audit is in
[`docs/18-M491-PUBLIC-RESULT.md`](docs/18-M491-PUBLIC-RESULT.md).

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
  -> M45 hybrid retrieval + reranking
  -> M48 context selection and bounded recovery policy
  -> M49.1 plain-text generation/repetition policy using M49 weights
  -> deterministic verification/rendering
  -> answer-only submission.json
```

M48 is retained as the last non-fine-tuned control. M49 retains the training
workflow and weights identity. M49.1 is the active online policy.

## 3. Models and parameter compliance

| Role | Model | Immutable revision |
|---|---|---|
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| Generator base | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Generator merged M49 | local merged tree | `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b` |

Approximate combined inventory is 3.466B parameters, below the BTC 4B total
limit. LoRA/quantization does not reduce the declared original parameter count.
Before another official run, verify that the exact base and fine-tuned model
identity has been registered or accepted under the current organizer process.

## 4. Official data and scorer identity

Only organizer-provided data may be used.

| Input | SHA-256 / identity |
|---|---|
| `selected-contexts.zip` package | `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97` |
| canonical corpus revision | `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e` |
| `train.json` | `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988` |
| `public-official.json` | `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5` |
| scorer ZIP | `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891` |
| frozen dev-200 ID list | `694825b5961a90a284ad0364ac4f31a1a85f446519c92274a784c8e2be9a48ad` |

The scorer uses whitespace-tokenized NLTK METEOR as the main metric and the
vendored ASCII-tokenized ROUGE-L as the secondary metric. Read
`docs/15-OFFICIAL-SCORING-CONTRACT.md` before changing answer rendering.

## 5. What is deliberately not in Git

The cleaned repository excludes:

- `selected-contexts.zip`, `train.json`, `public-official.json` and warm-up data;
- the official scorer archive;
- M45 BM25/vector/graph artifacts;
- M49 adapters and merged weights;
- Kaggle checkpoints (`results.jsonl`, `manifest.json`, `batch_state.json`);
- `submission.zip` and copied source ZIPs;
- caches, logs and temporary experiment directories.

Obtain official files from BTC and large private artifacts from the team owner.
Verify their hashes before use. Do not download substitute corpora or generate
synthetic training data.

## 6. Repository map

- `src/legal_agentic_rag/`: reusable offline/online implementation.
- `configs/uit-dsc-2026-task2-m45-qwen3-colab.example.json`: rebuild M45 index.
- `configs/uit-dsc-2026-task2-m48-qwen3-dev.example.json`: retained non-SFT control.
- `configs/uit-dsc-2026-task2-m49-qwen3-dev.example.json`: M49 merged generator.
- `configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json`: active M49.1 runtime.
- `notebooks/kaggle_candidate_dev_common.py`: shared frozen dev evaluator.
- `notebooks/kaggle_public_submission_common.py`: shared resumable public runner.
- `notebooks/m48_*`: retained M48 control runners.
- `notebooks/m49_*`: official-only SFT and M49 dev runner.
- `notebooks/m491_*`: M49.1 dev/public runners.
- `docs/runbooks/`: Kaggle execution guides.
- `tests/`: deterministic unit tests; no model or network download by default.

M45 files remain because M48/M49/M49.1 cannot run without the validated M45
artifact. They are infrastructure, not an additional active answer candidate.

## 7. Reproducing each retained stage

### 7.1 Build M45 artifacts

Use `notebooks/M45_KAGGLE_01_BUILD_DB.ipynb` and
`docs/runbooks/KAGGLE-M45-QWEN3.md`. Required input is the exact official corpus.
Save the validated artifact archive and checksum privately. This step is slow and
normally runs only once.

### 7.2 Run the M48 control

Use:

- `notebooks/M48_KAGGLE_01_DEV_EVAL.ipynb` for frozen dev-200;
- `notebooks/M48_KAGGLE_02_PUBLIC_SUBMISSION.ipynb` for public inference;
- `docs/runbooks/KAGGLE-M48.md`.

M48 uses the base Qwen3.5-2B generator and does not fine-tune or rebuild M45.

### 7.3 Train M49

Use `notebooks/M49_KAGGLE_01_TRAIN_GENERATOR.ipynb`, followed by
`notebooks/M49_KAGGLE_02_DEV_EVAL.ipynb`. The runbook is
`docs/runbooks/KAGGLE-M49-SFT-DEV.md`.

Training must consume only the 5,617 training records produced by the frozen
group-safe split. Expected merged model tree hash is:

```text
e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b
```

### 7.4 Run M49.1

Use `notebooks/M491_KAGGLE_01_DEV_EVAL.ipynb` for evaluation and
`notebooks/M491_KAGGLE_02_PUBLIC_SUBMISSION.ipynb` for resumable public inference.
Follow `docs/runbooks/KAGGLE-M491.md`.

The public batch directory is `/kaggle/working/m491-public-qwen3-v1`. The batch
CLI checkpoints after each completed question. To cross a Kaggle session boundary:

1. Quick Save with output enabled, or package the batch directory as a private
   dataset;
2. add that notebook output/dataset to the next notebook;
3. let the runner restore the highest valid matching checkpoint;
4. confirm the restored record count before running inference.

Never run a restore cell after new records already exist in `/kaggle/working`,
because that can overwrite newer progress with an older input checkpoint.

## 8. Latest measured behavior and the key technical issue

M49.1 produced 1,000 valid answers and scored substantially above M48. However,
900/1,000 records used `generator_model_error_fallback`. The likely cause is an
output-contract mismatch:

- M49 SFT teaches question -> natural-language answer;
- M49.1 inference expects natural language containing valid `[E#]` evidence markers;
- markerless output is rejected and becomes top-evidence fallback.

This is the central hypothesis for the next iteration. It must be tested, not
assumed. The top-evidence fallback itself is strong under the official lexical
metrics, so removing it can lower the score even if warning counts look better.

## 9. Recommended next milestone

Create M50 only as a controlled successor to M49.1:

1. freeze M49.1 source/config/result hashes as the control;
2. evaluate on the exact same group-safe dev-200 with the exact scorer;
3. test question-aware extractive trimming and answer-length selection first;
4. separately test an inference contract aligned with the answer-only SFT output;
5. preserve deterministic top-evidence fallback as an ablation arm;
6. measure METEOR, ROUGE-L, fallback types, answer length and latency;
7. promote only when METEOR improves and compliance/reproducibility gates pass.

Do not tune individual public question IDs, use public answers as supervision,
create synthetic QA/evidence, add external legal data or call model APIs.

## 10. Local development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Tests must remain offline and deterministic. Large model downloads belong only in
explicit Kaggle/Colab workflows. No dependency was added as part of repository
cleanup.

## 11. Before committing to GitHub

Run:

```bash
git status --short
git diff --check
python -m pytest
```

Inspect every untracked file. Commit code, configs, notebooks, tests and docs only.
Do not commit official data, private outputs, model weights, archives, tokens or
local caches. This cleanup does not perform `git commit` or `git push`; the owner
must review and publish the final diff.

After committing, create a Kaggle source package when needed:

```bash
python scripts/package_kaggle_source.py m491
```

The archive is written under ignored `dist/` and must be uploaded privately, not
committed back to the repository. Use `m48` or `m49` for the other retained flows.
