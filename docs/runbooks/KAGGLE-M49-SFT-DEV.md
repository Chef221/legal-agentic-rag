# Kaggle M49 official-only generator SFT

M49 fine-tunes only `Qwen/Qwen3.5-2B` with official answer supervision. Retrieval,
reranking and the M45 DB/index stay unchanged.

## Training

Use `notebooks/M49_KAGGLE_01_TRAIN_GENERATOR.ipynb` with:

1. current source packaged as `legal-agentic-rag-m49-source.zip` or an extracted
   source tree;
2. official `train.json` with SHA-256
   `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988`.

The script recreates the group-safe 5,617/678/705 split and trains one QLoRA epoch
with response-only causal-LM loss. It saves finite `checkpoint-*` directories,
merges the adapter into the base fp16 model and writes:

```text
/kaggle/working/m49-qwen3.5-2b-official-sft-v1
```

Completion requires `m49-training-manifest.json` with `complete: true`. The known
merged tree hash is:

```text
e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b
```

Quick Save with output enabled to persist checkpoints/weights. On resume, add the
previous notebook output and keep source, split, seed and hyperparameters unchanged.

## Dev evaluation

Use `notebooks/M49_KAGGLE_02_DEV_EVAL.ipynb` with source M49, official train,
official scorer, M45 artifacts and the completed M49 training output. It evaluates
the exact frozen dev-200 and never rebuilds the DB.

M49.1 later reused the same merged weights. M49 should therefore be treated as the
training lineage, not as the current public runtime.
