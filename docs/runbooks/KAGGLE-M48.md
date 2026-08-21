# Kaggle M48 control

M48 is the retained non-fine-tuned control. It reuses the immutable M45 artifact
and the base Qwen3.5-2B generator.

## Dev-200

Use `notebooks/M48_KAGGLE_01_DEV_EVAL.ipynb` with:

1. the current repository source packaged as `legal-agentic-rag-m48-source.zip`
   or exposed as an extracted Kaggle source tree;
2. official `train.json`;
3. the audited official scorer ZIP;
4. validated M45 artifacts as an archive/checksum pair or extracted directory.

The runner recreates the frozen dev-200 and compares M48 with immutable historical
control values. It does not need an M47 notebook output.

Expected M48 dev result:

- METEOR `0.26762720229432313`;
- ROUGE-L `0.3660979621381608`;
- 2/200 insufficient-evidence responses.

## Public

Use `notebooks/M48_KAGGLE_02_PUBLIC_SUBMISSION.ipynb` with source M48, official
`public-official.json` and M45 artifacts. No train/scorer input is required.

The batch checkpoints every completed question at:

```text
/kaggle/working/m48-public-qwen3-v1
```

Quick Save with output enabled before a session ends. Add the saved output in a
new session to resume. When 1,000 IDs are complete, the runner validates the batch
and writes `/kaggle/working/submission.zip`.

Measured M48 public score: METEOR `0.2685876695455311`, ROUGE-L
`0.3631401334440235`.
