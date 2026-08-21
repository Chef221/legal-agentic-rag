# Kaggle M49.1

M49.1 is the current best measured runtime. It reuses M45 retrieval artifacts and
the completed M49 merged generator; it neither rebuilds the DB nor fine-tunes.

## Dev-200

Use `notebooks/M491_KAGGLE_01_DEV_EVAL.ipynb` with:

1. current source packaged as `legal-agentic-rag-m491-source.zip` or an extracted
   source tree;
2. official `train.json` and official scorer ZIP;
3. validated M45 artifacts;
4. completed M49 training output with the expected merged hash.

The output directory is `/kaggle/working/m491-candidate-dev200`.

## Public 1,000

Use `notebooks/M491_KAGGLE_02_PUBLIC_SUBMISSION.ipynb` with source M49.1, official
`public-official.json`, M45 artifacts and completed M49 training output.

The batch checkpoints after each question at:

```text
/kaggle/working/m491-public-qwen3-v1
```

To cross sessions, Quick Save with output enabled (or package the batch directory
as a private dataset), add the saved output to the next notebook, restore it before
starting inference and verify the record count. Do not restore an older input over
a newer `/kaggle/working` batch.

At 1,000 records the runner validates IDs/answers and creates:

```text
/kaggle/working/submission.zip
```

Measured public-compatible result: METEOR `0.382772249`, ROUGE-L `0.473653736`.
See `docs/18-M491-PUBLIC-RESULT.md` for hashes and telemetry.
