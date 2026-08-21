# Kaggle M45 artifact build

M45 is retained only as the offline DB/index foundation for M48, M49 and M49.1.
Do not use M45 as an answer candidate.

## Inputs

1. Current repository source packaged for Kaggle or exposed as an extracted tree.
2. Exact official `selected-contexts.zip` with package SHA-256
   `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97`.
3. Internet enabled for exact registered model downloads.
4. Compatible GPU; T4 x2 was the working Kaggle choice.

Use `notebooks/M45_KAGGLE_01_BUILD_DB.ipynb`. It creates the artifact root:

```text
/kaggle/working/uit-dsc-2026-task2-m45-artifacts
```

The build is resumable by persisted stage/checkpoint identity. Vector embedding is
the long stage. Do not restart from a different source/config revision and call it
the same artifact.

## Completion gate

The build is complete only when all required BM25/vector/vector-serving files exist
and `build_validation_full_corpus.json` has `is_valid: true`. The validated corpus
contains 385,962 legal chunks.

Package the full artifact directory and its SHA-256 into private storage/Kaggle
notebook output. Do not commit the archive, checksum, vectors, indexes or source
corpus to Git.

M48/M49/M49.1 restore this artifact and never run the offline build during question
inference.
