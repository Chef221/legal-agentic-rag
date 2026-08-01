# Model Card — UIT DSC 2026 Task 2

> Tạo một bản riêng cho từng release candidate. Model chưa có registration
> evidence theo Form BTC không được dùng cho official run.

## 1. Identity

- Team/release ID: `[REQUIRED]`
- Code commit: `[REQUIRED]`
- Application config SHA-256: `[REQUIRED]`
- Docker image digest: `[REQUIRED]`

## 2. Components

| Role | Provider | Model ID | Immutable revision | License | BTC registration evidence |
|---|---|---|---|---|---|
| Embedding | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` |
| Reranker | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` |
| Generator | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` |
| Verifier | `[REQUIRED/NONE]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` |

## 3. Intended use

- Intended task: Vietnamese legal question answering over official BTC corpus.
- Intended users: `[REQUIRED]`
- Out-of-scope use: legal representation, autonomous legal decisions, or
  answers without verification against official legal text.

## 4. Training and adaptation

- Pretraining disclosure from model authors: `[REQUIRED]`
- Competition-data fine-tuning: `[NONE or REQUIRED]`
- Training data checksums/splits: `[REQUIRED]`
- Hyperparameters, seed and checkpoints: `[REQUIRED]`
- External data/augmentation confirmation: `[REQUIRED]`

## 5. Evaluation

| Dataset/revision | Metric | Result | Status |
|---|---|---:|---|
| `[REQUIRED]` | METEOR | `[REQUIRED]` | `[diagnostic|official]` |
| `[REQUIRED]` | ROUGE-L | `[REQUIRED]` | `[diagnostic|official]` |

- Retrieval metrics: `[REQUIRED]`
- Failure/abstention rate: `[REQUIRED]`
- Latency and hardware: `[REQUIRED]`
- Regression comparison report: `[REQUIRED]`

## 6. Grounding and safety

- Evidence selection policy: `[REQUIRED]`
- Citation verification policy: `[REQUIRED]`
- Insufficient-evidence behavior: `[REQUIRED]`
- Prompt injection/data leakage controls: `[REQUIRED]`

## 7. Limitations

- Corpus/date coverage: `[REQUIRED]`
- Hallucination and outdated-law risk: `[REQUIRED]`
- Metric limitations: `[REQUIRED]`
- Known failure cases and mitigations: `[REQUIRED]`

## 8. Reproduction

- Python/package freeze: `[REQUIRED]`
- OS/CUDA/driver/GPU: `[REQUIRED]`
- Model/cache acquisition procedure: `[REQUIRED]`
- Exact build and inference commands: `[REQUIRED]`
