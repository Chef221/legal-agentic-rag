# Hướng dẫn hệ thống cho thành viên mới

## Mục tiêu

Repository triển khai RAG pháp luật Việt Nam cho UIT DSC 2026 Task 2. M49.1 là
candidate tốt nhất đã đo; M48 là control không fine-tune; M49 là lineage weights
fine-tuned từ official train. Tất cả dùng DB/index M45.

Đọc theo thứ tự: `AGENTS.md` -> `HANDOFF.md` -> `docs/00-START-HERE.md` -> contract
dữ liệu/scorer -> tài liệu pipeline tương ứng.

## Pipeline offline

```text
selected-contexts.zip
  -> UIT DSC adapter + audit
  -> normalized/cleaned documents
  -> legal blocks and article-first chunks
  -> BM25 index
  -> Qwen3 embedding vector index
  -> bounded graph relationships
  -> serving metadata + manifests + validation
```

Offline chỉ chạy khi corpus/chunking/embedding/index config thay đổi. M49/M49.1
không thay các thành phần này nên dùng lại artifact M45.

Các package chính: `offline`, `indexing`, `embeddings`,
`competition/uit_dsc_2026` và `runtime/competition_offline.py`.

## Pipeline online

```text
question
  -> validation and query normalization
  -> BM25 + dense retrieval
  -> reciprocal-rank fusion
  -> Qwen3 cross-encoder reranking
  -> bounded graph expansion/retry when configured
  -> context selection
  -> Qwen3.5-2B answer generation
  -> deterministic claim/citation verification
  -> answer rendering
```

Online không được clean/chunk/embed/build index. Retry bị giới hạn bởi config và
không được dùng để che retrieval yếu.

## Retained profiles

| Profile | Generator | Mục đích |
|---|---|---|
| M48 | base Qwen3.5-2B | control không fine-tune |
| M49 | official-only merged SFT | training lineage/dev diagnostics |
| M49.1 | M49 weights + plain-text/repetition policy | current measured runtime |

M45 chỉ là offline artifact foundation. M46/M47 executable đã được archive khỏi
active tree; số đo lịch sử vẫn nằm trong design decisions.

## Package map

| Boundary | Package |
|---|---|
| Raw BTC adapter/submission | `competition/uit_dsc_2026` |
| Unified contracts | `schemas`, `contracts` |
| Configuration | `configuration` |
| Corpus processing | `offline` |
| Persistent indexes | `indexing` |
| Retrieval/reranking | `retrieval`, `reranking`, `embeddings` |
| Generation/verification | `generation` |
| Bounded orchestration | `agent`, `tools` |
| Runtime composition | `runtime` |
| CLI/API/UI | `serving` |
| Evaluation | `evaluation` |

## API và UI

FastAPI/UI are diagnostic surfaces around the same online runtime. They must not
bypass config/artifact validation or expose raw stack traces/secrets. Competition
submission does not require running the API/UI.

## Batch inference

`legal-rag-batch` checkpoints each completed question to a batch directory.
`legal-rag-submit` validates ID coverage/order and writes a deterministic ZIP with
only `submission.json`. Internal evidence markers are removed at the competition
renderer boundary.

Across Kaggle sessions, save the whole batch directory and restore it before new
inference. Never overwrite a newer working checkpoint with an older input copy.

## Evaluation và chọn model

Use the frozen group-safe dev-200 and exact scorer checksum. METEOR is primary,
ROUGE-L secondary. Also report fallback types, answer length, model errors and
latency. Change one hypothesis per experiment and preserve the preceding candidate
as an immutable control.

Do not treat a smoke test, partial dev run or public answer inspection as a quality
benchmark. Never tune rules by public question ID.

## Competition constraints

- official data only; no external corpus or synthetic training examples;
- no model/API intermediaries;
- combined original model parameters below 4B;
- exact models/revisions require organizer registration evidence;
- no dataset, model weights, indexes, outputs or secrets in Git;
- submission shape follows the audited scorer contract, not an old overview.

## Quy trình làm việc cho thành viên

```powershell
git clone <repository-url>
cd legal-agentic-rag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Before coding:

1. read `AGENTS.md` and task-specific required docs;
2. inspect `git status` and preserve unrelated work;
3. define the hypothesis, inputs, outputs and promotion gate;
4. list files/tests/dependencies expected to change;
5. create a short branch and keep the diff scoped.

Before PR:

```powershell
python -m pytest
python -m compileall -q src tests notebooks
python -m pip check
git diff --check
```

Review untracked files manually. Commit source, configs, tests, notebooks and docs
only.
