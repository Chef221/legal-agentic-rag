# M44.4 End-to-end Reranker A/B Runbook

## 1. Mục tiêu và biến kiểm soát

Thí nghiệm so sánh answer-level trên đúng 991 leakage-safe development records:

| Run | Strategy | top_k | candidate_k |
|---|---|---:|---:|
| Control | `hybrid_rerank` | 8 | 20 |
| Treatment | `hybrid_rerank` | 8 | 40 |

Hai config giữ nguyên official artifacts, query understanding, E5, mMARCO
reranker, Qwen generator, evidence selection, verifier và timeout. Không dùng
public answers, external data, synthetic data hoặc API model.

## 2. Kaggle inputs

Gắn hai private datasets đã dùng ở M44.3:

1. `khoanguyn221/uit-dsc-2026-task2-serving-v0430`;
2. dataset chứa đúng `development.json` của M44.1.

Development source hợp lệ có SHA-256:

```text
8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8
```

Notebook dùng accelerator Tesla T4. Không chạy hai batch song song.

## 3. Cài exact code

```python
!git clone https://github.com/Chef221/legal-agentic-rag.git /kaggle/working/legal-agentic-rag
%cd /kaggle/working/legal-agentic-rag
!git checkout <M44.4_COMMIT>
!python -m pip install -q -e . --no-deps
```

```python
import legal_agentic_rag
print(legal_agentic_rag.__version__)
assert legal_agentic_rag.__version__ == "0.44.4"
```

## 4. Xác minh source và artifacts

```python
from hashlib import sha256
from pathlib import Path

development_files = list(Path("/kaggle/input").rglob("development.json"))
assert len(development_files) == 1, development_files
DEVELOPMENT = development_files[0]

digest = sha256(DEVELOPMENT.read_bytes()).hexdigest()
print(DEVELOPMENT)
print(digest)
assert digest == "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"

ARTIFACT_ROOT = Path(
    "/kaggle/input/datasets/khoanguyn221/"
    "uit-dsc-2026-task2-serving-v0430/"
    "uit-dsc-2026-task2-serving-v0430"
)
for relative in (
    "build_validation_full_corpus.json",
    "bm25/index.sqlite3",
    "vector/vectors.npy",
    "vector_serving/metadata.sqlite3",
):
    path = ARTIFACT_ROOT / relative
    print("✅" if path.exists() else "❌", path)
    assert path.exists(), path
```

## 5. Chạy control k=20

```python
import subprocess

REPO = Path("/kaggle/working/legal-agentic-rag")
CONTROL_CONFIG = REPO / "configs/uit-dsc-2026-task2-qwen3b-rerank-k20-kaggle.example.json"
CONTROL_OUTPUT = Path("/kaggle/working/m44-e2e-rerank-k20-v0444")

subprocess.run(
    [
        "legal-rag-batch",
        "--config", str(CONTROL_CONFIG),
        "--questions", str(DEVELOPMENT),
        "--output", str(CONTROL_OUTPUT),
    ],
    check=True,
)
```

Nếu session ngắt, chạy lại đúng cell này với đúng output directory. Batch chỉ
resume khi source hash, config hash và code version vẫn khớp.

## 6. Chạy treatment k=40

Chỉ chạy sau khi control có `manifest.json`.

```python
TREATMENT_CONFIG = REPO / "configs/uit-dsc-2026-task2-qwen3b-rerank-k40-kaggle.example.json"
TREATMENT_OUTPUT = Path("/kaggle/working/m44-e2e-rerank-k40-v0444")

subprocess.run(
    [
        "legal-rag-batch",
        "--config", str(TREATMENT_CONFIG),
        "--questions", str(DEVELOPMENT),
        "--output", str(TREATMENT_OUTPUT),
    ],
    check=True,
)
```

## 7. Kiểm tra completeness

```python
import json

for label, output in (("k20", CONTROL_OUTPUT), ("k40", TREATMENT_OUTPUT)):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [record["question_id"] for record in records]
    print(label, len(records), len(set(ids)), manifest["record_count"])
    assert len(records) == len(set(ids)) == manifest["record_count"] == 991
```

## 8. Tạo submission nội bộ và chấm

```python
for label, output in (("k20", CONTROL_OUTPUT), ("k40", TREATMENT_OUTPUT)):
    submission_dir = Path(f"/kaggle/working/m44-e2e-score-{label}-v0444")
    submission_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "legal-rag-submit",
            "--questions", str(DEVELOPMENT),
            "--batch", str(output),
            "--output", str(submission_dir / "submission.zip"),
        ],
        check=True,
    )
```

Official-compatible scoring cần NLTK 3.7 và WordNet local. Resource này chỉ phục
vụ tái lập scorer BTC, không đi vào model/pipeline. Nếu môi trường Kaggle chưa
có đúng dependency/resource, tải hai submission ZIP và chấm trong môi trường
local đã pin; không silently dùng diagnostic score thay thế.

```python
for label in ("k20", "k40"):
    submission = Path(f"/kaggle/working/m44-e2e-score-{label}-v0444/submission.zip")
    score_output = Path(f"/kaggle/working/m44-e2e-official-score-{label}-v0444")
    subprocess.run(
        [
            "legal-rag-score-warmup",
            "--metric-mode", "official_compatible",
            "--references", str(DEVELOPMENT),
            "--submission", str(submission),
            "--output", str(score_output),
        ],
        check=True,
    )
```

## 9. Quy tắc quyết định

Đọc `warmup_score.json` của hai run. Candidate-k 40 chỉ thắng khi:

1. METEOR lớn hơn k=20 trên cùng 991 records;
2. ROUGE-L không regression nghiêm trọng;
3. không tăng mạnh abstention, generator error hoặc citation failure;
4. latency/memory vẫn chạy được trong budget thực tế.

Nếu METEOR không tăng, giữ k=20. Không chọn k=40 chỉ vì retrieval coverage tăng.
Không chạy k=60 trước khi review kết quả answer-level này.

## 10. Output cần lưu

Quick Save notebook và giữ:

- hai batch directories gồm `results.jsonl`, `batch_state.json`, `manifest.json`;
- hai `submission.zip` nội bộ;
- hai `warmup_score.json`;
- notebook log;
- exact commit hash và GPU identity.

Không commit các output này vào Git.
