# M44.3 Reranker Ablation Runbook

## 1. Mục tiêu

Chạy retrieval-only ablation cho approved
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` trên leakage-safe development
split. Giai đoạn này không chạy Qwen, không tạo submission và không coi
answer-term coverage là relevance gold.

Ba candidate độc lập:

```text
candidate_k = 20
candidate_k = 40
candidate_k = 60
```

Mọi candidate giữ `top_k=20`, cùng code, dev source, official artifacts, E5 và
reranker revision. Mỗi run dùng output directory mới.

## 2. Kaggle inputs

Notebook cần gắn hai private datasets:

1. `khoanguyn221/uit-dsc-2026-task2-serving-v0430`;
2. dataset chứa `development.json` và `split_manifest.json` được đóng gói từ
   M44.1 local split.

Không upload `train.json`, model cache, AIO artifact hoặc external data vào
serving dataset.

## 3. Cài code

```python
!git clone https://github.com/Chef221/legal-agentic-rag.git /kaggle/working/legal-agentic-rag
%cd /kaggle/working/legal-agentic-rag
!git checkout <M44.3_COMMIT>
!python -m pip install -q -e . --no-deps
```

Xác minh:

```python
import legal_agentic_rag
print(legal_agentic_rag.__version__)
assert legal_agentic_rag.__version__ == "0.44.2"
```

## 4. Tìm development source

```python
from pathlib import Path

development_files = list(Path("/kaggle/input").rglob("development.json"))
assert len(development_files) == 1, development_files
DEVELOPMENT = development_files[0]
print(DEVELOPMENT)
```

## 5. Smoke một câu

```python
import subprocess

repo = Path("/kaggle/working/legal-agentic-rag")
config = repo / "configs/uit-dsc-2026-task2-reranker-kaggle.example.json"
smoke = Path("/kaggle/working/m44-rerank-smoke-k20")

subprocess.run(
    [
        "legal-rag-diagnose-retrieval",
        "--config", str(config),
        "--questions", str(DEVELOPMENT),
        "--output", str(smoke),
        "--top-k", "20",
        "--candidate-k", "20",
        "--max-cases", "1",
        "--include-reranker",
    ],
    check=True,
)
```

Chỉ tiếp tục khi report có một successful case, zero failed case và bốn branch.

## 6. Chạy ba candidate

```python
for candidate_k in (20, 40, 60):
    output = Path(f"/kaggle/working/m44-rerank-k{candidate_k}")
    subprocess.run(
        [
            "legal-rag-diagnose-retrieval",
            "--config", str(config),
            "--questions", str(DEVELOPMENT),
            "--output", str(output),
            "--top-k", "20",
            "--candidate-k", str(candidate_k),
            "--include-reranker",
        ],
        check=True,
    )
```

Không chạy song song ba process trên một GPU. Model/index được nạp lại giữa các
CLI run; đổi candidate order không thay đổi report identity nhưng ảnh hưởng
thời gian cold start.

## 7. Gate kiểm tra

Với từng `retrieval_diagnostics.json`, kiểm tra:

- `question_count == 991`;
- `successful_case_count == 991`;
- `failed_case_count == 0`;
- `include_reranker == true`;
- mọi successful case có bốn branch;
- source SHA-256 bằng
  `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`;
- không có question, reference answer hoặc legal text trong report.

So sánh candidate bằng:

- reranker latency;
- mean hybrid/reranked Jaccard;
- mean absolute rank change;
- reranked document diversity;
- answer-term coverage delta, chỉ như non-gold diagnostic;
- explicit-reference regression.

Không chọn candidate chỉ vì coverage delta cao. Candidate được đưa sang full
generation phải có zero model error, latency/memory phù hợp và không làm giảm
explicit-reference behavior.

## 8. Output cần giữ

Quick Save notebook sau khi cả ba report hoàn tất. Tải hoặc publish đúng ba
thư mục report cùng notebook log. Không cần đóng gói model/index vì chúng đã là
input dataset riêng.
