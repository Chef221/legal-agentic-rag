# Kaggle M43 — Qwen2.5-3B Public Batch

## 1. Compliance gate

Chỉ dùng output M43 để nộp chính thức sau khi BTC đã duyệt đúng ba model/revision
đang active hoặc được cấu hình trong hệ thống:

- `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3`;
- `Qwen/Qwen2.5-3B-Instruct@a1d308dfcc03e09da285d49d912439a655a571e8`;
- reranker không được gọi trong profile M43.

Không thêm dataset, model hoặc API khác. Archive input phải có SHA-256:

```text
90d4d211a20f6d3a6f894d8dd33c0f187fcf141c1bcbc3814d8dcc7e003e729c
```

## 2. Tạo Kaggle Dataset

Tạo một private Kaggle Dataset và upload đúng hai file:

```text
uit-dsc-2026-task2-serving-v0430.tar.gz
uit-dsc-2026-task2-serving-v0430.tar.gz.sha256
```

Không upload artifact AIO hoặc dataset ngoài BTC. Sau khi dataset xử lý xong,
tạo notebook mới, add dataset này làm Input, bật Internet và chọn GPU.

## 3. Cài môi trường

P100 cần PyTorch còn hỗ trợ CUDA capability 6.0. Chạy cell sau một lần:

```bash
!pip uninstall -y torch torchvision torchaudio torchcodec
!pip install -q --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q --no-cache-dir "transformers==5.0.0" "sentence-transformers==5.4.1"
```

Sau đó chọn **Restart Session**, không chọn Factory Reset. Sau khi reconnect,
không chạy lại cell cài đặt nếu version check đã đúng.

## 4. Verify GPU và clone source

```python
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
assert torch.cuda.is_available()
```

```bash
%cd /kaggle/working
!git clone https://github.com/Chef221/legal-agentic-rag.git
%cd /kaggle/working/legal-agentic-rag
!python -m pip install -q -e . --no-deps
```

```python
import legal_agentic_rag
import sentence_transformers
import transformers

print("Project:", legal_agentic_rag.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)
print("Transformers:", transformers.__version__)
assert legal_agentic_rag.__version__ == "0.43.0"
assert sentence_transformers.__version__ == "5.4.1"
assert transformers.__version__ == "5.0.0"
```

## 5. Verify và giải nén artifact

```python
from hashlib import sha256
from pathlib import Path
import subprocess

EXPECTED = "90d4d211a20f6d3a6f894d8dd33c0f187fcf141c1bcbc3814d8dcc7e003e729c"
archives = list(Path("/kaggle/input").rglob("uit-dsc-2026-task2-serving-v0430.tar.gz"))
assert len(archives) == 1, archives
archive = archives[0]

digest = sha256()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
actual = digest.hexdigest()
print("Archive:", archive)
print("SHA-256:", actual)
assert actual == EXPECTED

root = Path("/kaggle/working/uit-dsc-2026-task2-serving-v0430")
assert not root.exists(), root
root.mkdir()
subprocess.run(["tar", "-xzf", str(archive), "-C", str(root)], check=True)
print("Extracted:", root)
```

```python
from pathlib import Path

root = Path("/kaggle/working/uit-dsc-2026-task2-serving-v0430")
required = [
    "legal_chunks/manifest.json",
    "bm25/index.sqlite3",
    "bm25/manifest.json",
    "vector/vectors.npy",
    "vector/chunks.jsonl",
    "vector/manifest.json",
    "vector_serving/metadata.sqlite3",
    "vector_serving/manifest.json",
    "graph/graph.json",
    "graph/manifest.json",
    "build_validation_full_corpus.json",
    "dataset_manifest.json",
    "public-official.json",
]
for relative in required:
    path = root / relative
    print("✅" if path.is_file() else "❌", path)
    assert path.is_file(), path
```

## 6. Smoke test một câu

Cell này load E5, vector lên GPU và Qwen lần đầu nên sẽ chậm hơn các câu sau.

```python
from pathlib import Path
from time import perf_counter

from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.runtime import OnlineRuntimeFactory
from legal_agentic_rag.schemas import RetrievalQuery
from legal_agentic_rag.serving.config_loader import load_application_config

repo = Path("/kaggle/working/legal-agentic-rag")
root = Path("/kaggle/working/uit-dsc-2026-task2-serving-v0430")
config = load_application_config(
    repo / "configs/uit-dsc-2026-task2-qwen3b-kaggle.example.json"
)
question = UitDsc2026DataLoader().load_questions(
    root / "public-official.json",
    require_reference_answers=False,
)[0]
runtime = OnlineRuntimeFactory(config).build()
query = RetrievalQuery(
    query_id=f"m43-smoke-{question.question_id}",
    original_question=question.question,
    normalized_question=question.question.strip(),
    top_k=8,
    candidate_k=60,
)
started = perf_counter()
result = runtime.answer(query)
print("Seconds:", round(perf_counter() - started, 2))
print("Stop reason:", result.stop_reason.value)
print("Insufficient:", result.response.insufficient_evidence)
print("Backend:", result.response.metadata.get("generator_backend"))
print("Model:", result.response.metadata.get("model_name"))
print("Answer:", result.response.answer)
print("Citations:", len(result.response.citations))
print("Warnings:", result.response.warnings)
assert result.response.metadata.get("generator_backend") == "transformers"
assert result.response.metadata.get("model_name") == "Qwen/Qwen2.5-3B-Instruct"
```

Chỉ chạy full batch nếu smoke test không có `generator:model_error` hoặc
`backend_initialization_error`.

## 7. Chạy toàn bộ public batch

Smoke process đã kết thúc sau cell, nên CLI sẽ load model lại đúng một lần cho
toàn batch. Không chạy hai batch song song.

```bash
%cd /kaggle/working/legal-agentic-rag
!legal-rag-batch \
  --config /kaggle/working/legal-agentic-rag/configs/uit-dsc-2026-task2-qwen3b-kaggle.example.json \
  --questions /kaggle/working/uit-dsc-2026-task2-serving-v0430/public-official.json \
  --output /kaggle/working/m43-public-qwen3b
```

CLI checkpoint sau từng câu và log tiến độ mỗi 25 câu. Nếu cell bị dừng nhưng
session còn sống, chạy lại nguyên cell trên để resume.

## 8. Kiểm tra batch và tạo submission

```python
from pathlib import Path
import json

root = Path("/kaggle/working/m43-public-qwen3b")
records = [
    json.loads(line)
    for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
responses = [record["response"] for record in records]
print("Records:", len(records))
print("Unique IDs:", len({record["question_id"] for record in records}))
print("Manifest records:", manifest["record_count"])
print("Insufficient:", sum(item["insufficient_evidence"] for item in responses))
print("Average answer chars:", round(sum(len(item["answer"]) for item in responses) / len(responses), 1))
print("Model errors:", sum("generator:model_error" in item["warnings"] for item in responses))
assert len(records) == 1000
assert len({record["question_id"] for record in records}) == 1000
assert manifest["record_count"] == 1000
```

```bash
!mkdir -p /kaggle/working/m43-submission
!legal-rag-submit \
  --questions /kaggle/working/uit-dsc-2026-task2-serving-v0430/public-official.json \
  --batch /kaggle/working/m43-public-qwen3b \
  --output /kaggle/working/m43-submission/submission.zip
```

```python
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json

submission = Path("/kaggle/working/m43-submission/submission.zip")
with ZipFile(submission) as archive:
    assert archive.namelist() == ["submission.json"]
    payload = json.loads(archive.read("submission.json"))
print("Submission records:", len(payload))
print("Archive SHA-256:", sha256(submission.read_bytes()).hexdigest())
assert len(payload) == 1000
assert all(isinstance(value.get("answer"), str) for value in payload.values())
```

## 9. Lưu output trước khi tắt GPU

Sau khi tất cả cell hoàn tất:

1. chọn **Save Version**;
2. chọn **Save & Run All** chỉ khi muốn chạy lại toàn notebook; nếu batch đã xong,
   dùng Quick Save/Save Version giữ output hiện tại;
3. chờ version có trạng thái hoàn tất và Output Data xuất hiện;
4. kiểm tra có `m43-submission/submission.zip`;
5. chỉ sau đó mới Stop Session/GPU.

Nếu session chết trước khi Save Version thì `/kaggle/working` bị mất. Nếu muốn
dừng chủ động giữa batch, interrupt cell, nén riêng thư mục
`m43-public-qwen3b`, Save Version, rồi ở session mới giải nén nó về đúng path và
chạy lại cell batch với cùng commit/config.
