# Kaggle M43 — Qwen2.5-3B Public Batch

## 1. Baseline identity và compliance gate

Baseline tái lập tại code version `0.43.1`, commit `96e6d5a`. Người dùng đã xác
nhận BTC duyệt ba model/revision trong inventory; M43 chỉ active embedding và
generator:

- `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3`;
- `Qwen/Qwen2.5-3B-Instruct@a1d308dfcc03e09da285d49d912439a655a571e8`;
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@1427fd652930e4ba29e8149678df786c240d8825`
  đã được duyệt nhưng **không được gọi** trong profile M43.

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

Không upload artifact AIO hoặc dataset ngoài BTC. Kaggle có thể hiển thị input
theo một trong hai dạng: giữ nguyên hai file archive/checksum, hoặc tự bung nội
dung thành một thư mục lồng. Runbook hỗ trợ cả hai dạng. Sau khi dataset xử lý
xong, tạo notebook mới, add dataset này làm Input, bật Internet và chọn GPU.

## 3. Chọn GPU và cài môi trường

Ưu tiên **một Tesla T4** cho baseline. T4 x2 vẫn chạy được nhưng code M43 hiện
chỉ sử dụng GPU0, nên GPU thứ hai không làm batch nhanh gấp đôi.

Kaggle từng cung cấp PyTorch `2.10.0+cu128` không hỗ trợ P100 `sm_60`. Nếu vẫn
dùng P100, bắt buộc cài PyTorch còn hỗ trợ capability 6.0 trước khi chạy:

```bash
!pip uninstall -y torch torchvision torchaudio torchcodec
!pip install -q --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Sau khi thay PyTorch trên P100, chọn **Restart Session**, không chọn Factory
Reset. Trên cả T4 và P100, cài đúng provider versions sau khi runtime cuối đã
sẵn sàng:

```bash
!pip install -q --no-cache-dir "transformers==5.0.0" "sentence-transformers==5.4.1"
```

Không chạy lại cell cài đặt nếu version check đã đúng. Với T4, vẫn phải kiểm tra
`torch.cuda.is_available()` và chạy một operation CUDA thực, không chỉ nhìn
Kaggle UI.

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
!git checkout 96e6d5a
!python -m pip install -q -e . --no-deps
```

```python
import legal_agentic_rag
import sentence_transformers
import transformers

print("Project:", legal_agentic_rag.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)
print("Transformers:", transformers.__version__)
assert legal_agentic_rag.__version__ == "0.43.1"
assert sentence_transformers.__version__ == "5.4.1"
assert transformers.__version__ == "5.0.0"
```

## 5. Xác định artifact input

Cell này nhận cả archive và thư mục Kaggle đã tự giải nén:

```python
from hashlib import sha256
from pathlib import Path
import os
import subprocess

EXPECTED = "90d4d211a20f6d3a6f894d8dd33c0f187fcf141c1bcbc3814d8dcc7e003e729c"
archives = list(Path("/kaggle/input").rglob("uit-dsc-2026-task2-serving-v0430.tar.gz"))
root = Path("/kaggle/working/uit-dsc-2026-task2-serving-v0430")
assert not root.exists(), root

if len(archives) == 1:
    archive = archives[0]
    digest = sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    print("Archive:", archive)
    print("SHA-256:", actual)
    assert actual == EXPECTED
    root.mkdir()
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(root)], check=True)
else:
    candidates = [
        path.parent
        for path in Path("/kaggle/input").rglob("build_validation_full_corpus.json")
        if (path.parent / "public-official.json").is_file()
        and (path.parent / "vector/vectors.npy").is_file()
        and (path.parent / "bm25/index.sqlite3").is_file()
    ]
    assert len(candidates) == 1, candidates
    # Symlink tránh copy thêm nhiều GB vào /kaggle/working.
    os.symlink(candidates[0], root, target_is_directory=True)

print("Artifact root:", root)
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
session còn sống, chạy lại nguyên cell trên để resume. Không resume bằng commit,
config, question source hoặc output identity khác.

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
print("Retrieval errors:", sum("retrieval:model_error" in item["warnings"] for item in responses))
assert len(records) == 1000
assert len({record["question_id"] for record in records}) == 1000
assert manifest["record_count"] == 1000
assert sum("retrieval:model_error" in item["warnings"] for item in responses) == 0
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

## 9. Lưu output/checkpoint trước khi tắt GPU

Sau khi tất cả cell hoàn tất:

1. chọn **Save Version**;
2. chọn **Save & Run All** chỉ khi muốn chạy lại toàn notebook; nếu batch đã xong,
   dùng Quick Save/Save Version giữ output hiện tại;
3. chờ version có trạng thái hoàn tất và Output Data xuất hiện;
4. kiểm tra có `m43-submission/submission.zip`;
5. chỉ sau đó mới Stop Session/GPU.

Nếu session chết trước khi Save Version thì `/kaggle/working` bị mất. Save
Version có thể fail hoặc không giữ output của interactive session. Cách bền hơn:

1. dừng process sạch nếu cần;
2. nén riêng thư mục batch có `results.jsonl` và `batch_state.json`;
3. tải archive về máy hoặc tạo **private Kaggle Dataset** từ archive;
4. session mới add dataset checkpoint làm Input;
5. copy hai file checkpoint vào đúng output directory;
6. checkout đúng `96e6d5a`, dùng đúng config/source rồi chạy lại CLI.

Không tìm file `.zip` nếu Kaggle đã tự bung dataset. Hãy `rglob("results.jsonl")`
và xác minh file đi cùng `batch_state.json`.

## 10. Theo dõi process

Nếu chạy batch nền, theo dõi bằng checkpoint thay vì chỉ nhìn GPU utilization:

```python
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

run = Path("/kaggle/working/m43-public-qwen3b")
results = run / "results.jsonl"
completed = sum(1 for line in results.open(encoding="utf-8") if line.strip()) if results.exists() else 0
print("Completed:", completed, "/ 1000")
print("Manifest:", (run / "manifest.json").exists())
print("Checked:", datetime.now(timezone.utc).isoformat())
print(subprocess.run(["bash", "-lc", "ps aux | grep '[l]egal-rag-batch'"], capture_output=True, text=True).stdout)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout)
```

GPU utilization 0% tại đúng thời điểm kiểm tra không kết luận process hỏng: BM25
và orchestration chạy CPU, còn CUDA theo đợt. Dấu hiệu đáng tin hơn là số record
tăng và log có stage completion. Nếu record đứng lâu, đọc log/traceback trước
khi restart.

## 11. Quality gate sau batch

Không format submission chỉ vì manifest tồn tại. Bắt buộc kiểm tra:

- đủ 1.000 unique IDs và đúng source order;
- không answer rỗng;
- `retrieval:model_error == 0`;
- thống kê `generator:model_error`, abstention và verification failures;
- đọc mẫu câu thành công/thất bại;
- lưu code/config/data/model/output checksums.

Sự cố đã biết: một lần P100 incompatible vẫn tạo đủ 1.000 records nhưng 615 câu
suffix đều có `retrieval:model_error`. Batch đó không hợp lệ và đã bị loại; suffix
được chạy lại trên T4.
