# Vietnamese Legal Agentic RAG

Hệ thống Agentic RAG cho bài toán trả lời câu hỏi pháp luật Việt Nam.

## Current Status

Dự án đã hoàn thành implementation cho:

`Milestone 18 — Model-backed Answer Generator`

Full-corpus execution chưa được tuyên bố hoàn thành cho tới khi có
`build_validation.json` thật với `is_full_corpus = true` và `is_valid = true`.

M18 bổ sung model-backed grounded generation qua endpoint OpenAI-compatible.
`extractive` vẫn là backend mặc định để local UI chạy không cần model service.
Model, revision, endpoint và secret đều đi qua configuration; core không khóa
vào một model hoặc nhà cung cấp cụ thể.

Hệ thống có composition root để build toàn bộ AIO artifacts, online factory để
reload BM25, vector, graph, reranker, tools và Agent, cùng FastAPI/Gradio để
chạy thử baseline. Startup kiểm tra checksum, dataset lineage và embedding
identity trước khi nhận query.

Evaluation framework hỗ trợ labeled JSONL benchmark, Recall/Precision/MRR/NDCG,
available generation metrics, latency/resource summary và error analysis.

## Local Baseline

```powershell
python -m pip install -e ".[dev]"
Copy-Item configs/baseline.example.json configs/baseline.local.json
# Chỉnh configs/baseline.local.json nếu cần.
legal-rag-build --config configs/baseline.local.json
legal-rag-validate --config configs/baseline.local.json
legal-rag-serve --config configs/baseline.local.json
```

Để bật model-backed generation, sửa `online.generation` trong config local:

```json
{
  "backend": "openai_compatible",
  "endpoint_url": "http://127.0.0.1:8001/v1/chat/completions",
  "api_key_env": null,
  "model_name": "<model-name>",
  "model_revision": "<pinned-revision>",
  "temperature": 0.0,
  "max_output_tokens": 1024
}
```

Nếu endpoint cần secret, chỉ đặt tên biến môi trường vào `api_key_env`; không
đặt API key trực tiếp trong file config.

Profile build toàn bộ AIO dành cho máy GPU/RAM phù hợp:

```powershell
Copy-Item configs/full-corpus.example.json configs/full-corpus.local.json
# Chọn artifact root mới và kiểm tra device trước khi chạy.
legal-rag-build --config configs/full-corpus.local.json
legal-rag-validate --config configs/full-corpus.local.json
```

Không commit corpus, model output hoặc artifact sinh ra.

Full profile dùng pinned multi-pass loading, giải phóng memory giữa các stage và
resume partial build sau normalized checkpoint. Resume chỉ hoạt động khi config
và code version tương thích; runtime không tự xóa artifact lỗi.

Từ phiên bản `0.19.1`, config hash được canonical hóa để ổn định giữa các
process Python. Partial state schema `1.0` từ bản cũ phải giữ nguyên để chẩn
đoán và chạy lại trong một artifact root mới; không sửa hoặc xóa state cũ.

Từ phiên bản `0.20.0`, parser/chunker xử lý từng document; block/chunk artifact
được ghi incremental, BM25 dùng disk-backed batched inserts và vector embedding
ghi theo batch vào NumPy memmap. Bản này sửa OOM đã đo ở legacy parser trên
runtime 12 GiB. Build `0.19.x` phải dùng artifact root khác khi chạy `0.20.0`.

Từ phiên bản `0.20.1`, vector build ghi checkpoint bền vững vào
`.vector.partial` theo batch và chỉ embedding phần chưa commit khi Colab
disconnect. Build state `0.20.0` được nâng có kiểm soát lên `0.20.1`; các
transition code/config khác vẫn bị từ chối.

Từ phiên bản `0.20.2`, online vector loader không còn giữ 1.278.201
`LegalChunk` objects trong RAM. Loader giữ compact byte offsets/filter postings,
validate vector và score exact cosine theo batch, rồi chỉ đọc full metadata cho
top-k hits. Artifact vector `0.20.1` hiện có được dùng lại, không cần rebuild.
Tiến độ startup xuất hiện trong log; các giới hạn nằm ở
`online.vector_runtime`.

Chạy benchmark có nhãn:

```powershell
legal-rag-evaluate `
  --config configs/baseline.local.json `
  --benchmark path/to/benchmark.jsonl `
  --output reports/evaluation-run
```

Sau khi server sẵn sàng:

- UI: `http://127.0.0.1:8000/ui`
- API docs: `http://127.0.0.1:8000/docs`
- health: `http://127.0.0.1:8000/api/v1/health`
- retrieval: `POST /api/v1/retrieve`
- answer: `POST /api/v1/answer`

UI hiện là giao diện chẩn đoán local, chưa có authentication hoặc cấu hình
deployment production.

## Current Dataset

Hugging Face:

`th1nhng0/vietnamese-legal-documents`

Dataset được dùng cho:

- legal corpus;
- legal metadata;
- document relationships;
- BM25 retrieval;
- dense retrieval;
- legal graph retrieval.

## High-Level Architecture

```text
Offline phase
Dataset
→ Audit
→ Normalize
→ Clean HTML
→ Parse legal structure
→ Chunk
→ Build indexes

Online phase
Question
→ Retrieval
→ Fusion
→ Rerank
→ Context grading
→ Answer generation
→ Citation verification
