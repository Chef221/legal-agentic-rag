# Vietnamese Legal Agentic RAG

Hệ thống Agentic RAG cho bài toán trả lời câu hỏi pháp luật Việt Nam.

## Current Status

Dự án đã hoàn thành implementation cho:

`Milestone 18 — Model-backed Answer Generator`

Full-corpus execution chưa được tuyên bố hoàn thành cho tới khi có
`build_validation.json` thật với `is_full_corpus = true` và `is_valid = true`.

M18 bổ sung model-backed grounded generation qua endpoint OpenAI-compatible
hoặc Hugging Face Transformers chạy local. `extractive` vẫn là backend mặc
định để local UI chạy không cần model. Model, revision, device, endpoint và
secret đều đi qua configuration; core không khóa vào một model hoặc nhà cung
cấp cụ thể.

Hệ thống có composition root để build toàn bộ AIO artifacts, online factory để
reload BM25, vector, graph, reranker, tools và Agent, cùng FastAPI/UI để
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

Để chạy model local trên GPU, dùng backend `transformers`:

```json
{
  "backend": "transformers",
  "endpoint_url": null,
  "api_key_env": null,
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "model_revision": "a1d308dfcc03e09da285d49d912439a655a571e8",
  "device": "cuda",
  "torch_dtype": "float16",
  "local_files_only": false,
  "max_context_tokens": 3072,
  "max_evidence": 3,
  "max_input_tokens": 8192,
  "temperature": 0.0,
  "max_output_tokens": 512,
  "max_structured_output_retries": 1,
  "timeout_seconds": 180.0
}
```

Đây là candidate tham chiếu vừa bộ nhớ GPU 16 GiB, không phải model production
đã được chốt. Provider lazy-load weights ở câu hỏi đầu tiên, không log prompt
hoặc nội dung evidence và không cắt ngầm legal text khi prompt vượt giới hạn.

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

Từ phiên bản `0.20.3`, full-corpus BM25 dùng bounded corpus-aware query planning
và FTS5 `rank`. Artifact đã có không cần rebuild. Sau khi
`legal-rag-validate` trả `is_valid = true`, cấu hình sau cho phép server tái sử
dụng validation report thay vì hash/integrity-scan lại hàng GB mỗi lần:

```json
{
  "online": {
    "startup_validation": {"mode": "validated_report"},
    "bm25_runtime": {
      "max_query_terms": 8,
      "max_document_frequency_ratio": 0.25
    }
  }
}
```

Model embedding chỉ lazy-load khi dense retrieval được gọi; BM25 startup không
phải chờ tải model weights.

Từ version `0.20.4`, chuẩn bị vector serving metadata đúng một lần:

```powershell
legal-rag-prepare-serving --config configs/full-corpus.local.json
```

Lệnh này chỉ scan `vector/chunks.jsonl` hiện có để tạo
`vector_serving/metadata.sqlite3`. Nó không đọc raw dataset, không embedding lại
và không sửa `vectors.npy`. Khi `online.vector_runtime.require_serving_metadata`
là `true`, server từ chối fallback về JSONL scan chậm.

Từ version `0.20.5`, `/ui` là diagnostic HTML dùng same-origin HTTP tới
`/api/v1/answer`. UI không còn dùng Gradio queue/SSE nên hoạt động qua Colab
port proxy mà không tạo runtime thứ hai.

Từ version `0.20.6`, full-corpus GPU profile có thể đặt
`online.vector_runtime.search_device = "cuda"`. Runtime chuyển matrix float32
hiện có lên GPU đúng một lần và giữ exact cosine search; không re-embed hoặc đổi
vector artifact. `cpu` vẫn là mặc định và CUDA được yêu cầu sẽ fail closed nếu
không khả dụng.

Full-corpus Colab validation trên 1.278.201 vector x 384 ghi nhận query ấm:
22,2 ms exact vector search, 35,6 ms dense retrieval, 398 ms reranking và khoảng
2,16 giây end-to-end. Query đầu tiên mất khoảng 31,47 giây do lazy-load embedding
và cross-encoder model.

Từ version `0.20.7`, generator local dùng Transformers qua cùng
`ChatModelProvider` với backend endpoint. Dependency `transformers` được khai báo
trực tiếp vì source code gọi API này; không thêm LLM SDK, LangChain hoặc
LangGraph.

Từ version `0.20.8`, model output có thể chứa một preamble/code fence vô hại
trước JSON. Parser chỉ lấy JSON object rồi vẫn áp dụng strict schema,
evidence-ID allowlist và marker verification. Nếu draft đầu tiên sai format,
generator được phép yêu cầu model sửa đúng một lần; raw completion và legal
content không được ghi vào log.

Từ version `0.20.9`, marker `[E#]` hiển thị sát nhận định là thứ tự citation
chuẩn. Danh sách `cited_evidence_ids` dư thừa của model được chuẩn hóa theo các
marker đã kiểm tra allowlist. Marker hoặc declared ID không tồn tại trong
selected evidence vẫn bị từ chối.

Từ version `0.20.10`, khi model trả valid declared evidence IDs nhưng bỏ marker
inline, hệ thống hiển thị deterministic các marker đã kiểm tra ở cuối answer.
Combined form như `[E1, E2]` cũng được nhận diện. Hệ thống không tự suy ra hoặc
tạo evidence ID mới.

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
