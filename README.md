# Vietnamese Legal Agentic RAG

Hệ thống Agentic RAG cho bài toán trả lời câu hỏi pháp luật Việt Nam.

## Current Status

Dự án đã hoàn thành:

`Milestone 16 — Evaluation`

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
legal-rag-serve --config configs/baseline.local.json
```

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
