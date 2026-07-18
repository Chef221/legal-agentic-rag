# Vietnamese Legal Agentic RAG

Hệ thống Agentic RAG cho bài toán trả lời câu hỏi pháp luật Việt Nam.

## Current Status

Dự án đã hoàn thành:

`Milestone 1 — Project Scaffold`

Hiện repository có unified schemas, backend contracts, typed
configuration, exception taxonomy, logging foundation và test structure.

Chưa triển khai offline data pipeline hoặc business logic.

Chưa triển khai Agent.

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
