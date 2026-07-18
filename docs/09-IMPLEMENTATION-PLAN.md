# 09. Implementation Plan

## 1. Development Order

Dự án được triển khai theo thứ tự:

```text
Documentation
→ Project Scaffold
→ Dataset Loader
→ Audit
→ Normalization
→ Cleaning
→ Legal Parsing
→ Chunking
→ BM25
→ Vector Retrieval
→ Hybrid RRF
→ Reranking
→ Graph Retrieval
→ Fixed RAG
→ Tool Wrappers
→ Agentic Workflow
→ Serving
→ Evaluation
→ Competition Adaptation
```

Không được code Agent trước các milestone retrieval.

---

## 2. Milestone 0 — Documentation Foundation

### Objectives

- đóng gói toàn bộ context;
- chốt architecture;
- chốt unified schema;
- chốt implementation order;
- chốt competition adaptation strategy.

### Outputs

- `AGENTS.md`;
- `README.md`;
- toàn bộ file trong `docs/`.

### Done When

- không có mâu thuẫn lớn;
- Codex tóm tắt đúng dự án;
- mọi quyết định hiện tại được ghi thành văn bản;
- không có implementation code.

---

## 3. Milestone 1 — Project Scaffold

**Status:** Completed

### Approved Foundation

- Python package: `legal_agentic_rag`;
- minimum Python: 3.11;
- layout: `src/`;
- schema/config: Pydantic v2;
- backend contracts: `typing.Protocol`;
- build backend: setuptools;
- logging: Python standard library;
- tests: pytest with small unified-schema fixtures.

Future milestone packages are not created empty. Tracing backend
contract is deferred until a real consumer exists.

Milestone được xác nhận bằng import, schema, configuration, contract,
logging và JSON round-trip tests. Không có concrete backend hoặc
business logic.

### Objectives

- tạo Python package;
- tạo configuration structure;
- tạo schemas;
- tạo interfaces;
- tạo test structure.

### Outputs

- package skeleton;
- configuration models;
- unified schema implementation;
- backend interfaces;
- basic import tests;
- schema tests.

### Not Included

- dataset download;
- cleaning;
- indexing;
- retrieval logic;
- model loading;
- Agent.

### Done When

- package import thành công;
- schemas serialize được;
- tests chạy thành công;
- không có fake production implementation.

---

## 4. Milestone 2 — Dataset Loader and Audit

### Objectives

- load metadata;
- load content;
- load relationships;
- hỗ trợ sample mode;
- audit dataset.

### Outputs

- AIO dataset loader;
- dataset adapter;
- audit service;
- JSON report;
- CSV error reports;
- fixtures;
- unit tests.

### Done When

- loader không load toàn bộ dataset trong unit tests;
- schema thực tế được ghi lại;
- join coverage được đo;
- invalid relationships được phát hiện;
- audit report có thể tái tạo.

---

## 5. Milestone 3 — Document Normalization

### Objectives

- map raw fields;
- chuẩn hóa ID;
- parse date;
- normalize effect status;
- join metadata và content.

### Outputs

- normalized documents;
- normalization warnings;
- normalized manifest;
- unit tests.

### Done When

- raw field không lan sang core;
- document ID deterministic;
- null values nhất quán;
- metadata thiếu không gây crash;
- invalid values được báo cáo.

---

## 6. Milestone 4 — HTML Cleaner

### Objectives

- chuyển HTML thành clean text;
- giữ legal structure markers;
- loại web noise.

### Outputs

- deterministic HTML cleaner;
- cleaning report;
- fixture HTML;
- unit tests.

### Done When

- script/style bị loại;
- Điều/Khoản được giữ;
- Unicode được bảo toàn;
- số và phủ định không bị mất;
- output deterministic.

---

## 7. Milestone 5 — Legal Structure Parser

### Objectives

- nhận diện Chương;
- nhận diện Mục;
- nhận diện Điều;
- nhận diện Khoản;
- nhận diện Điểm;
- tạo LegalBlock.

### Outputs

- structure parser;
- parsed blocks;
- parser diagnostics;
- unit tests.

### Done When

- parser xử lý được fixture chuẩn;
- parser không crash với văn bản không chuẩn;
- hierarchy được giữ;
- text coverage được đo.

---

## 8. Milestone 6 — Legal Chunker

### Objectives

- chunk theo Điều;
- fallback theo Khoản;
- token split khi cần;
- tạo deterministic chunk ID.

### Outputs

- legal chunks;
- chunk manifest;
- chunk validator;
- unit tests.

### Done When

- chunk ID duy nhất;
- chunk có document metadata;
- chunk truy ngược được;
- chunk không rỗng;
- chunk quá dài được xử lý.

---

## 9. Milestone 7 — BM25 Index

### Objectives

- build BM25 index;
- persist;
- reload;
- query.

### Outputs

- BM25 backend interface;
- BM25 builder;
- BM25 retriever;
- manifest;
- smoke tests.

### Done When

- index build thành công trên fixture;
- reload trả cùng kết quả;
- hit chứa đúng chunk metadata;
- query latency được đo.

---

## 10. Milestone 8 — Vector Index

### Objectives

- embedding interface;
- batch embedding;
- vector storage;
- dense retrieval.

### Outputs

- embedding provider interface;
- vector backend interface;
- index builder;
- dense retriever;
- manifest;
- tests.

### Done When

- chunk embedding thành công;
- dimension được kiểm tra;
- vector index reload được;
- search trả RetrievalHit;
- model name được ghi trong manifest.

---

## 11. Milestone 9 — Hybrid RRF

### Objectives

- chạy BM25 và dense;
- hợp nhất bằng RRF;
- deduplicate;
- giữ retrieval trace.

### Outputs

- RRF implementation;
- hybrid retriever;
- traceable contributions;
- tests.

### Done When

- raw score không bị cộng trực tiếp;
- rank contribution đúng;
- duplicate chunk được xử lý;
- kết quả deterministic.

---

## 12. Milestone 10 — Cross-Encoder Reranker

### Objectives

- rerank candidate set;
- giữ score;
- đo latency.

### Outputs

- reranker interface;
- pretrained backend;
- reranking service;
- tests.

### Done When

- reranker không chạy toàn corpus;
- output giữ metadata;
- candidate limit được kiểm tra;
- lỗi model được xử lý.

---

## 13. Milestone 11 — Graph Index and Retrieval

### Objectives

- normalize relationships;
- build document graph;
- seed expansion;
- hop limit.

### Outputs

- graph backend interface;
- graph builder;
- graph retriever;
- relationship mapping artifact;
- tests.

### Done When

- invalid edge bị loại;
- traversal có giới hạn;
- graph path được ghi;
- seed document truy xuất được related documents;
- graph không thay thế text retrieval.

---

## 14. Milestone 12 — Fixed End-to-End RAG

### Objectives

- context builder;
- answer generator interface;
- answer response;
- citation verifier;
- abstention.

### Outputs

- fixed RAG service;
- evidence builder;
- generation backend;
- rule-based verifier;
- integration tests.

### Done When

- question đi qua retrieval đến answer;
- answer chỉ dùng evidence;
- citation hợp lệ;
- thiếu evidence thì abstain;
- trace ID được trả về.

---

## 15. Milestone 13 — Tool Wrappers

### Objectives

Đóng gói các chức năng thành typed tools:

- BM25 search;
- dense search;
- hybrid search;
- rerank;
- graph search;
- context grading;
- answer generation;
- citation verification.

### Outputs

- tool contracts;
- tool registry;
- error handling;
- tool tests.

### Done When

- tool input/output đúng schema;
- tool không truy cập ngoài phạm vi;
- lỗi được chuẩn hóa;
- tool có mô tả rõ.

---

## 16. Milestone 14 — Agentic Workflow

### Objectives

- state graph;
- strategy selection;
- retry;
- query rewrite;
- stopping conditions;
- trace logging.

### Outputs

- AgentState implementation;
- workflow graph;
- router;
- retry logic;
- Agent tests.

### Done When

- max retry được enforce;
- Agent không build index;
- Agent chỉ gọi registered tools;
- workflow dừng đúng;
- trace có thể debug.

---

## 17. Milestone 15 — Serving

### Objectives

- API;
- optional UI;
- health check;
- query endpoint.

### Outputs

- FastAPI application;
- health endpoint;
- retrieval endpoint;
- answer endpoint;
- optional Gradio;
- API tests.

### Done When

- startup kiểm tra artifact;
- invalid request được xử lý;
- response đúng schema;
- trace ID được trả;
- secrets không bị lộ.

---

## 18. Milestone 16 — Evaluation

### Objectives

- retrieval evaluation framework;
- generation evaluation framework;
- latency benchmark;
- resource reporting.

### Outputs

- evaluator interfaces;
- benchmark runner;
- metric reports;
- error analysis artifacts.

### Retrieval Metrics

Khi có gold labels:

- Recall@k;
- MRR;
- NDCG@k;
- Precision@k.

### Generation Metrics

Khi có gold answers hoặc human labels:

- answer correctness;
- groundedness;
- citation precision;
- citation recall;
- unsupported claim rate.

---

## 19. Milestone 17 — Competition Adaptation

### Objectives

Tích hợp dữ liệu và yêu cầu chính thức của BTC.

### Possible Outputs

- competition corpus adapter;
- competition QA loader;
- train/dev/test loader;
- training pipeline;
- evaluator;
- submission writer;
- Docker package;
- resource configuration.

### Done When

- input đúng format BTC;
- output đúng format BTC;
- quy chế dữ liệu ngoài được tuân thủ;
- model/API constraints được tuân thủ;
- submission có thể tái tạo.

---

## 20. Milestone Execution Rules

Mỗi milestone phải:

1. đọc `AGENTS.md`;
2. đọc docs liên quan;
3. xác định phạm vi;
4. liệt kê file dự kiến thay đổi;
5. không mở rộng phạm vi;
6. viết tests;
7. chạy tests;
8. cập nhật docs nếu có decision mới;
9. báo rõ phần chưa làm;
10. không đánh dấu hoàn thành nếu test thất bại.
