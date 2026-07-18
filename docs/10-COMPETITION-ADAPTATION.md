# 10. Competition Adaptation

## 1. Purpose

Hệ thống được xây trước khi Ban tổ chức công bố dữ liệu và quy chế đầy đủ.

File này mô tả những phần phải thay đổi trong từng kịch bản.

Mục tiêu:

- không viết lại toàn bộ hệ thống;
- chỉ thay adapter hoặc backend cần thiết;
- giữ unified schema;
- giữ retrieval interfaces;
- giữ Agent workflow khi có thể.

---

## 2. Scenario A — BTC Provides a New Corpus

Ví dụ BTC cung cấp:

- văn bản pháp luật;
- điều luật;
- passage corpus;
- metadata riêng;
- document IDs riêng.

### Required Changes

- thêm `CompetitionCorpusAdapter`;
- map raw schema về `LegalDocument`;
- map cấu trúc về `LegalChunk`;
- điều chỉnh cleaner nếu format khác HTML;
- điều chỉnh parser nếu dữ liệu đã tách sẵn Điều/Khoản;
- rebuild BM25 index;
- re-embed corpus;
- rebuild vector index;
- rebuild graph nếu có relationships;
- tạo artifact manifest mới.

### Components Preserved

- unified schema;
- BM25 interface;
- vector interface;
- RRF;
- reranker interface;
- context builder;
- answer generator interface;
- verifier interface;
- Agent state;
- API contracts.

---

## 3. Scenario B — BTC Provides Training Questions and Relevant Evidence

Ví dụ:

```json
{
  "qid": "Q001",
  "question": "...",
  "relevant_document_ids": ["D10"],
  "relevant_article_ids": ["D10-A15"]
}
```

### Required Additions

- `CompetitionQALoader`;
- supervision schema;
- train/dev split;
- retrieval evaluator;
- hard-negative mining;
- retriever fine-tuning;
- reranker fine-tuning;
- experiment tracking.

### Required Checks

- ground truth ở cấp văn bản hay chunk;
- một câu có nhiều evidence hay không;
- ID có ánh xạ tới corpus không;
- duplicate question;
- leakage giữa train và dev;
- metric chính thức.

### After Retriever Fine-Tuning

- re-embed corpus;
- rebuild vector index;
- cập nhật model manifest;
- benchmark lại toàn bộ retrieval pipeline.

---

## 4. Scenario C — BTC Provides Questions, Evidence and Gold Answers

Ví dụ:

```json
{
  "qid": "Q001",
  "question": "...",
  "relevant_articles": ["..."],
  "answer": "..."
}
```

### Required Additions

- answer loader;
- generator evaluator;
- citation evaluator;
- groundedness evaluator;
- answer correctness evaluator;
- prompt optimization;
- optional generator fine-tuning;
- evidence grader training nếu phù hợp.

### Additional Metrics

- answer correctness;
- semantic similarity;
- groundedness;
- citation precision;
- citation recall;
- unsupported claim rate;
- refusal correctness.

---

## 5. Scenario D — BTC Provides Test Questions Only

Ví dụ:

```text
qid,question
```

### Required Additions

- `CompetitionTestLoader`;
- batch inference runner;
- output formatter;
- submission writer;
- validation script.

### Core Changes

Không cần thay core architecture nếu corpus và model được phép sử dụng.

---

## 6. Scenario E — BTC Provides a Corpus but No Training Labels

### Strategy

- dùng corpus BTC làm nguồn retrieval;
- dùng pretrained BM25, dense và reranker;
- rebuild toàn bộ indexes;
- tự xây internal validation nếu quy chế cho phép;
- không fine-tune nếu không có supervision phù hợp.

AIO corpus có thể được giữ cho nghiên cứu riêng nhưng không dùng trong
competition run nếu quy chế cấm external data.

---

## 7. Scenario F — BTC Forbids External Data

### Required Changes

- disable AIO corpus trong competition configuration;
- ingest corpus BTC;
- rebuild all indexes;
- không dùng artifact tạo từ AIO;
- kiểm tra model pretrained có được phép không;
- ghi rõ provenance của mọi artifact.

Core interfaces được giữ nguyên.

---

## 8. Scenario G — BTC Allows External Data

### Possible Strategy

- dùng corpus BTC làm nguồn ưu tiên;
- dùng AIO làm nguồn bổ sung nếu hợp lệ;
- gắn `source_dataset`;
- ưu tiên source chính thức;
- tránh duplicate documents;
- xây mapping giữa document IDs;
- đánh giá riêng tác động của external data.

Không trộn corpus nếu chưa xử lý deduplication và provenance.

---

## 9. Scenario H — BTC Forbids External LLM APIs

### Required Changes

Thay backend cho:

- answer generator;
- query rewriter;
- context grader;
- citation semantic verifier.

Interfaces được giữ nguyên.

Cần kiểm tra:

- model local;
- VRAM;
- quantization;
- latency;
- context length;
- licensing;
- offline availability.

---

## 10. Scenario I — BTC Forbids Internet Access

### Required Changes

- preload model;
- preload tokenizer;
- preload all artifacts;
- không gọi Hugging Face Hub khi runtime;
- không gọi external API;
- không web search;
- không remote database;
- đóng gói dependency đầy đủ.

---

## 11. Scenario J — BTC Requires File Submission

### Required Additions

- deterministic batch inference;
- submission formatter;
- schema validator;
- ordering validator;
- duplicate qid check;
- missing qid check;
- final checksum;
- reproducible config.

Output có thể là:

- CSV;
- JSON;
- JSONL;
- Parquet.

Format cụ thể phải theo quy chế.

---

## 12. Scenario K — BTC Requires Docker Submission

### Required Additions

- Dockerfile;
- dependency lock;
- local model paths;
- artifact packaging;
- entrypoint;
- health check;
- memory limits;
- CPU/GPU configuration;
- timeout handling;
- deterministic startup;
- no-network mode nếu cần.

---

## 13. Scenario L — BTC Requires API Submission

### Required Additions

- request schema;
- response schema;
- health endpoint;
- readiness endpoint;
- concurrency control;
- timeout;
- error contract;
- model preload;
- logging;
- trace ID;
- resource monitoring.

---

## 14. Scenario M — BTC Provides a Training Environment

Nếu BTC cung cấp notebook hoặc server:

- thêm environment-specific config;
- không hard-code local path;
- hỗ trợ download hoặc mount dataset;
- ghi dependency versions;
- hỗ trợ resume;
- giữ artifacts trong output directory quy định.

---

## 15. Ground-Truth Granularity

Khi quy chế được công bố, phải xác định ground truth ở cấp:

- văn bản;
- Chương;
- Mục;
- Điều;
- Khoản;
- passage;
- answer span;
- long-form answer.

Điều này ảnh hưởng trực tiếp tới:

- chunking;
- retrieval output;
- evaluation;
- submission format;
- citation format.

Không được mặc định ground truth ở cấp Điều.

---

## 16. Official Metric Impact

### Nếu metric là Recall@k

Ưu tiên:

- candidate recall;
- multi-evidence coverage;
- top-k selection.

### Nếu metric là MRR hoặc NDCG

Ưu tiên:

- ranking;
- reranking;
- score calibration;
- position của first relevant result.

### Nếu metric là answer accuracy

Ưu tiên:

- context quality;
- generation;
- answer formatting.

### Nếu metric là exact match

Cần tuân thủ chặt:

- output normalization;
- answer format;
- punctuation;
- label mapping.

### Nếu metric kết hợp retrieval và generation

Phải tối ưu end-to-end nhưng vẫn đánh giá từng module riêng.

---

## 17. Required Questions When Rules Are Released

1. BTC có cung cấp train data không?
2. BTC có cung cấp development set không?
3. BTC có cung cấp corpus riêng không?
4. Test có public và private split không?
5. Ground truth ở cấp nào?
6. Một câu hỏi có thể có nhiều evidence không?
7. Output là article IDs hay answer text?
8. Có yêu cầu citation không?
9. Metric chính thức là gì?
10. Có cho dùng external data không?
11. Có cho dùng pretrained model không?
12. Có cho fine-tune bằng dữ liệu ngoài không?
13. Có cho dùng Internet khi inference không?
14. Có cho dùng commercial LLM API không?
15. Có giới hạn model size không?
16. Có giới hạn GPU, RAM, disk không?
17. Có giới hạn latency không?
18. Nộp CSV, code, Docker hay API?
19. Có cần public source code không?
20. Có cần giải thích hoặc retrieval trace không?

---

## 18. Adaptation Principle

Kiến trúc thích ứng:

```text
Raw Competition Data
→ Competition Adapter
→ Unified Schema
→ Existing Core Pipeline
→ Output Adapter
→ Official Submission
```

Nếu corpus thay đổi:

```text
Competition Corpus
→ Cleaner and Chunker
→ Rebuild Indexes
→ Existing Retrieval Interfaces
```

Nếu có supervision:

```text
Competition Train Data
→ Training Adapter
→ Fine-Tune
→ Rebuild Relevant Artifacts
→ Evaluate
```

Phần cần thay đổi chủ yếu nằm ở:

- input adapter;
- corpus processing;
- model backend;
- evaluator;
- output adapter.

Agentic RAG core không được viết gắn cứng với dataset AIO.