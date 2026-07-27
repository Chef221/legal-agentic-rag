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

**Status:** Completed

Milestone đã được xác nhận bằng unit/integration tests không dùng network
và live smoke test sample 1 record cho cả ba config. Schema thực tế và
revision kiểm tra được ghi trong `docs/03-DATASET-AIO.md`.

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

**Status:** Completed

Implementation gồm AIO field projection, deterministic ID/null/date
normalization, conservative metadata-content join, structured issues,
normalized artifact manifest và local unit/integration tests.

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

**Status:** Completed

Implementation dùng `html.parser` của Python, exact-match noise policy,
Unicode NFC/whitespace normalization, typed cleaning result, upstream
manifest validation và local unit/integration tests. Không có parser cấu trúc
pháp lý trong milestone này.

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

**Status:** Completed

Implementation dùng deterministic line-based rules cho Phần/Chương/Mục/Tiểu
mục/Điều/Khoản/Điểm/Phụ lục và table rows. Output gồm non-overlapping
`LegalBlock`, per-document diagnostics, full text coverage, structured issues
và legal-block artifact manifest. Không có chunking trong milestone này.

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

**Status:** Completed

Implementation ưu tiên one-article chunks, group Khoản liên tiếp khi Điều dài,
và chỉ dùng overlapping Unicode token windows khi một legal unit vẫn vượt
giới hạn. Standalone blocks giữ preamble/unstructured text. Output gồm
validated `LegalChunk`, diagnostics, full block coverage và artifact manifest.

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

**Status:** Completed

Implementation dùng SQLite FTS5 reference backend qua `BM25Backend`, analyzer
Unicode bảo toàn dấu/số/từ phủ định, exact unified filters và deterministic
tie-breaking. Artifact gồm `index.sqlite3` cùng `manifest.json`, có checksum,
provenance, compatibility validation và không silently overwrite. Persist/reload
được xác nhận bằng unit và integration tests; không có dependency mới.

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

**Status:** Completed

Implementation dùng revision-pinned `intfloat/multilingual-e5-small` qua
`EmbeddingProvider`, passage/query prefixes và normalized 384-dimensional
embeddings. NumPy flat reference backend thực hiện exact cosine retrieval,
unified filters, persistence/reload, checksum và memory mapping. Model provider
đã được live-smoke trên CPU; unit/integration tests không gọi network.

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

**Status:** Completed

Implementation chạy BM25 và dense trên candidate pool, kiểm tra hai index cùng
legal-chunks source identity, deduplicate theo chunk ID và fuse bằng unweighted
RRF constant 60. Output giữ raw rank/score và contribution của từng nhánh,
deterministic tie-break, namespaced warnings và total latency. `FixedRetriever`
route BM25/dense/hybrid độc lập; unit/integration tests không dùng network hoặc
model nặng.

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

**Status:** Completed

Implementation thêm revision-pinned multilingual CrossEncoder phía sau
`Reranker`, lazy-load model, chấm bounded hybrid candidates và trả final top-k.
Reranking service giữ nguyên legal payload, artifact provenance cùng toàn bộ
BM25/dense/RRF trace, ghi thêm raw reranker score và total latency. Unit tests
dùng fixture model deterministic; integration test đi qua BM25 + dense + RRF +
reranker mà không cần network.

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

**Status:** Completed

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

### Implemented Baseline

- AIO relationship normalization với explicit mapping và structured issues;
- versioned relationship mapping artifact với checksum/no-overwrite;
- persisted directed `adjacency_json` reference backend;
- deterministic BFS 1 hop mặc định, tối đa 2 hop và relationship filter;
- hybrid text seeds, related-document chunk retrieval, bounded merge và final
  cross-encoder rerank;
- graph path/hop trace và fixed `graph` routing không cần Agent;
- unit/integration tests cho invalid edges, persistence/reload, traversal,
  artifact compatibility và end-to-end graph retrieval.

---

## 14. Milestone 12 — Fixed End-to-End RAG

**Status:** Completed

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

### Implemented Baseline

- bounded context builder giữ whole legal chunks và retrieval provenance;
- transparent structural context grader, không nhận là semantic grader;
- dependency-free extractive answer generator phía sau `AnswerGenerator`;
- exact rule-based citation verifier phía sau `CitationVerifier`;
- fixed retrieval-to-answer service với explicit abstention;
- fail-closed replacement khi generator trả citation không hợp lệ;
- typed context/generation/grading config và `ContextBuildResult`;
- unit/integration tests từ hybrid RRF + cross-encoder tới verified answer.

---

## 15. Milestone 13 — Tool Wrappers

**Status:** Completed

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

### Implemented Baseline

- closed enum và typed invocation/error/descriptor schemas;
- `TypedTool` Protocol không có generic base implementation;
- năm fixed retrieval wrappers;
- context grading, answer generation và citation verification wrappers;
- explicit eight-tool factory và deterministic closed registry;
- schema discovery, output-contract validation và sanitized error mapping;
- timeout-budget classification, payload-free logging và no hidden exceptions;
- unit/integration tests chạy retrieval → grade → generate → verify không Agent.

---

## 16. Milestone 14 — Agentic Workflow

**Status:** Completed

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

### Implemented Baseline

- dependency-free deterministic workflow phía sau `AgentWorkflow` Protocol;
- quality-first registered route plan `hybrid_rerank → graph → hybrid`;
- explicit requested-strategy priority và closed-registry filtering;
- conservative user-text-only query rewrite;
- bounded maximum of two retries and explicit terminal stop reasons;
- typed retrieval history, terminal state/result consistency validation;
- fail-closed abstention cho generation/citation/tool failure;
- payload-free invocation trace và workflow summary logging;
- unit/integration tests cho retry, stopping, tool restriction và error paths.

---

## 16.1 Runtime Assembly

**Status:** Completed

### Objectives

- ghép offline modules thành một reproducible artifact build;
- persist processed artifacts còn thiếu;
- load và validate complete online artifact set;
- tạo một application runtime sẵn sàng cho serving.

### Implemented Baseline

- `OfflineBuildRuntime` cho AIO audit-to-index pipeline;
- deterministic JSONL processed artifacts với manifest/checksum;
- configurable safe artifact layout và immutable preflight;
- BM25/vector/graph build và persistence trong cùng snapshot;
- `OnlineRuntimeFactory` fail-fast trên checksum, lineage và model identity;
- composition của fixed retrieval, tools và bounded Agent;
- Agent query rewrite được BM25/dense sử dụng thật;
- fixture integration test từ raw records tới verified answer sau reload.

### Limitations

- raw snapshot đang materialize trong memory;
- build chưa hỗ trợ resume hoặc transactional recovery;
- CLI chỉ nhận một file JSON typed config, chưa có environment composition.

---

## 17. Milestone 15 — Serving

**Status:** Completed

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

### Implemented Baseline

- FastAPI application factory với lifespan load đúng một `OnlineRuntime`;
- versioned endpoints `/api/v1/health`, `/api/v1/retrieve` và
  `/api/v1/answer`;
- bounded `LegalQuestionRequest`, NFC/whitespace normalization và UUID query ID;
- unified health/error schemas, error mapping fail-closed và không lộ backend
  detail;
- optional Gradio diagnostic UI mount tại `/ui`, dùng chung runtime;
- SQLite BM25 read được serialize cross-thread để Gradio worker dùng an toàn
  connection đã load trong FastAPI lifespan;
- explicit JSON config loader cùng `legal-rag-build` và `legal-rag-serve`;
- localhost default, configurable route/port/limits, OpenAPI docs tùy chọn;
- unit/integration tests cho lifecycle, schema, health, API errors và UI mount.

### Limitations

- chưa có authentication, rate limiting, CORS policy hoặc HTTPS termination;
- chưa có Docker/cloud deployment configuration;
- request pipeline đồng bộ và phù hợp baseline local, chưa benchmark tải đồng
  thời;
- Gradio chỉ dành cho chạy thử và quan sát, không phải production frontend.

---

## 18. Milestone 16 — Evaluation

**Status:** Completed

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

### Implemented Baseline

- backend-neutral `RetrievalEvaluator` và `GenerationEvaluator` Protocol;
- JSONL `EvaluationCase` dùng stable chunk/document IDs và graded relevance;
- Recall@k, Precision@k, MRR và NDCG@k bằng standard formulas;
- automatic exact match, abstention accuracy và citation precision/recall chỉ
  khi case có nhãn tương ứng;
- bounded runner trên immutable `OnlineRuntime`, tiếp tục theo case khi lỗi và
  hỗ trợ fail-fast cấu hình;
- benchmark SHA-256, artifact versions, code version và metric case counts;
- observed retrieval/generation latency, p50/p95, process CPU time và Python
  traced-memory;
- immutable `summary.json`, `cases.jsonl`, `errors.jsonl`;
- CLI `legal-rag-evaluate`;
- unit/integration tests cho metric arithmetic, document deduplication, missing
  labels, benchmark validation, no-overwrite và aggregation.

### Limitations

- chưa có official competition gold benchmark;
- không tự tạo synthetic QA làm gold;
- answer correctness, groundedness và unsupported claim rate chưa được công bố
  nếu thiếu gold/human labels;
- traced-memory chỉ đo Python allocations, không phải toàn bộ RSS/GPU memory.

---

## 19. Milestone 17 — Full Corpus Build and Baseline Validation

**Status:** Implementation complete; full-corpus execution pending capable
runtime

### Objectives

- tạo profile full AIO có revision và expected counts rõ ràng;
- từ chối sample hoặc unpinned input trước build;
- validate integrity, count và lineage của complete artifact set;
- tạo bằng chứng machine-readable trước khi gọi artifact ready.

### Implemented Baseline

- typed `BuildValidationConfig`;
- `configs/full-corpus.example.json` pin revision, bỏ sample limit và khai báo
  expected counts;
- `BuildValidationReport` cùng per-artifact `ArtifactValidationResult`;
- checksum/count validation cho JSONL, relationships, SQLite BM25, NumPy vector
  và adjacency graph;
- cross-artifact dataset identity và processing-hash lineage;
- immutable `build_validation.json` sau mỗi offline build;
- read-only `legal-rag-validate`;
- unit/integration tests cho policy, valid build và tampered payload.

### Milestone 17.1 Memory-Safe Execution

- full profile bật repeatable bounded source passes trên pinned revision;
- persist từng stage và giải phóng stage trước khỏi memory;
- giữ raw HTML trên disk nhưng bỏ reference khỏi downstream processing view;
- NumPy `float32` preallocation thay cho Python vector list;
- typed `build_state.json` khóa exact application config và code version;
- resume partial build sau normalized checkpoint;
- dependency validation giữa cleaned/blocks/chunks/indexes/graph;
- integration test cố ý làm vector stage thất bại rồi resume thành công;
- resume với config thay đổi bị từ chối.

### Milestone 17.2 Deterministic Resume Hash Hotfix

- dùng một canonical SHA-256 implementation cho application config và mọi
  processing config;
- sắp xếp mapping key và set/frozenset, giữ thứ tự list/tuple;
- nâng `OfflineBuildState` lên schema `1.1`;
- từ chối fail-closed state `1.0` vì không thể migrate digest cũ an toàn;
- kiểm thử hash của full application config qua nhiều process với
  `PYTHONHASHSEED` khác nhau;
- phát hành bản vá `0.19.1`, không thêm dependency.

### Milestone 17.3 Full-Corpus OOM Remediation

- measurement trên Colab 12 GiB xác nhận legacy parser bị kernel OOM-kill ở
  khoảng 10,8 GiB anonymous RSS;
- cleaned artifact được đọc bằng typed one-pass iterator;
- parser và chunker xử lý từng document, không giữ complete corpus lists;
- block/chunk JSONL được stage incremental và publish atomically;
- configurable document progress logging;
- BM25 disk-backed insert theo batch, không giữ corpus rows;
- vector embedding theo batch vào disk-backed NumPy memmap;
- persisted vector/chunk alignment giữ source-artifact order;
- regression tests cho one-pass behavior, output equivalence và failure safety;
- version `0.20.0`, không thêm dependency.

### Milestone 17.4 Resumable Vector Checkpoint Hotfix

- full run trên Colab Free xác nhận runtime termination làm mất toàn bộ vector
  progress dù memory đã bounded;
- deterministic `.vector.partial` workspace và typed checkpoint schema `1.0`;
- atomic committed chunk/vector offset theo configurable batch cadence;
- resume skip đã-commit chunks trước embedding;
- fail-closed checkpoint identity validation;
- atomic final directory publication;
- explicit compatible build-state upgrade `0.20.0 → 0.20.1`;
- regression tests cho interruption, resume, no re-embedding và incompatible
  checkpoint;
- version `0.20.1`, không thêm dependency.

### Done When

- full AIO build chạy xong trong một artifact root mới;
- persisted report có `is_full_corpus = true`;
- persisted report có `is_valid = true`;
- online runtime load được artifact set;
- smoke retrieval/answer chạy được trên full index;
- artifact và dataset lớn không bị commit.

### Current Limitation

Normalizer vẫn cần disk-derived ID indexes và cleaner hiện materialize stage
output trước khi persist, nhưng hai stage này đã hoàn thành trên Colab 12 GiB
trong full-corpus measurement. Parser/chunker/BM25/vector build đã chuyển sang
bounded execution sau OOM measurement. Milestone vẫn chưa được đánh dấu
Completed trước khi version `0.20.1` tạo được full validation report thật và
online smoke test load được artifact set.

---

### Milestone 17.5 Memory-bounded Online Vector Loader

- full-corpus validation đã hoàn thành với 1.278.201 chunks và vector shape
  `(1_278_201, 384)`;
- Colab 12 GiB đo được online server OOM-kill (`-9`) do materialize toàn bộ
  vector `chunks.jsonl` thành Pydantic objects;
- version `0.20.2` dùng validated disk-backed JSONL chunk store với byte offsets;
- compact inverted postings hỗ trợ đúng unified filters mà không giữ full chunk;
- vector validation và exact cosine scoring chạy theo configurable batches;
- top-k chỉ materialize final chunks và giữ deterministic chunk-ID tie-break;
- checksum/metadata/vector validation có progress logs;
- artifact format hiện có được reuse, không rebuild vector;
- build state `0.20.0`/`0.20.1` được nâng có kiểm soát lên `0.20.2`;
- unit/integration regression tests không dùng full corpus hoặc network;
- không thêm dependency.

Full-corpus online smoke và UI smoke phải chạy lại trên artifact thật trước khi
đánh dấu toàn bộ Milestone 17 Completed.

---

### Milestone 17.6 Full-corpus Query and Startup Performance Hotfix

- version `0.20.3` dùng FTS5 `rank` cùng bounded corpus-aware BM25 query planner;
- typed BM25 runtime limit giữ số và semantic modifier pháp lý;
- validated-report startup bỏ các checksum/integrity scan đã hoàn thành trước đó
  nhưng vẫn fail closed khi report hoặc current manifest lệch;
- embedding model không load chỉ để đọc configured dimension;
- startup stage latency được log riêng;
- artifact format hiện có được reuse, không preprocessing hoặc embedding lại;
- không thêm dependency.

Full-corpus latency và startup time phải được đo lại trên Colab artifact thật
trước khi chốt performance target.

---

## 20. Milestone 18 — Model-backed Answer Generator

**Status:** Implementation complete; model benchmark pending GPU and labeled
evaluation data

### Objectives

- sinh câu trả lời tiếng Việt thay vì chỉ trình bày evidence nguyên văn;
- giữ generator backend/model-neutral;
- từ chối citation/model output không truy ngược selected evidence;
- giữ UI hoạt động khi chưa có model server.

### Implemented Baseline

- `ChatModelProvider` Protocol;
- dependency-free OpenAI-compatible provider;
- typed backend/endpoint/model/revision/timeout/output-token configuration;
- API key lookup bằng tên environment variable;
- evidence-only Vietnamese prompt với untrusted-content boundary;
- strict `ModelAnswerDraft` JSON parsing;
- evidence-ID allowlist và required `[E#]` markers;
- deterministic system-built `Citation`;
- explicit model abstention và existing citation verifier;
- extractive default/fallback mode;
- unit tests không network cho prompt, parser, config, transport và failures.

### Done When

- model mode được bật bằng explicit local config;
- valid completion trở thành verified `AnswerResponse`;
- invented evidence ID, missing marker và invalid JSON đều fail closed;
- timeout/HTTP/model-envelope errors được phân loại;
- default test suite không gọi model thật;
- docs, schema và design decision thống nhất.

### Current Limitation

M18 chưa chọn hoặc benchmark model cuối cùng, chưa fine-tune và chưa
semantic-verify support của từng claim. OpenAI-compatible endpoint phải hỗ trợ
JSON response mode. Chất lượng thực tế chỉ được kết luận sau benchmark có nhãn
trên GPU; lựa chọn model không bị giới hạn bởi CPU của máy phát triển.

---

## 21. Future — Competition Adaptation

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

## 22. Milestone Execution Rules

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
