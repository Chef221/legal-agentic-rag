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

Milestone này là lịch sử của baseline cũ và đã bị D068/M25 supersede. Source,
fixture, dependency và tài liệu dataset-specific tương ứng không còn trong
active tree.

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

**Status:** Completed

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
bounded execution sau OOM measurement. Full-corpus run sau đó đã tạo report
thật với `is_full_corpus = true`, `is_valid = true`, load được online artifact
set và hoàn thành retrieval/answer smoke.

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

Full-corpus online smoke và UI smoke đã chạy trên artifact thật; lỗi OOM loader
không còn tái hiện sau memory-bounded implementation.

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

### Milestone 17.7 Persisted Vector Serving Metadata

- measured vector metadata startup bottleneck: 141,8 giây trên 1.278.201 chunks;
- version `0.20.4` thêm immutable SQLite vector metadata sidecar;
- one-time CLI preparation từ vector artifact hiện có;
- exact source/checksum/count compatibility và no-overwrite policy;
- read-only offset/chunk-ID/filter lookup khi server startup/search;
- optional fallback và production fail-closed config;
- không re-embed, không rebuild index và không thêm dependency.

Full-corpus sidecar preparation và startup reload phải được đo thật trên Colab
trước khi chuyển sang graph disk-backed loader.

---

### Milestone 17.8 Proxy-safe Diagnostic UI

- measured Colab failure: health `200`, không OOM nhưng Gradio submit không tới
  server;
- version `0.20.5` thay Gradio queue/SSE bằng same-origin HTTP UI;
- UI gọi public `/api/v1/answer`, không truy cập runtime/backend trực tiếp;
- local và Colab port proxy dùng cùng một trang, không cần public URL config;
- response được hiển thị bằng text-safe DOM APIs;
- bỏ runtime dependency Gradio, không thêm dependency mới;
- integration test xác nhận UI dùng public endpoint và chỉ load một runtime.

---

### Milestone 17.9 GPU-resident Exact Dense Search

- measured full-corpus dense latency: 52,28 giây cho 1.278.201 x 384 float32;
- version `0.20.6` thêm optional PyTorch CUDA scorer trong NumPy adapter;
- vector matrix được chuyển theo bounded batches và giữ resident trên GPU;
- unfiltered và filtered exact cosine đều giữ candidate order/output contract;
- explicit CUDA fail-closed, CPU default không đổi;
- reuse `vectors.npy`, sidecar và manifests hiện có; không re-embed/rebuild;
- unit test so sánh NumPy/Torch CPU, CUDA failure và optional live CUDA smoke.

Full-corpus CUDA validation đã pass: warm vector search 22,2 ms, dense retrieval
35,6 ms, reranker 398 ms và Agent workflow khoảng 2,04 giây. Cold query khoảng
31,47 giây do lazy model initialization, không phải vector scoring.

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
- local Hugging Face Transformers provider với lazy pinned-model loading;
- typed backend/endpoint/model/revision/timeout/output-token configuration;
- typed local device, dtype, cache policy và input-token bound;
- API key lookup bằng tên environment variable;
- evidence-only Vietnamese prompt với untrusted-content boundary;
- strict `ModelAnswerDraft` JSON parsing;
- evidence-ID allowlist và required `[E#]` markers;
- deterministic system-built `Citation`;
- explicit model abstention và existing citation verifier;
- extractive default/fallback mode;
- unit tests không network/model download cho prompt, parser, config, local
  inference boundary, transport và failures.

### Done When

- model mode được bật bằng explicit local config;
- valid completion trở thành verified `AnswerResponse`;
- invented evidence ID, missing marker và invalid JSON đều fail closed;
- timeout/HTTP/model-envelope errors được phân loại;
- default test suite không gọi model thật;
- docs, schema và design decision thống nhất.

### Current Limitation

Local Transformers provider đã hoàn thành full-corpus smoke với
Qwen2.5-3B-Instruct: model-backed answer, system-built citation và verifier đều
thành công. M18 vẫn chưa chọn model cuối cùng, chưa fine-tune, chưa
semantic-verify support của từng claim và chưa có benchmark chất lượng có nhãn.
OpenAI-compatible endpoint phải hỗ trợ JSON response mode. Chất lượng thực tế
chỉ được kết luận sau benchmark có nhãn trên GPU; lựa chọn model không bị giới
hạn bởi CPU của máy phát triển.

Full-corpus smoke đầu tiên của `0.20.7` xác nhận model load và inference thành
công nhưng completion bị strict draft validation từ chối. Version `0.20.8`
thêm safe JSON-object extraction, prompt contract rõ hơn, categorized
non-content logging và tối đa một structured-output correction attempt. Các
kiểm tra unknown evidence ID, marker mismatch, citation construction và
abstention không được nới lỏng.

Smoke `0.20.8` xác định cả hai model attempts đều tạo JSON hợp lệ nhưng redundant
`cited_evidence_ids` không khớp marker order/usage trong answer. Version
`0.20.9` chuẩn hóa citation IDs từ visible markers sau khi kiểm tra cả declared
IDs và markers trong selected-evidence allowlist. Unknown marker, answer không
có marker và insufficient answer có marker tiếp tục fail closed.

Smoke `0.20.9` cho thấy Qwen tiếp tục bỏ exact bracket marker ở cả correction
attempt dù structured list hợp lệ. Version `0.20.10` nhận combined bracket IDs
và render declared allowlisted IDs thành visible markers khi answer thiếu
marker. Citation metadata vẫn chỉ dựng từ selected Evidence.

### Milestone 18.1 — Legal-context Cross-encoder Input

- measured failure: answer đúng nhưng top citation thuộc văn bản chuyên biệt về
  người giúp việc gia đình;
- root cause: sparse/dense index dùng title-aware `search_text`, còn
  cross-encoder chỉ nhận chunk body nên mất document scope;
- version `0.20.11` thêm legal-context input từ named unified metadata;
- configurable `legal_context|text_only` để benchmark, mặc định legal context;
- không cộng heuristic score, không đổi public hit schema và không rebuild
  BM25/vector artifacts;
- unit/integration tests kiểm tra metadata selection, raw-field isolation,
  text-only comparison và payload/trace preservation.

---

## 21. Milestone 19 — Query Understanding and Multi-query Retrieval

**Status:** Completed

### Objectives

- hiểu tín hiệu pháp lý xuất hiện trực tiếp trong câu hỏi;
- tạo nhiều query forms có kiểm soát mà không thêm kiến thức;
- định tuyến Agent theo intent trong giới hạn hiện có;
- tăng recall bằng multi-query hybrid và giữ trace;
- không phụ thuộc dataset AIO hoặc format cuộc thi.

### Implemented Baseline

- typed `QueryAnalysis`, `QueryVariant` và `QueryVariantContribution`;
- deterministic extraction cho document number, Điều/Khoản/Điểm, năm, scope,
  relationship cues và conservative intent;
- normalized, framing-stripped và legal-reference variants, mặc định tối đa 3;
- runtime recompute analysis trước fixed retrieval hoặc Agent;
- BM25/dense execution theo từng variant và unweighted multi-branch RRF;
- aggregate sparse/dense contributions cùng per-variant trace;
- adaptive graph-first route cho relationship intent;
- BM25 retry route cho explicit-reference/quantitative intent;
- retry rewriter dùng user-derived variants trước original/normalized forms;
- legacy single-query behavior khi feature bị tắt hoặc chỉ có một variant;
- unit/integration tests không network, model hoặc dataset thật.

### Limitations

- đây là rule-based signal extraction, chưa semantic-grade intent;
- không dùng synonym expansion hoặc LLM query rewrite;
- generic natural-language question thường chỉ có một safe variant;
- semantic applicability và claim-level verification thuộc milestone tiếp theo;
- chưa tune variant policy hoặc RRF bằng official labeled data.

### Done When

- query analysis và variants serialize qua unified schema;
- explicit reference extraction giữ nguyên user text;
- multi-query RRF không cộng raw backend scores;
- conflicting duplicate payload fail closed;
- route vẫn tôn trọng requested strategy và `max_retry = 2`;
- artifacts hiện có load lại mà không rebuild;
- full local test suite pass.

---

## 22. Milestone 20 — Evidence Applicability and Context Selection

**Status:** Completed

### Objectives

- chọn context tốt hơn sau retrieval/reranking mà không phụ thuộc dataset AIO;
- ưu tiên evidence khớp explicit legal reference của người dùng;
- giữ nguyên whole legal chunks và token budget;
- giải thích được mọi quyết định chọn/bỏ evidence;
- fail closed khi explicit reference không được selected evidence hỗ trợ.

### Implemented Baseline

- typed `EvidenceApplicability`, `EvidenceSelectionReason` và
  `EvidenceSelectionTrace`;
- deterministic selector dùng source rank, reference match, lexical overlap và
  configured inactive-status penalty;
- exact chunk deduplication trước selection;
- whole-chunk count/token budgeting với reason riêng;
- selection trace được giữ trong `ContextBuildResult` và từng selected
  `Evidence`;
- explicit document/article coverage trong rule-based context grader;
- applicability score và status counts cho debugging;
- `legal_applicability_interpreted = false` để không overclaim;
- typed bounded `online.evidence_selection` config;
- không thêm dependency, model, dataset hoặc artifact format;
- unit/integration tests không network/model/full corpus.

### Limitations

- lexical overlap không phải semantic relevance model;
- effect status chỉ là dataset snapshot và explicit configured label;
- không xác định luật chung/chuyên ngành hoặc văn bản ưu tiên;
- không giải quyết xung đột, sửa đổi, thay thế hoặc hiệu lực theo thời điểm;
- chưa claim-level verify câu trả lời;
- weights chưa tune bằng official labeled benchmark.

### Done When

- explicit reference match có thể thay đổi context order deterministically;
- inactive/reference mismatch có trace và warning;
- every unique hit được selected hoặc có omission reason;
- grader abstains khi explicit reference coverage thất bại;
- legacy artifacts load không cần rebuild;
- full local test suite pass.

---

## 23. Milestone 21 — Claim-level Grounding and Citation Verification

**Status:** Completed

### Objectives

- phát hiện answer có citation đúng identity nhưng claim không được evidence hỗ
  trợ;
- bắt buộc inline evidence marker cho từng synthesized legal claim;
- bảo toàn số liệu và phủ định quan trọng;
- fail closed trước khi answer rời Agent/API;
- giữ per-claim verification trace.

### Implemented Baseline

- typed `ClaimSupportStatus` và `ClaimVerification`;
- bounded `ClaimVerificationConfig`;
- deterministic claim segmentation và combined-marker extraction;
- marker-to-response-citation-to-selected-evidence validation;
- per-claim lexical support score;
- exact numeric preservation check;
- conservative claim-negation preservation check;
- unused citation và uncited claim rejection;
- per-claim coverage trong `CitationVerificationResult`;
- Agent giữ verification result cho cả valid answer và abstention;
- extractive-answer exemption có cảnh báo rõ;
- unit/integration test cho supported, uncited, wrong-number, negation và
  end-to-end Agent abstention.

### Limitations

- lexical overlap không chứng minh semantic entailment;
- chưa dùng NLI/cross-encoder/LLM verifier;
- chưa xử lý đầy đủ paraphrase, coreference hoặc multi-hop claim;
- chỉ kiểm tra negation xuất hiện, chưa chứng minh đúng logical scope;
- sentence segmentation là deterministic baseline;
- threshold chưa tune trên labeled benchmark chính thức.

### Done When

- mỗi synthesized claim có typed verification record;
- inline marker phải thuộc response citation và selected evidence;
- changed quantity và introduced negation fail closed;
- unsupported claim làm Agent trả abstention;
- extractive mode vẫn hoạt động;
- không thêm dependency hoặc rebuild artifact;
- full local test suite pass.

---

## 24. Milestone 22 — Model-backed Semantic Claim Verification

**Status:** Completed

### Objectives

- kiểm tra semantic support sau hard checks của M21;
- phân biệt supported, contradicted và insufficient;
- không cho model tạo hoặc thay đổi citation identity;
- giữ backend tùy chọn và không buộc local runtime dùng GPU/API;
- fail closed khi model/provider/schema không đáng tin cậy.

### Implemented Baseline

- typed semantic draft, trusted assessment và aggregate result;
- `SemanticVerificationConfig` với backend mặc định `disabled`;
- provider reuse qua `ChatModelProvider`;
- load-once Transformers runtime sharing cho generator/verifier có exact runtime
  identity, với inference lock và consumer limits độc lập;
- two-stage `ModelBackedCitationVerifier`;
- strict claim completeness/order validation;
- trusted evidence reattachment từ M21 result;
- pinned model/provider provenance trong response metadata;
- bounded structured-output retry;
- Agent abstention cho contradicted/insufficient/model failure;
- unit/integration tests không network, model thật hoặc dataset thật.

### Limitations

- chưa benchmark model semantic verifier trên labeled legal claims;
- chưa calibrate theo competition metric;
- model có thể sai với điều kiện, ngoại lệ, temporal validity và multi-hop law;
- backend mặc định vẫn tắt cho đến khi có model/revision được duyệt;
- semantic verification không thay thế legal expert review.

### Done When

- hard checks luôn chạy trước model;
- model không thể cung cấp trusted evidence identity;
- mọi claim được đánh giá đúng một lần;
- mọi non-supported outcome fail closed;
- default runtime không cần GPU/network;
- full local test suite pass.

---

## 25. Milestone 23 — Reproducible Quality Benchmark Comparison

**Status:** Completed; official benchmark execution pending competition data

### Objectives

- compare multiple retrieval/generation/verifier candidates without assuming an
  unpublished competition metric;
- preserve exact benchmark, dataset, runtime, model, package, artifact, latency,
  and resource provenance;
- reject non-comparable runs instead of producing misleading rankings;
- expose quality/latency trade-offs as a Pareto frontier;
- select one candidate only under an explicit user-declared policy.

### Implemented Baseline

- sanitized evaluation runtime fingerprint and component provenance;
- pinned dataset name/revision in every new evaluation summary;
- typed candidate, objective, direction, threshold, and selection-mode config;
- strict benchmark SHA-256, case-count, cutoff, dataset-lineage, and label-count
  comparability checks;
- namespaced retrieval, generation, latency, failure, and resource metrics;
- optional accelerator name and peak allocated-memory observation without
  requiring GPU in the default test path;
- explicit candidate eligibility and exclusion reasons;
- deterministic Pareto dominance;
- optional ordered lexicographic selection, disabled by default;
- immutable `comparison.json` persistence;
- `legal-rag-compare` CLI and portable example config;
- deterministic unit/integration tests without network, GPU, model, or corpus.

### Limitations

- no official or reviewed labeled benchmark is included;
- benchmark label provenance still requires external review;
- Pareto membership does not prove legal correctness;
- no model is declared final;
- no hyperparameter search, fine-tuning, dataset download, or artifact rebuild.

### Done When

- different benchmark bytes or dataset revisions fail before comparison;
- unavailable labels never become zero-valued quality metrics;
- every excluded candidate has a reason;
- default comparison produces no artificial winner;
- explicit lexicographic policy is deterministic;
- full local test suite passes.

---

## 26. Milestone 24 — Benchmark Governance and Regression Gates

**Status:** Completed; trusted benchmark population pending reviewed/BTC data

### Objectives

- pin benchmark labels and corpus lineage before evaluation;
- distinguish diagnostic labels from reviewed or competition-official labels;
- prevent model-winner claims from untrusted labels;
- enforce bounded regressions against an explicit baseline candidate.

### Implemented Baseline

- typed benchmark manifest and label-status enum;
- exact benchmark SHA-256, case-count, and granularity validation;
- runtime artifact-lineage compatibility before case execution;
- manifest identity persisted in evaluation and comparison reports;
- diagnostic selection block with Pareto comparison still available;
- per-objective absolute regression tolerance against a named baseline;
- required `--benchmark-manifest` evaluation CLI input;
- portable diagnostic manifest and comparison examples;
- deterministic unit/integration tests without network, GPU, model, or corpus.

### Limitations

- no reviewed or competition-official benchmark is bundled;
- manifest trust is declared provenance, not independent legal review;
- no official competition metric or final model is selected.

### Done When

- altered benchmark bytes or manifest mismatch fail before evaluation;
- corpus revision mismatch fails before evaluation;
- diagnostic labels cannot produce a selected winner;
- trusted labels plus explicit policy may select deterministically;
- excessive objective regression makes a candidate ineligible;
- full local test suite passes.

---

## 27. Milestone 25 — Competition Data Reset

**Status:** Completed; official corpus ingestion moved to Milestone 26

### Objectives

- remove every active AIO-only source, config, fixture, test and dependency;
- make the runnable core reject legacy/external artifact lineage;
- preserve dataset-independent cleaning, parsing, chunking, indexing,
  retrieval, generation, verification, Agent, API and evaluation modules;
- add only the official QA/context boundary that current BTC documentation can
  support without guessing submission or corpus details.

### Implementation Scope

- competition-only provenance configuration;
- typed answer-record and context-record schemas;
- strict local JSON loaders with duplicate-key detection;
- removal of the AIO package, raw audit service and AIO offline composition
  root;
- removal of the AIO build CLI/profile and Hugging Face `datasets` dependency;
- neutral core fixtures and updated documentation;
- unit tests without official data, network or model download.

### Deferred

- selected-context normalization and official index build;
- official METEOR/ROUGE-L equivalence;
- final submission formatter (completed later in M28);
- fine-tuning and model selection;
- external-data policy claims beyond the fail-closed project decision.

### Done When

- active source/config/tests contain no AIO import or dataset identity;
- legacy artifacts fail competition provenance validation;
- official-format sample records load into typed competition schemas;
- no official data or artifact is committed;
- full local test suite passes.

---

## 28. Milestone 26 — Official Corpus Ingestion

**Status:** Completed; released-data validation and cleaner completed in M37

### Objectives

- read official `context_*.json` records directly from ZIP or a directory;
- validate the exact documented root shape and raw fields;
- produce dataset-neutral `LegalDocument` records without inferred metadata;
- pin corpus lineage by exact canonical source bytes;
- expose deterministic corpus audit and normalized-document manifests;
- connect official documents to the existing parser/chunker boundary.

### Implementation Scope

- direct ZIP/directory context source inspection;
- duplicate JSON-key, member-name and context-ID checks;
- deterministic source revision shared by equivalent ZIP/directory inputs;
- strict context adapter and corpus audit report;
- normalized and cleaned ingestion contracts (cleaner finalized in M37);
- unit and local integration tests with small official-format fixtures.

### Deferred

- full corpus counts and quality findings (completed in M37);
- persistence/build CLI and complete artifact build orchestration;
- BM25/vector/graph rebuild;
- official metric equivalence and submission formatting;
- training or model selection.

### Done When

- official-format ZIP and directory fixtures produce identical lineage;
- raw field names do not enter unified documents or downstream modules;
- malformed, duplicate or incomplete context input fails closed;
- normalized manifests carry official dataset identity and source revision;
- a context can flow into the existing legal parser/chunker in a local test;
- the full local test suite passes.

---

## 29. Milestone 27 — Official Offline Assembly and Batch Execution

**Status:** Implementation completed with fixtures; full-corpus execution
pending official data

### Objectives

- persist audited official documents and complete retrieval artifacts;
- resume long offline builds without silently accepting incompatible output;
- run question-only or warm-up batches with per-question checkpoints;
- prove batch completeness without guessing the Codabench submission format.

### Implementation Scope

- official build CLI and atomic build-state transitions;
- normalized/cleaned, empty relationship/graph, legal block/chunk, BM25, vector,
  and validation stages;
- existing vector batch checkpoint integration;
- internal answer-result JSONL, batch state, and immutable completion manifest;
- unit and integration tests using small organizer-shaped fixtures and fake
  model providers.

### Deferred

- organizer full-corpus counts and production artifact build;
- exact Codabench submission formatter;
- official METEOR/ROUGE-L scorer equivalence;
- training, fine-tuning, and final model selection.

### Done When

- interrupted compatible builds and batches resume only missing work;
- incompatible source/config/code identity fails closed;
- every completed batch has exactly one output per input question ID in order;
- no reference answer is used as a prediction;
- full local tests pass without network or model downloads.

---

## 30. Future — Competition Evaluation and Submission

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

## 31. Milestone 28 — Exact Codabench Submission Packaging

**Status:** Completed with local fixtures; official upload pending

### Objectives

- convert a complete internal batch into the organizer's exact answer-only ZIP;
- reject missing, duplicate, reordered, stale, or tampered predictions;
- make the final archive reproducible without adding data or model logic.

### Implementation Scope

- typed `id`/`answer` submission item;
- exact question-source and batch-manifest compatibility checks;
- UTF-8 `submission.json` output (the original array contract is superseded by
  D077 after live scorer verification);
- deterministic `submission.zip` containing no other member;
- no-overwrite publication and final archive self-validation;
- `legal-rag-submit` CLI and local tests.

### Deferred

- organizer-equivalent METEOR/ROUGE-L implementation until scorer details or
  official scoring behavior can be verified;
- official full-corpus execution, model selection, and Codabench upload.

### Done When

- output is named `submission.zip` and contains only `submission.json`;
- every official ID occurs exactly once with one string answer;
- changed question or result bytes fail before publication;
- archive bytes are reproducible and existing output is not overwritten;
- full local test suite passes.

---

## 32. Milestone 29 — Competition Text Metrics and Answer Rendering

**Status:** Completed with local fixtures; official scorer parity pending

### Objectives

- measure answer/reference overlap in the same optimization direction as BTC;
- keep local scores explicitly diagnostic while scorer details are unknown;
- remove verified internal citation markers from score-facing answer text.

### Implementation Scope

- exact-token Vietnamese METEOR diagnostic;
- token-level ROUGE-L F1 diagnostic;
- nullable per-case metrics and aggregate report values;
- explicit non-equivalence warning in evaluation reports;
- citation-marker-free Codabench answer rendering with unknown-marker rejection;
- deterministic unit and integration tests.

### Deferred

- official tokenizer, stemming/synonym rules, package version, parameters, and
  aggregation parity until BTC scorer implementation can be verified;
- metric-guided model selection until a pinned reviewed/official split exists.

### Done When

- labeled answers produce bounded METEOR/ROUGE-L diagnostics;
- unlabeled questions do not receive invented scores;
- fragmented/reordered text scores lower in deterministic regression tests;
- submission answers contain no verified internal `[E<number>]` markers;
- full local suite passes.

---

## 33. Milestone 30 — Answer-only Warm-up Scoring CLI

**Status:** Completed with local fixtures; official warm-up run pending

### Objectives

- score the exact Codabench archive directly against official warm-up answers;
- require complete ordered ID equality before computing any aggregate;
- persist reproducible diagnostics without copying legal answer content.

### Implementation Scope

- strict one-member submission ZIP loader with duplicate-field/ID rejection;
- official reference loader with required answers;
- per-question and mean exact match, METEOR, and ROUGE-L;
- three exact input checksums and code-version provenance;
- immutable content-free `warmup_score.json` report;
- `legal-rag-score-warmup` CLI;
- local tests requiring no runtime, corpus, model, network, or GPU.

### Deferred

- exact Codabench parity under D073;
- official warm-up score until a real generated submission is supplied;
- model selection policy based on reviewed train/dev splits.

### Done When

- valid official-shaped input produces bounded aggregate and per-ID scores;
- missing, extra, duplicate, or reordered IDs fail closed;
- malformed or multi-member ZIP fails closed;
- reports contain no question, gold, or prediction content;
- existing report destinations are not overwritten;
- full local suite passes.

---

## 34. Milestone 31 — Competition Compliance and Reproducibility

**Status:** Implementation completed; organizer corpus/model approvals pending

### Objectives

- turn the supplied organizer rules into fail-closed engineering gates;
- prepare mandatory Data Statement and Model Card evidence;
- provide a private-submission checklist and quota ledger;
- create a data-free, secret-free Docker reproduction scaffold;
- keep answer packaging/scoring usable without the serving stack.

### Implementation Scope

- organizer-rules checksum and compliance document;
- candidate model/license/approval register: E5 MIT, reranker Apache-2.0,
  Qwen 3B custom `qwen-research`; all candidates remain blocked until BTC
  approval;
- MIT source license;
- Data Statement, Model Card, private checklist and CSV ledger templates;
- non-root Python 3.11 CPU Dockerfile, `.dockerignore` and direct dependency
  constraints;
- lightweight competition CLI entry points for submission and warm-up scoring;
- import regression proving FastAPI is not loaded by lightweight tooling;
- version `0.33.0`, no new runtime dependency.

### Deferred

- corpus audit and all official indexes until `selected-contexts.zip` exists;
- organizer approval evidence, full transitive license review and final model
  selection;
- final GPU image, image digest and complete package freeze until organizer
  runtime constraints are known;
- exact Codabench scorer parity and portal-name clarification;
- completed release-specific Data Statement/Model Card until a real candidate
  and official artifacts exist.

### Done When

- compliance requirements and unresolved questions are explicit;
- no candidate with unknown approval is represented as competition-ready;
- Docker build context cannot include local data/model/artifact/secret paths;
- lightweight competition CLI import does not initialize FastAPI;
- compliance templates contain every organizer-required evidence category;
- full local test suite passes.

---

## 35. Milestone 32 — Live Codabench Submission Contract Correction

**Status:** Completed; BTC fixed WordNet and a submission scored successfully

### Objectives

- correct the documented-array mismatch exposed by the executable scorer;
- make local validation reproduce the scorer's `.items()` access pattern;
- preserve all existing completeness, ordering and archive safety gates.

### Implementation Scope

- serialize an ID-keyed root object with exact `{"answer": string}` values;
- reject the formerly documented list root and malformed answer objects;
- regression-test the exact scorer projection;
- update competition contract documentation and version to `0.34.0`.

### Done When

- targeted and full local test suites pass;
- a corrected archive passes local formatter/scorer validation;
- Codabench accepts the corrected root shape and reaches metric execution;
- BTC's repaired scorer produces a non-empty official warm-up score.

---

## 36. Milestone 33 — Team Onboarding and Confirmed Competition Rules

**Status:** Completed

### Objectives

- give all team members one practical end-to-end map of the repository;
- record the organizer's confirmed model/data/training/platform rules;
- distinguish implemented pipeline code from pending official artifacts;
- make contribution, testing and submission workflows reproducible.

### Implementation Scope

- detailed Vietnamese team onboarding guide;
- README entry point and package/CLI navigation;
- D078 official-only fine-tuning and synthetic-data decision;
- observed Codabench NLTK/WordNet limitation and its later BTC resolution;
  the earlier PyVi inference is superseded by M36 source analysis;
- version `0.35.0`, no dependency or business-logic change.

### Done When

- a new member can trace offline, online and submission data flows;
- model registration and prohibited-data rules are unambiguous;
- source/docs/tests are consistent and full local validation passes;
- repository changes are committed and pushed for team access.

---

## 37. Milestone 34 — Organizer Model, API and Reproducibility Rules

**Status:** Documentation completed; enforcement implementation pending

### Objectives

- record the organizer's system-wide parameter limit and local-control rule;
- remove resolved uncertainty about APIs, pretrained models and packaging;
- define the mandatory model inventory before an official run.

### Confirmed Scope

- total learned parameters across every active Task 2 model must be below 4B;
- embedding, reranker, generator and model-based verifier/grader counts are
  additive;
- quantization and LoRA do not make an over-limit base model eligible;
- all model APIs are prohibited; models must run under direct team control;
- only organizer data may be used directly and augmentation is prohibited;
- pretrained/distilled models and eligible research/non-commercial licenses are
  allowed subject to registration and compliance review;
- Docker, GitHub or ZIP delivery is acceptable when README reproduction is
  complete; eligible weights may be downloaded from the Internet.

### Next Implementation Gate

- extend the model register with verified parameter counts and sources;
- calculate the aggregate for each named experiment configuration;
- fail closed on missing counts or aggregate `>= 4_000_000_000`;
- record model registration evidence before an official batch.

### Deferred

- final model stack until parameter inventory and BTC registration Form exist;
- official corpus/train experiments until the files are released and audited;
- final execution image until hardware/runtime constraints are announced.

---

## 38. Milestone 35 — Official Data Overview Contract Alignment

**Status:** Superseded by completed Milestone 37 implementation

### Objectives

- preserve the complete organizer overview as an in-repository coding contract;
- distinguish confirmed fields from assumptions requiring byte-level audit;
- identify mismatches before the official corpus build.

### Confirmed Scope

- Vietnamese legal question input and grounded prose-answer output;
- five named resources and their documented phase roles;
- question/answer mapping and four documented context fields;
- numeric context ID and slug-like name observed in the official example;
- legal passage preservation, retrieval requirements and METEOR/ROUGE-L roles;
- no relationship table or retrieval relevance labels in the overview.

### Next Implementation Gate

- update the raw context adapter to canonicalize audited integer/string IDs;
- reject booleans, floats, nulls, blanks, duplicates and canonical collisions;
- add numeric-ID unit/integration fixtures based only on the official example;
- choose unknown-field policy only after inspecting the real corpus archive.

### Follow-up

- adapter and full-corpus audit completed in Milestone 37;
- official index build remains a separate measured run.

---

## 39. Milestone 36 — Official Scoring Source Analysis

**Status:** Documentation completed; compatible local implementation pending

### Objectives

- inventory and checksum the exact scoring ZIP supplied by BTC;
- derive executable prediction/reference/output contracts without running
  untrusted archive code;
- resolve METEOR/ROUGE-L tokenization and aggregation assumptions;
- identify exact differences from local M29/M30 diagnostics;
- update every active competition source-of-truth document.

### Confirmed Scope

- ZIP SHA-256
  `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`;
- NLTK METEOR over whitespace tokens with default runtime behavior;
- WordNet and OMW resources downloaded at scorer startup;
- vendored ROUGE-L with no stemming and lowercase ASCII-only tokenization;
- arithmetic macro mean for both metrics;
- no active PyVi tokenization;
- score output keys `rouge` and `meteor`;
- local diagnostic implementation is not official-equivalent.

### Deferred

- implementation of a separate official-compatible metric mode;
- exact NLTK/NumPy and WordNet/OMW pinning because the ZIP has no lock;
- golden-vector parity check against the BTC scorer runtime;
- verification that public/private phases use the same scorer checksum.

### Done When

- scorer artifact/member checksums and runtime I/O are documented;
- stale PyVi and unknown-aggregation claims are removed;
- D080 records the accepted behavior;
- documentation consistency checks pass;
- no scorer source, dataset, dependency or business logic is added.

---

## 40. Milestone 37 — Official Data Adapter and Passage Cleaner

**Status:** Completed

### Objectives

- adapt the exact released QA/context schemas without leaking raw types;
- preserve raw official passages and create a separate cleaned artifact;
- remove only noise demonstrated by byte-level corpus audit;
- pin cleaning policy and source lineage for later index builds.

### Implementation Scope

- numeric/string context-ID canonicalization and collision rejection;
- optional `name`, blank `passage`, exact field-set enforcement;
- dataset-specific deterministic cleaner for NFC/newlines/line whitespace,
  known HTML presentation markup and exact TVPL Pro boilerplate;
- expanded audit schema and normalized/cleaned ingestion result;
- offline runtime consumes cleaned documents for parsing while preserving raw
  normalized documents;
- full read-only audit of 8.532 official context records;
- focused unit/integration tests and source-of-truth documentation;
- version `0.36.0`, no dependency added.

### Deferred

- full official parser/chunker/BM25/vector artifact build;
- any relationship extraction or non-empty graph;
- leakage-safe train/dev split and fine-tuning experiments;
- official-compatible scorer implementation and final model benchmarking.

### Done When

- adapter accepts every official record and rejects unobserved schema drift;
- normalized and cleaned outputs have identical ID/order/count and lineage;
- full corpus audit records exact counts and deterministic cleaning statistics;
- targeted and full local test suites pass.

---

## 41. Milestone 38 — Official Parser/Chunker Stage Execution

**Status:** Completed

### Objectives

- run the reusable legal parser/chunker on official cleaned contexts only;
- stop before model-backed/index stages so structure quality can be audited;
- make the partial output immutable, checksummed and resumable.

### Implementation Scope

- `--through document_processing` on `legal-rag-build-competition`;
- explicit post-write validation of block/chunk manifests and payloads;
- stage-limited result with `validation_report = null`;
- exact resume into BM25/vector/final validation using the same source, config
  and code identity;
- integration coverage proving embedding is not invoked;
- version `0.37.0`, no dependency added.

### Measured Result

- canonical revision produced 1.215.092 blocks and 335.014 chunks;
- parser coverage was 261.550.497 / 261.550.497 non-whitespace characters;
- 7.637 documents were structured, 875 unstructured and 20 blank;
- chunk strategies: 131.806 article, 73.914 clause-group, 87.623 token-fallback,
  41.671 standalone;
- block/chunk payload SHA-256 and processing hashes matched across two builds;
- cross-process resume completed in about 11 seconds without reparsing;
- peak observed parser/chunker working set was about 116 MB after corpus ingest;
- typed application config is now hashed directly, fixing process-dependent
  ordering caused by prematurely converting frozensets to JSON lists.

### Out of Scope

- BM25/vector full build;
- inferred graph relationships;
- synthetic retrieval labels or training data;
- fine-tuning and model benchmarking.

---

## 42. Milestone 39 — Official Pre-GPU Quality Hardening

**Status:** Completed

### Objectives

- remove only organizer page-code residue demonstrated by the M38 audit;
- stop parser false positives without losing source text or valid Roman Điều;
- turn structural headings into context for a substantive retrieval unit;
- guarantee a bounded `search_text` proxy before exact embedding-tokenizer
  preflight;
- rebuild immutable official artifacts and BM25 on CPU before any GPU use.

### Implementation Scope

- UIT DSC passage cleaner `1.2` with balanced naked script/style removal and
  fail-closed unbalanced handling;
- complete-token and wrapped-line article markers plus conservative
  implicit-clause guards;
- heading attachment with complete source-block coverage;
- `max_tokens=384`, `max_search_tokens=448`, stored search-token diagnostics
  and content-preservation validation;
- version `0.38.1`, no dependency added;
- new artifact root; M38 diagnostic artifacts are not modified.

### Done When

- full test suite passes;
- all 8.532 official contexts rebuild with exact parser/chunker coverage;
- audited JavaScript and false-marker signatures are absent;
- every search text preserves its full chunk and respects the configured proxy
  budget;
- BM25 builds and reloads on CPU from the corrected chunks;
- exact embedding-model tokenizer preflight remains the only gate before the
  Kaggle GPU vector build.

### Full-corpus Verification

The immutable `0.38.1` root rebuilt all 8,532 official contexts into
1,145,383 legal blocks and 373,253 chunks. The independent audit found zero
audited JavaScript residues, zero nested-numeric false clauses, zero duplicate
chunk IDs, zero `search_text` budget violations and zero content-preservation
mismatches. It recognized 12,412 wrapped article markers. Three remaining
replacement characters are source corruption and are preserved rather than
guessed.

The CPU BM25 artifact contains 373,253 records, uses the canonical official
corpus revision and reloads in a fresh process with checksum/integrity
validation. Three smoke queries each returned five hits in approximately
0.05--0.12 seconds after load. Retrieval relevance is not claimed from this
unlabeled smoke test.

### Out of Scope

- embedding or vector build;
- model download, generation, fine-tuning or benchmark;
- inferred graph relationships or synthetic supervision.

---

## 43. Milestone Execution Rules

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
