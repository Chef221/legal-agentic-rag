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

## 43. Milestone 40 — Exact E5 Tokenizer Rechunk

**Status:** Completed

- replace the failed proxy gate with exact revision-pinned E5 token budgeting;
- preserve source text and prohibit silent truncation;
- rebuild official chunks and BM25 under `uit-dsc-2026-task2-v0400`;
- require zero inputs above 512 tokens before vector construction;
- keep `0.38.1` immutable as failed preflight evidence.

Completion evidence:

- rebuilt 330,768 official chunks and a matching 330,768-record BM25 index;
- retained the canonical official corpus revision
  `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`;
- independently retokenized all persisted E5 inputs;
- observed maximum length 512 and zero inputs above the model window.

## 44. Milestone 41 — Official Artifact Validation and Serving Readiness

**Status:** Completed

- persist một full-corpus validation report riêng cho artifact M40;
- tạo và kiểm tra vector-serving metadata;
- smoke test BM25, dense và hybrid trên câu hỏi public chính thức;
- smoke test grounded answer bằng extractive generator không cần GPU;
- không rebuild vector, không đánh giá retrieval quality khi chưa có relevance
  labels.

Completion evidence:

- full-corpus report hợp lệ cho 8.532 context/8.512 context có nội dung;
- vector sidecar tạo đủ 330.768 row và online startup bằng validated report;
- public question `80189` trả 5 hit cho cả BM25, dense và hybrid;
- measured local startup 0,08 giây, BM25 0,37 giây, cold dense 10,50 giây và
  warm hybrid 0,26 giây;
- extractive E2E kết thúc `answer_verified` với 5 citation trong 10,22 giây;
- các số đo là smoke evidence trên máy phát triển, không phải quality benchmark
  hoặc cam kết phần cứng BTC.

## 45. Milestone 42 — Public Extractive Baseline Submission

**Status:** Completed

- chạy đúng 1.000 public questions bằng hybrid retrieval;
- checkpoint từng answer và log tiến độ mỗi 25 câu;
- không gọi reranker, generator model, API hoặc dữ liệu ngoài;
- xác nhận exact question-ID coverage;
- đóng gói deterministic `submission.zip` theo scorer contract thực tế.

Completion evidence:

- xử lý đủ 1.000/1.000 public questions, 1.000 ID duy nhất và đúng source order;
- source SHA-256
  `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5`;
- batch records SHA-256
  `1c5eebd64f493ad6a97a84e43ff8bafae10e0ed059113f2a7b946e4e81b1054e`;
- 996 answer có evidence, 4 abstention, không answer rỗng;
- deterministic ZIP chỉ chứa `submission.json`, đủ 1.000 answer string;
- archive SHA-256
  `3cddb19b23d9b296b725dd1ed0c69b9de7770651f31d5b2e67ecc6cf39fa76d6`;
- answer extractive trung bình 9.850 ký tự và mọi record chạm context budget;
  đây là format/operational baseline, chưa phải quality candidate tối ưu metric.

## 46. Milestone 43 — Qwen3B Kaggle Generation Candidate

**Status:** Completed as operational baseline; quality improvement required

- pin Kaggle GPU config cho E5 + Qwen2.5-3B, tổng model dưới 4B;
- giữ hybrid-only retrieval, không load reranker hoặc semantic verifier;
- giới hạn 5 evidence, 3.072 context token và 256 output token;
- đóng gói online-only artifact cùng exact public source;
- smoke một câu trước khi chạy resumable batch 1.000 câu;
- validate exact ID coverage và deterministic submission ZIP;
- không nộp output nếu chưa xác nhận BTC model approval.

Prepared artifact:

- `uit-dsc-2026-task2-serving-v0430.tar.gz`, 17 members, khoảng 0,98 GiB;
- SHA-256
  `90d4d211a20f6d3a6f894d8dd33c0f187fcf141c1bcbc3814d8dcc7e003e729c`;
- chỉ chứa online artifacts M40, validation/provenance manifests và
  `public-official.json`; không chứa AIO hoặc offline intermediate payloads.

### M43.1 — Generator Abstention Hotfix

**Status:** Completed and verified on clean T4 suffix rerun

- giữ generator-level abstention là kết quả fail-closed;
- không gửi abstention qua citation verifier rồi gắn nhầm `answer_verified`;
- thêm regression test cho nhánh model tự báo evidence không đủ;
- không resume checkpoint `0.43.0` bằng code `0.43.1`; batch mới phải dùng output
  directory riêng để giữ đúng recovery identity.

Completion evidence:

- model E5 và Qwen2.5-3B active trong run đã được người dùng xác nhận BTC duyệt;
- 1.000/1.000 unique public IDs, không answer rỗng;
- 0 retrieval model error, 33 generator model error;
- 425 insufficient-evidence responses, 384 citation-verification failures;
- official Codabench METEOR `0.07862292376534387`, ROUGE-L
  `0.16735433212043324`;
- invalid P100 suffix với 615 retrieval errors bị loại; suffix 386–1.000 được
  chạy lại trên T4;
- batch archive SHA-256
  `d68c165366169fd0c567938682078025c49db71face7baa96a2d8a31e5fa7af5`.

## 47. Milestone 44 — Team Quality Improvement Program

**Status:** Ready for parallel implementation

Milestone này không phải một thay đổi end-to-end duy nhất. Nó là tập workstream
có control M43.1 chung:

1. leakage-safe dev split và official-compatible evaluator;
2. retrieval diagnostics và reranker ablation;
3. context selection/token telemetry;
4. prompt/output coverage và generator error reduction;
5. claim-level verification repair cùng grounded fallback;
6. official-only fine-tuning sau khi evaluator được chốt;
7. GPU compatibility, recovery và post-run quality gates.

Phạm vi, metric, file bắt đầu và tiêu chí nghiệm thu nằm tại
`docs/17-TEAM-IMPROVEMENT-BACKLOG.md`. Không merge thay đổi quality nếu thiếu
đối chứng trên cùng split hoặc vi phạm official-only/no-synthetic/no-API/model
approval gates.

## 48. Milestone Execution Rules

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

## 49. Milestone 44.1 — Leakage-safe Dev and Scorer-compatible Evaluation

**Status:** Completed

- `legal-rag-prepare-dev` tạo deterministic group-wise train/dev;
- quarantine training groups exact/near-duplicate với warm-up/public holdout;
- bảo toàn official record, không tạo label hoặc synthetic data;
- persist source/partition checksums, IDs, policy và warnings;
- thêm `official_compatible` mode với optional `nltk==3.7`;
- tái tạo ASCII ROUGE-L và whitespace-token NLTK METEOR theo scorer BTC;
- giữ `diagnostic` mode cũ và fail closed khi thiếu local WordNet.

Chưa thuộc M44.1: retrieval labels, reranker/context/generator/verifier changes,
bootstrap interval và phân tầng score theo question type.

Official-data smoke evidence với seed `2026`, dev fraction `0.15`, threshold
`0.92`: 7.000 train records → 5.617 training, 991 development và 392
quarantined; detector ghi 449 exact duplicate pairs và 1 near-duplicate pair.
Train SHA-256 là
`2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988`;
warm-up/public hashes khớp contract đã audit. Generated split chỉ được giữ local,
không commit vào repository.

## 50. Milestone 44.2 — Non-gold Retrieval Diagnostics

**Status:** Completed

- thêm typed, content-free retrieval diagnostic schemas;
- chạy riêng BM25, dense và hybrid với query identity validation;
- đo branch overlap/Jaccard, hybrid document diversity, explicit legal-reference
  match, latency và warning/error taxonomy;
- hỗ trợ optional lexical answer-term coverage nhưng đánh dấu rõ không phải
  relevance gold;
- persist report bất biến với source/config/code SHA identity;
- thêm `legal-rag-diagnose-retrieval` CLI và unit tests cho success, runtime
  error, immutable output và question-only source.

M44.2 không thay đổi retrieval ranking, không bật reranker, không chạy generator,
không tạo manual/synthetic labels và không có dependency mới. Bước thực nghiệm
tiếp theo là chạy CLI trên leakage-safe development split cùng official artifacts,
sau đó mới thiết kế M44.3 reranker ablation dựa trên report.

Full M44.2 control đã chạy local trên 991 development questions, official M40
artifacts, top-k 20/candidate-k 100: 991 success, 0 failure, mean BM25/dense
Jaccard `0.1578`, mean hybrid document diversity `0.5012`, 293 low-diversity
case, 144 zero-overlap case và 1 explicit-reference miss. Report source SHA-256
`8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`;
config SHA-256
`b530b785c63b0a53a8f1a3058b05ebc78a2ccf0585b42a7b9976d4ba72c7e08d`.

## 51. Milestone 44.3 — Retrieval-only Reranker Gate

**Status:** Completed; candidate-k 20 and 40 reviewed

- optional fourth `hybrid_rerank` branch trong diagnostics;
- typed top-k overlap/Jaccard, reranked diversity, explicit-reference match,
  answer-term coverage delta và rank-change metrics;
- `--include-reranker` CLI flag;
- Kaggle CUDA experiment config với exact approved E5 và mMARCO revisions;
- một report immutable cho mỗi candidate-k 20/40/60;
- comparison runtime tái sử dụng một sparse/dense execution cho direct branch,
  RRF và reranker thay vì gọi backend ba lần;
- chưa đổi baseline strategy và chưa chạy generator trong gate này.

Candidate-k 20 control trên 991 development cases đạt 991 success, 0 failure.
Vì `top_k == candidate_k == 20`, reranker chỉ đảo rank (mean absolute rank
change `5.0894`) mà không đổi membership: hybrid/rerank Jaccard `1.0`, diversity
delta `0` và non-gold answer-term coverage delta `0`. Candidate-k 40 là run đầu
tiên có thể đo candidate replacement; candidate-k 60 chỉ chạy nếu 40 có
tín hiệu/cost hợp lý.

Candidate-k 40 trên 991 cases đạt 991 success, 0 failure và zero log error.
Reranker thay trung bình `7.5409/20` kết quả, hybrid/rerank Jaccard trung bình
`0.4600`, coverage delta trung bình `+0.0052875` và diversity delta trung bình
`-0.0191726`. Backend reuse giảm BM25/dense completions từ 2.977 xuống 993.
Vì quality signal nhỏ và không phải gold, candidate-k 60 bị dừng; k=40 đi tiếp
qua answer-level A/B.

## 52. Milestone 44.4 — End-to-end Reranker Answer A/B

**Status:** Completed; k=40 selected for subsequent development experiments

- hai Kaggle profile Qwen `hybrid_rerank` cho candidate-k 20 và 40;
- giữ `top_k=8`, model revisions, context, verification và generation giống nhau;
- unit test fail nếu hai profile drift ngoài hai candidate-k fields;
- mỗi profile dùng batch directory/checkpoint/config hash riêng;
- tạo submission nội bộ cho đúng 991 development IDs;
- chấm `official_compatible` METEOR/ROUGE-L trên cùng references;
- candidate-k 40 chỉ được promote khi METEOR tăng và guardrails không regression
  nghiêm trọng.

M44.4 không fine-tune, không tạo dữ liệu, không dùng public labels, không đổi
artifact và không chạy candidate-k 60. Runbook thực thi nằm tại
`docs/19-M44-END-TO-END-RERANKER-ABLATION-RUNBOOK.md`.

M44.4.1 bổ sung low-memory Qwen startup qua `accelerate` sau khi run Kaggle stall
trước batch record đầu tiên. Thay đổi chỉ ảnh hưởng cách materialize weights và
thêm phase logs; không ảnh hưởng A/B variable hoặc answer contract.

Hai full-generation batch `v0446` đã hoàn tất trên cùng 991 development records,
được kiểm tra checksum/manifest, đóng gói qua submission formatter và chấm lại
với mode `official_compatible` (NLTK 3.7). k=40 đạt METEOR `0.07683363` so với
`0.07470770` của k=20, ROUGE-L `0.16459483` so với `0.16222435`, tăng 5
`answer_verified`, giảm 5 abstention, giữ nguyên 424 citation failures và tăng
mean latency `0.86 s` (+4.6%). Theo D094, k=40 được chọn làm development
candidate cho bước context/citation tiếp theo; k=20 vẫn là control. Raw batches,
submissions và score reports là local ignored artifacts, không commit vào Git.

## 53. Milestone 45 — Citation Failure Taxonomy and Evidence-selection Observability

**Status:** Completed; controlled k=40 quality candidate is ready to run

- audited all 424 `citation_verification_failed` records from the selected k=40
  development batch without using new data or labels;
- confirmed that failure was caused primarily by missing per-claim inline
  evidence markers, not invalid citation identities;
- made model-backed generation reject incomplete inline marker coverage before
  the fail-closed citation verifier, reusing only its existing one retry;
- made Context Builder distinguish count-cap omissions from actual token-budget
  omissions;
- persisted content-free context and `EvidenceSelectionTrace` metadata on Agent
  responses so the next batch can attribute failures to selection decisions;
- added focused unit coverage for partial marker correction/rejection, truthful
  context warnings and trace persistence.

M45 does not change corpus, artifacts, retrieval rank, evidence-selection score,
model identity, parameter inventory, data policy or submission output. A fresh
same-split k=40 run and official-compatible scoring are required before any
quality claim or evidence-selection policy change.

## 54. Milestone 46 — CPU-only Batch Outcome Analysis

**Status:** Completed

- add typed, content-free analysis for one completed checkpointed batch;
- validate manifest identity, result SHA-256, record count and unique IDs before
  analysis;
- aggregate stop reasons, abstentions, model-error warnings, citation claim-error
  taxonomy, selection traces and Agent latency;
- add a paired control/candidate comparison that rejects different official
  question bytes or ID/order and stores per-ID outcome deltas without answer text;
- add `legal-rag-analyze-batch` and `legal-rag-compare-batches` CLI commands;
- add unit coverage for analysis, comparison, immutable persistence and checksum
  rejection.

M46 does not invoke a model, require GPU, alter an index, modify retrieval/context
policy, create data or score quality. Its next consumer is the controlled M45
k=40 rerun: analyze the new trace-bearing batch, compare it with the M44 control,
then decide whether a bounded evidence-selection experiment is justified.

## 55. Milestone 47 — Batch Readiness Gate and Paired Score Comparison

**Status:** Completed

- add a typed explicit-policy gate for a completed internal batch;
- verify exact official question bytes, ordered IDs, batch manifest and result
  checksum before making a readiness decision;
- persist a content-free `batch_readiness.json` report and fail the CLI when its
  policy is violated;
- add a paired `warmup_score.json` comparison that rejects incomparable source,
  scorer or ID identities and reports per-ID/aggregate metric deltas;
- add `legal-rag-check-batch` and `legal-rag-compare-scores` commands;
- add unit coverage for accept/reject paths, content-free persistence and score
  comparison compatibility.

M47 is CPU-only. It does not load models, invoke a GPU, generate answers,
change output answers, score a new submission, alter official data/artifacts or
choose a retrieval/context policy. Every threshold remains an explicit
experiment input rather than a hard-coded production claim.

## 56. Milestone 48 — Bounded Context Document-diversity Ablation

**Status:** Implemented; requires a fresh GPU batch for quality evidence

- add optional typed `max_evidence_per_document`, defaulting to `null` to
  preserve all existing profiles;
- classify over-cap candidates as `document_cap` in typed selection trace and
  content-free warning telemetry;
- continue scanning candidates after a cap so a lower-ranked different document
  can be selected within the existing token/count limits;
- add an isolated k=40/Qwen candidate profile with document cap `2` and no other
  retrieval, model, generation, verifier, corpus, or artifact change;
- add unit coverage for default compatibility, validation, trace and selection
  behavior.

M48 does not assert a quality gain. The required next run is the same
development questions with M48 config, followed by batch analysis, readiness
gate, submission formatting, official-compatible scoring and paired comparison
against the k=40 control.

## 57. Milestone 49 — Claim-linked Generator Output

**Status:** Implemented; requires bounded GPU smoke validation

- replace answer-level model citation declarations with strict claim records;
- require one legal claim per record and at least one explicit evidence ID per
  grounded claim;
- validate all claim links against the selected evidence allowlist;
- reject model-written inline markers and multi-boundary claim records;
- render visible `[E#]` markers deterministically from the validated claim links;
- give the single structured-output retry a reason-specific correction prompt;
- retain the existing fail-closed citation verifier and Agent stopping policy;
- add schema and generator unit coverage for allowlist, abstention, rendering,
  boundary rejection and targeted retry.

M49 does not change official data, indexes, retrieval, context selection, model
identity, parameter inventory, artifact lineage, scoring, or submission format.
Before another 991-question batch, run a small fixed GPU smoke subset and compare
generator errors, citation-verification failures and latency with an equivalent
subset from the M48 candidate.

## 58. Milestone 49.1 — Bounded Structured Generation Recovery

**Status:** Implemented; requires rerunning the fixed 50-question GPU smoke set

- replace verbose Pydantic schema prompting with compact Vietnamese JSON examples;
- bound model output to at most four claims and 600 characters per claim;
- reject long ASCII-only claim drift with targeted Vietnamese correction feedback;
- distinguish JSON decode and Pydantic schema failures without logging content;
- log generated-token count and output-limit hits in the local provider;
- raise only the doc-cap-2 smoke profile from 256 to 384 output tokens;
- preserve evidence allowlist, explicit claim links, system marker rendering and
  fail-closed citation verification.

M49.1 does not rebuild artifacts or change official data, retrieval, model
identity, parameter count, scorer or submission format. Its next gate is the
same immutable 50-question M49 smoke source; do not run all 991 development
questions until that comparison meets the agreed generator-error threshold.

## 59. Milestone 49.2 — Numeric-only Grounded Regeneration

**Status:** Implemented; requires the bounded GPU smoke gate

- add an exact-number rule to the compact model prompt: quantitative text must
  be copied verbatim from evidence cited by that claim;
- add the sole typed `numeric_mismatch` correction signal at the generation-tool
  boundary, without rejected drafts or free-form feedback;
- allow one Agent-only regeneration only for isolated numeric-only claim failures;
- use the identical selected evidence/query/strategy and separate repair
  invocation phases; do not retrieve, rerank or rebuild context;
- retain one repair maximum, fail-closed abstention on every repair failure, and
  content-free repair outcome metadata;
- default the typed repair limit to `0`, enabling `1` only in the doc-cap-2
  k=40 candidate; `FixedRAGService` intentionally has no repair behavior.

The implementation is based on the M49.1 50-question smoke: 0 generator error,
45 verified, 3 model abstentions and 2 numeric-only citation failures (bundle
SHA-256 `5b8ca8a7200b0fc46d163410b2a44d4e120f70e3c33bd2f1fd03395b7e5d254d`).
It does not claim a GPU gain. First run either the exact five-question gate
(the two numeric failures plus three abstentions) or the immutable 50-question
smoke; do not run the 991-question development batch first.

## 60. Milestone 49.3 — Deterministic Numeric Claim Salvage and Diagnostics

**Status:** Implemented; requires the same bounded GPU smoke gate

- keep only verifier-supported claim text verbatim and only their already-linked
  citations when M49.2's numeric-only trigger is met;
- reverify the salvaged response in a dedicated invocation phase before any model
  fallback;
- fail closed on missing/ambiguous citation mapping or salvage verifier
  error/timeout; permit the existing single model regeneration only after a
  successful-but-rejected salvage verification or when no claim is supported;
- record only content-free salvage/regeneration outcomes and aggregate initial
  numeric verifier errors in completed-batch analysis;
- change no official data, artifact, retrieval, context selection, model,
  parameter count, scorer, or submission format.

The first acceptance gate is the existing five-question targeted smoke, followed
by the immutable 50-question M49 smoke. Do not run the 991-question development
batch until this gate and its paired analysis have been reviewed.

## 61. Milestone 49.4 — Typed Failure Diagnostics and Supported-claim Salvage

**Status:** Implemented; requires targeted GPU smoke validation

- classify only structured model-output failures with a closed tool-boundary code;
- persist no rejected draft, prompt, question, or evidence text;
- add a default-off one-shot supported-claim salvage for numeric/negation-only
  verifier failures with existing valid citation identities;
- retain numeric M49.3 semantics and prohibit model regeneration for negation;
- reverify every deterministic candidate and abstain on error, timeout, or rejection;
- aggregate code/salvage outcomes and add focused regression tests.

Run the seven targeted IDs (`134613`, `139655`, `25945`, `38915`, `53115`,
`6739`, `95519`), then the immutable 50-question smoke. Do not start the
991-question batch before reviewing both reports.

## 62. Milestone 49.5 — Bounded Terminal Schema Recovery

**Status:** Implemented; requires targeted and fixed-smoke GPU validation

- preserve `ModelAnswerDraft` strict validation and the existing single model
  correction retry;
- add default-off, one-shot, terminal-only recovery for explicitly allow-listed
  structural JSON mistakes;
- preserve claim text exactly and prohibit inferred semantic values;
- propagate only closed schema issue/repair/outcome telemetry through the tool,
  Agent and batch-analysis boundaries;
- add unit coverage for safe recovery, refusal cases, sanitization, config,
  Agent metadata and aggregate analysis;
- change no official data, corpus artifact, retrieval, model, scorer or
  submission behavior.

The acceptance sequence is: targeted IDs `139655` and `25945`, then the
immutable 50-question smoke set. Do not run the 991-question batch before both
reports are reviewed.

## 63. Milestone 49.6 — Bounded Missing-Required-Field Model Correction

**Status:** Implemented; requires targeted and fixed-smoke GPU validation

- preserve strict `ModelAnswerDraft` validation and precedence of M49.5 local
  structural repairs;
- add default-off, one-shot model correction (`max_missing_field_corrections` in
  `[0, 1]`, default 0) triggered only when terminal schema validation fails for
  unrecoverable missing required fields;
- classify missing required fields as top-level vs claim-level without leaking
  content;
- forbid local semantic guessing, default value synthesis, or claim invention;
- propagate only closed missing-field correction telemetry through generator,
  tool, Agent, and batch-analysis boundaries;
- add unit coverage for eligibility, precedence, call-count limits (at most 1
  extra call), failure fail-closed behavior, and batch diagnostics;
- change no official data, corpus artifact, retrieval, model, scorer or
  submission behavior.

The acceptance sequence is: targeted IDs `139655` and `25945`, then the
immutable 50-question smoke set. Do not run the 991-question batch before both
reports are reviewed.

## 64. Milestone 50.1 — Official-Data QLoRA Fine-Tuning Infrastructure and Direct Screening

**Status:** Completed; C1 executed and rejected due to free-generation degeneration

- preserve frozen M49.6 baseline and strict validation / citation / recovery gates;
- partition clean 5,617 `training.json` into deterministic three-way split (`sft_train.json` ~4,500, `sft_val.json` ~500, `screen_holdout.json` ~617) preserving duplicate groups with seed 2026;
- exclude `development.json` (991 records) as frozen historical benchmark and permanently exclude `quarantined.json` (392 records);
- implement exact ChatML answer-only SFT encoding with `-100` prompt loss masking and dynamic batch padding at $L=1536$;
- implement Candidate 1 QLoRA training runner with trainable parameter preflight check ($< 3\%$ trainable parameters) and comprehensive training manifest;
- implement cached BASE direct-QA generator and paired METEOR / ROUGE-L comparison with 95% bootstrap confidence intervals on `screen_holdout.json`;
- validate locally with unit test suite; do not run full 991 development benchmark until Candidate 1 passes direct screening and immutable 50-smoke.

## 65. Milestone 50.2 — M50-C2 Conservative QLoRA Pilot, SCREEN617 Holdout Evaluation, and Milestone Closure

**Status:** Completed & Closed (M50-C2 Rejected on SCREEN617; No Fine-Tuned Model Promoted)

- post-mortem of M50-C1: teacher-forced validation converged (1.09828) with positive lexical ROUGE-L signal (+0.04153), but free-generation collapsed into repetition loops (18/20 high repetition, 14/20 cap reached, 6/20 EOS);
- implement Candidate 2 conservative pilot recipe: rank $r=4, \alpha=8$, dropout $0.05$, target modules $\{q\_proj, v\_proj\}$ yielding exactly 921,600 trainable parameters;
- configure $\text{LR}=10^{-5}$ (5x lower than C1), cosine scheduler, warmup ratio 0.05, microbatch 2, accumulation 8, `paged_adamw_8bit`;
- bound pilot to `max_optimizer_steps=150` with evaluation and checkpoint gates executed at steps [50, 100, 150];
- implement EOS-preserving SFT encoding & ChatML suffix canonicalization: `encode_sft_example` strips post-EOS template whitespace tokens (e.g. `\n` token ID 198 after `<|im_end|>` token ID 151645) so the final supervised label is strictly `151645`, while sequence truncation preserves terminal `<|im_end|>`; `validate_sft_dataset_encoding` performs sub-second CPU preflight validation across all 5,000 records before GPU model loading;
- preserve strict holdout isolation: `screen_holdout.json` is strictly frozen during training and pilot probing; intermediate probing uses 20 deterministic questions extracted from `sft_val.json` via salted SHA-256 (`m50-c2-val-probe-v1:{question_id}`) with content-level lineage checks;
- implement checkpoint safety gate (0 errors, `cap_without_eos` $\le \text{BASE}+1$, `repeat8_high` $\le \text{BASE}+1$, `duplicate_line_high` $\le \text{BASE}+1$, `eos_emitted` $\ge \text{BASE}-1$, mean length $\le \text{BASE}\times 1.35$, median length $\le \max(\text{BASE}\times 1.35, \text{BASE}+64.0)$);
- implement semantic preservation gate ($\Delta\text{ROUGE-L} \ge -0.01$, $\Delta\text{METEOR} \ge -0.01$ if available, and at least one $> 0$);
- pilot execution on Kaggle GPU: all checkpoints [50, 100, 150] passed VAL20 gates; step 100 selected as pilot winner (combined delta $+0.01182$);
- formal holdout evaluation on SCREEN617: step 100 generated 617/617 cases (0 errors) but failed health gates (cap-no-EOS 6.48% vs 4.86%, repeat8-high 9.08% vs 6.32%, EOS rate 93.52% vs 95.14%) and semantic gates (METEOR delta -0.002803 with 95% CI `[-0.006577, +0.000909]`, Combined delta -0.002698; ROUGE-L delta -0.002594 was within -0.01 tolerance but directionally negative);
- final decision: M50-C2 is **REJECTED**; no fine-tuned model promoted to production;
- holdout consumption: `screen_holdout.json` is **CONSUMED** and must not be used for future adaptive candidate search;
- baseline preservation: frozen M49.6 reliability pipeline (pretrained `Qwen/Qwen2.5-3B-Instruct`) remains active;
- Milestone 50 is officially **CLOSED**; next technical direction requires a separate explicit decision.

## 66. Milestone 51.1 — Phase-A Current-System Census Execution and Baseline Characterization

**Status:** Completed & Closed (Authoritative report: `docs/22-PHASE-A-CLOSURE.md`)

- executed full M49.6-style competition pipeline against 991-question `development.json` benchmark on dual T4 GPUs with Qwen generator placed on GPU1 (`cuda:1`) to prevent illegal memory access;
- established Phase-A quality baseline under official competition scoring: METEOR `0.0980790959` (display rounded `0.098079`), ROUGE-L `0.1871225729` (display rounded `0.187123`), Exact Match `0.0`;
- established Phase-A reliability baseline: 991 records, 806 `answer_verified` (81.33%), 177 `generation_failed` (17.86%), 7 `citation_verification_failed` (0.71%), 1 `max_retry_reached` (0.10%), 185 `insufficient_evidence` (18.67%), 10 generator model errors;
- resolved batch failure blocker: hardened `ToolError` validator in `src/legal_agentic_rag/schemas/tools.py` to allow earlier schema issue diagnostics when accompanied by an attempted missing-field correction chain (`generation_missing_field_correction_attempted = True`); added permanent regression coverage in `tests/unit/schemas/test_tools.py`;
- conducted forensic static audit and empirical census of Graph Retrieval & Adaptive Routing: confirmed competition graph contains 0 edges by design, while `GRAPH_SEARCH` is unconditionally registered; confirmed `QueryIntent.RELATIONSHIP` keyword heuristic has top priority and triggers adaptive routing prepending `[GRAPH, HYBRID_RERANK, HYBRID]`; confirmed zero-edge graph retrieval evaluates only 20 seed candidates instead of configured 40; confirmed 22/22 relationship queries routed to `GRAPH_SEARCH` and terminated on Attempt 1 without fallback;
- architecture verdict: generic graph components retained (`KEEP_GENERIC_ONLY`), UIT competition graph integration designated candidate for removal (`REMOVE_COMPETITION_PATH`) pending Phase B1A paired ablation;
- package version bumped to `0.50.6`; Milestone 51.1 / Phase A formally **CLOSED**.

## 67. Milestone 51.2 — Phase B1A: Paired Graph-Routing Behavioral Ablation

**Status:** Completed & Inconclusive (METEOR delta: -0.00020, ROUGE-L delta: +0.00270; conflated graph traversal with candidate-k expansion 20->40).

## 68. Milestone 51.3 — Phase B1A.2: Graph Equivalence and Candidate-Pool Isolation

**Status:** Completed & Closed (Authoritative run: `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a`, Verdict: `GRAPH_REDUNDANCY_PROVEN`)

- isolated zero-edge graph traversal from candidate-pool depth by comparing ARM G (graph shell, top 20 seeds) against ARM S20 (direct seed rerank, top 20 seeds) and diagnostic ARM H40 (hybrid rerank, top 40 candidates);
- confirmed 22/22 seed hit matches, 22/22 final top-8 matches, and 22/22 score tolerance passes ($\le 10^{-6}$) between G and S20;
- confirmed 0 graph edges, 0 traversal steps, proving the competition graph traversal is entirely redundant;
- confirmed H40 differs in 17/22 cases, proving H40 is non-equivalent and must not replace S20 in Attempt 1;
- authorized Phase B1B structural graph removal.

## 69. Milestone 51.4 — Phase B1B: Structural Competition Graph Removal

**Status:** Completed & Closed (Authoritative run archive: `phase-b1b-graphless-equivalence-evidence.zip` SHA-256: `f392cc650699ecc562cb43ea0ea7f6e965455a36a621843ec6a882172913c9c3`, Verdict: `B1B_EQUIVALENCE_PASS`, Commit: `38a6feec8867a41454c453cce9c54b162801579e`)

- package version bumped to `0.50.7`;
- removed `ToolName.GRAPH_SEARCH` from online agent capabilities; added `ToolName.RELATIONSHIP_RERANK_SEARCH` with `RelationshipSeedRerankingRetriever` preserving exact S20 candidate isolation;
- routed `QueryIntent.RELATIONSHIP` across 3 attempts: Attempt 1 (`relationship_rerank_search` -> S20), Attempt 2 (`rerank_search` -> H40), Attempt 3 (`hybrid_search`);
- reduced online runtime manifest requirement to exact 3 artifacts (`legal_chunks`, `bm25_index`, `vector_index`);
- removed relationship mapping and graph index construction from offline competition build profile;
- preserved generic graph capabilities (`KEEP_GENERIC_ONLY`) in generic library packages;
- executed Kaggle post-change equivalence verification protocol, confirming 22/22 exact matches, 22/22 score tolerance passes ($\le 10^{-6}$), 22/22 branch depth 40 / candidate query / fusion limit / final top-k / route plan passes, 0 retrieval model errors against frozen B1A.2 baseline `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a`;
- officially closed Phase B1B with mechanical verdict `B1B_EQUIVALENCE_PASS`;
- next active research priority established as candidate-pool / reranker audit (S20 vs H40) without speculative promotion of H40.

## 70. Milestone 51.5 — Stage R1: S20 vs H40 Candidate-Pool / Reranker Mechanics Audit

**Status:** Completed & Closed (Authoritative run archive: `candidate-pool-reranker-audit-evidence.zip` SHA-256: `ce9b239b808c3d7b0e575ce1c1683db243bbea909f0e6d9c306df21cb2899860`, size `56,706 bytes`, Verdict: `CANDIDATE_POOL_AUDIT_PASS`, Commit: `9a5b708c2769425dbd65731feb8ede96975b5b46`)

- evaluated candidate-pool depth effects (fused 20 vs fused 40) under identical single-pass query understanding, branch retrieval (BM25 + dense depth 40), RRF fusion, and single-pass cross-encoder reranker scoring;
- verified 22/22 seed prefix passes, 22/22 shared S20 sequence passes, 22/22 legacy S20 frozen score passes ($\le 10^{-6}$), 22/22 H40 frozen score passes ($\le 10^{-6}$), 22/22 branch depth passes, 0 retrieval model errors;
- reproduced historical divergence: 5 identical top-8 cases, 17 changed top-8 cases;
- characterized candidate-pool mechanics: 35 total tail entrants entering top-8 across 17 cases, 20 document-level churn events;
- shared S20 maximum numerical delta of `1.621e-05` confirmed as batch-shape inference artifact, while observational legacy S20 probe reproduced all 22 frozen S20 scores within `1e-6`;
- runtime config evidence self-identity verified across disk and ZIP archive;
- H40 remains strictly unpromoted (`h40_promotion_authorized = false`); Stage R1 closed with mechanical verdict `CANDIDATE_POOL_AUDIT_PASS`;
- next active research frontier established as **Priority B — Verification-correctness audit**.

## 71. Milestone 51.6 — Priority B: Controlled V0 vs V1 Semantic-Verifier Benchmark Execution, Closure, and Holdout Governance

**Status:** Completed & Closed (Authoritative run archive: `verification-semantic-benchmark-evidence.zip` SHA-256: `bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6`, size `17,290 bytes`, Verdict: `VERIFIER_BENCHMARK_PASS`, Decision: `V1_EXISTING_SEMANTIC_VERIFIER_NOT_PROMOTED`, Commit: `d3aac626400cbe31ed0ed5ad109762fcb78d737d`)

- completed forensic packet materialization (Task B-FORENSIC-0, 4 questions, 8 historical arms, 100% chunk lookups, 100% verifier replay fidelity);
- completed human forensic label freezing (Task B-FORENSIC-1A, `verification-human-forensic-labels-v1.json`, 11 claims: 2 SUPPORTED, 5 CONTRADICTED, 4 INSUFFICIENT);
- completed positive-control packet sampling and materialization (Task B-FORENSIC-1B, `verification-positive-control-review-packets-v1.zip`, 16 PRIMARY + 8 RESERVE candidates across 4 strata);
- completed human positive-control label freezing (Task B-FORENSIC-1C, `verification-positive-control-human-labels-v1.json`, 27 claims: 16 SUPPORTED, 2 CONTRADICTED, 9 INSUFFICIENT); composite benchmark established at 38 claims (18 SUPPORTED, 7 CONTRADICTED, 13 INSUFFICIENT);
- implemented and hardened offline benchmark harness in `scripts/evaluate_verification_semantic_benchmark.py` and `tests/unit/evaluation/test_verification_semantic_benchmark.py` (28 unit tests, fail-closed SHA-256 source assertions, exact 22/22 V0 replay validation, 2-pass deterministic stability evaluation, observational provider call/retry logging, runtime execution identity persistence);
- executed canonical benchmark on Kaggle Tesla T4 (`d3aac626400cbe31ed0ed5ad109762fcb78d737d`): 0 model errors, 2 structured retries, 38/38 deterministic stable claims across passes, mechanical verdict `VERIFIER_BENCHMARK_PASS`;
- empirical outcome: V1 achieved 60.53% claim accuracy (vs V0 47.37%), 88.89% supported retention, 35.0% negative catch rate, $+5$ net correctness delta (+7 fixes, -2 regressions), and 63.64% answer accuracy (retaining 7/7 valid answers, catching 7/15 invalid answers);
- diagnostic failure modes: V1 caught 80% of actor inversions and 55.6% of wrong documents, but missed 100% of condition inversions/omissions, wrong articles, and quantity errors, and caught only 12.5% of scope overgeneralizations;
- formal decision: `V1_EXISTING_SEMANTIC_VERIFIER_NOT_PROMOTED` (`semantic_verifier_promotion_authorized = false` remains in effect);
- dataset lifecycle: 38-claim composite dataset permanently burned as DEVELOPMENT DATA (`verification_benchmark_v1_role = "development_after_first_evaluation"`);
- holdout governance: 8 positive-control reserves cannot be repurposed as a secret promotion holdout; pre-registered fresh holdout strategy required before future V2 promotion;
- active frontier advances to pre-registering fresh holdout protocol and exploring V2 verifier design hypotheses.

## 72. Milestone 51.7 — Priority B: Fresh V2 Holdout Pre-Registration and Sealed Materialization

**Status:** Completed & Closed (Authoritative selection artifact: `verification-v2-holdout-selection-v1.json` SHA-256: `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b`, size `16,788 bytes`; Sealed review package: `verification-v2-holdout-review-packets-v1.zip` SHA-256: `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4`, size `108,532 bytes`; Verdict: `V2_HOLDOUT_PRE_REGISTERED`)

- implemented and hardened holdout pre-registration harness in `scripts/preregister_verification_v2_holdout.py` and unit test suite `tests/unit/evaluation/test_preregister_verification_v2_holdout.py` (14 unit tests);
- validated Phase-A source integrity fail-closed (archive `df05a4...`, results `7b1bf8...`, development `867879...`, serving payload `330768` chunks);
- applied 46-QID contamination exclusion set (4 forensic + 16 primary controls + 8 reserve controls + 22 B1A relationship cases, SHA-256 `eefdd8967c39324bc7e88a8451ef8fb9241f765af1e68a0199db9ba33af01fda`);
- derived 772 eligible `answer_verified` records and stratified into 4 strata (`D_NEGATION_MODALITY`: 164, `C_NUMERIC`: 177, `B_MULTI_CLAIM_CLEAN`: 117, `A_SINGLE_CLAIM_CLEAN`: 314);
- deterministically sampled 16 PRIMARY (4/4/4/4) and 8 FRESH RESERVE (2/2/2/2) cases under salt `verification-v2-holdout-gen-v1:`;
- materialized 16 primary packets with `review_status = "sealed_unreviewed"` and null claim labels;
- confirmed 16/16 chunk lookups, 16/16 selection trace mappings, 16/16 citation metadata cross-checks, and 16/16 exact V0 RuleBasedCitationVerifier replays;
- emitted external selection report, content-free public commitment, and sealed review ZIP;
- strictly preserved holdout blindness: zero question IDs or prompts exposed in tracked documentation or console summary;
- next active action: develop V2 verifier using ONLY the 38-claim development benchmark while the fresh holdout remains sealed.

## 73. Milestone 51.8 — Priority B: Candidate V2-D3.1 Benchmark Execution and Closure

**Status:** Completed & Closed (Authoritative run archive: `verification-v2-d31-development-evidence.zip` SHA-256: `e14f9656a13a04b8e545d88a5dca13653fa317166ff530f45e4b13124f864041`, size `18,379 bytes`, 11 members, Verdict: `V2_D31_DEVELOPMENT_BENCHMARK_PASS`, Decision: `KEEP_D3`, Commit: `1383bf379a01c3f7456e3c41ba3be42846ceee2c`)

- evaluated monolithic two-gate verifier candidate V2-D3.1 on the 38-claim development benchmark;
- confirmed mechanical execution pass with 0 model errors, 0 retries, and 38/38 stable claims across 2 passes (76 calls);
- characterized behavioral trade-off: high contradiction sensitivity (6/7 recall) came with severe overcalling (14 false contradiction positives, 30% precision), dropping supported retention to 12/18 (66.67% vs D3's 94.44%) and regressing 6/7 D3 fixes;
- formally closed V2-D3.1 with decision `KEEP_D3` (`d31_supersedes_d3 = false`, `promotion_authorized = false`).

## 74. Milestone 51.9 — Priority B: Candidate V2-D3.2 Benchmark Execution and Closure

**Status:** Completed & Closed (Authoritative run archive: `verification-v2-d32-development-evidence.zip` SHA-256: `bf44b9d77172d4f1823b62c02abae9e462bfbb9fdc5c650ba87e192e4928878f`, size `28,738 bytes`, 13 members, Verdict: `V2_D32_DEVELOPMENT_BENCHMARK_PASS`, Decision: `KEEP_D3`, Commit: `e5db78f0796c53e973fc63f9dd98df6c95f43f6e`)

- evaluated asymmetric two-stage verifier candidate V2-D3.2 (Frozen D3 Base + Strict Contradiction Confirmation Overlay) on the 38-claim development benchmark;
- verified mechanical execution pass with 0 model errors, 0 retries, 38/38 stable claims, and 0 base D3 drift across 152 reconciled provider calls;
- confirmed overlay behavior: 0 false overrides produced, preserving 100% of D3 gains (7/7) and 17/18 supported retention, but caught 0 contradictions (net delta = 0 vs D3);
- formally closed V2-D3.2 with decision `KEEP_D3` (`d32_supersedes_d3 = false`, `promotion_authorized = false`).

## 75. Milestone 51.10 — Priority B: Candidate V2-D3 Freezing, Pre-H-LABEL Hardening, and Fresh Holdout Protocol Pre-Registration

**Status:** Completed & Ready for Review (Protocol: `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`, Label Freeze Script: `scripts/freeze_verification_v2_holdout_labels.py`, Evaluation Harness: `scripts/evaluate_verification_v2_d3_holdout.py`, Unit Tests: `tests/unit/evaluation/test_verification_v2_d3_holdout.py`, `tests/unit/evaluation/test_freeze_verification_v2_holdout_labels.py`)

- officially selected and frozen Candidate **V2-D3** (`Qwen/Qwen2.5-3B-Instruct` revision `a1d308dfcc03e09da285d49d912439a655a571e8`, System Prompt SHA `546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b`, Implementation SHA `a6e8bca15ad14d869e103e1f94fe94bb9a81f9ddc8bc650b280b69b7d57e9826`, Schema SHA `3591144a40b0519d5da9dd262e8edf8814531d798b69deea94fd81fae39f5f61`);
- permanently closed the 38-claim development benchmark for candidate tuning, prompt engineering, threshold tuning, and overlay creation (no D3.3);
- established two-phase irreversible lifecycle: **Phase H-LABEL** (human gold labeling without model predictions) and **Phase H-EXEC** (canonical one-shot model execution on Kaggle GPU);
- pre-registered immutable rate gates for holdout evaluation (`supp_ret >= 0.88`, `neg_catch >= 0.50`, `val_ans_ret >= 0.80`, `full_ans_acc >= 0.60`, `claim_bin_acc >= 0.70`), non-vacuous coverage gates, and mechanical gates (0 model errors, 0 execution errors, 0 unstable claims);
- executed final pre-H-LABEL integrity hardening pass (D124):
  - duplicate JSON key hook and fail-closed duplicate review checks in `freeze_verification_v2_holdout_labels.py`;
  - 2-stage governance status lifecycle (`FROZEN_PENDING_EXTERNAL_REVIEW` -> `EXTERNALLY_REVIEWED_FOR_H_EXEC`);
  - mandatory label commitment in canonical H-EXEC (blocking raw `--holdout-labels-sha256`);
  - label artifact metadata, claim counts, class counts, and exact per-claim `claim_text_sha256` matching against review packets;
  - exact prediction set equality ($\text{ExpectedKeys} \equiv \text{Pass1Keys} \equiv \text{Pass2Keys}$) and fail-closed stability evaluation;
  - content-safe error telemetry (`error_type`, `error_sha256`, `error_message_length`) eliminating raw exception strings and secret leakage;
  - provider call reconciliation gate ($\text{calls} == 2 \times N_{\text{claims}} + \text{retries}$) with exact system instruction SHA enforcement;
  - fail-closed canonical provenance gate (`V2-D3`, package `0.50.7`, `repeat_count=2`, clean git worktree, `cuda/float16/temp=0.0/8192/512/retries=1/timeout=180.0`, `transformers 4.47.1`);
  - Cell H1 pinned runtime assertions and Cell H6 independent evidence recomputation & verification assertions;
- strictly preserved holdout blindness (zero inspection of holdout questions, claims, evidence, or labels in this task);
- all 1,022 repository tests passing (38 unit tests across holdout evaluation and freezing suites);
- next action: external review of pre-registered frozen holdout evaluation protocol (`docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`) before human gold labeling (Phase H-LABEL).

## 76. Milestone 51.11 — Priority B: Phase H-LABEL Completion, Human Gold-Label Freeze, and External Chain-of-Custody Approval for Phase H-EXEC

**Status:** Completed & Authorized for H-EXEC (Frozen Labels: `verification-v2-holdout-reviewed-labels-v1.json` SHA-256 `85d348dbb7da1567398836b96156a9d08fcfe181b676c5ecd593535ec8904215`, size `9,383 bytes`; Approved Commitment: `configs/verification-v2-d3-holdout-label-commitment.json` SHA-256 `5cc7f58ed52c43d091bce98d5296ab68cded981495736a479644aefc1428b6dc`, size `1,060 bytes`; Review Status: `frozen_human_reviewed`, Governance Status: `EXTERNALLY_REVIEWED_FOR_H_EXEC`)

- unsealed 16 primary review packets from canonical holdout archive (`verification-v2-holdout-review-packets-v1.zip` SHA-256 `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4`, `verification-v2-holdout-selection-v1.json` SHA-256 `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b`);
- conducted neutral human forensic review across all 31 claims (16 questions, 16 arms) with strictly zero model predictions, zero prior verifier suggestions, and zero automatic labels;
- froze authoritative human gold labels into immutable artifact `verification-v2-holdout-reviewed-labels-v1.json` (31 claims: 24 SUPPORTED, 1 CONTRADICTED, 6 INSUFFICIENT) with per-claim `claim_text_sha256` binding;
- emitted historical pending commitment `verification-v2-d3-holdout-label-commitment.json` (SHA-256 `c7755e37e394e80484f73c52ee6965c34c65917c38fa83b1dc453bbb466bcf86`, size `823 bytes`, status `FROZEN_PENDING_EXTERNAL_REVIEW`);
- completed external chain-of-custody review (decision: `APPROVED`);
- created approved commitment tracked in repository at `configs/verification-v2-d3-holdout-label-commitment.json` (SHA-256 `5cc7f58ed52c43d091bce98d5296ab68cded981495736a479644aefc1428b6dc`, size `1,060 bytes`, status `EXTERNALLY_REVIEWED_FOR_H_EXEC`);
- candidate V2-D3 implementation (`a6e8bca1...`), prompt (`546cd8bd...`), schema (`3591144a...`), model revision (`a1d308df...`), and rate gates remain strictly frozen;
- no model inference (no Qwen, no D3, no Kaggle) executed;
- next action: external review of new execution-authority commit, then canonical one-shot Kaggle H-EXEC.

## 77. Milestone 51.12 — Priority B: Phase H-EXEC Attempt 0 Invalidation, Pre-Inference Provider Harness Correction, Preflight Constructor Smoke Gate, and Recovery Attempt 1 Authorization

**Status:** Completed & Ready for Review (Harness: `scripts/evaluate_verification_v2_d3_holdout.py`, Unit Tests: `tests/unit/evaluation/test_verification_v2_d3_holdout.py`, Decision: `D126`, Protocol: `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`)

- recorded and invalidated Attempt 0 (`H_EXEC_ATTEMPT_0_INVALID_PRE_INFERENCE_HARNESS_FAILURE` on commit `21b7ffcf10d4621b0fdcbf18dcd565e4d5186699`) due to synchronous `TypeError: TransformersChatProvider.__init__() got an unexpected keyword argument 'model_name'` before provider instantiation or Pass 1 start (0 provider calls, 0 D3 predictions, 0 holdout metrics; zero holdout scientific results consumed);
- ported proven provider construction logic from `scripts/evaluate_verification_v2_d3_development.py` to `scripts/evaluate_verification_v2_d3_holdout.py`, instantiating `TransformersChatProvider` via `SemanticVerificationConfig.as_generation_config()` with fail-closed GenerationConfig invariant assertions;
- added a model-free constructor-contract smoke gate in `--preflight-only` before declaring readiness (recording `provider_constructor_contract_verified: True`);
- added regression tests (`test_init_v3_provider_real_construction_contract`, `test_init_v3_provider_invalid_generation_config_fails_closed`, `test_preflight_verifies_provider_constructor_contract`) with monkeypatched runtime-load guards to guarantee zero weight loading during construction/preflight;
- verified `TransformersChatProvider` source, candidate V2-D3 implementation (`a6e8bca1...`), prompt (`546cd8bd...`), schema (`3591144a...`), frozen labels (`85d348db...`), tracked commitment (`5cc7f58ed5...`), and rate gates remain 100% byte-identical;
- no model inference (no Qwen, no D3, no Kaggle) executed;
- authorized exactly ONE recovery run as **H-EXEC Recovery Attempt 1** on a fresh Kaggle GPU session following external review of this corrected commit;
- next action: external review of corrected execution-authority commit, then canonical one-shot Kaggle H-EXEC Recovery Attempt 1.

## 78. Milestone 51.13 — Priority B: V2-D3 Fresh-Holdout Evaluation Closure, Promotion Rejection, Holdout Burning, and Forensic Failure Analysis

**Status:** Completed (Evidence Archive: `verification-v2-d3-holdout-evidence.zip` SHA-256 `9e2b38d4189f9c68901051a07b999845c660ec6ab4b4fa1e6ec69d3088fe6a5d`, Size `10,463 bytes`, Commit `77561aa7c4b242e12d011a84a21f3a262a17a0f8`, Postmortem: `docs/32-V2-D3-HOLDOUT-CLOSURE-AND-POSTMORTEM.md`, Decision: `D127`)

- formally verified canonical fresh-holdout evidence archive `verification-v2-d3-holdout-evidence.zip` (SHA-256 `9e2b38d4189f9c68901051a07b999845c660ec6ab4b4fa1e6ec69d3088fe6a5d`, 10,463 bytes, 9 members) executed on Kaggle GPU on reviewed authority commit `77561aa7c4b242e12d011a84a21f3a262a17a0f8`;
- recorded final holdout evaluation verdict: `V2_D3_HOLDOUT_EXECUTION_FAILURE`, decision: `REJECT_V2_D3_PROMOTION`, `promotion_recommended: false`, `promotion_authorized: false`;
- recorded exact Pass 1 scientific metrics: Supported Retention Rate = 22/23 = 95.65% (PASS), Negative Catch Rate = 2/7 = 28.57% (FAILED vs 50.0% minimum gate), Valid Answer Retention = 8/10 = 80.0% (PASS), Full Answer Accuracy = 10/16 = 62.50% (PASS), Claim Binary Accuracy = 24/30 = 80.00% (PASS);
- recorded operational telemetry: 61/62 provider calls succeeded, 0 retries, 1 call failure (`103383:PRIMARY:C1`) due to transient cold-start runtime loading on Call 1 (duration 54.55s), 30/30 stable claims;
- isolated operational failure from semantic quality: confirmed that eliminating runtime errors does not repair the failed negative catch rate;
- permanently closed V2-D3 development track, burned the 31 holdout claims for use as diagnostic development data only, and kept production semantic verifier disabled;
- conducted forensic failure analysis across all 9 non-trivial cases: classified 5 False Accepts (3x `ACTOR_ROLE_MISMATCH`, 1x `CONDITION_EXCEPTION_OMITTED`, 1x `ACTION_OBJECT_MISMATCH`/`QUANTITY_TEMPORAL_MISMATCH`), 1 False Reject (`SYNTAX_FRAGMENT_STRICTNESS`), and 2 True Negatives;
- ranked root causes: #1 Lack of explicit legal dimension decomposition, #2 Lexical entailment bias, #3 Context-blind fragment evaluation;
- proposed and recommended V3 architecture: Option C (Structured Dimension Decomposition with 3 boolean predicates and deterministic aggregation);
- assessed system-level engineering ROI: recommended shifting primary focus to Generation Grounding / Prompt Optimization (Task 2 metric leverage) and Retrieval / Reranking depth tuning;
- no production code or V3 implementation added in this diagnostic task;
- next action: shift engineering focus to Generation Grounding & Prompt Optimization, with V3 verifier design planned modularly.
