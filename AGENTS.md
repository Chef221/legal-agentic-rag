# AGENTS.md

## 1. Purpose of This File

File này là hướng dẫn bắt buộc dành cho mọi coding agent, AI assistant
và developer làm việc trong repository.

Mọi thay đổi đối với code, configuration, tests, scripts hoặc tài liệu
phải tuân thủ các quy tắc trong file này.

Coding agent phải đọc file này trước khi:

- phân tích repository;
- đề xuất kiến trúc;
- tạo file;
- sửa file;
- xóa file;
- cài dependency;
- chạy command;
- viết test;
- triển khai model;
- thay đổi configuration;
- thay đổi pipeline.

Không được xem prompt hội thoại hiện tại là nguồn sự thật duy nhất.

Nguồn sự thật lâu dài của dự án là:

1. `AGENTS.md`;
2. các file trong `docs/`;
3. source code và tests đã được phê duyệt;
4. các quyết định mới được người dùng xác nhận rõ ràng.

---

## 2. Project Mission

Dự án xây dựng hệ thống Agentic Retrieval-Augmented Generation cho bài
toán trả lời câu hỏi pháp luật Việt Nam.

Mục tiêu dài hạn là phục vụ UIT Data Science Challenge 2026 với chủ đề:

> Trả lời câu hỏi pháp luật Việt Nam.

Hệ thống phải có khả năng:

1. truy xuất các căn cứ pháp luật liên quan;
2. xếp hạng evidence;
3. sinh câu trả lời dựa trên evidence;
4. cung cấp citation;
5. cảnh báo khi evidence không đầy đủ;
6. hỗ trợ điều phối retrieval bằng Agent ở giai đoạn sau;
7. thích ứng với dữ liệu và format của Ban tổ chức khi được công bố.

---

## 3. Current Competition Context

Tại thời điểm hiện tại, Ban tổ chức chưa công bố đầy đủ:

- quy chế chính thức;
- train data;
- development data;
- test data;
- corpus pháp luật;
- ground-truth granularity;
- metric đánh giá;
- input format;
- output format;
- submission format;
- giới hạn tài nguyên;
- quy định external data;
- quy định pretrained model;
- quy định Internet;
- quy định commercial LLM API;
- yêu cầu Docker hoặc API deployment.

Do đó:

- không được thiết kế core system gắn cứng với dataset hiện tại;
- không được mặc định BTC sẽ cung cấp train data;
- không được mặc định BTC chỉ cung cấp test data;
- không được mặc định output là answer text;
- không được mặc định output là article IDs;
- không được mặc định ground truth ở cấp Điều;
- không được tối ưu theo một metric chưa được công bố.

Mọi thành phần liên quan tới competition phải được thiết kế thông qua:

- input adapter;
- corpus adapter;
- training adapter;
- evaluator;
- output adapter;
- submission formatter.

---

## 4. Current Data Scope

Trong giai đoạn hiện tại, nguồn dữ liệu duy nhất được sử dụng là:

`th1nhng0/vietnamese-legal-documents`

Dataset này được dùng làm:

- corpus văn bản pháp luật;
- nguồn metadata pháp lý;
- nguồn nội dung toàn văn;
- nguồn quan hệ giữa các văn bản;
- nguồn xây BM25 index;
- nguồn xây vector index;
- nguồn xây legal graph;
- nguồn evidence cho answer generation;
- nguồn metadata cho citation.

Không tích hợp bất kỳ dataset QA hoặc corpus pháp luật nào khác trong
baseline hiện tại.

Đặc biệt:

- không tích hợp bộ dữ liệu GitHub đã thảo luận trước đây;
- không trộn hai corpus;
- không xây mapping sang corpus khác;
- không thêm VLQA vào pipeline;
- không tạo synthetic QA làm ground truth chính thức.

Nếu phạm vi dữ liệu thay đổi, phải:

1. có xác nhận rõ ràng từ người dùng;
2. cập nhật `docs/08-DESIGN-DECISIONS.md`;
3. cập nhật tài liệu liên quan;
4. chỉ sau đó mới thay đổi code.

---

## 5. Dataset Logical Structure

Dataset hiện tại được hiểu theo ba thành phần logic:

### 5.1 Metadata

Chứa metadata của văn bản pháp luật.

Khóa định danh logic:

```text
metadata.id
```

### 5.2 Content

Chứa nội dung toàn văn, có thể ở dạng HTML.

Khóa nối logic:

```text
content.id → metadata.id
```

### 5.3 Relationships

Chứa quan hệ có hướng giữa các văn bản.

Khóa nối logic:

```text
relationships.doc_id → metadata.id
relationships.other_doc_id → metadata.id
```

Tên config, tên cột và data type thực tế phải được xác minh bằng loader
và audit.

Không được hard-code dựa hoàn toàn vào mô tả tài liệu.

---

## 6. Core System Tasks

Hệ thống gồm hai nhiệm vụ chính.

### 6.1 Task 1 — Legal Evidence Retrieval

Input:

- câu hỏi pháp luật bằng tiếng Việt.

Output:

- ranked legal chunks;
- document metadata;
- article metadata;
- retrieval score;
- rank;
- retrieval strategy;
- retrieval trace;
- artifact version.

### 6.2 Task 2 — Grounded Legal Answer Generation

Input:

- câu hỏi;
- selected evidence;
- legal metadata.

Output:

- câu trả lời bằng tiếng Việt;
- citations;
- warnings;
- insufficient-evidence status;
- trace ID.

Agent không phải một task độc lập.

Agent chỉ là lớp điều phối các module đã tồn tại.

---

## 7. Mandatory Development Principle

Thứ tự phát triển bắt buộc:

```text
Documentation
→ Unified Schema
→ Project Scaffold
→ Dataset Loader
→ Dataset Audit
→ Document Normalization
→ HTML Cleaning
→ Legal Structure Parsing
→ Legal Chunking
→ BM25 Index
→ Vector Index
→ Fixed Retrieval
→ RRF Fusion
→ Reranking
→ Graph Index
→ Graph Retrieval
→ Context Builder
→ Answer Generator
→ Citation Verifier
→ Tool Wrappers
→ Agentic Workflow
→ API and UI
→ Evaluation
→ Competition Adaptation
```

Không được:

- bỏ qua tầng dữ liệu;
- bỏ qua schema;
- viết Agent trước retrieval;
- viết UI trước core pipeline;
- dùng Agent để che giấu retrieval chưa hoàn chỉnh;
- triển khai end-to-end lớn trước khi từng module có test.

---

## 8. Offline and Online Separation

Offline phase và online phase phải tách biệt.

### 8.1 Offline Phase

Offline phase chịu trách nhiệm:

- tải dataset;
- audit dữ liệu;
- normalize dữ liệu;
- clean HTML;
- parse cấu trúc pháp lý;
- chunk;
- validate chunk;
- embedding corpus;
- build BM25 index;
- build vector index;
- normalize relationships;
- build graph index;
- tạo manifests;
- validate artifacts.

### 8.2 Online Phase

Online phase chịu trách nhiệm:

- validate query;
- normalize query;
- retrieve;
- fuse;
- rerank;
- graph expansion;
- context building;
- context grading;
- answer generation;
- citation verification;
- response packaging;
- tracing.

### 8.3 Prohibited Online Operations

Online pipeline không được:

- download dataset;
- audit toàn corpus;
- clean HTML toàn corpus;
- chunk lại toàn corpus;
- embedding lại toàn corpus;
- build lại BM25 index;
- build lại vector index;
- build lại graph;
- sửa raw data;
- ghi thay đổi vào corpus.

---

## 9. Unified Schema Rule

Mọi dữ liệu phải được chuyển về unified schema trước khi đi vào core.

Raw field names chỉ được xuất hiện trong:

- dataset loader;
- dataset adapter;
- audit module;
- raw fixtures.

Các module sau không được biết tên cột raw:

- cleaner;
- parser;
- chunker;
- indexer;
- retriever;
- reranker;
- generator;
- verifier;
- Agent;
- API.

Unified schema được định nghĩa trong:

`docs/07-UNIFIED-SCHEMA.md`

Nếu implementation khác tài liệu:

- dừng thay đổi;
- báo mâu thuẫn;
- không tự chọn bên nào;
- chờ xác nhận hoặc cập nhật tài liệu.

---

## 10. Legal Text Preservation Rules

Trong cleaning, parsing và chunking, phải giữ các thành phần quan trọng:

- tên văn bản;
- số ký hiệu;
- loại văn bản;
- ngày ban hành;
- ngày có hiệu lực;
- ngày hết hiệu lực;
- trạng thái hiệu lực;
- Chương;
- Mục;
- Tiểu mục;
- Điều;
- Khoản;
- Điểm;
- tên Điều;
- mức tiền;
- tỷ lệ;
- thời hạn;
- ngày tháng;
- tên cơ quan;
- đối tượng áp dụng;
- từ phủ định;
- từ chỉ ngoại lệ;
- từ chỉ điều kiện;
- bảng pháp lý có ý nghĩa.

Không được thực hiện mặc định:

- bỏ dấu tiếng Việt;
- xóa số;
- xóa dấu câu pháp lý;
- xóa từ phủ định;
- stemming;
- aggressive stopword removal;
- lowercase toàn bộ mà không có lý do;
- merge nội dung làm mất ranh giới Điều/Khoản;
- sửa nội dung pháp lý bằng suy đoán.

Các từ đặc biệt không được làm mất gồm:

```text
không
chưa
trừ
ngoại trừ
phải
được
chỉ khi
trong trường hợp
không áp dụng
hết hiệu lực
bãi bỏ
thay thế
sửa đổi
```

---

## 11. Legal Chunking Rules

Retrieval unit ưu tiên:

```text
1 chunk = 1 Điều
```

Nếu Điều quá dài:

```text
1 chunk = một hoặc nhiều Khoản liên tiếp
```

Nếu một Khoản vẫn quá dài:

```text
token-based splitting
```

Token-based splitting chỉ là fallback.

Mỗi chunk phải:

- có `chunk_id` deterministic;
- có `document_id`;
- giữ legal hierarchy;
- giữ document title;
- giữ document number;
- giữ article number nếu có;
- giữ clause number nếu có;
- có `text`;
- có `search_text`;
- có token count;
- truy ngược được về document;
- có source provenance;
- có effect status nếu có.

Không dùng line number hoặc row index làm ID chính nếu không ổn định.

---

## 12. Retrieval Architecture Rules

Fixed retrieval baseline phải gồm:

1. BM25 retrieval;
2. dense retrieval;
3. hybrid retrieval bằng RRF;
4. cross-encoder reranking;
5. graph-expanded retrieval.

### 12.1 BM25

BM25 dùng lexical matching.

Không yêu cầu training.

### 12.2 Dense Retrieval

Dense retrieval dùng embedding model pretrained ở baseline đầu tiên.

Embedding provider phải đi qua interface.

Không hard-code model vào core.

### 12.3 Hybrid Retrieval

Hybrid baseline phải dùng Reciprocal Rank Fusion.

Không cộng trực tiếp:

```text
BM25 raw score + dense similarity score
```

Mỗi result phải lưu contribution từ từng nhánh.

### 12.4 Reranker

Reranker chỉ chạy trên candidate set giới hạn.

Không chạy cross-encoder trên toàn corpus.

Reranker backend phải đi qua interface.

### 12.5 Graph Retrieval

Graph không thay thế BM25 hoặc dense.

Graph retrieval phải:

- bắt đầu từ seed documents;
- seed đến từ text retrieval;
- giới hạn hop;
- mặc định 1 hop;
- tối đa 2 hop trong baseline;
- lưu graph path;
- lưu relationship type;
- tránh mở rộng toàn graph.

---

## 13. Backend Abstraction Rules

Các backend sau phải đi qua interface hoặc protocol:

- dataset loader;
- BM25 backend;
- embedding provider;
- vector backend;
- reranker;
- graph backend;
- answer generator;
- context grader;
- citation verifier;
- Agent framework;
- tracing backend.

Core không được phụ thuộc trực tiếp vào:

- Chroma;
- Elasticsearch;
- FAISS;
- Milvus;
- Neo4j;
- NetworkX;
- một model Hugging Face cụ thể;
- một LLM API cụ thể;
- LangGraph;
- LangChain.

Backend cụ thể có thể được triển khai trong adapter hoặc provider riêng.

---

## 14. Current Unresolved Technology Decisions

Các thành phần sau chưa được chốt chính thức:

- BM25 backend;
- vector database;
- graph database;
- embedding model;
- reranker model;
- generator model;
- context grader model;
- Agent framework;
- API deployment strategy;
- artifact storage format.

Coding agent không được tự coi một lựa chọn là quyết định cuối cùng.

Nếu cần prototype:

1. đề xuất lựa chọn;
2. giải thích lý do;
3. liệt kê trade-off;
4. chờ xác nhận;
5. cập nhật design decision;
6. sau đó mới triển khai.

Không được tự thêm dependency nặng chỉ để thử nghiệm.

---

## 15. Model Training Rules

Baseline đầu tiên không fine-tune:

- dense retriever;
- reranker;
- generator;
- context grader.

Lý do:

- chưa có gold competition data;
- dataset hiện tại chủ yếu là corpus;
- chưa có benchmark chính thức;
- chưa có metric chính thức.

Fine-tuning chỉ được thêm khi có:

- supervision phù hợp;
- dataset hợp lệ;
- train/dev split;
- evaluator;
- experiment plan;
- xác nhận từ người dùng.

Sau khi thay embedding model hoặc fine-tune retriever:

- phải re-embed corpus;
- rebuild vector index;
- cập nhật artifact manifest;
- không được dùng index cũ.

---

## 16. Agent Rules

Agent chỉ được triển khai sau fixed baseline.

Agent được phép:

- phân tích query;
- chọn tool;
- gọi tool;
- quan sát result;
- grade context;
- rewrite query;
- đổi strategy;
- retry có giới hạn;
- gọi generator;
- gọi verifier;
- tạo trace.

Agent không được:

- tải dataset;
- clean corpus;
- chunk corpus;
- build index;
- sửa artifact;
- sửa configuration;
- truy cập raw database client;
- gọi tool không đăng ký;
- tự cài package;
- tự bật web search;
- tự thêm external data;
- lặp vô hạn.

Baseline Agent phải có:

```text
max_retry = 2
```

Giá trị này có thể cấu hình, nhưng không được bỏ giới hạn.

Agent phải có stopping conditions rõ ràng.

---

## 17. Generation Rules

Answer generator phải:

- chỉ dùng selected evidence;
- trả lời bằng tiếng Việt;
- trích dẫn theo evidence ID;
- không tạo citation giả;
- không tự thêm số Điều;
- không tự thêm tên luật;
- không che giấu thiếu evidence;
- cảnh báo nếu evidence có hiệu lực không rõ;
- hỗ trợ abstention.

Khi evidence không đủ, output phải thể hiện:

```text
insufficient_evidence = true
```

Không được đoán để tạo một câu trả lời có vẻ hợp lý.

---

## 18. Citation Rules

Mọi citation phải:

- trỏ tới evidence tồn tại;
- trỏ tới chunk tồn tại;
- trỏ tới document tồn tại;
- khớp article number nếu có;
- khớp document number nếu có;
- giữ source URL nếu có.

Baseline citation verifier có thể rule-based.

Không được tuyên bố semantic verification đã hoàn chỉnh nếu mới chỉ kiểm
tra ID.

---

## 19. Artifact Rules

Mọi artifact persisted phải có manifest.

Artifact gồm:

- normalized documents;
- cleaned documents;
- parsed legal blocks;
- legal chunks;
- BM25 index;
- vector index;
- graph index;
- embedding outputs;
- relationship mappings.

Manifest tối thiểu phải ghi:

- schema version;
- artifact type;
- artifact version;
- dataset name;
- dataset revision;
- processing config hash;
- code version;
- created time;
- record count;
- backend;
- model name nếu có;
- model revision nếu có;
- warnings.

Online pipeline phải từ chối load artifact không tương thích.

Không silently overwrite artifact cũ.

---

## 20. Configuration Rules

Không hard-code:

- đường dẫn;
- batch size;
- top-k;
- candidate-k;
- max token;
- overlap;
- RRF constant;
- hop limit;
- model name;
- device;
- timeout;
- API key;
- artifact path.

Các tham số phải đi qua configuration.

Configuration phải:

- có default hợp lý;
- có validation;
- có environment override khi cần;
- không chứa secrets trong repository;
- có config hash khi build artifact.

---

## 21. Dependency Rules

Trước khi thêm dependency:

1. kiểm tra dependency hiện có;
2. xác định dependency có thực sự cần thiết không;
3. ưu tiên thư viện nhẹ;
4. tránh dependency trùng chức năng;
5. kiểm tra license nếu có model hoặc package đặc biệt;
6. giải thích dependency mới trong báo cáo thay đổi.

Không được tự cài:

- framework Agent;
- vector database;
- graph database;
- model nặng;
- web crawler;
- OCR library;
- LLM SDK;

nếu chưa thuộc milestone hiện tại.

---

## 22. Coding Standards

### 22.1 Python

- sử dụng type hints;
- public functions phải có docstring;
- tên rõ nghĩa;
- tránh function quá dài;
- tránh class chỉ để chứa static methods;
- ưu tiên composition;
- không dùng mutable default arguments;
- không nuốt exception;
- không dùng bare `except`;
- không dùng `print` cho production logging;
- không dùng global mutable state;
- không hard-code path.

### 22.2 Module Design

Mỗi module phải có trách nhiệm rõ ràng.

Không tạo các file kiểu:

```text
utils.py
helpers.py
common.py
misc.py
```

chứa quá nhiều logic không liên quan.

Utility phải được đặt theo domain cụ thể.

### 22.3 Interfaces

Interfaces chỉ nên được tạo khi có ranh giới backend hoặc nhiều
implementation hợp lý.

Không over-engineer bằng abstract class cho mọi function.

### 22.4 Comments

Comment phải giải thích:

- lý do;
- invariants;
- edge case;
- constraint pháp lý;
- trade-off.

Không comment lại điều code đã thể hiện rõ.

---

## 23. Error Handling Rules

Mọi lỗi phải được phân loại rõ:

- configuration error;
- dataset schema error;
- data validation error;
- artifact compatibility error;
- backend initialization error;
- retrieval error;
- model error;
- timeout;
- external service error;
- invalid user input.

Không silently skip record nếu chưa có policy.

Nếu record bị bỏ:

- phải có reason;
- phải được log;
- phải xuất audit issue nếu thuộc offline phase.

Error message không được lộ:

- API key;
- token;
- secret;
- local sensitive path;
- raw stack trace cho end user.

---

## 24. Logging and Observability Rules

Không dùng `print` cho production workflow.

Logging cần có:

- timestamp;
- level;
- module;
- trace ID nếu online;
- document ID nếu offline;
- chunk ID nếu liên quan;
- strategy;
- latency;
- error type.

Online trace cần hỗ trợ:

- query ID;
- normalized query;
- selected strategy;
- candidate count;
- retrieval ranks;
- RRF contributions;
- reranker scores;
- graph hops;
- selected evidence;
- retry count;
- warnings;
- total latency.

Không log toàn bộ user query hoặc legal content trong môi trường có yêu
cầu privacy nếu chưa có policy.

---

## 25. Testing Rules

Mỗi milestone phải có tests phù hợp.

### 25.1 Unit Tests

Unit tests phải:

- dùng fixture nhỏ;
- không tải full dataset;
- không gọi network mặc định;
- không gọi model lớn;
- deterministic;
- chạy nhanh;
- kiểm tra edge cases.

### 25.2 Integration Tests

Integration tests có thể:

- dùng sample dataset;
- build index nhỏ;
- test persistence;
- test reload;
- test end-to-end trên fixture.

### 25.3 Prohibited Test Practices

Không được:

- dùng mock để che giấu production logic chưa viết;
- assert `True`;
- bỏ qua test lỗi mà không ghi lý do;
- phụ thuộc vào thứ tự test;
- gọi dịch vụ trả phí;
- download model nặng trong test mặc định;
- coi smoke test là benchmark chất lượng.

### 25.4 Minimum Tests by Module

Loader:

- schema mapping;
- missing field;
- duplicate ID;
- sample limit.

Cleaner:

- remove script;
- preserve Điều/Khoản;
- preserve Unicode;
- preserve numbers;
- deterministic output.

Parser:

- standard structure;
- missing levels;
- malformed numbering;
- no-structure document.

Chunker:

- deterministic IDs;
- article chunk;
- clause fallback;
- token fallback;
- metadata inheritance.

Retriever:

- output schema;
- ranking;
- empty query;
- top-k;
- persistence and reload.

Agent:

- retry limit;
- stopping conditions;
- tool restriction;
- error path.

---

## 26. Documentation Rules

Mọi thay đổi kiến trúc phải cập nhật:

`docs/08-DESIGN-DECISIONS.md`

Mọi thay đổi schema phải cập nhật:

`docs/07-UNIFIED-SCHEMA.md`

Mọi thay đổi pipeline phải cập nhật:

- `docs/05-OFFLINE-PIPELINE.md`;
- hoặc `docs/06-ONLINE-PIPELINE.md`.

Mọi thay đổi milestone phải cập nhật:

`docs/09-IMPLEMENTATION-PLAN.md`

Không để code và docs mâu thuẫn lâu dài.

Không cập nhật docs bằng các tuyên bố không đúng với implementation.

---

## 27. Source-of-Truth Priority

Khi có mâu thuẫn, dùng thứ tự ưu tiên:

1. yêu cầu rõ ràng mới nhất của người dùng;
2. quyết định `Accepted` trong `docs/08-DESIGN-DECISIONS.md`;
3. `AGENTS.md`;
4. unified schema;
5. pipeline docs;
6. implementation plan;
7. source code hiện tại;
8. prompt cũ hoặc comment cũ.

Nếu hai nguồn ở cùng mức mâu thuẫn:

- không tự chọn;
- báo rõ;
- yêu cầu xác nhận;
- không thay đổi code liên quan.

---

## 28. Required Reading by Task

### Dataset Work

Phải đọc:

- `AGENTS.md`;
- `docs/03-DATASET-AIO.md`;
- `docs/05-OFFLINE-PIPELINE.md`;
- `docs/07-UNIFIED-SCHEMA.md`;
- `docs/08-DESIGN-DECISIONS.md`.

### Retrieval Work

Phải đọc:

- `AGENTS.md`;
- `docs/02-PROBLEM-DEFINITION.md`;
- `docs/04-SYSTEM-ARCHITECTURE.md`;
- `docs/06-ONLINE-PIPELINE.md`;
- `docs/07-UNIFIED-SCHEMA.md`;
- `docs/08-DESIGN-DECISIONS.md`.

### Agent Work

Phải đọc toàn bộ tài liệu.

Agent work chỉ được bắt đầu khi milestone fixed RAG đã hoàn thành.

### Competition Adaptation

Phải đọc:

- `AGENTS.md`;
- `docs/01-PROJECT-CONTEXT.md`;
- `docs/08-DESIGN-DECISIONS.md`;
- `docs/10-COMPETITION-ADAPTATION.md`.

---

## 29. Mandatory Workflow Before Coding

Trước khi sửa code, coding agent phải:

1. đọc tài liệu liên quan;
2. kiểm tra trạng thái repository;
3. xác định milestone hiện tại;
4. xác định phạm vi;
5. liệt kê giả định;
6. liệt kê file dự kiến thay đổi;
7. nêu test dự kiến;
8. xác định dependency mới nếu có;
9. báo mâu thuẫn nếu có;
10. chỉ bắt đầu code khi phạm vi rõ.

Không được vừa đọc repository vừa tự động sửa hàng loạt file nếu chưa có
kế hoạch.

---

## 30. Mandatory Workflow After Coding

Sau khi sửa code, coding agent phải báo:

1. file đã tạo;
2. file đã sửa;
3. file đã xóa;
4. mục đích từng thay đổi;
5. test đã chạy;
6. kết quả test;
7. command đã chạy;
8. dependency đã thêm;
9. config đã thay đổi;
10. tài liệu đã cập nhật;
11. hạn chế còn lại;
12. phần chưa triển khai;
13. giả định đang giữ;
14. rủi ro kỹ thuật.

Không được chỉ nói “done” hoặc “implemented successfully”.

---

## 31. Scope Control

Mỗi nhiệm vụ chỉ thực hiện đúng milestone được yêu cầu.

Ví dụ, khi làm Dataset Loader:

Được phép:

- loader;
- adapter;
- audit;
- fixtures;
- tests.

Không được:

- cleaner;
- chunker;
- embedding;
- BM25;
- vector database;
- Agent;
- API.

Nếu phát hiện việc ngoài scope cần thiết:

- ghi nhận;
- báo người dùng;
- không tự triển khai.

---

## 32. Current Prohibitions

Cho đến khi có quyết định mới, không được:

- tích hợp dataset ngoài AIO;
- tích hợp dữ liệu GitHub QA;
- tích hợp VLQA;
- fine-tune model;
- xây autonomous Agent;
- xây multi-agent;
- bật web search;
- thêm web crawler;
- thêm OCR;
- parse PDF scan;
- gọi external legal API;
- tự cập nhật hiệu lực pháp lý;
- dùng synthetic QA làm gold benchmark;
- chọn production backend;
- triển khai cloud infrastructure;
- thêm authentication phức tạp;
- tối ưu premature performance;
- thêm cache phân tán;
- thêm message queue;
- thêm microservices.

---

## 33. Competition Adaptation Rule

Khi BTC công bố dữ liệu:

Không sửa core ngay lập tức.

Trước tiên phải xác định:

1. corpus có mới không;
2. có train labels không;
3. có gold answers không;
4. ground truth ở cấp nào;
5. metric là gì;
6. external data có được phép không;
7. API bên ngoài có được phép không;
8. runtime có Internet không;
9. output format là gì;
10. submission type là gì.

Sau đó cập nhật:

- competition docs;
- design decisions;
- adapters;
- evaluator;
- output formatter.

Nếu corpus BTC khác:

- rebuild indexes;
- không tái sử dụng index AIO một cách im lặng.

---

## 34. Security and Secrets

Không commit:

- API keys;
- Hugging Face tokens;
- passwords;
- database credentials;
- private URLs;
- `.env`;
- model license credentials;
- user data.

Dùng:

- `.env.example`;
- environment variables;
- secret manager nếu có;
- local ignored config.

Không hiển thị secret trong log hoặc error.

---

## 35. Data Storage Rules

Không commit:

- full dataset;
- model checkpoints lớn;
- vector index lớn;
- BM25 artifacts lớn;
- graph dump lớn;
- temporary files;
- cache;
- generated logs.

Các thư mục dữ liệu và artifact phải được quản lý bằng `.gitignore`.

Chỉ commit:

- fixtures nhỏ;
- schema examples;
- manifest examples;
- test samples;
- scripts tái tạo.

---

## 36. Performance Rules

Không tối ưu sớm khi chưa có baseline đúng.

Trước khi tối ưu phải đo:

- loading time;
- processing throughput;
- memory usage;
- embedding throughput;
- retrieval latency;
- reranking latency;
- end-to-end latency.

Không thay đổi thuật toán chỉ dựa trên cảm giác nhanh/chậm.

Batching, streaming, multiprocessing và caching chỉ được thêm khi có
bottleneck đo được.

---

## 37. Definition of Done

Một milestone chỉ được xem là hoàn thành khi:

- đúng phạm vi;
- đúng unified schema;
- không vi phạm design decisions;
- implementation không phải fake;
- unit tests pass;
- integration tests liên quan pass;
- lint/type checks pass nếu đã cấu hình;
- error cases được xử lý;
- config được validate;
- artifact hoặc output có manifest nếu cần;
- tài liệu được cập nhật;
- hạn chế được ghi rõ;
- không có secret;
- không có dữ liệu lớn bị commit;
- kết quả có thể tái tạo.

Nếu test chưa chạy được, milestone chưa được xem là hoàn thành.

---

## 38. Current Project Status

Trạng thái hiện tại:

```text
Milestone 1 — Project Scaffold (Completed)
```

Milestone 1 đã tạo:

- Python `src/` package structure;
- Pydantic v2 unified schemas;
- typed configuration schemas;
- backend contracts bằng `typing.Protocol`;
- exception taxonomy;
- standard-library logging foundation;
- unit và integration test structure;
- small unified-schema fixtures.

Milestone 1 không triển khai business logic.

Hiện chưa được phép triển khai nếu chưa có yêu cầu milestone mới:

- dataset loader production và dataset audit;
- cleaner;
- parser;
- chunker;
- index;
- retrieval;
- generator;
- Agent;
- API.

Bước tiếp theo chỉ sau khi người dùng phê duyệt phạm vi:

```text
Milestone 2 — Dataset Loader and Audit
```

---

## 39. Final Instruction to Coding Agents

Khi không chắc chắn:

- không đoán;
- không tự mở rộng kiến trúc;
- không thêm framework;
- không thêm dependency;
- không biến giả định thành quyết định;
- không viết code để “chuẩn bị trước” cho tính năng chưa được yêu cầu.

Hãy báo rõ:

- điều chưa rõ;
- các phương án;
- trade-off;
- tác động;
- quyết định cần người dùng xác nhận.

Mục tiêu không phải tạo nhiều code nhất.

Mục tiêu là xây một hệ thống đúng, kiểm thử được, truy vết được và có
thể thích ứng với dữ liệu chính thức của cuộc thi.
