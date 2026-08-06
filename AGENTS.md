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

Contract dữ liệu BTC được tóm tắt bắt buộc tại
`docs/13-UIT-DSC-2026-DATA-CONTRACT.md`. File overview gốc của BTC vẫn là bằng
chứng bên ngoài repository; nếu dữ liệu thật khác ví dụ, phải audit và cập nhật
decision trước khi thay đổi core.

Contract scorer BTC được tóm tắt bắt buộc tại
`docs/15-OFFICIAL-SCORING-CONTRACT.md`. Mọi scorer artifact mới phải được
checksum/diff; không được suy ra parity từ tên metric hoặc log cũ.

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

BTC UIT Data Science Challenge 2026 Task 2 đã công bố data overview,
`warmup.json`, `train.json`, `public-official.json` và
`selected-contexts.zip`. Các thông tin đã xác nhận:

- input là câu hỏi pháp luật tiếng Việt;
- output là câu trả lời văn xuôi tiếng Việt;
- corpus chính thức nằm trong `selected-contexts.zip`, có 8.532 context và
  canonical revision
  `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`;
- context có raw fields bắt buộc `id`, `link`, `passage`; `name` optional;
- raw context ID trong archive là non-negative integer và được adapter
  canonicalize sang unified string ID;
- 20 passage rỗng, 1.125 context thiếu name; không drop hoặc phát minh dữ liệu;
- `train.json` có 7.000 answer-labeled records, public có 1.000 question-only
  records và không có evidence/relevance labels;
- warm-up/train/public có overlap, nên mọi split/evaluation phải kiểm soát
  leakage;
- dữ liệu theo giai đoạn gồm warm-up, public test và private test;
- METEOR là metric xếp hạng chính, ROUGE-L là metric phụ;
- source scorer BTC checksum
  `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`
  dùng NLTK METEOR trên whitespace tokens, vendored ASCII-tokenized ROUGE-L và
  arithmetic macro mean; PyVi bị comment và không chạy;
- Codabench nhận `submission.zip` chứa duy nhất UTF-8 `submission.json`;
- scorer Codabench thực tế yêu cầu `submission.json` là object ánh xạ mỗi
  question ID sang đúng `{"answer": string}`; hướng dẫn dạng mảng đã bị scorer
  thực tế bác bỏ.
- model mã nguồn mở không có danh sách cố định nhưng phải đăng ký tên và URL qua
  Google Form BTC sẽ cung cấp;
- preprocessing, indexing, retrieval và fine-tuning được phép nếu chỉ dùng dữ
  liệu chính thức;
- synthetic data bị cấm kể cả khi sinh từ dữ liệu BTC;
- BTC sẽ cung cấp dữ liệu huấn luyện cho Task 2;
- tổng số tham số của tất cả model trong một hệ thống Task 2 phải nhỏ hơn
  4 tỷ, bao gồm embedding, reranker, generator, verifier/grader dùng model và
  mọi model phụ trợ khác;
- quantization, LoRA hoặc kỹ thuật giảm bộ nhớ không làm giảm số tham số được
  BTC tính; model gốc từ 4 tỷ tham số trở lên vẫn không hợp lệ;
- model distilled được phép nếu chính model/hệ thống cuối cùng vẫn có tổng số
  tham số nhỏ hơn 4 tỷ;
- cấm sử dụng mọi API model hoặc sản phẩm trung gian, kể cả API miễn phí hay
  phi lợi nhuận; đội phải tải, chạy và kiểm soát trực tiếp model/hệ thống;
- model mã nguồn mở hoặc có giấy phép phi thương mại/nghiên cứu/giáo dục được
  chấp nhận, nhưng vẫn phải qua đăng ký model và kiểm tra license;
- dữ liệu dùng để pretrain model/LLM không bị BTC xem là dữ liệu ngoài; đội
  không được trực tiếp đưa dữ liệu ngoài hoặc data augmentation vào pipeline;
- Docker không phải hình thức đóng gói duy nhất: GitHub hoặc ZIP source/weights
  cũng được chấp nhận nếu BTC có thể tái lập thực nghiệm theo README;
- được tải trọng số model hợp lệ từ Internet khi tái lập; quy trình tải, exact
  model identity và từng bước thực thi phải được ghi rõ;
- Codabench là nền tảng nộp chính thức của Task 2.

Các nội dung vẫn chưa được xác nhận đầy đủ:

- exact NLTK/NumPy versions, WordNet/OMW bytes và việc public/private có giữ
  đúng scorer checksum đã phân tích; lỗi WordNet warm-up đã được BTC sửa;
- giới hạn GPU, RAM, disk, thời gian thực thi và Internet trong môi trường chấm;
- Google Form và quy trình xác nhận đăng ký model;
- private-test data và thay đổi dữ liệu/scorer ở phase sau.

Core không được gắn cứng với raw schema của BTC. Mọi dữ liệu competition phải
đi qua adapter, evaluator và output boundary.

Mọi thành phần liên quan tới competition phải được thiết kế thông qua:

- input adapter;
- corpus adapter;
- training adapter;
- evaluator;
- output adapter;
- submission formatter.

---

## 4. Current Data Scope

Kể từ Milestone 25, phạm vi active của dự án là dữ liệu chính thức do BTC UIT
Data Science Challenge 2026 Task 2 cung cấp.

Chính sách mặc định là competition-only và fail-closed:

- không dùng corpus AIO trong build, runtime, evaluation hoặc submission;
- không load artifact có lineage từ corpus ngoài BTC;
- không trộn external corpus;
- không dùng synthetic QA làm gold benchmark;
- không tạo bất kỳ synthetic QA, answer, evidence, hard negative hoặc training
  example nào, kể cả từ dữ liệu BTC;
- `warmup.json` và `train.json` chỉ là answer-level supervision/evaluation data,
  không phải retrieval labels;
- archive thật đã được audit; official index chỉ được build từ canonical
  revision nêu trên hoặc revision mới đã audit lại.

Code, config, fixture và test dành riêng cho AIO được loại khỏi active tree.
Git history có thể giữ dấu vết lịch sử nhưng không được runtime import hoặc sử
dụng.

Nếu phạm vi dữ liệu thay đổi, phải:

1. có xác nhận rõ ràng từ người dùng;
2. cập nhật `docs/08-DESIGN-DECISIONS.md`;
3. cập nhật tài liệu liên quan;
4. chỉ sau đó mới thay đổi code.

---

## 5. Dataset Logical Structure

Hai contract raw đã được BTC mô tả:

```text
QA mapping: question_id → {question, answer?}
context file: {id, name, link, passage}
```

Ví dụ overview dùng numeric context `id`, `name` dạng slug và `passage` có
CRLF/Unicode cùng cấu trúc văn bản pháp luật. Raw context ID phải được audit;
adapter dự kiến canonicalize integer/string hợp lệ sang unified string ID mà
không làm kiểu raw lan vào core. `link` chỉ là provenance, không cho phép crawl.

Raw field names chỉ được tồn tại trong package adapter UIT DSC 2026. Cấu trúc
thật của corpus, số file, encoding, duplicate policy và record counts phải được
audit lại khi `selected-contexts.zip` được cung cấp; không suy đoán từ overview.

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

Lý do hiện tại:

- chưa có đầy đủ official train/corpus data;
- chưa audit train schema và supervision granularity;
- chưa có train/dev split và experiment plan đáng tin cậy;
- cấm tạo synthetic supervision để bù nhãn còn thiếu.

Fine-tuning chỉ được thêm khi có:

- supervision phù hợp;
- dataset hợp lệ;
- train/dev split;
- evaluator;
- experiment plan;
- xác nhận từ người dùng.

Mọi cấu hình model dùng cho competition phải có model inventory ghi:

- tên, URL và immutable revision;
- vai trò trong hệ thống;
- license;
- số tham số của chính model;
- bằng chứng/cách tính số tham số;
- trạng thái đăng ký với BTC.

Tổng tham số của toàn bộ model trong một Task phải nhỏ hơn `4_000_000_000`.
Embedding, reranker, generator, model-based grader/verifier và mọi model phụ đều
phải cộng vào tổng. Không được dùng quantization, pruning storage hoặc LoRA để
tuyên bố model gốc từ 4 tỷ tham số trở lên đã trở thành model dưới 4 tỷ.

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
- `docs/13-UIT-DSC-2026-DATA-CONTRACT.md`;
- `docs/10-COMPETITION-ADAPTATION.md`;
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
- `docs/10-COMPETITION-ADAPTATION.md`;
- `docs/13-UIT-DSC-2026-DATA-CONTRACT.md`;
- `docs/15-OFFICIAL-SCORING-CONTRACT.md` khi task liên quan evaluation,
  answer rendering hoặc submission scoring.

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

- tích hợp dataset ngoài dữ liệu chính thức của BTC;
- tải hoặc dùng lại AIO corpus/artifacts;
- tích hợp dữ liệu GitHub QA;
- tích hợp VLQA;
- fine-tune model trước khi có official supervision, split, evaluator, experiment
  plan và model registration phù hợp;
- xây autonomous Agent;
- xây multi-agent;
- bật web search;
- thêm web crawler;
- thêm OCR;
- parse PDF scan;
- gọi external legal API;
- gọi bất kỳ model API hoặc sản phẩm AI trung gian nào trong quá trình xây dựng
  phương pháp hay chạy competition;
- dùng cấu hình model có tổng tham số từ 4 tỷ trở lên hoặc chưa kiểm kê được
  tổng tham số;
- dùng model mà đội không thể tải, chạy, kiểm soát và cung cấp quy trình tái lập;
- tự cập nhật hiệu lực pháp lý;
- tạo hoặc dùng synthetic QA, answer, evidence, hard negative hay training data;
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

Khi corpus BTC được cung cấp:

- audit raw schema trước khi map sang unified schema;
- rebuild toàn bộ indexes từ corpus BTC;
- từ chối mọi artifact không có đúng BTC lineage.

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
Milestone 37 — Official Data Adapter and Passage Cleaner
(Adapter/Cleaner Completed; Official Index Build and Model Experiments Pending)
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

Milestone 2 đã tạo:

- production AIO dataset source qua Hugging Face `datasets`;
- raw-field adapter cô lập theo dataset;
- sample limit, streaming, revision và dataset manifest;
- schema, identity, join, content, metadata và relationship audit;
- versioned JSON audit report và CSV issue reports;
- small raw AIO fixtures, unit tests và local integration test.

Milestone 3 đã tạo:

- AIO raw-field projection sang stable normalization inputs;
- deterministic ID, null, date, URL và configurable label normalization;
- conservative metadata-content join;
- duplicate/invalid record rejection có structured issue;
- unified `LegalDocument` outputs và normalized artifact manifest;
- unit tests và loader-to-normalizer integration test.

Milestone 4 đã tạo:

- deterministic HTML cleaner dùng Python standard library;
- explicit tag/class/id và hidden-element noise policy;
- Unicode NFC, entity, whitespace và line-break normalization;
- bảo toàn legal markers, số, phủ định, bảng và visible legal text;
- typed cleaning result, structured issues và cleaned artifact manifest;
- fixture HTML, unit tests và normalizer-to-cleaner integration test.

Milestone 5 đã tạo:

- deterministic line-based Vietnamese legal structure parser;
- hierarchy cho Phần, Chương, Mục, Tiểu mục, Điều, Khoản và Điểm;
- document, appendix và table blocks không chồng lấp text;
- deterministic block IDs, parent links và inherited `LegalStructure`;
- per-document coverage diagnostics và structured parser issues;
- legal-block artifact manifest, fixtures, unit và integration tests.

Milestone 6 đã tạo:

- article-first legal chunker với clause grouping và token fallback;
- dependency-free Unicode baseline tokenizer có bounded overlap;
- deterministic chunk IDs, per-document indexes và search text;
- source block provenance và document/legal metadata inheritance;
- legal chunk validator cùng per-document block coverage diagnostics;
- legal-chunks artifact manifest, unit và integration tests.

Milestone 7 đã tạo:

- SQLite FTS5 reference backend phía sau `BM25Backend`;
- deterministic Unicode analyzer giữ dấu tiếng Việt, số và từ phủ định;
- BM25 build/search cùng unified filters, hit metadata, trace và latency;
- versioned SQLite artifact, JSON manifest và SHA-256 checksum;
- compatibility validation, persistence, reload và no-overwrite policy;
- serialized cross-thread reads cho FastAPI/Gradio shared runtime;
- unit và integration tests từ legal chunks đến persisted BM25 query.

Milestone 7 không triển khai embedding, vector index hoặc hybrid retrieval.

Milestone 8 đã tạo:

- pinned multilingual E5 embedding provider qua `EmbeddingProvider`;
- passage/query prefixes, normalized 384-dimensional embeddings và CPU default;
- batched offline vector index builder;
- NumPy float32 exact cosine backend qua `VectorBackend`;
- versioned vector/chunk artifacts, manifests, checksums và memory-mapped reload;
- strict provider/model/revision/dimension compatibility trước dense query;
- unified filters, dense retrieval trace, model compatibility và total latency;
- unit/integration tests cùng live model smoke test không dùng AIO data.

Milestone 8 không triển khai hybrid RRF, reranking hoặc graph retrieval.

Milestone 9 đã tạo:

- standard unweighted Reciprocal Rank Fusion với configurable constant 60;
- candidate-k retrieval cho BM25/dense và final top-k fusion;
- chunk-ID deduplication cùng per-branch rank, raw score và RRF contribution;
- source-artifact compatibility giữa BM25 và vector index;
- deterministic tie-breaking, namespaced warnings và fail-closed branch errors;
- fixed routing cho BM25, dense và hybrid không cần Agent;
- unit/integration tests dùng persisted reference backends và fixture embedding.

Milestone 9 không triển khai reranking, graph retrieval, generator hoặc Agent.

Milestone 10 đã tạo:

- revision-pinned multilingual cross-encoder qua `Reranker` Protocol;
- typed model, device, batching, max-length và candidate-limit configuration;
- lazy model loading và tách lỗi initialization khỏi lỗi inference;
- bounded hybrid-candidate reranking với deterministic tie-breaking;
- bảo toàn legal payload, BM25/dense/RRF provenance và artifact versions;
- fixed routing cho strategy `hybrid_rerank` không cần Agent;
- unit/integration tests với fixture model và live multilingual model smoke test.

Milestone 10 không triển khai graph retrieval, generator hoặc Agent.

Milestone 11 đã tạo:

- AIO relationship normalization với explicit canonical mapping;
- structured rejection cho invalid endpoint, orphan, self-loop và duplicate;
- versioned relationship artifact với manifest, checksum và no-overwrite;
- persisted directed adjacency graph phía sau `GraphBackend`;
- deterministic BFS 1 hop mặc định và tối đa 2 hop;
- hybrid text seeds, related-document chunk retrieval và bounded final rerank;
- graph hop/path trace cùng fixed strategy `graph` không cần Agent;
- unit/integration tests cho normalization, persistence, traversal và retrieval.

Milestone 11 không triển khai generator, Agent hoặc API.

Milestone 12 đã tạo:

- bounded context builder giữ whole legal chunks và retrieval provenance;
- structural context grader công khai chưa semantic-grade;
- dependency-free extractive generator qua `AnswerGenerator`;
- exact rule-based verifier qua `CitationVerifier`;
- fixed retrieval-to-answer orchestration với deterministic trace ID;
- explicit abstention khi context thiếu hoặc citation verification thất bại;
- response metadata giữ context grade, verification và artifact versions;
- unit/integration tests tới verified grounded answer.

Milestone 12 không triển khai LLM generator, tool wrappers, Agent hoặc API.

Milestone 13 đã tạo:

- closed `ToolName` enum với đúng tám approved capabilities;
- typed inputs, descriptors, invocation results và sanitized errors;
- năm fixed retrieval wrappers;
- context grading, answer generation và citation verification wrappers;
- explicit registry/factory không plugin discovery hoặc dynamic import;
- output-contract validation, timeout budget và safe domain-error mapping;
- payload-free invocation logging;
- unit/integration workflow tests không dùng Agent.

Milestone 13 không triển khai strategy selection, retry workflow, Agent framework
hoặc API.

Milestone 14 đã tạo:

- dependency-free deterministic Agent phía sau `AgentWorkflow` Protocol;
- quality-first route plan chỉ dùng registered retrieval tools;
- conservative query rewrite không tạo thêm legal term;
- `max_retry = 2` và explicit stopping reasons;
- typed terminal `AgentRunResult` giữ answer, state và latency;
- fail-closed abstention khi generation/citation/tool failure;
- payload-free invocation trace và workflow logging;
- unit/integration tests cho retry, stopping, tool restriction và error paths.

Milestone 14 không triển khai runtime assembly, API, UI hoặc LLM generator.

Runtime Assembly đã tạo:

- `OfflineBuildRuntime` ghép AIO source tới BM25/vector/graph artifacts;
- deterministic JSONL persistence cho processed documents/blocks/chunks;
- manifest, checksum, no-overwrite và configurable artifact layout;
- `OnlineRuntimeFactory` load/validate artifacts và ghép complete Agent runtime;
- startup compatibility cho dataset, chunk/index source, graph lineage và
  embedding identity;
- BM25/dense dùng Agent rewritten query khi retry;
- integration test từ raw fixture tới verified answer sau persistence/reload.

Milestone 15 đã tạo:

- FastAPI lifespan load một immutable `OnlineRuntime` và fail-fast startup;
- versioned health, retrieval và answer endpoints;
- typed bounded public request, health và safe-error schemas;
- serving service chuẩn hóa Unicode/whitespace và tạo query ID;
- optional Gradio diagnostic UI dùng chung runtime;
- explicit JSON config loader cùng build/serve CLI;
- localhost-safe defaults và configurable serving limits;
- unit/integration tests cho API lifecycle, errors, health và UI mount.

Milestone 16 đã tạo:

- competition-neutral JSONL evaluation cases với chunk/document labels;
- `RetrievalEvaluator` và `GenerationEvaluator` Protocol;
- Recall@k, Precision@k, MRR và graded NDCG@k;
- available exact match, abstention và citation metrics;
- benchmark/config/artifact/code provenance;
- per-case results, sanitized errors, latency và resource summary;
- immutable JSON/JSONL reports cùng `legal-rag-evaluate` CLI;
- unit/integration tests cho metric arithmetic và complete runner.

Milestone 16 không tạo official/synthetic gold benchmark, không fine-tune model
và không tuyên bố semantic metrics khi thiếu gold/human labels.

Milestone 17 đã tạo:

- typed full-corpus validation policy;
- AIO full profile pin revision, bỏ sample limit và khai báo expected counts;
- complete artifact-set checksum, count, identity và lineage validation;
- immutable `build_validation.json` sau offline build;
- read-only `legal-rag-validate` CLI;
- unit/integration tests cho policy, valid build và payload tampering.

Milestone 17.1 đã tạo:

- pinned bounded source passes cho full profile;
- immediate stage persistence và explicit memory release;
- NumPy float32 vector preallocation thay Python list-of-lists;
- typed `build_state.json` với config hash và code version;
- opt-in resume sau normalized checkpoint;
- fail-closed stage dependency/config/code validation;
- integration test cho late-stage failure và successful resume.

Milestone 17.2 đã sửa:

- canonical config hashing ổn định giữa các process Python;
- một shared hashing implementation cho build identity và processing manifests;
- `OfflineBuildState` schema `1.1`;
- fail-closed legacy state `1.0` vì không thể migrate digest cũ an toàn;
- subprocess regression test với nhiều `PYTHONHASHSEED`;
- patch version `0.19.1`, không thêm dependency.

Milestone 17.3 đã sửa:

- xác nhận legacy parser OOM-kill ở khoảng 10,8 GiB RSS trên Colab 12 GiB;
- one-document-at-a-time parser/chunker production path;
- incremental atomic legal-block/legal-chunk artifact writers;
- configurable document-processing progress logs;
- disk-backed batched SQLite BM25 build;
- batched embedding trực tiếp vào disk-backed NumPy memmap;
- one-pass/equivalence/failure-safety regression tests;
- version `0.20.0`, không thêm dependency.

Milestone 17.4 đã sửa:

- vector stage dùng deterministic `.vector.partial` workspace;
- typed checkpoint ghi committed vector/chunk offset theo batch;
- resume không embedding lại các chunk đã commit;
- checkpoint identity validation fail closed;
- final vector artifact chỉ publish sau count và checksum;
- build state `0.20.0` được nâng có kiểm soát lên `0.20.1`;
- version `0.20.1`, không thêm dependency.

Milestone 17 đã được xác nhận bằng full AIO build và report thật:

```text
is_full_corpus = true
is_valid = true
```

Milestone 18 đã tạo:

- backend-neutral `ChatModelProvider` phía sau `AnswerGenerator`;
- OpenAI-compatible provider không cần LLM SDK;
- typed endpoint, model/revision, timeout, output-token và API-key-env config;
- evidence-only prompt cùng strict `ModelAnswerDraft`;
- evidence-ID allowlist, required answer marker và system-built citations;
- fail-closed model/transport/schema handling;
- explicit extractive default để UI không phụ thuộc model service;
- unit tests không network cho prompt, parsing, citation và provider failures.

Milestone 18 chưa chọn/fine-tune model cuối cùng, chưa semantic-verify từng claim
và chưa benchmark chất lượng trên labeled data. Full-corpus Qwen2.5-3B smoke đã
trả model-backed answer với citation hợp lệ.

Milestone 19 đã tạo:

- typed query analysis, bounded query variants và per-variant contribution;
- deterministic extraction chỉ từ user text cho document number,
  Điều/Khoản/Điểm, năm, scope cue, relationship cue và conservative intent;
- runtime recompute analysis trước fixed retrieval hoặc Agent;
- multi-query BM25/dense với unweighted RRF, không cộng raw scores;
- adaptive graph-first route cho relationship query;
- retry rewriter ưu tiên user-derived variants;
- fail-closed duplicate-payload validation và trace đầy đủ;
- unit/integration tests không network, model hoặc dataset thật.

Milestone 19 không triển khai synonym/LLM query expansion, semantic context
grading, claim-level citation verification hoặc learned fusion.

Milestone 20 đã tạo:

- typed evidence applicability, selection reason và per-hit selection trace;
- deterministic context ranking dùng source rank, explicit reference match,
  lexical overlap và configured inactive-status penalty;
- whole-chunk token/count budgeting với omission reason;
- selection provenance trong `ContextBuildResult` và selected `Evidence`;
- explicit document/article reference coverage trong context grader;
- fail-closed context chỉ có inactive hoặc reference-mismatched evidence;
- typed bounded `online.evidence_selection` configuration;
- unit/integration tests không network, model hoặc dataset thật.

Milestone 20 không diễn giải semantic legal applicability, không xác minh hiệu
lực pháp luật hiện tại, không giải quyết xung đột văn bản và không claim-level
verify answer.

Milestone 21 đã tạo:

- typed claim support status và per-claim verification result;
- bounded deterministic claim segmentation;
- inline evidence marker coverage cho synthesized answer;
- exact mapping từ claim marker tới response citation và selected evidence;
- lexical support, numeric preservation và negation preservation checks;
- unused citation và unsupported claim rejection;
- fail-closed Agent abstention cùng persisted verification metadata;
- extractive-mode exemption có explicit warning;
- unit/integration tests không network, model hoặc dataset thật.

Milestone 21 chưa semantic-entailment verify, chưa contradiction-detect đầy đủ,
chưa xử lý complex coreference/multi-hop claim và chưa tune threshold trên
labeled competition data.

Milestone 22 đã tạo:

- optional two-stage model-backed verifier sau toàn bộ M21 hard checks;
- typed three-way semantic label: supported, contradicted, insufficient;
- strict model draft chỉ gồm claim ID và label;
- trusted evidence IDs được gắn lại từ deterministic claim result;
- exact claim completeness/order validation;
- provider/model/revision provenance trong verification metadata;
- fail-closed Agent abstention cho non-supported, invalid schema và model error;
- backend mặc định disabled, không buộc local runtime dùng GPU hoặc network;
- reuse `ChatModelProvider`, không thêm dependency và không rebuild artifact;
- generator/verifier cùng exact Transformers runtime identity dùng chung một
  bộ weights, giữ token limits riêng và shared inference lock;
- unit/integration tests không model thật, network hoặc dataset thật.

Milestone 22 chưa benchmark model trên labeled legal claims, chưa giải quyết đầy
đủ temporal validity, exception scope, conflict of laws hoặc multi-hop legal
reasoning và không thay thế legal expert review.

Milestone 23 đã tạo:

- sanitized runtime configuration hash và component/model/package provenance
  trong evaluation summary;
- pinned dataset name/revision cho từng evaluation run mới;
- typed multi-run candidate, objective, direction, threshold và selection mode;
- strict comparability cho benchmark SHA-256, case count, cutoffs, dataset
  lineage và labeled metric case counts;
- namespaced quality, latency, failure và resource metrics;
- optional accelerator identity và peak allocated-memory observation;
- explicit eligibility/exclusion reason cho từng candidate;
- deterministic Pareto frontier, mặc định không chọn một winner;
- optional lexicographic selection chỉ khi policy khai báo rõ;
- immutable comparison report và `legal-rag-compare` CLI;
- unit/integration tests không network, GPU, model hoặc dataset thật.

Milestone 23 chưa kèm official/reviewed labeled benchmark, chưa chốt model cuối
cùng, không fine-tune và không tuyên bố competition quality.

Milestone 24 đã tạo:

- typed benchmark manifest với exact benchmark SHA-256, case count, target
  granularity và pinned dataset lineage;
- label status rõ ràng: `diagnostic`, `human_reviewed` hoặc
  `competition_official`;
- timestamped provenance bắt buộc cho trusted label status;
- validation benchmark bytes/manifest/runtime corpus trước evaluation;
- benchmark manifest identity trong evaluation và comparison reports;
- chặn winner selection khi benchmark chỉ là diagnostic;
- optional maximum regression theo từng objective so với explicit baseline;
- required `--benchmark-manifest` cho evaluation CLI;
- unit/integration tests không network, GPU, model hoặc dataset thật.

Milestone 24 chưa kèm reviewed/official benchmark, chưa tự xác minh chất lượng
nhãn, chưa chọn model cuối cùng và không giả định metric của BTC.

Competition adaptation chỉ bắt đầu sau khi người dùng yêu cầu và có thông tin
BTC:

```text
Future — Competition Adaptation
```

---

Milestone 17.5 đã sửa:

- online vector `chunks.jsonl` dùng disk-backed byte-offset store thay vì
  materialize toàn bộ corpus thành Pydantic objects;
- compact postings cho unified filters;
- bounded vector validation và exact cosine scoring batches;
- chỉ materialize final top-k chunks;
- startup progress logging không chứa legal content;
- giữ nguyên vector artifact format và retrieval semantics;
- build state `0.20.0`/`0.20.1` được nâng có kiểm soát lên `0.20.2`;
- version `0.20.2`, không thêm dependency.

Full-corpus validation đã được người dùng xác nhận `is_full_corpus = true` và
`is_valid = true`. Online/UI smoke trên artifact thật cũng đã hoàn thành sau
memory-safe loader.

Milestone 17.6 đã sửa:

- BM25 full-corpus query dùng FTS5 `rank` và bounded corpus-aware term planning;
- giữ số và legal semantic modifier, không aggressive stopword removal;
- typed `online.bm25_runtime` bounds;
- optional `validated_report` startup chỉ reuse deep validation khi exact
  manifests và required checks khớp;
- embedding dimension compatibility không load model weights;
- startup stage latency logging;
- reuse artifact `0.20.2`, không rebuild dataset/index/vector;
- partial build `0.20.2` không resume bằng `0.20.3`; complete artifacts được
  reuse mà không rebuild;
- version `0.20.3`, không thêm dependency.

Milestone 17.7 đã tạo:

- persisted SQLite `vector_serving_metadata` sidecar từ validated vector chunks;
- one-time `legal-rag-prepare-serving` command;
- atomic/no-overwrite build, source checksum/manifest compatibility;
- read-only row offset, chunk ID và unified filter lookup;
- optional fallback hoặc required fail-closed runtime policy;
- không re-embed, không rebuild vector/BM25 và không thêm dependency;
- version `0.20.4`.

Milestone 17.8 đã sửa:

- thay Gradio queue/SSE diagnostic UI bằng same-origin HTTP UI;
- UI gọi public `/api/v1/answer`, không gọi runtime/backend trực tiếp;
- hoạt động qua Colab port proxy mà không cần public URL config;
- bỏ runtime dependency Gradio, không thêm dependency;
- version `0.20.5`.

Milestone 17.9 đã tạo:

- optional GPU-resident exact dense scorer trong NumPy vector adapter;
- typed `cpu|cuda` search device và bounded device transfer batch;
- giữ float32 cosine, filter, tie-break, artifact và public schema;
- explicit CUDA fail-closed, CPU default không đổi;
- reuse vector/sidecar hiện có, không re-embed hoặc rebuild;
- version `0.20.6`.

Full-corpus CUDA validation đã pass trên 1.278.201 x 384 vectors:

- warm vector search 22,2 ms;
- warm dense retrieval 35,6 ms;
- warm reranker 398 ms;
- warm Agent workflow khoảng 2,04 giây;
- cold query khoảng 31,47 giây do lazy model initialization.

Milestone 25 đã tạo:

- active data policy `competition_only` cho UIT DSC 2026 Task 2;
- fail-closed online artifact lineage theo official corpus identity;
- typed `CompetitionQuestion` và `CompetitionContext` schemas;
- strict local UIT DSC JSON loader với duplicate-key, unknown-field,
  missing-field và duplicate-context checks;
- question-only và question-with-answer support mà không phát minh labels;
- loại AIO source/adapter/audit/normalizer/relationship normalizer, raw fixtures,
  tests, build profile và offline composition root khỏi active tree;
- loại runtime dependency `datasets` và legacy build CLI;
- dataset-neutral core fixtures và generic build-count validation;
- cập nhật competition, schema, architecture và implementation documents;
- validation loader trên `warmup.json` thật: 500 records;
- local suite: 350 passed, 1 skipped; compileall pass.

Milestone 25 chưa ingest `selected-contexts.zip`, chưa build official indexes,
chưa triển khai official-equivalent METEOR/ROUGE-L và không xóa external
artifacts ngoài repository. Submission formatter được hoàn thành sau đó ở M28.

Milestone 26 đã tạo:

- official `context_*.json` loader cho ZIP hoặc thư mục đã giải nén;
- strict one-object-per-file, exact-field, duplicate-key, duplicate-member và
  duplicate-context checks;
- canonical corpus SHA-256 giống nhau cho ZIP và directory có cùng bytes;
- one-to-one context adapter sang unified `LegalDocument`;
- không suy diễn document number, dates, effect status hoặc relevance labels;
- deterministic corpus audit, dataset manifest, normalized manifest và
  plain-text pass-through cleaned manifest;
- local integration từ official-format context qua parser tới legal chunks;
- local suite: 356 passed, 1 skipped.

Milestone 26 chưa chạy trên `selected-contexts.zip` thật, chưa persist full
official corpus, chưa build official BM25/vector/graph, chưa đo memory/latency
toàn corpus và chưa tạo build CLI mới.

Milestone 27 đã tạo:

- official offline build CLI với stage state atomic và strict source/config/code
  recovery identity;
- persistence cho corpus audit, normalized/cleaned pass-through documents và
  complete artifact lineage;
- explicit zero-record relationship artifact và zero-edge graph vì raw schema
  BTC không có relationship fields;
- streaming parser/chunker, disk-backed BM25 và resumable vector checkpoint
  trong composition root mới;
- submission-neutral batch inference JSONL với per-question checkpoint,
  ordered completeness validation và final checksum manifest;
- không dùng warm-up gold answer làm prediction;
- local suite: 361 passed, 1 skipped.

Milestone 27 chưa chạy full corpus thật, chưa xác nhận official counts/resource
usage và chưa implement official-equivalent METEOR/ROUGE-L.

Milestone 28 đã tạo:

- exact `CompetitionSubmissionItem` chỉ gồm string `id` và `answer`;
- fail-closed validation cho question source, batch checksum, count, ID và order;
- deterministic UTF-8 `submission.json` trong ZIP chỉ có một member;
- output bắt buộc tên `submission.zip` và không overwrite;
- `legal-rag-submit` CLI;
- unit tests cho exact contract, Unicode, tampering và reproducibility;
- local suite: 366 passed, 1 skipped; compileall và CLI help pass.

Milestone 28 chưa upload Codabench thật, chưa chạy official full corpus và chưa
tuyên bố evaluator local tương đương scorer METEOR/ROUGE-L của BTC.

Milestone 29 đã tạo:

- nullable local diagnostic `meteor` và `rouge_l` cho labeled answer cases;
- NFC/casefold Vietnamese letter/number tokens, không bỏ dấu;
- exact-token METEOR với precision/recall và fragmentation penalty;
- token-level ROUGE-L F1 bằng longest common subsequence;
- report warning rằng metric local chưa official-equivalent;
- submission rendering loại verified `[E<number>]` markers nhưng giữ citation
  đầy đủ trong internal batch;
- unknown evidence marker fail closed;
- version `0.31.0`, không thêm dependency;
- local suite: 369 passed, 1 skipped; compileall pass.

Tại thời điểm Milestone 29, official tokenizer, stemming/synonym policy và
aggregation chưa biết. Milestone 36 đã phân tích source BTC; implementation M29
vẫn là diagnostic và chưa được thay bằng official-compatible mode.

Milestone 30 đã tạo:

- strict loader cho `submission.zip` chỉ có `submission.json`;
- duplicate JSON field, duplicate ID, missing/extra/reordered ID rejection;
- direct warm-up reference scoring không cần retrieval labels;
- per-ID và aggregate exact match, diagnostic METEOR, diagnostic ROUGE-L;
- checksum cho exact reference, archive và submission JSON bytes;
- immutable content-free `warmup_score.json`;
- `legal-rag-score-warmup` CLI;
- version `0.32.0`, không thêm dependency;
- local suite: 372 passed, 1 skipped; compileall và CLI help pass.

Milestone 30 chưa chạy trên submission/warm-up thật và vẫn không tuyên bố local
scorer tương đương tuyệt đối với Codabench.

Milestone 31 đã tạo:

- compliance source of truth từ thể lệ BTC được cung cấp ngày 2026-08-01;
- official-only data gate, model/license approval register (E5 MIT, reranker
  Apache-2.0, Qwen 3B custom `qwen-research`) và unresolved-rule register;
- MIT source license;
- Data Statement, Model Card, private submission checklist và quota ledger
  templates;
- non-root CPU Docker reproducibility scaffold không chứa data/model/artifact,
  secret hoặc submission;
- direct dependency constraints cho Docker scaffold;
- lightweight `legal-rag-submit` và `legal-rag-score-warmup` entry points không
  import FastAPI/Uvicorn;
- version `0.33.0`, không thêm dependency.

Milestone 31 chưa phê duyệt model nào, chưa hoàn tất transitive license review,
chưa tạo final GPU image, chưa có full transitive dependency lock và chưa build
artifact official vì còn chờ dữ liệu/hạ tầng/xác nhận của BTC.

Milestone 32 đã sửa contract theo bằng chứng từ scorer Codabench thực tế:

- root `submission.json` là object keyed by official question ID;
- mỗi value chỉ có string field `answer`;
- formatter và warm-up scorer từ chối array contract cũ;
- regression test thực thi đúng phép chiếu `.items()` của scorer;
- version `0.34.0`, không thêm dependency.

Milestone 32 đã được xác nhận về output shape: Codabench đọc object keyed by ID.
BTC sau đó đã bổ sung NLTK WordNet và một submission đã được chấm thành công với
non-empty official warm-up score.

Milestone 33 đã tạo/cập nhật:

- `docs/12-TEAM-ONBOARDING.md` giải thích end-to-end pipeline, package map,
  artifact, CLI, evaluation, submission và workflow cho thành viên;
- phản hồi BTC về model registration, official-only data, fine-tuning và cấm
  synthetic data;
- bằng chứng log warm-up ban đầu về NLTK/rouge-score và lỗi WordNet phía BTC;
  kết luận PyVi cũ đã bị source chính thức ở Milestone 36 supersede;
- README onboarding entry point;
- version `0.35.0`, không thêm dependency hoặc business logic.

Milestone 34 đã ghi nhận thông báo BTC mới nhất:

- tổng tham số của toàn bộ model trong Task 2 phải nhỏ hơn 4 tỷ;
- cấm mọi API và yêu cầu đội chạy/kiểm soát model trực tiếp;
- pretrained/distilled model phù hợp được phép, nhưng augmentation bị cấm;
- dữ liệu pretraining không được xem là dữ liệu ngoài trực tiếp;
- Docker, GitHub hoặc ZIP đều có thể dùng để tái lập;
- parameter inventory và aggregate budget là gate bắt buộc trước official run.

Milestone 35 đã ghi nhận toàn bộ data overview của BTC tại
`docs/13-UIT-DSC-2026-DATA-CONTRACT.md`, gồm task contract, resource names, QA và
context raw schema, metric, graph implication và checklist audit. Milestone 37
đã thay các giả định từ overview bằng kết quả audit archive thật.

Milestone 36 đã phân tích read-only source scorer BTC:

- ZIP SHA-256
  `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`;
- METEOR dùng whitespace split + NLTK defaults + WordNet/OMW;
- ROUGE-L dùng vendored lowercase ASCII `[a-z0-9]` tokenizer, không stemming;
- cả hai metric dùng arithmetic macro mean;
- PyVi bị comment và không tham gia scoring;
- runtime I/O, member checksums và local parity gap được ghi tại
  `docs/15-OFFICIAL-SCORING-CONTRACT.md`;
- không sửa evaluator, không thêm dependency và không commit source scorer.

Milestone 36 chưa implement official-compatible scoring mode và chưa pin exact
NLTK/NumPy/WordNet identities vì scorer ZIP không chứa dependency lock.

Milestone 37 đã hoàn thành:

- audit read-only `train.json`, `public-official.json` và
  `selected-contexts.zip` thật;
- canonical corpus revision
  `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`;
- adapter cho numeric/string ID, optional `name`, blank `passage` và strict
  audited field sets;
- raw-preserving normalized artifact và dataset-specific cleaned artifact;
- cleaner versioned cho NFC/newline/line whitespace, known HTML presentation
  markup và exact TVPL Pro notice;
- graph vẫn zero-edge; không tạo retrieval labels hoặc synthetic data;
- version `0.36.0`, không thêm dependency.

Milestone 37 chưa build full parser/chunker/BM25/vector artifacts, chưa chạy
model experiment và chưa implement official-compatible scorer.

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
