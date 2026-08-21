# 08. Design Decisions

## 1. Purpose

File này là nguồn chính thức lưu các quyết định kiến trúc.

Codex hoặc developer không được tự ý thay đổi quyết định đã có trạng thái
`Accepted`.

Mọi thay đổi phải:

1. thêm decision mới;
2. ghi lý do;
3. ghi tác động;
4. đánh dấu decision cũ là `Superseded` nếu cần;
5. cập nhật tài liệu liên quan.

---

## 2. Status Values

- `Proposed`: đang xem xét.
- `Accepted`: đã chốt và phải tuân thủ.
- `Superseded`: đã bị thay thế.
- `Rejected`: không sử dụng.

---

## D001 — Primary Dataset

**Status:** Superseded by D068

Trong giai đoạn hiện tại, chỉ sử dụng dataset:

`th1nhng0/vietnamese-legal-documents`

Dataset được dùng làm corpus, metadata source và relationship source.

Không tích hợp corpus hoặc QA dataset khác vào baseline hiện tại.

---

## D002 — Dataset Independence

**Status:** Accepted

Core system không phụ thuộc trực tiếp vào schema thô của dataset AIO.

Raw fields phải được chuyển qua adapter và unified schema.

**Reason:**

Dữ liệu Ban tổ chức có thể có schema khác.

---

## D003 — Offline and Online Separation

**Status:** Accepted

Ingestion, cleaning, chunking và indexing chạy offline.

Online pipeline chỉ đọc artifact và xử lý query.

Agent không được build lại index trong thời gian inference.

---

## D004 — Retrieval Unit

**Status:** Accepted

Retrieval unit ưu tiên là Điều luật.

Nếu Điều quá dài, chia theo Khoản.

Token-based splitting chỉ được dùng như fallback.

---

## D005 — Legal Structure Preservation

**Status:** Accepted

Cleaner và chunker phải giữ:

- tên văn bản;
- số ký hiệu;
- Chương;
- Mục;
- Điều;
- Khoản;
- Điểm;
- số tiền;
- ngày tháng;
- thời hạn;
- từ phủ định;
- metadata hiệu lực.

---

## D006 — Retrieval-First Development

**Status:** Accepted

Fixed retrieval baseline phải được xây trước Agentic workflow.

Agent không được triển khai để thay thế hoặc che giấu retrieval chưa hoàn
chỉnh.

---

## D007 — Baseline Retrieval Strategies

**Status:** Accepted

Baseline gồm:

- BM25;
- dense retrieval;
- hybrid RRF;
- cross-encoder reranking;
- graph-expanded retrieval.

---

## D008 — Hybrid Fusion

**Status:** Accepted

Hybrid baseline dùng Reciprocal Rank Fusion.

Không cộng trực tiếp raw BM25 score và dense similarity score.

---

## D009 — Reranking Scope

**Status:** Accepted

Cross-encoder reranker chỉ chạy trên candidate set đã được retrieval.

Không chạy reranker trên toàn corpus.

---

## D010 — Graph Retrieval

**Status:** Accepted

Graph retrieval không thay thế text retrieval.

Graph dùng seed documents từ BM25, dense hoặc hybrid retrieval.

Baseline traversal:

- mặc định 1 hop;
- tối đa 2 hop.

---

## D011 — Graph Granularity

**Status:** Accepted

Baseline graph sử dụng document-level nodes.

Chunk-level graph chưa thuộc baseline.

---

## D012 — Model Training

**Status:** Accepted

Baseline đầu tiên dùng pretrained models.

Chưa fine-tune:

- dense retriever;
- reranker;
- answer generator;
- context grader.

Fine-tuning chỉ được bổ sung khi có supervision phù hợp.

---

## D013 — Agent Responsibility

**Status:** Accepted

Agent chỉ:

- chọn tool;
- gọi tool;
- quan sát kết quả;
- grade context;
- retry có giới hạn;
- gọi generator;
- gọi verifier.

Agent không:

- ingest dataset;
- clean corpus;
- build index;
- sửa corpus;
- thay config;
- truy cập raw database client trực tiếp.

---

## D014 — Agent Retry Limit

**Status:** Accepted

Baseline Agent có:

```text
max_retry = 2
```

Giá trị có thể được đưa vào configuration nhưng không được bỏ giới hạn.

---

## D015 — Grounded Generation

**Status:** Accepted

Answer generator chỉ được dùng selected evidence.

Nếu evidence không đủ, hệ thống phải abstain.

---

## D016 — Citation Verification

**Status:** Accepted

Mọi answer có citation phải được kiểm tra citation tồn tại và trỏ về
evidence hợp lệ.

Baseline dùng rule-based verification.

---

## D017 — Legal Validity Metadata

**Status:** Accepted

`effect_status`, `effective_date` và `expiry_date` được xem là metadata
snapshot.

Hệ thống không mặc định đây là trạng thái pháp lý mới nhất.

---

## D018 — Raw Data Preservation

**Status:** Accepted

Raw dataset không được chỉnh sửa trực tiếp.

Mọi processing stage phải có artifact hoặc khả năng tái tạo.

---

## D019 — Artifact Versioning

**Status:** Accepted

BM25, vector, graph và processed data phải có manifest.

Online pipeline phải kiểm tra compatibility của artifact khi startup.

---

## D020 — Competition Adaptation

**Status:** Accepted

Dữ liệu Ban tổ chức sau này được tích hợp bằng adapter.

Nếu corpus thay đổi, hệ thống rebuild indexes nhưng giữ core interfaces.

---

## D021 — Backend Selection

**Status:** Proposed

Các lựa chọn sau chưa được chốt ở mức production:

- vector database;
- graph database;
- BM25 backend;
- embedding model;
- reranker model;
- LLM generator;
- agent framework.

Prototype có thể tham khảo lựa chọn trong tài liệu AIO nhưng core không
được hard-code implementation.

---

## D022 — Agent Framework

**Status:** Proposed

LangGraph là ứng viên ưu tiên do khả năng kiểm soát state, edge và retry.

Chưa triển khai trước milestone Agentic Workflow.

---

## D023 — Evaluation Dataset

**Status:** Proposed

Dataset AIO không có gold QA labels đầy đủ.

Evaluation có nhãn sẽ được bổ sung khi:

- BTC cung cấp dữ liệu;
- hoặc nhóm xây benchmark hợp lệ.

Synthetic QA không được dùng làm ground truth chính thức nếu chưa được
kiểm chứng.

---

## D024 — External Web Search

**Status:** Rejected for Initial Baseline

Không sử dụng web fallback trong baseline hiện tại.

---

## D025 — OCR and PDF Processing

**Status:** Rejected for Initial Baseline

Không triển khai OCR hoặc PDF crawler trong baseline đầu tiên.

---

## D026 — Milestone 1 Python Foundation

**Status:** Accepted

Milestone 1 sử dụng:

- package import name `legal_agentic_rag`;
- Python 3.11 trở lên;
- `src/` layout;
- Pydantic v2 cho unified schema và typed configuration;
- `typing.Protocol` cho backend contracts;
- setuptools làm Python build backend;
- Python standard-library logging.

Milestone 1 không dùng configuration framework, environment loader,
concrete backend, model, dataset loader hoặc business logic.

---

## D027 — Milestone 1 Schema Contract Clarifications

**Status:** Accepted

Milestone 1 áp dụng các contract sau:

- unknown top-level schema fields bị từ chối;
- extension data chỉ nằm trong `metadata` hoặc `raw_metadata`;
- `source_dataset` required và không mặc định thành `aio`;
- unmapped `LegalRelationship.relationship_type` dùng null;
- `ArtifactManifest` ghi dataset name, processing config hash và warnings;
- artifact type bao phủ mọi processed/index artifact trong `AGENTS.md`;
- `AnswerResponse.insufficient_evidence` là required field;
- retrieval trace lưu contribution riêng của BM25 và dense trong RRF;
- typed nested contracts chỉ được thêm khi có consumer rõ ràng.

Các nested contract được chấp nhận:

- `RetrievalFilters`;
- `GraphPathStep`;
- `RetrievalTrace`;
- `CitationVerificationResult`;
- `ArtifactValidationResult`;
- `RetrievalHistoryItem`.

---

## D028 — Deferred Milestone 1 Packages and Tracing Contract

**Status:** Accepted

Không tạo package rỗng cho offline, indexing, retrieval, generation,
tools, Agent, serving hoặc evaluation trong Milestone 1.

Chưa tạo tracing backend contract khi chưa có consumer. Milestone 1 chỉ
tạo logging context cơ bản; tracing backend vẫn phải đi qua abstraction
khi được triển khai ở milestone phù hợp.

---

## D029 — Milestone 2 Dataset Loader and Audit Boundary

**Status:** Superseded by D068

Milestone 2 dùng `datasets>=5,<6` là concrete dependency duy nhất để đọc
dataset Hugging Face đã phê duyệt. Implementation:

- chỉ `AioDatasetSource` bị giới hạn với
  `th1nhng0/vietnamese-legal-documents`;
- load riêng `metadata`, `content`, `relationships`, split `data`;
- hỗ trợ revision, streaming và sample limit;
- raw field names bị cô lập trong `offline/datasets/aio` hoặc audit;
- audit ghi schema profile, ID, join coverage, content findings, relationship
  findings và metadata findings;
- persisted output gồm `DatasetAuditReport` và bốn CSV issue extracts;
- không silently overwrite report cũ;
- không normalize, clean, parse, chunk hoặc index.

Do schema Arrow của config `content` và kiểu Parquet thực tế không khớp,
loader áp dụng `large_string` feature override ngay tại AIO boundary. Quyết
định này không thay đổi raw content.

---

## D030 — Milestone 3 Conservative Document Normalization

**Status:** Accepted

Document normalization áp dụng policy bảo thủ:

- metadata tạo document; orphan content không tạo metadata giả;
- duplicate metadata ID bị reject toàn bộ thay vì chọn tùy ý;
- chỉ join content khi có đúng một record hợp lệ;
- ambiguous content không merge và không tự chọn;
- invalid date thành null kèm issue, không đoán hoặc sửa ngày;
- effect status/document type chỉ canonicalize bằng mapping configuration;
- unknown raw metadata được giữ trong `raw_metadata`;
- `nguon_thu_thap` không được coi là URL;
- normalized HTML được giữ nguyên và cleaning vẫn thuộc Milestone 4;
- chưa chọn artifact storage format; Milestone 3 tạo typed result và
  manifest nhưng chưa có persistence writer.

Policy này ưu tiên không gắn nhầm nội dung hoặc metadata pháp lý khi raw
dataset có record trùng hoặc không nhất quán.

---

## D031 — Milestone 4 Conservative HTML Cleaning

**Status:** Accepted

HTML cleaning dùng `html.parser` của Python standard library và chỉ nhận
unified `LegalDocument` cùng normalized artifact manifest. Policy:

- loại `script`, `style`, `nav`, các non-content tag được cấu hình, phần tử
  hidden và class/id noise theo exact token;
- không xóa `header` hoặc `footer` chỉ dựa vào tag vì chúng có thể chứa tiêu
  đề, số văn bản, chữ ký hoặc phụ lục pháp lý;
- giữ visible table text bằng ranh giới dòng/cột đơn giản, chưa parse cấu trúc
  bảng;
- decode HTML entities, normalize Unicode NFC, line break và whitespace;
- giữ dấu tiếng Việt, số, dấu câu, từ phủ định và legal markers;
- HTML lỗi nhẹ được parser dung nạp; output rỗng và content thiếu tạo
  structured `AuditIssue` thay vì silently skip document;
- output giữ toàn bộ input documents, nối provenance qua cleaned artifact
  manifest và processing config hash;
- chưa chọn artifact storage format, vì vậy result vẫn in-memory và chưa có
  persistence writer.

Repeated header/footer không bị suy đoán từ text lặp trên toàn corpus trong
baseline này vì có nguy cơ xóa nhầm nội dung pháp lý. Chỉ marker web-noise rõ
ràng mới được loại.

---

## D032 — Milestone 5 Conservative Legal Structure Parsing

**Status:** Accepted

Legal structure parser dùng Python standard-library regex trên unified
`clean_text`; không biết raw field AIO. Policy:

- tạo non-overlapping blocks cho document preamble, Phần, Chương, Mục, Tiểu
  mục, Điều, Khoản, Điểm, Phụ lục và table rows;
- Khoản/Điểm chỉ được nhận diện trong context của Điều;
- hỗ trợ marker số thường/La Mã và các delimiter phổ biến được xác nhận từ
  live sample, đặc biệt dạng `Điều 1:`;
- title dòng kế tiếp chỉ được nhận khi không giống marker/table, không kết
  thúc bằng sentence punctuation và nằm trong character/word limits có config;
- marker không chắc chắn không bị sửa hoặc bỏ; parser giữ text trong block cha
  và có thể phát `unrecognized_structure_marker`;
- mỗi non-whitespace source character phải xuất hiện đúng một lần trong block
  output; diagnostic ghi coverage theo document;
- block ID dùng content hash deterministic; parent luôn cùng document và đứng
  trước child;
- documents thiếu `clean_text` vẫn có diagnostic/issue và không bị silently
  drop khỏi result;
- chưa chọn artifact storage format nên result và `legal_blocks` manifest vẫn
  in-memory, chưa có persistence writer.

Parser này là structural baseline, không phải semantic legal parser. Văn bản
không có marker rõ ràng được giữ trong một `document` block thay vì đoán cấu
trúc.

---

## D033 — Milestone 6 Article-First Legal Chunking

**Status:** Accepted

Legal chunker chỉ dùng unified documents/blocks và áp dụng policy:

- một Điều cùng descendants là retrieval unit ưu tiên;
- Điều vượt giới hạn được group theo direct legal child units, thường là các
  Khoản liên tiếp;
- một unit vẫn vượt giới hạn mới dùng overlapping token windows;
- non-article blocks tạo standalone chunks để bảo toàn toàn bộ parsed text;
- baseline tokenizer là dependency-free `unicode_word_v1`, với defaults
  `max_tokens=512`, `min_tokens=50`, `overlap_tokens=50`;
- `min_tokens` chỉ tạo informational issue; không bỏ short legal chunk;
- chunk ID hash deterministic từ document, strategy, source blocks, split và
  text; chunk index liên tục theo document;
- search text chỉ dùng document metadata, hierarchy và chunk text có thật;
- validator phải xác nhận token limit, source coverage, identity, ordering và
  metadata inheritance trước khi trả result;
- output nối provenance bằng processing config hash và manifest
  `legal_chunks`; chưa persist vì artifact storage format vẫn chưa chốt.

Tokenizer baseline chỉ phục vụ deterministic chunk limits. Nó không đại diện
cho tokenizer của embedding/generator model chưa được lựa chọn. Thay tokenizer
phải tạo config hash/version mới và rebuild legal chunks cùng downstream
artifacts.

---

## D034 — Milestone 7 SQLite FTS5 BM25 Reference Backend

**Status:** Accepted

Milestone 7 dùng SQLite FTS5 từ Python standard library làm reference backend,
không coi đây là lựa chọn production cuối cùng. Policy:

- concrete implementation nằm sau `BM25Backend`;
- `build` nhận validated `LegalChunk` và source manifest `legal_chunks`;
- analyzer `unicode_word_casefold_v1` dùng Unicode NFC và casefold cho lexical
  terms, giữ dấu tiếng Việt, số và từ phủ định, không stemming/stopword removal;
- default match mode là `any`; `all` có thể cấu hình;
- search hỗ trợ unified exact filters, deterministic tie-breaking và trả
  `RetrievalResponse` cùng BM25 trace/latency;
- artifact format riêng của backend gồm `index.sqlite3` và `manifest.json`;
- manifest/checksum/integrity/count phải tương thích trước khi load;
- loaded connection cho phép cross-thread read nhưng mọi operation trên một
  connection được serialize bằng backend lock để FastAPI/Gradio dùng chung an
  toàn;
- destination tồn tại không bị silently overwrite;
- thay analyzer/backend/source chunks tạo config hash mới và yêu cầu rebuild.

Quyết định production backend trong D021 vẫn để mở; core không phụ thuộc trực
tiếp vào SQLite FTS5.

---

## D035 — Milestone 8 Multilingual E5 and NumPy Vector Reference Baseline

**Status:** Accepted

Người dùng ủy quyền lựa chọn tối ưu cho baseline hiện tại. Milestone 8 chọn:

- `intfloat/multilingual-e5-small`, revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, license MIT;
- `sentence-transformers>=5,<6` làm concrete provider sau
  `EmbeddingProvider`, không hard-code model vào retrieval core;
- E5 `passage:`/`query:` prefixes, dimension 384, max length 512 và normalized
  embeddings; CPU mặc định phù hợp runtime không GPU;
- `numpy_flat` float32 exact cosine làm reference `VectorBackend`;
- offline builder batch 16; artifact ghi concrete provider name/version và
  online dense retriever kiểm tra provider/model/revision/dimension trước query
  embedding;
- artifact `vectors.npy` + `chunks.jsonl` + `manifest.json`, checksums và
  memory-mapped reload;
- exact unified metadata filters, deterministic chunk-ID tie-breaking và
  total latency gồm query embedding;
- không fine-tune, không FAISS/vector database và không approximate search khi
  chưa đo bottleneck trên corpus đầy đủ.

Model hoặc revision thay đổi bắt buộc re-embed và rebuild. Production vector
backend/model trong D021 vẫn để mở; core chỉ phụ thuộc Protocol và unified
schemas.

---

## D036 — Milestone 9 Fixed Retrieval and Reciprocal Rank Fusion

**Status:** Accepted

Milestone 9 dùng unweighted Reciprocal Rank Fusion chuẩn với constant mặc định
60 qua typed `RetrievalConfig`. Policy:

- BM25 và dense mỗi nhánh lấy `candidate_k`, final fusion trả `top_k`;
- chỉ rank tham gia `1 / (60 + rank)`; raw backend scores không được cộng;
- deduplicate theo `chunk_id`, giữ rank/raw score/contribution riêng từng nhánh;
- cùng chunk ID nhưng khác document, text hoặc metadata là lỗi;
- RRF score giảm dần và `chunk_id` là deterministic tie-break;
- backend công khai legal-chunks source artifact identity; hybrid từ chối hai
  index khác source type/version/processing hash trước khi query;
- hai nhánh chạy tuần tự và fail-closed. Cách này tránh thay đổi SQLite threading
  policy, giữ behavior đơn giản và không silently degrade;
- `FixedRetriever` hỗ trợ độc lập BM25, dense và hybrid, mặc định hybrid;
- không thêm dependency và chưa triển khai reranker hoặc graph retrieval.

Nếu corpus source khác, cả BM25 và vector index phải được rebuild từ cùng một
legal-chunks artifact trước khi hybrid retrieval được phép chạy.

---

## D037 — Milestone 10 Multilingual Cross-Encoder Reranker

**Status:** Accepted

Người dùng tiếp tục ủy quyền lựa chọn baseline tối ưu. Milestone 10 chọn
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, revision
`1427fd652930e4ba29e8149678df786c240d8825`, license Apache-2.0, qua
`sentence-transformers` và contract `Reranker`.

Lựa chọn này ưu tiên multilingual retrieval, có tiếng Việt trong training scope,
kích thước khoảng 0.1B parameters và khả năng chạy CPU. Model
`BAAI/bge-reranker-v2-m3` đa ngôn ngữ mạnh hơn nhưng lớn hơn đáng kể, nên chưa
phù hợp reference baseline nhẹ khi chưa có benchmark chính thức.

Policy:

- model/revision/device/batch/max length/candidate limit đều qua typed config;
- model được lazy-load, không fine-tune và không hard-code trong retrieval core;
- reranker chỉ nhận tối đa 100 hybrid candidates và trả final `top_k`;
- dùng raw single-label logits để giữ thông tin xếp hạng, không diễn giải như xác
  suất và không cộng với score từ retrieval stage khác;
- deterministic tie-break dùng prior retrieval rank rồi `chunk_id`;
- output giữ nguyên candidate legal payload và toàn bộ RRF provenance;
- backend error fail-closed với taxonomy initialization/model/retrieval;
- không tạo reranker artifact vì đây là online inference với revision-pinned model;
- dependency `sentence-transformers>=5,<6` đã có từ Milestone 8, không thêm package.

Đây là reference baseline, không phải quyết định model production cuối cùng.
Việc thay model cần benchmark trên evaluation data phù hợp sau khi thể lệ và dữ
liệu chính thức được công bố.

---

## D038 — Model Selection Is Quality-Benchmark Driven

**Status:** Accepted

Cấu hình phần cứng local hiện tại không phải giới hạn lựa chọn model dài hạn.
Các model Milestone 8 và 10 vẫn là reference implementations có thể tái tạo,
không phải quality ceiling.

Khi có evaluation set và GPU:

- shortlist phải ưu tiên benchmark chất lượng retrieval/reranking tiếng Việt và
  multilingual phù hợp domain pháp luật;
- đo Recall/MRR/NDCG, latency, VRAM và throughput trước khi thay model;
- model mạnh hơn được phép chạy trên GPU thuê ngoài qua typed config;
- thay embedding model/revision bắt buộc re-embed và rebuild vector index;
- fine-tuning vẫn chỉ bắt đầu khi có supervision, split và experiment plan hợp lệ;
- core tiếp tục chỉ phụ thuộc backend Protocol, không hard-code model.

---

## D039 — Milestone 11 Directed Adjacency Graph Reference Baseline

**Status:** Accepted

Milestone 11 dùng `adjacency_json` từ Python standard library làm reference
`GraphBackend`; đây không phải lựa chọn graph database production cuối cùng.

Policy:

- raw AIO relationship fields chỉ tồn tại trong AIO adapter/normalizer;
- canonical relationship type chỉ được tạo từ explicit configuration mapping;
- orphan endpoint, self-loop, missing endpoint/label và exact duplicate bị loại
  khỏi production graph, đồng thời giữ structured `AuditIssue`;
- normalized relationship mapping và graph index đều có versioned manifest,
  SHA-256 và no-overwrite policy;
- graph là directed document-level adjacency; legal content vẫn nằm trong chunk
  artifacts, không bị copy vào graph;
- traversal dùng deterministic BFS, 1 hop mặc định, tối đa 2 hop, hỗ trợ exact
  relationship filter và trả BFS discovery path;
- online graph retrieval luôn bắt đầu từ hybrid text seeds, chỉ retrieve chunk
  trong reached documents, merge trong `candidate_k`, rồi cross-encoder rerank
  đúng một lần;
- graph không thay BM25/dense, không chạy toàn graph và không cần Agent;
- graph/chunk artifacts phải cùng dataset/revision trước khi online query.

Không thêm dependency mới. Neo4j, NetworkX hoặc backend khác chỉ được xem xét khi
có bottleneck/consumer và benchmark thực tế.

---

## D040 — Milestone 12 Fail-Closed Extractive Fixed RAG

**Status:** Accepted

Khi chưa có gold answer, semantic evaluator và generator benchmark chính thức,
Milestone 12 dùng `ExtractiveAnswerGenerator` dependency-free làm reference
`AnswerGenerator`.

Policy:

- context builder chọn whole legal chunks theo bounded count/token budget,
  không cắt legal text;
- effect status chỉ ảnh hưởng ordering khi inactive labels được cấu hình rõ,
  không suy đoán trạng thái pháp lý mới nhất;
- fixed context grader chỉ xác nhận structural sufficiency và khai báo rõ chưa
  semantic-grade;
- extractive generator chỉ ghép nguyên văn evidence với `[E#]`, không thêm Điều,
  tên luật hay kết luận pháp lý mới;
- empty/insufficient context tạo explicit abstention không citation;
- rule-based verifier kiểm tra exact evidence, chunk, document, article,
  document number và source URL;
- citation invalid làm generated answer bị loại và chuyển thành abstention;
- verification baseline không tuyên bố semantic claim support;
- fixed service giữ retrieval provenance, context grade và verification result
  trong response metadata;
- query ID là deterministic trace ID cho tới khi tracing backend có consumer;
- không thêm LLM SDK, model hoặc dependency.

Backend extractive không phải quality target cuối cùng. Generator model mạnh hơn
sẽ được chọn qua benchmark trên GPU/evaluation data và vẫn phải đi qua
`AnswerGenerator`.

---

## D041 — Milestone 13 Closed Typed Tool Registry

**Status:** Accepted

Milestone 13 dùng closed registry dependency-free thay vì plugin framework.

Policy:

- chỉ có tám `ToolName` được phê duyệt: năm retrieval strategies, context
  grading, answer generation và citation verification;
- retrieval tools tái sử dụng `RetrievalQuery`/`RetrievalResponse`;
- generation-side tools có typed Pydantic inputs chứa đúng query/evidence/
  response cần thiết;
- wrapper chỉ gọi injected fixed service/Protocol, không được truy cập raw
  dataset, database client hoặc artifact mutation;
- registry không auto-discover, dynamic import, decorator-register hoặc load
  third-party tool;
- descriptor công bố name, description, input/output JSON schema và timeout;
- known validation/domain failures được map sang sanitized error types;
- exception message nội bộ, path, secret, payload và legal content không đi vào
  error/log;
- unexpected programming errors không bị nuốt;
- synchronous elapsed-time budget không trả output quá hạn; hard cancellation
  vẫn là trách nhiệm cooperative timeout của concrete external provider;
- tool workflow được integration-test độc lập, chưa có strategy selection,
  retry loop hoặc Agent state mutation.

Không thêm dependency hoặc Agent framework.

---

## D042 — Milestone 14 Dependency-Free Bounded Agent Reference

**Status:** Accepted

Milestone 14 dùng deterministic state machine thuần Python phía sau
`AgentWorkflow` Protocol; đây là reference implementation, không khóa lựa chọn
Agent framework production.

Policy:

- Agent chỉ gọi tool qua closed `ToolRegistry.execute`;
- route mặc định theo chất lượng là `hybrid_rerank → graph → hybrid`, có thể
  cấu hình nhưng tối đa ba strategy duy nhất;
- explicit requested strategy được ưu tiên, nhưng tool chưa đăng ký luôn bị
  loại khỏi route plan;
- `max_retry = 2`, tương ứng tối đa ba retrieval attempts;
- query rewrite bảo thủ chỉ đổi giữa hai query form do user/input adapter cung
  cấp, không sinh thêm legal term;
- mỗi attempt được ghi bằng `RetrievalHistoryItem`; mọi invocation chỉ ghi ID,
  tool name, success, sanitized error type và latency, không ghi payload;
- timeout và lỗi không retry được dừng ngay;
- context đủ mới được generate; citation verification fail hoặc generation
  fail đều trả explicit abstention;
- terminal result giữ đồng thời public `AnswerResponse`, serialized
  `AgentState`, stop reason và total latency;
- workflow không tải dataset, preprocess, build/sửa index hoặc truy cập raw
  backend client.

Không thêm LangGraph, LangChain hoặc dependency mới. Khi có benchmark/consumer
cho framework khác, implementation có thể được thay phía sau `AgentWorkflow`.

---

## D043 — Runtime Artifact Layout and Composition Roots

**Status:** Accepted

Runtime Assembly dùng hai composition roots thuần Python:

- `OfflineBuildRuntime` cho offline build;
- `OnlineRuntimeFactory` cho immutable online load.

Policy:

- mọi artifact directory là một safe relative segment cấu hình trong
  `ArtifactConfig`;
- processed records còn thiếu persistence dùng deterministic UTF-8 JSONL,
  manifest và SHA-256;
- runtime không overwrite build cũ, kể cả khi generic config bật overwrite;
- offline runtime materialize raw components một lần để audit/normalization dùng
  cùng snapshot; chưa tuyên bố tối ưu full-corpus memory;
- legal-chunk manifest lưu normalized-document lineage để đối chiếu graph;
- online startup kiểm tra chunk checksum, dataset/revision, BM25/vector source
  identity, graph lineage và embedding identity;
- online factory chỉ gọi backend `load`, không gọi `build` hoặc `persist`;
- concrete AIO/reference backend chỉ xuất hiện trong composition root, không đi
  vào Agent/API core;
- runtime nhận injected source/provider/backend cho deterministic test;
- không thêm CLI, API, UI, framework config hoặc dependency mới trong bước này.

---

## D044 — Milestone 15 Single-Process FastAPI and Gradio Serving

**Status:** Accepted

Milestone 15 dùng một FastAPI process và optional Gradio UI mount trong cùng
application:

- FastAPI lifespan load một immutable `OnlineRuntime` và fail-fast nếu artifacts
  không tương thích;
- versioned API prefix mặc định `/api/v1`;
- health, retrieval và answer dùng unified Pydantic schemas;
- `ServingService` tạo query ID, bảo toàn Unicode tiếng Việt và enforce giới hạn;
- API không truy cập concrete backend, raw dataset hoặc offline pipeline;
- Gradio chỉ là diagnostic UI và dùng chung runtime, không build runtime thứ hai;
- JSON config loader là explicit baseline, chưa thêm Hydra,
  `pydantic-settings` hoặc environment composition;
- default bind là `127.0.0.1`; chưa tuyên bố production security;
- domain errors được phân loại thành response an toàn, không trả exception detail;
- CLI gồm build artifacts và serve, không tự build artifacts khi server startup.

Dependencies mới:

- runtime: FastAPI, Uvicorn và Gradio;
- development: HTTPX2 cho ASGI integration tests.

Không thêm authentication, CORS, rate limiting, Docker, reverse proxy, cache,
message queue hoặc cloud deployment trong milestone này.

---

## D045 — Milestone 16 Label-Aware Evaluation Baseline

**Status:** Accepted

Evaluation dùng JSONL cases độc lập competition format:

- mỗi case có stable ID, question, target granularity `chunk|document` và
  positive graded relevance map;
- document-level ranking deduplicate nhiều chunks cùng document trước khi tính
  metric;
- retrieval metrics là Recall@k, Precision@k, MRR và graded NDCG@k;
- generation chỉ tính exact match, abstention accuracy và citation
  precision/recall khi nhãn tương ứng tồn tại;
- không tự suy diễn answer correctness, groundedness hoặc unsupported claim
  rate khi chưa có gold/human labels;
- benchmark exact bytes được nhận diện bằng SHA-256;
- report ghi artifact versions, code version, metric sample counts, latency và
  portable resource observations;
- report directory immutable, không silently overwrite;
- evaluator đi qua Protocol và runner chỉ gọi `OnlineRuntime`, không biết
  concrete retrieval/model backend;
- không thêm dependency hoặc dataset ngoài AIO.

Fixture labels chỉ dùng để test code metric, không được tuyên bố là official
benchmark hay gold chất lượng.

---

## D046 — Milestone 17 Full-Corpus Profile and Artifact-Set Validation

**Status:** Accepted

Milestone 17 tách hai khái niệm:

- build profile mô tả một lần chạy full corpus có thể tái tạo;
- artifact-set validator kiểm tra kết quả đã persist mà không biết raw field AIO.

Policy:

- full-corpus profile pin dataset revision
  `0a39ad7eae8e6c188cb225c4b1443c3b346461d8`;
- `sample_limit` phải là `null`;
- expected component counts nằm trong config dataset-specific, không hard-code
  vào core validator;
- profile hiện tại kỳ vọng 153.420 metadata, 178.665 content và 897.890
  relationships;
- build chỉ thành công sau khi manifest schema, dataset identity, SHA-256,
  payload/index count và cross-artifact lineage đều hợp lệ;
- validation report được persist một lần dưới artifact root và không overwrite;
- `legal-rag-validate` revalidate read-only một artifact set đã tồn tại;
- không commit full corpus, index, model output hoặc validation report sinh ra;
- không được tuyên bố đã hoàn thành full-corpus run nếu chưa có report thật với
  `is_full_corpus = true` và `is_valid = true`.

Milestone này không đổi embedding/reranker/generator model, không fine-tune và
không thêm dependency. Tại thời điểm D046, runtime vẫn materialize các stage lớn;
D047 sau đó bổ sung bounded source passes, stage release và safe resume nhưng
không thay yêu cầu phải đo full-run thực tế.

---

## D047 — Milestone 17.1 Staged Memory Release and Safe Resume

**Status:** Accepted

Full-corpus profile dùng staged offline execution mà không đổi unified data
contract:

- source được đọc nhiều pass chỉ khi dataset revision đã pin;
- pass đầu chỉ xác nhận component counts, audit và normalization đọc lại cùng
  immutable revision;
- normalized, relationships, graph, cleaned, blocks, chunks, BM25 và vector
  được persist ngay sau khi hoàn thành thay vì giữ tới cuối build;
- raw `content_html` vẫn tồn tại trong persisted normalized và cleaned artifact
  theo schema; sau khi cleaned artifact đã checksum, runtime chỉ bỏ reference
  HTML trong processing view dùng cho parser/chunker;
- vector batches được ghi vào một NumPy `float32` matrix thay vì tích lũy Python
  list-of-lists;
- explicit garbage collection chỉ chạy tại stage boundary;
- `build_state.json` giữ typed schema version, application config SHA-256, code
  version và timestamp;
- resume là opt-in và chỉ chấp nhận partial build đã có dataset manifest, audit,
  normalized checkpoint cùng exact config/code identity;
- stage dependency thiếu, checksum sai hoặc config/code đổi đều fail closed;
- complete/failed `build_validation.json` không được resume hay overwrite.

Resume không che giấu lỗi và không tự xóa partial artifact. Failure trước
normalized checkpoint phải dùng artifact root mới. Milestone này không thêm
dependency hoặc thay model.

---

## D048 — Milestone 18 Backend-neutral Model Generation

**Status:** Accepted

M18 thêm model-backed generation nhưng không chọn model production khi chưa có
benchmark chính thức:

- `AnswerGenerator` tiếp tục là contract mà fixed service/tools/Agent sử dụng;
- `ChatModelProvider` cô lập concrete model endpoint khỏi core;
- reference provider dùng OpenAI-compatible Chat Completions qua standard
  library, không thêm LLM SDK;
- model name và revision phải pin khi bật model mode;
- API key không nằm trong config, chỉ tên environment variable được lưu;
- prompt chỉ chứa original question và selected evidence;
- completion phải là `ModelAnswerDraft` có schema strict;
- model chỉ chọn evidence ID; citation metadata luôn dựng từ evidence thật;
- unknown ID, thiếu marker, JSON/schema sai, timeout hoặc backend failure đều
  fail closed qua exception taxonomy và bounded Agent policy;
- model-declared insufficient evidence luôn trở thành standard abstention;
- `extractive` vẫn là default để local UI và test không phụ thuộc network/model.

M18 không fine-tune, không hard-code model, không tuyên bố semantic citation
verification, không gửi raw corpus ngoài selected evidence và không log prompt
hoặc legal content. Model cuối cùng sẽ được chọn bằng benchmark trên GPU sau khi
có dữ liệu/metric phù hợp.

---

## D049 — Canonical Configuration Hashing for Reproducible Resume

**Status:** Accepted

Mọi application/processing config hash dùng chung một canonical SHA-256
implementation:

- Pydantic model được chuyển sang Python values trước khi canonical hóa;
- mapping key và set/frozenset được sắp xếp ổn định;
- thứ tự list/tuple được giữ nguyên vì có thể mang ý nghĩa cấu hình;
- `Path`, `Enum`, date/time và primitive values có biểu diễn ổn định;
- NaN, infinity, non-string mapping key và type không hỗ trợ bị từ chối.

Caller phải truyền typed Pydantic config (hoặc Python-mode values) trực tiếp
vào canonical hasher. Không được gọi `model_dump(mode="json")` trước vì bước đó
có thể biến set/frozenset thành list với process-dependent order. M38 sửa cả
competition offline-build identity và batch-checkpoint identity theo invariant
này; cross-process resume test bảo vệ regression.

Quy tắc này áp dụng cho dataset source, audit, normalization, cleaning, parsing,
chunking, BM25, vector, graph và full-build recovery identity.
`OfflineBuildState` được nâng lên schema `1.1`.

State schema `1.0` chỉ lưu nondeterministic digest, không lưu config gốc, nên
không thể migrate hoặc chứng minh tương đương an toàn. Runtime phải fail closed,
giữ artifact cũ để chẩn đoán và yêu cầu artifact root mới. Cross-process tests
với nhiều `PYTHONHASHSEED` bảo vệ invariant này. Quyết định không thêm dependency.

---

## D050 — Measured OOM Requires Bounded Document and Index Processing

**Status:** Accepted

Full-corpus run trên Colab 12 GiB đo được legacy `legal-rag-build` bị Linux
OOM-kill trong legal structure parser ở khoảng 10,8 GiB anonymous RSS. Đây là
bottleneck đã đo, không phải suy đoán hoặc GPU limitation.

Version `0.20.0` áp dụng policy:

- parser/chunker giữ tối đa một document cùng blocks/chunks của document đó;
- cleaned, block và chunk JSONL đi qua typed one-pass iterator;
- processed artifact dùng staging writer incremental và chỉ publish atomically
  sau count/manifest/checksum validation;
- per-document output phải tương đương thuật toán parser/chunker hiện có;
- BM25 build dùng disk-backed SQLite và configurable bounded insert batches;
- vector builder embed bounded batches trực tiếp vào disk-backed NumPy memmap;
- vector rows và chunk records giữ deterministic source-artifact order;
- source artifact checksum thay corpus-sized chunk-ID list trong processing
  hash;
- progress logging là configurable nhưng không log legal content;
- không thêm dependency, model, dataset hoặc thay retrieval/generation logic.

In-process `ParsedLegalDocument`, `ChunkedLegalDocument` và `VectorBuildBatch`
có consumer rõ ràng nhưng không trở thành persisted unified schema. Legacy
list-based parser/chunker API được giữ cho fixture và bounded callers; production
offline composition dùng streaming path.

Do exact code/config identity là resume invariant, partial build `0.19.x` không
được sửa state để chạy tiếp bằng `0.20.0`. Full rerun phải dùng artifact root
mới. Full-corpus completion vẫn cần report thật `is_full_corpus = true` và
`is_valid = true`.

---

## D051 — GPU Interruption Requires Durable Vector Batch Checkpoints

**Status:** Accepted

Full-corpus execution trên Colab Free đo được vector stage bị runtime
termination sau khi model đã khởi tạo nhưng trước khi 1.278.201 chunks được
embedding xong. Version `0.20.0` chỉ bounded memory; random staging directory
không có committed offset nên mỗi GPU session phải embedding lại từ đầu.

Version `0.20.1` áp dụng policy:

- một deterministic `.vector.partial` workspace cho mỗi destination;
- NumPy memmap và chunk JSONL được flush trước khi atomic checkpoint công bố
  `next_offset` và chunk byte boundary;
- resume bỏ qua unified chunk stream tới committed offset mà không gọi embedding
  provider cho phần đã hoàn thành;
- source manifest/model/provider/dimension/dtype/batch identity không khớp thì
  fail closed và không sửa checkpoint;
- cadence checkpoint là execution-only config, không thay đổi final artifact;
- final `vector/` chỉ được publish bằng directory rename sau count/checksum;
- random `.vector-*` legacy staging không được suy diễn thành checkpoint;
- chỉ transition build state `0.20.0 → 0.20.1` được chấp nhận vì thay đổi này
  giữ nguyên canonical config hash và artifact lineage; transition khác vẫn bị
  từ chối.

Không thêm dependency, không đổi embedding model, vector format online,
retrieval score hoặc corpus.

---

## D052 — Memory-bounded Online Vector Loading

**Status:** Accepted

Full-corpus serving trên Colab 12 GiB đã đo được process bị OOM-kill (`-9`) sau
khi BM25 load xong. Nguyên nhân là online NumPy backend materialize 1.278.201
`LegalChunk` Pydantic objects, đồng thời validation/search có thể tạo bản sao
ma trận ở kích thước corpus.

Version `0.20.2` áp dụng policy:

- giữ nguyên artifact `vectors.npy` + `chunks.jsonl` + `manifest.json`; không
  rebuild hoặc migrate vector artifact đã validation thành công;
- scan `chunks.jsonl` một record mỗi lần, validate schema/checksum/count/order,
  rồi chỉ giữ byte offsets, chunk IDs và compact postings cho unified filters;
- chỉ parse full `LegalChunk` cho các hit cuối cùng được trả về;
- kiểm tra finite/unit-norm của vector theo configurable batch;
- exact cosine scoring theo configurable batch, không advanced-index toàn ma
  trận;
- exact top-k và tie-break theo `chunk_id` giữ nguyên;
- startup log checksum, metadata-scan và vector-validation progress nhưng không
  log legal content;
- execution bounds nằm trong `online.vector_runtime`, không tham gia artifact
  processing hash;
- build state `0.20.0` hoặc `0.20.1` được phép nâng lên `0.20.2` vì thay đổi này
  không đổi offline output, config hash, model identity hoặc artifact lineage.

Không thêm dependency, backend/model/dataset, cache phân tán hoặc business
logic. Full-corpus online smoke vẫn phải chạy lại sau khi cài version này.

---

## D053 — Full-corpus BM25 Planning and Validated Fast Startup

**Status:** Accepted

Full-corpus measurement ghi nhận BM25-only query mất 279 giây và startup lặp lại
nhiều checksum/integrity scan trên artifact set đã deep-validate. Version
`0.20.3` áp dụng:

- SQLite FTS5 top-k dùng hidden `rank` và không global secondary-sort theo
  `chunk_id`;
- corpus-aware query planner dùng temporary `fts5vocab`, ưu tiên term có
  document frequency thấp, giữ số và legal semantic modifier, với typed bounds
  tại `online.bm25_runtime`;
- query planner không static/aggressive stopword removal và không thay đổi
  analyzer hay BM25 artifact format;
- startup `full` vẫn là mặc định và giữ deep integrity behavior;
- startup `validated_report` chỉ bỏ corpus-sized scan sau khi exact online
  manifests khớp một `BuildValidationReport` hợp lệ có đủ checksum, count,
  SQLite integrity và vector-shape checks;
- Sentence Transformer provider công bố configured dimension mà không load
  weights; actual model dimension vẫn được kiểm tra khi model lazy-load;
- startup log latency riêng cho manifest, BM25, vector, graph và tổng runtime;
- partial offline build `0.20.2` không tự resume bằng `0.20.3` vì canonical
  application config có thêm online runtime fields; complete artifacts vẫn được
  load lại mà không rebuild.

Không rebuild dataset, legal chunks, BM25 hoặc vector artifact; không thêm
dependency hay backend mới.

---

## D054 — Persisted Vector Serving Metadata Sidecar

**Status:** Accepted

Full-corpus `0.20.3` measurement giảm startup xuống 159 giây nhưng
`vector/chunks.jsonl` metadata scan vẫn mất 141,8 giây. Version `0.20.4` áp dụng:

- `legal-rag-prepare-serving` tạo sidecar đúng một lần từ validated vector chunk
  JSONL;
- sidecar SQLite lưu row index, byte offset, chunk ID và exact unified filter
  columns; không lưu embedding hoặc legal text;
- manifest khóa exact vector source identity và `chunks_sha256`;
- build sidecar streaming/batched, atomic, checksum-validated và không overwrite;
- runtime mở sidecar read-only/immutable, query final offsets/filter indexes và
  chỉ Pydantic-parse final hit records;
- `prefer_serving_metadata` cho phép fallback tương thích; production/full-corpus
  có thể bật `require_serving_metadata` để fail closed thay vì scan chậm;
- vector matrix, cosine scoring, chunk-ID tie-break và public retrieval schemas
  không thay đổi.

Không re-embed, không rebuild vector/BM25, không đọc raw dataset và không thêm
dependency. Sidecar không được commit vào Git.

---

## D055 — Same-origin HTTP Diagnostic UI

**Status:** Accepted

Full-corpus smoke trên Colab đo được FastAPI health vẫn `200`, không có OOM và
không có request xử lý câu hỏi tới server khi Gradio báo mất kết nối. Nguyên nhân
là Gradio 6 dùng SSE với public root suy ra từ internal host sau Colab proxy.
Version `0.20.5` áp dụng:

- `/ui` là một trang HTML nhỏ do FastAPI phục vụ;
- UI gọi public `/api/v1/answer` bằng same-origin `fetch`;
- không dùng queue, SSE, WebSocket hoặc public proxy URL trong config;
- UI không gọi trực tiếp runtime, retriever, model hoặc artifact backend;
- answer, citation và warning được gán bằng `textContent`;
- FastAPI lifespan vẫn chỉ load đúng một immutable online runtime;
- bỏ dependency Gradio vì không còn consumer.

Đây vẫn là diagnostic baseline UI, không phải production frontend và không thêm
authentication, CORS, reverse proxy hay deployment framework.

---

## D056 — Optional GPU-resident Exact Dense Scoring

**Status:** Accepted

Full-corpus smoke `0.20.5` trên Colab GPU đo được dense retrieval mất 52,28 giây
trong khi reranker CUDA mất 17,02 giây. Vector artifact là normalized float32
matrix 1.278.201 x 384 và vừa bộ nhớ GPU 16 GiB. Version `0.20.6` áp dụng:

- `online.vector_runtime.search_device` là typed execution choice `cpu|cuda`;
- CUDA path dùng PyTorch đã có qua dependency model hiện tại và import lazy;
- matrix NumPy memory-map được chuyển theo bounded batches rồi giữ resident trên
  GPU cho toàn bộ vòng đời server;
- unfiltered query dùng exact GPU matrix-vector product;
- filtered query dùng bounded `index_select`, không copy toàn matrix lần nữa;
- output score/rank/schema, vector manifest và sidecar không thay đổi;
- explicit CUDA request fail startup nếu device không khả dụng hoặc load lỗi;
- CPU NumPy path vẫn là default và không phụ thuộc CUDA;
- không thêm FAISS/vector database, không re-embed và không rebuild artifact.

Quyết định này là runtime acceleration của reference backend, không phải quyết
định vector database production.

Full-corpus validation thực tế trên Colab GPU với 1.278.201 x 384 vectors:

- cold query: 31,47 giây end-to-end, gồm lazy-load embedding và cross-encoder;
- warm exact vector search: 22,21 ms;
- warm dense retrieval: 35,58 ms;
- warm cross-encoder rerank trên 30 candidates: 398,14 ms;
- warm Agent workflow: 2.041,90 ms; UI quan sát 2,16 giây;
- response không timeout và `insufficient_evidence = false`.

---

## D057 — Local Transformers Generator Behind Existing Contract

**Status:** Accepted

Full-corpus GPU smoke xác nhận warm retrieval/reranking đã giảm còn khoảng hai
giây nhưng extractive generator vẫn chỉ ghép nguyên văn evidence. Version
`0.20.7` mở rộng M18:

- thêm concrete local Transformers adapter phía sau `ChatModelProvider`;
- backend được chọn bằng `online.generation.backend`, core service/Agent không
  import model cụ thể;
- model name và immutable revision luôn phải pin;
- device, dtype, local-files policy, input/output token bounds và temperature
  đều là typed configuration;
- local model lazy-load, shared inference được serialize và deterministic khi
  temperature bằng 0;
- prompt vượt giới hạn bị từ chối thay vì truncate nội dung pháp luật;
- model completion vẫn phải qua strict `ModelAnswerDraft`, evidence-ID allowlist,
  system-built citations và CitationVerifier hiện có;
- `extractive` vẫn là safe default; OpenAI-compatible provider không đổi;
- `transformers` được khai báo direct dependency vì source code dùng trực tiếp,
  không thêm LLM SDK, Agent framework hoặc quantization package.

`Qwen/Qwen2.5-3B-Instruct` revision
`a1d308dfcc03e09da285d49d912439a655a571e8` được chọn làm candidate smoke test
cho GPU 16 GiB nhờ multilingual/structured-output capability và kích thước 3B.
Đây không phải quyết định model production: các model mạnh hơn vẫn phải được
đánh giá bằng benchmark có nhãn khi dữ liệu cuộc thi được công bố.

Không fine-tune, không thay retrieval artifact, không re-embed corpus và không
gửi selected evidence ra external service trong local mode.

---

## D058 — Bounded Structured-output Repair Without Citation Relaxation

**Status:** Accepted

Full-corpus Qwen2.5-3B smoke trên `0.20.7` ghi nhận model initialization và
completion đều thành công nhưng draft bị từ chối với `model_error`. Do raw
completion không được log, runtime chỉ công bố categorized validation failure.
Version `0.20.8` áp dụng:

- system prompt yêu cầu concise answer, JSON-only output và exact equality giữa
  `cited_evidence_ids` với marker `[E#]`;
- user prompt công bố explicit evidence-ID allowlist và output rules;
- parser có thể lấy một JSON object hợp lệ khỏi model preamble/code fence nhưng
  không tự thêm, xóa hoặc đổi field;
- strict `ModelAnswerDraft`, unknown-ID rejection, exact marker matching và
  system-built citation vẫn giữ nguyên;
- `max_structured_output_retries` cho phép tối đa một correction attempt;
- correction prompt dùng lại trusted question/evidence và failure contract,
  không đưa raw invalid completion trở lại prompt;
- log chỉ ghi schema/unknown-ID/marker category cùng attempt number, không ghi
  question, evidence hoặc completion.

Không thêm constrained-decoding dependency, không silently repair citation và
không thay Agent retrieval retry limit.

---

## D059 — Visible Evidence Markers Are the Canonical Citation Selection

**Status:** Accepted

Full-corpus `0.20.8` log xác nhận Qwen trả valid structured draft ở cả hai
attempt nhưng `cited_evidence_ids` khác marker usage/order trong answer. Vì
marker `[E#]` nằm trực tiếp cạnh legal claim còn list ID là dữ liệu dư thừa,
version `0.20.9` áp dụng:

- mọi declared evidence ID vẫn phải thuộc selected-evidence allowlist;
- mọi marker trong answer cũng phải thuộc cùng allowlist;
- sufficient answer phải có ít nhất một marker;
- thứ tự marker xuất hiện lần đầu là canonical citation order;
- redundant ID list được normalize theo marker order;
- Citation objects vẫn do hệ thống dựng từ trusted Evidence;
- insufficient answer có marker, unknown marker hoặc answer không marker đều
  fail closed.

Đây là deterministic normalization của hai biểu diễn citation trong cùng model
draft, không phải semantic citation verification và không cho phép model tạo
căn cứ mới.

---

## D060 — Deterministic Rendering of Verified Declared Citation IDs

**Status:** Accepted

Full-corpus `0.20.9` xác nhận Qwen tiếp tục bỏ exact `[E#]` marker trong cả hai
attempt dù JSON draft và declared citation list đã hợp lệ. Version `0.20.10`
áp dụng:

- nhận diện từng evidence ID trong combined bracket marker như `[E1, E2]`;
- khi sufficient answer không có bracket marker, render declared IDs đã qua
  selected-evidence allowlist thành marker ở cuối answer;
- citation objects tiếp tục được dựng từ trusted Evidence, không từ model text;
- không tạo ID mới hoặc suy đoán một evidence không được model declare;
- unknown declared ID, unknown visible marker và insufficient answer có marker
  vẫn bị từ chối.

Đây là bước presentation normalization cho citation IDs đã được model declare
và hệ thống xác minh, không phải semantic verification của từng claim.

---

## D061 — Cross-encoder Must See Unified Legal Scope Metadata

**Status:** Accepted

Full-corpus answer smoke testing returned the correct “12 ngày” conclusion,
but the highest-ranked citation came from Article 23 of a document whose scope
was limited to domestic workers. BM25 and dense retrieval use title-aware
`search_text`, while the cross-encoder previously received only the chunk body.
The reranker therefore could not reliably distinguish a generally applicable
provision from a provision with a narrower document scope.

Starting with version `0.20.11`:

- the default cross-encoder input mode is `legal_context`;
- the candidate text places named unified metadata before the legal body;
- eligible metadata includes document title, number, type, issuing authority,
  legal field, effect metadata, and legal structure;
- raw dataset fields, arbitrary metadata, and source URLs are excluded;
- `text_only` remains available for controlled A/B comparison;
- the raw cross-encoder score remains the final reranking score;
- no heuristic boost or penalty is added;
- retrieval-hit identity, trace provenance, and artifact identity remain
  unchanged;
- existing chunks, embeddings, BM25, vector, and graph artifacts do not need to
  be rebuilt.

Effect metadata remains a dataset snapshot under D017 and must not be presented
as an independent confirmation of current legal validity.

---

## D062 — Deterministic Query Understanding and Bounded Multi-query RRF

**Status:** Accepted

The baseline router previously followed a fixed strategy order and its retry
rewriter could only alternate between original and normalized question forms.
Version `0.21.0` adds query intelligence without depending on AIO fields,
competition labels, an LLM, or external legal knowledge:

- `QueryUnderstandingService` extracts only explicit document numbers,
  Điều/Khoản/Điểm references, years, scope cues, relationship cues and a
  conservative intent;
- runtime recomputes trusted analysis instead of accepting client-supplied
  analysis;
- variants are limited to normalized text, removal of generic framing, and
  concatenation of references already present in the question;
- `online.query_understanding.max_variants` is bounded from 1 to 5 and defaults
  to 3;
- hybrid retrieval executes BM25 and dense once per active variant and applies
  standard unweighted RRF over every branch rank;
- backend raw scores are never added or normalized together;
- per-variant branch rank, raw score and contribution are persisted in typed
  retrieval trace fields;
- duplicate chunk IDs with conflicting legal payload fail closed;
- relationship intent prioritizes graph retrieval, while explicit requested
  strategy remains first and Agent retries remain capped at two;
- disabled query understanding preserves the legacy single-query path;
- existing legal chunks, BM25, vector and graph artifacts remain compatible.

This is deterministic query planning, not semantic question understanding.
Synonym expansion, legal-term invention, LLM rewriting, learned fusion weights
and semantic applicability grading remain outside M19.

---

## D063 — Deterministic Evidence Applicability Screening Before Generation

**Status:** Accepted

Retrieval and reranking scores alone do not explain whether selected context
matches an explicit legal reference in the user's question. Version `0.22.0`
adds a deterministic selection layer without claiming semantic legal judgment:

- the selector reads only unified retrieval metadata and trusted
  `QueryAnalysis`;
- source rank remains a signal but an exact document/article match may outrank
  a higher raw retrieval rank;
- lexical overlap is computed deterministically and used only inside context
  selection;
- inactive effect status is penalized only when the corresponding label is
  explicitly configured;
- unknown effect status remains selectable and produces a warning;
- reference mismatch is demoted and traced, not silently deleted;
- every unique retrieval hit is classified as selected, count-omitted or
  token-budget-omitted;
- selection score is not added back into BM25, dense, RRF or cross-encoder
  scores;
- the structural grader requires selected coverage for explicit document and
  article references and fails closed for context containing only inactive or
  reference-mismatched evidence;
- metadata explicitly records that semantic relevance and legal applicability
  were not interpreted;
- existing corpus and index artifacts remain compatible.

This policy improves evidence precision and debuggability but does not verify
current legal validity, resolve conflicts between instruments, infer exceptions
or prove claim-level support.

---

## D064 — Synthesized Legal Claims Require Inline Grounding

**Status:** Accepted

Exact citation identity does not prove that the sentence beside a citation is
supported by that evidence. Version `0.23.0` adds a fail-closed deterministic
claim-grounding layer:

- only synthesized answers are split into bounded claims;
- every legal claim requires an inline evidence marker by default;
- each marker must exist in both response citations and selected evidence;
- response citations unused by any claim are rejected;
- lexical support is measured against only the evidence linked to that claim;
- quantities in a claim must occur in linked evidence;
- negation terms introduced by a claim must occur in linked evidence;
- one marker on a later sentence cannot silently cover earlier sentences;
- any unsupported claim invalidates the whole answer and triggers abstention;
- per-claim results and coverage are persisted in verification metadata;
- extractive answers skip claim grounding because their legal text is verbatim,
  but exact identity verification remains mandatory;
- checks are bounded by typed configuration and add no model dependency.

This is deterministic claim grounding, not semantic entailment, contradiction
detection or complete legal reasoning. A learned verifier may later implement
the existing `CitationVerifier` boundary after labeled evaluation is available.

---

## D065 — Semantic Verification Is Optional, Two-stage and Fail-closed

**Status:** Accepted

Version `0.24.0` adds an optional model-backed semantic verifier behind the
existing `CitationVerifier` contract:

- deterministic M21 identity, marker, lexical, numeric and negation checks
  always run first;
- the model runs only when all hard checks pass on a synthesized answer;
- the model receives only each claim and its already-linked evidence;
- untrusted output contains only `claim_id` and a three-way support label;
- trusted evidence IDs and citation metadata are never accepted from the model;
- every expected claim must appear exactly once and in deterministic order;
- `contradicted`, `insufficient`, malformed output, provider failure and
  incomplete assessment all fail closed;
- provider, provider version, model name and pinned model revision are persisted;
- `disabled` is the default backend, preserving GPU/network-independent local
  behavior;
- OpenAI-compatible and local Transformers providers remain replaceable behind
  `ChatModelProvider`;
- generator and semantic verifier share one local Transformers weight set only
  when model, revision, device, dtype and local-files policy match exactly;
- shared consumers retain independent input/output bounds and serialize access
  through one reentrant inference lock;
- no corpus artifact is rebuilt and no new dependency is added.

This verifier improves claim-level semantic screening but is not a proof of
legal correctness. Threshold/model selection still requires a labeled
competition benchmark or a reviewed legal evaluation set.

---

## D066 — Model Comparison Requires Reproducible Like-for-like Reports

**Status:** Accepted

Version `0.25.0` adds a generic comparison layer over immutable M16 evaluation
reports:

- every new evaluation report records pinned dataset lineage, a sanitized
  runtime-configuration hash, component/model identities, and relevant package
  versions;
- comparison requires exact benchmark bytes, equal case counts, equal cutoffs,
  equal dataset/revision, and equal labeled-case counts for quality objectives;
- failed runs, missing objective metrics, missing runtime provenance, and
  threshold failures remain explicit ineligible candidates;
- quality, latency, failures, and resource observations use namespaced metrics;
- accelerator identity and peak allocated memory are recorded when available
  but remain null on CPU or runtimes without compatible telemetry;
- the default result is a Pareto frontier and does not invent one winner;
- a single candidate is selected only when the comparison configuration
  explicitly requests ordered lexicographic objectives;
- comparison output is immutable and never overwrites prior evidence;
- no official metric, model winner, or benchmark quality is inferred.

M23 does not add QA data, download a model, rebuild artifacts, fine-tune, or
claim competition readiness. A winner can be trusted only after the same
pipeline is run on official competition labels or a reviewed legal benchmark.

---

## D067 — Benchmark Trust Must Be Explicit Before Model Selection

**Status:** Accepted

Version `0.26.0` requires every new evaluation run to use a typed benchmark
manifest:

- exact benchmark SHA-256, case count, and target granularities are validated
  before runtime evaluation;
- benchmark dataset name/revision must equal the loaded artifact lineage;
- label status is explicitly `diagnostic`, `human_reviewed`, or
  `competition_official`;
- trusted statuses require timestamped provenance, while this declaration is
  never presented as independent legal certification;
- diagnostic labels may support pipeline checks and Pareto analysis but cannot
  select one winner;
- optional per-objective maximum regression is measured against one explicit
  baseline candidate;
- reports with different benchmark manifest bytes cannot be compared.

M24 adds no dataset, model, training, artifact rebuild, or competition metric.

---

## D068 — Active Data Scope Is UIT DSC 2026 Task 2 Only

**Status:** Accepted

Từ Milestone 25, active repository không còn hỗ trợ corpus AIO. Quyết định này
supersede D001 và các phần AIO-specific của D015, D016, D017, D027, D028,
D029, D033, D034, D035 và D036:

- source, adapter, audit service, normalizer, relationship normalizer, build
  runtime, config profile, raw fixture và tests chỉ dành cho AIO bị loại bỏ;
- dependency `datasets` bị loại khi không còn consumer;
- online runtime chỉ chấp nhận artifact có dataset identity chính thức được
  competition config khai báo;
- dữ liệu QA và context của BTC đi qua package adapter riêng trước khi vào
  unified schema;
- raw field names của BTC không được lan vào cleaner, parser, chunker, indexer,
  retrieval, generation, Agent hoặc serving;
- artifact AIO bên ngoài repository không bị tự động xóa, nhưng không còn được
  runtime chấp nhận;
- Git history không bị rewrite.

BTC đã xác nhận answer-text task, context raw fields, METEOR primary và ROUGE-L
secondary. Tuy nhiên scorer implementation, train corpus và
selected-context bytes vẫn phải được kiểm tra trước khi triển khai phần phụ
thuộc chúng. Warm-up answers không tạo ra retrieval relevance labels.

---

## D069 — Official Context Files Map One-to-one to Unified Documents

**Status:** Accepted

The one-file/one-document mapping remains accepted. Raw-ID, optional-title and
cleaning clauses below are superseded by the released-data evidence in D081.

The organizer overview describes each `context_*.json` file as one selected legal
document with `id`, `name`, `link`, and `passage`. Milestone 26 applies
the following fail-closed mapping:

- canonicalized `id` becomes the unified string `document_id`;
- optional `name` becomes nullable `title`;
- `link` becomes `source_url`;
- `passage` becomes raw-preserving normalized `clean_text`, then D081 creates a
  separately cleaned `clean_text`;
- `source_dataset` is the official UIT DSC 2026 Task 2 corpus identity;
- missing legal metadata is left null rather than inferred;
- raw organizer field names remain inside the UIT DSC adapter boundary;
- one file must contain exactly one JSON object; list roots are rejected;
- ZIP members are read without extraction and duplicate member names or context
  IDs are rejected;
- corpus revision is a deterministic SHA-256 over ordered context filenames and
  exact member bytes, so an archive and its equivalent extracted directory have
  the same lineage;
- dataset-specific cleaning is inserted before parsing under D081;
- ingestion emits distinct normalized and cleaned manifests;
- Warm-up answers remain answer-level references and are not converted into
  document/chunk relevance labels.

The released archive confirms numeric JSON context IDs and optional `name`.
The raw adapter accepts non-negative integers or non-blank strings and
canonicalizes them to unified string IDs, while rejecting booleans, floats,
nulls, blanks and canonical collisions. Only the two audited field sets are
accepted; future changed bytes require a new audit.

No BM25, vector, graph, or training artifact may be claimed as official unless
its source bytes have passed the corpus audit and its manifest pins that exact
revision.

---

## D070 — Official Builds Are Staged, Immutable, and Resumable

**Status:** Accepted

Milestone 27 composes the official corpus boundary into the existing offline
pipeline under these recovery rules:

- a build is identified by exact source revision, application-config hash, and
  code version;
- completed stages are recorded atomically and may be reused only after their
  persisted payloads and manifests validate;
- a mismatched source, config, code version, missing stage output, or broken
  checksum fails closed instead of rebuilding over an existing destination;
- vector embedding uses the existing durable batch checkpoint;
- normalized and cleaned pass-through artifacts remain explicit so lineage is
  auditable even though organizer passages are already plain text;
- the documented corpus provides no relationship field, therefore the build
  emits a truthful zero-record relationship artifact and a zero-edge graph over
  the official documents; it never infers or fabricates legal relationships.

The build can be exercised with fixtures now. Full-corpus counts and resource
requirements remain unclaimed until the organizer archive is available.

---

## D071 — Batch Inference Checkpoints Are Not Submission Files

**Status:** Accepted

Competition batch execution persists one internal typed `AnswerResponse` per
official question ID. It validates exact question-source bytes, preserves input
order, rejects duplicate or unknown checkpoint IDs, and resumes only missing
questions. A completed run requires exactly one result for every input ID.

This internal JSONL plus manifest remains intentionally submission-neutral.
D077 defines the separate Codabench formatter after executable scorer behavior
was verified. Reference answers, when present in warm-up data, are never copied
into predictions.

---

## D072 — Codabench Submission Is an Exact Answer-only ZIP Contract

**Status:** Superseded by D077

The organizer prose available at implementation time described a file named
`submission.zip` whose only member is UTF-8 `submission.json`. It described the
JSON root as an array in official question order with items containing:

```text
id: string
answer: string
```

The formatter consumes only a completed M27 batch. Before publishing, it
requires the exact question-source SHA-256, result JSONL SHA-256, record count,
and ordered IDs to agree. Missing, duplicate, reordered, unknown, or tampered
records fail closed. Existing output is never overwritten, ZIP member metadata
is fixed for reproducibility, and reference answers are never read as
predictions. Internal citations and other `AnswerResponse` metadata are not
included as separate submission fields. D073 defines the score-facing removal
of verified internal evidence markers from the answer string.

The live Codabench scoring program later contradicted this prose by calling
`.items()` on the submitted root. D077 records the executable contract that now
governs packaging.

---

## D073 — Competition Text Metrics Are Diagnostic Until Scorer Equivalence

**Status:** Accepted

BTC has confirmed METEOR as the primary ranking metric and ROUGE-L as the
secondary metric, both compared against expert reference answers and optimized
upward. Version `0.31.0` therefore adds dependency-free local diagnostics:

- NFC/casefold tokenization over Vietnamese letters and numbers;
- exact-token METEOR precision/recall with standard weighted harmonic mean and
  fragmentation penalty;
- token-level ROUGE-L F1 using longest common subsequence;
- null scores when no reference answer exists;
- an explicit report warning that these values are not official-equivalent.

No stemming, synonym resource, external language model, or organizer tokenizer
is inferred. These local values may compare answer variants on one pinned
benchmark, but cannot be represented as exact Codabench scores until scorer
implementation or verified parity evidence is available.

D080 later records the official scorer source. D073 remains the accepted
contract for the existing diagnostic implementation; it does not become
official-equivalent merely because the official algorithm is now known.

Because `[E<number>]` is internal grounding notation rather than expert answer
text, the competition formatter removes only markers whose evidence IDs exist
in the verified `AnswerResponse.citations`. Unknown markers fail closed. The
internal batch keeps full citations and verification metadata.

---

## D074 — Warm-up Scoring Is Answer-only, Immutable, and Content-free

**Status:** Accepted

Version `0.32.0` adds a local scorer for the exact official warm-up answer
contract. It accepts only `warmup.json` records with reference answers and an
exact one-member `submission.zip`. Submission IDs must equal reference IDs once
and in source order.

The report records per-ID and mean exact match, diagnostic METEOR, and diagnostic
ROUGE-L, together with SHA-256 identities for the reference file, archive, and
submitted JSON bytes. It never persists questions, reference answers, or
prediction text. Output is a new immutable directory and is not overwritten.

This scorer requires no corpus, index, model, GPU, network, or external metric
package. It is intended for fast local regression on organizer-provided
references and retains the D073 non-equivalence warning.

---

## D075 — Organizer Compliance Is a Fail-closed Runtime Gate

**Status:** Accepted

The organizer rules supplied on 2026-08-01 are identified by SHA-256
`c88b2eec6bccf2bc809e0b7982cbe113c56928671f99c7acb5e741fc310091be`.
They confirm that competition work may use only organizer-provided data and may
not add manual labels, external collection, or external data augmentation.

M31 applies these rules as follows:

- external/AIO data and artifacts remain rejected by active runtime lineage;
- model candidates are not competition-approved models;
- any model whose organizer approval or original license is unknown is blocked
  from an official run;
- a model proposal records immutable identity, purpose, license and approval
  evidence before its status changes to approved;
- private submissions require a human-reviewed preflight and a ledger enforcing
  awareness of the three-per-day limit;
- Data Statement and Model Card snapshots are required release evidence but are
  not inserted into the exact answer-only ZIP unless the organizer changes its
  output contract;
- source owned by the team is MIT licensed, while third-party data, packages,
  tokenizers and weights retain their own licenses;
- platform naming, pretrained-model scope, Internet availability, final Docker
  interface and local scorer implementation parity remain unresolved questions.

This decision records compliance controls only. It does not approve a model,
model license, dataset byte sequence, Docker GPU image or official score.

---

## D076 — Competition Packaging Tools Must Not Require Serving Dependencies

**Status:** Accepted

Submission formatting and warm-up scoring operate only on local typed files.
Their console entry points therefore live under the UIT DSC competition package
and must not import FastAPI, Uvicorn, API routes, runtime artifacts or models.

The M31 Docker context excludes data, artifacts, models, checkpoints, logs,
reports, secrets and submission files. Its CPU image is a non-root compliance
scaffold with a pinned Python base and direct dependency constraints. It is not
the final private-test GPU image: the final base digest, complete package freeze,
model acquisition policy and reproduction command require organizer runtime
details and approved models.

---

## D077 — Executable Codabench Scorer Defines an ID-keyed Submission Object

**Status:** Accepted

The first real warm-up upload failed before scoring with:

```text
y_pred = {k: v['answer'] for k, v in y_pred.items()}
AttributeError: 'list' object has no attribute 'items'
```

Therefore the executable scorer requires the UTF-8 `submission.json` root to
be an object keyed by official question ID:

```text
{
  "<question_id>": {"answer": "<answer text>"}
}
```

The ZIP still contains only `submission.json`; each value contains only one
string `answer`; and IDs must remain complete and unique. D080 source analysis
shows the scorer aggregates by prediction keys and does not depend on object
order. The repository nevertheless preserves official source order as a
stricter deterministic packaging/reproducibility gate. The formatter and local
scorer reject the formerly documented array. Executable scoring behavior takes
precedence over contradictory prose until BTC publishes a corrected scorer or
contract.

---

## D078 — Official-only Training Is Allowed but Synthetic Data Is Prohibited

**Status:** Accepted

BTC's written reply on 2026-08-01 confirms:

- Task 2 submissions use Codabench;
- there is no predeclared model allow-list;
- each intended open-source model must be registered by name and official URL
  through the organizer's forthcoming Google Form;
- preprocessing, indexing, retrieval and fine-tuning are allowed when they use
  only official competition data;
- synthetic data is prohibited even when generated from official data.

The active runtime therefore remains official-only and all current model
candidates remain blocked from an official run until registration evidence is
available. Fine-tuning may begin only after real official supervision, split,
leakage control, evaluator and experiment plan exist. The project must not
manufacture QA pairs, answers, evidence labels, hard negatives or other
training examples to compensate for missing supervision.

Parameter budget, pretrained-model data treatment, API prohibition and flexible
reproduction packaging are resolved by D079. Final runtime hardware, scoring
environment and Private Test interface remain unresolved.

---

## D079 — Competition Model Budget, Local Control and Reproducibility

**Status:** Accepted

Thông báo chung mới nhất của BTC xác nhận các ràng buộc sau cho từng task:

- tổng tham số của toàn bộ model/hệ thống phải nhỏ hơn 4 tỷ; embedding,
  reranker, generator, verifier/grader dùng model và mọi model phụ trợ đều được
  cộng vào cùng một ngân sách;
- distillation được phép nếu model/hệ thống cuối cùng thực sự dưới 4 tỷ tham số;
- quantization, LoRA và kỹ thuật giảm bộ nhớ không thay đổi số tham số để xét
  điều kiện; model gốc từ 4 tỷ tham số trở lên không trở thành hợp lệ;
- không được dùng bất kỳ API nào, kể cả API miễn phí hoặc phi lợi nhuận; model
  phải được đội tải, chạy và kiểm soát trực tiếp;
- model mã nguồn mở hoặc có license phi thương mại/nghiên cứu/giáo dục có thể
  được chấp nhận, nhưng vẫn phải đăng ký, kiểm tra license và tái lập được;
- chỉ dữ liệu BTC được dùng trực tiếp và data augmentation vẫn bị cấm; dữ liệu
  pretraining của pretrained model/LLM không được xem là dữ liệu ngoài;
- BTC sẽ cung cấp training data cho Task 2;
- Docker, GitHub hoặc ZIP source/weights đều là hình thức giao nộp có thể chấp
  nhận nếu README cho phép BTC tái lập; việc tải trọng số model hợp lệ từ
  Internet cũng được phép.

Vì vậy một candidate chỉ đủ điều kiện competition khi model inventory ghi exact
identity, URL, revision, license, vai trò, parameter count có bằng chứng và trạng
thái đăng ký BTC. Gate phải tính tổng tham số theo cấu hình thực tế và fail closed
nếu thiếu bất kỳ count nào hoặc tổng `>= 4_000_000_000`.

Quyết định này chưa xác nhận cấu hình model hiện tại dưới giới hạn. Việc kiểm kê
và cộng tham số là bước bắt buộc riêng trước official run.

---

## D080 — Organizer Scorer Source Defines Metric Tokenization and Aggregation

**Status:** Accepted

BTC cung cấp `Scoring-Program-Task-LegalQA.zip` có SHA-256
`4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`.
Read-only source analysis xác nhận:

- prediction root là object; mỗi ID được chiếu sang `value["answer"]`;
- METEOR chuyển reference/prediction sang `str`, tách bằng whitespace
  `str.split()` và gọi NLTK `meteor_score` với defaults;
- scorer tải `wordnet` và `omw-1.4` khi chạy;
- PyVi import/call bị comment và không tham gia scoring;
- ROUGE-L dùng package `rouge_score` vendored trong ZIP với
  `use_stemmer=False`;
- vendored default tokenizer lowercase, thay ký tự ngoài ASCII `[a-z0-9]` bằng
  khoảng trắng, rồi tính LCS precision/recall/F1;
- METEOR và ROUGE-L đều được arithmetic macro mean trên prediction IDs;
- output `/app/output/scores.json` có keys `rouge` và `meteor`.

Local M29/M30 scorer không parity vì dùng NFC/casefold Unicode tokenization và
exact-token-only METEOR. Nó tiếp tục mang diagnostic warning cho tới khi một
official-compatible mode được implement và verified bằng golden vectors.

Repository giữ submission validation nghiêm hơn scorer: exact complete IDs,
source order, duplicate-key rejection và string-only answer. Không nới lỏng các
gate này chỉ vì official script mới kiểm count và coerce answer qua `str`.

Archive không chứa dependency lock hoặc exact NLTK/NumPy/WordNet identities.
Do đó công thức, tokenizer ROUGE và aggregation đã resolved; absolute runtime
reproduction vẫn phải pin các dependency/resource còn thiếu. Chi tiết và member
checksums nằm tại `docs/15-OFFICIAL-SCORING-CONTRACT.md`.

---

## D081 — Official Context Boundary and Audited Passage Cleaning

**Status:** Accepted

Read-only audit ngày 2026-08-06 xác nhận `selected-contexts.zip` có 8.532 JSON
records trên canonical revision
`sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`.
Raw IDs là non-negative integers; 1.125 records thiếu `name`; 20 passages rỗng;
không có field ngoài `id/name/link/passage` và không có relationship/evidence
labels.

Boundary được chốt như sau:

- adapter canonicalize non-negative integer hoặc non-blank string ID sang
  unified string ID và từ chối boolean, float, null, blank, duplicate/collision;
- `name` optional map sang title nullable; blank passage được giữ với
  `has_content = false`;
- normalized artifact giữ raw passage sau schema mapping;
- cleaned artifact dùng dataset-specific cleaner versioned: NFC, newline,
  Unicode space/control và line-whitespace normalization, known HTML
  presentation markup removal, exact audited TVPL Pro notice removal;
- cleaner không drop duplicate records, không xóa repeated opening sequences,
  không suy diễn document metadata và không thay đổi legal meaning;
- cleaner policy identity tham gia processing hash; online/offline artifact
  compatibility tiếp tục fail closed;
- graph artifact giữ zero-edge cho tới khi official relationship evidence tồn
  tại.

Audit chính thức cũng xác nhận warm-up/train/public overlap đáng kể và train chỉ
có answer supervision. Do đó không được xem warm-up là independent dev hoặc suy
diễn retrieval/reranker labels từ answer text. Full index build và model
experiments không thuộc quyết định này.

---

## D082 — Official Parser/Chunker Is a Durable Stage Boundary

**Status:** Accepted

The official build may stop after `document_processing` so the released corpus
can be parsed, chunked and audited without initializing any embedding model or
building retrieval indexes. This is an execution boundary, not a second parser
implementation:

- the existing dataset-independent `LegalStructureParser` and `LegalChunker`
  consume only cleaned unified documents;
- raw BTC fields remain confined to the UIT DSC adapter;
- processing stays one-document-at-a-time and writes atomic JSONL artifacts;
- legal-block and legal-chunk manifests and payload checksums must validate
  before the stage is recorded complete;
- 20 blank official passages remain documents with no fabricated blocks/chunks;
- `validation_report` is absent until the final full-artifact validation stage;
- a later invocation with identical source/config/code identity resumes from
  persisted chunks and does not parse the corpus again;
- incompatible partial roots fail closed and are never silently overwritten.

This stage does not approve chunk quality, build BM25/vector indexes, infer
relationships, generate labels, or use any non-organizer data. Full-corpus
statistics must come from the measured official run rather than fixtures.

Measured evidence on 2026-08-06 confirms 1,215,092 blocks and 335,014 chunks
from all 8,532 official contexts, with exact non-whitespace character coverage.
The 20 blank contexts produced no fabricated chunks. Independent builds had
identical block/chunk payload and processing hashes. A typed-config hashing bug
found by cross-process resume was fixed before accepting the reusable root;
the incompatible diagnostic root was not migrated or edited in place.

---

## D083 — Official Pre-GPU Text and Retrieval-Unit Quality Gates

**Status:** Accepted

Full-corpus M38 audit found deterministic integrity but four quality defects
that must be fixed before any GPU/vector work:

- naked Thư Viện Pháp Luật page JavaScript remained in three organizer
  passages after known-tag cleaning;
- the article regex could parse the first Roman-numeral letter of ordinary
  phrases such as `điều của`, while implicit clause parsing confused decimals,
  tariff codes and bare years with Khoản markers;
- structural headings became tens of thousands of tiny standalone retrieval
  units despite having a clear following consumer;
- `search_text` could exceed a 512-token embedding window even when chunk text
  itself satisfied the former 512-token proxy limit.

The accepted correction is official-data-specific and fail-closed:

- cleaner `1.2` removes only balanced naked script/style blocks with audited start
  signatures, exact adjacent UI labels and the audited custom `huongdan` tag;
  an unbalanced candidate is preserved rather than deleting following text;
- legal marker parsing requires a complete article-number token and rejects
  implicit blank/decimal-nested clauses while preserving the source line as
  ordinary legal text; a bare marker keyword may join only its immediately
  following valid number line, covering organizer wraps such as `Điều\n23.`;
- `Phần/Chương/Mục/Tiểu mục` are attached to the next substantive unit and
  remain traceable through `source_block_ids`; trailing orphan headings remain
  standalone so source coverage stays complete;
- official defaults become 384 proxy tokens for chunk content and 448 for
  `search_text`; metadata is budgeted before content and complete chunk text is
  mandatory;
- exact model-tokenizer preflight remains mandatory before embedding because
  `unicode_word_v1` is not the E5 tokenizer.

All corrected artifacts must be rebuilt under a new root and lineage. The M38
root remains immutable diagnostic evidence and must not be resumed for M39.

The accepted `0.38.1` full-corpus verification produced 1,145,383 blocks and
373,253 chunks from all 8,532 official contexts. The quality scan found no
audited JavaScript residue, nested-numeric false clause, duplicate chunk ID,
search-budget violation or content-preservation mismatch. The persisted BM25
artifact has the same 373,253-record official lineage and passed a fresh-process
integrity reload plus non-empty CPU retrieval smoke. These checks establish
artifact correctness only; they do not establish retrieval quality without
labeled retrieval relevance.

---

## D084 — Exact Embedding Tokenizer Governs Competition Chunk Windows

**Status:** Accepted

Kaggle preflight on the complete `0.38.1` chunk artifact found 6,239 of 373,253
E5 inputs over the 512-token model window, with a maximum of 1,043 tokens. The
proxy budget cannot authorize vector construction, and truncation is prohibited
because it would silently remove legal content.

Competition chunking now injects `embedding_model_v1`, pinned to the same model
name, revision and document prefix as `EmbeddingConfig`. Counts include prefix
and special tokens. Source-span splitting keeps original text rather than model
decode output; metadata is dropped before legal content; tokenizer identity is
part of chunk metadata, manifest metadata and processing hash. Generic fixtures
retain `unicode_word_v1`. Corrected chunks, BM25 and vector artifacts must share
one new immutable `0.40.0` lineage; `0.38.1` remains diagnostic only.

The full `0.40.0` rebuild produced 330,768 chunks and a matching 330,768-record
BM25 index. Independent post-build preflight observed a maximum of exactly 512
tokens and zero violations across all persisted chunks. This authorizes vector
construction from `uit-dsc-2026-task2-v0400`; it does not authorize reuse of any
earlier vector artifact.

---

## D085 — Separate Immutable Build Identity From Serving Validation Policy

**Status:** Accepted

Không sửa `baseline.example.json` sau khi artifact M40 đã được tạo vì toàn bộ
ApplicationConfig thuộc build identity. Full-corpus policy và online runtime
được đặt trong `uit-dsc-2026-task2-serving.example.json`, trỏ tới cùng artifact
root nhưng không dùng để resume build.

Validation report mới phải được tạo bằng CLI với `--persist`, có tên riêng và
không được ghi đè. Online dùng report này cùng `vector_serving` sidecar để tránh
deep scan lặp lại lúc startup. Sidecar chỉ là metadata dẫn tới vector/chunk đã
checksum; nó không thay đổi retrieval corpus hoặc lineage.

Provider version là một phần compatibility contract của vector. Vector M40 được
build bằng `sentence-transformers==5.4.1`, do đó runtime cũng pin đúng version
này thay vì chấp nhận một range 5.x hoặc làm yếu kiểm tra manifest.

---

## D086 — Public Extractive Baseline Uses Hybrid-only Agent Routing

**Status:** Accepted

Batch extractive đầu tiên dùng đúng một strategy `hybrid`. Không dùng
`hybrid_rerank` mặc định vì M42 nhằm xác nhận full-corpus inference, checkpoint,
ID coverage và submission boundary trên CPU; reranker làm batch chậm hơn nhưng
chưa có relevance labels để chứng minh lợi ích chất lượng.

Query understanding và tối đa ba biến thể vẫn được giữ. Mọi run dùng reranker
hoặc model generator phải có config/output directory mới; không resume vào batch
M42 vì config hash và code version là recovery identity.

---

## D087 — M43 Qwen Candidate Is a Separate GPU Experiment

**Status:** Accepted

M43 không sửa hoặc resume batch extractive M42. Candidate dùng hybrid retrieval,
E5 query embedding và exact dense search trên GPU, sau đó sinh answer bằng
`Qwen/Qwen2.5-3B-Instruct` revision đã pin. Reranker và semantic model verifier
không được gọi.

Generation nhận tối đa 5 evidence trong context budget 3.072 token và sinh tối
đa 256 token ở temperature 0. Profile nhằm giảm độ dài answer so với trung bình
9.850 ký tự của M42; chưa được coi là candidate chính thức cho tới khi model
registration/approval evidence được xác nhận. Không dùng output M43 để nộp nếu
gate này chưa đạt.

---

## D088 — Generator Abstention Is a Fail-Closed Terminal Outcome

**Status:** Accepted

Khi generator trả về `insufficient_evidence=true`, workflow không gọi citation
verifier và không gắn `answer_verified`. Kết quả được giữ là abstention, kết
thúc với `generation_failed` và warning `generator:insufficient_evidence`.
Quy tắc này ngăn output không có evidence bị hiểu nhầm là câu trả lời đã xác
minh.

---

## D089 — M43.1 Is the Operational Control, Not the Quality Target

**Status:** Accepted

M43.1 tại commit `96e6d5a` đã chạy đủ 1.000 public questions bằng official-only
artifacts, hybrid retrieval và local Qwen2.5-3B. Submission được Codabench chấm
METEOR `0.07862292376534387` và ROUGE-L `0.16735433212043324`. Batch có đủ ID,
không answer rỗng và không retrieval model error, nhưng có 425 abstention, 384
citation-verification failure, 33 generator model error và mọi câu đều chạm
context budget.

Vì vậy M43.1 được giữ làm immutable operational control cho mọi ablation tiếp
theo, nhưng không được mô tả là quality-complete baseline. Team improvement phải
bắt đầu bằng leakage-safe dev/evaluator, sau đó thay một nhóm biến tại một thời
điểm và so sánh quality, error rate, latency cùng compliance.

Fail-closed grounding vẫn là invariant. Không được giảm abstention bằng cách tắt
verifier hoặc cho phép unsupported claim. Các hướng hợp lệ gồm retrieval/rerank
tốt hơn, context selection, verifier-guided regeneration, giữ claim supported và
grounded extractive fallback có giới hạn.

Lần resume P100 tạo 615 retrieval model error do PyTorch build không hỗ trợ
`sm_60`; suffix đó bị loại và chạy lại trên T4. Kể từ quyết định này, batch
completion chỉ là structural gate. Official candidate còn phải qua post-run
quality gate đếm retrieval/generator errors, abstention, verification failures
và exact ID coverage trước khi format submission.

Chi tiết bằng chứng và workstream nằm tại:

- `docs/16-M43-BASELINE-POSTMORTEM.md`;
- `docs/17-TEAM-IMPROVEMENT-BACKLOG.md`.

---

## D090 — M44 Uses Group-wise Leakage Quarantine and Dual Scoring Modes

**Status:** Accepted

Local development split được tạo từ official `train.json` theo nhóm câu hỏi.
Exact duplicate và near-duplicate heuristic không được phép nằm hai partition;
mọi training group giao với warm-up/public holdout bị đưa vào `quarantined`.
Thuật toán, seed, threshold, source hash, partition IDs và output hashes đều
được ghi trong manifest. Đây là protocol local, không phải gold retrieval split
và không tạo synthetic data.

Evaluator giữ hai mode tách biệt. `diagnostic` bảo toàn metric Unicode cũ;
`official_compatible` tái tạo METEOR NLTK 3.7 và ASCII ROUGE-L từ scorer BTC
checksum `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`.
Mode này fail closed nếu thiếu NLTK 3.7/WordNet và không tự tải network resource.
Không tuyên bố absolute parity khi chưa pin exact WordNet/OMW bytes hoặc BTC đổi
scorer ở phase sau.

---

## D091 — Retrieval Diagnostics Remain Content-free and Non-gold

**Status:** Accepted

M44.2 chạy BM25, dense và hybrid độc lập trên cùng official question để quan
sát overlap, diversity, explicit-reference match, latency và warning/error.
Khi development source có reference answer, lexical answer-term coverage được
phép dùng làm diagnostic hypothesis nhưng không được gọi là relevance, recall
hoặc ground truth.

Persisted report chỉ chứa question ID, retrieval identities và số liệu; không
chứa question text, reference answer hay legal passage. Output immutable và
khóa source/config/code identity. Diagnostics không thay đổi runtime strategy,
index, reranker, context builder hay generator; mọi ablation sau phải dùng config
và output riêng.

---

## D092 — Reranker Uses a Retrieval-only Gate Before Generation Ablation

**Status:** Accepted

M44.3 không chạy ngay 991 câu qua Qwen cho mọi candidate count. Trước tiên,
approved mMARCO MiniLM reranker được đánh giá read-only với candidate-k 20/40/60
trên cùng M44.1 development split. Mỗi run giữ cùng official artifacts, model
revision, top-k và query policy; chỉ candidate-k thay đổi.

Gate ghi rank churn, hybrid/reranked overlap, diversity, latency và non-gold
answer-term coverage delta. Không metric nào được gọi là retrieval relevance.
Chỉ candidate có behavior/cost hợp lý mới chạy full generation rồi chấm bằng
official-compatible METEOR/ROUGE-L. Baseline mặc định không đổi trước kết quả đó.

---

## D093 — Retrieval Comparison Reuses Shared Branch Work

**Status:** Accepted

M44.3 comparison không gọi BM25/dense lặp lại cho BM25, hybrid và
hybrid-rerank observations. Runtime enrich query một lần, lấy sparse/dense
candidate tối đa `candidate_k` một lần cho mỗi active query variant, fuse cùng
responses và rerank hybrid candidate đã có. Direct branch và pre-rerank hybrid
diagnostics chỉ project final `top_k` từ candidate response.

Đây là execution optimization, không phải thay đổi ranking policy. Nó không
thêm cache persisted, không đổi artifact/model/config contract và không dùng
answer text trong retrieval. Latency hybrid-rerank vẫn biểu diễn candidate
retrieval cộng reranker latency để so sánh end-to-end.

---

## D094 — Candidate-k 40 Must Win an End-to-end Answer A/B Before Promotion

**Status:** Accepted

Retrieval-only gate trên 991 development cases cho candidate-k 40 đạt 991
success, zero model error, thay trung bình `7.5409/20` membership so với hybrid
và tăng non-gold answer-term coverage trung bình `0.0052875`. Tuy nhiên document
diversity giảm trung bình `0.0191726`, 303 cases giảm coverage và các metric này
không phải relevance gold. Vì vậy không chạy candidate-k 60 và không thay
baseline chỉ từ retrieval diagnostics.

M44.4 chạy hai batch full generation độc lập trên cùng leakage-safe 991-question
development source. Cả hai dùng cùng code, official artifacts, query policy,
`top_k=8`, approved E5, approved mMARCO reranker, approved Qwen generator,
context builder và verifier. Hai profile chỉ khác `candidate_k=20` hoặc `40`;
agent bắt buộc dùng `hybrid_rerank` để thí nghiệm thật sự quan sát reranker.

Candidate-k 40 chỉ được promote nếu METEOR chính tăng trên cùng source; ROUGE-L,
abstention, model error, citation verification và latency là guardrail phụ. Mỗi
run có output/checkpoint/config hash riêng. Public baseline M43.1 không bị thay
đổi trước khi A/B hoàn tất và được review.

**Kết quả M44.4 (2026-08-12):** Hai output `v0446` đã được kiểm tra checksum,
đóng gói lại qua `legal-rag-submit` và chấm bằng
`legal-rag-score-warmup --metric-mode official_compatible` trên đúng 991 records
của development source SHA-256
`8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`. Runtime
chấm dùng source scorer BTC SHA-256
`4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`,
NLTK `3.7`, NumPy `2.1.3`, WordNet ZIP SHA-256
`cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59` và
OMW 1.4 ZIP SHA-256
`3b941e664852f3297b6040236626065796a2aaf7d7f9eec8779a3beaa1096c2d`.

| Metric/guardrail | k=20 | k=40 | Delta k40 − k20 |
|---|---:|---:|---:|
| METEOR (primary) | 0.07470770 | 0.07683363 | +0.00212592 |
| ROUGE-L | 0.16222435 | 0.16459483 | +0.00237049 |
| `answer_verified` | 507 | 512 | +5 |
| `insufficient_evidence` | 484 | 479 | -5 |
| `generator:model_error` | 48 | 47 | -1 |
| `citation_verification_failed` | 424 | 424 | 0 |
| mean agent latency | 18.59 s | 19.45 s | +0.86 s (+4.6%) |

Ở paired comparison, k=40 tốt hơn/kém hơn/hòa lần lượt ở 166/158/667 câu theo
METEOR và 161/160/670 theo ROUGE-L. Vì METEOR chính và ROUGE-L đều tăng, các
reliability guardrail không xấu đi và latency vẫn nằm trong khả năng chạy của
T4, k=40 được chọn làm **development candidate mặc định** cho thí nghiệm quality
kế tiếp. Điều này không thay public submission hoặc artifact; k=20 được giữ làm
control tái lập. Parity cuối cùng với container Codabench vẫn phụ thuộc exact
WordNet bytes/image của BTC, nên report giữ cảnh báo tương ứng.

---

## D095 — Qwen Startup Uses Low-memory Loading Before CUDA Transfer

**Status:** Accepted

M44.4 Kaggle run stall trước câu trả lời đầu tiên khi Qwen đang materialize
weights (`173/434`), CPU/GPU đều gần zero và chưa có batch record. Provider dùng
`accelerate` với `low_cpu_mem_usage=true` để tránh tạo thêm full CPU copy của
model 3B trước khi chuyển sang CUDA. Startup log được tách thành tokenizer,
weights và device-transfer phases.

Đây là runtime-loading safeguard, không thay Qwen identity, number of parameters,
artifact, retrieval, prompt, generation policy hay answer content. `accelerate`
là dependency runtime nhẹ để kích hoạt API chính thức của Transformers; không
phải model, dữ liệu hoặc external service.

---

## D096 — Competition Batch Progress Is a Durable, Foreground Signal

**Status:** Accepted

`legal-rag-batch` retains its default progress interval of 25 durable answers
for ordinary command-line use, but accepts `--progress-interval` for an explicit
operator choice. The Kaggle A/B runbook invokes it directly in the notebook cell
with `PYTHONUNBUFFERED=1`, merged stderr/stdout, and interval `1`.

Each progress message is emitted only after the corresponding result has been
flushed, fsynced and reflected in `batch_state.json`; it is therefore a
checkpoint signal, not an optimistic "started" counter. This changes only
observability. It does not run work in the background, alter inference,
retrieval, model identity, artifacts, answers or recovery identity.

---

## D097 — Citation Coverage Must Be Repaired Before Changing Evidence Policy

**Status:** Accepted

The k=40 end-to-end batch contained 424 `citation_verification_failed` cases.
All 424 stopped fail-closed at verification, while all draft citation identities
were structurally valid: 294 drafts cited one valid evidence item, 81 cited two,
and the remaining 49 cited three to five; none contained an invalid citation.
The dominant claim-level failures were `missing_inline_evidence` and
`no_linked_evidence` (910 each), followed by numeric mismatch (330) and negation
mismatch (145). This is evidence that incomplete claim-to-marker coverage is the
first repair target, not evidence that a citation ID or corpus chunk is invalid.

`ModelBackedAnswerGenerator` must therefore validate the same sentence/list-item
boundaries as the rule-based claim verifier before returning a draft. A legal
claim without a visible inline `[E#]` marker is rejected and may use the existing
single bounded structured-output retry; the system must not fabricate or copy a
marker onto an uncited claim. This keeps the citation verifier fail-closed.

The batch did not preserve context-selection decisions, so it does not justify a
ranking-policy change yet. Every Agent final response now records content-free
context count/budget telemetry, selected evidence identities and the existing
`EvidenceSelectionTrace`. `context_budget_exhausted` is emitted only for an
actual token-budget omission; `context_max_evidence_reached:<n>` distinguishes a
count cap. The next controlled k=40 run must use these traces to determine
whether a context/evidence-selection policy experiment is warranted.

---

## D098 — Batch Outcomes Must Be Analyzed Before a New Evidence-selection Policy

**Status:** Accepted

M45 added content-free context telemetry, but a fresh controlled batch is needed
before interpreting it. M46 therefore adds CPU-only analysis and paired batch
comparison boundaries rather than changing ranking or context selection now. The
tools validate the finished batch manifest and result checksum, reject a
comparison across different official question bytes or ID/order, and persist only
identities, aggregates and per-ID state deltas. Answer content, questions, legal
passages and synthetic labels are deliberately excluded.

This makes the next evidence-selection decision measurable: first classify
citation/abstention transitions and token/count-cap selection reasons on the same
official development split; only then propose a bounded policy experiment. M46
does not load models, use GPUs, alter artifacts, change retrieval/generation, or
make a quality claim.

---

## D099 — Completed Batches Require an Explicit Quality Gate and Paired Score Evidence

**Status:** Accepted

M47 adds two CPU-only decision boundaries without changing the corpus, models,
retrieval, context builder, generator, verifier, artifacts, or answer text.

First, `legal-rag-check-batch` evaluates one complete internal batch against a
typed JSON policy selected by the experiment owner. There are no hidden quality
thresholds: the policy declares allowed retrieval/generator model errors,
citation-verification failures, insufficient-evidence rate, and trace
requirement. The command verifies source bytes, ordered IDs, manifest/result
checksums, persists a content-free report, and exits non-zero if the policy is
violated. A failed gate is evidence to investigate, not permission to silently
rewrite or drop predictions. `legal-rag-submit` requires a passed report and
re-checks its question/record identities before it can produce `submission.zip`.

Second, `legal-rag-compare-scores` performs a paired comparison of two local
warm-up score reports only if their reference checksum, metric mode, scorer
provenance and ordered IDs are identical. It reports per-ID numeric deltas and
aggregate improvements/regressions/ties for exact match, METEOR, and ROUGE-L.
It never exposes answer/reference content and cannot make an
official-equivalence claim that is absent from input reports.

This enforces the sequence: complete batch -> explicit quality gate -> format
submission -> official-compatible local score when references are available ->
paired score comparison -> only then decide whether an experiment is promoted.

---

## D100 — Context Diversity Is an Explicit, Reversible Experiment Variable

**Status:** Accepted

The selected k=40 batch needs a bounded evidence-selection experiment before
changing retrieval or models. M48 introduces only an optional
`max_evidence_per_document` cap after deterministic ranking. The default is
`null`, so existing profiles retain exact behavior. When configured, a later
chunk from an already capped document is classified as `document_cap`; the
builder continues through lower-ranked candidates instead of stopping early.

The first candidate uses cap `2`, while keeping k=40, model revisions,
generation max-context tokens, max evidence, verifier, and corpus artifacts
identical to control. This is not a claim that diversity is intrinsically better:
it may omit useful same-document chunks. It must therefore be evaluated by a
fresh trace-bearing GPU batch, readiness gate, official-compatible score, and
paired comparison before any promotion.

---

## D101 — Model Output Uses Explicit Claim-level Evidence Links

**Status:** Accepted

The M48 development candidate improved local answer metrics but produced 342
`generator:model_error` outcomes. Runtime logs classified the dominant rejection
as `evidence_marker_mismatch`: the model returned answer-level evidence IDs while
placing inline markers at boundaries that did not match the deterministic claim
splitter. Appending all declared markers at the end of an answer could ground only
the final claim and was incompatible with the fail-closed verifier.

M49 replaces that error-prone internal boundary. The model now returns a list of
claim records; every record contains exactly one claim text and the evidence IDs
supporting that specific claim. The generator validates every ID against selected
evidence, rejects model-written markers, rejects a claim record that spans more
than one verifier boundary, and then renders `[E#]` itself from the explicit
claim-level links. A bounded retry receives the exact validation category and a
targeted correction instruction.

This does not weaken citation verification and does not infer or copy an evidence
ID onto an unlinked claim. It changes no corpus, retrieval ranking, context
selection, model identity, model parameters, official data, artifact, or public
answer schema. M49 is an implementation candidate until a small GPU smoke set
shows the failure taxonomy improved; a full development run is required only
after that gate passes.

---

## D102 — M49.1 Uses Compact Bounded JSON and Content-free Truncation Telemetry

**Status:** Accepted

The fixed 50-question M49 smoke set intentionally sampled M48 generator-error
cases. M49 recovered 25 cases, left 25 `generation_failed`, introduced no
retrieval error, and exposed three drafts that the existing fail-closed verifier
correctly rejected. The runtime log contained 50 schema rejections: two failed
structured attempts for every remaining generator error. Rejected completions
were materially slower than accepted completions, while the profile allowed only
256 new tokens, but the old telemetry could not distinguish malformed JSON,
schema validation, or output-limit truncation.

M49.1 removes the verbose generated Pydantic schema from the prompt and replaces
it with compact Vietnamese grounded/abstention examples. The untrusted draft is
bounded to four claims and 600 characters per claim. A long ASCII-only claim is
rejected as non-Vietnamese, and correction feedback distinguishes JSON decoding,
schema validation, language, boundary, marker, and allowlist failures. The local
Transformers provider logs only generated-token count and whether generation hit
the configured output limit; it never logs completion content. The isolated
doc-cap-2 smoke profile raises the output budget from 256 to 384 tokens.

No citation is inferred, copied, repaired, or accepted outside the selected
evidence allowlist. Marker rendering and the existing citation verifier remain
fail closed. M49.1 changes no official data, index, retrieval result, model
identity, parameter inventory, artifact lineage, public response, scorer, or
submission format. Promotion still requires rerunning the same 50-question smoke
set before any full development batch.

---

## D103 — M49.2 Numeric-only Regeneration Is Typed, One-shot, and Fail-closed

**Status:** Accepted

M49.1 fixed generator-format failures on the fixed 50-question smoke but left
two citation failures caused only by `numeric_mismatch`: question IDs `163099`
and `7157`. The same smoke had 0 generator error, 45 verified answers, 3
model-reported abstentions, and 2 numeric-only citation failures; its immutable
bundle checksum is
`5b8ca8a7200b0fc46d163410b2a44d4e120f70e3c33bd2f1fd03395b7e5d254d`.

M49.2 adds an exact-number prompt rule and one optional Agent-only repair. The
closed `numeric_mismatch` signal contains no draft or legal content. It is used
only when every unsupported claim has exactly that error, citations have no
identity failure, and top-level errors are only the corresponding unsupported
claim errors. The Agent then regenerates from the unchanged selected evidence,
query and strategy and verifies once more with separate trace phases. It never
retrieves, reranks, rebuilds context, increments retrieval retry count, edits
text deterministically, or loops. Any repair error, abstention, contract mismatch
or second verification failure is an abstention.

The default config value is `0`; only the M49.2 doc-cap-2 candidate enables
`max_numeric_mismatch_repairs=1`. Repair telemetry is content-free. This changes
no official data, artifact, retrieval, model/revision/parameter inventory,
scorer, submission format, or verifier threshold. It is not a claimed GPU
improvement until the defined smoke gate completes.

---

## D104 — M49.3 Salvages Only Already-supported Claims Before Numeric Regeneration

**Status:** Accepted

M49.2's targeted smoke showed that numeric regeneration can abstain or fail even
when the initial verifier has already marked other claims supported. M49.3 first
constructs a candidate from those exact supported claim texts and their exact
existing evidence IDs/citations. It never edits a number, paraphrases a claim,
copies a citation to a different claim, or creates a citation. The candidate is
verified in the separate `numeric_salvage_verification` phase.

If no claim is supported, salvage is not applicable and the one M49.2 model
regeneration remains available. If a salvage verification succeeds but rejects
the candidate, that same one model regeneration is available. Missing/ambiguous
citation mappings and verifier timeout/error fail closed without model fallback.
No retrieval, reranking, context rebuild, extra Agent retry, model identity,
official data, artifact, scorer, or submission behavior changes. The default
repair limit stays `0`; only the already-approved candidate may set it to `1`.
Persisted repair telemetry and aggregate reports remain content-free.

---

## D105 — M49.4 Classifies Structured Generation and Salvages Only Supported Claims

**Status:** Accepted

M49.3's smoke recovered numeric-only failures but left a negation-only citation
failure and model errors with no persisted category. M49.4 adds a closed,
sanitized structured-generation failure code at the tool boundary. It classifies
only output-contract failures; provider, CUDA, timeout, and unknown runtime
failures must not be relabelled as structured output failures.

M49.4 also generalizes deterministic salvage under a separate default-off limit.
It applies only when claims already marked supported can be retained exactly and
all removed claims are rejected solely for numeric and/or negation mismatch. The
Agent re-verifies the candidate. A negation path never calls the model again.
Missing mappings, timeout/error, or a rejected candidate remain an abstention.
Numeric-only handling retains M49.3's bounded model fallback after its existing
salvage conditions. No corpus, artifact, retrieval, model, scorer, or submission
contract changes. Promotion requires the targeted and fixed 50-question gates.

---

## D106 — M49.5 Recovers Only Structurally Safe Terminal Generator Drafts

**Status:** Accepted

M49.4 left a small number of terminal Pydantic schema failures after its one
model correction attempt. M49.5 preserves the strict `ModelAnswerDraft`
contract and adds a local, default-off recovery only after that correction is
exhausted. The allow-list is limited to removing extra fields, normalizing one
claim object or already-valid scalar evidence ID into a list, deduplicating
identifiers/warnings, and dropping complete excess claims above the existing
claim-count limit. It never changes legal text, numeric or negation tokens,
evidence identity, insufficiency state, model retries, retrieval, context,
models, artifacts, scorer or submission contract.

The recovered draft must pass all existing generator and citation checks. A
missing/invalid semantic value, revalidation failure, timeout or verification
failure remains fail-closed. Persisted telemetry contains only closed schema
issue, repair and outcome codes. Promotion requires the two-ID terminal-schema
gate and the immutable 50-question smoke regression; no 991-case batch may be
started automatically.

---

## D107 — M49.6 Recovers Terminal Missing Required Fields via Bounded Model Correction

**Status:** Accepted (Local Implementation Complete; GPU Validation Pending)

M49.5 targeted validation on Kaggle failed fail-closed on IDs `139655` and `25945`
because both drafts omitted required fields (`missing_required_field`), which
M49.5 correctly refused to invent locally. M49.6 retains this strict local
refusal and introduces one final bounded model correction specifically for
unrecoverable missing-required-field schema failures
(`max_missing_field_corrections`, bounded to [0, 1], default 0).

The correction prompt reuses the base prompt with sanitized Vietnamese guidance
requiring complete strict `ModelAnswerDraft` fields without exposing raw Pydantic
errors or rejected drafts. M49.5 safe local structural repairs take precedence;
only truly unrecoverable missing-field drafts without disqualifying errors
(e.g. JSON decode error, unknown evidence ID, grounding-state mismatch) are
eligible. Telemetry records closed `missing_field_correction` counts and
outcomes (`succeeded` / `failed`). All completions re-enter normal strict
validation and citation verification. Promotion requires the two-ID targeted
GPU gate and the immutable 50-question smoke regression.

---

## D108 — M50 Official-Data QLoRA Fine-Tuning Infrastructure and Screening Ladder

**Status:** Accepted (Local Implementation Complete; GPU Execution Pending)

M50 establishes the infrastructure to fine-tune `Qwen/Qwen2.5-3B-Instruct`
using exclusively official organizer supervision without changing retrieval,
reranking, context selection, or strict verification gates. The 991-record
`development.json` is preserved as a frozen historical development benchmark
and is strictly excluded from gradient updates and intermediate model selection.

The clean 5,617 training pool (`training.json`) is partitioned into a
deterministic three-way split (`sft_train.json` ~4,500, `sft_val.json` ~500,
`screen_holdout.json` ~617) preserving all exact and near-duplicate groups with
seed 2026. `quarantined.json` (392 records) remains permanently excluded. Exact
tokenizer audit determines $L=1536$ truncation ceiling (truncating only 39/5617
assistant answers, or 0.694%), using answer-only ChatML loss masking (`-100` for
all prompt tokens) and dynamic batch padding.

Candidate 1 (M50-C1) defines a conservative QLoRA recipe ($r=8, \alpha=16,
\text{lr}=5\times 10^{-5}$, attention-only projections, 1 epoch, microbatch 2,
accumulation 8). Candidate evaluation uses a staged ladder: (1) training loss on
validation, (2) cheap direct-QA paired screening on `screen_holdout.json` (617
records) with cached BASE outputs and paired bootstrap 95% confidence intervals
for $\Delta\text{METEOR}$ and $\Delta\text{ROUGE-L}$, (3) immutable 50-question
production RAG smoke, and (4) full 991 development benchmark run exactly once
for the final selected candidate.

---

## D109 — M50-C2 Conservative QLoRA Pilot with Generation Health and Semantic Preservation Gates

**Status:** Accepted (Local Infrastructure Complete; Kaggle Pilot Execution Ready)

M50-C1 GPU execution revealed a fundamental dissociation: while teacher-forced
validation loss decreased normally (1.09828) and reference-anchored direct-QA
ROUGE-L showed positive lexical signal (+0.04153 over BASE on 20 paired cases),
free-generation health collapsed (EOS emission dropped from 15/20 to 6/20,
cap-reached rate rose from 5/20 to 14/20, 8-gram repetition ratio $\ge 0.25$
rose from 4/20 to 18/20, mean generated tokens increased from 293.30 to 439.15,
and median generated tokens rose from 262.50 to 512.00). Candidate 1 was
conclusively rejected for promotion.

M50-C2 introduces a conservative, generation-safe pilot recipe and strict
multi-checkpoint health gates:
1. **Architecture & Parameters**: rank $r=4, \alpha=8$, dropout $0.05$, target
   modules $\{q\_proj, v\_proj\}$ yielding exactly 921,600 trainable parameters,
   with exact runtime parameter counting and module allowlists.
2. **Learning Rate & Schedule**: $\text{LR}=10^{-5}$ (5x lower than C1), cosine
   decay, warmup ratio 0.05, microbatch 2, accumulation 8, `paged_adamw_8bit`.
3. **Pilot Step Bound**: capped at `max_optimizer_steps=150` with evaluation and
   checkpoint gates executed at steps [50, 100, 150].
4. **EOS-Preserving SFT Encoding & ChatML Suffix Canonicalization**: `encode_sft_example`
   strictly guarantees that ChatML template trailing whitespace tokens (such as `\n` token ID 198
   after `<|im_end|>` token ID 151645) are canonicalized so that the final supervised label is the actual
   EOS token `151645`. Truncated assistant targets retain the terminal EOS token in the final sequence
   position with unmasked label `eos_token_id`. `validate_sft_dataset_encoding` performs sub-second
   CPU preflight validation across all 5,000 records before GPU model loading.
5. **Strict Holdout Isolation**: `screen_holdout.json` is strictly frozen and
   never touched during training or intermediate probing. Probing operates on
   20 deterministic questions extracted exclusively from `sft_val.json` via
   salted SHA-256 (`m50-c2-val-probe-v1:{question_id}`) with content-level
   SHA verification.
6. **Checkpoint Safety Gate**: checkpoints must satisfy: 0 generation errors,
   `cap_without_eos` $\le \text{BASE} + 1$, `repeat8_high` $\le \text{BASE} + 1$,
   `duplicate_line_high` $\le \text{BASE} + 1$, `eos_emitted` $\ge \text{BASE} - 1$,
   mean length $\le \text{BASE} \times 1.35$, and median length $\le \max(\text{BASE} \times 1.35, \text{BASE} + 64.0)$.
7. **Semantic Preservation Gate**: $\Delta\text{ROUGE-L} \ge -0.01$,
   $\Delta\text{METEOR} \ge -0.01$ (if METEOR available), and at least one metric $> 0$.
8. **Multi-Checkpoint Selection**: eligible checkpoints are ranked by average
   semantic delta $\to$ lowest cap $\to$ lowest repetition $\to$ lowest val loss.
   If zero checkpoints pass both gates, the pilot returns `no_promotable_checkpoint`
   and prevents premature promotion without manual review.
9. **Observability & Durability**: visible logging every 10 steps, arbitrary-hour
   elapsed/ETA formatting (`format_duration`), atomic `progress.json` updates,
   durable JSONL history, recovery from steps 50 and 100, and complete archive
   bundling (`m50-c2-pilot-complete.zip`) with SHA-256 checksum manifests.

---

## D110 — M50-C2 Holdout Rejection, SCREEN617 Consumption, and Milestone 50 Closure

**Status:** Accepted

**Context:**
M50-C2 conservative QLoRA pilot was executed on Kaggle GPU across 150 optimizer steps.
All three checkpoints (50, 100, 150) passed the 20-case VAL probe safety gate (20/20 EOS emissions,
0 cap without EOS) and semantic gate. Step 100 was selected as the pilot winner due to highest
combined free-generation semantic delta (+0.01182).

Following pilot selection, the frozen M50-C2 Step 100 adapter was evaluated once on the entire
immutable 617-question holdout partition (`screen_holdout.json`, SHA256 `a165d4a6fba2e2ec460f856a2a67580607d72648f1012fb6dbd5b779c1eb7367`).
Both BASE and C2 Step 100 completed 617/617 generations with 0 errors.

**Authoritative Findings:**
1. **Generation Health Regressed**:
   - Cap without EOS rose from 4.86% (30/617) under BASE to 6.48% (40/617) under C2 (+1.62% regression);
   - High 8-gram repetition ($\ge 0.25$) rose from 6.32% (39/617) under BASE to 9.08% (56/617) under C2 (+2.76% regression);
   - Terminal EOS emission rate dropped from 95.14% (587/617) under BASE to 93.52% (577/617) under C2 (-1.62% regression).
   - The frozen health gate yielded **FAIL**.
2. **Semantic Signal Reversed Negative**:
   - METEOR mean delta: **-0.002803** (95% paired bootstrap CI `[-0.006577, +0.000909]`, 291 wins / 15 ties / 311 losses);
   - ROUGE-L mean delta: **-0.002594** (95% paired bootstrap CI `[-0.007145, +0.001844]`, within frozen $\ge -0.01$ non-regression tolerance but directionally negative);
   - Combined mean delta: **-0.002698** (95% paired bootstrap CI `[-0.006544, +0.001103]`);
   - The semantic gate yielded **FAIL** because METEOR and Combined mean deltas were $\le 0$.

**Decisions & Invariants:**
1. **Candidate Rejection**: M50-C2 is conclusively **REJECTED**. No fine-tuned model from Milestone 50 is promoted to production.
2. **Holdout Consumption**: `screen_holdout.json` (617 questions) has been **CONSUMED** by the formal evaluation of M50-C2. It is now part of the historical evaluation record and **must not** be treated as an untouched final holdout for future adaptive candidate search, hyperparameter tuning, or checkpoint cherry-picking.
3. **No Unwarranted Adaptive Cherry-Picking**: Evaluating Step 50 or Step 150 against SCREEN617 is explicitly forbidden, as SCREEN617 was reserved solely for the pre-registered pilot winner.
4. **Methodological Invariant**: Teacher-forced validation loss and small generation probes (20 cases) are insufficient proxies for holdout generalization. Free-generation health and untouched holdout evaluation remain mandatory criteria for generator promotion.
5. **Frozen Baseline Preserved**: M49.6 (pretrained `Qwen/Qwen2.5-3B-Instruct` with bounded missing-field recovery) remains the active, frozen reliability baseline for competition answering.
6. **Milestone Closure**: Milestone 50 is officially **CLOSED**. No M50-C3 candidate or M51 milestone is opened without an explicit user decision.

---

## D111 — Phase-A Current-System Census Baseline for Phase B

**Status:** Accepted

**Context:**
The Phase-A Current-System Census executed the full M49.6-style competition pipeline against the canonical 991-question `development.json` engineering benchmark using the official competition scoring contract.

**Decisions & Invariants:**
1. **Benchmark Nature**: The 991-question development set (`development.json`, SHA256 `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`) is a historical engineering benchmark, not an untouched generalization holdout.
2. **Current-System Quality Baseline**:
   - Exact Match: `0.0`
   - METEOR: `0.0980790959`
   - ROUGE-L: `0.1871225729`
3. **Current-System Operational Baseline**:
   - 991 records evaluated; 806 `answer_verified` (81.33%), 177 `generation_failed` (17.86%), 7 `citation_verification_failed` (0.71%), 1 `max_retry_reached` (0.10%), 185 `insufficient_evidence` (18.67%), 10 generator model errors.
4. **Phase B Reference**: These metrics serve as the primary current-system reference point for Phase-B improvements.

---

## D112 — ToolError Correction-Chain Diagnostic Envelope Contract

**Status:** Accepted

**Context:**
During batch execution, when an initial structured generation draft failed with missing fields and an attempted missing-field correction subsequently failed with a terminal non-schema error (such as `JSON_DECODE_ERROR`), preserving earlier schema diagnostics caused the `ToolError` model validator to reject the error envelope, terminating the entire batch.

**Decisions & Invariants:**
1. **Separation of Schema and Correction Detail**: `ToolError` distinguishes schema details (`generation_schema_issue_codes`, `generation_schema_repair_codes`, `generation_schema_recovery_outcome`) from correction details (`generation_missing_field_correction_attempted`, `generation_missing_field_correction_outcome`).
2. **Correction-Chain Preservation**: A terminal non-schema generation failure is permitted to carry earlier schema issue codes if and only if `generation_missing_field_correction_attempted = True`.
3. **Rejection of Free-Floating Schema Detail**: Free-floating schema diagnostics attached to a non-schema terminal failure without an attempted correction chain remain strictly forbidden.
4. **Reliability Invariant**: Individual question generation failures must yield durable `generation_failed` records within the batch without crashing the overall evaluation process.

---

## D113 — UIT Competition Graph Integration Architectural Assessment

**Status:** Accepted

**Context:**
Static architecture auditing and empirical census analysis on the 991-question benchmark revealed:
1. The official UIT DSC 2026 data contract lacks relationship annotations; `relationships.jsonl` has `record_count = 0` by design.
2. The competition graph artifact contains 8,532 nodes and 0 edges, and build validation strictly asserts `record_count == 0`.
3. `QueryUnderstandingService` assigns `QueryIntent.RELATIONSHIP` top priority on any query matching 8 substring cues (`"sửa đổi"`, `"bổ sung"`, `"thay thế"`, `"bãi bỏ"`, `"hướng dẫn"`, `"dẫn chiếu"`, `"còn hiệu lực"`, `"hết hiệu lực"`).
4. `DeterministicStrategyRouter` prepends `[GRAPH, HYBRID_RERANK, HYBRID]` when `adaptive_routing_enabled = True`.
5. On zero-edge graphs, `graph_search` requests hybrid seeds capped by `min(graph_seed_chunk_k, candidate_k - 1) = 20`. Traversal finds 0 edges, so the cross-encoder evaluates only 20 candidates instead of the configured 40 in `hybrid_rerank`.
6. Empirical census showed 22/22 relationship-matching questions executed `GRAPH_SEARCH` and terminated on Attempt 1 without falling back to `hybrid_rerank`.

**Decisions & Invariants:**
1. **Generic Graph Library Retained**: `GraphBackend`, `AdjacencyGraphBackend`, and `GraphExpandedRetriever` remain part of the generic library codebase (`KEEP_GENERIC_ONLY`).
2. **Competition Path Deletion Candidate**: The UIT competition-specific graph integration is designated as a candidate for removal (`REMOVE_COMPETITION_PATH`), but will NOT be removed during Phase A.
3. **Mandatory Paired Ablation**: Structural modification or deletion of the graph path is deferred until the Phase B1A paired counterfactual ablation explicitly measures the causal delta on the 22 affected cases.

---

## D114 — Non-Causal Interpretation of Observational Subgroup Metrics

**Status:** Accepted

**Context:**
In the 991-question census, the graph-routed subgroup achieved an 81.82% `answer_verified` rate (18/22) compared to 81.32% (788/969) for the non-graph group.

**Decisions & Invariants:**
1. **Observational vs Causal**: The graph and non-graph subgroups evaluated entirely distinct legal questions with different complexities, references, and vocabularies.
2. **Prohibition of Causal Quality Inferences**: The repository strictly forbids claiming that graph retrieval is superior or inferior based on observational subgroup averages.
3. **Controlled Counterfactual Standard**: All future routing and retrieval architectural evaluations must use paired counterfactual comparisons where the exact same questions are evaluated under candidate versus control pipelines.

---

## D115 — Phase B1A.2 Graph Redundancy and Candidate Pool Isolation Verdict

**Status:** Accepted

**Context:**
Phase B1A.2 evaluated the 22 canonical relationship questions under three isolated arms:
- **ARM G**: Current graph path (branch depth 40 -> RRF top 20 -> zero-edge graph traversal -> cross-encoder rerank 20 -> final top 8).
- **ARM S20**: Seed-equivalent direct path (branch depth 40 -> RRF top 20 -> NO graph -> cross-encoder rerank 20 -> final top 8).
- **ARM H40**: Diagnostic standard hybrid-rerank path (branch depth 40 -> RRF top 40 -> cross-encoder rerank 40 -> final top 8).

Canonical run archive `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a` confirmed:
1. G vs S20: 22/22 seed match (100.0%), 22/22 final top-8 match (100.0%), 22/22 score tolerance passes ($\le 10^{-6}$).
2. Traversal verification: 0 graph records, 0 edges, 0 yielded steps, 1 traversal call/case.
3. Diagnostic H40: 17/22 cases differed in top-8 results (mean overlap 6.4091/8).
4. Official verdict: **`GRAPH_REDUNDANCY_PROVEN`**.

---

## D116 — Phase B1B Structural Competition Graph Removal and S20 Preservation

**Status:** Accepted (Verified via `B1B_EQUIVALENCE_PASS`)

**Context:**
Authorized by verdict `GRAPH_REDUNDANCY_PROVEN` from Phase B1A.2, Phase B1B removes the zero-edge graph traversal mechanism from the UIT DSC competition online path and offline build while preserving exact S20 retrieval behavior.
Post-change equivalence verification on Kaggle (reviewed commit `38a6feec8867a41454c453cce9c54b162801579e`, canonical evidence archive `phase-b1b-graphless-equivalence-evidence.zip` SHA-256 `f392cc650699ecc562cb43ea0ea7f6e965455a36a621843ec6a882172913c9c3`, size 14,157 bytes) achieved mechanical verdict **`B1B_EQUIVALENCE_PASS`** (`b1b_verified = true`).

**Decisions & Invariants:**
1. **Tool Surface**: `ToolName.GRAPH_SEARCH` is removed from active online agent capabilities; `ToolName.RELATIONSHIP_RERANK_SEARCH` (`"relationship_rerank_search"`) is introduced.
2. **Strategy Preservation**: `relationship_rerank_search` emits `RetrievalStrategy.HYBRID_RERANK`. No new public retrieval strategy enum is created.
3. **Candidate Pool Isolation (S20)**: `RelationshipSeedRerankingRetriever` executes branch candidate depth 40, hybrid fusion limit $\le 20$, cross-encoder rerank limit $\le 20$, final top 8.
4. **Adaptive Relationship Routing**: Query intent `RELATIONSHIP` plans:
   - Attempt 1: `(HYBRID_RERANK, relationship_rerank_search)` (S20)
   - Attempt 2: `(HYBRID_RERANK, rerank_search)` (H40)
   - Attempt 3: `(HYBRID, hybrid_search)`
5. **Online Runtime Artifact Set**: Exactly 3 active artifacts (`legal_chunks`, `bm25_index`, `vector_index`). Startup does not require or validate `graph/` or `relationships/`.
6. **Offline Competition Build**: Produces exactly 6 artifacts (`normalized_documents`, `cleaned_documents`, `legal_blocks`, `legal_chunks`, `bm25_index`, `vector_index`).
7. **Generic Graph Infrastructure (`KEEP_GENERIC_ONLY`)**: Generic graph contracts, implementations, and tests remain intact outside the competition path.
8. **Verification Invariants**: 22/22 exact matches, chunk sequence matches, document sequence matches, score tolerance passes ($\le 10^{-6}$), branch depth 40 passes, candidate query passes, fusion limit passes ($\le 20$), final top-k passes ($\le 8$), and route plan passes confirmed against frozen B1A.2 baseline `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a`. H40 remains a distinct non-promoted path.

---

## D117 — Stage R1 Candidate-Pool / Reranker Mechanics Characterization and Non-Promotion of H40

**Status:** Accepted (Verified via `CANDIDATE_POOL_AUDIT_PASS`)

**Context:**
Following Phase B1B, Stage R1 audited the candidate-pool depth effects (fused 20 vs fused 40) under identical single-pass query understanding, branch retrieval (BM25 + dense depth 40), RRF fusion, and single-pass cross-encoder reranker scoring on the 22 canonical relationship queries.
Kaggle execution (reviewed commit `9a5b708c2769425dbd65731feb8ede96975b5b46`, canonical evidence archive `candidate-pool-reranker-audit-evidence.zip` SHA-256 `ce9b239b808c3d7b0e575ce1c1683db243bbea909f0e6d9c306df21cb2899860`, size 56,706 bytes) achieved mechanical verdict **`CANDIDATE_POOL_AUDIT_PASS`** (`audit_verified = true`, `h40_promotion_authorized = false`).

**Decisions & Invariants:**
1. **Candidate-Pool Depth is Behaviorally Material**: Expanding the fused candidate pool from 20 to 40 changes the final top-8 evidence in 17/22 relationship cases, introducing 35 tail entrants and 20 document-level churn events.
2. **Prohibition of Unsubstantiated H40 Promotion**: The official competition dataset lacks retrieval ground-truth labels. The presence of tail entrants does not prove that H40 is semantically superior or legally correct. Therefore, H40 remains strictly unpromoted (`h40_promotion_authorized = false`).
3. **Production S20 Routing Preserved**: Production Attempt 1 routing remains strictly bound to `relationship_rerank_search` (S20). Attempt 2 remains `rerank_search` (H40).
4. **Verification Fidelity**: 22/22 seed prefix passes, 22/22 shared S20 sequence passes, 22/22 legacy S20 frozen score passes ($\le 10^{-6}$), 22/22 H40 frozen score passes ($\le 10^{-6}$), 22/22 branch depth passes, and 0 retrieval model errors verified.
5. **Next Active Research Frontier**: The active development frontier advances to **Priority B — Verification-correctness audit**.

---

## D118 — Controlled V0 vs V1 Semantic-Verifier Benchmark Verdict, Non-Promotion Decision, and 38-Claim Development Status

**Status:** Accepted (Verdict: `VERIFIER_BENCHMARK_PASS`, Decision: `V1_EXISTING_SEMANTIC_VERIFIER_NOT_PROMOTED`)

**Context:**
Milestone 51 Priority B conducted the first controlled offline benchmark comparing the deterministic `RuleBasedCitationVerifier` (V0) against the pre-existing `ModelBackedCitationVerifier` (V1, `Qwen/Qwen2.5-3B-Instruct`) over the frozen 38-claim composite human-labeled dataset (18 `SUPPORTED`, 7 `CONTRADICTED`, 13 `INSUFFICIENT`).
The benchmark was executed on Kaggle GPU under execution commit `d3aac626400cbe31ed0ed5ad109762fcb78d737d` with evidence archived in `verification-semantic-benchmark-evidence.zip` (SHA-256 `bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6`, size 17,290 bytes, 8 members).

**Empirical Summary:**
1. **Mechanical Execution**: `VERIFIER_BENCHMARK_PASS` (0 model errors, 2 structured output retries, 38/38 stable claims across 2 passes, 22/22 exact V0 replay passes).
2. **Claim-Level Binary**: V1 achieved 60.53% accuracy (vs V0 47.37%), 88.89% supported retention (16/18), 35.0% negative catch rate (7/20), precision 55.17%, F1 0.6809, balanced accuracy 61.94%.
3. **Paired Deltas**: 16 both correct, 2 V0-only correct (regressions), 7 V1-only correct (fixes), 13 both wrong; net correctness delta $+5$ claims.
4. **Answer-Level**: V1 achieved 63.64% accuracy (vs V0 31.82%), 100.0% valid answer retention (7/7), 46.67% invalid answer catch rate (7/15).
5. **Diagnostic Failure Modes**: V1 exhibited 80% catch on actor role inversions and 55.6% on wrong documents, but 0% catch on condition omissions/inversions, wrong articles, and quantity errors, and only 12.5% on scope overgeneralizations. Three-way `CONTRADICTED` recall was 14.29% (1/7) and `INSUFFICIENT` recall was 15.38% (2/13).

**Decisions & Invariants:**
1. **Formal Non-Promotion of Existing V1**: `V1_EXISTING_SEMANTIC_VERIFIER_NOT_PROMOTED`. The existing V1 implementation remains strictly unpromoted in production (`semantic_verifier_promotion_authorized = false`).
2. **Production Baseline Preserved**: `RuleBasedCitationVerifier` (V0) remains the active verification engine in the production answering pipeline.
3. **38-Claim Benchmark Burned as Development Data**: The 38-claim composite dataset is permanently transitioned to `verification_benchmark_v1_role = "development_after_first_evaluation"`. It may be used for diagnostic forensic analysis and verifier-development evaluation under its frozen usage policy. Its human labels remain prohibited for training, fine-tuning, retrieval relevance supervision, public/private test annotation, and manual submission correction unless a separate explicit governance decision changes that policy. It MUST NOT serve as final promotion evidence for any future tuned V2 verifier.
4. **Prohibition of Repurposing Control Reserves**: The 8 positive-control reserve cases (`27503`, `31317`, `33177`, `85651`, `112105`, `112833`, `130283`, `137453`) cannot be repurposed as a secret promotion holdout.
5. **Mandatory Fresh Holdout Pre-Registration**: Any future V2 semantic verifier evaluation requires a fresh, independently sampled holdout drawn from the $\approx 764$ untouched Phase-A records with explicit exclusion of all 28 previously audited QIDs.

---

## D119 — Fresh V2 Verification Holdout Pre-Registration and Sealed Protocol

**Status:** Accepted (Verdict: `V2_HOLDOUT_PRE_REGISTERED`)

**Context:**
Following the closure of the V1 benchmark and the classification of the 38 human-reviewed claims as development data, a fresh, independent holdout dataset has been pre-registered before any V2 tuning, prompt engineering, or candidate modeling.
The holdout was selected deterministically from frozen Phase-A census records under pre-registered salt `verification-v2-holdout-gen-v1:`, algorithm `deterministic_sha256_stratified_v2`, and strict exclusion of all 46 previously audited/control QIDs (SHA-256 `eefdd8967c39324bc7e88a8451ef8fb9241f765af1e68a0199db9ba33af01fda`).
The selection commitment is archived in `verification-v2-holdout-selection-v1.json` (SHA-256 `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b`, size 16,788 bytes) and review packets in `verification-v2-holdout-review-packets-v1.zip` (SHA-256 `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4`, size 108,532 bytes).

**Decisions & Invariants:**
1. **Mechanical Verdict**: `V2_HOLDOUT_PRE_REGISTERED` (`16` PRIMARY across 4 strata 4/4/4/4, `8` FRESH RESERVE across 4 strata 2/2/2/2, 16/16 chunk lookup passes, 16/16 trace mapping passes, 16/16 metadata cross-checks, 16/16 V0 RuleBasedCitationVerifier replay passes).
2. **Holdout Blindness & Sealing Invariant**: Zero selected question IDs or raw prompts are recorded in tracked documentation, console summaries, or agent communication. The holdout review packet archive remains strictly sealed and unreviewed (`review_status = "sealed_unreviewed"`, `claim_labels = null`).
3. **Strict Development vs Holdout Separation**: All future V2 development, tuning, and prompt engineering must use ONLY the 38-claim development benchmark (`verification_benchmark_v1_role = "development_after_first_evaluation"`).
4. **Mandatory Freezing Before Unsealing**: The fresh holdout MUST NOT be unsealed for human forensic review until the candidate V2 system (code, prompt, model, inference parameters, promotion gates) is completely frozen.
5. **Reserve Replacement Invariant**: The 8 fresh reserve cases may only be used for mechanical reconstruction/provenance failure, never for undesirable human labels or performance optimization.

---

## D120 — Candidate V2-D3.1 Benchmark Outcome, High-Recall Contradiction Overcalling, and Formal Decision: `KEEP_D3`

**Status:** Accepted (Verdict: `V2_D31_DEVELOPMENT_BENCHMARK_PASS`, Decision: `KEEP_D3`)

**Context:**
Candidate V2-D3.1 (monolithic two-gate verifier: Gate 1 contradiction check $\to$ Gate 2 support/insufficient check) was evaluated on Kaggle GPU under execution commit `1383bf379a01c3f7456e3c41ba3be42846ceee2c`.
Evidence was archived in `verification-v2-d31-development-evidence.zip` (SHA-256 `e14f9656a13a04b8e545d88a5dca13653fa317166ff530f45e4b13124f864041`, size `18,379` bytes, 11 members).

**Empirical Summary & Findings:**
1. **Mechanical Execution**: `V2_D31_DEVELOPMENT_BENCHMARK_PASS` (0 model errors, 0 retries, 38/38 stable claims, 76 provider calls).
2. **Behavioral Characterization**: D3.1 operated as a high-recall / low-precision contradiction detector. It caught 6/7 gold contradictions (85.71% recall), but emitted 14 false contradiction positives (10 on gold `INSUFFICIENT`, 4 on gold `SUPPORTED`), yielding a contradiction precision of only 30.00%.
3. **Severe Supported Retention Regression**: Supported claim retention dropped from 17/18 (94.44% in D3) down to 12/18 (66.67%), and 6 of 7 historical D3 fixes regressed.
4. **Decision**: `KEEP_D3`. Candidate V2-D3.1 does not supersede D3 (`d31_supersedes_d3 = false`, `promotion_authorized = false`).

---

## D121 — Candidate V2-D3.2 Benchmark Outcome, Strict Conflict Calibration, and Formal Decision: `KEEP_D3`

**Status:** Accepted (Verdict: `V2_D32_DEVELOPMENT_BENCHMARK_PASS`, Decision: `KEEP_D3`)

**Context:**
Candidate V2-D3.2 (asymmetric two-stage verifier: Call A Frozen D3 Base + Call B Strict Contradiction Confirmation Overlay) was evaluated on Kaggle GPU under execution commit `e5db78f0796c53e973fc63f9dd98df6c95f43f6e`.
Evidence was archived in `verification-v2-d32-development-evidence.zip` (SHA-256 `bf44b9d77172d4f1823b62c02abae9e462bfbb9fdc5c650ba87e192e4928878f`, size `28,738` bytes, 13 members).

**Empirical Summary & Findings:**
1. **Mechanical Execution**: `V2_D32_DEVELOPMENT_BENCHMARK_PASS` (0 model errors, 0 retries, 38/38 stable claims, 152 provider calls reconciled across 2 passes).
2. **Base D3 Fidelity**: 0 drift across Pass 1 (38/38) and Pass 2 (38/38).
3. **Zero False Overrides**: The strict conflict overlay emitted 0 conflict positives on the 38 development claims, producing 0 false overrides and preserving 100% of D3's high supported retention (17/18) and all 7/7 historical D3 fixes.
4. **Contradiction Sensitivity**: Under the strict two-gate formulation (`same_material_proposition = true` AND `cannot_both_be_true = true`), the overlay caught 0 contradictions, achieving Net Delta = 0 vs D3 while doubling inference calls.
5. **Decision**: `KEEP_D3`. Candidate V2-D3.2 does not supersede D3 (`d32_supersedes_d3 = false`, `promotion_authorized = false`). Candidate V2-D3.2 is formally closed.

---

## D122 — Formal Freezing of Candidate V2-D3, Closure of Development Phase, and Pre-Registration of Fresh Holdout Evaluation Protocol

**Status:** Accepted (Protocol: `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`)

**Context:**
Following the completion and formal closure of development iterations V2-D3.1 and V2-D3.2, candidate V2-D3 is officially selected and frozen as the exclusive V2 semantic verification candidate to proceed to the Fresh Holdout evaluation.

**Decisions & Invariants:**
1. **Development Benchmark Permanently Closed**: The 38-claim development benchmark is permanently closed for candidate tuning, prompt engineering, threshold tuning, and overlay creation. There is NO D3.3.
2. **Frozen Candidate Identity (V2-D3)**:
   - Model: `Qwen/Qwen2.5-3B-Instruct` (Immutable Revision: `a1d308dfcc03e09da285d49d912439a655a571e8`).
   - Implementation SHA-256: `a6e8bca15ad14d869e103e1f94fe94bb9a81f9ddc8bc650b280b69b7d57e9826`.
   - System Instruction SHA-256: `546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b`.
   - Runtime: Transformers 4.47.1, CUDA, float16, temperature 0.0, max retries 1.
3. **Pre-Registered Promotion Rate Gates (Pass 1 Authoritative)**:
   - `min_supported_retention_rate = 0.88` (88.00%)
   - `min_negative_catch_rate = 0.50` (50.00%)
   - `min_valid_answer_retention_rate = 0.80` (80.00%)
   - `min_full_answer_accuracy_rate = 0.60` (60.00%)
   - `min_claim_binary_accuracy_rate = 0.70` (70.00%)
4. **Pre-Registered Mechanical Gates**: Zero model errors, zero execution errors, zero unstable claims between Pass 1 and Pass 2, complete 2-label verification per claim, exact frozen source SHA matches.
5. **Strict Holdout Blindness**: Fresh holdout dataset (`verification-v2-holdout-selection-v1.json` SHA-256 `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b`, `verification-v2-holdout-review-packets-v1.zip` SHA-256 `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4`) remains sealed and unreviewed prior to Phase H-LABEL authorization.
6. **Promotion vs Authorization Boundary (Fail-Closed)**: Harness outputs `promotion_recommended: true` only upon satisfying all gates. Harness output `promotion_authorized: false` is an invariant fail-closed security boundary. Enabling the semantic verifier in production requires explicit external human governance sign-off and subsequent production codebase configuration.

---

## D123 — Fresh Holdout Evaluation Lifecycle, Label-Freeze Commitment, and Non-Vacuous Coverage Gates

**Status:** Accepted (Protocol: `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`)

**Context:**
Following the formal freeze of candidate V2-D3 and closure of the 38-claim development benchmark, the holdout evaluation governance and harness contract was hardened to define a two-phase irreversible lifecycle and non-vacuous gate semantics.

**Decisions & Invariants:**
1. **Two-Phase Holdout Lifecycle**:
   - **Phase H-LABEL (Human Gold-Label Freezing)**: Candidate V2-D3, prompt SHA, implementation SHA, and rate thresholds remain frozen. Human reviewers unseal ONLY the 16 primary review packets to assign gold labels (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`). Model predictions MUST NOT exist or be visible during review. Labels are frozen using `scripts/freeze_verification_v2_holdout_labels.py`, producing `verification-v2-holdout-reviewed-labels-v1.json` and content-free commitment `configs/verification-v2-d3-holdout-label-commitment.json`.
   - **Phase H-EXEC (Canonical Model Execution)**: Executed once on Kaggle GPU ONLY after the label commitment SHA is externally reviewed and frozen. Requires exact SHA match across packets, selection, and labels against the commitment.
2. **Scientifically Correct Blindness Invariants**:
   - Before Phase H-LABEL authorization: Zero holdout inspection.
   - During Phase H-LABEL: Human inspection permitted solely to establish independent gold labels without D3 outputs.
   - After gold labels are frozen: Zero label edits.
   - During/After Phase H-EXEC: Zero candidate tuning, zero prompt edits, zero threshold edits, zero label edits, and zero reruns to improve metrics.
3. **Non-Vacuous Coverage Denominator Gates**:
   - Promotion eligibility requires: `gold_supported_claims > 0`, `gold_negative_claims > 0`, `gold_valid_answers > 0`, and `gold_invalid_answers > 0`.
   - If any denominator is zero: `verdict = "V2_D3_HOLDOUT_COVERAGE_INSUFFICIENT"`, `promotion_recommended = False`, `promotion_authorized = False`.
   - Zero-denominator rates report `null` / `None` (never `1.0`).
4. **Strict Fail-Closed Label Loading**: Zero fallbacks permitted. Exact packet-label claim set equality `(question_id, arm_id, claim_id)` is enforced before model initialization. Missing, extra, duplicate, or invalid labels fail closed immediately with `DataValidationError`.
5. **Single Model Loading & Pinned Kaggle Environment**:
   - Kaggle environment uses pinned `transformers==4.47.1 tokenizers==0.21.4 huggingface-hub==0.27.1 accelerate==1.2.1` and `python -m pip install -q -e . --no-deps`.
   - Cell H4 verifies tokenizer/access only without retaining a Qwen model object in notebook memory. The model is loaded exclusively by the H5 evaluation subprocess.

---

## D124 — Pre-H-LABEL Integrity Hardening, Governance Status Lifecycle, Content-Safe Telemetry, and Fail-Closed Verification Gates

**Status:** Accepted (Harness: `scripts/freeze_verification_v2_holdout_labels.py`, `scripts/evaluate_verification_v2_d3_holdout.py`, Protocol: `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`)

**Context:**
Prior to initiating Phase H-LABEL human review, a final pre-holdout integrity hardening pass was conducted across the holdout freezing harness, evaluation harness, and runbook protocol to eliminate all subtle fail-open edge cases, enforce strict governance state transitions, prevent secret leakage in telemetry, and harden verification assertions.

**Decisions & Invariants:**
1. **Frozen V2-D3 Candidate Byte-Identity Preserved**:
   - Implementation SHA-256: `a6e8bca15ad14d869e103e1f94fe94bb9a81f9ddc8bc650b280b69b7d57e9826`
   - System Instruction SHA-256: `546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b`
   - Schema SHA-256: `3591144a40b0519d5da9dd262e8edf8814531d798b69deea94fd81fae39f5f61` (sorted `D3StructuredClaimAssessmentDraft.model_json_schema()`)
   - Pre-registered rate thresholds and quality gates remain completely unchanged.
2. **Fail-Closed Duplicate Human Review Detection**:
   - `scripts/freeze_verification_v2_holdout_labels.py` uses a custom JSON object pairs hook (`_reject_duplicate_json_keys`) to detect and reject duplicate keys at parse time.
   - The label freezer enforces unique `(question_id, arm_id, claim_id)` tuples across all input structures, rejecting duplicate review entries fail-closed (`HOLD_OUT_LABEL_DUPLICATE`). Never silently overwrite.
3. **Two-Stage Governance Status Lifecycle**:
   - Label freezing initializes commitment status to `FROZEN_PENDING_EXTERNAL_REVIEW`. The script does not self-authorize execution.
   - Canonical H-EXEC strictly requires `--label-commitment` with `reviewer_governance_status: "EXTERNALLY_REVIEWED_FOR_H_EXEC"`. Direct bypass via raw `--holdout-labels-sha256` without an approved commitment is blocked in canonical execution.
4. **Label Artifact Metadata & Claim SHA Enforcement**:
   - Label loading validates `artifact_type == "verification_v2_holdout_reviewed_labels"`, `review_status == "frozen_human_reviewed"`, total claim counts, and class count sums.
   - Every claim in the label artifact must contain `claim_text_sha256` matching the exact SHA-256 of the packet claim text.
5. **Exact Prediction-Set Equality & Fail-Closed Stability**:
   - Stability evaluation requires exact set equality: $\text{ExpectedClaimKeys} \equiv \text{Pass1Keys} \equiv \text{Pass2Keys}$. Exactly one prediction per claim per pass.
   - Both passes must produce valid 3-way semantic predictions (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`). Any missing, duplicate, extra, or invalid prediction is marked as an execution failure, eliminating the `None == None -> stable` fail-open vulnerability.
6. **Content-Safe Error Telemetry**:
   - Raw exception strings (`str(exc)`) and unmasked stack traces are removed from call history and telemetry. Replaced by `error_type`, `error_sha256`, and `error_message_length` to prevent accidental secret or prompt leakage in logs.
7. **Provider Call Reconciliation Gate**:
   - Provider call row count in `telemetry/provider_calls.jsonl` must reconcile with decision report: $\text{total\_provider\_calls} == 2 \times N_{\text{claims}} + \text{total\_structured\_retries}$. Every call must match the frozen system instruction SHA.
8. **Hardened Canonical Provenance Validation**:
   - `_validate_canonical_provenance()` enforces fail-closed checks on candidate ID (`V2-D3`), source package version (`0.50.7`), installed package version (`0.50.7`), clean git worktree, `repeat_count == 2`, `device == "cuda"`, `torch_dtype == "float16"`, `temperature == 0.0`, token bounds (`8192/512`), `max_retries == 1`, `timeout == 180.0`, backend (`transformers`), provider version (`4.47.1`), model name, immutable revision, and exact D3 implementation, system instruction, and schema SHA-256 digests.
9. **Independent Recomputation in Cell H6**:
   - Runbook Cell H6 recomputes all coverage booleans, mechanical pass criteria, provider call reconciliation, and quality rate gates directly from evidence metrics, asserting exact match against decision reports and verifying `promotion_authorized == False`.

---

## D125 — Phase H-LABEL Completion, Human Gold-Label Freeze, and External Chain-of-Custody Approval for Phase H-EXEC

**Status:** Accepted (Artifacts: `verification-v2-holdout-reviewed-labels-v1.json`, `configs/verification-v2-d3-holdout-label-commitment.json`, Protocol: `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`)

**Context:**
Phase H-LABEL human review and gold-label freezing was executed across the 16 primary holdout review packets. Following human label entry and freeze verification, an external chain-of-custody review was conducted on the content-free label commitment, transitioning the governance state to authorize the canonical one-shot Phase H-EXEC run on Kaggle GPU.

**Decisions & Invariants:**
1. **Holdout Unsealing & Neutral Human Review Workflow**:
   - The 16 primary review packets from canonical archive `verification-v2-holdout-review-packets-v1.zip` (SHA-256 `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4`) and selection binding `verification-v2-holdout-selection-v1.json` (SHA-256 `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b`) were unsealed strictly for neutral human forensic review.
   - Zero model predictions, zero prior verifier suggestions, and zero automatic labels were produced or displayed.
2. **Authoritative Human Gold Labels Frozen**:
   - The human reviewer assigned gold entailment labels across all 31 claims (16 questions, 16 arms).
   - Label distribution: **24 SUPPORTED, 1 CONTRADICTED, 6 INSUFFICIENT**.
   - Every claim is bound to its canonical `claim_text_sha256`.
   - Immutable label artifact created: `verification-v2-holdout-reviewed-labels-v1.json` (SHA-256 `85d348dbb7da1567398836b96156a9d08fcfe181b676c5ecd593535ec8904215`, size `9,383` bytes, `review_status: "frozen_human_reviewed"`).
3. **Chain of Custody & Approved Commitment**:
   - Historical pending commitment: `verification-v2-d3-holdout-label-commitment.json` (SHA-256 `c7755e37e394e80484f73c52ee6965c34c65917c38fa83b1dc453bbb466bcf86`, size `823` bytes, `reviewer_governance_status: "FROZEN_PENDING_EXTERNAL_REVIEW"`).
   - External chain-of-custody review approved: Created approved commitment `configs/verification-v2-d3-holdout-label-commitment.json` (SHA-256 `5cc7f58ed52c43d091bce98d5296ab68cded981495736a479644aefc1428b6dc`, size `1,060` bytes, `reviewer_governance_status: "EXTERNALLY_REVIEWED_FOR_H_EXEC"`).
   - Provenance fields link `prior_pending_commitment_sha256`, `external_review_scope: "content_free_chain_of_custody_review"`, and timestamp.
4. **Scope of Authorization**:
   - Authorization is granted exclusively for the pre-registered one-shot Phase H-EXEC Kaggle execution.
   - Candidate V2-D3 remains strictly frozen (`a6e8bca1...`, prompt `546cd8bd...`, schema `3591144a...`, `Qwen/Qwen2.5-3B-Instruct` rev `a1d308df...`).
   - Rate thresholds remain strictly frozen (`supp_ret >= 0.88`, `neg_catch >= 0.50`, `val_ans_ret >= 0.80`, `full_ans_acc >= 0.60`, `claim_bin_acc >= 0.70`).
   - Zero post-hoc modifications, candidate edits, prompt edits, threshold edits, label edits, or rerun loops permitted.

---

## D126 — Phase H-EXEC Attempt 0 Invalidation, Pre-Inference Provider Harness Correction, Preflight Constructor Smoke Gate, and Recovery Attempt 1 Authorization

**Status:** Accepted (Harness: `scripts/evaluate_verification_v2_d3_holdout.py`, Tests: `tests/unit/evaluation/test_verification_v2_d3_holdout.py`, Protocol: `docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md`)

**Context:**
The initial canonical H-EXEC execution attempt on Kaggle GPU (`21b7ffcf10d4621b0fdcbf18dcd565e4d5186699`) encountered a mechanical pre-inference harness defect during provider initialization (`TypeError: TransformersChatProvider.__init__() got an unexpected keyword argument 'model_name'`). An emergency pre-inference harness correction was conducted to align the holdout evaluation harness with the proven development provider construction architecture.

**Decisions & Invariants:**
1. **Attempt 0 Invalidation & Classification**:
   - Attempt 0 failed synchronously at `V2D3HoldoutBenchmarkEvaluator._init_v3_provider()` before any provider object was instantiated, before model weights were loaded, and before Pass 1 started.
   - Telemetry count: 0 provider calls, 0 model weights loaded, 0 D3 predictions produced, 0 holdout performance metrics generated.
   - Formally classified as `H_EXEC_ATTEMPT_0_INVALID_PRE_INFERENCE_HARNESS_FAILURE`. Zero scientific holdout results were consumed.
2. **Root Cause & Construction Port**:
   - `scripts/evaluate_verification_v2_d3_holdout.py` incorrectly passed keyword arguments directly to `TransformersChatProvider.__init__()` rather than wrapping them in `GenerationConfig`.
   - Repaired by porting the proven, validated provider construction logic from `scripts/evaluate_verification_v2_d3_development.py`:
     `cfg = SemanticVerificationConfig(...); generation_cfg = cfg.as_generation_config(); return TransformersChatProvider(generation_cfg)`
   - Enforced fail-closed assertions on `GenerationConfig` canonical constants (`transformers`, `Qwen/Qwen2.5-3B-Instruct`, `a1d308df...`, `cuda`, `float16`, `local_files_only=False`, `timeout=180.0`, `8192/512`, `retries=1`, `temperature=0.0`).
3. **Model-Free Provider Constructor Preflight Smoke Gate**:
   - Updated the `--preflight-only` path to execute a model-free provider-constructor smoke verification check before declaring readiness.
   - Validates that `_init_v3_provider()` and `_validate_runtime_provider_identity()` succeed without calling `complete()`, without calling `_require_runtime()`, and without loading model weights.
   - Records `provider_constructor_contract_verified: True` in preflight output.
4. **Regression Testing**:
   - Added unit regression tests (`test_init_v3_provider_real_construction_contract`, `test_init_v3_provider_invalid_generation_config_fails_closed`, `test_preflight_verifies_provider_constructor_contract`) with monkeypatched `_load_runtime` guards to verify lazy loading and eliminate fail-open regressions.
5. **Frozen Invariants Unchanged**:
   - Core provider source (`src/legal_agentic_rag/generation/transformers_provider.py`) is completely unchanged.
   - Candidate V2-D3 implementation (`a6e8bca1...`), prompt (`546cd8bd...`), and schema (`3591144a...`) remain 100% byte-identical.
   - Frozen human gold labels (`85d348db...`, 9,383 bytes), label commitment (`5cc7f58ed5...`), and promotion thresholds remain 100% immutable.
6. **Recovery Governance**:
   - Exactly ONE recovery execution is authorized on a fresh Kaggle session as **H-EXEC Recovery Attempt 1** following external review of the corrected execution-authority commit.

---

## D127 — V2-D3 Fresh-Holdout Evaluation Closure, Promotion Rejection, Holdout Burning, and Forensic Failure Analysis

**Status:** Accepted (Evidence: `verification-v2-d3-holdout-evidence.zip` SHA-256 `9e2b38d4189f9c68901051a07b999845c660ec6ab4b4fa1e6ec69d3088fe6a5d`, Size `10,463` bytes, Commit `77561aa7c4b242e12d011a84a21f3a262a17a0f8`, Postmortem: `docs/32-V2-D3-HOLDOUT-CLOSURE-AND-POSTMORTEM.md`)

**Context:**
The canonical one-shot Phase H-EXEC evaluation of candidate V2-D3 (`StructuredSemanticCitationVerifierD3`) was executed on Kaggle GPU on commit `77561aa7c4b242e12d011a84a21f3a262a17a0f8`. The evaluation completed 62 provider calls across 31 claims with authoritative Pass 1 metrics and stability Pass 2 validation.

**Decisions & Invariants:**
1. **Formal Evaluation Verdict & Decision**:
   - Verdict: `V2_D3_HOLDOUT_EXECUTION_FAILURE`
   - Evaluation Decision: `REJECT_V2_D3_PROMOTION`
   - `promotion_recommended = False`
   - `promotion_authorized = False` (Strict Invariant)
   - Production semantic verifier remains **DISABLED** in production pipelines.
2. **Scientific Metric Evaluation**:
   - Pass 1 Supported Retention Rate: $22 / 23 = 95.65\%$ (Passed $\ge 0.88$ gate).
   - Pass 1 Negative Catch Rate: $2 / 7 = 28.57\%$ (**FAILED** $\ge 0.50$ gate).
   - Pass 1 Valid Answer Retention Rate: $8 / 10 = 80.00\%$ (Passed $\ge 0.80$ gate).
   - Pass 1 Full Answer Accuracy: $10 / 16 = 62.50\%$ (Passed $\ge 0.60$ gate).
   - Pass 1 Claim Binary Accuracy: $24 / 30 = 80.00\%$ (Passed $\ge 0.70$ gate).
   - Negative catch failed significantly ($28.57\%$ vs $50.0\%$ threshold); 5 out of 7 negative claims escaped as False Accepts.
3. **Operational Telemetry & Error Isolation**:
   - 61/62 provider calls succeeded. 1 call (`103383:PRIMARY:C1`) failed with `BackendInitializationError` on Call 1 due to transient cold-start runtime loading. In Pass 2, Call 32 for `103383:PRIMARY:C1` executed cleanly and predicted `SUPPORTED`.
   - Stability: 30/30 successfully executed claims were 100% deterministic between Pass 1 and Pass 2.
   - Operational error remediation is separated from semantic verifier quality: fixing cold-start initialization would not repair the failed negative catch rate.
4. **Permanent Closure & Holdout Burning Invariants**:
   - Candidate V2-D3 development track is **PERMANENTLY CLOSED**.
   - **NO D3.3** will be created.
   - The 31 holdout claims are **BURNED / CONSUMED** and are converted to diagnostic development data only.
   - **NO holdout reruns** or post-hoc threshold changes permitted.
   - Any future candidate promotion requires a brand-new, untouched holdout.
5. **Post-Holdout Forensic Analysis Findings**:
   - 5 False Accepts classified: 3 $\times$ `ACTOR_ROLE_MISMATCH` (`125893:C1`, `125893:C3`, `90897:C1`), 1 $\times$ `CONDITION_EXCEPTION_OMITTED` (`45427:C1`), 1 $\times$ `ACTION_OBJECT_MISMATCH` / `QUANTITY_TEMPORAL_MISMATCH` (`95695:C1`).
   - 1 False Reject classified: `SYNTAX_FRAGMENT_STRICTNESS` (`61523:C1`).
   - Comparison with True Negatives (`162759:C1`, `3339:C1`) proves D3 only catches negatives with overt lexical antonyms or total keyword absence, failing on subtle legal subject substitutions and condition omissions.
6. **Future Architecture Direction (V3 Candidate)**:
   - Root cause: Single-call holistic entailment is dominated by lexical similarity, masking legal actor and scope boundaries.
   - Recommended next architecture: Option C (Structured Dimension Decomposition with 3 focused boolean checks: `legal_actor_aligned`, `activity_and_scope_aligned`, `conditions_and_numbers_accurate` with deterministic aggregation).
   - Strategic Priority: Shift primary engineering focus to Generation Grounding / Prompt Optimization (Task 2 metric leverage) and Retrieval / Reranking depth.

---

## D128 — Generation Grounding G1 Material-Fidelity Candidate, Prompt Identity, and A/B Evaluation Protocol

**Status:** Accepted (Implementation: `src/legal_agentic_rag/generation/model_generator.py`, Harness: `scripts/evaluate_generation_grounding_g1.py`, Specification: `docs/33-GENERATION-GROUNDING-G1-DEVELOPMENT.md`)

**Context:**
Following the forensic failure analysis of V2-D3 (D127), the repository shifted primary engineering resources to generation-stage grounding to prevent material proposition errors (actor substitutions, dropped conditions, object conflations) before claims are emitted.

**Decisions & Invariants:**
1. **G1 Material-Fidelity Grounding Candidate**:
   - Implements candidate profile `material_fidelity_v1` alongside `baseline`.
   - Added Vietnamese natural-language prompt instructions preserving:
     - Actor / Role: Exact legal subject without substitution.
     - Action / Object: Exact regulated activity without alteration.
     - Conditions / Exceptions: Preservation of all statutory prerequisites (no unconditioned broadening).
     - Legal Scope: Preservation of public/private entity and jurisdiction limits.
     - Numeric / Temporal: Retains strict verbatim numeric copying.
     - Full Coverage: Every material component must be supported by cited evidence.
     - List Noun Phrases: Direct faithful noun phrases from evidence are validated for list questions (mitigating `61523:C1` syntax strictness).
2. **Strict Invariants Preserved**:
   - Production default remains `grounding_profile="baseline"`. G1 is NOT promoted in this task.
   - Output contract is strictly unchanged: `ModelAnswerDraft` (`claims`, `insufficient_evidence`, `warnings`). No extra model-emitted fields.
   - Call count contract is unchanged: exactly 1 normal provider call per generation.
   - No second-pass reflection, critic, or verifier calls.
3. **Diagnostic Dataset Status**:
   - The 16 review packets from `verification-v2-holdout-review-packets-v1.zip` are burned diagnostic development data only.
   - Model sees only question and retrieved evidence; no labels or error tags.
4. **A/B Development Evaluation Harness**:
   - Created `scripts/evaluate_generation_grounding_g1.py` executing Baseline vs G1 on the 16 diagnostic cases.
   - Implemented deterministic pairwise blinding for `results/generation_g1_human_review_worksheet.md` with separate `generation_g1_blinding_key.json`.
5. **Pre-Registered Development Success Criteria**:
   - Criterion A: 0 execution errors for G1.
   - Criterion B: Eliminates material error on $\ge 4 / 5$ historical error mechanisms without new errors.
   - Criterion C: Preserves valid answer on $\ge 9 / 10$ gold valid cases.
   - Criterion D: G1 abstention rate does not exceed baseline by $> 15\%$.
   - Criterion E: 0 schema regressions.
   - Criterion F: Provider call parity.

---

## D129 — Phase D0 Official Dataset Census, Chunking Unit Boundary Audit, and Development Governance Contract

**Status:** Accepted (Parent Authority: `469dd45834f6ef2406198e3368669459bebeb264`, Evidence: `C:\Users\Nguyen\Downloads\data-d0-official-data-audit-evidence.zip` SHA-256 `eca404a749a45c00b6b7b94c7dee246fea39de385882e51343f6f1a20d93c27f`, Size `40,549` bytes, Report: `docs/34-DATA-CENSUS-AND-RETRIEVAL-UNIT-AUDIT-D0.md`)

**Context:**
Before investing in large-scale offline chunking or dense embedding overhauls, Phase D0 conducted a comprehensive, empirical census of the official dataset (`selected-contexts.zip`, `train.json`, `public-official.json`) and the current serving artifacts (`artifacts/uit-dsc-2026-task2-v0400`).

**Decisions & Invariants:**
1. **Official Data Census Authority & Checksum Distinction**:
   - Raw Contexts Archive: `selected-contexts.zip` byte identity SHA-256 is `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97` (97,276,888 bytes, 8,532 member files).
   - Canonical Context Content Revision: `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e` (verified across all 8,532 JSON member contents).
   - Corpus Distribution: 8,512 non-empty, 20 empty, 7,407 with title, 1,125 without title, 4 exact duplicate clusters (9 records). Largest document is `context_68843` (5,983,358 characters, ~1.24M words / 1,236,787 words).
   - Official Train: 7,000 question-answer pairs (`train.json`). Mean question length 88.1 chars / 19.7 words; mean answer length 1,575.7 chars / 347.4 words (median answer-to-question ratio 16.51x).
   - Official Public: 1,000 questions (`public-official.json`), all answers `null`. 1 exact question overlap with train.
2. **Serving Chunks & Boundary Risk Census**:
   - Total Chunks: 330,768 chunks across 8,512 unique documents.
   - Strategy: `article` (39.53%), `token_fallback` (32.65%), `clause_group` (24.53%), `standalone_block` (3.29%).
   - Boundary Risk Pairs: 2,056 adjacent chunk boundary risk pairs identified (971 cross-reference splits, 649 list header splits, 436 condition left-boundary truncations).
   - Critical Deficiency: All 108,009 `token_fallback` chunks (32.65% of corpus) currently have empty header context in `search_text`.
   - Metadata Deficiencies: 0% of chunks currently have `document_number`, `document_type`, or `effect_status` extracted from passage headers or document slugs.
3. **Train Q&A Linkability & High-Confidence Retrieval Proxy**:
   - 1,333 train questions (19.04%) possess unambiguous, high-confidence links to canonical context documents based on statutory document number citations in answers.
   - 639 train questions (9.13%) link unambiguously to specific articles.
   - 3,453 train citations reference external legal documents not in the 8,532 context corpus.
   - The 1,333 unambiguous links define a **HIGH-CONFIDENCE OFFICIAL-DATA RETRIEVAL PROXY** (not official relevance ground truth) without violating competition data rules (no synthetic QA).
4. **Historical BM25 Retrieval Proxy Baseline**:
   - Evaluated on SQLite FTS5 index (`bm25_documents`) across the historical 200-QA subset: Document Recall @ 1 = 48.0%, Recall @ 5 = 71.5%, Recall @ 10 = 79.5%, Recall @ 20 = 86.0%.
5. **Selection of Exactly ONE Phase D1 Candidate**:
   - Selected: **D1 — Parent-Context Enriched Token-Fallback Search Representation**.
   - Single Causal Variable: For existing `token_fallback` chunks only, enrich `search_text` with parent/legal context ALREADY deterministically available in existing chunk/source lineage.
   - Strict Invariants: Exact chunk count (330,768), chunk IDs, chunk boundaries, and raw chunk text strictly unchanged. Retrieval parameters, dense model, reranker, answer generator, and verifier unchanged.
   - Strict Prohibitions: No resegmenting, no metadata extraction, no adjacent-window stitching, no parameter tuning, no metadata fabrication.
   - Pre-Registered Measurement: Evaluate on high-confidence official-data proxy (all 1,333 links if practical + 200 historical subset), reporting ALL proxy vs AFFECTED subset (questions whose target has $\ge 1$ `token_fallback` chunk).
   - Pre-Registered Success Gate: Structural invariants pass; Primary: BM25 Recall@5 improves by $\ge 2.0$ absolute percentage points on same evaluation population; Secondary: BM25 Recall@10 must not regress; Tertiary: BM25 Recall@20 must not regress by $> 0.5$ absolute percentage points.
6. **Architectural Backlog (Future Hypotheses)**:
   - Backlog 1: Hierarchical Legal Chunking (Clause/Point resegmentation).
   - Backlog 2: Online Adjacent Chunk Window Expansion & Parent Stitching.
   - Backlog 3: Offline Legal Metadata Extraction from Slugs and Headers.
