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
