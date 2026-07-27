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

**Status:** Accepted

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

**Status:** Accepted

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
