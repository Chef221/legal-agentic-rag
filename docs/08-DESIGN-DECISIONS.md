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
