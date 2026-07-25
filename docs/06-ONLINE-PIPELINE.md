# 06. Online Pipeline

## 1. Purpose

Online pipeline nhận câu hỏi người dùng và trả về câu trả lời có căn cứ
pháp luật.

Online pipeline chỉ sử dụng artifact đã được build từ offline phase.

---

## 2. Input

Input tối thiểu:

```json
{
  "question": "..."
}
```

Input nội bộ:

```json
{
  "query_id": "query-001",
  "question": "...",
  "top_k": 10,
  "strategy": "hybrid_rerank",
  "filters": {}
}
```

---

## 3. Query Validation

Kiểm tra:

- question tồn tại;
- question không rỗng;
- độ dài hợp lệ;
- encoding hợp lệ;
- top-k hợp lệ;
- strategy được hỗ trợ;
- filters hợp lệ.

Query không hợp lệ phải fail với lỗi rõ ràng.

---

## 4. Query Normalization

Normalization nhẹ:

- Unicode normalization;
- trim whitespace;
- normalize repeated spaces;
- normalize line breaks;
- loại control character không cần thiết.

Không được:

- bỏ dấu;
- xóa số;
- lowercase bắt buộc toàn bộ;
- loại stopword mạnh;
- xóa từ phủ định;
- xóa document number;
- rewrite bằng LLM trong fixed baseline.

---

## 5. Fixed Retrieval Strategies

### 5.1 BM25 Retrieval

Luồng:

```text
Normalized Question
→ BM25 Index
→ Top-K Legal Chunks
```

Phù hợp với:

- số hiệu văn bản;
- thuật ngữ pháp lý;
- tên thủ tục;
- tên cơ quan;
- mức tiền;
- thời hạn;
- câu hỏi có lexical overlap cao.

Milestone 7 triển khai BM25 search bằng SQLite FTS5 phía sau protocol chung.
Query chỉ dùng `normalized_question`; analyzer case-insensitive nhưng giữ dấu
tiếng Việt và số. `RetrievalFilters` được áp dụng exact trên document ID, loại
văn bản, lĩnh vực và trạng thái hiệu lực. Hit trả `RetrievalHit` cùng BM25 rank,
score, legal chunk metadata, artifact version và latency; backend không log
toàn bộ nội dung query.

### 5.2 Dense Retrieval

Luồng:

```text
Normalized Question
→ Query Embedding
→ Vector Search
→ Top-K Legal Chunks
```

Phù hợp với:

- câu hỏi đời thường;
- semantic matching;
- từ đồng nghĩa;
- diễn đạt khác với văn bản luật.

Milestone 8 triển khai `DenseRetriever` theo luồng:

```text
RetrievalQuery.normalized_question
→ EmbeddingProvider.embed_query (query prefix)
→ VectorBackend.search
→ RetrievalResponse
```

Online provider phải khớp tuyệt đối provider name/version, model name, revision
và dimension trong vector artifact. NumPy reference backend normalize query vector, exact cosine
trên matrix đã normalize, áp dụng unified exact filters và tie-break theo
`chunk_id`. `RetrievalHit` giữ dense rank/score và toàn bộ legal metadata;
latency tổng gồm cả query embedding. Không log toàn bộ query text.

### 5.3 Hybrid RRF

Luồng:

```text
BM25 Top-N
+
Dense Top-N
→ Reciprocal Rank Fusion
→ Hybrid Top-N
```

RRF kết hợp thứ hạng thay vì cộng trực tiếp raw score.

Mỗi hit phải ghi:

- BM25 rank nếu có;
- dense rank nếu có;
- RRF contribution;
- final fused rank.

Milestone 9 triển khai fixed hybrid retrieval theo policy:

- mỗi nhánh nhận `top_k = candidate_k`; fusion chỉ trả final `top_k`;
- contribution chuẩn là `1 / (rrf_constant + branch_rank)`, mặc định
  `rrf_constant = 60`;
- raw BM25/dense score chỉ được giữ trong trace, không tham gia phép cộng;
- candidate trùng được deduplicate theo `chunk_id`; cùng ID nhưng khác
  document/text/metadata là retrieval error;
- nhánh không chứa candidate có contribution `0.0` và rank tương ứng là null;
- tie theo RRF score được break bằng `chunk_id`;
- BM25 và vector index phải cùng legal-chunks source identity trước khi search;
- lỗi một nhánh làm hybrid request fail, không silently degrade thành một nhánh;
- warning được namespace theo `bm25:` hoặc `dense:`.

`FixedRetriever` route trực tiếp `bm25`, `dense`, `hybrid` mà không cần Agent;
strategy mặc định là `hybrid`. Rerank và graph chưa thuộc Milestone 9.

### 5.4 Hybrid Rerank

Luồng:

```text
Hybrid Candidates
→ Cross-Encoder Reranker
→ Final Top-K
```

Reranker nhận:

```text
question + legal chunk
```

Reranker chỉ chạy trên candidate set giới hạn.

Không chạy cross-encoder trên toàn corpus.

Milestone 10 triển khai policy:

- hybrid retrieval lấy tối đa `candidate_k`, mặc định và giới hạn cứng là 100;
- cross-encoder mặc định là
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` tại revision bất biến
  `1427fd652930e4ba29e8149678df786c240d8825`;
- inference mặc định chạy CPU, batch 8, max length 512 và không fine-tune;
- model nhận `rewritten_question` nếu có, nếu không dùng `normalized_question`,
  ghép với `RetrievalHit.text`;
- final score là raw cross-encoder logit, không cộng với raw BM25, dense hoặc
  RRF score;
- tie theo reranker score được break bằng retrieval rank trước đó rồi `chunk_id`;
- output chỉ được đổi rank, final score và strategy; document ID, text và legal
  metadata phải giữ nguyên;
- BM25/dense rank, raw score, RRF contribution và RRF score vẫn được giữ trong
  `RetrievalTrace`, đồng thời thêm `reranker_score`;
- lỗi tải model và lỗi inference được phân loại riêng, không silently trả lại
  hybrid ranking;
- model được lazy-load nên import package không tải model hoặc gọi network.

`FixedRetriever` route strategy `hybrid_rerank` khi được inject một implementation
`Reranker`. Strategy `rerank` chỉ là stage nội bộ và không phải entry point của
fixed retrieval.

### 5.5 Graph Retrieval

Luồng:

```text
Text Retrieval
→ Seed Chunks
→ Seed Documents
→ Graph Expansion
→ Related Documents
→ Related Chunks
→ Candidate Merge
→ Rerank
```

Graph traversal mặc định:

- 1 hop;
- tối đa 2 hop;
- filter relationship type khi có thể;
- không mở rộng toàn graph;
- phải ghi retrieval trace.

Milestone 11 triển khai fixed graph flow:

```text
Hybrid seed retrieval
→ Unique seed documents
→ Directed BFS expansion
→ Hybrid retrieval restricted to reached documents
→ Merge bounded candidates
→ Cross-encoder rerank once
→ Graph RetrievalResponse
```

Policy:

- graph không chạy nếu text retrieval không tạo seed;
- graph không thay BM25/dense và không tự quét toàn corpus;
- `graph_seed_chunk_k`, `graph_seed_document_k`,
  `graph_related_document_k`, hop và relationship filters đều qua typed config;
- explicit document filters không bị graph expansion nới rộng;
- expanded hit ghi đủ `graph_hop` và ordered `graph_path`;
- graph/chunk artifacts phải cùng dataset/revision;
- candidate pool không vượt `candidate_k` hoặc reranker limit;
- `FixedRetriever` chỉ route `graph` khi đã inject graph backend, chunk manifest
  và reranker.

---

## 6. Fixed Baseline Flow

Fixed baseline chính:

```text
Question
→ Normalize
→ BM25 Search
→ Dense Search
→ RRF Fusion
→ Candidate Deduplication
→ Cross-Encoder Reranking
→ Context Builder
→ Answer Generator
→ Citation Verifier
→ Final Response
```

Đây là luồng phải hoàn chỉnh trước Agentic workflow.

---

## 7. Candidate Deduplication

Candidate merger phải:

- deduplicate theo `chunk_id`;
- nhóm theo `document_id` khi cần;
- giữ score contribution;
- giữ strategy provenance;
- giữ rank từ từng nhánh;
- không làm mất metadata;
- không hợp nhất hai chunk khác nhau chỉ vì text gần giống.

---

## 8. Context Builder

Context builder nhận top-ranked hits và tạo evidence.

Công việc:

- deduplicate evidence;
- giới hạn token;
- ưu tiên rank cao;
- ưu tiên văn bản còn hiệu lực;
- giữ cảnh báo với văn bản hết hiệu lực;
- nhóm evidence theo document nếu cần;
- giữ article number;
- giữ document number;
- giữ source URL;
- gán evidence ID.

Ví dụ:

```text
[E1]
Văn bản: ...
Số ký hiệu: ...
Điều 15: ...
Nội dung: ...

[E2]
Văn bản: ...
Điều 20: ...
Nội dung: ...
```

---

## 9. Context Grading

Context grader đánh giá:

- relevance;
- coverage;
- consistency;
- legal metadata availability;
- multi-document sufficiency;
- effect status warning;
- evidence duplication.

Output logic:

```json
{
  "is_sufficient": true,
  "score": 0.85,
  "missing_aspects": [],
  "warnings": []
}
```

Fixed baseline có thể dùng rule hoặc cấu hình đơn giản.

LLM-based context grading thuộc giai đoạn sau.

---

## 10. Answer Generation

Generator phải nhận:

- original question;
- evidence list;
- generation instruction;
- output schema.

Generator phải:

- chỉ dùng evidence;
- citation theo evidence ID;
- không thêm căn cứ ngoài context;
- trả lời bằng tiếng Việt;
- nói rõ khi thiếu căn cứ;
- giữ cảnh báo hiệu lực;
- tránh khẳng định pháp lý tuyệt đối.

### 10.1 Milestone 18 Model-backed Generation

M18 giữ `AnswerGenerator` làm core boundary và thêm `ChatModelProvider` cho ranh
giới model server. Concrete baseline gọi một endpoint Chat Completions tương
thích OpenAI bằng Python standard library; model, revision, endpoint, timeout,
output-token limit và tên biến môi trường chứa API key đều do typed config cung
cấp.

Luồng model-backed:

```text
Selected Evidence
→ evidence-only Vietnamese prompt
→ JSON-mode model completion
→ ModelAnswerDraft validation
→ evidence-ID allowlist
→ system-built Citation metadata
→ existing CitationVerifier
→ AnswerResponse hoặc Abstention
```

Model chỉ được tạo:

- answer text có marker `[E#]`;
- danh sách `cited_evidence_ids`;
- `insufficient_evidence`;
- warnings.

Model không được tự tạo `chunk_id`, `document_id`, số văn bản, số Điều hoặc URL
citation. Các field này luôn được hệ thống ánh xạ từ `Evidence` đã chọn. Unknown
evidence ID, JSON sai schema hoặc marker bị thiếu đều fail closed thành model
error; Agent hiện tại xử lý lỗi theo retry/stopping policy có giới hạn.

`extractive` vẫn là default backend. `openai_compatible` chỉ được bật rõ bằng
config và phải có endpoint cùng model name/revision đã pin. M18 chưa chọn model
production, chưa fine-tune, chưa semantic-verify từng claim và không gọi model
thật trong test mặc định.

---

## 11. Citation Verification

Citation verifier kiểm tra:

- evidence ID có tồn tại;
- chunk ID có tồn tại;
- citation không bị lặp sai;
- answer không trích evidence không được cung cấp;
- article number khớp metadata;
- document ID khớp evidence;
- warning khi answer không có citation.

Baseline verifier có thể là rule-based.

Semantic claim verification được bổ sung sau.

### 11.1 Milestone 12 Fixed RAG Policy

Milestone 12 triển khai:

```text
Fixed Retriever
→ Context Builder
→ Structural Context Grader
→ Extractive Answer Generator
→ Rule-based Citation Verifier
→ AnswerResponse hoặc Abstention
```

Context builder:

- deduplicate theo `chunk_id` và từ chối duplicate có legal payload khác nhau;
- giữ nguyên toàn bộ chunk text, không cắt giữa chừng nội dung pháp luật;
- giới hạn `max_evidence` và `max_context_tokens` qua typed config;
- ưu tiên retrieval rank; chỉ hạ ưu tiên effect status khi inactive label được
  cấu hình rõ ràng;
- giữ article/document/source metadata và retrieval trace trong evidence;
- cảnh báo khi effect status không rõ hoặc context budget hết.

Structural grader chỉ kiểm tra số lượng, identity và metadata bắt buộc theo
config. Nó luôn ghi `semantic_relevance_checked = false`; không được mô tả như
semantic context grading.

`ExtractiveAnswerGenerator` chỉ trình bày nguyên văn selected evidence với
marker `[E#]`. Đây là backend tham chiếu an toàn, không phải model sinh câu trả
lời cuối cùng.

Rule-based verifier kiểm tra exact evidence/chunk/document/article/document
number/source URL. Nó không kiểm tra semantic support của từng claim.

Fixed service:

- không gọi generator khi structural context không đủ;
- citation invalid làm answer bị loại và chuyển thành abstention;
- không silently trả answer chưa verify;
- giữ context grade, verification result, artifact versions và selected
  retrieval trace trong response metadata;
- dùng query ID làm trace ID deterministic trong fixed baseline.

---

## 12. Failure and Abstention Behavior

Nếu không đủ evidence, hệ thống phải:

- không đoán;
- không tạo citation giả;
- không trả lời chắc chắn;
- trả thông báo thiếu căn cứ;
- ghi warnings;
- giữ retrieval trace.

Ví dụ:

```json
{
  "answer": "Hệ thống chưa tìm thấy căn cứ pháp luật đủ rõ trong dữ liệu hiện có để trả lời chắc chắn.",
  "citations": [],
  "insufficient_evidence": true,
  "warnings": [
    "Insufficient retrieval evidence"
  ]
}
```

---

## 13. Agentic Workflow

### 13.1 Milestone 13 Tool Boundary

Trước khi có Agent, fixed capabilities được đóng gói thành:

```text
bm25_search
dense_search
hybrid_search
rerank_search
graph_search
context_grading
answer_generation
citation_verification
```

Mỗi tool có:

- closed name;
- description rõ phạm vi;
- Pydantic input/output model;
- timeout budget;
- typed direct invocation;
- JSON-compatible registry envelope.

Registry:

- chỉ chạy tool đã register explicit;
- không có plugin discovery, dynamic import hoặc tool tự đăng ký;
- validate payload trước invocation;
- từ chối output sai contract;
- map validation/domain errors sang safe `ToolErrorType`;
- không đưa exception text, path, secret hoặc stack trace ra output;
- không nuốt unexpected programming exception;
- không log payload hoặc legal content;
- loại output nếu elapsed time vượt configured budget.

Baseline timeout là wall-clock budget check sau synchronous invocation để không
chuyển SQLite connection sang thread khác. Backend/model bên ngoài vẫn phải có
cooperative transport timeout riêng; registry không tuyên bố có thể cưỡng bức
dừng một synchronous provider đang treo.

Agentic workflow chỉ được triển khai sau fixed baseline.

Milestone 14 reference workflow:

- route mặc định `hybrid_rerank → graph → hybrid`;
- explicit `requested_strategy` được ưu tiên nếu tool tương ứng đã đăng ký;
- route luôn được lọc qua descriptor của closed registry;
- query rewrite chỉ tái sử dụng `original_question` hoặc
  `normalized_question`, không tự thêm thuật ngữ pháp lý;
- context builder không phải tool vì chỉ chuyển typed retrieval output thành
  bounded evidence, không truy cập backend;
- generation hoặc citation verification thất bại tạo abstention;
- terminal output là `AgentRunResult` gồm `AnswerResponse`, `AgentState`,
  `AgentStopReason` và total latency.

Luồng:

```text
Question
→ Analyze
→ Select Retrieval Tool
→ Retrieve
→ Observe
→ Grade Context
→ Sufficient?
    Yes → Generate Answer
    No  → Rewrite Query or Select Another Tool
→ Retry
→ Verify Citation
→ Final Response
```

---

## 14. Agent State

State logic:

```json
{
  "trace_id": "trace-001",
  "original_question": "...",
  "normalized_question": "...",
  "current_query": "...",
  "selected_strategy": null,
  "retrieval_history": [],
  "candidate_hits": [],
  "selected_evidence": [],
  "context_grade": null,
  "retry_count": 0,
  "answer": null,
  "citations": [],
  "warnings": []
}
```

---

## 15. Retry Policy

Baseline:

```text
max_retry = 2
```

Agent phải dừng khi:

- context đủ;
- đạt max retry;
- không còn strategy mới;
- query rewrite không thay đổi;
- không có evidence đáng tin;
- tool lỗi không thể phục hồi;
- timeout.

Agent không được lặp vô hạn.

Query rewrite không thay đổi không buộc dừng nếu còn strategy mới: workflow đi
theo nhánh “Select Another Tool”. Khi không còn cả query form mới lẫn strategy
mới, workflow dừng với `no_new_strategy`.

---

## 16. Online Output

```json
{
  "question": "...",
  "answer": "...",
  "citations": [
    {
      "evidence_id": "E1",
      "chunk_id": "...",
      "document_id": "...",
      "document_title": "...",
      "document_number": "...",
      "article_number": "...",
      "source_url": "..."
    }
  ],
  "insufficient_evidence": false,
  "warnings": [],
  "retrieval_strategy": "hybrid_rerank",
  "trace_id": "trace-001"
}
```

---

## 17. Online Observability

Cần đo:

- total latency;
- query normalization latency;
- retrieval latency;
- fusion latency;
- reranker latency;
- graph latency;
- context building latency;
- generation latency;
- verification latency;
- candidate counts;
- retry count;
- selected strategy;
- errors and warnings.

---

## 18. Online Runtime Assembly

`OnlineRuntimeFactory` thực hiện startup theo thứ tự:

1. load và checksum-validate legal-chunks manifest/payload;
2. load BM25, vector và graph manifests;
3. kiểm tra cùng dataset/revision và source processing lineage;
4. kiểm tra embedding provider/model/revision/version/dimension;
5. load concrete reference backends;
6. ghép dense, hybrid, rerank và graph retrieval;
7. tạo closed tool registry và bounded Agent;
8. trả `OnlineRuntime`.

Factory fail trước serving nếu artifact hoặc model identity không tương thích.
Factory không download dataset, preprocess, build hoặc persist artifact.

---

## 19. Serving Flow

Milestone 15 thêm một boundary mỏng trên `OnlineRuntime`:

```text
HTTP/Gradio request
→ LegalQuestionRequest validation
→ Vietnamese NFC + whitespace normalization
→ RetrievalQuery with query ID and bounded limits
→ OnlineRuntime.retrieve() or OnlineRuntime.answer()
→ Unified RetrievalResponse or AnswerResponse
```

FastAPI lifespan load runtime đúng một lần. Nếu manifest, checksum, lineage,
model identity hoặc backend load lỗi, process fail startup thay vì báo ready.

Public endpoints:

- `GET /api/v1/health`;
- `POST /api/v1/retrieve`;
- `POST /api/v1/answer`.

Gradio tại `/ui` là optional diagnostic consumer và dùng chung runtime. Serving
không download dataset, build index, truy cập raw field hoặc giữ concrete
database client.

Validation error và domain error được map sang stable error category; nội dung
exception nội bộ, local path và request payload không được trả về client.

---

## 20. Evaluation Flow

```text
Labeled EvaluationCase JSONL
→ benchmark validation + SHA-256
→ RetrievalQuery with configured fixed strategy
→ OnlineRuntime.retrieve / OnlineRuntime.answer
→ per-case metrics and sanitized failures
→ aggregate metrics, latency, resources
→ immutable summary/cases/errors reports
```

Evaluation không thay đổi online artifacts. Retrieval metric có thể dùng nhãn
chunk hoặc document; generation metric thiếu label được giữ là unavailable,
không tự gán bằng 0.
