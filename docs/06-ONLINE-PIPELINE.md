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

## 4.1 Query Understanding and Variants

Milestone 19 phân tích deterministic trên `normalized_question`:

- số hiệu văn bản;
- Điều, Khoản và Điểm được nhắc trực tiếp;
- năm;
- tín hiệu phạm vi như `đối với`, `trong trường hợp`;
- tín hiệu quan hệ như `sửa đổi`, `thay thế`, `bãi bỏ`;
- intent bảo thủ: reference, relationship, quantitative, procedure,
  eligibility, obligation, prohibition hoặc definition.

Runtime không tin query-analysis do client cung cấp mà luôn tính lại. Query
variants chỉ được tạo bằng:

1. câu normalized đầy đủ;
2. bỏ framing như `xin hỏi` hoặc `theo quy định của pháp luật`;
3. ghép lại reference xuất hiện trực tiếp trong câu hỏi.

Không dùng synonym dictionary, LLM rewrite hoặc tự thêm thuật ngữ pháp lý.
`online.query_understanding.max_variants` giới hạn từ 1 đến 5, mặc định 3.

Khi có nhiều variant, hybrid retrieval gọi BM25 và dense cho từng variant rồi
RRF toàn bộ branch ranks. Raw score không được cộng trực tiếp. Mỗi per-variant
rank, raw score và RRF contribution được giữ trong `RetrievalTrace`.

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
Mỗi branch dùng `rewritten_question` khi có, nếu không dùng
`normalized_question`; analyzer case-insensitive nhưng giữ dấu
tiếng Việt và số. `RetrievalFilters` được áp dụng exact trên document ID, loại
văn bản, lĩnh vực và trạng thái hiệu lực. Hit trả `RetrievalHit` cùng BM25 rank,
score, legal chunk metadata, artifact version và latency; backend không log
toàn bộ nội dung query.

Từ version `0.20.3`, SQLite backend dùng hidden FTS5 `rank` cho bounded top-k
thay vì global secondary sort theo `bm25(...)` và `chunk_id`. Query planner tra
document frequency bằng temporary `fts5vocab`, luôn giữ số và legal semantic
modifier, rồi chọn tối đa `online.bm25_runtime.max_query_terms` term có độ phân
biệt cao. Đây không phải static stopword removal; Unicode, số và phủ định vẫn
được analyzer bảo toàn. Planner phát warning khi phải giới hạn query terms.

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
  ghép với legal-context candidate text;
- legal-context mặc định gồm tên/số/loại văn bản, cơ quan ban hành, lĩnh vực,
  metadata hiệu lực, cấu trúc Điều/Khoản/Điểm và `RetrievalHit.text`;
- `input_mode = "text_only"` giữ input cũ cho controlled A/B benchmark;
- chỉ named unified metadata được đưa vào model; URL và arbitrary metadata bị
  loại;
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

Milestone 20 chèn bước applicability screening trước token-budget selection:

```text
Ranked Retrieval Hits
→ exact chunk deduplication
→ explicit document/article reference matching
→ lexical overlap + configured effect-status screening
→ deterministic evidence ordering
→ whole-chunk token/count budgeting
→ Evidence + EvidenceSelectionTrace
```

Reference chỉ được lấy từ trusted `QueryAnalysis` do runtime dựng lại. Mismatch
không bị xóa tuyệt đối vì có thể là supporting evidence, nhưng bị hạ thứ tự và
được cảnh báo. Inactive status chỉ có tác động nếu label được cấu hình rõ trong
`generation.inactive_effect_statuses`.

Selection score không được mô tả là xác suất relevance hoặc kết luận hiệu lực
pháp lý. Nó chỉ là policy score có trace, không cộng vào retrieval/reranker score.

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

Milestone 20 mở rộng rule-based grader:

- explicit document/article reference phải có ít nhất một selected evidence
  khớp;
- context chỉ gồm `inactive` hoặc `reference_mismatch` không được xem là đủ;
- lexical overlap được báo cáo để chẩn đoán nhưng không tự tuyên bố semantic
  relevance;
- metadata ghi `legal_applicability_interpreted = false`.

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
config và phải có endpoint cùng model name/revision đã pin.
`transformers` chạy một causal language model local có chat template, cũng yêu
cầu pin model/revision cùng device và dtype rõ ràng. Provider:

- lazy-load tokenizer/model ở lần dùng đầu tiên;
- serialize inference trên một shared model instance;
- dùng deterministic decoding khi `temperature = 0`;
- chỉ decode token mới sinh, không lẫn prompt vào completion;
- từ chối prompt vượt `max_input_tokens`, không silently truncate legal text;
- không log prompt, question hoặc evidence content;
- phân loại model-load và inference failure theo exception taxonomy.

Version `0.20.8` cho phép parser bóc đúng một JSON object khỏi preamble/code
fence thường gặp nhưng không tự sửa field hoặc citation. Draft vẫn phải qua
strict schema, evidence-ID allowlist và exact `[E#]` marker matching. Khi draft
không hợp lệ, generator có tối đa một correction attempt theo typed
`max_structured_output_retries`; retry không gửi lại raw completion và log chỉ
ghi failure category.

Version `0.20.9` dùng thứ tự marker `[E#]` xuất hiện trong answer làm canonical
citation selection. Điều này cho phép sửa deterministic trường hợp model trả
`cited_evidence_ids` khác thứ tự hoặc dư ID không được dùng trong answer. Mọi
declared ID và marker vẫn phải thuộc selected-evidence allowlist; answer đủ căn
cứ vẫn phải có ít nhất một marker và citation metadata vẫn do hệ thống dựng.

Version `0.20.10` xử lý model không tuân thủ inline-marker formatting:

- combined bracket form như `[E1, E2]` được parse thành từng verified ID;
- nếu sufficient draft không có bracket marker, declared IDs đã qua allowlist
  được render deterministic ở cuối answer;
- không suy luận claim-to-evidence mapping mới và không tạo evidence ID;
- unknown declared ID, unknown visible marker và insufficient draft có marker
  vẫn fail closed.

Qwen2.5-3B-Instruct revision
`a1d308dfcc03e09da285d49d912439a655a571e8` là candidate tham chiếu cho smoke
test GPU 16 GiB, không phải model production đã được chốt. M18 chưa fine-tune,
chưa semantic-verify từng claim và không gọi model thật trong test mặc định.

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

Semantic claim verification được triển khai tùy chọn từ Milestone 22.

### 11.2 Milestone 21 Claim-level Grounding

Với synthesized answer:

```text
Answer text
→ sentence/claim segmentation
→ inline [E#] extraction
→ response Citation + selected Evidence allowlist
→ lexical support check
→ exact numeric preservation
→ claim-negation preservation
→ per-claim result
→ verified answer hoặc abstention
```

Mỗi claim pháp lý phải có marker riêng. Citation object tồn tại nhưng không được
dùng trong answer không đủ để ground claim. Một marker ở câu cuối không tự bao
phủ các câu trước.

Rule-based baseline chỉ xác nhận:

- claim có linked evidence thật;
- claim/evidence có minimum lexical overlap;
- mọi số trong claim xuất hiện trong linked evidence;
- từ phủ định trong claim không được tự thêm so với evidence.

Nó không chứng minh semantic entailment, không xử lý logic ngoại lệ phức tạp và
không phát hiện mọi trường hợp đảo nghĩa. Extractive answer không chạy claim
verification vì nội dung evidence được trình bày nguyên văn; identity verifier
vẫn chạy bình thường.

### 11.3 Milestone 22 Model-backed Semantic Claim Verification

Khi `online.semantic_verification.backend` khác `disabled`:

```text
M21 identity + claim hard checks
→ chỉ giữ claim đã qua hard checks
→ gửi claim và đúng linked evidence qua ChatModelProvider
→ strict JSON: claim_id + supported|contradicted|insufficient
→ kiểm tra đủ, đúng thứ tự và không trùng claim_id
→ gắn lại trusted evidence IDs và model provenance
→ verified answer hoặc abstention
```

Semantic verifier không được tạo citation, evidence ID, document metadata hoặc
legal conclusion mới. `contradicted`, `insufficient`, output sai schema, lỗi
provider và prompt vượt giới hạn đều fail closed. Model không chạy nếu hard
checks đã thất bại, answer là abstention hoặc answer dùng extractive backend.
Backend mặc định `disabled`, vì vậy test mặc định không tải model, không gọi
network và không cần GPU.

Với hai backend `transformers` cùng model name, revision, device, dtype và
`local_files_only`, runtime chia sẻ model weights. Sai bất kỳ phần nào trong
runtime identity thì mỗi provider phải tự load backend tương ứng; không được
silently reuse weights khác cấu hình.

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
- query rewrite tái sử dụng bounded query variants trước
  `original_question`/`normalized_question`, không tự thêm thuật ngữ pháp lý;
- context builder không phải tool vì chỉ chuyển typed retrieval output thành
  bounded evidence, không truy cập backend;
- generation hoặc citation verification thất bại tạo abstention;
- terminal output là `AgentRunResult` gồm `AnswerResponse`, `AgentState`,
  `AgentStopReason` và total latency.

Milestone 19 giữ explicit requested strategy ở vị trí đầu. Nếu không có
requested strategy, relationship intent ưu tiên `graph`; query có reference
hoặc quantitative intent giữ `hybrid_rerank` đầu và đưa BM25 vào retry plan.
Route vẫn bị giới hạn bởi `max_retry = 2` và closed registry.

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

1. load legal-chunks, BM25, vector và graph manifests;
2. chạy deep validation hoặc đối chiếu validated build report;
3. kiểm tra cùng dataset/revision và source processing lineage;
4. kiểm tra embedding provider/model/revision/version/dimension;
5. load concrete reference backends;
6. ghép dense, hybrid, rerank và graph retrieval;
7. tạo closed tool registry và bounded Agent;
8. trả `OnlineRuntime`.

Factory fail trước serving nếu artifact hoặc model identity không tương thích.
Factory không download dataset, preprocess, build hoặc persist artifact.

`online.startup_validation.mode` có hai policy:

- `full` mặc định: checksum, SQLite integrity, record count và vector-value
  validation được chạy lại;
- `validated_report`: chỉ dùng khi artifact set immutable đã có
  `build_validation.json` hợp lệ. Runtime bắt buộc đối chiếu exact current
  manifests và required deep checks trong report trước khi bỏ các corpus-sized
  integrity scans.

Embedding provider công bố pinned dimension mà không load model weights.
Concrete model chỉ lazy-load ở dense query đầu tiên. Startup log thời gian riêng
cho manifest validation, BM25, vector, graph và tổng runtime.

---

## 19. Serving Flow

Milestone 15 thêm một boundary mỏng trên `OnlineRuntime`:

```text
HTTP/API UI request
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

Diagnostic UI tại `/ui` là optional same-origin consumer của public answer API
và dùng chung runtime. Serving
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

---

## 21. Memory-bounded Full-corpus Vector Load

Online NumPy backend không được materialize toàn bộ `chunks.jsonl` thành
`LegalChunk` objects. Startup thực hiện:

1. checksum vector payload có progress;
2. memory-map `vectors.npy`;
3. scan và validate từng JSONL record, chỉ giữ byte offset, chunk ID và compact
   postings cho `document_id`, `document_type`, `legal_field`, `effect_status`;
4. kiểm tra finite và unit norm theo batch;
5. log tiến độ mà không log legal content.

Dense query tính exact cosine theo batch. Query không filter không tạo advanced
index copy của toàn ma trận; query có filter chỉ copy từng bounded batch.
Backend chỉ đọc và Pydantic-validate full metadata cho top-k hit cuối cùng.
Artifact format, score, filter semantics và deterministic chunk-ID tie-break
không thay đổi.

Các giới hạn execution nằm tại `online.vector_runtime`:

- `validation_batch_size`;
- `search_batch_size`;
- `load_progress_interval_records`;
- `checksum_progress_interval_bytes`;
- `prefer_serving_metadata`;
- `require_serving_metadata`.

---

## 22. M44.2 Non-gold Retrieval Diagnostics

`RetrievalDiagnosticsRunner` là đường chạy evaluation read-only, không nằm
trong request path của API. Với mỗi question, runner gọi đúng ba strategy:

```text
BM25
Dense
Hybrid RRF
```

Mỗi response phải trả đúng strategy và query ID được yêu cầu. Report ghi:

- ordered chunk/document identities của từng branch;
- hit count, unique-document count và latency;
- BM25/dense overlap và Jaccard;
- document diversity của hybrid;
- kết quả khớp tham chiếu Điều/Khoản/Điểm/số văn bản nếu query nêu rõ;
- warning/error taxonomy;
- optional lexical answer-term coverage khi source có reference answer.

Coverage chỉ là diagnostic hypothesis, không phải gold relevance. Runner không
gọi generator/reranker/Agent workflow, không sửa artifact và không persist nội
dung question, answer hoặc legal passage. Source và application config được
khóa bằng SHA-256; output là immutable directory.

### 22.1 M44.3 optional reranker branch

Khi CLI nhận `--include-reranker`, runner gọi thêm `hybrid_rerank` với cùng
question, top-k và candidate-k. Report so sánh hybrid trước rerank với output
sau rerank bằng top-k overlap/Jaccard, document diversity, mean absolute rank
change, explicit-reference match và optional answer-term coverage delta.

Comparison runtime enrich query đúng một lần. Sparse và dense mỗi nhánh chạy
đúng một lần ở `candidate_k`; primary branch output được chiếu về final `top_k`,
trong khi cùng candidate pool được fusion và chuyển thẳng cho reranker.
Multi-query variant bổ sung vẫn chạy đúng một lần cho từng variant. Runner
không gọi lại BM25/dense chỉ để quan sát từng strategy.

Đây là retrieval-only gate. Nó không gọi generator và coverage delta vẫn không
phải relevance metric. Candidate-k 20/40/60 được chạy thành ba report immutable;
chỉ cấu hình có cost/behavior hợp lý mới chuyển sang full answer scoring.

Version `0.20.4` bổ sung optional persisted `vector_serving` sidecar:

- `legal-rag-prepare-serving` scan validated `vector/chunks.jsonl` đúng một lần;
- SQLite lưu `row_index`, JSONL byte offset, chunk ID và các unified filter
  columns;
- source vector manifest identity và `chunks_sha256` khóa compatibility;
- runtime mở SQLite read-only/immutable và chỉ Pydantic-parse final hits;
- thiếu sidecar có thể fallback về legacy scan hoặc fail closed theo config;
- sidecar không chứa embeddings và không thay đổi vector score/tie-break.

Version `0.20.5` thay Gradio diagnostic consumer bằng một trang HTML same-origin:

- submit trực tiếp JSON tới `/api/v1/answer`;
- không dùng queue, SSE hoặc WebSocket;
- không suy ra public URL từ internal host `127.0.0.1`;
- hoạt động qua Colab port proxy và vẫn dùng đúng một online runtime;
- chỉ render response bằng text-safe DOM APIs.

Version `0.20.6` bổ sung exact GPU-resident dense scoring:

- `online.vector_runtime.search_device` chọn `cpu` hoặc `cuda`;
- CUDA load matrix float32 hiện có theo bounded transfer batches đúng một lần;
- query không filter dùng một matrix-vector product trên GPU;
- filtered query dùng bounded GPU index-select batches;
- score vẫn là exact normalized inner product và final tie-break vẫn theo
  `chunk_id`;
- CUDA không khả dụng hoặc thiếu bộ nhớ làm startup fail rõ ràng, không silently
  fallback CPU;
- không re-embed, rebuild artifact hoặc thêm vector database.

## Official M40 Serving Profile

`configs/uit-dsc-2026-task2-serving.example.json` là consumer profile của
artifact M40, không phải cấu hình rebuild. Profile này yêu cầu full-corpus
validation report và `vector_serving` sidecar tương thích, giữ vector search
trên CPU mặc định, giới hạn Agent ở đúng hybrid strategy và dùng extractive
generator cho smoke test/batch baseline không cần GPU.

Model-backed generation là profile thực nghiệm riêng và chỉ chạy khi model đã
đăng ký/phê duyệt cùng tài nguyên phù hợp. Profile không dùng API, dữ liệu ngoài
hoặc artifact AIO.

Smoke test M41 trên public question `80189` xác nhận startup validated-report
khoảng 0,08 giây, cả BM25/dense/hybrid đều trả đủ 5 hit và extractive workflow
kết thúc `answer_verified` với 5 citation. Cold dense mất khoảng 10,50 giây do
nạp E5; warm hybrid mất khoảng 0,26 giây. Đây chỉ là kiểm tra vận hành vì public
data không có retrieval relevance labels.
