# 12. Hướng dẫn hệ thống dành cho thành viên UIT-SV5T

## 1. Đọc tài liệu này để làm gì?

Tài liệu này là bản đồ thực hành của repository. Sau khi đọc xong, thành viên
cần trả lời được năm câu hỏi:

1. Dữ liệu đi vào hệ thống ở đâu và được biến đổi như thế nào?
2. Một câu hỏi đi qua những module nào trước khi có câu trả lời?
3. Mỗi package trong `src/legal_agentic_rag` chịu trách nhiệm gì?
4. Cách build, chạy, đánh giá và tạo file nộp là gì?
5. Phần nào đã có code và phần nào vẫn phải chờ BTC?

Đây là tài liệu onboarding, không thay thế schema và design decisions. Khi cần
chi tiết contract, xem `docs/07-UNIFIED-SCHEMA.md`; khi cần biết lý do kiến trúc,
xem `docs/08-DESIGN-DECISIONS.md`.

## 2. Tóm tắt hệ thống trong một phút

Hệ thống là một Agentic RAG trả lời câu hỏi pháp luật Việt Nam:

```text
Corpus chính thức BTC
→ kiểm tra và chuyển về unified schema
→ parse cấu trúc pháp luật
→ chunk
→ build BM25 + vector + graph
→ validate và lưu artifact

Câu hỏi
→ hiểu và chuẩn hóa query
→ BM25 + dense
→ RRF fusion
→ cross-encoder rerank
→ graph expansion khi cần
→ chọn evidence
→ grade context
→ sinh câu trả lời
→ kiểm tra claim và citation
→ trả AnswerResponse hoặc abstain

Tập câu hỏi chính thức
→ batch inference có checkpoint
→ submission formatter
→ submission.zip
→ Codabench
```

Hai nửa quan trọng:

- **Offline:** xử lý corpus một lần và tạo artifact.
- **Online:** chỉ load artifact có sẵn để trả lời câu hỏi; tuyệt đối không build
  lại corpus/index trong request.

## 3. Trạng thái và giới hạn hiện tại

### Đã có

- unified schema bằng Pydantic v2;
- adapter dữ liệu UIT DSC 2026;
- cleaner, legal parser và article-first chunker;
- SQLite FTS5 BM25;
- multilingual E5 embedding và NumPy exact vector search;
- RRF fusion, cross-encoder reranker và bounded graph retrieval;
- query understanding, evidence selection và context grading;
- extractive/model-backed generation;
- rule-based claim/citation verification và optional semantic verification;
- deterministic bounded Agent workflow;
- FastAPI, same-origin diagnostic UI;
- evaluation, benchmark governance và comparison;
- resumable competition build, batch inference và Codabench formatter.

### Baseline M43.1 đã chạy thật

- official corpus đã được parse/chunk thành 330.768 chunk;
- BM25, vector 384 chiều, vector-serving metadata và zero-edge graph đã build,
  checksum và validation;
- public batch đã hoàn tất 1.000/1.000 câu bằng hybrid + Qwen2.5-3B;
- submission đã được Codabench chấm METEOR `0.07862292376534387` và ROUGE-L
  `0.16735433212043324`;
- model E5, mMARCO reranker và Qwen2.5-3B đã được người dùng xác nhận BTC duyệt.

### Chưa hoàn tất về chất lượng

- 425/1.000 câu abstain và 384 câu fail citation verification;
- 100% câu chạm context budget;
- answer median 92 ký tự, ngắn hơn nhiều so với reference train median 1.410;
- approved reranker chưa active trong M43;
- chưa chốt leakage-safe train/dev strategy;
- chưa có local scorer parity tuyệt đối vì scorer chính thức không pin exact
  NLTK/WordNet dependency/resource versions;
- chưa chốt final model stack, GPU image và giới hạn runtime BTC.

Đọc hậu kiểm và backlog trước khi sửa:

- `docs/16-M43-BASELINE-POSTMORTEM.md`;
- `docs/17-TEAM-IMPROVEMENT-BACKLOG.md`.

### Ràng buộc BTC phải nhớ

- chỉ dùng dữ liệu chính thức của BTC;
- được preprocessing, indexing, retrieval và fine-tuning trên dữ liệu BTC;
- cấm synthetic data, kể cả sinh từ dữ liệu BTC;
- Codabench là nền tảng nộp Task 2;
- chỉ model đã đăng ký và được BTC duyệt mới được dùng;
- không dùng lại corpus hoặc artifact AIO trong competition runtime.

## 4. Kiến trúc theo lớp

| Lớp | Package chính | Vai trò |
|---|---|---|
| Competition boundary | `competition/uit_dsc_2026` | Đọc raw field BTC, audit, map schema, batch và đóng gói submission |
| Unified contracts | `schemas`, `contracts` | Ngôn ngữ chung giữa các module và ranh giới backend |
| Configuration | `configuration` | Tất cả đường dẫn, model, top-k, timeout, device và policy |
| Offline processing | `offline` | Clean, parse, chunk và validate document |
| Indexing | `indexing` | Persist/load BM25, vector và graph artifacts |
| Retrieval | `retrieval`, `reranking`, `embeddings` | Tìm candidate, fuse, rerank và mở rộng graph |
| Generation | `generation` | Chọn context, sinh answer, kiểm tra grounding/citation |
| Orchestration | `tools`, `agent` | Tool registry đóng và workflow retry có giới hạn |
| Composition | `runtime` | Ghép backend cụ thể thành offline/online runtime |
| Serving | `serving` | CLI, API, UI và chuyển request thành unified query |
| Evaluation | `evaluation` | Metric, provenance, report, regression gate và model comparison |
| Observability | `observability` | Logging có trace/query/document/chunk/latency/error context |

Quy tắc quan trọng: core chỉ biết unified schema và Protocol. Tên raw của BTC
`id`, `name`, `link`, `passage` chỉ được biết ở competition adapter.

Đọc contract đầy đủ tại `docs/13-UIT-DSC-2026-DATA-CONTRACT.md`. Lưu ý ví dụ BTC
dùng numeric context ID, `name` dạng slug và passage có xuống dòng/Unicode; adapter
phải chuẩn hóa ID sang string, còn core không được biết kiểu raw. `link` chỉ dùng
làm provenance, không phải quyền crawl thêm dữ liệu.

## 5. Pipeline offline: từ corpus BTC đến artifact

Composition root là `CompetitionOfflineBuildRuntime` trong
`runtime/competition_offline.py`. CLI:

```powershell
legal-rag-build-competition --config <config.json> --source <selected-contexts.zip>
```

### 5.1 Corpus stage

1. `UitDsc2026DataLoader` đọc ZIP hoặc thư mục `context_*.json`.
2. Loader canonicalize numeric/string ID, chấp nhận optional `name` và blank
   passage đã audit; mọi schema drift và ID collision bị từ chối.
3. `UitDsc2026ContextAdapter` ánh xạ một context thành một `LegalDocument`.
4. `UitDsc2026CorpusIngestor` tạo audit report, dataset manifest, raw-preserving
   normalized artifact và cleaned artifact.
5. `UitDsc2026PassageCleaner` chỉ chuẩn hóa Unicode/newline/whitespace và loại
   known HTML + exact TVPL notice đã audit; adapter không suy diễn metadata.

Mục đích: cô lập raw schema và chứng minh chính xác corpus byte nào được dùng.

### 5.2 Document-processing stage

`StreamingDocumentProcessor` xử lý tuần tự/bounded thay vì materialize toàn bộ
corpus trong RAM:

```text
Cleaned LegalDocument
→ LegalStructureParser
→ LegalChunker
→ LegalChunkValidator
```

- Cleaner bảo toàn Unicode, số, phủ định và marker pháp lý.
- Parser nhận diện Phần/Chương/Mục/Điều/Khoản/Điểm.
- Chunker ưu tiên `1 chunk = 1 Điều`; Điều dài mới tách theo Khoản/token.
- Chunk ID deterministic và truy ngược được document/block nguồn.

Đầu ra chính: `legal_blocks` và `legal_chunks` cùng manifest/checksum.

### 5.3 BM25 stage

`SQLiteFTS5BM25Backend` ghi lexical index xuống SQLite:

- tokenizer Unicode giữ dấu tiếng Việt, số và phủ định;
- query planner giới hạn term quá phổ biến để truy vấn full corpus ổn định;
- artifact có manifest và checksum;
- backend load lại được mà không rebuild.

BM25 mạnh khi câu hỏi chứa đúng từ khóa, số Điều, tên luật hoặc cụm pháp lý.

### 5.4 Vector stage

`VectorIndexBuilder` gọi `EmbeddingProvider` theo batch rồi persist:

- vectors float32;
- chunk metadata;
- model name, revision, dimension và provider version;
- checkpoint theo batch để resume khi GPU/session bị ngắt.

Backend tham chiếu là `NumpyVectorBackend`, hỗ trợ memory map và optional exact
GPU scorer. Dense retrieval mạnh ở câu hỏi diễn đạt khác từ ngữ với corpus.

### 5.5 Graph và validation

Schema corpus BTC hiện chưa có relationship field, nên graph competition mặc
định là zero-edge artifact hợp lệ thay vì phát minh quan hệ. Nếu BTC bổ sung
quan hệ thật, phải audit và adapter trước khi thay đổi.

`ArtifactSetValidator` cuối cùng kiểm tra:

- dataset lineage;
- schema/artifact version;
- processing config hash;
- record counts;
- checksum;
- model/revision/dimension compatibility;
- liên kết nguồn giữa chunks, BM25, vector và graph.

Runtime online từ chối artifact không tương thích thay vì âm thầm dùng tiếp.

### 5.6 Cấu trúc artifact dự kiến

```text
<artifact-root>/
├── audit/
├── normalized_documents/
├── cleaned_documents/
├── legal_blocks/
├── legal_chunks/
├── relationships/
├── bm25/
├── vector/
├── vector_serving/
├── graph/
├── dataset_manifest.json
├── build_state.json
└── build_validation.json
```

Không commit thư mục này lên GitHub.

## 6. Pipeline online: từ câu hỏi đến câu trả lời

`OnlineRuntimeFactory` trong `runtime/online.py` là composition root. Nó validate
lineage trước, sau đó load BM25, vector, graph, model providers và tool registry.

### 6.1 Request và query understanding

`ServingService` chuyển `LegalQuestionRequest` thành `RetrievalQuery` có query ID,
original/normalized question, filters, top-k, candidate-k và strategy.

`QueryUnderstandingService` phân tích deterministic:

- intent;
- tham chiếu Điều/Khoản/Điểm hoặc số văn bản;
- query variants có giới hạn;
- routing hint.

Không dùng LLM để tùy ý viết lại query. `ConservativeQueryRewriter` chỉ rewrite
có giới hạn khi Agent retry.

### 6.2 Retrieval branches

- **BM25:** lexical search trên SQLite FTS5.
- **Dense:** embed query bằng đúng provider/revision đã build vector rồi cosine
  search.
- **Hybrid:** chạy hai nhánh và dùng Reciprocal Rank Fusion.
- **Hybrid rerank:** lấy candidate đã fuse và cho cross-encoder chấm lại.
- **Graph:** bắt đầu từ seed text retrieval, mở rộng tối đa hop cấu hình.

RRF không cộng raw BM25 score với cosine score. Nó cộng contribution theo rank:

```text
contribution = 1 / (rrf_constant + rank)
```

Mỗi `RetrievalHit` giữ provenance: branch rank, raw score, RRF contribution,
reranker score, graph path và artifact version.

### 6.3 Evidence và context

`ContextBuilder` biến top hits thành `Evidence`. `EvidenceSelector` ưu tiên:

- khớp tham chiếu pháp lý trong query;
- lexical overlap;
- văn bản có khả năng áp dụng;
- giới hạn số evidence và token budget.

`RuleBasedContextGrader` quyết định context có đủ để trả lời hay không. Nếu chưa
đủ, Agent có thể đổi strategy/rewrite query và retry.

### 6.4 Generation

Generator nằm sau `AnswerGenerator` Protocol:

- `ExtractiveAnswerGenerator`: baseline không cần model sinh;
- `ModelBackedAnswerGenerator`: dùng provider Transformers hoặc endpoint tương
  thích OpenAI theo config.

Generator chỉ nhận selected evidence, phải trả tiếng Việt và dùng marker `[E1]`,
`[E2]` để khai báo evidence. Nó không được tự thêm Điều/luật không có trong
context.

### 6.5 Verification và abstention

Verification có hai tầng:

1. Rule-based claim grounding kiểm tra marker, ID, lexical support, số và phủ
   định.
2. Optional semantic verifier kiểm tra supported/contradicted khi backend được
   bật.

Nếu generation lỗi, citation sai, claim contradicted, timeout hoặc context không
đủ, workflow trả lời an toàn với `insufficient_evidence = true` thay vì đoán.

### 6.6 Agent thực sự làm gì?

`DeterministicAgentWorkflow` không phải autonomous Agent. Nó:

1. nhận danh sách route từ `DeterministicStrategyRouter`;
2. gọi retrieval tool đã đăng ký;
3. build và grade context;
4. nếu đủ thì gọi generation và verification;
5. nếu thiếu thì thử route/query khác;
6. dừng sau tối đa `max_retry = 2` lần retry.

Agent không được tải dataset, build index, gọi web hoặc dùng tool ngoài registry.

## 7. API và UI

Khởi động:

```powershell
legal-rag-serve --config <config.json>
```

Với default prefix `/api/v1`:

| Endpoint | Chức năng |
|---|---|
| `GET /api/v1/health` | Trạng thái runtime và artifact |
| `POST /api/v1/retrieve` | Chỉ chạy retrieval, trả ranked chunks |
| `POST /api/v1/answer` | Chạy Agent RAG và trả `AnswerResponse` |
| `GET /ui` | UI chẩn đoán same-origin |
| `GET /docs` | OpenAPI docs khi được bật |

UI chỉ gọi API; nó không truy cập backend/index trực tiếp. Đây là UI chẩn đoán,
không phải giao diện production.

## 8. Batch inference và file nộp Codabench

### 8.1 Chạy/resume batch

```powershell
legal-rag-batch --config <config.json> --questions <questions.json> --output <batch-dir>
```

Batch lưu:

- `results.jsonl`: full internal `AnswerResponse` theo từng ID;
- `batch_state.json`: checkpoint atomic;
- `manifest.json`: chỉ xuất hiện khi đủ mọi câu hỏi đúng source order.

Nếu session ngắt, chạy lại đúng source/config/code/output để resume. Nếu checksum
khác, hệ thống fail closed.

### 8.2 Đóng gói submission

```powershell
legal-rag-submit --questions <questions.json> --batch <batch-dir> --output <dir>/submission.zip
```

ZIP chỉ có `submission.json`. Contract thực tế đã xác minh từ scorer:

```json
{
  "question_id": {
    "answer": "Câu trả lời"
  }
}
```

Không thêm citation, trace, warning hoặc manifest làm field nộp. Formatter kiểm
tra đủ ID, đúng thứ tự, checksum và không overwrite output cũ.

### 8.3 Scorer warm-up đã quan sát

Source scorer BTC checksum
`4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`
xác nhận:

- dùng object `.items()` thay vì array như hướng dẫn ban đầu;
- METEOR dùng whitespace `split()` rồi gọi NLTK `meteor_score` defaults;
- ROUGE-L dùng vendored `rouge_score`, lowercase và chỉ giữ ASCII `[a-z0-9]`;
- cả hai metric được arithmetic macro mean;
- PyVi đã bị comment và không được dùng;
- scoring container từng thiếu NLTK `wordnet`, khiến METEOR crash với câu trả
  lời có token chưa exact-match; BTC đã sửa và submission sau đó đã được chấm.

Đây từng là lỗi hạ tầng BTC, không phải lý do để sửa output contract.
Local scorer vẫn chỉ là diagnostic vì tokenizer/cách match khác và scorer ZIP
không pin NLTK/WordNet bytes. Xem
[`15-OFFICIAL-SCORING-CONTRACT.md`](15-OFFICIAL-SCORING-CONTRACT.md).

## 9. Evaluation và chọn model

Evaluation CLI:

```powershell
legal-rag-evaluate --config <config.json> --benchmark <cases.jsonl> `
  --benchmark-manifest <manifest.json> --output <new-report-dir>
```

So sánh nhiều run:

```powershell
legal-rag-compare --comparison <comparison.json> --output <new-report-dir>
```

Evaluation hỗ trợ retrieval metrics, generation metrics, latency, failure và
resource provenance. Benchmark phải có manifest ghi checksum và trust status:
diagnostic, human-reviewed hoặc competition-official. Diagnostic benchmark
không được dùng để tuyên bố model chiến thắng.

Model selection phải so sánh cùng benchmark, corpus lineage, cutoffs và config.
Mọi model open-source dùng cho cuộc thi phải đăng ký tên + URL với BTC khi Form
được phát hành.

## 10. Configuration: thay đổi hành vi ở đâu?

`ApplicationConfig` ghép các nhóm sau:

- `artifacts`: root và tên thư mục artifact;
- `competition`: official-only lineage policy;
- `offline`: cleaner/parser/chunker/BM25/embedding/vector/graph build;
- `online`: retrieval, query understanding, reranker, generation, verifier,
  Agent và startup validation;
- `serving`: host, port, API prefix, UI và request limits;
- `evaluation`: strategy, cutoffs và candidate limit;
- `logging`: level và format.

File mẫu là `configs/baseline.example.json`. Đây là template, không phải config
đã sẵn sàng cho artifact thật. Không hard-code path/model/device/top-k trong core.

## 11. Schemas, contracts và exceptions

### Schemas

`schemas/` chứa dữ liệu đi qua module. Ví dụ:

- `LegalDocument` → `LegalBlock` → `LegalChunk`;
- `RetrievalQuery` → `RetrievalHit` → `RetrievalResponse`;
- `Evidence` → `ContextGrade` → `AnswerResponse`;
- `AgentState`, `RetrievalHistoryItem`;
- manifests, validation, competition batch/submission và evaluation records.

Schema giúp lỗi xuất hiện ngay tại boundary thay vì lan truyền dict không rõ
cấu trúc.

### Contracts

`contracts/` dùng `typing.Protocol` cho backend có thể thay thế: BM25, embedding,
vector, reranker, graph, generator, grader, verifier, Agent và evaluator. Core
không phụ thuộc trực tiếp một model/database cụ thể.

### Exceptions

`exceptions.py` phân loại configuration, data validation, artifact compatibility,
backend initialization, retrieval, model, timeout và external-service errors.
API chuyển chúng thành response an toàn, không lộ stack trace/secret.

## 12. Logging và cách debug

Production code dùng standard-library logging, không dùng `print`. Context log có
thể mang:

```text
trace_id, query_id, document_id, chunk_id, strategy,
latency_ms, error_type
```

Khi query chậm, tìm lần lượt:

1. `embedding_model_initialized` — cold model load;
2. `dense_vector_search_completed` — vector compute;
3. `hybrid_retrieval_completed` — retrieval tổng;
4. `cross_encoder_rerank_completed` — reranker;
5. `transformers_chat_completion_completed` — generator;
6. `agent_workflow_completed` — end-to-end.

Cold request có thể lâu do lazy model loading; warm request mới phản ánh latency
ổn định.

## 13. Cách tìm code khi nhận task

| Muốn sửa | Bắt đầu đọc |
|---|---|
| Raw data BTC | `competition/uit_dsc_2026/loader.py`, `raw_schema.py`, `context_adapter.py` |
| Build official corpus | `runtime/competition_offline.py` |
| Clean/parse/chunk | `offline/cleaning`, `offline/parsing`, `offline/chunking` |
| BM25/vector/graph | `indexing/` |
| Retrieval/fusion | `retrieval/` và `reranking/` |
| Prompt/generation | `generation/model_generator.py`, provider tương ứng |
| Citation/claims | `generation/claim_grounding.py`, `citation_verifier.py`, `semantic_verifier.py` |
| Agent retry/routing | `agent/workflow.py`, `router.py`, `query_rewriter.py` |
| API/UI | `serving/api.py`, `query_service.py`, `ui.py` |
| Config | `configuration/` và `configs/baseline.example.json` |
| Schema | `schemas/` và `docs/07-UNIFIED-SCHEMA.md` |
| Evaluation | `evaluation/` |
| Submission | `competition/uit_dsc_2026/batch_inference.py`, `submission.py` |

## 14. Quy trình làm việc cho thành viên

```powershell
git clone https://github.com/Chef221/legal-agentic-rag.git
cd legal-agentic-rag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Trước khi sửa:

1. đọc `AGENTS.md` và docs liên quan;
2. tạo branch ngắn theo task;
3. xác định input/output schema và boundary;
4. không dùng dữ liệu ngoài BTC;
5. thêm unit test nhỏ, không download model/network mặc định.

Trước khi mở PR:

```powershell
python -m pytest
python -m compileall -q src tests
python -m pip check
git diff --check
```

Không commit dataset, artifact, model, token, `.env`, cache hoặc log.

## 15. Lộ trình cải thiện sau baseline M43.1

1. Giữ M43.1 làm control, không sửa config/output cũ.
2. Chốt leakage-safe train/dev manifest và evaluator.
3. Tạo retrieval/error diagnostics theo từng question type.
4. Benchmark hybrid với approved reranker trên cùng dev split.
5. Đo context token allocation và giảm duplicate evidence.
6. Benchmark prompt/output length, không đồng thời thay retrieval.
7. Thêm claim-level repair hoặc grounded fallback, không tắt verifier.
8. Chỉ fine-tune bằng official train sau khi split/evaluator được duyệt.
9. Chạy full public candidate mới đúng một lần sau local gate.
10. Ghi score, commit, config, checksum và error rates vào experiment ledger.

## 16. Những hiểu lầm cần tránh

- Có official index không có nghĩa retrieval đã đạt chất lượng tốt.
- `warmup.json` không phải corpus.
- Agent không thay thế retrieval; nó chỉ điều phối module đã kiểm thử.
- Citation nội bộ giúp grounding nhưng không được đưa thành field submission.
- Điểm local không mặc nhiên bằng điểm Codabench.
- Model mạnh hơn không tự động tốt hơn nếu retrieval/context sai.
- Không được tạo synthetic QA/hard negative dưới bất kỳ hình thức nào cho cuộc
  thi hiện tại.
- Không tái sử dụng AIO artifact dù format có vẻ tương thích.

## 17. Thứ tự tài liệu nên đọc tiếp

1. `docs/00-START-HERE.md` — điểm vào và cách nhận việc.
2. `docs/16-M43-BASELINE-POSTMORTEM.md` — score và root cause.
3. `docs/17-TEAM-IMPROVEMENT-BACKLOG.md` — workstream.
4. File này — bản đồ thực hành.
5. `docs/14-SYSTEM-ARCHITECTURE-REFERENCE.md` — I/O as-built.
6. `docs/04-SYSTEM-ARCHITECTURE.md` — boundary kiến trúc.
7. `docs/05-OFFLINE-PIPELINE.md` và `docs/06-ONLINE-PIPELINE.md`.
8. `docs/07-UNIFIED-SCHEMA.md` — field contract.
9. `docs/10-COMPETITION-ADAPTATION.md` và compliance/scoring docs.
10. `docs/08-DESIGN-DECISIONS.md`, `docs/09-IMPLEMENTATION-PLAN.md`.
