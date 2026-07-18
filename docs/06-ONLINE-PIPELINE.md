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

Agentic workflow chỉ được triển khai sau fixed baseline.

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