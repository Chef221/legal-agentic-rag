# 02. Problem Definition

## 1. Problem Statement

Hệ thống nhận một câu hỏi pháp luật bằng tiếng Việt và thực hiện hai
nhiệm vụ chính:

1. Truy xuất các căn cứ pháp luật liên quan.
2. Sinh câu trả lời dựa trên các căn cứ đã truy xuất.

Agentic RAG là cơ chế điều phối hệ thống, không phải một bài toán riêng.

---

## 2. System Input

Input tối thiểu:

```json
{
  "question": "Câu hỏi pháp luật bằng tiếng Việt"
}
```

Các trường có thể được bổ sung trong tương lai:

```json
{
  "question": "...",
  "requested_date": null,
  "legal_domain": null,
  "jurisdiction": null,
  "preferred_answer_format": null,
  "user_context": null
}
```

Những trường mở rộng trên chưa thuộc baseline đầu tiên.

---

## 3. Task 1: Legal Evidence Retrieval

### 3.1 Input

Task retrieval nhận:

- câu hỏi người dùng;
- legal chunks;
- metadata văn bản;
- BM25 index;
- vector index;
- legal graph;
- retrieval configuration.

### 3.2 Output

Output là một danh sách các legal chunks được xếp hạng:

```json
{
  "query": {
    "query_id": "query-001",
    "original_question": "...",
    "normalized_question": "...",
    "top_k": 10
  },
  "strategy": "hybrid_rerank",
  "hits": [
    {
      "chunk_id": "doc-001::article-15::chunk-0",
      "document_id": "doc-001",
      "rank": 1,
      "score": 0.91,
      "strategy": "hybrid_rerank",
      "text": "...",
      "metadata": {}
    }
  ]
}
```

### 3.3 Retrieval Objectives

Retrieval cần đáp ứng:

- tìm đúng quy định pháp luật;
- tìm đủ evidence cần thiết;
- hỗ trợ câu hỏi cần nhiều văn bản;
- giữ metadata để citation;
- giữ thông tin hiệu lực;
- ưu tiên evidence có liên quan cao;
- hạn chế tài liệu nhiễu;
- truy xuất được câu hỏi diễn đạt đời thường;
- truy xuất được câu hỏi chứa thuật ngữ pháp lý chính xác.

---

## 4. Task 2: Grounded Legal Answer Generation

### 4.1 Input

Answer generation nhận:

- câu hỏi gốc;
- selected evidence;
- metadata của evidence;
- trạng thái hiệu lực nếu có;
- generation configuration.

### 4.2 Output

```json
{
  "question": "...",
  "answer": "...",
  "citations": [
    {
      "evidence_id": "E1",
      "chunk_id": "...",
      "document_id": "...",
      "article_number": "15"
    }
  ],
  "insufficient_evidence": false,
  "warnings": [],
  "trace_id": "trace-001"
}
```

### 4.3 Generation Constraints

Generator phải tuân thủ:

- chỉ sử dụng thông tin trong evidence;
- không tự tạo tên văn bản;
- không tự tạo số Điều hoặc số Khoản;
- không tạo citation không tồn tại;
- không khẳng định khi evidence chưa đủ;
- phải nói rõ khi không tìm thấy căn cứ;
- phải cảnh báo nếu evidence có dấu hiệu hết hiệu lực;
- không được che giấu sự không chắc chắn.

---

## 5. Agent Role

Agent chịu trách nhiệm:

- phân tích câu hỏi;
- lựa chọn retrieval strategy;
- gọi retrieval tool;
- quan sát kết quả;
- đánh giá context;
- quyết định dừng hoặc retry;
- rewrite query khi cần;
- chuyển sang strategy khác khi cần;
- gọi answer generator;
- gọi citation verifier;
- trả về trace có thể kiểm tra.

Agent không chịu trách nhiệm:

- tải dataset trong lúc trả lời;
- clean HTML trong lúc trả lời;
- chunk lại corpus;
- build lại index;
- chỉnh sửa raw corpus;
- tự cập nhật metadata pháp lý;
- tự xác nhận hiệu lực ngoài dữ liệu;
- tự thay đổi cấu hình production.

---

## 6. Fixed Baseline

Trước khi xây Agent, hệ thống phải hỗ trợ trực tiếp:

```text
BM25 retrieval
Dense retrieval
Hybrid RRF retrieval
Hybrid retrieval + reranker
Graph-expanded retrieval
```

Luồng fixed baseline:

```text
Question
→ BM25 + Dense
→ RRF
→ Candidate Top-N
→ Cross-Encoder Reranker
→ Final Top-K
→ Context Builder
→ Answer Generator
→ Citation Verifier
```

Fixed baseline phải chạy được mà không cần Agent.

---

## 7. Success Criteria

### Retrieval

- evidence đúng xuất hiện trong top-k;
- ranking có tính liên quan;
- metadata đầy đủ;
- kết quả có thể truy ngược về corpus;
- strategy và score được ghi lại;
- latency được đo.

### Generation

- trả lời đúng trọng tâm;
- câu trả lời phù hợp với evidence;
- không có unsupported claim rõ ràng;
- citation tồn tại;
- citation trỏ đúng evidence;
- hệ thống biết từ chối khi thiếu căn cứ.

### Agent

- chọn tool hợp lý;
- không lặp vô hạn;
- có giới hạn retry;
- trace rõ ràng;
- không gọi tool ngoài phạm vi;
- không bỏ qua verifier.

---

## 8. Current Evaluation Limitation

Dữ liệu BTC hiện xác nhận answer-level references nhưng chưa cung cấp
retrieval relevance labels trong warm-up.

Warm-up hiện tại không phải gold retrieval benchmark hoàn chỉnh dạng:

```text
question
→ relevant evidence
→ gold answer
```

Vì vậy baseline có thể được xây và chạy inference, nhưng chưa thể đánh
giá đầy đủ các metric như:

- Recall@k;
- MRR;
- NDCG;
- answer correctness;
- citation recall;
- citation precision.

Các metric này sẽ được áp dụng khi có:

- dữ liệu từ Ban tổ chức;
- tập benchmark do nhóm xây dựng;
- hoặc dữ liệu có nhãn được chấp nhận theo quy chế.
