# 07. Unified Schema

## 1. Purpose

Unified schema tách core system khỏi schema thô của dataset AIO.

Raw field names chỉ được sử dụng trong dataset adapter.

Các module phía sau chỉ sử dụng schema trong tài liệu này.

Schema trong file này là logical contract.

Milestone 1 triển khai logical contract bằng Pydantic v2 trên Python
3.11. Backend contract sử dụng `typing.Protocol` và nằm ngoài schema
package.

---

## 2. Common Conventions

- Tất cả ID nội bộ dùng string.
- Empty string được normalize thành null khi phù hợp.
- Date dùng ISO 8601.
- Datetime dùng ISO 8601 có timezone.
- Metadata không hiểu phải được giữ trong `raw_metadata` hoặc `metadata`.
- Không làm mất provenance.
- Schema phải hỗ trợ JSON serialization.
- Schema phải có version khi persistence.
- Unknown top-level fields bị từ chối; extension data phải nằm trong
  `metadata` hoặc `raw_metadata`.
- Field required không có implicit default.
- Optional collection dùng default factory, không dùng mutable default.
- `source_dataset` là required string và không mặc định thành `aio` để
  giữ dataset independence.

---

## 3. LegalDocument

```json
{
  "document_id": "string",
  "title": "string|null",
  "document_number": "string|null",
  "document_type": "string|null",
  "issuance_date": "date|null",
  "effective_date": "date|null",
  "expiry_date": "date|null",
  "effect_status": "string|null",
  "issuing_authority": "string|null",
  "position_title": "string|null",
  "signer": "string|null",
  "sector": "string|null",
  "legal_field": "string|null",
  "scope": "string|null",
  "application_info": "string|null",
  "publication_date": "date|null",
  "source_url": "string|null",
  "content_html": "string|null",
  "clean_text": "string|null",
  "has_content": true,
  "source_dataset": "aio",
  "raw_metadata": {}
}
```

### Required Fields

- `document_id`;
- `has_content`;
- `source_dataset`.

### Notes

- `content_html` chỉ tồn tại ở processing layer.
- `clean_text` chỉ tồn tại sau cleaning.
- document thiếu content vẫn có thể tồn tại trong graph.

---

## 4. LegalStructure

```json
{
  "part": "string|null",
  "chapter": "string|null",
  "section": "string|null",
  "subsection": "string|null",
  "article_number": "string|null",
  "article_title": "string|null",
  "clause_numbers": ["string"],
  "point_numbers": ["string"],
  "structure_path": ["string"]
}
```

`structure_path` lưu hierarchy theo thứ tự.

Ví dụ:

```json
{
  "structure_path": [
    "Chương II",
    "Mục 1",
    "Điều 15",
    "Khoản 2"
  ]
}
```

---

## 5. LegalBlock

Schema trung gian sau legal structure parsing:

```json
{
  "block_id": "string",
  "document_id": "string",
  "block_type": "document|part|chapter|section|subsection|article|clause|point|paragraph|table|appendix",
  "block_number": "string|null",
  "title": "string|null",
  "text": "string",
  "parent_block_id": "string|null",
  "order_index": 0,
  "structure": {},
  "metadata": {}
}
```

---

## 6. LegalChunk

```json
{
  "chunk_id": "string",
  "document_id": "string",
  "chunk_index": 0,
  "text": "string",
  "search_text": "string",
  "token_count": 0,
  "structure": {},
  "document_title": "string|null",
  "document_number": "string|null",
  "document_type": "string|null",
  "issuance_date": "date|null",
  "effective_date": "date|null",
  "expiry_date": "date|null",
  "effect_status": "string|null",
  "issuing_authority": "string|null",
  "legal_field": "string|null",
  "source_url": "string|null",
  "source_dataset": "aio",
  "metadata": {}
}
```

### Required Fields

- `chunk_id`;
- `document_id`;
- `chunk_index`;
- `text`;
- `search_text`;
- `token_count`;
- `source_dataset`.

---

## 7. LegalRelationship

```json
{
  "source_document_id": "string",
  "target_document_id": "string",
  "relationship_type": "string|null",
  "raw_relationship": "string",
  "is_directed": true,
  "source_dataset": "aio",
  "metadata": {}
}
```

### Notes

- `relationship_type` là canonical label hoặc null nếu chưa map được.
- `raw_relationship` giữ nguyên giá trị nguồn.
- Nếu chưa map được canonical label, relationship phải được đánh dấu
  thay vì tự suy diễn.

---

## 8. DatasetManifest

```json
{
  "schema_version": "string",
  "dataset_name": "string",
  "dataset_revision": "string|null",
  "loaded_at": "datetime",
  "configs": [
    "metadata",
    "content",
    "relationships"
  ],
  "record_counts": {},
  "processing_config_hash": "string",
  "code_version": "string|null",
  "warnings": []
}
```

---

## 9. ArtifactManifest

```json
{
  "schema_version": "string",
  "artifact_type": "normalized_documents|cleaned_documents|legal_blocks|legal_chunks|bm25_index|embedding_output|vector_index|relationship_mapping|graph_index",
  "artifact_version": "string",
  "dataset_name": "string",
  "dataset_revision": "string|null",
  "created_at": "datetime",
  "record_count": 0,
  "processing_config_hash": "string",
  "code_version": "string|null",
  "backend": "string|null",
  "model_name": "string|null",
  "model_revision": "string|null",
  "warnings": [],
  "metadata": {}
}
```

### ArtifactValidationResult

```json
{
  "manifest": {},
  "is_valid": true,
  "checked_at": "datetime",
  "passed_checks": [],
  "errors": [],
  "warnings": []
}
```

`is_valid` phải là false khi `errors` không rỗng.

---

## 10. RetrievalQuery

```json
{
  "query_id": "string",
  "original_question": "string",
  "normalized_question": "string",
  "rewritten_question": "string|null",
  "filters": {
    "document_ids": [],
    "document_types": [],
    "legal_fields": [],
    "effect_statuses": []
  },
  "top_k": 10,
  "candidate_k": 100,
  "requested_strategy": "string|null",
  "metadata": {}
}
```

`RetrievalFilters` chỉ chứa field chung của unified schema. Empty list
nghĩa là không áp dụng filter tương ứng. Backend-specific filter không
được đưa vào core contract.

---

## 11. RetrievalHit

```json
{
  "chunk_id": "string",
  "document_id": "string",
  "rank": 1,
  "score": 0.0,
  "strategy": "bm25|dense|hybrid|hybrid_rerank|rerank|graph",
  "text": "string",
  "metadata": {},
  "retrieval_trace": {}
}
```

`retrieval_trace` có thể chứa:

```json
{
  "bm25_rank": 3,
  "bm25_score": 12.4,
  "dense_rank": 5,
  "dense_score": 0.78,
  "bm25_rrf_contribution": 0.015,
  "dense_rrf_contribution": 0.016,
  "rrf_score": 0.031,
  "reranker_score": 0.92,
  "graph_hop": null,
  "graph_path": []
}
```

Mỗi item trong `graph_path` có:

```json
{
  "source_document_id": "string",
  "target_document_id": "string",
  "relationship_type": "string",
  "hop": 1
}
```

Rank bắt đầu từ 1. Graph hop mặc định 1 và tối đa 2.

---

## 12. RetrievalResponse

```json
{
  "query": {},
  "strategy": "string",
  "hits": [],
  "latency_ms": 0.0,
  "warnings": [],
  "artifact_versions": {}
}
```

---

## 13. Evidence

```json
{
  "evidence_id": "E1",
  "chunk_id": "string",
  "document_id": "string",
  "text": "string",
  "article_number": "string|null",
  "article_title": "string|null",
  "document_title": "string|null",
  "document_number": "string|null",
  "document_type": "string|null",
  "effective_date": "date|null",
  "expiry_date": "date|null",
  "effect_status": "string|null",
  "source_url": "string|null",
  "metadata": {}
}
```

---

## 14. ContextGrade

```json
{
  "is_sufficient": true,
  "score": 0.0,
  "relevance_score": 0.0,
  "coverage_score": 0.0,
  "consistency_score": 0.0,
  "missing_aspects": [],
  "warnings": [],
  "metadata": {}
}
```

---

## 15. Citation

```json
{
  "evidence_id": "E1",
  "chunk_id": "string",
  "document_id": "string",
  "document_title": "string|null",
  "document_number": "string|null",
  "article_number": "string|null",
  "source_url": "string|null"
}
```

---

## 16. AnswerResponse

```json
{
  "question": "string",
  "answer": "string",
  "citations": [],
  "insufficient_evidence": false,
  "warnings": [],
  "retrieval_strategy": "string",
  "trace_id": "string",
  "metadata": {}
}
```

`insufficient_evidence` là required field và không có implicit default.

### CitationVerificationResult

```json
{
  "is_valid": true,
  "valid_citations": [],
  "invalid_citations": [],
  "errors": [],
  "warnings": []
}
```

Baseline result chỉ xác nhận structural và referential validity. Nó
không tuyên bố semantic claim verification.

---

## 17. AgentState

```json
{
  "trace_id": "string",
  "original_question": "string",
  "normalized_question": "string",
  "current_query": "string",
  "selected_strategy": "string|null",
  "retrieval_history": [
    {
      "attempt_number": 1,
      "query": {},
      "strategy": "hybrid_rerank",
      "response": null,
      "context_grade": null,
      "error_type": null,
      "warnings": []
    }
  ],
  "candidate_hits": [],
  "selected_evidence": [],
  "context_grade": null,
  "retry_count": 0,
  "answer": null,
  "citations": [],
  "warnings": [],
  "metadata": {}
}
```

Một `RetrievalHistoryItem` lưu đúng một retrieval attempt. Với
`max_retry = 2`, attempt number nằm trong khoảng 1 đến 3.

---

## 18. AuditIssue

```json
{
  "issue_type": "string",
  "severity": "info|warning|error",
  "record_id": "string|null",
  "message": "string",
  "raw_value": null,
  "metadata": {}
}
```

---

## 19. Schema Evolution

Mọi persisted schema phải có version.

Khi thay đổi schema:

- cập nhật file này;
- cập nhật design decision nếu thay đổi kiến trúc;
- thêm migration hoặc rebuild strategy;
- không silently load artifact không tương thích.

---

## 20. Milestone 1 Module Layout

```text
schemas/
├── legal_documents.py
├── legal_relationships.py
├── manifests.py
├── retrieval.py
├── answering.py
├── agent_state.py
└── auditing.py
```

Các schema bổ sung có consumer rõ ràng được đặt cùng domain:

- `ArtifactValidationResult` trong `manifests.py`;
- `RetrievalFilters`, `GraphPathStep` và `RetrievalTrace` trong
  `retrieval.py`;
- `CitationVerificationResult` trong `answering.py`;
- `RetrievalHistoryItem` trong `agent_state.py`.
