# 07. Unified Schema

## 1. Purpose

Unified schema tách core system khỏi schema thô của dữ liệu competition.

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
- `source_dataset` là required string và không có implicit default để
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
  "source_dataset": "uit-dsc-2026-task2-selected-contexts",
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
- `content_html` và `clean_text` không bị strip khi có nội dung; chỉ
  whitespace-only mới được normalize thành null.

### HtmlCleaningResult

```json
{
  "documents": [],
  "issues": [],
  "manifest": {},
  "input_document_count": 0,
  "cleaned_document_count": 0,
  "missing_content_count": 0,
  "empty_output_count": 0
}
```

`manifest.artifact_type` phải là `cleaned_documents`. Cleaner giữ đúng một
output `LegalDocument` cho mỗi input, giữ nguyên `content_html`, và chỉ gán
`clean_text` khi còn visible text sau cleaning. Mỗi input được phân loại đúng
một lần thành cleaned, missing content hoặc empty output. `issues` tái sử dụng
`AuditIssue` với `metadata.stage = "html_cleaning"`.

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

`LegalBlock` của baseline parser không chồng lấp text. Preamble hoặc văn bản
không có marker nằm trong block `document`; title nằm ở dòng kế tiếp được giữ
trong `text` và đồng thời gán vào `title` khi vượt qua heuristic bảo thủ. Table
rows kế tiếp nhau tạo block `table` và kế thừa `LegalStructure` của parent.

### DocumentParsingDiagnostic

```json
{
  "document_id": "string",
  "block_count": 0,
  "recognized_structure_count": 0,
  "source_non_whitespace_characters": 0,
  "covered_non_whitespace_characters": 0,
  "text_coverage": 0.0,
  "has_recognized_structure": false
}
```

### LegalStructureParsingResult

```json
{
  "documents": [],
  "blocks": [],
  "diagnostics": [],
  "issues": [],
  "manifest": {},
  "input_document_count": 0,
  "parsed_document_count": 0,
  "missing_clean_text_count": 0,
  "structured_document_count": 0,
  "unstructured_document_count": 0
}
```

`manifest.artifact_type` phải là `legal_blocks` và record count bằng số
block. Mỗi document có đúng một diagnostic. Parent block phải thuộc cùng
document và đứng trước child. `order_index` liên tục trong từng document.
Coverage đo trên non-whitespace characters để việc thay đổi line break không
che giấu text bị rơi.

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
  "source_dataset": "uit-dsc-2026-task2-selected-contexts",
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

`LegalChunk.metadata` của Milestone 6 ghi tối thiểu:

```json
{
  "source_block_ids": ["string"],
  "source_block_types": ["article", "clause"],
  "chunk_strategy": "article|clause_group|token_fallback|standalone_block",
  "tokenizer_name": "unicode_word_v1",
  "split_index": 0,
  "split_count": 1
}
```

### DocumentChunkingDiagnostic

```json
{
  "document_id": "string",
  "source_block_count": 0,
  "covered_block_count": 0,
  "chunk_count": 0,
  "article_unit_count": 0,
  "token_fallback_chunk_count": 0,
  "block_coverage": 0.0,
  "has_chunks": false
}
```

### LegalChunkingResult

```json
{
  "documents": [],
  "blocks": [],
  "chunks": [],
  "diagnostics": [],
  "issues": [],
  "manifest": {},
  "input_document_count": 0,
  "input_block_count": 0,
  "documents_with_chunks_count": 0,
  "documents_without_chunks_count": 0,
  "article_chunk_count": 0,
  "clause_fallback_chunk_count": 0,
  "token_fallback_chunk_count": 0,
  "standalone_chunk_count": 0
}
```

`manifest.artifact_type` phải là `legal_chunks`. Mọi source block phải xuất
hiện trong ít nhất một chunk; token overlap có thể khiến cùng source block xuất
hiện trong nhiều token-fallback chunks. Chunk IDs unique và `chunk_index` liên
tục trong từng document.

---

## 7. LegalRelationship

```json
{
  "source_document_id": "string",
  "target_document_id": "string",
  "relationship_type": "string|null",
  "raw_relationship": "string",
  "is_directed": true,
  "source_dataset": "uit-dsc-2026-task2-selected-contexts",
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

### BM25 Artifact Metadata

Milestone 7 không thêm public schema mới. `ArtifactManifest` với
`artifact_type = "bm25_index"` ghi thêm metadata có consumer rõ ràng:

- `analyzer_name`;
- `match_mode`;
- `source_artifact_type` và `source_artifact_version`;
- `source_processing_config_hash`;
- `sqlite_version`;
- sau persistence: `index_filename`, `manifest_filename`, `index_sha256`.

`BM25Backend.build` nhận `LegalChunk` cùng source `ArtifactManifest` để không
làm mất dataset revision và processing provenance. Search tiếp tục trả đúng
public `RetrievalResponse`; backend payload không trở thành unified schema.

### Vector Artifact Metadata

Milestone 8 cũng không thêm public response schema. `ArtifactManifest` với
`artifact_type = "vector_index"` ghi:

- `model_name`, pinned `model_revision`;
- `embedding_provider_name`, `embedding_provider_version`;
- `dimension`, `distance_metric`, `dtype`, `normalized_vectors`;
- `embedding_batch_size`;
- source artifact type/version/config hash;
- NumPy version;
- sau persistence: filenames và SHA-256 của vector/chunk payload.

`VectorBackend.build` nhận aligned chunks/vectors, source manifest và model
identity. Backend công khai provider name/version, `model_name`, `model_revision`
và `dimension` để `DenseRetriever` kiểm tra compatibility trước khi embed query. Dense
search vẫn trả unified `RetrievalResponse`/`RetrievalHit`/`RetrievalTrace`.

### Vector Serving Metadata

Version `0.20.4` dùng một internal persisted artifact
`artifact_type = "vector_serving_metadata"`. Đây không phải public retrieval
schema và không chứa embedding. Manifest ghi:

- backend `sqlite_chunk_metadata`;
- source vector type/version/processing hash;
- source `chunks_sha256`;
- database filename/checksum và backend schema version;
- record count bằng vector artifact.

SQLite payload chỉ chứa vector row index, JSONL byte offset, chunk ID và các
unified filter fields. Runtime vẫn dựng final `LegalChunk` từ validated source
JSONL nên public schema không thay đổi.

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
  "query_analysis": "QueryAnalysis|null",
  "query_variants": [],
  "metadata": {}
}
```

`RetrievalFilters` chỉ chứa field chung của unified schema. Empty list
nghĩa là không áp dụng filter tương ứng. Backend-specific filter không
được đưa vào core contract.

### QueryAnalysis

Milestone 19 thêm nested contract có consumer rõ ràng là router, query rewriter
và multi-query retriever:

```json
{
  "intent": "general|reference_lookup|relationship|quantitative|procedure|eligibility|obligation|prohibition|definition",
  "document_numbers": [],
  "article_numbers": [],
  "clause_numbers": [],
  "point_numbers": [],
  "year_mentions": [],
  "scope_cues": [],
  "relationship_cues": []
}
```

Mọi value phải xuất hiện trong câu hỏi normalized. Runtime tính lại analysis;
không dùng analysis do public client tự khai báo.

### QueryVariant

```json
{
  "variant_id": "qv1",
  "text": "string",
  "kind": "normalized|framing_stripped|legal_reference"
}
```

Variant ID và text phải unique. Variant không được thêm kiến thức hoặc thuật ngữ
không có trong câu hỏi. Số lượng do typed config giới hạn.

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
  "graph_path": [],
  "query_variant_contributions": []
}
```

Mỗi `QueryVariantContribution` ghi:

```json
{
  "variant_id": "qv1",
  "strategy": "bm25|dense",
  "rank": 1,
  "raw_score": 12.4,
  "rrf_contribution": 0.016
}
```

Cặp `(variant_id, strategy)` phải unique trong trace của một hit. Chỉ BM25 và
dense được phép tham gia query fusion.

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

Trong Milestone 9, hybrid RRF tái sử dụng schema hiện có, không thêm response
schema mới. Với hybrid hit:

- `score` bằng `retrieval_trace.rrf_score`;
- raw BM25/dense scores chỉ giữ để trace;
- contribution của nhánh không có chunk bằng `0.0`, rank/score nhánh đó là null;
- `rank` là final fused rank và `strategy = "hybrid"`;
- metadata của duplicate chunk phải giống nhau giữa hai branch.

Milestone 19 mở rộng cùng phép RRF sang nhiều bounded query variants:

- mỗi variant có một BM25 và một dense branch;
- `bm25_rrf_contribution` và `dense_rrf_contribution` là tổng contribution theo
  loại branch;
- `rrf_score` là tổng của hai giá trị trên;
- `RetrievalHit.score` tiếp tục bằng `rrf_score`;
- best rank/raw score của mỗi loại branch giữ trong field cũ;
- mọi per-variant contribution giữ trong typed list mới;
- duplicate chunk có payload khác nhau giữa variants bị từ chối.

Trong Milestone 10, cross-encoder tiếp tục tái sử dụng `RetrievalQuery`,
`RetrievalHit`, `RetrievalTrace` và `RetrievalResponse`; không cần nested schema
mới. Quy ước:

- `strategy = "rerank"` chỉ mô tả output trực tiếp của backend reranker;
- response fixed pipeline dùng `strategy = "hybrid_rerank"`;
- `RetrievalHit.score` bằng raw cross-encoder logit và
  `retrieval_trace.reranker_score` giữ cùng giá trị;
- các BM25/dense/RRF field trong trace không bị xóa;
- rank được đánh lại liên tục từ 1 sau rerank;
- reranker chỉ được trả subset của input candidates và không được thay đổi
  `chunk_id`, `document_id`, `text` hay `metadata`;
- `artifact_versions` của fixed response tiếp tục chỉ nguồn BM25/vector artifacts;
  model identity và revision thuộc typed config/logging, vì reranking không tạo
  persisted artifact.

Trong Milestone 11, graph retrieval tiếp tục dùng các schema hiện có:

- `LegalRelationship` là directed edge dataset-independent;
- `GraphPathStep` là một cạnh trong BFS discovery path;
- `RetrievalTrace.graph_path` chứa ordered path từ một text seed tới document
  của expanded hit;
- seed hit không giả lập graph path;
- expanded hit có `graph_hop` bằng hop lớn nhất trong path;
- final fixed response dùng `strategy = "graph"` và vẫn giữ BM25/dense/RRF cùng
  reranker provenance.

`RelationshipNormalizationResult` là output typed của bước chuẩn hoá quan hệ:

```json
{
  "relationships": [],
  "issues": [],
  "manifest": {},
  "input_count": 0,
  "rejected_count": 0,
  "duplicate_count": 0
}
```

Manifest phải có type `relationship_mapping`, record count khớp accepted
relationships và source processing hash trỏ về normalized documents.

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
  "applicability_score": 0.0,
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

### ModelAnswerDraft

`ModelAnswerDraft` là nested boundary có consumer rõ ràng: parser của
model-backed generator. Nó không phải public API response.

```json
{
  "answer": "string có marker [E#]",
  "cited_evidence_ids": ["E1"],
  "insufficient_evidence": false,
  "warnings": []
}
```

Rules:

- extra field bị từ chối;
- `answer` không rỗng;
- evidence ID dùng format `E[1-9][0-9]*` và không lặp;
- grounded draft phải có ít nhất một evidence ID;
- insufficient draft không được có evidence ID;
- generator kiểm tra từng ID thuộc selected `Evidence`;
- `Citation` đầy đủ được dựng bởi hệ thống, không deserialize trực tiếp từ model.

### ContextBuildResult

Milestone 12 thêm typed result cho consumer rõ ràng là fixed RAG service:

```json
{
  "evidence": [],
  "input_hit_count": 0,
  "selected_count": 0,
  "omitted_hit_count": 0,
  "duplicate_hit_count": 0,
  "estimated_token_count": 0,
  "truncated": false,
  "warnings": [],
  "selection_trace": []
}
```

`truncated` chỉ cho biết có whole hit bị bỏ vì count/token budget. Legal text
không bị cắt một phần. Mọi input hit phải được selected, omitted hoặc classified
là exact duplicate.

### EvidenceSelectionTrace

Milestone 20 thêm một record cho mỗi unique retrieval hit:

```json
{
  "chunk_id": "string",
  "source_rank": 1,
  "selection_rank": 1,
  "applicability": "explicit_match",
  "document_reference_match": true,
  "article_reference_match": true,
  "lexical_overlap_score": 0.5,
  "selection_score": 5.5,
  "selected": true,
  "reason": "selected"
}
```

`applicability` chỉ nhận:

- `explicit_match`;
- `compatible`;
- `unknown`;
- `inactive`;
- `reference_mismatch`.

`reason` chỉ nhận `selected`, `max_evidence` hoặc `token_budget`.
Trace phải phân loại mọi unique hit; selected trace order phải khớp danh sách
`Evidence`. Đây là selection provenance, không phải semantic/legal verdict.

`insufficient_evidence` là required field và không có implicit default.

### CitationVerificationResult

```json
{
  "is_valid": true,
  "valid_citations": [],
  "invalid_citations": [],
  "claim_verifications": [],
  "claim_coverage_score": null,
  "claim_level_verification_performed": false,
  "semantic_verification": null,
  "errors": [],
  "warnings": []
}
```

Milestone 21 thêm `ClaimVerification`:

```json
{
  "claim_id": "C1",
  "claim_text": "Người lao động được nghỉ 12 ngày.",
  "evidence_ids": ["E1"],
  "status": "supported",
  "lexical_support_score": 0.8,
  "numeric_match": true,
  "negation_match": true,
  "errors": []
}
```

`status` chỉ nhận `supported` hoặc `unsupported`. Unsupported claim phải có
failure reason; supported claim phải có evidence ID và không có error.
`claim_coverage_score` là tỷ lệ claim vượt deterministic grounding checks,
không phải semantic entailment probability.

Milestone 22 thêm nested `SemanticVerificationResult` chỉ khi model semantic
thực sự đã chạy:

```json
{
  "is_valid": true,
  "assessments": [
    {
      "claim_id": "C1",
      "evidence_ids": ["E1"],
      "label": "supported"
    }
  ],
  "provider_name": "transformers",
  "provider_version": "x.y.z",
  "model_name": "pinned-model-name",
  "model_revision": "pinned-revision",
  "errors": [],
  "warnings": []
}
```

Untrusted model draft chỉ gồm `claim_id` và `label`. `evidence_ids` trong result
được gắn lại từ `ClaimVerification`, không đọc từ model output. `label` chỉ nhận
`supported`, `contradicted` hoặc `insufficient`; mọi claim phải xuất hiện đúng
một lần và đúng thứ tự.

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

Milestone 14 bổ sung terminal contract:

```json
{
  "state": {},
  "response": {},
  "stop_reason": "answer_verified",
  "total_latency_ms": 1.0
}
```

`AgentRunResult` bắt buộc `trace_id`, answer, citations và retry count của
`AnswerResponse`/`AgentState` nhất quán. `AgentStopReason` là closed enum gồm:
`answer_verified`, `max_retry_reached`, `no_new_strategy`,
`non_retryable_tool_error`, `timeout`, `generation_failed` và
`citation_verification_failed`.

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

### DatasetAuditReport

`DatasetAuditReport` là persisted output có version của Milestone 2:

```json
{
  "schema_version": "1.0",
  "audit_config_hash": "string",
  "dataset_manifest": {},
  "created_at": "datetime",
  "components": {
    "metadata": {
      "component": "metadata",
      "total_records": 0,
      "unique_ids": 0,
      "duplicate_ids": 0,
      "empty_ids": 0,
      "malformed_ids": 0,
      "field_profiles": []
    }
  },
  "joins": {
    "metadata_with_content": 0,
    "metadata_without_content": 0,
    "orphan_content_ids": 0,
    "metadata_with_multiple_content_records": 0,
    "invalid_relationship_sources": 0,
    "invalid_relationship_targets": 0
  },
  "effect_status_distribution": {},
  "relationship_distribution": {},
  "issues": []
}
```

Mỗi `AuditFieldProfile` ghi `field_name`, `present_count`, `null_count` và
`observed_types`. `ComponentAuditSummary` dùng ID đơn cho metadata/content
và composite edge key cho relationships. Các report schema này nằm trong
`schemas/auditing.py` vì có consumer thực tế là audit service và report
writer.

---

## 18.1 Milestone 13 Tool Schemas

`schemas/tools.py` chứa các contract có consumer trực tiếp là closed registry:

- `ContextGradingInput`;
- `AnswerGenerationInput`;
- `CitationVerificationInput`;
- `ToolDescriptor`;
- `ToolInvocationRequest`;
- `ToolError`;
- `ToolInvocationResult`;
- closed enums `ToolName` và `ToolErrorType`.

Retrieval tools dùng trực tiếp `RetrievalQuery` và `RetrievalResponse`, không
tạo schema backend-specific.

`ToolInvocationResult` có đúng một trong hai trạng thái:

```json
{
  "invocation_id": "invoke-1",
  "tool_name": "hybrid_search",
  "success": true,
  "output": {},
  "error": null,
  "latency_ms": 1.0
}
```

Failure có `output = null` và một sanitized `ToolError`. Unexpected programming
exception không bị đổi thành fake successful output.

---

## 18.2 Milestone 15 Serving Schemas

`schemas/serving.py` chứa các contract có consumer trực tiếp là FastAPI API và
diagnostic UI:

- `LegalQuestionRequest`: câu hỏi, unified filters, optional top/candidate limit
  và public retrieval strategy;
- `HealthResponse` và `ArtifactHealth`: readiness cùng artifact identity không
  chứa local path;
- `ApiErrorResponse` và `ApiErrorDetail`: error envelope đã loại backend detail;
- `ServiceStatus`: closed readiness state.

Serving chuyển `LegalQuestionRequest` thành `RetrievalQuery`; raw dataset field
không xuất hiện ở API boundary.

---

## 18.3 Milestone 16 Evaluation Schemas

`schemas/evaluation.py` chứa:

- `EvaluationCase`: question, target granularity, graded stable IDs và optional
  generation labels;
- `RetrievalCaseMetrics` và `GenerationCaseMetrics`; generation metrics include
  nullable diagnostic `meteor` and `rouge_l` only when a reference answer exists;
- `EvaluationCaseResult` với compact retrieved IDs và sanitized failure;
- `LatencySummary`, `EvaluationResourceUsage`, `EvaluationSummary`;
- `EvaluationRunResult` cho persistence.

Metric không có nhãn giữ giá trị `null`; framework không biến “không đo được”
thành điểm `0`.

---

### 18.4.1 Milestone 23 Evaluation Comparison Schemas

M23 emits `EvaluationSummary` schema version `1.1` and extends it with pinned
dataset lineage, a sanitized runtime
configuration SHA-256, and component provenance for embedding, BM25, vector,
reranker, generator, semantic verifier, retrieval policy, and package versions.
Local artifact paths, endpoint URLs, API-key environment names, prompts, and
legal content are not persisted in this provenance.

The comparison contract adds:

- `EvaluationMetricDirection`: `maximize` or `minimize`;
- `EvaluationSelectionMode`: `pareto_only` or explicitly requested
  `lexicographic`;
- `EvaluationObjective`: one named metric, direction, and optional eligibility
  threshold;
- `EvaluationCandidateResult`: projected objective values, reproducibility
  identity, eligibility reasons, and dominance relationships;
- `EvaluationComparisonReport`: exact benchmark identity, candidate set,
  Pareto frontier, optional selected candidate, and warnings.

`EvaluationResourceUsage` retains portable CPU/Python observations and optional
accelerator name/peak allocated bytes when an already-loaded runtime exposes a
CUDA-compatible API. Missing accelerator telemetry remains `null`.

Reports from different benchmark bytes, case counts, retrieval cutoffs, or
dataset revisions are not comparable. Missing labels remain missing and cannot
silently become zero or a model-selection score.

---

### 18.4.2 Milestone 24 Benchmark Governance Schemas

M24 emits `EvaluationSummary` schema version `1.2` and adds:

- `EvaluationBenchmarkLabelStatus`: `diagnostic`, `human_reviewed`, or
  `competition_official`;
- `EvaluationBenchmarkManifest`: benchmark/version identity, exact benchmark
  SHA-256, pinned dataset name/revision, case count, target granularities, and
  declared label provenance;
- `benchmark_manifest_sha256`, `benchmark_version`,
  `benchmark_label_status`, verification timestamp, and label-provenance
  reference in each evaluation summary;
- optional `maximum_regression` on each objective;
- `baseline_candidate_id` and benchmark-manifest identity in each comparison
  report.

Trusted label statuses require a timezone-aware verification timestamp and a
non-empty provenance reference. This is an auditable declaration, not an
independent legal-quality certification. Diagnostic benchmarks may produce
metrics and Pareto frontiers, but cannot produce a selected winner.

### 18.4.3 Milestone 29 Competition Text Metrics

M29 emits `EvaluationSummary` schema version `1.3`.
`GenerationCaseMetrics` adds nullable `meteor` and `rouge_l`; they are populated
only from an explicit reference answer. Reports include a warning that local
tokenization and parameters are diagnostic rather than official Codabench
equivalence.

---

## 18.4 Milestone 17 Build Validation Schema

`schemas/build_validation.py` chứa `BuildValidationReport`, có consumer trực tiếp
là validation CLI và online startup validation:

- dataset manifest đã đọc được, nếu tồn tại;
- map `ArtifactValidationResult` theo `ArtifactType.value`;
- expected raw component counts lấy từ typed config;
- `is_full_corpus` chỉ đúng khi revision được pin, counts khớp và không có
  sample warning;
- `is_valid` phản ánh cả top-level completeness/lineage và từng artifact result;
- passed checks, sanitized errors và warnings.

`OfflineBuildState` là recovery identity có consumer trực tiếp là offline

- schema version `1.1`;
- canonical application config SHA-256;
- code version;
- timezone-aware creation time.

State không chứa raw record, legal content, model output hoặc secret.
Canonical hashing sắp xếp mapping key và set/frozenset nhưng giữ thứ tự
list/tuple. State schema `1.0` bị từ chối vì digest cũ không đủ dữ liệu để
migrate hoặc chứng minh config tương đương một cách an toàn.

Streaming execution từ version `0.20.0` không thay field của `LegalDocument`,
`LegalBlock`, `LegalChunk` hoặc `ArtifactManifest`. `ParsedLegalDocument`,
`ChunkedLegalDocument` và `VectorBuildBatch` chỉ là bounded in-process transfer
objects có consumer trực tiếp; chúng không phải persisted unified schemas.
Manifest vector mới có metadata `chunk_order = "source_artifact_order"` để mô
tả alignment giữa chunk JSONL và vector rows.

`VectorBuildCheckpoint` là persisted recovery schema có consumer trực tiếp là
NumPy vector artifact store:

- `schema_version = "1.0"`;
- `artifact_manifest: ArtifactManifest` — identity và final count dự kiến;
- `next_offset: int` — số row/chunk đã commit;
- `chunks_byte_count: int` — byte boundary an toàn của `chunks.jsonl`;
- `updated_at: datetime` — timezone-aware checkpoint time.

Schema này không chứa embedding text ngoài các unified `LegalChunk` đã được ghi
trong payload, không phải online contract và không được publish như final vector
artifact. `next_offset` không được vượt `artifact_manifest.record_count`.

---

## 19. Schema Evolution

Mọi persisted schema phải có version.

Khi thay đổi schema:

- cập nhật file này;
- cập nhật design decision nếu thay đổi kiến trúc;
- thêm migration hoặc rebuild strategy;
- không silently load artifact không tương thích.

---

## 20. Schema Module Layout

```text
schemas/
├── competition.py
├── competition_execution.py
├── legal_documents.py
├── legal_relationships.py
├── manifests.py
├── retrieval.py
├── answering.py
├── agent_state.py
├── auditing.py
├── cleaning.py
├── parsing.py
├── chunking.py
├── build_validation.py
├── evaluation.py
├── tools.py
└── serving.py
```

Các schema bổ sung có consumer rõ ràng được đặt cùng domain:

- `ArtifactValidationResult` trong `manifests.py`;
- `RetrievalFilters`, `GraphPathStep` và `RetrievalTrace` trong
  `retrieval.py`;
- `CitationVerificationResult` trong `answering.py`;
- `RetrievalHistoryItem` trong `agent_state.py`.
- `AgentRunResult` và `AgentStopReason` trong `agent_state.py`.
- `HtmlCleaningResult` trong `cleaning.py`.
- `DocumentParsingDiagnostic` và `LegalStructureParsingResult` trong
  `parsing.py`.
- `DocumentChunkingDiagnostic` và `LegalChunkingResult` trong `chunking.py`.
- `ContextBuildResult` trong `answering.py`.
- tool input, descriptor, invocation và safe-error schemas trong `tools.py`.
- public request, health và safe-error schemas trong `serving.py`.
- benchmark, metric và report schemas trong `evaluation.py`.
- complete artifact-set report trong `build_validation.py`.
- typed partial-build recovery state trong `build_validation.py`.
- typed resumable vector-build checkpoint trong `build_validation.py`.
- official build, batch, and submission contracts trong
  `competition_execution.py`.

---

## 21. Competition Boundary Records

Các schema này là output ổn định của adapter, chưa phải submission schema.

### 21.1 `CompetitionQuestion`

```text
question_id: str, required
question: str, required
reference_answer: str | null, default null
```

Text được giữ nguyên; validator chỉ từ chối chuỗi rỗng/whitespace-only.
`reference_answer = null` hỗ trợ public/private question-only files mà không
phát minh label.

### 21.2 `CompetitionContext`

```text
context_id: str, required
title: str | null, default null
source_url: str, required
passage: str, required
```

Raw fields `id`, `name`, `link`, `passage` chỉ được map trong competition
adapter. Schema này không tự suy diễn document number, Điều, Khoản, hiệu lực
hoặc relationship. M26 map một context sang một `LegalDocument`, sau đó parser
và chunker reusable mới tạo cấu trúc/chunk.

Overview BTC có ví dụ raw `id` là JSON integer. Unified `context_id` vẫn luôn là
string: competition adapter chịu trách nhiệm canonicalize raw non-negative
integer hoặc non-blank string thành string deterministic sau audit. Boolean,
float, null, blank và canonicalization collision phải bị từ chối. Đây là raw
boundary rule, không thay đổi type của unified schema. Archive thật xác nhận
`name` thiếu ở 1.125/8.532 records và `passage` rỗng ở 20 records; hai trường
hợp này được giữ minh bạch, không phát minh title hoặc content.

### 21.3 `CompetitionCorpusAuditReport`

Audit report giữ dataset/revision, source kind, member/record/unique counts,
content/blank/missing-title counts, raw/cleaned character totals, passage bounds,
duplicate title/URL/passage counts, HTML/boilerplate/modified counts, passed checks
và warnings. Contract bắt buộc một member tạo đúng một unique context record.

### 21.4 `CompetitionCorpusIngestionResult`

```text
normalized_documents: list[LegalDocument]
cleaned_documents: list[LegalDocument]
dataset_manifest: DatasetManifest
normalized_manifest: ArtifactManifest(normalized_documents)
cleaned_manifest: ArtifactManifest(cleaned_documents)
audit: CompetitionCorpusAuditReport
```

`normalized_documents` giữ raw passage hợp lệ sau schema mapping.
`cleaned_documents` là output của `UitDsc2026PassageCleaner`: NFC/newline và
whitespace normalization, known HTML presentation markup removal và exact
audited TVPL Pro notice removal. Hai danh sách phải cùng count/order/ID; mọi
documents, manifests và audit phải cùng official dataset/revision.

### 21.5 Competition execution records

`CompetitionBuildState` records the exact official source revision,
application-config hash, code version, timestamps, and an ordered prefix of
`CompetitionBuildStage` values. `CompetitionOfflineBuildResult` returns that
identity with the final `BuildValidationReport`.

`CompetitionBatchRecord` maps one official `question_id` to one full
`AnswerResponse`. `CompetitionBatchState` records the ordered completed IDs for
resume. `CompetitionBatchManifest` is written only after exact completeness and
pins the question bytes, config, code, record count, and result JSONL checksum.
These are internal execution contracts and are not the Codabench submission
schema.

### 21.6 Official submission records

`CompetitionSubmissionItem` is the exact item written to `submission.json`:

```text
id: str, required
answer: str, required
```

Unknown fields are rejected. In the published JSON, `id` is the key of a
non-empty root object and each value is exactly `{"answer": str}`. Root keys
must match the official question source once and in source order. The internal
item schema retains `id` so validation and batch comparison remain typed.
`CompetitionSubmissionResult` is not placed in the ZIP; it only returns the
output path, question count, payload SHA-256, and archive SHA-256 to the local
caller. The official ZIP contains only `submission.json`.

### 21.7 Warm-up score records

`CompetitionWarmupCaseScore` contains only `question_id`, `exact_match`,
`meteor`, and `rouge_l`. `CompetitionWarmupScoreReport` schema `1.0` contains:

- timezone-aware creation time and code version;
- reference-source, submission-archive, and submission-JSON SHA-256;
- question count and mean exact match/METEOR/ROUGE-L;
- ordered content-free case scores;
- explicit diagnostic/non-equivalence warnings.

The report does not contain question text, gold answer text, prediction text,
citations, or model prompts.

D080/source analysis does not change this schema: M29/M30 values still use the
local diagnostic algorithm and remain non-equivalent to the BTC scorer described
in `docs/15-OFFICIAL-SCORING-CONTRACT.md`.
