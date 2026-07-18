# 05. Offline Pipeline

## 1. Purpose

Offline pipeline chuyển dataset pháp luật thô thành các artifact phục vụ
retrieval và answer generation.

Luồng chính:

```text
Dataset
→ Load
→ Audit
→ Normalize
→ Clean
→ Parse legal structure
→ Chunk
→ Validate
→ Build indexes
→ Validate artifacts
```

---

## 2. Step 1: Load Dataset

Load riêng các config:

- metadata;
- content;
- relationships.

Yêu cầu:

- hỗ trợ sample mode;
- hỗ trợ giới hạn số record;
- hỗ trợ streaming nếu backend cho phép;
- không bắt buộc load toàn bộ vào RAM;
- ghi dataset revision;
- ghi config name;
- không sửa raw record;
- không làm cleaning trong loader;
- không thực hiện indexing trong loader.

Output:

- iterable raw metadata records;
- iterable raw content records;
- iterable raw relationship records;
- dataset manifest sơ bộ.

---

## 3. Step 2: Audit Dataset

Audit được thực hiện trước normalization.

### 3.1 Schema Audit

Kiểm tra:

- field names;
- data types;
- nested structures;
- nullability;
- unexpected fields;
- incompatible records.

### 3.2 ID Audit

Kiểm tra:

- total records;
- unique IDs;
- duplicate IDs;
- empty IDs;
- malformed IDs.

### 3.3 Join Audit

Kiểm tra:

```text
metadata.id ↔ content.id
relationships.doc_id ↔ metadata.id
relationships.other_doc_id ↔ metadata.id
```

Xuất:

- missing content records;
- orphan content records;
- invalid relationship source;
- invalid relationship target.

### 3.4 Value Audit

Kiểm tra:

- missing titles;
- missing document numbers;
- invalid dates;
- unknown effect status;
- empty content;
- duplicate content;
- relationship distribution.

Output:

- `data_audit.json`;
- `missing_content.csv`;
- `orphan_content.csv`;
- `invalid_relationships.csv`;
- `duplicate_records.csv`;
- audit logs.

---

## 4. Step 3: Normalize Documents

Raw metadata và content được chuyển thành `LegalDocument`.

Công việc:

- map raw fields;
- chuẩn hóa `document_id` thành string;
- chuẩn hóa empty string thành null;
- parse date;
- chuẩn hóa effect status;
- chuẩn hóa document type;
- chuẩn hóa source URL;
- ghép metadata với content;
- gắn `has_content`;
- giữ `raw_metadata`;
- ghi source dataset.

Không được:

- xóa raw metadata không hiểu;
- tự đoán ngày;
- tự tạo document number;
- tự suy luận effect status khi không đủ dữ liệu.

Output:

- normalized documents;
- normalization warnings;
- normalized document manifest.

---

## 5. Step 4: Clean HTML

Input:

- `content_html`.

Output:

- `clean_text`;
- optional structured blocks;
- cleaning metadata.

### 5.1 Elements to Remove

- script;
- style;
- navigation;
- repeated header;
- repeated footer;
- invisible control characters;
- tracking elements;
- unrelated page elements;
- empty HTML nodes.

### 5.2 Elements to Preserve

- document title;
- preamble;
- Chương;
- Mục;
- Tiểu mục;
- Điều;
- Khoản;
- Điểm;
- legal tables;
- signatures when relevant;
- appendices when useful;
- dates;
- amounts;
- document numbers;
- punctuation;
- negation.

### 5.3 Text Normalization

- normalize Unicode;
- normalize whitespace;
- normalize line breaks;
- decode HTML entities;
- preserve Vietnamese diacritics;
- preserve numbering;
- preserve legal markers.

Không thực hiện:

- remove diacritics;
- aggressive lowercase;
- aggressive stopword removal;
- stemming;
- removal of numbers;
- removal of negative words.

---

## 6. Step 5: Parse Legal Structure

Parser nhận `clean_text` và tạo cấu trúc phân cấp:

```text
Document
→ Part
→ Chapter
→ Section
→ Subsection
→ Article
→ Clause
→ Point
```

Parser phải xử lý:

- thiếu một hoặc nhiều cấp;
- Điều không có title;
- Khoản không đánh số;
- numbering không liên tục;
- Điểm dùng chữ cái;
- Điều dùng số La Mã hoặc số thường;
- văn bản hành chính không có cấu trúc luật chuẩn;
- annex hoặc appendix;
- table nằm trong Điều;
- malformed line breaks.

Output có thể gồm các `LegalBlock`:

```json
{
  "block_type": "article",
  "block_number": "15",
  "title": "...",
  "text": "...",
  "parent_path": {
    "chapter": "...",
    "section": "..."
  }
}
```

Parser không được tự sửa nội dung pháp lý.

---

## 7. Step 6: Legal Chunking

### 7.1 Chunking Priority

Ưu tiên:

1. một Điều là một chunk;
2. Điều quá dài thì chia theo Khoản;
3. nhóm nhiều Khoản liên tiếp nếu phù hợp;
4. Khoản quá dài mới chia theo token;
5. token split phải có overlap;
6. mỗi chunk phải giữ legal hierarchy.

### 7.2 Chunk Header

`search_text` nên chứa:

```text
Tên văn bản
Số ký hiệu
Loại văn bản
Chương hoặc Mục
Điều
Khoản
Nội dung
```

Ví dụ:

```text
Văn bản: Luật ...
Số ký hiệu: ...
Chương II: ...
Điều 15. Điều kiện ...
Khoản 1: ...
```

### 7.3 Chunk Identity

`chunk_id` phải:

- deterministic;
- unique;
- reproducible;
- truy ngược được về document;
- không phụ thuộc vào vị trí dòng trong raw dataset.

Ví dụ:

```text
{document_id}::article-{article_number}::chunk-{chunk_index}
```

Nếu không có article number:

```text
{document_id}::section-{structure_hash}::chunk-{chunk_index}
```

### 7.4 Token Split Fallback

Token split chỉ được dùng khi legal boundary không đủ.

Cần cấu hình:

- maximum token count;
- minimum token count;
- overlap token count;
- tokenizer name.

Các giá trị này chưa được hard-code trong architecture docs.

---

## 8. Step 7: Validate Chunks

Kiểm tra:

- chunk ID duy nhất;
- document ID tồn tại;
- text không rỗng;
- search text không rỗng;
- token count hợp lệ;
- legal structure hợp lệ;
- source metadata đầy đủ;
- chunk có thể truy ngược về clean document;
- tỷ lệ text coverage;
- chunk quá ngắn;
- chunk quá dài;
- duplicated chunks.

Output:

- chunk validation report;
- invalid chunks;
- chunk manifest;
- processed legal chunks.

---

## 9. Step 8: Build BM25 Index

Input:

- validated legal chunks.

Index phải lưu hoặc truy xuất được:

- chunk ID;
- searchable text;
- document ID;
- legal metadata;
- effect status;
- legal field;
- document number;
- article number.

BM25 builder phải hỗ trợ:

- deterministic build;
- persistence;
- reload;
- index manifest;
- simple smoke query;
- configurable analyzer.

BM25 index không được chứa raw HTML.

---

## 10. Step 9: Build Vector Index

Input:

- validated legal chunks.

Pipeline:

```text
LegalChunk.search_text
→ Embedding model
→ Embedding vector
→ Vector backend
```

Manifest phải ghi:

- embedding model name;
- model revision;
- embedding dimension;
- normalization policy;
- distance metric;
- batch size;
- number of indexed chunks;
- build timestamp;
- backend version.

Vector backend phải hỗ trợ:

- insert;
- batch insert;
- search;
- persistence;
- reload;
- metadata filtering;
- index version validation.

Embedding model cụ thể chưa được chốt trong tài liệu kiến trúc.

---

## 11. Step 10: Normalize Relationships

Raw relationship được chuyển thành `LegalRelationship`.

Công việc:

- normalize source ID;
- normalize target ID;
- preserve raw relationship label;
- map sang canonical relationship khi có mapping rõ;
- ghi direction;
- ghi provenance;
- loại invalid edge khỏi production graph;
- giữ invalid edge trong audit artifact.

Không được map relationship type bằng phỏng đoán không kiểm chứng.

---

## 12. Step 11: Build Graph Index

### 12.1 Nodes

Baseline graph sử dụng document nodes.

Node metadata:

- document ID;
- title;
- document number;
- document type;
- effect status;
- effective date;
- expiry date;
- legal field;
- has content.

### 12.2 Edges

Edge lưu:

- source document ID;
- target document ID;
- canonical relationship type;
- raw relationship;
- direction;
- provenance.

### 12.3 Graph Requirements

Graph backend phải hỗ trợ:

- add nodes;
- add directed edges;
- lookup document;
- traverse neighbors;
- filter by relationship;
- hop limit;
- persistence;
- reload;
- graph manifest.

---

## 13. Step 12: Validate Artifacts

Trước khi offline pipeline hoàn thành:

- document count khớp manifest;
- chunk count khớp manifest;
- BM25 index load được;
- BM25 smoke query chạy được;
- vector index load được;
- vector smoke query chạy được;
- graph load được;
- graph traversal chạy được;
- sample chunk truy ngược được về document;
- artifact config được lưu;
- backend version được lưu;
- lỗi được ghi rõ.

Không artifact nào được đánh dấu ready nếu validation thất bại.

---

## 14. Offline Outputs

Thư mục artifact logic:

```text
artifacts/
├── manifests/
├── audits/
├── normalized_documents/
├── cleaned_documents/
├── legal_structures/
├── legal_chunks/
├── bm25/
├── vector/
└── graph/
```

Tên thư mục vật lý có thể thay đổi khi scaffold được chốt.

---

## 15. Idempotency and Reproducibility

Offline pipeline phải:

- có thể chạy lại;
- không tạo duplicate artifact;
- sử dụng deterministic IDs;
- ghi processing config hash;
- ghi dataset revision;
- ghi code version;
- hỗ trợ resume khi khả thi;
- không silently overwrite incompatible artifacts.