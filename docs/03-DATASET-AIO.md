# 03. AIO Dataset

## 1. Dataset Information

Dataset hiện tại:

`th1nhng0/vietnamese-legal-documents`

Nguồn:

Hugging Face Datasets.

Dataset là nguồn dữ liệu duy nhất của baseline hiện tại.

---

## 2. Dataset Role

Dataset được sử dụng làm:

- legal document corpus;
- legal metadata source;
- relationship source;
- BM25 indexing source;
- vector indexing source;
- graph indexing source;
- retrieval evidence source;
- citation metadata source.

Dataset không được mặc định là gold QA dataset.

---

## 3. Logical Dataset Components

Dataset gồm ba nhóm dữ liệu chính:

1. Metadata.
2. Content.
3. Relationships.

Tên config và schema chính xác phải được xác minh bằng dataset loader và
data audit.

Không được dựa hoàn toàn vào mô tả tài liệu mà bỏ qua schema thực tế.

---

## 4. Metadata Component

Metadata chứa thông tin mô tả văn bản pháp luật.

Khóa định danh logic:

```text
metadata.id
```

Các trường có thể gồm:

- `id`;
- `title`;
- `so_ky_hieu`;
- `loai_van_ban`;
- `ngay_ban_hanh`;
- `ngay_co_hieu_luc`;
- `ngay_het_hieu_luc`;
- `tinh_trang_hieu_luc`;
- `co_quan_ban_hanh`;
- `chuc_danh`;
- `nguoi_ky`;
- `nganh`;
- `linh_vuc`;
- `pham_vi`;
- `thong_tin_ap_dung`;
- `nguon_thu_thap`;
- `ngay_dang_cong_bao`;
- `source_url`.

Tên trường thực tế có thể khác.

Dataset adapter chịu trách nhiệm ánh xạ raw field sang unified schema.

---

## 5. Content Component

Content chứa nội dung toàn văn của văn bản.

Khóa nối logic:

```text
content.id → metadata.id
```

Nội dung có thể được lưu dưới dạng HTML.

Content chưa phải retrieval unit và chưa được đưa trực tiếp vào index.

Pipeline xử lý:

```text
content_html
→ HTML cleaning
→ clean text
→ legal structure parsing
→ legal chunks
→ indexes
```

Một số văn bản có thể:

- không có content;
- chỉ có metadata;
- có nhiều content records;
- có nội dung trùng;
- chứa HTML không nhất quán;
- chứa table;
- chứa nội dung scan không parse được.

Các trường hợp này phải được audit.

---

## 6. Relationships Component

Relationships chứa các quan hệ có hướng giữa các văn bản.

Cấu trúc logic:

```text
doc_id
other_doc_id
relationship
```

Khóa nối:

```text
relationships.doc_id → metadata.id
relationships.other_doc_id → metadata.id
```

Quan hệ có thể mô tả:

- sửa đổi;
- bổ sung;
- thay thế;
- bãi bỏ;
- hết hiệu lực;
- dẫn chiếu;
- hướng dẫn;
- quy định chi tiết;
- liên quan;
- quan hệ khác trong raw dataset.

Danh sách relationship type thực tế phải được audit.

Không được tự suy diễn canonical relationship trước khi xem giá trị
thực tế.

---

## 7. Required Dataset Audit

Audit phải được thực hiện trước cleaner và chunker.

### 7.1 Schema Audit

Kiểm tra:

- config names;
- column names;
- data types;
- nested fields;
- nullable fields;
- unexpected columns.

### 7.2 Identity Audit

Kiểm tra:

- số dòng;
- số ID duy nhất;
- duplicate metadata IDs;
- duplicate content IDs;
- empty IDs;
- malformed IDs.

### 7.3 Join Audit

Kiểm tra:

- metadata có content;
- metadata không có content;
- content có metadata;
- content không có metadata;
- một metadata ID có nhiều content records;
- relationship source tồn tại;
- relationship target tồn tại.

### 7.4 Content Audit

Kiểm tra:

- content rỗng;
- content quá ngắn;
- content quá dài;
- HTML lỗi;
- duplicate content;
- content chỉ chứa navigation;
- content chỉ chứa bảng;
- content không có dấu hiệu cấu trúc pháp lý.

### 7.5 Relationship Audit

Kiểm tra:

- relationship distribution;
- source ID không tồn tại;
- target ID không tồn tại;
- self-loop;
- duplicate edge;
- reciprocal edge;
- unknown relationship label;
- empty relationship label.

### 7.6 Metadata Audit

Kiểm tra:

- ngày không parse được;
- ngày hiệu lực trước ngày ban hành;
- ngày hết hiệu lực trước ngày có hiệu lực;
- trạng thái hiệu lực không nhất quán;
- loại văn bản bị thiếu;
- số ký hiệu bị thiếu;
- title bị thiếu;
- source URL bị thiếu.

---

## 8. Join Policy

### 8.1 Metadata and Content Both Exist

Document được:

- normalize;
- clean;
- parse;
- chunk;
- đưa vào BM25 index;
- đưa vào vector index;
- đưa vào graph.

### 8.2 Metadata Exists but Content Is Missing

Document:

- không được đưa vào text retrieval;
- vẫn có thể được giữ làm graph node;
- phải có `has_content = false`;
- phải xuất hiện trong audit report.

### 8.3 Content Exists but Metadata Is Missing

Record:

- không được silently drop;
- không được tự tạo metadata giả;
- phải được ghi vào audit report;
- chỉ được xử lý khi có quyết định mới.

### 8.4 Invalid Relationship

Relationship:

- không được đưa vào production graph;
- phải được ghi vào invalid relationships report;
- không được tự tạo ghost node.

---

## 9. Raw Data Preservation

Raw records không được sửa trực tiếp.

Pipeline dữ liệu:

```text
Raw dataset
→ Raw snapshot or reproducible loader
→ Normalized records
→ Cleaned documents
→ Structured legal blocks
→ Legal chunks
→ Index artifacts
```

Mỗi tầng phải:

- có schema;
- có version;
- có manifest;
- có khả năng tái tạo;
- có validation.

---

## 10. Dataset Versioning

Mỗi lần ingestion phải ghi:

- dataset name;
- Hugging Face revision hoặc commit hash;
- config names;
- load timestamp;
- record counts;
- code version;
- processing config hash;
- generated artifact paths;
- warnings;
- audit summary.

Không được mặc định dataset luôn bất biến.

---

## 11. Known Limitations

- metadata có thể là snapshot cũ;
- tình trạng hiệu lực có thể chưa cập nhật;
- một số văn bản có thể thiếu content;
- một số nội dung có thể là bản scan;
- HTML có thể không đồng nhất;
- cấu trúc Điều/Khoản có thể bị lỗi;
- relationship labels có thể chưa chuẩn hóa;
- cùng một văn bản có thể xuất hiện nhiều bản ghi;
- corpus chưa cung cấp gold QA labels;
- dataset không thay thế nguồn pháp luật chính thức.