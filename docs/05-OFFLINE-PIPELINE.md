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

### 3.5 Milestone 2 Implementation Boundary

Milestone 2 triển khai audit theo các nguyên tắc:

- raw source field names chỉ xuất hiện trong competition adapter và audit;
- audit chỉ đọc, thống kê và phân loại issue, không sửa raw record;
- relationship label và effect status không bị tự canonicalize;
- unknown-value check chỉ bật khi có accepted set trong configuration;
- JSON/CSV report không silently overwrite kết quả cũ;
- normalization và join thành `LegalDocument` vẫn thuộc Milestone 3.

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

### 4.1 Milestone 3 Implementation Policy

- metadata là nguồn authoritative để tạo `LegalDocument`;
- metadata ID trùng bị reject toàn bộ theo ID, không tự chọn một record;
- content chỉ được gắn khi ID có đúng một content record hợp lệ;
- nhiều content cùng ID không bị merge hoặc tự chọn; document được giữ với
  `has_content = false` và issue rõ ràng;
- orphan content không tạo metadata giả;
- ngày không parse được thành null và sinh warning, ngày parse được nhưng
  thứ tự bất thường vẫn được giữ nguyên;
- effect status và document type chỉ canonicalize qua mapping configuration
  rõ ràng; nếu không có mapping thì giữ nhãn nguồn sau khi trim;
- `nguon_thu_thap` không được suy diễn thành URL và vẫn nằm trong
  `raw_metadata`;
- HTML không bị clean hoặc sửa trong bước này;
- result trả về in-memory cùng `ArtifactManifest`; persistence writer được
  trì hoãn cho tới khi artifact storage format được phê duyệt.

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

### 5.4 Milestone 4 Implementation Policy

Milestone 4 dùng Python standard-library `html.parser` và nhận duy nhất
unified `LegalDocument` cùng manifest `normalized_documents`:

- tag non-content và noise class/id được cấu hình bằng exact token;
- phần tử `hidden`, `aria-hidden=true`, `display:none` hoặc
  `visibility:hidden` bị loại;
- `header` và `footer` được giữ mặc định để tránh mất title, document number,
  signature hoặc appendix;
- table text được giữ với newline giữa row và ` | ` giữa cell; cấu trúc pháp
  lý của bảng chưa được parse;
- Unicode được normalize về NFC; HTML entity, control character, whitespace
  và line break được normalize deterministic;
- missing content và empty clean output tạo structured issue, không làm mất
  document;
- output là `HtmlCleaningResult` in-memory với cleaned artifact manifest;
  persistence writer tiếp tục trì hoãn tới khi artifact storage format được
  phê duyệt.

Không tự suy đoán repeated header/footer bằng corpus-wide text matching vì
policy đó có thể xóa nhầm nội dung pháp lý lặp hợp lệ.

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

### 6.1 Milestone 5 Implementation Policy

Milestone 5 dùng parser quy tắc theo line và chỉ nhận unified
`LegalDocument.clean_text` cùng manifest `cleaned_documents`:

- nhận diện Phần, Chương, Mục, Tiểu mục, Điều, Khoản, Điểm và Phụ lục bằng
  marker tiếng Việt rõ ràng;
- hỗ trợ Điều số thường hoặc La Mã, delimiter `.`, `:`, `-`, `–`, `—` và
  title inline hoặc một dòng title ngắn kế tiếp;
- chỉ nhận dạng Khoản/Điểm khi đang có Điều để tránh biến danh sách hành chính
  độc lập thành hierarchy pháp lý giả;
- numbering thiếu cấp hoặc không liên tục vẫn được giữ, không tự sửa;
- marker không đủ chắc chắn được giữ như ordinary text và có structured issue;
- preamble/văn bản không cấu trúc tạo block `document`; table rows liên tiếp
  tạo block `table` kế thừa hierarchy;
- block text không chồng lấp và coverage đo bằng non-whitespace characters;
- block ID là hash deterministic từ document ID, order, type và preserved
  block text;
- output là `LegalStructureParsingResult` in-memory cùng manifest
  `legal_blocks`; persistence writer tiếp tục trì hoãn tới khi artifact storage
  format được phê duyệt.

Parser không suy diễn semantic heading, không sửa chính tả và không gộp text
vào chunk. Chunking vẫn thuộc Milestone 6.

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

### 7.4 Milestone 6 Implementation Policy

Milestone 6 chỉ nhận unified `LegalDocument`, `LegalBlock` và manifest
`legal_blocks`:

- Điều cùng toàn bộ descendant blocks tạo một chunk khi không vượt
  `max_tokens`;
- Điều dài được chia bằng cách gom các direct child legal units, ưu tiên các
  Khoản liên tiếp và không vượt giới hạn;
- nếu một legal unit vẫn quá dài, tokenizer `unicode_word_v1` tạo sliding
  windows với overlap cấu hình;
- `Phần/Chương/Mục/Tiểu mục` liên tiếp được gắn vào retrieval unit nội dung kế
  tiếp; heading cuối tài liệu không có consumer vẫn được giữ thành standalone
  chunk để bảo toàn source-block coverage;
- block ngoài Điều còn lại tạo standalone chunk để không mất preamble, phụ lục,
  table hoặc văn bản không cấu trúc;
- defaults official baseline là `max_tokens=384`, `max_search_tokens=448`,
  `min_tokens=50`, `overlap_tokens=50`; `min_tokens` là ngưỡng cảnh báo, không
  phải lý do xóa chunk hợp lệ;
- chunk ID là content hash deterministic; `chunk_index` liên tục theo từng
  document;
- mỗi chunk giữ source block IDs, strategy, tokenizer, split index/count,
  inherited document metadata và `LegalStructure`;
- `search_text` luôn giữ nguyên toàn bộ chunk text; metadata/hierarchy có thật
  chỉ được thêm khi còn nằm trong `max_search_tokens`;
- validator kiểm tra ID, chunk/search token count, content preservation,
  source-block coverage, metadata inheritance và chunk ordering;
- output là `LegalChunkingResult` in-memory cùng manifest `legal_chunks`;
  persistence writer tiếp tục trì hoãn tới khi artifact storage format được
  phê duyệt.

`unicode_word_v1` không được xem là tokenizer của embedding model hoặc LLM.
Khi model chính thức được chọn, tokenizer/config và artifacts phải được đánh
giá lại, version và rebuild rõ ràng.

`max_search_tokens=448` là budget proxy có chừa headroom cho prefix/provider
tokens của model 512-token hiện tại. Đây không phải bằng chứng tokenizer parity;
trước GPU build phải chạy exact tokenizer preflight của embedding model đã đăng
ký và fail nếu bất kỳ input nào bị truncate.

### 7.5 Token Split Fallback

Token split chỉ được dùng khi legal boundary không đủ.

Cần cấu hình:

- maximum token count;
- minimum token count;
- overlap token count;
- tokenizer name.

Các giá trị này chưa được hard-code trong architecture docs.

---

### 7.6 Exact embedding-tokenizer gate

Competition config `0.40.0` uses `embedding_model_v1`, loaded from the exact
embedding model name and immutable revision. Counts include the `passage:`
prefix and special tokens. The competition budgets are 448 tokens for chunk
content and 512 tokens for `search_text`; source-span splitting preserves the
original legal text and metadata is removed before content.

The complete M39 preflight found 6,239 of 373,253 inputs above 512 E5 tokens,
with a maximum of 1,043. Therefore `0.38.1` is diagnostic only and must not be
embedded. Exact preflight remains mandatory even after tokenizer-aware rebuild.

The complete M40 rebuild produced 330,768 chunks from 8,532 official contexts.
An independent batched preflight retokenized every persisted `search_text` with
the pinned E5 tokenizer, including the `passage:` prefix and special tokens. The
maximum input length was 512 tokens and the violation count was zero. The M40
BM25 index contains the same 330,768 records and is the only chunk/BM25 lineage
authorized for the next vector build.

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

### 9.1 Milestone 7 Implementation Policy

Baseline dùng SQLite FTS5 như một reference backend phía sau `BM25Backend`:

- input là toàn bộ `LegalChunk` cùng manifest `legal_chunks` tương ứng;
- analyzer `unicode_word_casefold_v1` normalize NFC/case cho lexical matching
  nhưng không sửa `text` hoặc metadata gốc;
- giữ dấu tiếng Việt, số và từ phủ định; không stemming hoặc stopword removal;
- build theo thứ tự chunk ID ổn định và tie-break theo chunk ID;
- artifact versioned gồm `index.sqlite3` và `manifest.json` với SHA-256 checksum;
- persist từ chối destination đã tồn tại; reload kiểm tra manifest, backend,
  analyzer, match mode, checksum, SQLite integrity và record count;
- manifest ghi source artifact version/config hash, SQLite version và backend;
- không thêm dependency vì dùng `sqlite3` trong Python standard library.

Đây là reference baseline, không phải quyết định production backend cuối cùng.
Khi đổi backend hoặc analyzer phải tạo artifact version/config hash mới và
rebuild index.

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

### 10.1 Milestone 8 Implementation Policy

Dense baseline hiện dùng:

- model `intfloat/multilingual-e5-small` pin tại revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3` (MIT);
- `sentence-transformers>=5,<6` phía sau `EmbeddingProvider`;
- prefix `passage:` cho `LegalChunk.search_text`, prefix `query:` cho query;
- dimension 384, max sequence length 512 và L2-normalized embeddings;
- CPU là device mặc định vì runtime hiện không có NVIDIA GPU; device vẫn cấu hình;
- batching mặc định 16 và model download chỉ xảy ra khi cache chưa có cùng
  `local_files_only = false`;
- NumPy float32 flat matrix làm reference `VectorBackend`, exact cosine search;
- artifact gồm `vectors.npy`, `chunks.jsonl`, `manifest.json` và SHA-256 cho
  numeric/chunk payload;
- manifest ghi cả embedding provider name/version; reload memory-map vector
  matrix và kiểm tra provider, model, revision, dimension, normalization, dtype,
  metric, checksums, record count và chunk order;
- persist từ chối destination đã tồn tại.

Chưa dùng FAISS hoặc vector database vì chưa đo corpus-scale bottleneck và
chưa có resource limit chính thức. Khi đổi model/revision/dimension/provider,
phải re-embed toàn corpus và rebuild vector index; không tái sử dụng artifact cũ.

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

### 11.1 Milestone 11 Implementation Policy

- raw relationship fields chỉ được đọc qua dataset-specific adapter;
- chỉ canonicalize label khi có mapping cấu hình rõ ràng;
- label chưa map vẫn được giữ ở `raw_relationship` và canonical type là null;
- orphan endpoint, self-loop, endpoint/label thiếu và exact duplicate bị loại
  khỏi graph, đồng thời tạo `AuditIssue`;
- accepted edges được sort deterministic;
- relationship artifact gồm `relationships.jsonl`, `manifest.json` và SHA-256;
- persistence từ chối ghi đè destination đã tồn tại.

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

### 12.4 Milestone 11 Reference Backend

Milestone 11 dùng `adjacency_json` từ Python standard library làm reference
`GraphBackend`, không phải quyết định production cuối cùng.

- graph chỉ lưu document IDs và unified directed relationships, không nhân bản
  toàn bộ legal content;
- source document manifest và relationship manifest phải cùng dataset/revision;
- relationship manifest phải trỏ đúng processing hash của normalized documents;
- artifact gồm `graph.json`, `manifest.json` và SHA-256;
- load kiểm tra backend/version/checksum/count/order/endpoints;
- traversal là deterministic BFS, 1 hop mặc định và tối đa 2 hop;
- filter theo canonical type; nếu canonical type chưa có thì dùng raw label;
- mỗi reached document chỉ có một BFS discovery path ngắn nhất trong trace.

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
├── build_validation.json
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

---

## 16. Runtime Assembly

Official ZIP/directory loader, context adapter, audit, dataset-specific cleaner
và resumable offline composition root đã có. Pipeline đã chốt sau audit corpus
BTC thật là:

```text
Official competition context snapshot
→ audit
→ normalize documents (giữ raw passage hợp lệ)
→ UIT DSC passage cleaner
→ parse legal blocks
→ legal chunks
→ normalize relationships
→ build BM25/vector/graph
→ persist immutable artifact set
```

Processed payloads `normalized_documents`, `cleaned_documents`,
`legal_blocks` và `legal_chunks` dùng deterministic JSONL kèm manifest và
SHA-256. Relationship/BM25/vector/graph tiếp tục dùng artifact store riêng của
từng module. Tất cả directory nằm dưới configured `ArtifactConfig.root_path`;
future runtime preflight phải từ chối overwrite trước khi load dataset.

Corpus audit xác nhận 8.532 records, trong đó 20 blank passages và 1.125 missing
titles. Cleaner chuẩn hóa NFC/newline/line whitespace, loại known HTML
presentation markup và exact TVPL Pro notice; không drop record hoặc suy diễn
metadata. Relationship artifact và graph giữ rỗng vì source không có edge.

Mỗi processed stage phải được persist/checksum ngay sau khi tạo. Runtime tương
lai phải giải phóng stage không còn consumer và không giữ toàn corpus trong RAM.

---

## 17. Full-Corpus Profile and Post-Build Validation

Full competition corpus profile phải:

- pin dataset revision;
- đặt `sample_limit = null`;
- khai báo expected counts tại dataset-specific config boundary;
- dùng artifact root mới;
- không silently overwrite build cũ.

Sau khi persist đủ tám artifact, `ArtifactSetValidator` kiểm tra:

1. dataset và artifact manifest parse được;
2. mọi artifact cùng dataset/revision;
3. payload checksum khớp;
4. JSONL, SQLite và vector record count khớp manifest;
5. normalized → cleaned → blocks → chunks lineage liên tục;
6. BM25/vector cùng trỏ tới legal chunks;
7. relationships và graph cùng trỏ tới normalized documents;
8. full-corpus count policy được thỏa.

`build_validation.json` là bằng chứng readiness của lần build. Report invalid
được persist để chẩn đoán và offline command fail closed. Validation không tải
lại dataset và không sửa artifact.

### 17.1 Partial-Build Resume

Full profile tạo `build_state.json` schema `1.1` trước ingestion, gồm:

- schema version;
- exact canonical application-config SHA-256;
- code version;
- timezone-aware creation time.

Canonical hash sắp xếp key của mapping và phần tử của set/frozenset, nhưng giữ
nguyên thứ tự list/tuple. Vì vậy cùng một typed config phải có cùng hash giữa
các process Python độc lập.

Resume chỉ được phép khi:

- config bật `resume_partial_build`;
- dataset revision được pin;
- chưa có `build_validation.json`;
- dataset manifest, audit và normalized checkpoint tồn tại;
- config hash và code version khớp tuyệt đối;
- các stage hiện có thỏa dependency order.

Runtime load/checksum checkpoint đã hoàn thành, chỉ chạy các stage còn thiếu.
Failure trước normalized checkpoint, partial file không hợp lệ hoặc config/code
đổi phải dùng artifact root mới; runtime không tự xóa hay overwrite dữ liệu.
State schema `1.0` không thể migrate an toàn vì chỉ lưu digest cũ, không lưu
config gốc; runtime từ chối state này và yêu cầu artifact root mới.

### 17.2 Memory-Bounded Document Processing and Indexing

Full profile từ version `0.20.0` không materialize complete cleaned, block hoặc
chunk stage trong parser/index build:

- cleaned JSONL được checksum rồi đọc bằng one-pass typed iterator;
- parser và chunker xử lý một document tại một thời điểm;
- legal blocks và chunks được ghi incremental vào staging JSONL;
- staging artifact chỉ được publish sau khi count, manifest và checksum hợp lệ;
- progress được log theo
  `offline.execution.document_processing_progress_interval`;
- BM25 đọc chunk stream một lần, insert theo
  `offline.bm25.write_batch_size` vào SQLite disk-backed;
- vector builder embed theo batch và ghi thẳng vào NumPy memmap cùng chunk JSONL;
- vector row order giữ deterministic source-artifact order;
- lỗi Python giữa chừng không publish destination artifact không hoàn chỉnh.

Thay đổi này xuất phát từ full-corpus measurement trên Colab 12 GiB: legacy
parser bị OOM-kill ở khoảng 10,8 GiB anonymous RSS. GPU không giải quyết stage
CPU/RAM này. Thuật toán nhận dạng legal structure, chunk boundaries và unified
record schema không đổi.

Build `0.19.x` không được resume bằng code `0.20.0` vì build state khóa exact
code/config identity. Phải giữ artifact cũ để chẩn đoán và dùng artifact root
mới; không sửa tay `build_state.json`.

### 17.3 Official Competition Build (M27)

`legal-rag-build-competition` composes the UIT DSC adapter with the reusable
offline modules. The fixed stage order is:

```text
corpus -> document_processing -> bm25 -> vector -> validation
```

`competition_build_state.json` pins the canonical source revision,
application-config SHA-256, code version, timestamps, and completed stage
prefix. Resume validates already-published payloads before reuse.

The corpus stage persists provenance, audit, raw-preserving normalized documents
and dataset-specifically cleaned documents, a zero-record relationship artifact, and a zero-edge
graph. Empty relationships are truthful because the documented official fields
contain no relationship information. Document processing remains streaming,
BM25 remains disk-backed, and vector build retains `.vector.partial` batch
checkpoints. Final validation still requires all eight artifact types.

Từ version `0.37.0`, CLI nhận `--through` để dừng ở một durable stage. Target
`document_processing` chạy chính xác:

```text
corpus audit/normalize/clean
→ legal structure parser
→ legal chunker
→ checksum legal_blocks/legal_chunks
→ stop
```

Chế độ này không khởi tạo embedding provider, không build BM25/vector và không
tạo `build_validation.json`, vì report cuối chỉ hợp lệ khi đủ toàn bộ artifact.
`competition_build_state.json` vẫn ghi ordered completed-stage prefix. Lần chạy
sau với cùng source revision, application-config hash và code version xác minh
lại payload đã persist rồi resume từ BM25. Destination không tương thích tiếp
tục fail closed và không bị overwrite.

Measured run ngày 2026-08-06 trên canonical corpus revision tạo 1.215.092 legal
blocks và 335.014 legal chunks từ 8.532 documents. Source/covered non-whitespace
characters đều là 261.550.497; 7.637 documents có recognized structure, 875
documents không có explicit marker và đúng 20 blank documents không có chunk.
Chunk strategies gồm 131.806 article, 73.914 clause-group, 87.623 token-fallback
và 41.671 standalone chunks. Hai run độc lập tạo cùng payload SHA-256 cho blocks
và chunks; invocation thứ hai trên root hợp lệ chỉ verify/resume trong khoảng
11 giây. Số liệu thời gian full stage phụ thuộc CPU/I/O và không phải benchmark
phần cứng chuẩn của BTC.

### 17.4 Resumable Vector Batches

Compatibility note: version `0.20.2` only changes online vector loading/search,
not offline output, canonical config hash, or artifact lineage. Partial build
state schema `1.1` from `0.20.0` or `0.20.1` may therefore be upgraded directly
to `0.20.2`; all other version/config transitions remain fail closed.

Từ version `0.20.1`, vector build dùng workspace bền vững
`.vector.partial` trong artifact root thay vì thư mục staging ngẫu nhiên:

- `vectors.npy` được preallocate một lần và mở lại bằng memory map;
- `chunks.jsonl` chỉ giữ các record đã commit;
- `checkpoint.json` schema `1.0` ghi vector manifest identity, `next_offset`,
  byte offset của chunk payload và thời điểm cập nhật;
- payload được flush trước khi checkpoint mới được publish atomically;
- khi resume, builder đọc lại legal-chunks artifact đã checksum, bỏ qua
  `next_offset` record và chỉ embedding phần còn lại;
- checkpoint chỉ được chấp nhận khi source artifact, model, provider,
  dimension, dtype, batch size và processing hash tương thích;
- `offline.vector_index.checkpoint_interval_batches` điều khiển cadence, mặc
  định 100 batch; đây là execution tuning và không thay đổi final artifact hash;
- destination `vector/` chỉ xuất hiện sau khi đủ record, checksum hoàn tất và
  toàn bộ workspace được rename.

Nếu process bị SIGKILL giữa hai checkpoint, lần sau chỉ mất phần chưa commit.
Các thư mục staging ngẫu nhiên `.vector-*` từ version `0.20.0` không có
checkpoint đáng tin cậy và không được tự động tái sử dụng.

Build state `0.20.0` schema `1.1` được phép nâng một chiều lên `0.20.1` chỉ cho
thay đổi recovery này, sau khi canonical application-config hash vẫn khớp.
Mọi code-version transition khác tiếp tục bị từ chối fail closed.

## 19. M45 Qwen3 Embedding Build

M45 là artifact lineage mới, không resume hoặc ghi đè vector E5 của M43. Corpus
vẫn là canonical official revision và giữ nguyên cleaning/parser/chunker policy.
Dense stage dùng
`Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`,
dimension 1.024, normalized cosine và fp16 trên CUDA. Passage không nhận prefix;
query nhận instruction tiếng Anh cố định qua provider adapter. Instruction,
dtype, revision và dimension đều thuộc application config hash/manifest, nên mọi
thay đổi bắt buộc rebuild vector.

Build vẫn checkpoint theo batch và chỉ publish vector artifact sau khi đủ record,
finite/unit-norm validation và checksum hoàn tất. Không có dữ liệu ngoài, synthetic
record hoặc relation suy diễn được thêm vào corpus.

## 18. M41 Full-corpus Validation Boundary

Artifact M40 giữ nguyên cấu hình build và lineage đã tạo ra nó. Chính sách
full-corpus dùng cho online được đặt trong cấu hình serving riêng, yêu cầu đúng
8.532 context và 8.512 context có nội dung. `legal-rag-validate --persist` tạo
một report bất biến với tên cấu hình; command từ chối ghi đè report có sẵn.

Vector M40 được build từ 330.768 chunk chính thức với E5 revision đã pin. Việc
tạo `vector_serving` chỉ sinh metadata tra cứu theo row/offset, không re-embed,
không thay đổi vector và không tạo dữ liệu mới.
