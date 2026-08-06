# 13. UIT DSC 2026 Task 2 Official Data Contract

## 1. Purpose and authority

Tài liệu này ghi lại nội dung kỹ thuật trong file BTC cung cấp
`DSC2026_Task2_LegalQA_Data_Overview.docx` dưới dạng contract để code, test và
review cùng bám theo. Đây là bản diễn giải trong repository, không thay thế file
gốc của BTC.

Khi byte dữ liệu thật khác ví dụ overview, phải dừng phần phụ thuộc, audit dữ
liệu thật và cập nhật decision trước khi sửa core.

## 2. Task contract

Input là một câu hỏi pháp luật tiếng Việt. Hệ thống phải:

1. truy xuất văn bản/context liên quan;
2. tạo một câu trả lời tự nhiên bằng văn xuôi;
3. dựa câu trả lời trên căn cứ pháp lý đã truy xuất.

Reference answer do chuyên gia pháp lý xây dựng. Output chấm điểm chỉ là answer
text; citation, trace và evidence metadata là dữ liệu nội bộ, không tự thêm vào
submission nếu output contract không yêu cầu.

## 3. Released resource names

| Tệp | Vai trò đã được BTC mô tả |
|---|---|
| `train.json` | Dữ liệu huấn luyện để phát triển phương pháp |
| `warmup.json` | Dữ liệu mẫu cho Warm-up và quy trình submission |
| `public-official.json` | Dữ liệu chính thức giai đoạn Public Test |
| `private-official.json` | Dữ liệu chính thức giai đoạn Private Test |
| `selected-contexts.zip` | Kho văn bản chính thức đã phát hành, gồm 8.532 `context_*.json` |

Không suy đoán file đã tồn tại chỉ từ tên trong overview. Mỗi file thật phải
được checksum, inventory và audit trước khi sử dụng.

## 4. Question/answer contract

Ví dụ của BTC mô tả JSON object ánh xạ ID câu hỏi sang record:

```text
question_id -> {
  question: string,
  answer: string
}
```

`answer` tồn tại trong dữ liệu có reference như train/warm-up. Public/private
phải được loader hỗ trợ ở dạng question-only nếu file thật không có answer;
không được suy đoán answer trước khi audit.

Question và answer phải được giữ nguyên Unicode và nội dung pháp lý. Reference
answer có thể dài, chứa xuống dòng, bullet, căn cứ sửa đổi, ngày hiệu lực và cả
mô tả quy định trước đây. Không được coi answer là relevance label cho một
context/chunk nếu BTC không cung cấp liên kết đó.

## 5. Context contract

BTC mô tả mỗi `context_*.json` là một JSON object có các raw field:

| Raw field | Ý nghĩa |
|---|---|
| `id` | Mã định danh duy nhất của văn bản/context |
| `name` | Tên hoặc tiêu đề do BTC cung cấp; optional trong archive thật |
| `link` | Đường dẫn nguồn |
| `passage` | Nội dung văn bản dùng làm ngữ cảnh/căn cứ |

Ví dụ overview dùng numeric JSON `id` (`740`), `name` dạng slug, URL nguồn và
`passage` dài với CRLF, dòng trống, Unicode, tiêu đề, số hiệu, ngày tháng, căn
cứ và chữ ký.

Ranh giới adapter phải tuân thủ:

- raw `id` trong archive thật là non-negative JSON integer; adapter chấp nhận
  non-negative integer hoặc non-blank string để hỗ trợ contract đã công bố và
  canonicalize sang unified string ID; boolean, float, null, blank, ID trùng và
  collision sau canonicalization đều bị từ chối;
- `name` được giữ nguyên làm title nguồn; không tự biến slug thành tên pháp lý;
- `link` chỉ là provenance; không được crawl URL hoặc lấy thêm nội dung ngoài
  dữ liệu BTC;
- `passage` phải giữ nguyên ở raw boundary; cleaning/parsing chỉ được thực hiện
  bằng policy đã kiểm thử và không làm mất cấu trúc, số, ngày, phủ định hoặc
  điều kiện pháp lý;
- archive thật có đúng hai field set: `id/link/passage` và
  `id/name/link/passage`; mọi field ngoài tập này bị từ chối;
- không suy diễn số hiệu, hiệu lực, cơ quan, quan hệ hoặc citation từ tên file,
  slug hay URL khi passage không chứng minh.

## 6. Core mapping

Raw field chỉ tồn tại trong adapter UIT DSC 2026:

```text
id      -> CompetitionContext.context_id -> LegalDocument.document_id
name?   -> CompetitionContext.title      -> LegalDocument.title
link    -> CompetitionContext.source_url -> LegalDocument.source_url
passage -> CompetitionContext.passage    -> normalized LegalDocument.clean_text
        -> UitDsc2026PassageCleaner      -> cleaned LegalDocument.clean_text
```

Unified/core schema tiếp tục dùng string ID. Việc chấp nhận numeric raw ID là
trách nhiệm của adapter, không được làm kiểu raw lan vào core.

Một context file được coi là một organizer-selected source record. Parser và
chunker có thể chia passage thành retrieval units nhưng phải giữ provenance về
context ID ban đầu.

## 7. Retrieval and graph implications

Overview không cung cấp relevance labels hoặc relationship table. Vì vậy:

- BM25, dense, RRF và reranker có thể build từ official passages;
- không tạo synthetic positive/negative pairs từ reference answers;
- graph không được tự giả định có edge;
- graph retrieval chỉ được bật nếu quan hệ được trích xuất có kiểm chứng từ
  chính official passage hoặc BTC cung cấp relationship data;
- nếu không có quan hệ đáng tin cậy, graph artifact phải rỗng/disabled và fixed
  retrieval vẫn hoạt động độc lập.

## 8. Evaluation contract

- METEOR là metric chính dùng để xếp hạng;
- scorer BTC hiện dùng whitespace `str.split()` và NLTK `meteor_score` defaults
  cho METEOR, có WordNet/OMW resource;
- ROUGE-L là metric phụ dựa trên Longest Common Subsequence, dùng vendored
  `rouge_score` default tokenizer lowercase chỉ giữ ASCII `[a-z0-9]`, không
  stemming;
- cả hai metric được arithmetic macro mean trên prediction IDs;
- scorer hiện không dùng PyVi;
- cả hai metric càng cao càng tốt;
- scorer source được định danh bằng ZIP SHA-256
  `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`.

Chi tiết thuật toán, runtime I/O, member checksum và khác biệt với evaluator
local nằm tại `docs/15-OFFICIAL-SCORING-CONTRACT.md`. ZIP không pin NLTK/NumPy
hoặc WordNet bytes, nên các phần môi trường đó vẫn phải được khóa trước khi
tuyên bố tái lập tuyệt đối.

Optimization phải giữ cân bằng giữa similarity với reference và grounded legal
correctness. Không được thêm thông tin không có căn cứ chỉ để tăng overlap.

## 9. Released-data audit

Audit read-only ngày 2026-08-06 xác nhận:

| Artifact | SHA-256 / canonical revision | Records |
|---|---|---:|
| `train.json` | `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988` | 7.000 |
| `public-official.json` | `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5` | 1.000 |
| `selected-contexts.zip` | `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97` | 8.532 |
| canonical corpus content | `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e` | 8.532 |

Corpus inventory:

- 7.407 records có đủ `id/name/link/passage`; 1.125 records thiếu `name`;
- 8.512 passage có nội dung, 20 passage rỗng;
- mọi raw `id` là integer duy nhất, không âm; filename và ID khớp;
- URL là HTTPS duy nhất từ `thuvienphapluat.vn`; URL chỉ là provenance;
- 24 records dư do passage trùng chính xác, gồm nhóm 20 passage rỗng và bốn
  nhóm non-blank; không tự drop record;
- 8.509 passages có CRLF, 608 passages chưa NFC;
- cleaner xác nhận 10 contexts có known HTML markup và loại 13.066 lần exact
  TVPL Pro notice trong 5 contexts;
- raw passage có 353.691.230 ký tự; cleaned passage có 342.135.431 ký tự;
- passage dài nhất có 5.983.358 ký tự; pipeline phải tiếp tục xử lý bounded;
- 7.622 documents có dòng Điều; parser/chunker hiện tại có cơ sở để tái sử dụng;
- corpus không có relationship table hoặc evidence/relevance labels; graph
  artifact phải rỗng ở baseline.

QA inventory:

- `train.json` có 7.000 câu hỏi với answer string; `public-official.json` có
  1.000 câu hỏi với `answer = null`;
- train có 16 nhóm câu hỏi trùng sau normalize; 14 nhóm có answer khác nhau;
- warm-up không còn là independent dev: 387/500 record trùng exact ID và nội
  dung với train;
- 41 public questions có normalized question đã xuất hiện trong train/warm-up;
  mọi split/evaluation phải có leakage policy rõ ràng;
- answer supervision có thể phục vụ generator experiment sau khi có split và
  model approval, nhưng không được biến thành retrieval/reranker labels hoặc
  synthetic examples.

## 10. Cleaning policy accepted from the audit

`UitDsc2026PassageCleaner` chạy sau raw normalization và trước parser:

- giữ nguyên raw passage trong normalized artifact;
- chuẩn hóa newline, Unicode NFC, Unicode space/control an toàn và khoảng trắng
  đầu/cuối dòng;
- loại đúng known HTML presentation tags, giữ visible text và `img alt`;
- loại đúng chuỗi TVPL Pro notice đã audit, kèm dot placeholder nếu có;
- không xóa repeated opening sequences, duplicate records, số, ngày, phủ định,
  điều kiện, unknown angle-bracket fields hoặc legal metadata;
- blank passage vẫn tồn tại với `has_content = false` và không sinh chunk;
- mọi phép biến đổi được pin bằng cleaner version và processing hash trong
  cleaned manifest.

## 11. Mandatory audit before every future official build

Khi nhận dữ liệu thật, phải kiểm tra tối thiểu:

1. archive/file checksum, encoding và inventory;
2. JSON root shape và duplicate JSON keys;
3. question ID/context ID types, uniqueness và canonicalization collisions;
4. required, null, blank và unknown fields;
5. số record, độ dài passage và duplicate title/URL/content;
6. newline, Unicode và markup/noise distribution;
7. một context là full document hay excerpt trong dữ liệu thật;
8. train/public/private có answer hay evidence labels nào;
9. leakage giữa train, warm-up, public, private và corpus;
10. khả năng parse/chunk và số relationship có căn cứ.

Audit lần đầu đã hoàn tất cho ba file nêu trên. Nếu BTC thay file hoặc phase,
checksum/revision thay đổi thì phải chạy lại toàn bộ checklist trước khi build
index hoặc training run mới.
