# 10. Competition Adaptation

## 1. Confirmed Task Contract

UIT Data Science Challenge 2026 Task 2 nhận câu hỏi pháp luật tiếng Việt và
yêu cầu câu trả lời văn xuôi tiếng Việt. Reference answers do chuyên gia pháp
lý cung cấp.

BTC công bố các tên tài nguyên:

- `warmup.json`;
- `train.json`;
- `public-official.json`;
- `private-official.json`;
- `selected-contexts.zip` với `context_*.json`.

Context raw schema được mô tả bằng `id`, `name`, `link`, `passage`.

Contract chi tiết và audit checklist nằm tại
`docs/13-UIT-DSC-2026-DATA-CONTRACT.md`. Ví dụ overview dùng numeric `id`,
slug-like `name`, source URL và passage chứa full legal structure/newline noise.
Không được suy ra rằng ID luôn là string hoặc corpus không có field bổ sung
trước khi audit archive thật.

Phản hồi chính thức của BTC ngày 2026-08-01 xác nhận thêm:

- chỉ được dùng dữ liệu chính thức của cuộc thi;
- preprocessing, pipeline, indexing, retrieval và fine-tuning được phép nếu chỉ
  dùng dữ liệu chính thức;
- synthetic data bị cấm kể cả khi sinh từ dữ liệu BTC;
- model mã nguồn mở phải đăng ký tên và URL qua Google Form BTC sẽ cung cấp;
- Codabench là nền tảng chính thức của Task 2;
- runtime/Docker/GPU/RAM/network/private-test rules sẽ công bố sau.

Thông báo chung tiếp theo của BTC xác nhận:

- tổng tham số của tất cả model trong một hệ thống Task 2 phải nhỏ hơn 4 tỷ,
  tính cả embedding, reranker, generator và model phụ;
- distillation hợp lệ nếu model cuối dưới giới hạn; quantization và LoRA không
  làm giảm parameter count được xét;
- cấm mọi API, kể cả miễn phí/phi lợi nhuận; đội phải trực tiếp sở hữu khả năng
  tải, chạy và kiểm soát model;
- pretrained model được phép và dữ liệu pretraining không bị xem là dữ liệu
  ngoài, nhưng pipeline chỉ được trực tiếp dùng dữ liệu BTC và không được data
  augmentation;
- BTC sẽ cung cấp training data cho cả hai task;
- có thể giao Docker, GitHub hoặc ZIP; được tải open weights hợp lệ từ Internet
  nếu README mô tả đầy đủ quy trình tái lập.

## 2. Metric

- METEOR: metric xếp hạng chính;
- ROUGE-L: metric phụ;
- higher is better.

Chưa được coi local implementation nào là tương đương official scorer cho tới
khi có scorer code hoặc đối chiếu kết quả Codabench. Cần xác minh tokenizer,
normalization, library/version và cách aggregate.

Warm-up logs đã quan sát scorer dùng PyVi tokenization, NLTK `meteor_score` và
`rouge-score`. Scoring image từng thiếu NLTK `wordnet`; BTC đã sửa và một
submission đã hoàn tất chấm điểm. Đây là bằng chứng một phần về implementation,
nhưng vẫn chưa đủ để tuyên bố local scorer parity khi parameters/aggregation
chưa được công bố đầy đủ.

## 3. Active Data Policy

Repository dùng policy `competition_only`:

- chỉ ingest corpus và QA data chính thức của BTC;
- không trộn external corpus;
- không tạo synthetic QA, synthetic evidence, synthetic hard negative hoặc dữ
  liệu huấn luyện tổng hợp;
- không load legacy/external artifacts;
- mọi artifact phải ghi đúng competition dataset identity và revision;
- raw data không được đi trực tiếp vào core.

Đây là quyết định fail-closed của dự án và hiện cũng phù hợp Điều 4, Điều 9 của
thể lệ BTC được cung cấp ngày 2026-08-01: chỉ dùng dữ liệu BTC, không manual
label, external collection hoặc external augmentation.

## 4. Current Adapter Boundary

```text
Official JSON
→ UitDsc2026DataLoader
→ CompetitionQuestion / CompetitionContext
→ corpus/training adapter
→ unified schema
```

Loader hiện kiểm tra duplicate JSON keys, unknown fields, missing fields,
duplicate context IDs và blank values. Loader giữ nguyên legal text hợp lệ.

M26 map một `CompetitionContext` sang một `LegalDocument` theo D069. Legal
structure và chunks chỉ được tạo bởi parser/chunker reusable; adapter không tự
suy diễn Điều, Khoản hoặc metadata pháp lý còn thiếu.

## 5. Warm-up Adaptation

`warmup.json` là mapping từ question ID đến:

```json
{
  "question": "...",
  "answer": "..."
}
```

Có thể dùng ngay để:

- kiểm tra question/answer loader;
- xây answer-level dev/holdout split deterministic;
- đánh giá generation ở mức diagnostic;
- phân tích độ dài và phong cách gold answer;
- kiểm tra batch runner/checkpoint sau này.

Không được dùng warm-up answer để giả lập relevant document/chunk IDs.

## 6. Corpus Adaptation When Released

Sau khi nhận `selected-contexts.zip`:

1. checksum và inventory archive;
2. audit encoding, JSON root shape, field types, nulls, duplicates và counts;
3. xác định một context là document, passage hay retrieval unit;
4. chốt deterministic ID mapping;
5. map sang unified schema;
6. quyết định có cần cleaner/parser/chunker hay không;
7. build mới BM25/vector và graph rỗng nếu không có relationships;
8. persist manifests với official lineage;
9. validate artifact set;
10. benchmark end-to-end.

Không tái sử dụng index từ corpus khác.

The accepted M26 mapping is one context file to one unified document:

```text
id      -> document_id
name    -> title
link    -> source_url
passage -> clean_text
```

Raw numeric/string `id` phải được competition adapter canonicalize sang unified
string ID sau audit; URL chỉ giữ provenance và không được dùng để crawl thêm dữ
liệu. Slug-like `name` được bảo toàn, không tự diễn giải thành metadata pháp lý.

No document number, legal dates, authority, effect status, or relevance label
is inferred from the four raw fields. The direct context loader accepts the
official ZIP or an equivalent extracted directory and computes one canonical
content revision over the ordered filenames and bytes.

## 7. Training Adaptation When Released

Trước fine-tuning phải kiểm tra:

- train schema;
- duplicate/leakage giữa splits;
- có evidence labels hay chỉ answer labels;
- metric và evaluator;
- model/license/resource policy;
- experiment plan và baseline comparison.

Trước mọi official experiment còn phải lập model inventory và cộng parameter
count của toàn bộ model active. Inventory tối thiểu gồm model ID, URL, revision,
vai trò, license, parameter count, nguồn xác minh count và trạng thái đăng ký
BTC. Thiếu count hoặc tổng từ 4 tỷ trở lên phải fail closed.

BTC đã xác nhận fine-tuning được phép nếu chỉ sử dụng dữ liệu chính thức. Quyền
này không tự tạo ra supervision: chỉ fine-tune khi train schema thật cung cấp
nhãn phù hợp, có split/leakage control và không sinh synthetic examples.

Không fine-tune retriever nếu chỉ có answer text mà không có supervision phù
hợp. Thay embedding model yêu cầu re-embed và rebuild vector index.

## 8. Submission Adaptation

Các mục sau đã được BTC/Codabench xác nhận cho formatter M28:

- JSON/JSONL/CSV/ZIP;
- field names;
- ordering;
- encoding;
- filename;
- missing/duplicate ID policy;
- có cần citation hay chỉ answer text.

Batch inference phải deterministic, có resume/checkpoint, completeness check và
final checksum.

## 9. Implemented M27 Execution Boundary

Official build and internal batch commands are now available:

```text
legal-rag-build-competition --config <config.json> --source <zip-or-directory>
legal-rag-batch --config <config.json> --questions <questions.json> --output <directory>
```

Batch output contains `results.jsonl`, `batch_state.json`, and, only after every
question is present exactly once in source order, `manifest.json`. Each record
keeps the official question ID and full typed `AnswerResponse`. Warm-up gold
answers are never passed to the answerer.

This remains recovery/diagnostic output, not the submission itself. M28 converts
only a completed compatible batch with:

```text
legal-rag-submit --questions <questions.json> --batch <batch-directory> --output <path>/submission.zip
```

The generated ZIP contains only UTF-8 `submission.json`. Its root object maps
each official question ID to exactly `{"answer": string}` in source order.
Retrieval trace, citations, warnings, manifests, and warm-up reference answers
are not added as separate submission fields.

## 10. Verified Codabench Output Contract

The earlier deferral in Section 8 is resolved by live scorer behavior. The
organizer prose described an array, but the executable scorer calls `.items()`
on the root and therefore requires:

- archive filename: `submission.zip`;
- only archive member: `submission.json`;
- root: JSON object keyed by string question ID;
- each value: an object containing only string `answer`;
- every question ID exactly once, with no missing question;
- UTF-8 Vietnamese text is preserved.

M28 validates exact question and batch bytes before atomically publishing a
deterministic, no-overwrite archive. Official METEOR/ROUGE-L implementation
equivalence remains unresolved until scorer code or verified scoring behavior
is available.

## 11. Implemented Diagnostic Metrics

M29 computes local METEOR and ROUGE-L whenever `reference_answer` exists. The
implementation uses deterministic NFC/casefold Vietnamese word/number tokens,
exact-token METEOR matching with a fragmentation penalty, and token-level
ROUGE-L F1. It deliberately does not invent stemming, synonyms, tokenizer
details, or hidden Codabench parameters.

Evaluation reports carry
`competition_text_metrics_are_diagnostic_not_official_equivalent`. These scores
are suitable for comparing candidates on the same pinned warm-up/dev split,
not for claiming an exact leaderboard score. The submission renderer removes
verified `[E<number>]` markers because the organizer grades answer prose only;
the internal result retains citations for grounding and audit.

## 12. Implemented Warm-up Scoring CLI

M30 can score a finished archive without loading corpus artifacts or models:

```text
legal-rag-score-warmup \
  --references <warmup.json> \
  --submission <submission.zip> \
  --output <new-report-directory>
```

The scorer requires exact ordered ID equality and writes only checksums,
per-question numeric scores, aggregate exact match/METEOR/ROUGE-L, code version,
timestamp, and the D073 diagnostic warning. It does not persist question, gold,
or prediction text.

## 13. Remaining Questions

1. Khi nào `selected-contexts.zip` và `train.json` được mở?
2. Official METEOR/ROUGE-L implementation là gì?
3. Form đăng ký model yêu cầu trạng thái khai báo hay phê duyệt?
4. Môi trường chấm cuối có Internet không, ngoài việc BTC đã cho phép tải trọng
   số model hợp lệ để tái lập?
5. Có giới hạn GPU, RAM, disk, thời gian hoặc số submission không?
6. Public/private question files có answer field hay không?
7. Context IDs có xuất hiện trong supervision không?
8. Data Statement và Model Card được nộp ở đâu và theo format nào?

## 14. Competition Compliance Boundary

M31 thêm compliance source of truth tại
`docs/11-COMPETITION-COMPLIANCE.md`. Những gate bắt buộc trước official run:

- exact official-only data lineage;
- verified third-party license;
- organizer model registration evidence khi áp dụng;
- complete Data Statement và Model Card snapshot;
- immutable config/code/model/artifact identities;
- private submission preflight và ledger dưới giới hạn 3 lần/ngày;
- Docker image digest cùng reproduction evidence.

`legal-rag-submit` và `legal-rag-score-warmup` được tách khỏi serving CLI để
không yêu cầu FastAPI/Uvicorn khi chỉ đóng gói hoặc chấm file local. Docker M31
không chứa dữ liệu/model và chưa phải final GPU image.
