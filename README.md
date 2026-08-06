# Vietnamese Legal Agentic RAG

Hệ thống Retrieval-Augmented Generation cho UIT Data Science Challenge 2026,
Task 2 — Legal Question Answering.

## Trạng thái

Milestone hiện tại là **M37 — Official Data Adapter and Passage Cleaner**.
`train.json`, `public-official.json` và `selected-contexts.zip` đã được audit;
adapter/cleaner chính thức đã hoàn thành. Full BM25/vector build và model
experiments vẫn là bước riêng tiếp theo.

Thành viên mới nên bắt đầu tại
[`docs/12-TEAM-ONBOARDING.md`](docs/12-TEAM-ONBOARDING.md). Tài liệu này giải
thích toàn bộ offline/online pipeline, package map, CLI, artifact, evaluation và
quy trình nộp Codabench bằng ngôn ngữ thực hành.

Repository giữ lại core độc lập dữ liệu:

- unified legal schemas;
- cleaning, legal structure parsing và chunking;
- BM25, dense retrieval, RRF, reranking và bounded graph retrieval;
- context selection, grounded answer generation và citation verification;
- deterministic Agent workflow;
- FastAPI/UI;
- evaluation, reproducibility và regression gates.

Active data policy là `competition_only`. Runtime chỉ chấp nhận artifact có
lineage `uit-dsc-2026-task2-selected-contexts`. Corpus, index và config cũ từ
nguồn ngoài BTC không còn được hỗ trợ.

## Dữ liệu BTC đã biết

Các file thật đã xác nhận:

- `warmup.json`, `train.json`, `public-official.json`,
  `private-official.json`;
- `selected-contexts.zip` chứa các file `context_*.json`;
- mỗi context có raw fields bắt buộc `id`, `link`, `passage`; `name` optional;
- input là câu hỏi pháp luật tiếng Việt;
- output là câu trả lời văn xuôi tiếng Việt;
- METEOR là metric chính và ROUGE-L là metric phụ.

Contract dữ liệu và checklist audit trước official build được ghi tại
[`docs/13-UIT-DSC-2026-DATA-CONTRACT.md`](docs/13-UIT-DSC-2026-DATA-CONTRACT.md).
Ví dụ BTC dùng numeric context ID và passage có nguyên văn xuống dòng/Unicode;
adapter phải xử lý kiểu raw, còn core tiếp tục dùng unified string ID.

Contract scorer chính thức, checksum, tokenizer, aggregation và khác biệt với
local evaluator nằm tại
[`docs/15-OFFICIAL-SCORING-CONTRACT.md`](docs/15-OFFICIAL-SCORING-CONTRACT.md).

Raw field names của BTC chỉ nằm trong
`legal_agentic_rag.competition.uit_dsc_2026`.

BTC xác nhận chỉ được dùng dữ liệu chính thức, cấm synthetic data kể cả sinh từ
dữ liệu BTC, cho phép preprocessing/indexing/retrieval/fine-tuning trên dữ liệu
chính thức, và yêu cầu đăng ký tên + URL model mã nguồn mở qua Form sắp công bố.
Private test tối đa 3 submission/ngày, và Top 7
phải cung cấp Docker image cùng mã nguồn MIT. Quy trình chi tiết, model approval
register và các điểm cần BTC làm rõ nằm trong
`docs/11-COMPETITION-COMPLIANCE.md`.

## Cài đặt phát triển

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Python tối thiểu: 3.11.

## Configuration

File mẫu:

```text
configs/baseline.example.json
```

File này không chứa đường dẫn dataset hay artifact thật. Để chạy online, cần
một config local trỏ tới artifact BTC đã build và validated. Không commit
config chứa secret hoặc đường dẫn nhạy cảm.

## CLI hiện có

```text
legal-rag-build-competition
legal-rag-batch
legal-rag-submit
legal-rag-score-warmup
legal-rag-validate
legal-rag-prepare-serving
legal-rag-serve
legal-rag-evaluate
legal-rag-compare
```

Build CLI mới chỉ nhận ZIP/thư mục context chính thức, persist theo stage và
resume khi source/config/code identity khớp. Batch CLI ghi output nội bộ có
checkpoint và completeness manifest. Tạo file nộp sau khi batch hoàn tất bằng:

```text
legal-rag-submit --questions <questions.json> --batch <batch-directory> --output <path>/submission.zip
```

Lệnh này không chạy model. Nó xác minh checksum, số lượng và thứ tự ID rồi tạo
ZIP tái tạo được, chứa duy nhất UTF-8 `submission.json`. Root là object ánh xạ
`id` sang `{"answer": string}` theo đúng scorer Codabench thực tế. Các marker
citation nội bộ đã được xác minh như `[E1]` bị loại khỏi answer nộp, nhưng vẫn
được giữ trong batch nội bộ.

Khi benchmark có `reference_answer`, evaluator báo local diagnostic `meteor` và
`rouge_l`. Source BTC xác nhận METEOR dùng whitespace tokens + NLTK defaults,
ROUGE-L dùng vendored ASCII-only default tokenizer và cả hai được macro mean;
PyVi không chạy. Local score vẫn chưa tương đương official vì dùng tokenizer và
METEOR matching khác. Scorer ZIP cũng chưa pin exact NLTK/WordNet versions.

Chấm trực tiếp một bài nộp trên warm-up có đáp án:

```text
legal-rag-score-warmup \
  --references <warmup.json> \
  --submission <submission.zip> \
  --output <new-report-directory>
```

Lệnh không cần GPU, model hay index. Report không lưu nội dung câu hỏi, gold
answer hoặc prediction.

`legal-rag-submit` và `legal-rag-score-warmup` dùng entry point competition nhẹ,
không import FastAPI/Uvicorn hoặc khởi tạo serving runtime.

## Compliance và Docker

- `LICENSE`: source license MIT theo yêu cầu công khai mã nguồn;
- `Dockerfile`: CPU reproducibility scaffold chạy non-root và không chứa data,
  model, artifact hoặc secret;
- `.dockerignore`: chặn các file competition/local lớn khỏi build context;
- `constraints/competition-direct.txt`: pin direct Python dependencies cho
  image M31, chưa phải transitive lock;
- `docs/templates/`: Data Statement, Model Card, private-submission checklist
  và submission ledger.

GPU image, model weights và final reproduction command chỉ được chốt sau khi
BTC xác nhận hạ tầng và model được phép sử dụng.

## Ranh giới competition

```text
Official BTC JSON
→ UIT DSC 2026 adapter
→ typed competition records
→ unified legal schema
→ reusable core pipeline
→ competition answer/output adapter
```

`UitDsc2026DataLoader` hiện hỗ trợ:

- question mapping có hoặc không có reference answer;
- `context_*.json` theo fields BTC đã mô tả;
- duplicate JSON key detection;
- unknown/missing field rejection;
- duplicate context ID rejection;
- đọc trực tiếp ZIP hoặc thư mục đã giải nén;
- canonical corpus SHA-256 không phụ thuộc ZIP packaging;
- mapping context sang unified `LegalDocument`;
- audit cùng normalized/cleaned manifests;
- canonicalize raw integer/string ID sang string;
- giữ raw passage ở normalized artifact và làm sạch có kiểm soát bằng policy
  được pin trong cleaned manifest.

Loader không tự tải dữ liệu, không tạo index và không suy đoán submission
format.

## Việc còn lại

- build parser/chunker/BM25/vector artifacts từ official corpus trong artifact
  root mới và benchmark memory/latency;
- implement và verify scorer METEOR/ROUGE-L tương thích source BTC;
- chốt leakage-safe train/dev strategy trước fine-tuning;
- model benchmark cuối cùng.

Không commit full dataset, model checkpoint, BM25/vector/graph artifact, log,
cache hoặc token.
