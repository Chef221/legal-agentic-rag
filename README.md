# Vietnamese Legal Agentic RAG

Hệ thống Retrieval-Augmented Generation cho UIT Data Science Challenge 2026,
Task 2 — Legal Question Answering.

## Trạng thái

Baseline chất lượng đối chứng vẫn là **M43.1 — Qwen3B public baseline**. Code
hiện tại là `0.44.6`, bổ sung evaluator, development protocol, retrieval
diagnostics M44 nhưng chưa
thay đổi retrieval/model của baseline.
Hệ thống đã build official-only BM25/vector artifacts, chạy đủ 1.000 câu public,
tạo submission hợp lệ và được Codabench chấm:

- METEOR: `0.07862292376534387`;
- ROUGE-L: `0.16735433212043324`;
- 1.000/1.000 ID, không answer rỗng;
- 425 abstention, 384 citation-verification failure và 33 generator model error.

Đây là baseline vận hành hoàn chỉnh nhưng chất lượng còn yếu. Thành viên mới
phải bắt đầu tại [`docs/00-START-HERE.md`](docs/00-START-HERE.md), sau đó đọc:

- [`docs/16-M43-BASELINE-POSTMORTEM.md`](docs/16-M43-BASELINE-POSTMORTEM.md)
  để hiểu score, lỗi và root cause;
- [`docs/17-TEAM-IMPROVEMENT-BACKLOG.md`](docs/17-TEAM-IMPROVEMENT-BACKLOG.md)
  để nhận workstream;
- [`docs/12-TEAM-ONBOARDING.md`](docs/12-TEAM-ONBOARDING.md) và
  [`docs/14-SYSTEM-ARCHITECTURE-REFERENCE.md`](docs/14-SYSTEM-ARCHITECTURE-REFERENCE.md)
  để hiểu package, I/O và kiến trúc as-built.

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

- `warmup.json`, `train.json`, `public-official.json`;
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
chính thức, và chỉ cho phép model đã đăng ký/được duyệt. Ba model baseline E5,
mMARCO MiniLM reranker và Qwen2.5-3B đã được người dùng xác nhận BTC duyệt; bằng
chứng duyệt gốc phải được giữ trong hồ sơ đội. Private test tối đa 3
submission/ngày, và Top 7
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
legal-rag-prepare-dev
legal-rag-diagnose-retrieval
legal-rag-validate
legal-rag-prepare-serving
legal-rag-serve
legal-rag-evaluate
legal-rag-compare
```

Build CLI nhận ZIP/thư mục context chính thức, persist theo stage và
resume khi source/config/code identity khớp. Có thể dừng ngay sau parser/chunker
mà không khởi tạo embedding model hoặc build index:

```text
legal-rag-build-competition \
  --config <config.json> \
  --source <selected-contexts.zip> \
  --through document_processing
```

Lần chạy sau với cùng source, config, code version và không truyền `--through`
sẽ xác minh blocks/chunks đã persist rồi resume BM25, vector và validation.
Batch CLI ghi output nội bộ có
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

Tạo local train/dev split chống leakage từ dữ liệu chính thức:

```text
legal-rag-prepare-dev \
  --train <train.json> \
  --holdout <warmup.json> \
  --holdout <public-official.json> \
  --output <new-split-directory>
```

Lệnh giữ nguyên từng question/answer, gom exact/near-duplicate theo nhóm, cách ly
mọi train record trùng holdout vào `quarantined.json`, và ghi checksum/ID vào
`split_manifest.json`. Output là dữ liệu local bị `.gitignore` loại trừ, không
phải retrieval labels và không được commit.

Chấm trực tiếp một bài nộp trên warm-up/dev có đáp án:

```text
legal-rag-score-warmup \
  --references <warmup.json> \
  --submission <submission.zip> \
  --output <new-report-directory> \
  --metric-mode official_compatible
```

Chế độ `official_compatible` cần cài `.[official-scoring]`, dùng NLTK 3.7
METEOR trên whitespace tokens và ROUGE-L ASCII đúng scorer đã audit. Không tự
tải WordNet; report ghi scorer checksum và cảnh báo parity tuyệt đối còn phụ
thuộc exact WordNet bytes. Mặc định `diagnostic` cũ vẫn tồn tại để tương thích.
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

Baseline không cần được một người tiếp tục “hoàn thiện hết” trước khi cả đội
tham gia. Các việc ưu tiên hiện tại là:

- chốt leakage-safe train/dev protocol và official-compatible evaluator;
- phân tích retrieval theo branch và benchmark approved reranker;
- sửa context selection vì 100% câu chạm budget;
- tăng answer coverage nhưng vẫn giữ grounding;
- giảm generic abstention bằng claim-level repair/fallback có kiểm chứng;
- chỉ sau đó mới fine-tune bằng official train data.

Chi tiết đầu vào, metric, file cần sửa và tiêu chí nghiệm thu của từng nhánh nằm
trong `docs/17-TEAM-IMPROVEMENT-BACKLOG.md`.

Chạy retrieval diagnostics trên cùng một tập câu hỏi mà không tạo relevance
label giả:

```text
legal-rag-diagnose-retrieval \
  --config <online-config.json> \
  --questions <development.json> \
  --output <new-diagnostics-directory> \
  --top-k 20 \
  --candidate-k 100
```

Lệnh chạy riêng BM25, dense và hybrid, rồi ghi branch overlap, document
diversity, explicit-reference match, latency, warning/error và optional
answer-term coverage. Answer-term coverage chỉ là dấu hiệu chẩn đoán ở mức từ;
nó không phải retrieval relevance label hay Recall@k. Report không chứa câu hỏi,
đáp án tham chiếu hoặc toàn văn pháp luật và không ghi đè output cũ.

M44.3 retrieval-only reranker ablation dùng cùng lệnh với
`--include-reranker`. Mỗi candidate count 20/40/60 phải dùng output directory
riêng. Chỉ candidate qua gate diagnostics mới được chạy full generation và chấm
METEOR/ROUGE-L.

Từ `0.44.5`, local Transformers provider dùng `accelerate` low-memory loading
trước khi chuyển Qwen sang CUDA và log riêng tokenizer, weights và device transfer.
Điều này xử lý bottleneck startup đã quan sát trên Kaggle, không thay model,
prompt, ranking hay output contract.

Từ `0.44.6`, `legal-rag-batch` nhận `--progress-interval`. Kaggle runbook dùng
shell foreground với interval `1`, do đó log streaming chỉ báo những câu đã được
durably checkpoint, không phải tiến độ ước lượng.

Từ `0.44.3`, comparison runtime enrich query một lần, chạy sparse/dense branch
một lần ở `candidate_k`, rồi tái sử dụng cùng candidate cho RRF và reranker.
Phép tối ưu chỉ loại bỏ backend work trùng lặp; ranking contract, model,
artifact và persisted diagnostic schema không đổi.

M44.4 đưa candidate-k 20 và 40 qua A/B end-to-end trên cùng 991 development
questions. Hai profile Qwen `hybrid_rerank` giữ nguyên mọi tham số ngoài
`candidate_k`; quyết định cuối chỉ dựa trên METEOR chính và ROUGE-L phụ theo
scorer-compatible mode, không dựa riêng vào retrieval diagnostics.

Không commit full dataset, model checkpoint, BM25/vector/graph artifact, log,
cache hoặc token.
