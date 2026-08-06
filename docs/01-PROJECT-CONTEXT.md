# 01. Project Context

## 1. Competition

Dự án phục vụ UIT Data Science Challenge 2026, Task 2 — Legal Question
Answering. Hệ thống nhận câu hỏi pháp luật tiếng Việt và sinh câu trả lời văn
xuôi tiếng Việt dựa trên legal context.

BTC đã mô tả:

- khoảng 8.500 văn bản/context pháp luật;
- khoảng 10.000 câu hỏi;
- các phase warm-up, public test và private test;
- METEOR là metric chính;
- ROUGE-L là metric phụ.

BTC đã cung cấp source scoring chính thức. ZIP checksum, exact tokenizer,
aggregation, runtime I/O và các dependency chưa pin được ghi tại
`docs/15-OFFICIAL-SCORING-CONTRACT.md`. Local evaluator hiện vẫn là diagnostic,
không tương đương scorer đó.

Thể lệ được cung cấp ngày 2026-08-01 còn xác nhận:

- chỉ dùng dữ liệu chính thức do BTC phát hành;
- không gán nhãn thủ công, không external data augmentation;
- model mã nguồn mở mới phải được đề xuất trước hạn private test ít nhất 10
  ngày;
- private test giới hạn 3 submission/ngày;
- Top 7 phải cung cấp Docker image, source MIT và bằng chứng tái lập;
- mỗi bài nộp phải có Data Statement và Model Card.

Chi tiết compliance và các điểm chưa rõ nằm trong
`docs/11-COMPETITION-COMPLIANCE.md`.

## 2. Active Data Scope

Active repository chỉ dùng dữ liệu chính thức do BTC Task 2 cung cấp. Chính
sách runtime là `competition_only` và không cho phép external corpus.

Các raw contract hiện biết:

```text
question_id → {question, answer?}
context → {id, name, link, passage}
```

Chi tiết authoritative-in-repository nằm tại
`docs/13-UIT-DSC-2026-DATA-CONTRACT.md`. Ví dụ BTC dùng numeric context ID,
`name` dạng slug và passage chứa nguyên văn pháp lý với xuống dòng/Unicode; raw
ID cần được adapter canonicalize sang string sau khi audit corpus thật.

`warmup.json` cung cấp question và reference answer. Nó không cung cấp
retrieval relevance ID, vì vậy không được tạo relevance label giả.

## 3. Architecture Principle

```text
Official competition data
→ competition adapter
→ typed competition record
→ unified legal schema
→ offline/online core
→ competition output adapter
```

Core không biết raw field names của BTC và không hard-code dataset path,
revision, backend hoặc model.

## 4. Preserved Core

- unified schemas and manifests;
- legal cleaning, parsing and chunking;
- lexical/dense/hybrid/graph retrieval;
- reranking and evidence selection;
- grounded generation and citation verification;
- bounded deterministic Agent workflow;
- API/UI and observability;
- reproducible evaluation and regression gates.

## 5. Current Reset Boundary

M25 loại source, adapter, raw audit, normalizer, build runtime, fixtures,
profiles và dependency chỉ phục vụ corpus cũ. External artifacts không bị xóa
tự động nhưng bị competition runtime từ chối.

M37 đã audit archive thật, hoàn thiện numeric-ID/optional-title adapter và tạo
raw-preserving normalized + dataset-specifically cleaned manifests. Full
parser/chunker/BM25/vector build vẫn là bước riêng chưa chạy.

## 6. Legal Safety

Output chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chuyên nghiệp. Hệ
thống phải giữ abstention, citation verification và provenance; không được tự
khẳng định hiệu lực hiện hành nếu evidence không chứng minh điều đó.
