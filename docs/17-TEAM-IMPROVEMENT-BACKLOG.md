# Backlog cải thiện baseline dành cho cả đội

## 1. Mục tiêu

Tài liệu này chuyển các vấn đề M43.1 thành workstream có thể giao việc độc lập.
Mỗi workstream phải tạo bằng chứng định lượng và PR nhỏ, không phải “thử model
mới xem sao”.

Thứ tự ưu tiên:

```text
P0: evaluator + dev protocol
→ P1: retrieval diagnostics/reranker/context
→ P1: generation/verifier/fallback
→ P2: official-only fine-tuning
→ P2: runtime/reproducibility
```

## 2. Definition of Done chung

Một cải tiến chỉ được coi hoàn thành khi có:

1. giả thuyết viết trước;
2. commit và config riêng;
3. input/output lineage;
4. metric quality + error-rate + latency;
5. so sánh với M43.1 trên cùng split;
6. tests deterministic;
7. compliance check;
8. docs/decision cập nhật nếu merge.

## 3. WS-A — Evaluator và leakage-safe experiment protocol

**Ưu tiên:** P0, chặn các workstream quality khác.

**Vấn đề:** Không có vòng lặp local đáng tin cậy. Train/public/warm-up có overlap;
official scorer dùng behavior khác local diagnostic.

**Mục tiêu:** Tạo một split và evaluator đủ ổn định để so sánh thí nghiệm mà
không tối ưu trực tiếp public leaderboard.

**Đầu vào hợp lệ:** `train.json`, source scorer BTC, official corpus. Không dùng
public answer vì không có gold; không tạo synthetic label.

**Công việc:**

- audit duplicate/near-duplicate question giữa các partition;
- định nghĩa group key để các câu trùng/near-duplicate không nằm hai phía;
- tạo train/dev manifest chỉ chứa ID/checksum, không copy data vào Git;
- implement official-compatible scoring mode theo source đã checksum;
- tạo golden vectors để phát hiện drift NLTK/tokenizer;
- report macro score và distribution theo question type/answer length;
- lưu per-ID diagnostic local ngoài Git.

**File bắt đầu:** `evaluation/`, `competition/uit_dsc_2026/warmup_scoring.py`,
`docs/15-OFFICIAL-SCORING-CONTRACT.md`.

**Metric:** scorer parity trên golden cases; zero split leakage theo group key;
METEOR/ROUGE-L macro; bootstrap confidence interval nếu đủ mẫu.

**Nghiệm thu:** cùng prediction/reference cho kết quả deterministic; split
manifest tái tạo được; không có ID/group overlap; tài liệu nêu rõ phần nào parity
và phần nào còn phụ thuộc resource runtime.

## 4. WS-B — Retrieval diagnostics và error taxonomy

**Ưu tiên:** P1.

**Vấn đề:** Không biết câu nào thất bại do BM25, dense, fusion, missing corpus hay
question intent. Official data không có gold evidence.

**Mục tiêu:** Tạo diagnostic không biến answer thành retrieval label giả.

**Công việc:**

- phân loại query bằng rule/metadata không learned: hỏi điều kiện, thẩm quyền,
  thời hạn, mức tiền, thủ tục, định nghĩa, liệt kê;
- log top candidates từng branch và RRF contribution;
- đo lexical answer-term coverage chỉ như diagnostic, không gọi là gold recall;
- lấy một tập review nhỏ từ official train theo protocol đội phê duyệt; nếu gán
  relevance thủ công bị quy định cấm thì chỉ review lỗi định tính, không dùng
  làm training data;
- xác định failure: no relevant context, relevant below top-k, fusion demotion,
  intent mismatch, chunk fragmentation.

**File bắt đầu:** `retrieval/`, `runtime/online.py`, `observability/`,
`evaluation/diagnostics.py`.

**Metric:** candidate diversity, document diversity, branch overlap, answer-term
coverage, latency p50/p95, abstention downstream theo query type.

**Nghiệm thu:** report giúp truy ngược ít nhất mỗi abstention tới stage có khả
năng gây lỗi; không tuyên bố recall khi không có relevance gold.

## 5. WS-C — Reranker ablation

**Ưu tiên:** P1.

**Vấn đề:** Approved cross-encoder đã có implementation nhưng M43 không bật.

**Giả thuyết:** Reranking top fused candidates sẽ giảm intent mismatch và
verification failure với chi phí latency chấp nhận được.

**Công việc:**

- benchmark `hybrid` so với `hybrid_rerank` trên cùng dev split;
- thử candidate counts nhỏ có kiểm soát, ví dụ 20/40/60;
- đo batch size, GPU memory, p50/p95;
- kiểm tra truncation input 512 của reranker;
- không thay generator/context trong cùng experiment.

**File/config:** `reranking/cross_encoder.py`, `retrieval/rerank.py`, config mới
không ghi đè M43.

**Metric:** local METEOR/ROUGE-L, abstention, verification failure, latency,
model error, candidate intent diagnostics.

**Nghiệm thu:** chỉ bật mặc định nếu quality tăng lặp lại trên dev và tổng model
parameter vẫn dưới 4B; ghi rõ cost/query.

## 6. WS-D — Context selection và token budget

**Ưu tiên:** P1.

**Vấn đề:** 1.000/1.000 câu báo `context_budget_exhausted`.

**Giả thuyết:** Context hiện trùng lặp hoặc phân bổ token chưa tốt, làm mất các ý
cần thiết trước generation.

**Công việc:**

- log token của question/instruction/metadata/text riêng;
- ghi evidence nào được giữ/bỏ và lý do;
- deduplicate overlap theo document/article/source block;
- benchmark diversity-aware selection;
- thử budget 3.072/4.096/6.144 nhưng giữ generator/output cố định;
- bảo toàn số, phủ định, điều kiện, exception và citation marker.

**File:** `generation/context_builder.py`, `generation/context_grader.py`, schemas
trace nếu cần.

**Metric:** tỷ lệ exhausted, selected unique documents/articles, context token
utilization, METEOR/ROUGE-L, verification failure, OOM/latency.

**Nghiệm thu:** warning không còn là hằng số mù; tăng budget phải tạo coverage
tốt hơn chứ không chỉ nhiều token hơn.

## 7. WS-E — Generation prompt, answer coverage và model errors

**Ưu tiên:** P1.

**Vấn đề:** Prediction median 92 ký tự so với reference median 1.410; 33 model
errors; 636 answer dưới 30 từ.

**Giả thuyết:** Prompt/style và output cap đang tối ưu độ ngắn, trái với metric
recall và cấu trúc answer tham chiếu.

**Công việc:**

- phân loại 33 `generator:model_error` từ raw internal log;
- prompt yêu cầu trả lời trực tiếp rồi nêu đủ điều kiện/ngoại lệ/căn cứ có trong
  evidence;
- benchmark 256/384/512 output token;
- giữ deterministic decoding ở vòng đầu;
- đo repetition, unsupported claims và answer length distribution;
- retry chỉ khi parser/marker lỗi, tối đa theo workflow contract.

**File:** `generation/model_generator.py`, `generation/prompts.py`,
`generation/transformers_provider.py`, config experiment.

**Metric:** METEOR/ROUGE-L, answer length quantiles, unsupported claims,
verification pass, model error, latency.

**Nghiệm thu:** tăng coverage và metric mà không tăng hallucination/unsupported
claim quá ngưỡng đội định trước.

## 8. WS-F — Verifier và grounded fallback

**Ưu tiên:** P1.

**Vấn đề:** 384 citation failures tạo nhiều generic abstentions. Verifier kiểm
grounding nhưng chưa chắc answer đúng intent.

**Giả thuyết:** Có thể giữ safety và giảm abstention bằng claim-level repair,
không cần tắt verification.

**Công việc:**

- thêm check quan hệ question-answer: ai/cơ quan, bao nhiêu, khi nào, điều kiện,
  thủ tục;
- nếu một số claim fail, loại claim fail và giữ claim supported;
- retry generation một lần với structured verifier feedback;
- nếu vẫn fail, grounded extractive fallback từ evidence tốt nhất;
- generic abstention chỉ khi thật sự không có claim hợp lệ;
- phân biệt `retrieval_insufficient`, `generation_error`,
  `verification_rejected`, `corpus_gap` trong metadata.

**File:** `generation/citation_verifier.py`, `generation/model_generator.py`,
`agent/workflow.py`.

**Metric:** generic abstention, verification pass, unsupported claim, relevance
taxonomy, METEOR/ROUGE-L.

**Nghiệm thu:** không nới invariant citation; mọi non-abstaining claim vẫn trỏ
evidence tồn tại và supported; failure response có lý do chính xác.

## 9. WS-G — Official-only supervised fine-tuning

**Ưu tiên:** P2, chỉ bắt đầu sau WS-A.

**Vấn đề:** Qwen generic chưa học style và độ phủ answer official.

**Ràng buộc:** Chỉ dùng 7.000 official train records và official corpus. Không
synthetic QA, answer, evidence, hard negative hay external data. Model output
fine-tuned vẫn phải được đăng ký/duyệt nếu BTC yêu cầu cập nhật.

**Công việc:**

- chốt leakage-safe train/dev split;
- xác định objective chỉ từ labels thật;
- baseline prompt-only trước fine-tune;
- fine-tune generator bằng official question-answer;
- không dùng public questions để train/tune;
- lưu training config, seed, base revision, adapter/full weights identity;
- kiểm kê parameter theo model gốc; LoRA không giảm parameter count BTC tính.

**Metric:** dev official-compatible METEOR/ROUGE-L, hallucination/verification,
length distribution, reproducibility nhiều seed nếu có tài nguyên.

**Nghiệm thu:** improvement vượt prompt-only baseline trên held-out dev; Data
Statement/Model Card cập nhật; không vi phạm tổng dưới 4B.

## 10. WS-H — Runtime, GPU và reproducibility

**Ưu tiên:** P2 nhưng một số guard cần làm sớm.

**Vấn đề:** P100 incompatibility tạo batch “hoàn tất” nhưng 615 retrieval errors;
T4 x2 chỉ dùng một GPU; dependency transitive chưa freeze đầy đủ.

**Công việc:**

- startup preflight kiểm tra GPU capability với PyTorch build;
- fail fast nếu configured CUDA backend không chạy được;
- batch completion gate bắt buộc zero retrieval model error hoặc explicit
  approved threshold;
- pin environment lock cho Kaggle/T4 và ghi CUDA/driver;
- benchmark single T4 trước khi nghiên cứu multi-GPU;
- xem multi-GPU là experiment riêng, không mặc định cần thiết;
- chuẩn hóa save checkpoint thành private Kaggle Dataset.

**File:** runtime/provider initialization, batch inference, runbook, constraints.

**Metric:** cold startup, p50/p95 query, throughput, peak RAM/VRAM, recovery,
error count.

**Nghiệm thu:** incompatible GPU fail trước câu đầu; resume không làm mất/ghi
đè record; post-run validator chặn submission lỗi.

## 11. Phân công song song đề xuất

| Nhánh | Người/nhóm | Phụ thuộc |
|---|---|---|
| A — evaluator/dev | 1–2 người | Không |
| B — retrieval diagnostics | 1 người | Có thể chạy song song A, nhưng kết luận quality chờ A |
| C — reranker | 1 người | A + benchmark artifact |
| D — context | 1 người | A, có thể chuẩn bị telemetry trước |
| E/F — generator/verifier | 1–2 người | A; phối hợp để tránh sửa cùng file |
| G — fine-tuning | 1–2 người | A hoàn tất, GPU, approval/compliance |
| H — runtime | 1 người | Chạy song song, không thay quality logic |

Nếu đội ít người, thứ tự tốt nhất là `A → B+C → D → E+F → G`, còn H làm các
guard fail-fast ngay khi có thời gian.

## 12. Sprint đầu tiên đề xuất

1. Hoàn tất WS-A split/evaluator contract.
2. Tạo M43 error-analysis report theo question type và stop reason.
3. Chạy ablation hybrid vs approved reranker trên dev, không chạy full public.
4. Thêm context token telemetry và xử lý duplicate context.
5. Prompt/output-length ablation trên cùng dev.
6. Chọn đúng một candidate tốt hơn M43 để chạy public tiếp theo.

Không cần một người “hoàn thành hết” trước khi nhóm tham gia. Ngược lại, M43.1
đã là điểm bàn giao phù hợp: baseline, score, checksums, lỗi và workstream đều đã
có định nghĩa.
