# Bắt đầu tại đây — UIT-SV5T LegalQA

## 1. Mục đích

Đây là trang vào chính của repository dành cho thành viên mới. Baseline M43.1
đã chạy đủ 1.000 câu public và nộp thành công lên Codabench. Từ thời điểm này,
repository chuyển từ giai đoạn “một người dựng khung” sang giai đoạn “cả đội
thực nghiệm và cải thiện có kiểm soát”.

Không đọc riêng README rồi sửa code. Tùy vai trò, hãy theo lộ trình dưới đây.

## 2. Trạng thái thật hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Code | `0.44.2`; quality control baseline remains `0.43.1` at `96e6d5a` |
| Corpus | 8.532 context chính thức, competition-only |
| Retrieval units | 330.768 legal chunks, exact E5 window tối đa 512 token |
| Index | SQLite FTS5 BM25 + exact dense vector 384 chiều |
| Online M43 | Hybrid RRF, không reranker, không graph expansion |
| Generator | `Qwen/Qwen2.5-3B-Instruct`, local Transformers |
| Public batch | 1.000/1.000 ID, không answer rỗng |
| Codabench | METEOR `0.07862292376534387`, ROUGE-L `0.16735433212043324` |
| Kết luận | Vận hành end-to-end thành công; chất lượng còn yếu |

Điểm số thấp không có nghĩa repository “chưa chạy”. Nó cho biết ranh giới kỹ
thuật đã hoạt động nhưng retrieval, context selection, generation và policy
abstention chưa phù hợp metric/câu trả lời tham chiếu.

## 3. Sơ đồ một trang

```text
selected-contexts.zip
  → official adapter/audit/cleaner
  → legal parser
  → exact-token legal chunker
  → BM25 + E5 vector + zero-edge graph
  → manifests/checksums/validation

public-official.json
  → query understanding
  → BM25 + dense retrieval
  → RRF fusion
  → evidence selection/context budget
  → Qwen2.5-3B generation
  → claim/citation verification
  → internal AnswerResponse
  → competition renderer
  → submission.zip
  → Codabench METEOR/ROUGE-L
```

M43 có code reranker, graph retrieval và semantic verifier trong repository,
nhưng cấu hình public baseline không kích hoạt chúng. “Có implementation” không
đồng nghĩa “đã được chứng minh cải thiện điểm”.

## 4. Thứ tự đọc

### Đọc trong 15 phút

1. File này.
2. [`16-M43-BASELINE-POSTMORTEM.md`](16-M43-BASELINE-POSTMORTEM.md) — baseline
   chạy ra sao và yếu ở đâu.
3. [`17-TEAM-IMPROVEMENT-BACKLOG.md`](17-TEAM-IMPROVEMENT-BACKLOG.md) — nhận
   một workstream có đầu vào, metric và tiêu chí nghiệm thu.

### Đọc trước khi sửa module

1. [`12-TEAM-ONBOARDING.md`](12-TEAM-ONBOARDING.md) — bản đồ package và luồng
   thực hành.
2. [`14-SYSTEM-ARCHITECTURE-REFERENCE.md`](14-SYSTEM-ARCHITECTURE-REFERENCE.md)
   — kiến trúc as-built và I/O từng lớp.
3. `07-UNIFIED-SCHEMA.md` và `08-DESIGN-DECISIONS.md` — contract và lý do.
4. Tài liệu pipeline tương ứng: `05-OFFLINE-PIPELINE.md` hoặc
   `06-ONLINE-PIPELINE.md`.

### Đọc trước khi chạy competition

1. `11-COMPETITION-COMPLIANCE.md`.
2. `13-UIT-DSC-2026-DATA-CONTRACT.md`.
3. `15-OFFICIAL-SCORING-CONTRACT.md`.
4. `runbooks/KAGGLE-M43-QWEN3B.md`.

## 5. Nơi sửa theo loại vấn đề

| Muốn cải thiện | Package/file bắt đầu | Không sửa trước khi đo |
|---|---|---|
| Parse/chunk | `offline/parsing`, `offline/chunking` | Không rebuild vector cũ bằng lineage mới |
| BM25 | `indexing/bm25`, `retrieval/sparse.py` | Không cộng raw score với dense |
| Dense | `embeddings`, `indexing/vector`, `retrieval/dense.py` | Không đổi model mà giữ vector cũ |
| Fusion | `retrieval/fusion.py`, `retrieval/fixed.py` | Không đánh giá chỉ bằng vài câu nhìn tay |
| Reranker | `reranking`, `retrieval/rerank.py` | Không rerank toàn corpus |
| Context | `generation/context_builder.py` | Không cắt mất số, phủ định, điều kiện |
| Generator | `generation/model_generator.py`, provider | Không dùng API hoặc dữ liệu ngoài BTC |
| Verifier | `generation/citation_verifier.py` | Không biến ID-check thành semantic claim |
| Agent | `agent/workflow.py` | Không tăng retry vô hạn |
| Batch/submission | `competition/uit_dsc_2026` | Không sửa scorer contract theo phỏng đoán |

## 6. Quy tắc phối hợp

- Mỗi thí nghiệm có branch, config và output directory riêng.
- Ghi rõ commit, config hash, model revision, corpus revision và seed.
- Chỉ dùng dữ liệu BTC; không synthetic data, không external corpus, không API.
- Không chỉnh tay public answer để tối ưu leaderboard.
- Một PR nên kiểm tra một giả thuyết chính; không gộp nhiều thay đổi khiến không
  biết điểm tăng/giảm vì đâu.
- Không commit dataset, model weights, artifacts, batch outputs hoặc token.
- Mọi thay đổi schema/architecture/pipeline phải cập nhật tài liệu nguồn sự thật.

## 7. Baseline phải được giữ lại

M43.1 là mốc so sánh, không phải code “bỏ đi”. Mọi cải tiến phải trả lời:

1. metric local leakage-safe thay đổi thế nào;
2. tỷ lệ abstention/model error/verification failure thay đổi thế nào;
3. latency và tài nguyên thay đổi thế nào;
4. có còn đáp ứng compliance và tổng model dưới 4 tỷ tham số không;
5. có thể tái lập từ commit + config + artifact lineage hay không.

Nếu chưa trả lời đủ năm câu này, thay đổi chưa sẵn sàng merge.
