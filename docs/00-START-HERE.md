# Bắt đầu tại đây

## Trạng thái

Repository hiện giữ ba mốc có ích:

| Mốc | Vai trò | Trạng thái |
|---|---|---|
| M48 | control không fine-tune | giữ để đối chứng |
| M49 | fine-tune generator bằng official train | giữ training/weights lineage |
| M49.1 | runtime tốt nhất hiện tại | METEOR 0.382772249 |

M45 vẫn được giữ ở tầng offline vì cả ba mốc trên dùng DB/index M45. M46/M47 chỉ
còn số đo lịch sử trong design decisions, không còn notebook/config thực thi.

## Thứ tự đọc

1. [`../AGENTS.md`](../AGENTS.md) — quy tắc bắt buộc.
2. [`../HANDOFF.md`](../HANDOFF.md) — trạng thái, cách tái lập và hướng tiếp theo.
3. [`18-M491-PUBLIC-RESULT.md`](18-M491-PUBLIC-RESULT.md) — kết quả/audit hiện tại.
4. `13-UIT-DSC-2026-DATA-CONTRACT.md` và
   `15-OFFICIAL-SCORING-CONTRACT.md` — contract BTC.
5. `05-OFFLINE-PIPELINE.md` hoặc `06-ONLINE-PIPELINE.md` tùy phạm vi sửa.
6. `07-UNIFIED-SCHEMA.md` và `08-DESIGN-DECISIONS.md` trước thay đổi kiến trúc.

## Bản đồ nhanh

```text
Official corpus
  -> M45 offline artifacts
  -> retrieval/reranking/context
  -> M48 base generator policy
  -> M49 official-only SFT weights
  -> M49.1 runtime policy
  -> verified answer-only submission
```

Code chính nằm trong `src/legal_agentic_rag`. Config retained nằm trong `configs`.
Notebook/runbook Kaggle nằm trong `notebooks` và `docs/runbooks`.

## Nguyên tắc thực nghiệm

- Mỗi giả thuyết dùng config, output directory và report riêng.
- So paired trên exact dev-200 và exact scorer trước khi chạy public.
- Ưu tiên METEOR; báo thêm ROUGE-L, fallback, latency và answer length.
- Không đổi embedding mà dùng lại vector index cũ.
- Không dùng public answer để train hoặc viết rule theo question ID.
- Không dùng dữ liệu ngoài, synthetic data hoặc model API.
- Không commit dữ liệu, weights, index, checkpoint hay submission.

## Hướng ưu tiên

M49.1 có 900/1.000 generator fallbacks nhưng vẫn tăng mạnh score nhờ retrieval và
top-evidence fallback. Hướng kế tiếp phải giữ M49.1 làm control, rồi tách riêng hai
thử nghiệm:

1. question-aware extractive trimming/answer-length selection;
2. contract inference khớp answer-only SFT mà vẫn grounded.

Không loại top-evidence fallback chỉ để giảm warning count; phải chứng minh METEOR
tăng trên split cố định.
