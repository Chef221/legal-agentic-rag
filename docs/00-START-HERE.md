# Bắt đầu tại đây

## Trạng thái

Repository hiện giữ các mốc có ích trên nhánh chính `main` (Canonical Main Promotion: **COMPLETE**):

| Mốc | Vai trò | Trạng thái / Score |
|---|---|---|
| M48 | control không fine-tune | giữ để đối chứng (METEOR 0.2685876695, ROUGE-L 0.3631401334) |
| M49 | fine-tune generator bằng official train | giữ training/weights lineage |
| M49.1 | baseline reranker Qwen3 | METEOR 0.382772249, ROUGE-L 0.473653736 |
| **M49.1-JINA35** | **baseline chuẩn hiện tại của repository trên `main` (đã tích hợp source)** | **METEOR 0.406858976, ROUGE-L 0.496260842** |

M45 vẫn được giữ ở tầng offline vì các mốc trên dùng DB/index M45. M46/M47 chỉ
còn số đo lịch sử trong design decisions, không còn notebook/config thực thi.

Pre-promotion GitHub main được lưu giữ tại `archive/main-before-m491-canonical-e1a7916` (`e1a79162c394411ab45349353678be4278dcce71`).
Lịch sử adoption merge trên `main`: `6ebc0e5bde118e8c83e810251557a2f66c69a0d8`.

## Thứ tự đọc

1. [`../AGENTS.md`](../AGENTS.md) — quy tắc bắt buộc.
2. [`../CURRENT-WORK.md`](../CURRENT-WORK.md) — trạng thái baseline canonical và kết quả Public-1000.
3. [`../HANDOFF.md`](../HANDOFF.md) — trạng thái, cách tái lập và hướng tiếp theo.
4. [`21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md`](21-M491-JINA35-PUBLIC1000-FINAL-SUBMISSION.md) — hồ sơ postmortem, hotfix V2, kết quả Codabench và checksum chính thức của M49.1-JINA35.
5. [`18-M491-PUBLIC-RESULT.md`](18-M491-PUBLIC-RESULT.md) — bảng so sánh benchmark giữa các mốc.
6. `13-UIT-DSC-2026-DATA-CONTRACT.md` và
   `15-OFFICIAL-SCORING-CONTRACT.md` — contract BTC.
7. `05-OFFLINE-PIPELINE.md` hoặc `06-ONLINE-PIPELINE.md` tùy phạm vi sửa.
8. `07-UNIFIED-SCHEMA.md` và `08-DESIGN-DECISIONS.md` trước thay đổi kiến trúc.
9. [`19-M491-RERANKER-RESEARCH-STORY.md`](19-M491-RERANKER-RESEARCH-STORY.md) — câu chuyện nghiên cứu reranker M49.1 và validation Clean100.
10. [`20-M491-JINA35-PRODUCTION-INTEGRATION.md`](20-M491-JINA35-PRODUCTION-INTEGRATION.md) — hồ sơ tích hợp kỹ thuật Jina v3.5 (Phase A).

## Bản đồ nhanh

```text
Official corpus
  -> M45 offline artifacts
  -> retrieval / reranker (Qwen3 / Jina v3.5) / context
  -> M48 base generator policy
  -> M49 official-only SFT weights
  -> M49.1-JINA35 canonical runtime policy
  -> verified answer-only submission (1000/1000 valid)
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

## Hướng ưu tiên và việc đã đóng

- **M49.1-JINA35 Public-1000 execution là CLOSED**: 1.000/1.000 câu hợp lệ, submission Codabench đạt ROUGE-L `0.496260842`, METEOR `0.406858976`.
- Việc đã đóng không được chạy lại chỉ để kiểm tra hoặc tái lập. Một lần chạy Public-1000 tương lai chỉ được phép khi có giả thuyết/mục tiêu mới được xác định rõ ràng với execution authority mới.
- **Repository reconciliation & Main promotion:** **HOÀN THÀNH**. Hệ thống M49.1-JINA35 là baseline chuẩn trên nhánh `main`.
