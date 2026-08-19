# Phase A — Current-System Census

## 1. Mục tiêu

Phase A đo **hệ thống hiện tại trước khi đơn giản hóa kiến trúc**. Không thay đổi
retrieval, routing, evidence selection, generator, verifier, artifacts hoặc model
identity. Kết quả của Phase A là baseline kỹ thuật để Phase B có thể xóa hoặc
đơn giản hóa component mà vẫn biết chính xác score, reliability, routing và cost
đã thay đổi như thế nào.

Benchmark sử dụng lại leakage-safe `development.json` 991 câu đã được dùng trong
các milestone M44–M49. Tập này là **historical engineering benchmark**, không phải
untouched holdout và kết quả không được diễn giải như bằng chứng generalization.

## 2. Snapshot cần đo

Profile census giữ nguyên active M49.6 reliability behavior:

- official corpus artifacts M40/M43 serving package;
- E5 multilingual small;
- BM25 + dense + RRF;
- approved mMARCO cross-encoder reranker;
- `candidate_k=40`, `top_k=8`;
- query understanding + multi-query + adaptive routing **giữ nguyên để đo current behavior**;
- evidence document cap `2`;
- tối đa 5 evidence / 3,072 context tokens;
- pretrained `Qwen/Qwen2.5-3B-Instruct`;
- 384 generated tokens;
- M49.5 schema recovery;
- M49.6 missing-required-field correction;
- numeric repair + supported-claim salvage;
- deterministic citation/claim verification;
- semantic verifier disabled.

Config: `configs/phase-a-current-system-census-kaggle.example.json`.

Không được chỉnh graph hoặc adaptive routing trước census. Một mục tiêu của census
là đo xem graph route có thực sự bị chạm trên 991 câu hay không.

## 3. Preconditions

Cần có trên môi trường GPU:

1. repository checkout đúng commit/branch muốn census;
2. official serving artifact package tại đường dẫn config;
3. exact historical `development.json` 991 records, source SHA-256:
   `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`;
4. NLTK 3.7 + exact local WordNet/OMW resources đã dùng cho M44 official-compatible scoring;
5. không sửa content của `development.json` hoặc artifacts giữa batch và scoring.

## 4. Environment setup

```bash
set -euo pipefail
export PYTHONUNBUFFERED=1

python -m pip install -e ".[official-scoring]"

CONFIG=configs/phase-a-current-system-census-kaggle.example.json
QUESTIONS=/kaggle/working/development.json
BATCH=/kaggle/working/phase-a-current-system-census-batch
ANALYSIS=/kaggle/working/phase-a-current-system-census-analysis
READINESS=/kaggle/working/phase-a-current-system-census-readiness
SUBMISSION=/kaggle/working/phase-a-current-system-census-submission.zip
SCORE=/kaggle/working/phase-a-current-system-census-score
CENSUS=/kaggle/working/phase-a-current-system-census.json
```

Trước khi chạy GPU batch, xác nhận question bytes:

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path
p = Path('/kaggle/working/development.json')
print(sha256(p.read_bytes()).hexdigest())
PY
```

Expected:

```text
8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8
```

Nếu khác, dừng run.

## 5. Run the 991-question current-system batch

```bash
legal-rag-batch \
  --config "$CONFIG" \
  --questions "$QUESTIONS" \
  --output "$BATCH" \
  --progress-interval 1 \
  2>&1 | tee /kaggle/working/phase-a-current-system-census.log
```

Batch runner checkpoint từng record sau flush + fsync. Nếu notebook bị gián đoạn,
chạy lại **đúng command, config, code version và output directory** để resume.
Không sửa config rồi resume vào cùng directory.

## 6. Validate and analyze the completed batch

```bash
legal-rag-analyze-batch \
  --batch "$BATCH" \
  --output "$ANALYSIS"
```

Report này phải cung cấp ít nhất:

- record count / unique ID count;
- insufficient-evidence count;
- generator model errors;
- structured generation failure taxonomy;
- schema recovery outcomes;
- missing-field correction outcomes;
- retrieval model errors;
- stop-reason distribution;
- citation-verification failures + claim error taxonomy;
- numeric repair / supported-claim salvage outcomes;
- context selection reason counts;
- Agent latency distribution.

## 7. Build the Phase A routing/context census

Existing batch analysis intentionally không tổng hợp final strategy và từng
retrieval-tool invocation. Phase A thêm một standalone content-free summarizer:

```bash
python scripts/phase_a_census.py \
  --batch "$BATCH" \
  --output "$CENSUS"
```

Census report chỉ lưu aggregate/statistical fields, không lưu question, answer,
legal passage hoặc citation text. Nó đo thêm:

- final retrieval strategy distribution;
- retrieval tool attempt/success counts;
- số record từng chạm `graph_search`;
- số record kết thúc với `retrieval_strategy=graph`;
- evidence-count distribution;
- estimated context-token distribution;
- answer character/whitespace-word distribution;
- Agent latency distribution.

Hai field graph là gate đặc biệt cho Phase B:

```text
graph_touched_record_count
graph_final_strategy_record_count
```

Nếu một trong hai > 0, current adaptive routing thực sự đang sử dụng graph trên
benchmark dù official graph artifact zero-edge. Không được suy diễn score impact
cho đến khi Phase B chạy controlled ablation.

## 8. Create a census-only readiness report

Phase A không dùng readiness gate để tuyên bố candidate tốt. Policy census chỉ
xác nhận batch hoàn chỉnh và yêu cầu context-selection trace ở toàn bộ records;
các quality thresholds khác cố tình permissive để ta quan sát failure thật thay
vì chặn formatter.

```bash
legal-rag-check-batch \
  --questions "$QUESTIONS" \
  --batch "$BATCH" \
  --policy configs/phase-a-census-readiness-policy.json \
  --output "$READINESS"
```

## 9. Format and score exactly the same 991 records

```bash
legal-rag-submit \
  --questions "$QUESTIONS" \
  --batch "$BATCH" \
  --readiness-report "$READINESS" \
  --output "$SUBMISSION"

legal-rag-score-warmup \
  --references "$QUESTIONS" \
  --submission "$SUBMISSION" \
  --output "$SCORE" \
  --metric-mode official_compatible
```

Primary census quality number: **METEOR**. Secondary: **ROUGE-L**.
Retain scorer provenance and the existing warning that exact parity depends on
exact WordNet/OMW bytes.

## 10. Required evidence bundle

Giữ local/ignored, không commit raw outputs:

```text
phase-a-current-system-census-batch/
phase-a-current-system-census-analysis/
phase-a-current-system-census-readiness/
phase-a-current-system-census-submission.zip
phase-a-current-system-census-score/
phase-a-current-system-census.json
phase-a-current-system-census.log
```

Sau khi run xong, review tối thiểu các số sau trước khi mở Phase B:

| Nhóm | Required values |
|---|---|
| Identity | code version, config hash, question SHA, records SHA |
| Quality | METEOR, ROUGE-L |
| Reliability | verified/abstain, generator errors, retrieval errors, citation failures |
| Recovery | schema recovery, missing-field correction, numeric repair, supported-claim salvage |
| Routing | final strategies, retrieval-tool attempts, graph touched/final counts |
| Context | selected evidence, selection reasons, token estimate |
| Output shape | mean/p50/p95 answer chars and whitespace words |
| Cost | mean/p50/p95/max Agent latency |

## 11. Phase A exit gate

Phase A hoàn thành khi:

1. batch đủ 991 IDs và manifest/checksum hợp lệ;
2. `batch_analysis.json` được tạo;
3. `phase-a-current-system-census.json` được tạo;
4. submission formatter chạy trên đúng batch;
5. official-compatible METEOR + ROUGE-L được ghi nhận;
6. routing distribution và graph-touch count được review;
7. không có code/config change quality-facing nào xảy ra giữa census batch và score.

Không promote candidate và không xóa component trong Phase A.

Phase B chỉ bắt đầu sau khi baseline này được đóng băng. Experiment đầu tiên dự
kiến là loại graph khỏi competition routing/runtime surface và tắt adaptive graph
routing, sau đó rerun cùng 991 questions để đo score/reliability/latency delta.
