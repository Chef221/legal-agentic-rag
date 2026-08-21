# Vietnamese Legal RAG — UIT DSC 2026 Task 2

Hệ thống RAG trả lời câu hỏi pháp luật Việt Nam bằng dữ liệu chính thức của UIT
Data Science Challenge 2026 Task 2.

## Trạng thái hiện tại

- Candidate tốt nhất đã đo: **M49.1**.
- Public-compatible score: **METEOR `0.382772249`**, **ROUGE-L `0.473653736`**.
- Control còn giữ: **M48** — METEOR `0.2685876695`, ROUGE-L `0.3631401334`.
- M49 là generator Qwen3.5-2B được fine-tune bằng official `train.json`; M49.1 dùng
  chính weights đó với runtime policy mới.
- M48/M49/M49.1 dùng chung DB/index M45 đã build từ canonical
  `selected-contexts.zip`.

Thành viên mới bắt đầu tại
[`docs/00-START-HERE.md`](docs/00-START-HERE.md), sau đó đọc
[`HANDOFF.md`](HANDOFF.md). Kết quả và audit M49.1 nằm tại
[`docs/18-M491-PUBLIC-RESULT.md`](docs/18-M491-PUBLIC-RESULT.md).

## Pipeline

```text
selected-contexts.zip
  -> audit + normalize + legal chunking
  -> BM25 + dense vector + graph artifacts (M45)

question
  -> BM25/dense retrieval
  -> RRF + Qwen3 reranker
  -> bounded context selection
  -> Qwen3.5-2B generation
  -> deterministic grounding/citation checks
  -> answer-only submission.json
```

M49 adds official-only response SFT. M49.1 aligns online output/repetition policy
while retaining M45 retrieval, M48 recovery and M49 weights.

## Retained Kaggle workflows

- M45 artifact build: `notebooks/M45_KAGGLE_01_BUILD_DB.ipynb`.
- M48 dev/public control: `notebooks/M48_KAGGLE_01_DEV_EVAL.ipynb` and
  `notebooks/M48_KAGGLE_02_PUBLIC_SUBMISSION.ipynb`.
- M49 training/dev: `notebooks/M49_KAGGLE_01_TRAIN_GENERATOR.ipynb` and
  `notebooks/M49_KAGGLE_02_DEV_EVAL.ipynb`.
- M49.1 dev/public: `notebooks/M491_KAGGLE_01_DEV_EVAL.ipynb` and
  `notebooks/M491_KAGGLE_02_PUBLIC_SUBMISSION.ipynb`.

Các script dùng chung nằm tại `notebooks/kaggle_candidate_dev_common.py` và
`notebooks/kaggle_public_submission_common.py`. M46/M47 không còn là executable
candidate trong repository.

Sau khi commit source sạch, tạo ZIP riêng để upload Kaggle bằng
`python scripts/package_kaggle_source.py m491` (hoặc `m48`, `m49`). File sinh ra
trong `dist/` và không được commit.

## Quy định dữ liệu và model

- Chỉ dùng dữ liệu chính thức BTC: `selected-contexts.zip`, `train.json`,
  `warmup.json` và test chính thức.
- Không dùng external corpus, synthetic QA/evidence, model API hoặc chỉnh tay đáp
  án public.
- Tổng tham số mọi model trong hệ thống phải dưới 4 tỷ. Inventory hiện tại khoảng
  3.466B.
- Fine-tuning chỉ dùng partition train của split chống leakage; dev/holdout không
  đi vào optimizer.
- Submission phải là `submission.zip` chứa duy nhất UTF-8 `submission.json` dạng
  object `question_id -> {"answer": string}`.

Xem `AGENTS.md`, `docs/11-COMPETITION-COMPLIANCE.md`,
`docs/13-UIT-DSC-2026-DATA-CONTRACT.md` và
`docs/15-OFFICIAL-SCORING-CONTRACT.md` trước khi thay đổi pipeline.

## Cài đặt phát triển

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Python tối thiểu: 3.11. Unit test mặc định không tải model lớn và không gọi mạng.

## CLI chính

```text
legal-rag-build-competition
legal-rag-prepare-serving
legal-rag-batch
legal-rag-submit
legal-rag-score-warmup
legal-rag-validate
```

## Không đưa lên Git

Không commit full dataset, scorer archive, source ZIP đóng gói, model weights,
BM25/vector/graph artifacts, Kaggle checkpoint, `submission.zip`, cache, log hoặc
secret. Repository chỉ giữ code, config, notebook, test, tài liệu và fixture nhỏ.
