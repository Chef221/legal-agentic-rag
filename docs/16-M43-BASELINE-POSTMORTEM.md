# M43.1 Public Baseline — Báo cáo hậu kiểm

## 1. Executive summary

M43.1 đã chứng minh hệ thống có thể xử lý official corpus, load artifacts, chạy
hybrid retrieval, sinh answer bằng model local, kiểm tra citation, checkpoint
batch, tạo đúng submission và được Codabench chấm thành công.

Kết quả chính thức:

| Metric | Giá trị |
|---|---:|
| METEOR — metric xếp hạng chính | `0.07862292376534387` |
| ROUGE-L — metric phụ | `0.16735433212043324` |

Đây là baseline vận hành hợp lệ nhưng baseline chất lượng yếu. Nguyên nhân lớn
nhất không phải lỗi định dạng submission mà là chuỗi vấn đề chất lượng:

```text
candidate chưa đủ đúng
→ context bị giới hạn ở mọi câu
→ generator tạo câu quá ngắn hoặc lỗi
→ verifier loại nhiều câu
→ 42,5% output trở thành cùng một câu abstention
→ recall token rất thấp so với reference dài
→ METEOR thấp
```

## 2. Identity và provenance

### 2.1 Code và dữ liệu

| Identity | Giá trị |
|---|---|
| Project version | `0.43.1` |
| Baseline commit | `96e6d5a` |
| Official corpus name | `uit-dsc-2026-task2-selected-contexts` |
| Corpus canonical revision | `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e` |
| Public question SHA-256 | `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5` |
| Context count | 8.532 |
| Context có nội dung | 8.512 |
| Legal chunks | 330.768 |

Không có AIO/external lineage trong serving package M43. Artifact lịch sử trên
ổ đĩa cá nhân không phải input của runtime này.

### 2.2 Model active

| Vai trò | Model/revision | Active trong M43 |
|---|---|---|
| Embedding | `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3` | Có |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@1427fd652930e4ba29e8149678df786c240d8825` | Không |
| Generator | `Qwen/Qwen2.5-3B-Instruct@a1d308dfcc03e09da285d49d912439a655a571e8` | Có |
| Semantic verifier | Không có model active | Không |

Người dùng đã xác nhận cả ba model trên đã được BTC duyệt. Đội vẫn phải giữ
bằng chứng duyệt gốc ngoài repository và kiểm kê parameter/license khi tạo
release candidate.

### 2.3 Artifact

| Artifact | Thông tin |
|---|---|
| Serving package | `uit-dsc-2026-task2-serving-v0430.tar.gz` |
| Package SHA-256 | `90d4d211a20f6d3a6f894d8dd33c0f187fcf141c1bcbc3814d8dcc7e003e729c` |
| BM25 | SQLite FTS5, khoảng 2.372,52 MB trên Kaggle |
| Vector matrix | `(330768, 384)`, float32, khoảng 484,52 MB |
| Vector chunk metadata | khoảng 1.392,81 MB |
| Serving metadata | SQLite, khoảng 43,86 MB |
| Graph | zero-edge; không có relationship labels chính thức |

M40 vector SHA-256:
`591840809c024ad697a88efc8623ba0708ed9aa3727f0866c13362de57636e24`.
Vector chunk metadata SHA-256:
`c1be6044019335753edd432196653bd4395707aa148af5bc695e9e652b605223`.

## 3. Cấu hình thực tế

M43 dùng `configs/uit-dsc-2026-task2-qwen3b-kaggle.example.json`:

- E5 document/query embedding, exact dense search trên CUDA;
- `top_k=8`, `candidate_k=60`, RRF hybrid;
- query understanding và tối đa ba query variants;
- không cross-encoder reranker;
- không graph expansion;
- tối đa năm evidence;
- context budget 3.072 token;
- Qwen input tối đa 6.144 token, output tối đa 256 token;
- deterministic decoding (`temperature=0`);
- một structured retry;
- rule-based claim/citation verifier;
- semantic verifier tắt;
- Agent chỉ có strategy `hybrid`, không rewrite/retry retrieval.

Điểm quan trọng: repository có nhiều khả năng hơn cấu hình này, nhưng official
score chỉ phản ánh đúng active path trên.

## 4. Lịch sử chạy và sự cố

### 4.1 Hotfix workflow

Code `0.43.0` có thể tạo `AgentRunResult` không hợp lệ khi generator tự abstain.
D088 và version `0.43.1` sửa invariant: chỉ `answer_verified` mới được trả
non-abstaining response; generator abstention kết thúc fail-closed.

### 4.2 Checkpoint 385 câu

Lần chạy đầu hoàn tất 385/1.000 câu và được lưu thành checkpoint. Checkpoint có
code version `0.43.1`, exact question count 1.000 và 385 ID đã hoàn tất.

### 4.3 Lần resume P100 bị loại

Kaggle P100 có compute capability `sm_60`, trong khi PyTorch mặc định lúc đó
chỉ hỗ trợ `sm_70+`. Process vẫn tiếp tục nhưng 615/615 câu suffix phát sinh
`retrieval:model_error` và abstain. Việc “batch đạt 100%” không chứng minh output
hợp lệ. Toàn bộ suffix này bị loại.

### 4.4 Lần resume T4 hợp lệ

Suffix 386–1.000 được chạy lại trên T4 x2. Pipeline hiện chỉ dùng GPU0; GPU1
không tự động được tận dụng. Lần chạy hợp lệ có:

- 0 retrieval model error;
- 17 generator model error trong suffix;
- đủ 1.000 records và manifest hoàn chỉnh khi ghép checkpoint;
- cold first resumed query khoảng 87,7 giây do load model;
- warm tail: BM25 khoảng 3,7–5,0 giây, dense matrix score khoảng 33–45 ms,
  generation khoảng 6,6–7,2 giây, E2E khoảng 11–12 giây;
- 615 câu suffix mất khoảng 169 phút, trung bình thực tế khoảng 16,5 giây/câu.

Batch archive cuối:

```text
m43-public-qwen3b-v0431-t4-complete.zip
sha256:d68c165366169fd0c567938682078025c49db71face7baa96a2d8a31e5fa7af5
```

Submission đã kiểm tra trước khi nộp:

```text
sha256:636d50f076cb8e9fc336ce3bb1ea44ef4e1b336bc8aa1065bef40bbfba8bab03
```

Các file lớn trên không được commit; checksum là dấu vết đối chiếu.

## 5. Kết quả cấu trúc batch

| Chỉ số | Giá trị | Tỷ lệ |
|---|---:|---:|
| Records/unique IDs | 1.000/1.000 | 100% |
| Answer rỗng | 0 | 0% |
| Insufficient evidence | 425 | 42,5% |
| Non-abstaining verified answer | 575 | 57,5% |
| Generator backend `transformers` | 579 | 57,9% |
| Generator backend `None` | 421 | 42,1% |
| Retrieval model error | 0 | 0% |
| Generator model error | 33 | 3,3% |
| Citation verification failed | 384 | 38,4% |

`generator backend=None` không hoàn toàn đồng nghĩa model crash: nhiều record
đã abstain trước hoặc sau generation và response metadata không ghi backend.

## 6. Warning distribution

| Warning | Count | Ý nghĩa |
|---|---:|---|
| `effect_status_unknown:E1..E5` | 1.000 cho từng marker | Corpus không cung cấp trạng thái hiệu lực có cấu trúc |
| `context_budget_exhausted` | 1.000 | Mọi câu đều chạm budget selection |
| `citation_verification_failed` | 384 | Draft không qua claim/citation gate |
| `unsupported_claim:C1` | 340 | Claim đầu không đủ support theo verifier |
| `unsupported_claim:C2` | 214 | Claim thứ hai không đủ support |
| `unsupported_claim:C3` | 130 | Tương tự cho claim thứ ba |
| `unsupported_claim:C4` | 57 | Tương tự |
| `unsupported_claim:C5` | 40 | Tương tự |
| `generator:model_error` | 33 | Provider/model generation failure |
| `unsupported_claim:C6` | 28 | Tương tự |
| `unsupported_claim:C7` | 12 | Tương tự |

Hai warning đầu hiện gần như hằng số nên ít giá trị phân loại lỗi giữa các câu.
Chúng cần được tách thành telemetry chi tiết hơn: token requested/selected,
evidence dropped, lý do unknown effect status và tác động đến quyết định.

## 7. Phân tích độ dài answer

### 7.1 Prediction M43

- trung bình 147,8 ký tự;
- median 92 ký tự;
- p10/p90: 89/274 ký tự;
- trung bình 33,1 từ, median 21 từ;
- 636/1.000 answer dưới 30 từ;
- 425 answer là cùng một câu abstention;
- chỉ có 576 chuỗi answer duy nhất.

### 7.2 Reference trong official train

- 7.000 answers;
- trung bình 1.575,7 ký tự;
- median 1.410 ký tự;
- p10/p90: 670/2.612 ký tự;
- trung bình 347,4 từ, median 312 từ.

Median prediction ngắn hơn reference khoảng 15 lần theo ký tự. Vì METEOR nhấn
mạnh token recall và ROUGE-L đo chuỗi con chung dài nhất, output một câu dù đúng
ý thường không thể đạt recall cao so với reference nhiều đoạn.

Không được suy ra rằng “càng dài càng tốt”. Cần sinh đủ các ý pháp lý liên quan,
không lặp, không bịa và không padding. Độ dài là triệu chứng của thiếu coverage,
không phải target độc lập.

## 8. Chẩn đoán theo thành phần

### 8.1 Data adapter/cleaner — tương đối ổn định

Điểm mạnh:

- official-only và fail-closed;
- giữ blank/missing source fields thay vì bịa;
- canonical checksum, deterministic mapping;
- không để raw BTC fields lan vào core.

Rủi ro còn lại:

- corpus không có effect status/relationship labels;
- không có retrieval relevance labels;
- train/public overlap phải được kiểm soát khi tạo dev split.

Không nên tiếp tục chỉnh cleaner nếu chưa có lỗi corpus cụ thể. Đây không phải
nút thắt có bằng chứng cho official score hiện tại.

### 8.2 Parser/chunker — đúng integrity, chưa chứng minh retrieval quality

M40 bảo đảm 330.768 chunk, exact E5 max 512 và không truncation. Tuy nhiên:

- integrity không chứng minh granularity tối ưu cho câu hỏi;
- chunk metadata dài có thể chiếm budget search/input;
- điều/khoản liên quan có thể bị tách khiến answer cần nhiều ý nhưng retrieval
  chỉ thấy một phần;
- không có labeled retrieval set để chọn chunk policy bằng metric.

### 8.3 BM25 — bottleneck latency và lexical recall cần đo

Warm tail cho thấy BM25 khoảng 3,7–5 giây/câu, trong khi dense matrix score chỉ
vài chục ms. Cần tách thời gian query planning, SQLite I/O, multi-query và merge.
Không nên thay backend chỉ vì cảm giác; trước hết profile và lập recall proxy
trên split hợp lệ.

### 8.4 Dense retrieval — nhanh sau load, model chưa được task-adapt

E5-small cho exact vector search nhanh nhưng chưa fine-tune bằng dữ liệu Task 2.
Official train chỉ có answer-level supervision, không có gold evidence. Không
được tự tạo synthetic hard negatives/relevance labels. Mọi cách weak supervision
cần được đối chiếu điều lệ trước khi dùng cho training.

### 8.5 Fusion — cấu hình cố định chưa được tune

M43 dùng RRF, `candidate_k=60`, `top_k=8`, tối đa ba query variants. Chưa có
ablation cho:

- BM25-only, dense-only, hybrid;
- số query variants;
- RRF constant;
- branch candidate depth;
- duplicate/document diversity.

### 8.6 Reranker — implementation có nhưng M43 tắt

Approved cross-encoder chưa được dùng trong official batch. Đây là một khoảng
trống lớn: top fused candidates có thể cùng từ khóa nhưng không trả lời đúng
quan hệ hỏi. Cần benchmark reranker trên candidate set hữu hạn, không bật mặc
định mà thiếu ablation latency/quality.

### 8.7 Graph — không có dữ liệu để phát huy

Official graph zero-edge vì BTC không cung cấp relationship labels. Không nên
tự suy diễn relationship hoặc crawl ngoài. Graph generic vẫn hữu ích về kiến
trúc nhưng không phải workstream ưu tiên cho dataset hiện tại.

### 8.8 Context builder — mọi câu đều chạm budget

`context_budget_exhausted` 1.000/1.000 là tín hiệu mạnh nhất ở giữa pipeline.
Context builder có thể đang:

- lấy quá nhiều evidence dài;
- tiêu tốn token vào metadata/boilerplate;
- chọn các đoạn trùng nhau;
- cắt trước khi đủ các ý trả lời;
- ưu tiên rank thay vì marginal information gain.

Cần log token/evidence selection và benchmark diversity/coverage trước khi chỉ
tăng 3.072 lên một số lớn hơn.

### 8.9 Generator — output quá ngắn và 33 lỗi model

Prompt/config 256 output token và policy concise tạo answer rất ngắn so với gold.
Model chưa fine-tune Task 2. Model marker repair giúp format citation nhưng không
bảo đảm question relevance. Cần phân loại 33 model errors, prompt ablation và
official-only supervised fine-tuning plan.

### 8.10 Citation verifier — grounded không đồng nghĩa trả lời đúng câu hỏi

Verifier kiểm tra claim có support trong evidence, numeric/negation và citation
mapping. Nó chưa kiểm tra đầy đủ claim có trực tiếp trả lời intent của câu hỏi.

Một failure đã quan sát thuộc nhóm hỏi **thẩm quyền quyết định thành lập**, nhưng
answer lại mô tả **thành phần thành viên**. Nội dung có thể xuất hiện trong
evidence và qua một số grounding check, nhưng sai quan hệ cần trả lời. Báo cáo
chỉ lưu taxonomy, không commit nguyên câu hỏi/answer official.

384 verification failures cũng cho thấy policy hiện quá dễ rơi từ draft sang
generic abstention, làm mất toàn bộ token overlap thay vì giữ phần claim hợp lệ.

### 8.11 Agent workflow — deterministic nhưng ít đường phục hồi

M43 chỉ cho strategy hybrid, không rewrite và không retrieval retry. Khi context
yếu hoặc verification fail, workflow chủ yếu abstain. Safety invariant là đúng,
nhưng cần một fallback vẫn grounded, chẳng hạn:

- bỏ claim không được support, giữ claim hợp lệ;
- retry generation với feedback verifier;
- fallback extractive có giới hạn;
- query rewrite/retrieve lại tối đa theo `max_retry=2`.

Không được nới verifier để ép answer qua gate.

### 8.12 Evaluator — thiếu vòng lặp chất lượng trước submission

Official scorer đã được phân tích, nhưng local evaluator chưa có parity tuyệt
đối do NLTK/WordNet runtime chưa pin. Quan trọng hơn, chưa có dev protocol rõ để
tune mà không nhìn public leaderboard. Đây là nút thắt tổ chức lớn nhất.

## 9. Root-cause tree

```text
METEOR 0.0786
├─ 42,5% generic abstention
│  ├─ 38,4% citation verification failed
│  ├─ retrieval/context không đủ hoặc lệch intent
│  ├─ generator/model errors
│  └─ workflow thiếu grounded fallback
├─ 57,5% answer còn lại quá ngắn
│  ├─ prompt ưu tiên concise
│  ├─ output cap 256 token
│  ├─ context selection thiếu coverage
│  └─ model chưa học style/coverage của official answers
├─ retrieval chưa được task-evaluate
│  ├─ không relevance labels
│  ├─ reranker tắt
│  ├─ fusion/top-k chưa ablate
│  └─ BM25 latency cao
└─ evaluation loop chưa hoàn chỉnh
   ├─ chưa có leakage-safe dev protocol được chốt
   ├─ chưa có error taxonomy dashboard
   └─ official scorer runtime chưa fully pinned
```

## 10. Những điều không nên làm

- Không chỉnh tay answer public hoặc dùng leaderboard như tập train.
- Không dùng AIO/external corpus hay synthetic QA/evidence/hard negative.
- Không tăng output length mà không đo factuality/verification.
- Không tắt verifier để giảm abstention.
- Không bật tất cả reranker/graph/semantic verifier cùng lúc rồi không biết thay
  đổi nào tạo hiệu quả.
- Không dùng model/API chưa được BTC duyệt.
- Không resume checkpoint khác code/config identity.
- Không coi `Completed: 1000/1000` là quality gate; phải đếm error/warning.

## 11. Kết luận bàn giao

Baseline đã hoàn thành đúng vai trò: tạo một hệ thống end-to-end có provenance,
recovery và submission boundary. Việc tiếp theo không phải xây lại từ đầu, mà là
thiết lập evaluator/dev split rồi cải thiện từng điểm yếu bằng thí nghiệm có
đối chứng. Backlog thực thi nằm tại
[`17-TEAM-IMPROVEMENT-BACKLOG.md`](17-TEAM-IMPROVEMENT-BACKLOG.md).
