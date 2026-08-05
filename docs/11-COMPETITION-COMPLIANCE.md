# 11. Competition Compliance and Reproducibility

## 1. Status and source identity

Tài liệu này ghi lại các điều khoản BTC được người dùng cung cấp ngày
2026-08-01. Bản text được đối chiếu có SHA-256:

```text
c88b2eec6bccf2bc809e0b7982cbe113c56928671f99c7acb5e741fc310091be
```

Checksum chỉ định danh bản thể lệ đã đọc, không thay thế URL hay văn bản chính
thức của BTC. Khi BTC cập nhật thể lệ, phải lưu bản mới, checksum mới và rà
soát lại tài liệu này trước khi thay đổi pipeline.

Email phản hồi chính thức tiếp theo của BTC ngày 2026-08-01 xác nhận model
registration, official-only data, synthetic-data prohibition, Codabench và phạm
vi kỹ thuật được phép. Nội dung email phải được lưu cùng hồ sơ đội; tài liệu này
ghi lại quyết định nhưng không thay thế email gốc.

Thông báo chung tiếp theo của BTC xác nhận giới hạn tổng tham số dưới 4 tỷ, cấm
mọi API, cho phép pretrained/distilled model phù hợp, xác định dữ liệu pretraining
không phải external data trực tiếp, và cho phép nhiều hình thức đóng gói tái lập.
Thông báo gốc vẫn phải được lưu trong hồ sơ đội.

## 2. Ràng buộc dữ liệu bắt buộc

- Chỉ dùng dữ liệu chính thức do BTC phát hành.
- Không gán nhãn thủ công.
- Không thu thập corpus hoặc QA data bên ngoài.
- Không dùng data augmentation từ nguồn bên ngoài.
- Không tạo synthetic data, kể cả khi sinh hoàn toàn từ dữ liệu BTC. Lệnh cấm
  bao gồm synthetic QA, answer, evidence, hard negative và training example.
- Không dùng artifact/index có lineage AIO hoặc corpus ngoài BTC.
- Warm-up answer chỉ là supervision/evaluation chính thức; không được dùng làm
  prediction cho chính case đang được đánh giá hoặc giả lập relevance label.
- Dữ liệu BTC, artifact lớn và submission thật không được commit vào Git.

Mọi input phải qua adapter UIT DSC 2026 và mọi artifact phải ghi dataset name,
revision/checksum, config hash và code version.

## 3. Model registration gate

BTC không ban hành danh sách model cố định. Mọi model mã nguồn mở đội muốn sử
dụng phải đăng ký tên model và URL chính thức qua Google Form BTC sẽ gửi. Cho
tới khi Form xuất hiện và registration evidence được lưu, repository áp dụng
fail-closed:

1. model candidate không đồng nghĩa model được phép thi;
2. không chạy official competition batch bằng model có trạng thái approval
   `pending` hoặc `unknown`;
3. registration record phải ghi model ID, URL, immutable revision, license, mục
   đích, parameter count có bằng chứng, dữ liệu huấn luyện công bố bởi model
   author và ngày gửi Form;
4. chỉ đổi trạng thái thành `registered`/`approved` khi có bằng chứng chính thức
   phù hợp hướng dẫn Form;
5. đổi embedding model yêu cầu rebuild toàn bộ vector index.

### Candidate register

| Thành phần | Candidate và revision | License | BTC registration | Trạng thái |
|---|---|---|---|---|
| Embedding | `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3` | MIT theo model card tại revision | Chờ Form | Không được dùng cho official run |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@1427fd652930e4ba29e8149678df786c240d8825` | Apache-2.0 theo model card tại revision | Chờ Form | Không được dùng cho official run |
| Generator smoke candidate | `Qwen/Qwen2.5-3B-Instruct@a1d308dfcc03e09da285d49d912439a655a571e8` | `qwen-research` tùy chỉnh | Chờ Form | Không được dùng cho official run; cần legal/BTC review |

Nguồn model card theo exact revision:

- <https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3>;
- <https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825>;
- <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/a1d308dfcc03e09da285d49d912439a655a571e8>.

Metadata license là bước kiểm kê đầu tiên, chưa thay thế việc đọc toàn văn
license và kiểm tra dataset/base-model transitive obligations. Đặc biệt,
`qwen-research` không được mặc định tương đương OSI open source hoặc tương
thích yêu cầu BTC.

Không dùng bất kỳ model API hoặc sản phẩm AI trung gian nào trong quá trình xây
dựng phương pháp hoặc competition run, kể cả API miễn phí hay phi lợi nhuận.
Mọi model phải được đội tải, chạy và kiểm soát trực tiếp. Mọi model hoặc package
mới phải được thêm vào register trước khi thử.

BTC đã cho phép fine-tuning, preprocessing, indexing và retrieval nếu chỉ dùng
dữ liệu chính thức. Synthetic augmentation vẫn bị cấm và mọi model fine-tuned
vẫn phải đi qua registration/license/reproducibility gate.

## 3.1. Parameter-budget gate

- Tổng tham số của tất cả model active trong Task 2 phải nhỏ hơn
  `4_000_000_000`.
- Phải cộng embedding, reranker, generator, semantic grader/verifier và mọi
  model phụ trợ; BM25, RRF và code rule-based không có learned parameters.
- Distilled model được phép nếu model cuối thực sự nằm dưới giới hạn.
- Quantization, LoRA hoặc giảm số bit lưu trữ không làm giảm parameter count để
  xét điều kiện.
- Thiếu parameter count/bằng chứng cho bất kỳ model active nào hoặc tổng
  `>= 4_000_000_000` phải chặn official run.
- Candidate register hiện tại chưa phải bằng chứng rằng tổng cấu hình đang dùng
  hợp lệ; cần hoàn tất inventory trước khi chọn final stack.

## 4. Submission governance

- Private test: tối đa 03 submission trong một ngày.
- Chỉ submit archive đã qua formatter và preflight local.
- Không upload trực tiếp output nội bộ `results.jsonl`.
- `submission.json` phải là object keyed by question ID với mỗi value đúng
  `{"answer": string}`, theo contract thực thi đã xác minh trên Codabench.
- Ghi SHA-256 của question source, batch manifest, config, code commit,
  `submission.json` và final `submission.zip`.
- Mỗi lần upload phải được ghi vào ledger riêng; không suy ra số lượt còn lại
  từ file local.
- Local METEOR/ROUGE-L chỉ là diagnostic cho tới khi scorer parity được chứng
  minh.
- Source scorer BTC checksum
  `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`
  xác nhận NLTK METEOR, vendored `rouge_score` và macro mean; PyVi hiện bị
  comment và không chạy.
- Scorer tải `wordnet`/`omw-1.4` lúc thực thi nhưng không pin NLTK hoặc resource
  bytes. BTC đã sửa lỗi WordNet của warm-up image; không thay đổi submission để
  né lỗi hạ tầng scorer.
- Phân tích đầy đủ nằm tại `docs/15-OFFICIAL-SCORING-CONTRACT.md`.

Template thực thi nằm tại:

- `docs/templates/PRIVATE-SUBMISSION-CHECKLIST.md`;
- `docs/templates/SUBMISSION-LEDGER.csv`.

## 5. Data Statement and Model Card

Mỗi bài nộp phải có Data Statement và Model Card theo thể lệ. Hai tài liệu này
không nằm trong `submission.zip` trừ khi BTC thay đổi exact output contract.
Template:

- `docs/templates/DATA-STATEMENT.md`;
- `docs/templates/MODEL-CARD.md`.

Không được điền nội dung suy đoán. Mỗi release candidate phải tạo bản snapshot
riêng, ghi code commit, config hash, dataset checksum, model revision và kết
quả kiểm thử.

## 6. Reproducibility and Docker

Top 7 phải cung cấp Docker image và source MIT để BTC tái lập Private Test.
Repository áp dụng:

- source license `LICENSE` là MIT;
- Docker build context loại data, model, artifact, log, secret và submission;
- process trong image chạy bằng non-root user;
- Python base image và direct dependency constraints được pin;
- model weights và official artifacts phải được mount/copy theo quy trình BTC
  cho phép, không được commit hoặc silently download khi reproduction yêu cầu
  offline;
- trước private release phải lưu image digest, `pip freeze`, OS/CUDA/driver,
  GPU, seed, config và exact reproduction command;
- CPU Dockerfile hiện tại là compliance scaffold. GPU base image cuối cùng chỉ
  được chốt sau khi biết hạ tầng BTC và model được phê duyệt.

## 7. Licensing and disclosure

- Source do đội sở hữu được phát hành theo MIT.
- License MIT của repository không thay đổi license của dependency hoặc model.
- Model weights, tokenizer, dataset và package bên thứ ba phải được trích dẫn
  và tuân thủ license gốc.
- Không đưa private URL, token, credential hoặc dữ liệu cá nhân vào source,
  image, log, Model Card hay Data Statement.

## 8. Open organizer clarifications

1. Google Form đăng ký model được phát hành khi nào và registration có cần BTC
   phản hồi chấp thuận riêng hay chỉ cần khai báo?
2. Google Form yêu cầu khai báo parameter count theo nguồn nào và có cần BTC
   phản hồi chấp thuận riêng không?
3. Môi trường inference/chấm cuối có Internet không và model weights được
   cung cấp/mount thế nào?
5. Docker command, CUDA, RAM, disk, timeout và interface chấm cuối cùng là gì?
6. Data Statement và Model Card được nộp ở đâu, thời điểm nào và theo format
   nào?
7. Image public/private có dùng đúng scorer checksum đã phân tích và exact
   NLTK/WordNet resource versions nào?

Mọi điểm trên giữ trạng thái unresolved; code không được biến chúng thành giả
định ngầm.
