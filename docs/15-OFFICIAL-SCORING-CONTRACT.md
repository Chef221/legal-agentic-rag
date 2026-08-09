# 15. UIT DSC 2026 Task 2 Official Scoring Contract

## 1. Purpose and authority

Tài liệu này ghi lại hành vi của mã nguồn scorer do BTC cung cấp trong file:

```text
Scoring-Program-Task-LegalQA.zip
```

Artifact được phân tích read-only ngày 2026-08-05, không chạy code trong ZIP và
không commit artifact vào repository.

| Thuộc tính | Giá trị |
|---|---|
| Kích thước ZIP | 41.775 bytes |
| SHA-256 | `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891` |
| Số archive entries | 34 |
| Entry path traversal/absolute path | Không phát hiện |

Khi BTC phát hành scorer mới, checksum phải được kiểm tra lại. Không được mặc
định tài liệu này còn đúng với artifact có checksum khác.

## 2. Archive inventory

Hai file điều khiển ở root:

| File | Vai trò | SHA-256 |
|---|---|---|
| `metadata.yaml` | Chạy `python3 scoring.py` | `7aa33a3c190a9cd687e40a0a1aa99aeefebf9c6c86bf7f083c05f6ea7254ad3a` |
| `scoring.py` | Đọc prediction/reference, tính điểm và ghi output | `f04843fbfad26d41356506d8e49692a7c8a0ed1b9f065a3a8472fa6398a5aa95` |

ZIP còn chứa:

- 11 entries dưới `rouge_score/` cho source/runtime package vendored;
- 5 test modules của `rouge_score`;
- 6 entries `__pycache__`;
- 10 entries metadata `__MACOSX`;
- `.DS_Store` và directory markers.

Các entry `__MACOSX`, bytecode cache và `.DS_Store` không tham gia thuật toán.
Scorer import trực tiếp package `rouge_score` được đóng cùng ZIP. Archive không
có `requirements.txt`, dependency lock hoặc version metadata cho NLTK/NumPy.

Các file vendored trực tiếp quyết định ROUGE-L có checksum:

| File | SHA-256 |
|---|---|
| `rouge_score/rouge_scorer.py` | `9484c5fd05e22cd28b5053bf9de586b3620cf65b0c38052b8690badd51f31d1a` |
| `rouge_score/tokenize.py` | `dc91cea8f09507f744549160c458031d0956e35fa230c80d01f990eba20a7403` |
| `rouge_score/tokenizers.py` | `2b7b9dae505ce8739064ba8e4791b871b13aa1f65bc54e25692f1a9ba8600508` |
| `rouge_score/scoring.py` | `b3fc153499e484665294ddbaeb876452b40a8fa2120e824b804928c96e3f2c1a` |

## 3. Runtime file contract

Scorer dùng các path cố định trong Codabench container:

```text
/app/input/ref/metadata.json
/app/input/ref/<reference filename>
/app/input/res/<input filename>
/app/output/scores.json
```

Tên reference và prediction được lấy từ:

```text
metadata["files"]["reference"]
metadata["files"]["input"]
```

Prediction phải là JSON object. Với mỗi question ID, scorer lấy:

```text
prediction[question_id]["answer"]
```

Reference nội bộ của scoring bundle được dùng như mapping trực tiếp:

```text
reference[question_id] -> reference answer value
```

Đây là contract nội bộ của scoring program, không thay đổi raw contract
`warmup.json` public là `question_id -> {question, answer}`. Submission formatter
vẫn phải tạo root object `question_id -> {"answer": string}` theo D077.

Scorer chỉ so sánh số lượng prediction ID với số lượng reference ID. Sau đó nó
iterate prediction IDs và truy cập reference bằng cùng key. Vì vậy:

- thiếu/thừa ID làm fail do count mismatch hoặc missing key;
- thứ tự JSON object không ảnh hưởng aggregate;
- scorer không tự kiểm tra duplicate JSON keys;
- scorer không enforce `answer` là string trước khi gọi `str(...)`.

Repository giữ validator nghiêm hơn: exact ID set/order theo source, duplicate
key rejection và string-only answer. Các gate này tương thích scorer và không
được nới lỏng theo những thiếu sót validation của scorer.

## 4. Official METEOR algorithm

Mỗi prediction/reference được chuyển sang `str`, sau đó dùng Python
`str.split()` không truyền separator. Hệ quả:

- token boundary là whitespace;
- Unicode tiếng Việt và dấu câu gắn token được giữ khi split;
- PyVi **không được dùng**; import và lời gọi PyVi trong source đã bị comment.

Scorer gọi trực tiếp:

```text
nltk.translate.meteor_score.meteor_score([reference_tokens], prediction_tokens)
```

Không truyền preprocessing, `alpha`, `beta`, `gamma`, stemmer hoặc WordNet
override, nên hành vi phụ thuộc defaults của NLTK đang cài trong image. Log
warm-up trước đó cho thấy image dùng NLTK 3.7. Source NLTK 3.7 định nghĩa:

- `preprocess=str.lower`, vì vậy METEOR không phân biệt hoa/thường sau bước
  whitespace split;
- `PorterStemmer()` mặc định;
- `nltk.corpus.wordnet` mặc định;
- `alpha=0.9`, `beta=3.0`, `gamma=0.5`;
- thứ tự matching: exact, stem, WordNet synonym, rồi fragmentation penalty.

Nguồn đối chiếu:
<https://github.com/nltk/nltk/blob/3.7/nltk/translate/meteor_score.py>.

Khi import, scorer gọi:

```text
nltk.download("wordnet")
nltk.download("omw-1.4")
```

Do ZIP không pin NLTK/version resource và yêu cầu tải resource lúc chạy, exact
METEOR reproduction vẫn còn phụ thuộc image + NLTK data cụ thể. Đây là phần
chưa được artifact tự thân khóa hoàn toàn, dù công thức gọi và aggregation đã
được xác nhận.

Dataset score là arithmetic macro mean bằng `numpy.mean` trên điểm từng
prediction ID.

## 5. Official ROUGE-L algorithm

Scorer tạo:

```text
RougeScorer(["rougeL"], use_stemmer=False)
```

`build_in_tokenizer` trả nguyên chuỗi; không dùng PyVi. Tuy nhiên vendored
`RougeScorer` tiếp tục chạy `DefaultTokenizer` của chính package:

1. lowercase;
2. thay mọi ký tự không khớp `[a-z0-9]` bằng khoảng trắng;
3. split whitespace;
4. bỏ token rỗng/không hợp lệ;
5. không Porter stemming vì `use_stemmer=False`.

Điểm đặc biệt cần ghi nhớ: regex này chỉ giữ ASCII `a-z` và chữ số. Các ký tự
tiếng Việt có dấu bị loại hoặc trở thành ranh giới token trong nhánh ROUGE-L.
Không được nhầm behavior này với Unicode-aware tokenizer local hiện tại.

ROUGE-L tính độ dài longest common subsequence, sau đó:

```text
precision = LCS / prediction_token_count
recall    = LCS / reference_token_count
F1        = 2 * precision * recall / (precision + recall)
```

Dataset score là arithmetic macro mean bằng `numpy.mean` trên F1 từng
prediction ID.

## 6. Scorer output

Output duy nhất của scoring script là:

```json
{
  "rouge": 0.0,
  "meteor": 0.0
}
```

File được ghi tại `/app/output/scores.json`. Key thực thi là `rouge` và
`meteor`; METEOR vẫn là metric xếp hạng chính theo quy định BTC, ROUGE-L là
metric phụ.

## 7. Difference from the current local evaluator

Local M29/M30 evaluator hiện dùng:

- NFC + casefold;
- Unicode word/number tokens;
- exact-token-only METEOR, không stemming/synonym resource;
- Unicode token-level ROUGE-L F1.

Vì vậy local score **không tương đương** scorer BTC:

| Chi tiết | Scorer BTC | Local M29/M30 |
|---|---|---|
| METEOR tokenization | whitespace `split()` | Unicode word/number regex |
| METEOR case/punctuation | lowercase nội bộ; punctuation còn gắn token | casefold, bỏ punctuation |
| METEOR stem/synonym | NLTK default + WordNet | không |
| ROUGE-L tokenizer | lowercase ASCII `[a-z0-9]` | Unicode-aware |
| Aggregation | arithmetic macro mean | arithmetic macro mean |

Các report local phải tiếp tục mang warning
`competition_text_metrics_are_diagnostic_not_official_equivalent` cho tới khi
official-compatible implementation được viết và verified bằng test vectors.

## 8. Engineering consequences

1. Không còn coi tokenizer/aggregation của scorer là unknown.
2. Không thêm PyVi vào scorer parity plan; source chính thức hiện không dùng nó.
3. Không xóa dấu hoặc làm hỏng tiếng Việt để chạy theo ROUGE tokenizer. METEOR
   là metric chính và answer vẫn phải đúng, grounded, UTF-8.
4. Một implementation official-compatible phải pin NLTK/runtime resource,
   reproduce đúng vendored ROUGE files hoặc hành vi có test parity, và giữ
   scorer mode tách khỏi diagnostic mode hiện tại.
5. Trước khi thay local evaluator, cần test golden vectors gồm dấu tiếng Việt,
   hoa/thường, punctuation, whitespace, reordered fragments, empty text và ID
   mismatch.
6. Không copy toàn bộ source scorer vào core nếu chưa cần; giữ checksum và
   provenance, đồng thời review license/dependency trước khi thêm package.

## 9. Remaining uncertainties

Scoring ZIP đã giải quyết công thức, tokenizer ROUGE, lời gọi METEOR và macro
aggregation. Các điểm còn thiếu để tái lập tuyệt đối:

- exact NLTK và NumPy versions của image chính thức;
- exact WordNet/OMW resource versions/bytes;
- việc image chấm public/private có giữ nguyên scorer checksum này hay không;
- Codabench leaderboard configuration ngoài `scores.json`.

Mọi scorer artifact mới phải được checksum và diff trước khi dùng.

## 10. M44 Local Official-compatible Mode

M44 triển khai mode `official_compatible` tách khỏi diagnostic evaluator:

- METEOR gọi NLTK 3.7 `meteor_score([reference.split()], prediction.split())`;
- ROUGE-L dùng lowercase + regex ASCII `[^a-z0-9]` + LCS F1;
- report pin scorer SHA-256, code version, NLTK/NumPy versions và input hashes;
- thiếu NLTK 3.7 hoặc WordNet local là lỗi khởi tạo backend;
- không tự download resource và không silently fallback.

Implementation tương thích thuật toán/source đã audit. Absolute parity vẫn phụ
thuộc exact WordNet/OMW bytes và việc phase scorer sau giữ nguyên checksum. Mô
tả “local evaluator chưa có official mode” ở phần lịch sử phía trên chỉ áp dụng
M29/M30 và được mục này thay thế từ M44.
