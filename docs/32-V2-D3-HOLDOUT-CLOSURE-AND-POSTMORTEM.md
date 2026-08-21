# 32 — V2-D3 Fresh-Holdout Evaluation Closure & Postmortem Forensic Failure Analysis

## 1. Executive Summary & Final Governance Decision

This document records the formal closure of the **V2-D3 Structured Semantic Citation Verifier** fresh-holdout evaluation (Phase H-EXEC) on Kaggle GPU, details the exact scientific and operational metrics obtained, and provides a forensic failure analysis across all holdout errors to establish requirements for future verifier architectures.

### Final V2 Decision & Invariants
- **Final Verdict:** `V2_D3_HOLDOUT_EXECUTION_FAILURE`
- **Evaluation Decision:** `REJECT_V2_D3_PROMOTION`
- **Promotion Recommended:** `False`
- **Promotion Authorized:** `False` (Unconditional Invariant)
- **V2-D3 Development Track:** **PERMANENTLY CLOSED**
- **Production Semantic Verifier:** **REMAINS DISABLED**
- **Fresh Holdout Status:** **BURNED & PERMANENTLY CONSUMED**
- **Governance Invariants:**
  - **NO D3.3** will be created.
  - **NO holdout reruns** on these 31 claims.
  - **NO post-hoc threshold adjustments**.
  - **NO human label modifications**.
  - The 31 fresh-holdout claims are now diagnostic development data only; any future candidate promotion will require a brand-new, untouched holdout.

---

## 2. Canonical Holdout Evidence Identity

The authoritative benchmark execution was conducted on Kaggle GPU and packaged into canonical evidence archive `verification-v2-d3-holdout-evidence.zip`:

| Evidence Parameter | Canonical Value |
| :--- | :--- |
| **Execution Authority Commit** | `77561aa7c4b242e12d011a84a21f3a262a17a0f8` |
| **Evidence Archive Filename** | `verification-v2-d3-holdout-evidence.zip` |
| **Archive SHA-256 Checksum** | `9e2b38d4189f9c68901051a07b999845c660ec6ab4b4fa1e6ec69d3088fe6a5d` |
| **Archive Size** | `10,463` bytes |
| **Total Archive Members** | `9` files |
| **Execution Timestamp** | `2026-08-21T04:18:35.241981+00:00` |
| **Hardware Environment** | Kaggle GPU (Tesla T4, CUDA 12.x, PyTorch float16) |
| **Candidate Model** | `Qwen/Qwen2.5-3B-Instruct` (rev `a1d308dfcc03e09da285d49d912439a655a571e8`) |
| **Runtime Backend** | `transformers==4.47.1`, `tokenizers==0.21.4`, `accelerate==1.2.1` |
| **D3 Implementation SHA** | `a6e8bca15ad14d869e103e1f94fe94bb9a81f9ddc8bc650b280b69b7d57e9826` |
| **D3 System Prompt SHA** | `546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b` |
| **D3 Schema SHA** | `3591144a40b0519d5da9dd262e8edf8814531d798b69deea94fd81fae39f5f61` |
| **Frozen Human Labels SHA** | `85d348dbb7da1567398836b96156a9d08fcfe181b676c5ecd593535ec8904215` (9,383 bytes) |
| **Approved Commitment SHA** | `5cc7f58ed52c43d091bce98d5296ab68cded981495736a479644aefc1428b6dc` (1,060 bytes) |

---

## 3. Authoritative Benchmark Results & Promotion Gate Evaluation

### 3.1 Pre-Registered Promotion Gate Evaluation (Authoritative Pass 1)

| Promotion Gate Metric | Pre-Registered Minimum | Observed Pass 1 Value | Gate Status |
| :--- | :--- | :--- | :--- |
| **Coverage Denominators** | Non-vacuous ($> 0$) | $N_{\text{supp}}=23, N_{\text{neg}}=7, N_{\text{val}}=10, N_{\text{inval}}=6$ | **PASS** |
| **Supported Retention Rate** | $\ge 0.88$ ($88.0\%$) | $22 / 23 = 95.65\%$ | **PASS** |
| **Negative Catch Rate** | $\ge 0.50$ ($50.0\%$) | $2 / 7 = 28.57\%$ | **FAILED** |
| **Valid Answer Retention Rate** | $\ge 0.80$ ($80.0\%$) | $8 / 10 = 80.00\%$ | **PASS** |
| **Full-Denominator Answer Accuracy** | $\ge 0.60$ ($60.0\%$) | $10 / 16 = 62.50\%$ | **PASS** |
| **Claim Binary Accuracy** | $\ge 0.70$ ($70.0\%$) | $24 / 30 = 80.00\%$ | **PASS** |
| **Mechanical Stability / Zero Errors** | Zero Execution Errors | $1$ Execution Error (`103383:PRIMARY:C1`) | **FAILED** |

### 3.2 Key Scientific Conclusion
The primary scientific blocker is that **Negative Catch Rate Failed** ($28.57\%$ vs $50.00\%$ minimum threshold).
Even if the single operational execution error did not occur, V2-D3 still would **NOT** qualify for promotion because it allowed $5$ out of $7$ invalid/unsupported claims to escape as `SUPPORTED`.

---

## 4. Exact Scientific Metrics Breakdown

### 4.1 Claim-Level Binary Metrics (Pass 1)
- **Total Claims:** 31
- **Evaluated Claims:** 30 (1 Execution Error)
- **True Positives (TP):** 22 (Gold Supported $\rightarrow$ Predicted Accept)
- **False Positives (FP):** 5 (Gold Negative $\rightarrow$ Predicted Accept)
- **True Negatives (TN):** 2 (Gold Negative $\rightarrow$ Predicted Reject)
- **False Negatives (FN):** 1 (Gold Supported $\rightarrow$ Predicted Reject)
- **Claim Binary Accuracy:** $24 / 30 = 80.00\%$
- **Precision:** $22 / (22 + 5) = 81.48\%$
- **Supported Retention Rate:** $22 / 23 = 95.65\%$
- **Negative Catch Rate:** $2 / 7 = 28.57\%$
- **F1 Score:** $0.8800$
- **Balanced Accuracy:** $(0.9565 + 0.2857) / 2 = 62.11\%$

### 4.2 Three-Way Entailment Confusion Matrix (Pass 1)

```text
                  Predicted SUPPORTED   Predicted CONTRADICTED   Predicted INSUFFICIENT
Gold SUPPORTED:          22                      0                         1
Gold CONTRADICTED:        0                      0                         1
Gold INSUFFICIENT:        5                      0                         1
```

- **Three-Way Accuracy:** $23 / 30 = 76.67\%$
- **Note on Contradiction Detection:** D3 detected zero explicit contradictions in 3-way classification ($0/1$). The single contradicted claim (`3339:PRIMARY:C1`) was classified as `INSUFFICIENT` (`DOES_NOT_ESTABLISH`), which is binary-correct (rejected) but fails granular three-way classification.

### 4.3 Answer-Level Metrics (16 Answers)
- **Gold Valid Answers:** 10
- **Gold Invalid Answers:** 6
- **Valid Answers Retained:** $8 / 10 = 80.00\%$
- **Invalid Answers Caught:** $2 / 6 = 33.33\%$
- **Full-Denominator Answer Accuracy:** $10 / 16 = 62.50\%$

---

## 5. Operational Telemetry & Stability Analysis

### 5.1 Provider Telemetry
- **Total Provider Calls:** 62 (Pass 1: 31 calls; Pass 2: 31 calls)
- **Successful Provider Calls:** 61
- **Failed Provider Calls:** 1 (`call_index: 1`)
- **Structured JSON Retries Used:** 0
- **Call Reconciliation Gate:** Reconciled ($62 == 2 \times 31 + 0$)

### 5.2 Deterministic Stability (Pass 1 vs Pass 2)
- **Claims Evaluated Twice:** 30
- **Stable Semantic Claims:** 30 ($100.0\%$ stability across all successfully executed claims)
- **Unstable Semantic Claims:** 0
- **Pass 1 Execution Errors:** 1 (`103383:PRIMARY:C1`)
- **Pass 2 Execution Errors:** 0

### 5.3 Operational Error Analysis (`103383:PRIMARY:C1`)
- **Error Observed:** `BackendInitializationError` on Call 1 (duration 54.55s).
- **Subsequent Calls:** Call 2 through 62 succeeded in ~1.5s per claim. In Pass 2, `103383:PRIMARY:C1` executed cleanly and predicted `SUPPORTED`.
- **Root Cause Hypothesis:** Cold-start model weight loading and CUDA initialization in `TransformersChatProvider._require_runtime()` during the first inference call encountered a transient initialization delay.
- **Operational vs Semantic Decoupling:** Remediating the cold-start initialization eliminates the runtime error but leaves Negative Catch at $28.57\%$. Operational fixes cannot substitute for semantic verifier redesign.

---

## 6. Post-Holdout Forensic Failure Analysis

Now that the holdout is burned, all 9 non-trivial claims (5 False Accepts, 1 False Reject, 2 True Negatives, 1 Execution Error) are unsealed for architectural forensics.

### 6.1 Forensic Failure Taxonomy

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC FAILURE TAXONOMY                              │
├─────────────────────────────┬───────────────────────────────────────────────┤
│ Failure Category            │ Description                                   │
├─────────────────────────────┼───────────────────────────────────────────────┤
│ ACTOR_ROLE_MISMATCH         │ Right/obligation assigned to wrong legal party│
│ ACTION_OBJECT_MISMATCH      │ Violation/action conflated with another topic │
│ CONDITION_EXCEPTION_OMITTED │ Statutory prerequisites/exemptions ignored    │
│ SYNTAX_FRAGMENT_STRICTNESS  │ Valid noun phrase fragment rejected           │
│ QUANTITY_TEMPORAL_MISMATCH  │ Penalty/deadline numbers mapped to wrong rule │
│ LEXICAL_SIMILARITY_BIAS     │ Verbatim token overlap masks semantic defects │
└─────────────────────────────┴───────────────────────────────────────────────┘
```

---

### 6.2 The Five False Accepts (Primary Failure Target)

#### 1. Target `125893:PRIMARY:C1`
- **Question:** Đương sự có những quyền và nghĩa vụ nào khi tham gia tố tụng hành chính?
- **Claim:** "Đương sự có quyền tham gia tố tụng từ khi khởi kiện hoặc bất cứ giai đoạn nào trong quá trình tố tụng hành chính."
- **Human Gold:** `INSUFFICIENT`
- **D3 Prediction:** `SUPPORTED` (relation: `ENTAILS`, flags: all `False`)
- **Cited Evidence:** Điều 61/64 Luật TTHC 2015 (Evidence E1, chunk `5514594eccb6f582654cd49c`):
  *"6. Người bảo vệ quyền và lợi ích hợp pháp của đương sự có các quyền, nghĩa vụ sau đây: a) Tham gia tố tụng từ khi khởi kiện hoặc bất cứ giai đoạn nào trong quá trình tố tụng hành chính..."*
- **Primary Failure Mechanism:** `ACTOR_ROLE_MISMATCH`
- **Forensic Diagnosis:** The statute grants the right to participate from filing to the **representative/advocate** (*Người bảo vệ quyền và lợi ích hợp pháp của đương sự*), NOT to the **litigant** (*Đương sự*), whose rights are governed by Điều 55. D3 matched the phrase *"quyền và lợi ích hợp pháp của đương sự"* and treated "đương sự" as the grammatical subject, failing to recognize the legal actor substitution.

#### 2. Target `125893:PRIMARY:C3`
- **Question:** Đương sự có những quyền và nghĩa vụ nào khi tham gia tố tụng hành chính?
- **Claim:** "Đương sự có quyền tham gia phiên tòa, phiên họp hoặc trong trường hợp không tham gia thì được gửi văn bản bảo vệ quyền và lợi ích hợp pháp của đương sự cho Tòa án xem xét."
- **Human Gold:** `INSUFFICIENT`
- **D3 Prediction:** `SUPPORTED` (relation: `ENTAILS`, flags: all `False`)
- **Cited Evidence:** Điều 61/64 Luật TTHC 2015 (Evidence E1):
  *"c) Tham gia phiên tòa, phiên họp hoặc trong trường hợp không tham gia thì được gửi văn bản bảo vệ quyền và lợi ích hợp pháp của đương sự cho Tòa án xem xét;"*
- **Primary Failure Mechanism:** `ACTOR_ROLE_MISMATCH`
- **Forensic Diagnosis:** Identical to C1. The right to submit a written defense if absent belongs to the **advocate**, not the litigant. D3 observed a near-$100\%$ lexical overlap with point c and overlooked the missing legal subject.

#### 3. Target `45427:PRIMARY:C1`
- **Question:** Nhà ở xã hội do doanh nghiệp đầu tư, xây dựng được không?
- **Claim:** "Nhà ở xã hội do doanh nghiệp đầu tư được hưởng các ưu đãi như miễn tiền sử dụng đất và thuế."
- **Human Gold:** `INSUFFICIENT`
- **D3 Prediction:** `SUPPORTED` (relation: `ENTAILS`, flags: all `False`)
- **Cited Evidence:** Điều 58 Luật Nhà ở 2014 (Evidence E1, chunk `a6b484dd8dcc80e2dd0efbb4`):
  *"1. Doanh nghiệp, hợp tác xã tham gia đầu tư xây dựng nhà ở xã hội không phải bằng nguồn vốn hoặc hình thức quy định tại khoản 1 Điều 53 của Luật này để cho thuê, cho thuê mua, bán thì được hưởng các ưu đãi sau đây: a) Được miễn tiền sử dụng đất... b) Được miễn, giảm thuế GTGT..."*
- **Primary Failure Mechanism:** `CONDITION_EXCEPTION_OMITTED`
- **Forensic Diagnosis:** Statutory incentives are conditional on investing **without state capital** (*không phải bằng nguồn vốn hoặc hình thức quy định tại khoản 1 Điều 53*) and for specific purposes (*để cho thuê, cho thuê mua, bán*). Furthermore, incentives are granted to the corporate investor, not the building. D3 overlooked the statutory conditions and blanket-endorsed an unqualified claim.

#### 4. Target `90897:PRIMARY:C1`
- **Question:** Ai có quyền quyết định giải ngạch đối với sĩ quan dự bị hết tuổi phục vụ?
- **Claim:** "Chỉ huy trưởng Ban Chỉ huy quân sự cấp huyện có quyền quyết định giải ngạch đối với sĩ quan dự bị hết tuổi phục vụ."
- **Human Gold:** `INSUFFICIENT`
- **D3 Prediction:** `SUPPORTED` (relation: `ENTAILS`, flags: all `False`)
- **Cited Evidence:** Điều 29 Luật Nghĩa vụ quân sự 2015 (Evidence E1, chunk `b6f965dec5ca4fe168255bf8`):
  *"Hạ sĩ quan, binh sĩ dự bị hết độ tuổi hoặc không còn đủ sức khỏe phục vụ trong ngạch dự bị thì được giải ngạch theo quyết định của Chỉ huy trưởng Ban Chỉ huy quân sự cấp huyện."*
- **Primary Failure Mechanism:** `ACTOR_ROLE_MISMATCH`
- **Forensic Diagnosis:** E1 applies strictly to **NCOs and enlisted reserve soldiers** (*Hạ sĩ quan, binh sĩ dự bị*). For **reserve officers** (*Sĩ quan dự bị*), discharge authority belongs to the appointing authority (Minister of Defense, Prime Minister, etc. under Điều 25/44 Luật Sĩ quan QĐNDVN). D3 conflated "Hạ sĩ quan" with "Sĩ quan", treating two legally distinct ranks as equivalent.

#### 5. Target `95695:PRIMARY:C1`
- **Question:** Sản xuất thức ăn chăn nuôi không có người phụ trách kỹ thuật có trình độ đại học trở lên bị xử phạt như thế nào?
- **Claim:** "Sản xuất thức ăn chăn nuôi không có người phụ trách kỹ thuật có trình độ đại học trở lên bị phạt tiền từ 15.000.000 đồng đến 20.000.000 đồng."
- **Human Gold:** `INSUFFICIENT`
- **D3 Prediction:** `SUPPORTED` (relation: `ENTAILS`, flags: all `False`)
- **Cited Evidence:** Điều 21 Nghị định 14/2021/NĐ-CP (Evidence E1, chunk `15fb151787bcfbf4246814a1`):
  *"1.b) Phạt tiền từ 10.000.000 đồng đến 15.000.000 đồng đối với hành vi cơ sở khảo nghiệm không có người phụ trách kỹ thuật có trình độ từ đại học trở lên...; 1.c) Phạt tiền từ 15.000.000 đồng đến 20.000.000 đồng đối với hành vi cơ sở khảo nghiệm không có cơ sở vật chất, trang thiết bị kỹ thuật..."*
- **Primary Failure Mechanism:** `ACTION_OBJECT_MISMATCH` & `QUANTITY_TEMPORAL_MISMATCH`
- **Forensic Diagnosis:** E1 regulates *khảo nghiệm* (testing), not *sản xuất* (production). Furthermore, the penalty for lacking technical personnel is 10M–15M VND (point b), while 15M–20M VND (point c) is for lacking facilities. D3 merged keywords across points b and c, misattributing the fine range and confusing production with testing.

---

### 6.3 The False Reject (`61523:PRIMARY:C1`)
- **Question:** Có các loại trang thiết bị dùng trong thi đấu và tập huấn thể thao nào?
- **Claim:** "Trang thiết bị tập thể lực chung cho vận động viên đội tuyển quốc gia các môn thể thao."
- **Human Gold:** `SUPPORTED`
- **D3 Prediction:** `INSUFFICIENT` (relation: `DOES_NOT_ESTABLISH`)
- **Cited Evidence:** Điều 5 Thông tư 05/2021/TT-BVHTTDL (Evidence E1, chunk `93022c2ce1e97866f29657cf`):
  *"Điều 5. Phân loại trang thiết bị tập huấn, thi đấu thể thao: Trang thiết bị tập huấn, thi đấu thể thao bao gồm: 1. Trang thiết bị tập thể lực chung cho vận động viên đội tuyển quốc gia các môn thể thao;..."*
- **Primary Failure Mechanism:** `SYNTAX_FRAGMENT_STRICTNESS`
- **Forensic Diagnosis:** The claim text is a noun phrase representing an enumerated list item. When evaluated in isolation without conversational context, D3 deemed the noun phrase incomplete as a standalone legal assertion, even though it matches Điều 5.1 verbatim.
- **Safety Boundary for Future Architectures:** Future verifiers must evaluate list items in context with the question, avoiding false rejections of valid verbatim statutory provisions.

---

### 6.4 True-Negative Comparison

| True Negative Target | Gold Label | D3 Prediction | Why D3 Successfully Rejected | Key Differential Signal |
| :--- | :--- | :--- | :--- | :--- |
| **`162759:PRIMARY:C1`** | `INSUFFICIENT` | `INSUFFICIENT` | Evidence E3 explicitly stated the procedure was simplified for *trường đại học tư thục* (private), while the claim asserted *công lập* (public). | **Explicit lexical antonym** (*công lập* vs *tư thục*) present in text. |
| **`3339:PRIMARY:C1`** | `CONTRADICTED` | `INSUFFICIENT` | The claim asserted that "không tuân thủ Điều 23" excluded coverage, whereas Điều 23 itself enumerates non-covered medical cases. | **Total absence of rule predicate** in cited text. |

**Key Takeaway from Comparison:**
D3 succeeds on negative detection **only when there is an overt lexical contradiction or total keyword absence**. It fails completely when the negative claim shares high lexical overlap with the evidence but introduces subtle legal substitutions (e.g. *Hạ sĩ quan* $\leftrightarrow$ *Sĩ quan*, *Người bảo vệ* $\leftrightarrow$ *Đương sự*, *Khảo nghiệm* $\leftrightarrow$ *Sản xuất*).

---

## 7. Root-Cause Ranking

1. **#1 Primary Root Cause: Lack of Explicit Legal Dimension Decomposition**
   - Single-call global entailment asks the LLM to make a holistic judgment. In legal text, holistic judgment is dominated by topical and lexical similarity, completely masking critical statutory boundaries (legal actor, activity type, condition scope).
2. **#2 Secondary Root Cause: Lexical Entailment Bias in Unconstrained Prompting**
   - When 80%+ of tokens in a claim match statutory text verbatim, standard prompting cannot overcome the prior that high lexical overlap implies entailment.
3. **#3 Tertiary Root Cause: Context-Blind Fragment Evaluation**
   - Evaluating atomic claims as isolated propositions without question-answer conversational grounding produces false rejections on noun-phrase list items.

---

## 8. Architecture Options for Next-Generation Verifier (V3)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VERIFIER ARCHITECTURE OPTIONS                          │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Option A                 │ Option B                 │ Option C (Recommended)│
│ Single-Call Prompt Tune  │ Two-Stage Filter/Audit   │ Structured Dimension  │
│                          │                          │ Decomposition         │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ - Same D3 architecture   │ - Stage 1: Coarse filter │ - 3 targeted boolean  │
│ - Few-shot / rules       │ - Stage 2: Adversarial   │   dimension checks    │
│ - Low complexity         │ - 1.5x call cost         │ - Deterministic logic │
│ - Fails on lexical bias  │ - Risk of overcalling    │ - Minimal schema      │
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

### 8.1 Detailed Option Comparison

| Dimension | Option A (Tuned D3) | Option B (Two-Stage) | Option C (Dimension Decomposition) |
| :--- | :--- | :--- | :--- |
| **Expected Negative Catch** | Low ($30-40\%$) | Medium ($50-60\%$) | **High ($> 65\%$)** |
| **Supported Retention Risk** | Low | Medium | **Low (Explicit checks avoid false rejects)** |
| **Provider Call Cost** | $1.0\times$ calls | $1.5\times$ calls | **$1.0\times$ calls** |
| **Schema Fragility** | Very Low | Medium | **Low (3 simple booleans vs D2's 6 enums)** |
| **Implementation Complexity** | Minimal | High | **Moderate** |
| **Observability & Diagnostics** | Opaque | Mixed | **High (Exact failing dimension logged)** |

### 8.2 Recommended Architecture: Option C (Structured Dimension Decomposition)
Design a streamlined, single-pass verifier with 3 focused boolean checks:
1. `legal_actor_aligned: bool`: Is the right/obligation assigned to the exact party stated?
2. `activity_and_scope_aligned: bool`: Is the legal activity (*sản xuất* vs *khảo nghiệm*) and scope exact?
3. `conditions_and_numbers_accurate: bool`: Are all statutory prerequisites and numerical figures verified?

**Deterministic Rule:**
$$\text{Label} = \begin{cases} \text{SUPPORTED} & \text{if all 3 checks are True} \\ \text{INSUFFICIENT} & \text{if any check is False and no contradiction} \\ \text{CONTRADICTED} & \text{if direct negation detected} \end{cases}$$

---

## 9. Pre-Registered Development Targets for Future Verifiers

Any future candidate (e.g. V3) is only viable if it meets the following pre-registered development targets on diagnostic benchmark data:
- **Supported Retention Rate:** $\ge 90.0\%$
- **Negative Catch Rate:** $\ge 60.0\%$
- **Execution Errors:** Strictly $0$
- **Development Iterations Capped:** Maximum 1–2 iterations before reassessing engineering allocation.

---

## 10. System-Level Strategic Assessment: Engineering Allocation Priorities

Based on current repository evidence across Milestone 49 to 51:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 ENGINEERING LEVERAGE / ROI COMPARISON                       │
├────────────────────────┬─────────┬──────────────────────────────────────────┤
│ System Component       │ ROI     │ Strategic Rationale                      │
├────────────────────────┼─────────┼──────────────────────────────────────────┤
│ 1. Generation Grounding│ HIGHEST │ Preventing actor/condition errors during │
│    & Prompt Design     │         │ generation improves answer accuracy and  │
│    (Task 2 Core)       │         │ Codabench METEOR/ROUGE-L directly.       │
├────────────────────────┼─────────┼──────────────────────────────────────────┤
│ 2. Retrieval & Rerank  │ HIGH    │ Passage relevance directly drives answer │
│    Depth Tuning        │         │ grounding and lexical overlap scores.    │
├────────────────────────┼─────────┼──────────────────────────────────────────┤
│ 3. Semantic Verifier   │ MEDIUM  │ Verifier is currently disabled in prod;  │
│    (Safety Layer)      │         │ serves as safety guardrail, not metric.  │
└────────────────────────┴─────────┴──────────────────────────────────────────┘
```

**Recommendation:** Shift primary engineering focus to **Generation Grounding & Prompt Optimization** and **Retrieval/Reranker Depth Tuning** for direct competition metric impact, while keeping verifier improvements lightweight and modular.
