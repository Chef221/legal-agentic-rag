# V2-D3 Development Forensic Audit & Architectural Failure Analysis

---

## 1. Executive Summary & Purpose

Candidate **V2-D3** (`StructuredSemanticCitationVerifierD3`) was executed on the frozen 38-claim composite development benchmark under canonical execution parameters (Qwen2.5-3B-Instruct, float16, greedy temperature 0.0, 2 evaluation passes) and achieved:
- **Verdict:** `V2_DEVELOPMENT_BENCHMARK_PASS`
- **Mechanical Stability:** 100.0% (38/38 claims identical across Pass 1 and Pass 2)
- **Zero Execution Errors:** `model_errors = 0`, `provider_invocation_errors = 0`, `structured_retries = 0` (76/76 successful provider calls)
- **Binary Claim Accuracy:** **28 / 38 (73.68%)** (TP=17, FP=9, TN=11, FN=1), exceeding V1 baseline (23/38, 60.53%) by **+5 net correct claims** (7 fixes, 2 regressions).
- **Three-Way Semantic Accuracy:** **24 / 38 (63.16%)** (SUPPORTED: 17/18, CONTRADICTED: 0/7, INSUFFICIENT: 7/13).
- **Answer-Level Accuracy:** **14 / 22 (63.64%)** (Valid Retained: 6/7 [85.71%], Invalid Caught: 8/15 [53.33%]).

Although V2-D3 meets all mechanical freeze criteria (`CANDIDATE_FREEZE_ELIGIBLE`), **V2-D3 is intentionally not frozen yet**. The fresh holdout remains strictly **SEALED**. 

This document performs an exhaustive, evidence-grounded forensic development audit of all **14 three-way errors** and **7 exact fixes** (21 unique claims) in V2-D3 to diagnose the root semantic mechanisms and inform the design of **V2-D3.1**.

---

## 2. Canonical Source Identities & Checksums

All analysis in this audit is verified against the canonical development artifacts:

| Artifact Role | Filename | SHA-256 Checksum | Record Count |
| :--- | :--- | :--- | :--- |
| **V2-D3 Execution Evidence** | `verification-v2-d3-development-evidence.zip` | `0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff` | 10 members (76 calls) |
| **Forensic Review Packets** | `verification-forensic-review-packets.zip` | `996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a` | 11 claims (Slice A) |
| **Forensic Labels** | `verification-human-forensic-labels-v1.json` | `bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733` | 11 claims (Slice A) |
| **Control Review Packets** | `verification-positive-control-review-packets-v1.zip` | `cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4` | 27 claims (Slice B) |
| **Control Labels** | `verification-positive-control-human-labels-v1.json` | `60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373` | 27 claims (Slice B) |
| **V1 Baseline Evidence** | `verification-semantic-benchmark-evidence.zip` | `bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6` | 38 claims baseline |

---

## 3. The 14 D3 Three-Way Semantic Errors

While D3 exhibits only 10 binary classification errors (9 FP, 1 FN), it exhibits **14 three-way semantic errors** because D3 failed to predict `CONTRADICTS` on any of the 7 human-contradicted claims (predicting 3 as `SUPPORTED` and 4 as `INSUFFICIENT`).

```
                              D3 Three-Way Confusion Matrix
                    ┌──────────────┬──────────────┬──────────────┐
                    │  SUPPORTED   │ CONTRADICTED │ INSUFFICIENT │
     ┌──────────────┼──────────────┼──────────────┼──────────────┤
     │ SUPPORTED    │      17      │      0       │      1       │  (1 Error: Group D)
     ├──────────────┼──────────────┼──────────────┼──────────────┤
Gold │ CONTRADICTED │       3      │      0       │      4       │  (7 Errors: Groups A & B)
     ├──────────────┼──────────────┼──────────────┼──────────────┤
     │ INSUFFICIENT │       6      │      0       │      7       │  (6 Errors: Group C)
     └──────────────┴──────────────┴──────────────┴──────────────┘
```

### Group A: False Entailment of Contradiction (Human: `CONTRADICTED` $\to$ D3: `SUPPORTED`)
1. **`102047:BASE:C1`** — Rule condition inverted (Bậc 3 electricity price applies to $<12$ months without declaration, not $\ge 12$ months with registration).
2. **`102047:CANDIDATE:C1`** — Statutory prerequisites omitted (unconditional direct power purchase contract vs mandatory homeowner commitment/representative).
3. **`103983:PRIMARY:C3`** — Statutory duration/article conflation (24 months active service required for peacetime completion vs 3 months reserve training).

### Group B: Contradiction Undercall (Human: `CONTRADICTED` $\to$ D3: `INSUFFICIENT`)
4. **`147239:CANDIDATE:C2`** — Actor duty inverted (Department of Tourism must issue certificates within 10 days vs Student completing course).
5. **`95861:BASE:C1`** — Entirely wrong subject matter / document (school management directive cited for expressway maintenance).
6. **`95861:CANDIDATE:C1`** — Actor entity mismatch (advisory bodies assisting the Minister vs the Minister himself).
7. **`150131:PRIMARY:C1`** *(V1 Regression)* — Authority conflict (Auditor General has exclusive replacement authority vs Head of unit proposing).

### Group C: False Entailment of Insufficient Evidence (Human: `INSUFFICIENT` $\to$ D3: `SUPPORTED`)
8. **`30405:PRIMARY:C1`** — Definition overgeneralization (endangered species specimens vs general forest animal specimens).
9. **`40489:PRIMARY:C1`** — Legal instrument mismatch (Program / Chương trình vs Scheme / Đề án).
10. **`5967:PRIMARY:C1`** — Professional rank mismatch (Hạng III $\to$ II criteria cited for Hạng II $\to$ I promotion).
11. **`5967:PRIMARY:C2`** — Professional rank mismatch (Hạng III $\to$ II criteria cited for Hạng II $\to$ I promotion).
12. **`5967:PRIMARY:C3`** — Professional rank mismatch (Hạng III $\to$ II criteria cited for Hạng II $\to$ I promotion).
13. **`75171:PRIMARY:C1`** — Procedural rule overgeneralized (dossier contents when submitting to Government vs universal submission mandate).

### Group D: Overrejection of Supported Claim (Human: `SUPPORTED` $\to$ D3: `INSUFFICIENT`)
14. **`31883:PRIMARY:C1`** *(V1 Regression)* — Overconservatism on internal cross-reference ("Đủ điều kiện theo quy định tại Điều 11 Thông tư này" rejected due to phrasing / internal reference).

---

## 4. The 7 Exact D3 Fixes Over V1

D3 fixed 7 distinct errors present in the V1 baseline:

1. **`26541:BASE:C1`** (`INSUFFICIENT`): V1 predicted `SUPPORTED` (due to "90 ngày" lexical match); D3 correctly rejected because reappointment rules do not establish dismissal replacement (`WRONG_ARTICLE_SCOPE`).
2. **`95861:BASE:C3`** (`INSUFFICIENT`): V1 predicted `SUPPORTED`; D3 correctly rejected school management directive cited for expressway maintenance contracts (`WRONG_DOCUMENT_SCOPE`).
3. **`95861:CANDIDATE:C2`** (`INSUFFICIENT`): V1 predicted `SUPPORTED`; D3 correctly recognized internal advisory department duties do not equal direct Ministry responsibility (`ACTOR_ROLE_MISMATCH`).
4. **`95861:CANDIDATE:C3`** (`SUPPORTED`): V1 predicted `CONTRADICTED` (false contradiction); D3 correctly identified exact literal statutory entailment of road inspection supervision (`SUPPORTED`).
5. **`103983:PRIMARY:C2`** (`SUPPORTED`): V1 predicted `INSUFFICIENT` (false rejection); D3 correctly recognized exact statutory support for militia completion (`SUPPORTED`).
6. **`108497:PRIMARY:C1`** (`INSUFFICIENT`): V1 predicted `CONTRADICTED` (hallucinated conflict); D3 correctly recognized that uninformative drug pricing text merely fails to establish the claim (`DOES_NOT_ESTABLISH`).
7. **`155139:PRIMARY:C1`** (`INSUFFICIENT`): V1 predicted `SUPPORTED` (fooled by clause number "7."); D3 correctly recognized that item index "7." is not a quantity of 7 incident levels (`QUANTITY_SEMANTIC_ROLE_MISMATCH`).

---

## 5. Claim-Level Forensic Matrix (21 Unique Records)

| Claim Key | Human | V1 | D3 Rel | D3 Pred | Outcome | Frozen Error Tags | Primary Failure / Gain Family | Semantic Mechanism | Contradiction? | Established? | Material Issue Axis | Recommended D3.1 Behavior | Regression Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `102047:BASE:C1` | CONT | SUP | ENTAILS | SUP | ERROR | CONDITION_INVERTED, SCOPE_OVERGENERALIZED | FALSE_ENTAILMENT_OF_CONTRADICTION | CONDITION_INVERTED | YES | NO | Condition | Detect inverted $<12$m vs $\ge 12$m condition $\to$ `CONTRADICTS` | Low |
| `102047:CANDIDATE:C1` | CONT | SUP | ENTAILS | SUP | ERROR | CONDITION_OMITTED, SCOPE_OVERGENERALIZED | FALSE_ENTAILMENT_OF_CONTRADICTION | CONDITION_OMITTED | YES | NO | Condition | Detect missing mandatory prerequisite $\to$ `CONTRADICTS` | Low |
| `103983:PRIMARY:C3` | CONT | SUP | ENTAILS | SUP | ERROR | CONDITION_OMITTED, QUANTITY_ERROR, WRONG_ARTICLE | FALSE_ENTAILMENT_OF_CONTRADICTION | WRONG_ARTICLE_SCOPE | YES | NO | Article, Quantity | Detect 24m active vs 3m reserve conflict $\to$ `CONTRADICTS` | Low |
| `147239:CANDIDATE:C2` | CONT | INS | DOES_NOT | INS | ERROR | ACTOR_ROLE_INVERTED | CONTRADICTION_UNDERCALLED | ACTOR_ROLE_MISMATCH | YES | NO | Actor | Detect Dept duty vs Student duty conflict $\to$ `CONTRADICTS` | Medium |
| `95861:BASE:C1` | CONT | INS | DOES_NOT | INS | ERROR | ACTOR_ROLE_INVERTED, WRONG_DOCUMENT | CONTRADICTION_UNDERCALLED | WRONG_DOCUMENT_SCOPE | UNCLEAR | NO | Source Scope | Maintain non-support, evaluate conflict $\to$ `INSUFFICIENT` / `CONTRADICTS` | Low |
| `95861:CANDIDATE:C1` | CONT | INS | DOES_NOT | INS | ERROR | ACTOR_ROLE_INVERTED, WRONG_DOCUMENT | CONTRADICTION_UNDERCALLED | ACTOR_ROLE_MISMATCH | YES | NO | Actor | Detect assisting bodies vs Minister conflict $\to$ `CONTRADICTS` | Medium |
| `150131:PRIMARY:C1` | CONT | CONT | DOES_NOT | INS | ERROR | ACTOR_ROLE_INVERTED, WRONG_DOCUMENT | CONTRADICTION_UNDERCALLED | ACTOR_AUTHORITY_CONFLICT | YES | NO | Actor, Authority | Detect Auditor General vs Unit Head authority conflict $\to$ `CONTRADICTS` | High |
| `30405:PRIMARY:C1` | INS | SUP | ENTAILS | SUP | ERROR | SCOPE_OVERGENERALIZED, WRONG_DOCUMENT | FALSE_ENTAILMENT_OF_INSUFFICIENT | CLAIM_OVERGENERALIZATION | NO | NO | Source Scope | Reject endangered vs general forest animals $\to$ `INSUFFICIENT` | Low |
| `40489:PRIMARY:C1` | INS | SUP | ENTAILS | SUP | ERROR | OTHER | FALSE_ENTAILMENT_OF_INSUFFICIENT | LEGAL_INSTRUMENT_MISMATCH | NO | NO | Instrument Type | Reject Program vs Scheme mismatch $\to$ `INSUFFICIENT` | Low |
| `5967:PRIMARY:C1` | INS | SUP | ENTAILS | SUP | ERROR | SCOPE_OVERGENERALIZED, OTHER | FALSE_ENTAILMENT_OF_INSUFFICIENT | RANK_LEVEL_MISMATCH | NO | NO | Article, Rank | Reject Hạng III $\to$ II cited for Hạng II $\to$ I $\to$ `INSUFFICIENT` | Low |
| `5967:PRIMARY:C2` | INS | SUP | ENTAILS | SUP | ERROR | SCOPE_OVERGENERALIZED, OTHER | FALSE_ENTAILMENT_OF_INSUFFICIENT | RANK_LEVEL_MISMATCH | NO | NO | Article, Rank | Reject Hạng III $\to$ II cited for Hạng II $\to$ I $\to$ `INSUFFICIENT` | Low |
| `5967:PRIMARY:C3` | INS | SUP | ENTAILS | SUP | ERROR | SCOPE_OVERGENERALIZED, OTHER | FALSE_ENTAILMENT_OF_INSUFFICIENT | RANK_LEVEL_MISMATCH | NO | NO | Article, Rank | Reject Hạng III $\to$ II cited for Hạng II $\to$ I $\to$ `INSUFFICIENT` | Low |
| `75171:PRIMARY:C1` | INS | SUP | ENTAILS | SUP | ERROR | SCOPE_OVERGENERALIZED, WRONG_ARTICLE | FALSE_ENTAILMENT_OF_INSUFFICIENT | CLAIM_OVERGENERALIZATION | NO | NO | Condition, Scope | Reject submission dossier rule as universal mandate $\to$ `INSUFFICIENT` | Low |
| `31883:PRIMARY:C1` | SUP | SUP | DOES_NOT | INS | ERROR | NONE | OVERREJECTION_OF_SUPPORTED | SUPPORTED_OVERREJECTED | NO | YES | None | Retain cross-reference support $\to$ `SUPPORTED` | High |
| `26541:BASE:C1` | INS | SUP | DOES_NOT | INS | FIX | WRONG_DOCUMENT, WRONG_ARTICLE | GAIN_PRESERVATION | WRONG_ARTICLE_SCOPE | NO | NO | Article | Maintain rejection of reappointment for dismissal $\to$ `INSUFFICIENT` | High (Guardrail) |
| `95861:BASE:C3` | INS | SUP | DOES_NOT | INS | FIX | WRONG_DOCUMENT | GAIN_PRESERVATION | WRONG_DOCUMENT_SCOPE | NO | NO | Source Scope | Maintain rejection of school directive for expressway $\to$ `INSUFFICIENT` | High (Guardrail) |
| `95861:CANDIDATE:C2` | INS | SUP | DOES_NOT | INS | FIX | ACTOR_ROLE_INVERTED, WRONG_DOCUMENT | GAIN_PRESERVATION | ACTOR_ROLE_MISMATCH | NO | NO | Actor | Maintain rejection of advisory unit for Ministry duty $\to$ `INSUFFICIENT` | High (Guardrail) |
| `95861:CANDIDATE:C3` | SUP | CONT | ENTAILS | SUP | FIX | NONE | GAIN_PRESERVATION | EXACT_LITERAL_SUPPORT | NO | YES | None | Maintain literal statutory support $\to$ `SUPPORTED` | High (Guardrail) |
| `103983:PRIMARY:C2` | SUP | INS | ENTAILS | SUP | FIX | NONE | GAIN_PRESERVATION | EXACT_LITERAL_SUPPORT | NO | YES | None | Maintain literal statutory support $\to$ `SUPPORTED` | High (Guardrail) |
| `108497:PRIMARY:C1` | INS | CONT | DOES_NOT | INS | FIX | SCOPE_OVERGENERALIZED, WRONG_DOCUMENT | GAIN_PRESERVATION | AVOID_FALSE_CONTRADICTION | NO | NO | None | Maintain `INSUFFICIENT` without hallucinated conflict $\to$ `INSUFFICIENT` | High (Guardrail) |
| `155139:PRIMARY:C1` | INS | SUP | DOES_NOT | INS | FIX | QUANTITY_ERROR, WRONG_ARTICLE | GAIN_PRESERVATION | QUANTITY_ROLE_DISCRIMINATION | NO | NO | Quantity | Maintain rejection of clause number as incident level $\to$ `INSUFFICIENT` | High (Guardrail) |

---

## 6. Aggregate Root Cause Analysis

### 6.1 Distribution by Failure Family (14 Errors)
1. **`FALSE_ENTAILMENT_OF_INSUFFICIENT_EVIDENCE`**: **6 claims (42.9%)** — Model entails claims that have strong lexical overlap but describe a different rank (`5967`), overgeneralized scope (`30405`, `75171`), or mismatched legal instrument (`40489`).
2. **`CONTRADICTION_UNDERCALLED_AS_INSUFFICIENT`**: **4 claims (28.6%)** — Model correctly realizes the evidence does not support the claim, but defaults to `DOES_NOT_ESTABLISH` instead of identifying the direct legal incompatibility (`147239`, `95861:BASE:C1`, `95861:CANDIDATE:C1`, `150131`).
3. **`FALSE_ENTAILMENT_OF_CONTRADICTION`**: **3 claims (21.4%)** — Model completely misses direct legal contradictions (inverted condition in `102047:BASE:C1`, omitted prerequisite in `102047:CANDIDATE:C1`, 24m vs 3m conflict in `103983:PRIMARY:C3`).
4. **`OVERREJECTION_OF_SUPPORTED_CLAIM`**: **1 claim (7.1%)** — Model over-rejects valid claim due to internal article cross-reference (`31883:PRIMARY:C1`).

### 6.2 Capabilities Required to Resolve the 14 Errors
- **Explicit Conflict & Contradiction Discrimination:** 7 claims (all 7 gold CONTRADICTED cases).
- **Strict 100% Full-Establishment Checking (Anti-False Entailment):** 9 claims (all 9 false ENTAILS).
- **Statutory Article & Professional Rank Scoping:** 6 claims (`5967:C1,C2,C3`, `103983:C3`, `95861:BASE:C1`, `75171:C1`).
- **Condition & Exception Verification:** 3 claims (`102047:BASE:C1`, `102047:CANDIDATE:C1`, `75171:C1`).
- **Actor & Authority Role Discrimination:** 3 claims (`147239:C2`, `95861:C1`, `150131:C1`).
- **Quantity & Numerical Role Discrimination:** 2 claims (`103983:C3`, `102047:BASE:C1`).

---

## 7. Contradiction-Specific Forensic Audit

In the entire 38-claim benchmark, there are **7 human-CONTRADICTED claims**. D3 predicted `CONTRADICTS: 0`.

### Why D3 Failed on Contradictions:
1. **Lack of Incompatibility Focus in Prompt:** D3 prompt defined `CONTRADICTS` as "Bằng chứng khẳng định điều trái ngược trực tiếp". For a 3B LLM, "trái ngược trực tiếp" was interpreted as requiring explicit syntactic negation words (e.g. "không được", "cấm").
2. **Failure on Structural & Inverted Contradictions:**
   - **`150131:PRIMARY:C1`** (Authority conflict): Evidence says *Tổng Kiểm toán nhà nước decides* while *Unit Head proposes*. The claim says *Unit Head has authority*. This is an explicit legal conflict, but lacks the word "không". D3 treated it as merely unestablished.
   - **`102047:BASE:C1`** (Condition inversion): Evidence says *Bậc 3 applies to $<12$ months*. Claim says *Bậc 3 applies to $\ge 12$ months*. The condition is inverted, meaning the claim asserts the opposite of statutory law. D3 saw the matching keywords and called it `ENTAILS`.
   - **`147239:CANDIDATE:C2`** (Actor duty inversion): Evidence assigns 10-day deadline to the *Department of Tourism*. Claim assigns 10-day deadline to the *Student*.

---

## 8. False-Entailment Forensic Audit

D3 produced **26 ENTAILS** predictions across the 38 claims, of which **9 were false entailments** (3 gold CONTRADICTED, 6 gold INSUFFICIENT).

### Tested Hypotheses for False Entailment:
- **H1: Topical Overlap False Entailment (Confirmed in 6/9 claims):** When claim and evidence share high vocabulary density (e.g., `5967` public health promotion terms, `30405` wildlife specimen definitions), D3 defaults to `ENTAILS`.
- **H2: Sub-Proposition Cherry-Picking (Confirmed in 4/9 claims):** D3 verified that the predicate existed in the text, but failed to verify that the *subject/rank/condition* matched (e.g., verifying "đơn vị có nhu cầu" without verifying "Y tế công cộng cao cấp" in `5967`).
- **H3: Condition & Scope Blindness (Confirmed in 3/9 claims):** Omission of statutory prerequisites was ignored (`102047:CANDIDATE:C1`, `75171:PRIMARY:C1`).

---

## 9. Overrejection Guardrail Audit (`31883:PRIMARY:C1`)

- **Claim:** "Để được bổ nhiệm Giám định viên cao cấp, cần đủ điều kiện theo quy định tại Điều 11 Thông tư này ."
- **Evidence:** Điều 14 khoản 1 Thông tư 32/2022/TT-BCA: "1. Đủ điều kiện theo quy định tại Điều 11 Thông tư này."
- **Why D3 Over-rejected:** D3 was overly conservative regarding internal statutory references ("theo quy định tại Điều 11 Thông tư này") and slight paraphrasing ("Để được bổ nhiệm" vs "Tiêu chuẩn chức danh").
- **D3.1 Guardrail:** D3.1 must recognize that valid statutory claims frequently summarize article standards or cross-reference internal sections. Retaining **$\ge 17/18$ supported claims** is a non-negotiable quality invariant.

---

## 10. D3 Gain-Preservation Guardrails

From the 7 D3 fixes over V1, the following mandatory guardrails are established for D3.1:

- **G1 (Statutory Event Discrimination):** Do not conflate distinct statutory events (e.g., reappointment rules do not validate dismissal replacement; `26541:BASE:C1`).
- **G2 (Source Document Integrity):** Maintain strict rejection of completely irrelevant statutory documents (`95861:BASE:C3`).
- **G3 (Internal Sub-Unit vs Ministry Authority):** Maintain rejection when internal departmental advice is claimed as direct Ministry executive responsibility (`95861:CANDIDATE:C2`).
- **G4 (Literal Statutory Support):** Preserve exact literal statutory entailment without hallucinating contradictions (`95861:CANDIDATE:C3`, `103983:PRIMARY:C2`).
- **G5 (Absence of Evidence is Not Contradiction):** Do not classify uninformative evidence as `CONTRADICTED` when the evidence merely fails to establish the claim (`108497:PRIMARY:C1`).
- **G6 (Clause Index vs Numerical Quantity):** Maintain discrimination between document structure (clause/item numbers like "7.") and actual quantitative metrics (`155139:PRIMARY:C1`).

---

## 11. Evaluation of Two-Stage Verification Hypothesis

We formally evaluate the hypothesis: *Should D3.1 split verification into Stage A (Conflict/Contradiction Check) followed by Stage B (Full Establishment Check)?*

- **Evaluation:** **`PARTIALLY_SUPPORTED`**
- **Rationale:**
  - *Supporting Evidence:* D3's 0/7 performance on contradictions confirms that a single multi-class relation choice causes Qwen-3B to collapse toward `ENTAILS` or `DOES_NOT_ESTABLISH`. Decomposing conflict detection from entailment verification is semantically sound.
  - *Contra-Indications:* Executing **two separate model calls per claim** doubles operational latency (76 calls $\to$ 152 calls across 2 passes), increases timeout risk, and creates compounding rejection failure modes.
  - *Synthesis:* The decomposition is necessary **logically and semantically**, but should ideally be achieved via a **structured multi-step single-call prompt** (Option A) rather than two physically separate inference calls (Option B).

---

## 12. Proposed D3.1 Candidate Architectures

### Option A: Hierarchical Single-Call Verifier with Explicit 2-Gate Decomposition *(RECOMMENDED)*
- **Architecture:** Single provider call per claim. The prompt guides the model through two explicit, ordered diagnostic questions before deriving the final label:
  1. *Gate 1 (Conflict / Incompatibility Check):* Does the evidence assert an opposing rule, an incompatible actor/authority, an inverted condition, or a conflicting quantity? $\to$ `is_contradicted: boolean`.
  2. *Gate 2 (Full Establishment Check):* Does the evidence explicitly and fully establish 100% of all material propositions, subjects, ranks, and conditions in the claim? $\to$ `is_fully_established: boolean`.
  - **Deterministic Label Rule:** If `is_contradicted` is True $\to$ `CONTRADICTED`. Else if `is_fully_established` is True $\to$ `SUPPORTED`. Else $\to$ `INSUFFICIENT`.
- **Targeted Errors:** Fixes Group A & B (by explicitly priming conflict checks) and Group C (by requiring 100% material proposition coverage).
- **Protected Gains:** Preserves all 7 D3 fixes and maintains 1 call per claim (38 calls/pass).
- **Model Calls per Claim:** **1 call**.
- **Complexity:** Low; straightforward boolean schema.

### Option B: Two-Stage Sequential Provider Verifier (Stage 1: Conflict $\to$ Stage 2: Entailment)
- **Architecture:** Two independent model invocations per claim:
  - *Call 1:* Dedicated Contradiction Classifier (Focuses strictly on legal conflict, authority clash, inverted conditions). If CONTRADICTED $\to$ Stop.
  - *Call 2:* Dedicated Entailment Classifier (Focuses strictly on complete vs partial/insufficient support).
- **Targeted Errors:** Maximum separation for Group A/B errors.
- **Model Calls per Claim:** **1 to 2 calls** (up to 76 calls/pass, 152 calls total).
- **Risks:** Doubled execution time; potential over-rejection in Call 1.

### Option C: Propositional Breakdown & Fact-Checking Verifier
- **Architecture:** Single call where the model breaks the claim into atomic sub-propositions `[P1, P2, ...]` and evaluates each as `SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`.
- **Targeted Errors:** Highly effective against partial false entailment (Group C).
- **Risks:** High schema complexity; 3B model may struggle with dynamic array generation and JSON parsing stability.

### Architecture Recommendation Ranking:
1. **OPTION A (Hierarchical Single-Call 2-Gate Verifier):** **Rank 1 (Recommended)** — Highest performance/simplicity ratio, zero added latency, directly resolves contradiction and partial-entailment failure modes.
2. **OPTION B (Two-Stage Sequential Verifier):** **Rank 2 (Fallback)** — Strong semantic isolation, but high latency overhead.
3. **OPTION C (Propositional Breakdown):** **Rank 3** — Excess schema complexity for 3B parameter ceiling.

---

## 13. D3.1 Development Quality Targets

When D3.1 is eventually implemented and evaluated, it must satisfy:

- **Mechanical Targets:**
  - `model_errors = 0`
  - `provider_invocation_errors = 0`
  - `unstable_semantic_claims = 0`
  - `38 / 38` successfully evaluated claims
- **Quality Targets vs D3:**
  - Binary Claim Accuracy: **$> 28 / 38$ ($> 73.68\%$)**
  - Supported Retained: **$\ge 17 / 18$ ($\ge 94.44\%$)**
  - Negative Caught: **$> 11 / 20$ ($> 55.00\%$)**
  - Paired Net Correctness Delta vs D3: **$> 0$**
  - Three-Way Claim Accuracy: **$> 24 / 38$ ($> 63.16\%$)** (specifically recovering `CONTRADICTED` discrimination).
  - Answer-Level Accuracy: **$\ge 14 / 22$ ($\ge 63.64\%$)**.

---

## 14. Explicit Governance & Compliance Statements

- **V2-D3 FORENSIC DEVELOPMENT AUDIT COMPLETED**
- **V2-D3 REMAINS CURRENT BEST DEVELOPMENT CANDIDATE**
- **V2-D3 NOT YET FROZEN**
- **D3.1 NOT IMPLEMENTED IN THIS PASS**
- **FRESH HOLDOUT REMAINS STRICTLY SEALED**
- **NO HOLDOUT DATA ACCESSED, OPENED, UNZIPPED, OR RECONSTRUCTED**
- **NO HOLDOUT QIDS COMPUTED OR INSPECTED**
- **NO REAL QWEN INFERENCE RUN**
- **NO KAGGLE EXECUTION**
- **NO PRODUCTION WIRING MODIFIED**
- **NO RETRIEVAL / RERANKING / GENERATION CHANGES**
- **NO TRAINING / FINE-TUNING**
- **ZERO CODE CHANGES IN `src/`, `scripts/`, `tests/`, `configs/`**
