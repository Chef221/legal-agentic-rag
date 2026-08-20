# 27 — Priority B: Verification-Correctness Forensic Audit

## 1. Executive Summary & Audit State

- **Frontier**: Priority B — Verification-Correctness Audit
- **Milestone Task**: B-FORENSIC-0 (Materialization) & B-FORENSIC-1A (Human Labels Freeze v1)
- **Status**: `HUMAN_FORENSIC_LABELS_FROZEN`
- **Production Code Changes**: NONE (`src/` untouched)
- **Model / Verifier Status**:
  - `ModelBackedCitationVerifier` remains DISABLED (`semantic_verification.enabled = false`).
  - Production `RuleBasedCitationVerifier` behavior is UNCHANGED.
  - S20 remains the active production configuration; H40 remains UNPROMOTED.
- **Human Review Package**:
  - Source Review Archive: `verification-forensic-review-packets.zip` (SHA-256 `996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a`, `42,826 bytes`)
  - Frozen Label Artifact JSON: `verification-human-forensic-labels-v1.json` (SHA-256 `bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733`, `11,849 bytes`)
  - Frozen Label Artifact ZIP: `verification-human-forensic-labels-v1.zip` (SHA-256 `25c23b80fb94a59976ccd821944355ff80aa60c7360e32a3dd8dea19dae12cbb`, `3,118 bytes`)

---

## 2. Frozen Historical Evidence Provenance

The forensic materializer operates strictly on frozen historical data without executing any live retrieval, reranking, or generative models.

### 2.1 External B1A Evidence Source
- **Archive Filename**: `phase-b1a-graph-routing-ablation-evidence.zip`
- **Observed Archive Size**: `42,993 bytes`
- **Observed Archive SHA-256**:
  `b88ccce928b4cecfc9239d490c59f91405834c5a3199f917d757c5735b1d6631`
- **Archive Member Validation (12/12 Verified)**:
  - `configs/phase-b1a-graph-routing-cases.json`: `b1efe824f320d9323af462869fd8842ef8544fa14d5f81ae35decca99e1ee99f`
  - `evidence/materialized_questions_identity.json`: `abad62cb31dc24bc40213ada580f8b464bfe2f98d1340d3820fd10de338ebcd3`
  - `configs/base_runtime_config.json`: `03a32009c0dc9a68ac93538710ec741b7a7e68a8ef9e160116ecc6bcb76d64fc`
  - `configs/candidate_runtime_config.json`: `27a490b947336a2f3aa0c34e9f7a19494be28193f10e5934c5909930fea7f99a`
  - `results/phase_b1a_paired_report.json`: `ed7b5129539a4f31b4f8b9153ef8060d75d58cebee1d7932b346d63b0dc1e0e7`
  - `results/phase_b1a_decision_report.json`: `6fce0e2daf6af1a50fdc1ed41bba271b99c7f6844e73edd1f3f52babdd365c5e`
  - `base_batch/manifest.json`: `72cb07ae18d9539357e693b0bd4385565980b9d22b5cac054cb7d3a0a0012406`
  - `candidate_batch/manifest.json`: `c64baf71ff13ccb2d864e84e875b06bdeb204dd636a45ad25ba6b3bed2499908`
  - `base_batch/results.jsonl`: `c72dc38f37945831095c71ba4b0f24f328de2e01dc4f7ecac6e527b997dc3cac`
  - `candidate_batch/results.jsonl`: `420d2297d529c3d8c246de499111840d4a4a98b010bd95e42639b7ba9f3fb6ad`
  - `base_batch/batch_state.json`: `af15d1676144570ea75c9183119ea6c6890aed062554db516c0a3ddb642cb159`
  - `candidate_batch/batch_state.json`: `11054fe5a9568da70f4e3f694185b846a8097eb90e37233825db1a02635ae219`

### 2.2 Historical Arms Identity
- **BASE Batch (`base_batch`)**:
  - `record_count`: 22
  - `code_version`: `0.50.6`
  - `question_source_sha256`: `f5d681c447a2bb964de90298207af0363c76b3546bfa027603d7fa98322a3ce3`
  - `records_sha256`: `c72dc38f37945831095c71ba4b0f24f328de2e01dc4f7ecac6e527b997dc3cac` (Exact match)
- **CANDIDATE Batch (`candidate_batch`)**:
  - `record_count`: 22
  - `code_version`: `0.50.6`
  - `question_source_sha256`: `f5d681c447a2bb964de90298207af0363c76b3546bfa027603d7fa98322a3ce3`
  - `records_sha256`: `420d2297d529c3d8c246de499111840d4a4a98b010bd95e42639b7ba9f3fb6ad` (Exact match)

### 2.3 Benchmark & Serving Artifact Provenance
- **Canonical Development Benchmark**:
  - Filename: `development.json`
  - Record Count: 991
  - SHA-256: `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8` (Exact match)
- **Canonical Serving Artifact Root**:
  - Artifact Directory: `uit-dsc-2026-task2-v0400/legal_chunks`
  - `artifact_type`: `legal_chunks`
  - `dataset_name`: `uit-dsc-2026-task2-selected-contexts`
  - `dataset_revision`: `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`
  - `record_count`: 330,768
  - `payload_integrity_verified`: `true`
  - `payload_sha256`: `3a769121f07aa1c65b69569ce296b416f40048ba47b9761a393c245ece609872`

---

## 3. Four Target Cases & Reconstructed Outcomes

The initial audit investigates 4 target questions across both historical arms (total 8 historical execution arms):

| Question ID | BASE Stop Reason | BASE Verified | CANDIDATE Stop Reason | CANDIDATE Verified | Selected Chunks | Lookup Pass | Source Mapping Pass | Replay Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **102047** | `answer_verified` | Yes (`is_valid=True`) | `answer_verified` | Yes (`is_valid=True`) | 5 / 5 | 100% Pass | 100% Pass | Matched 100% |
| **147239** | `generation_failed` | No (Replay N/A) | `answer_verified` | Yes (`is_valid=True`) | 5 / 5 | 100% Pass | 100% Pass | Matched 100% |
| **26541** | `answer_verified` | Yes (`is_valid=True`) | `generation_failed` | No (Replay N/A) | 5 / 5 | 100% Pass | 100% Pass | Matched 100% |
| **95861** | `answer_verified` | Yes (`is_valid=True`) | `answer_verified` | Yes (`is_valid=True`) | 5 / 5 | 100% Pass | 100% Pass | Matched 100% |

### 3.1 Source Mapping & Verifier Replay Results
- **Source Mapping Invariant**: 8 / 8 arms passed exact 1-to-1 cross-checks between `selected_evidence` (`E1..En`) and `context.selection_trace` (`selected == True` ordered by `selection_rank`).
- **Applicable Arms**: 6 (where historical `citation_verification` was reached and `stop_reason == answer_verified`)
- **Non-Applicable Arms**: 2 (`147239` BASE and `26541` CANDIDATE, preserved with `reason: "historical_verifier_not_reached"`)
- **Replay Pass Rate**: 6 / 6 (100% bit-for-bit fidelity with frozen citation verification records including claim IDs, claim texts, token overlap, numeric matches, and negation checks).

---

## 4. Human Forensic Labels v1 (B-FORENSIC-1A)

The human forensic labels are an immutable overlay over frozen review packets, capturing verified human judgments on claim entailment and granular error modes.

### 4.1 Provenance & Authority Statement
- **Approval Kind**: `explicit_user_human_approval`
- **Approval Date**: `2026-08-20`
- **Reviewer Identifier**: `human_reviewer_1`
- **Scope Contract**:
  > *"These labels are internal human forensic annotations over frozen train-derived development outputs and supplied frozen evidence. They are not official UIT DSC relevance or legal-answer ground-truth labels."*
- **Usage Policy**:
  - **Allowed**: `verification_correctness_evaluation`, `forensic_analysis`
  - **Strictly Prohibited**: `training`, `fine_tuning`, `retrieval_relevance_supervision`, `public_test_annotation`, `private_test_annotation`, `manual_submission_correction`

### 4.2 Exact Label Counts

Across the 8 historical arms (6 verified arms + 2 generation-failed arms):

| Metric | Count |
| :--- | :--- |
| **Target Questions** | 4 |
| **Historical Arms** | 8 |
| **Total Labeled Claims** | 11 |
| **`SUPPORTED`** | 2 |
| **`CONTRADICTED`** | 5 |
| **`INSUFFICIENT`** | 4 |
| **Generation-Failed Unlabeled Arms** | 2 |

> [!NOTE]
> These cases represent a deliberately selected suspicious forensic subset and **must not** be used to infer an overall system-wide verifier false-positive rate.

### 4.3 Detailed Case-by-Case Claim Matrix

| Question ID | Arm | Historical Status | Claim ID | Human Entailment Label | Granular Error Tags |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **102047** | BASE | `answer_verified` | C1 | `CONTRADICTED` | `CONDITION_INVERTED`, `SCOPE_OVERGENERALIZED` |
| **102047** | CANDIDATE | `answer_verified` | C1 | `CONTRADICTED` | `CONDITION_OMITTED`, `SCOPE_OVERGENERALIZED` |
| **147239** | BASE | `generation_failed` | — | *(Unlabeled)* | *(None — generation_failed)* |
| **147239** | CANDIDATE | `answer_verified` | C1 | `SUPPORTED` | `NONE` |
| **147239** | CANDIDATE | `answer_verified` | C2 | `CONTRADICTED` | `ACTOR_ROLE_INVERTED` |
| **26541** | BASE | `answer_verified` | C1 | `INSUFFICIENT` | `WRONG_DOCUMENT`, `WRONG_ARTICLE` |
| **26541** | CANDIDATE | `generation_failed` | — | *(Unlabeled)* | *(None — generation_failed)* |
| **95861** | BASE | `answer_verified` | C1 | `CONTRADICTED` | `ACTOR_ROLE_INVERTED`, `WRONG_DOCUMENT` |
| **95861** | BASE | `answer_verified` | C2 | `INSUFFICIENT` | `WRONG_DOCUMENT` |
| **95861** | BASE | `answer_verified` | C3 | `INSUFFICIENT` | `WRONG_DOCUMENT` |
| **95861** | CANDIDATE | `answer_verified` | C1 | `CONTRADICTED` | `ACTOR_ROLE_INVERTED`, `WRONG_DOCUMENT` |
| **95861** | CANDIDATE | `answer_verified` | C2 | `INSUFFICIENT` | `ACTOR_ROLE_INVERTED`, `WRONG_DOCUMENT` |
| **95861** | CANDIDATE | `answer_verified` | C3 | `SUPPORTED` | `NONE` |

---

## 5. Frozen Label Overlay Schema

The human label artifact overlay is formatted as follows:

```json
{
  "schema_version": "1.0",
  "artifact_type": "verification_human_forensic_labels",
  "label_version": "v1",
  "verdict": "HUMAN_FORENSIC_LABELS_FROZEN",
  "source_review_package": {
    "filename": "verification-forensic-review-packets.zip",
    "sha256": "996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a",
    "size_bytes": 42826
  },
  "approval": {
    "approval_kind": "explicit_user_human_approval",
    "approval_date": "2026-08-20",
    "reviewer_id": "human_reviewer_1",
    "organizer_ground_truth": false,
    "legal_expert_credential_asserted": false,
    "approval_statement": "..."
  },
  "usage_policy": {
    "allowed_initial_uses": ["verification_correctness_evaluation", "forensic_analysis"],
    "prohibited_initial_uses": ["training", "fine_tuning", "retrieval_relevance_supervision", "public_test_annotation", "private_test_annotation", "manual_submission_correction"]
  },
  "questions": {
    "102047": {
      "question_id": "102047",
      "arms": {
        "BASE": {
          "historical_stop_reason": "answer_verified",
          "claim_review_applicable": true,
          "claims": {
            "C1": {
              "claim_id": "C1",
              "claim_text_sha256": "...",
              "claim_text": "...",
              "entailment_label": "CONTRADICTED",
              "error_tags": ["CONDITION_INVERTED", "SCOPE_OVERGENERALIZED"],
              "diagnostic_metadata": {
                "status": "diagnostic_not_ground_truth",
                "historical_rule_status": "supported",
                "historical_evidence_ids": ["E1"]
              }
            }
          }
        },
        "CANDIDATE": { ... }
      }
    }
  },
  "aggregate": {
    "question_count": 4,
    "historical_arm_count": 8,
    "labeled_claim_count": 11,
    "supported": 2,
    "contradicted": 5,
    "insufficient": 4,
    "generation_failed_unlabeled_arms": 2
  }
}
```

---

## 6. Execution Invariants & Verification Checklist

- [x] Zero retrieval reruns
- [x] Zero generation reruns
- [x] Zero model-backed semantic verifier calls
- [x] Zero auto-generated correctness labels
- [x] Paired BASE and CANDIDATE arms preserved
- [x] 100% deterministic chunk reconstruction from canonical `uit-dsc-2026-task2-v0400/legal_chunks`
- [x] In-protocol payload integrity verification of serving artifacts (`records.jsonl` SHA verified)
- [x] 100% source mapping cross-check between `selected_evidence` and `selection_trace`
- [x] 100% verifier replay fidelity across all 6 applicable historical arms
- [x] Exact human claim-text binding: all 11 claims bound to exact frozen UTF-8 claim text SHA-256 digests
- [x] Immutable separation: source review packets remain unmutated; human labels generated as overlay
- [x] 20/20 freezer unit tests passing (`tests/unit/evaluation/test_freeze_verification_forensic_labels.py`)
- [x] 35/35 materializer unit tests passing (`tests/unit/evaluation/test_verification_forensic_packets.py`)
- [x] Full pytest test suite passing
- [x] Real human label artifacts written outside tracked repository content

---

## 7. Task B-FORENSIC-1B — Positive-Control Candidate Selection

### 7.1 Objective & Pre-Registration Rationale

The initial human forensic set (B-FORENSIC-0 / B-FORENSIC-1A) was deliberately drawn from known failure and divergence cases. Benchmarking a candidate semantic verifier solely on negative/suspicious cases would introduce severe selection bias (a trivial verifier that unconditionally rejects every claim would appear 100% accurate).

To establish an unbiased evaluation benchmark, a balanced set of **positive-control candidates** was sampled and pre-registered from historical Phase-A `answer_verified` outputs prior to running any semantic verifier.

> [!IMPORTANT]
> **Candidate Status**: These cases are designated as **positive-control candidates**, not confirmed positives. Human review has not yet been performed on them. If any candidate receives a negative human label (`CONTRADICTED` or `INSUFFICIENT`), it will be retained in the evaluation set as an additional negative case and will not be filtered or replaced.

### 7.2 Source Authority & Pre-Filter

- **Phase-A Evidence Archive**: `phase-a-current-system-census-final-evidence.zip` (SHA-256: `df05a401599c43a28e39136d72b225841b242d10a40dc5bc475b9be6ed86be8b`, 1,036,904 bytes)
- **Phase-A Results Source**: `phase-a-current-system-census-batch/results.jsonl` (SHA-256: `7b1bf802c752e37cee7386c0b24f6e0ee5ea2f65056b22eaa9488d73161aaee6`, 991 records)
- **Historical Stop-Reason Invariants**:
  - `answer_verified`: 806
  - `generation_failed`: 177
  - `citation_verification_failed`: 7
  - `max_retry_reached`: 1
- **Exclusions**:
  - All 22 historical relationship-case IDs (including the 4 forensic target questions `102047`, `147239`, `26541`, `95861`) were excluded to broaden legal domain coverage.
- **Eligible Pool**: Exactly 788 records passed all eligibility gates (`stop_reason == "answer_verified"`, `is_valid == true`, non-empty `selected_evidence`, non-empty `selection_trace`, non-empty `claim_verifications`).

### 7.3 Deterministic Stratification & Precedence

Each eligible record was assigned to exactly one stratum based on deterministic pre-review telemetry:

```text
Precedence: D_NEGATION_MODALITY -> C_NUMERIC -> B_MULTI_CLAIM_CLEAN -> A_SINGLE_CLAIM_CLEAN
```

1. **`D_NEGATION_MODALITY`** (Count: 168): At least one verified claim containing negation terms (`bãi`, `cấm`, `chưa`, `hủy`, `không`, `ngoại`, `trừ`) with historical `negation_match == true`.
2. **`C_NUMERIC`** (Count: 181): At least one verified claim containing numeric tokens with historical `numeric_match == true`.
3. **`B_MULTI_CLAIM_CLEAN`** (Count: 121): $\ge 2$ verified claims with no numeric or negation mismatch.
4. **`A_SINGLE_CLAIM_CLEAN`** (Count: 318): Exactly 1 verified claim with no numeric or negation mismatch.

### 7.4 Deterministic Sampling Formula

For each eligible record, a deterministic selection key was computed:

$$\text{selection\_key} = \text{SHA256}(\text{"verification-positive-control-v1|"} + \text{question\_id})$$

Within each stratum, records were sorted by `(selection_key, question_id)` ascending:
- **PRIMARY**: First 4 candidates per stratum (Total: 16 PRIMARY)
- **RESERVE**: Next 2 candidates per stratum (Total: 8 RESERVE)

### 7.5 Pre-Registered Candidate Matrix

#### PRIMARY Candidates (16 IDs)

| Stratum | Question ID | Claim Count | Selection Key Prefix | Replay Status |
| :--- | :--- | :--- | :--- | :--- |
| `D_NEGATION_MODALITY` | `108497` | 1 | `030097439e919c2b...` | `PASS` |
| `D_NEGATION_MODALITY` | `4031` | 1 | `03bcff91673a52ba...` | `PASS` |
| `D_NEGATION_MODALITY` | `103983` | 3 | `0401c1d1b2c2e970...` | `PASS` |
| `D_NEGATION_MODALITY` | `140693` | 1 | `043f73fc2caf6419...` | `PASS` |
| `C_NUMERIC` | `34351` | 2 | `00e5cbf63c58c267...` | `PASS` |
| `C_NUMERIC` | `31883` | 1 | `03773ca0b205ddf0...` | `PASS` |
| `C_NUMERIC` | `40489` | 1 | `0660a3cf6117f804...` | `PASS` |
| `C_NUMERIC` | `155139` | 1 | `09fbe8a9cada128c...` | `PASS` |
| `B_MULTI_CLAIM_CLEAN` | `116877` | 3 | `00c726d5ced2a42c...` | `PASS` |
| `B_MULTI_CLAIM_CLEAN` | `15181` | 3 | `00d8453fe03dad72...` | `PASS` |
| `B_MULTI_CLAIM_CLEAN` | `5967` | 3 | `03060247ca025fd0...` | `PASS` |
| `B_MULTI_CLAIM_CLEAN` | `139413` | 3 | `0375feb861d1b3b4...` | `PASS` |
| `A_SINGLE_CLAIM_CLEAN` | `75171` | 1 | `00006bf26e15dd26...` | `PASS` |
| `A_SINGLE_CLAIM_CLEAN` | `150131` | 1 | `000b97c303c42041...` | `PASS` |
| `A_SINGLE_CLAIM_CLEAN` | `30405` | 1 | `01962a615eadc572...` | `PASS` |
| `A_SINGLE_CLAIM_CLEAN` | `36801` | 1 | `03c6989e359db4b8...` | `PASS` |

#### RESERVE Candidates (8 IDs)

| Stratum | Question ID | Claim Count | Selection Key Prefix |
| :--- | :--- | :--- | :--- |
| `D_NEGATION_MODALITY` | `79625` | 1 | `063267e3207870f7...` |
| `D_NEGATION_MODALITY` | `70663` | 3 | `08fd19a44a611fd3...` |
| `C_NUMERIC` | `88089` | 3 | `0dac44ce4e35c0c9...` |
| `C_NUMERIC` | `12991` | 1 | `0ea810cba8b525bd...` |
| `B_MULTI_CLAIM_CLEAN` | `76335` | 3 | `0d95e047f1e793c6...` |
| `B_MULTI_CLAIM_CLEAN` | `154189` | 3 | `11f721bf5a3526cd...` |
| `A_SINGLE_CLAIM_CLEAN` | `166539` | 1 | `04127c06fa69d08d...` |
| `A_SINGLE_CLAIM_CLEAN` | `82009` | 1 | `050b01628decc1ca...` |

### 7.6 Review Package Artifact

- **External Review Package ZIP**: `verification-positive-control-review-packets-v1.zip`
- **SHA-256**: `cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4`
- **Size**: `110,095 bytes`
- **Contents**:
  - `execution/control_source_identity.json`
  - `results/control_selection_report.json`
  - `results/primary_control_ids.json`
  - `results/reserve_control_ids.json`
  - `positive_control_packets/<16 primary QIDs>.json` (human review status: `unreviewed`, null labels)
- **Mechanical Verdict**: `POSITIVE_CONTROL_SOURCE_READY`

---

## 8. Task B-FORENSIC-1C — Freeze Human-Approved Positive-Control Labels v1

### 8.1 Approval Provenance & Invariants

- **Approval Kind**: `explicit_user_human_approval`
- **Approval Date**: `2026-08-20`
- **Reviewer Identifier**: `human_reviewer_1`
- **Organizer Ground Truth**: `false`
- **Legal Expert Credential Asserted**: `false`
- **Scope Disclaimer**:
  > *"These labels are internal human forensic annotations over frozen, train-derived development outputs and their exact supplied frozen evidence. They are not official UIT DSC ground truth, retrieval relevance labels, public/private test annotations, or training labels."*
- **Usage Policy**:
  - **Allowed Initial Uses**: `["verification_correctness_evaluation", "forensic_analysis"]`
  - **Prohibited Initial Uses**: `["training", "fine_tuning", "retrieval_relevance_supervision", "public_test_annotation", "private_test_annotation", "manual_submission_correction"]`

### 8.2 Positive-Control Label Overlay Counts

| Scope | Count |
| :--- | :--- |
| **Questions** | 16 |
| **Historical Arms** | 16 |
| **Total Labeled Claims** | 27 |
| **`SUPPORTED`** | 16 |
| **`CONTRADICTED`** | 2 |
| **`INSUFFICIENT`** | 9 |

> [!NOTE]
> **Claim-Level Entailment Boundary**: These annotations evaluate strict claim-to-evidence entailment against cited frozen evidence only. They do not encode overall answer completeness or question responsiveness.

### 8.3 Canonical External Artifacts

- **Canonical JSON Overlay**: `verification-positive-control-human-labels-v1.json`
  - **SHA-256**: `60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373`
  - **Size**: `27,642 bytes`
- **Packaged Transport ZIP**: `verification-positive-control-human-labels-v1.zip`
  - **SHA-256**: `bf8efc191c74786af76b1580247d02989254fa2ee65063bfaccaea0035a4ecf2`
  - **Size**: `6,427 bytes`
  - **Members**: `verification-positive-control-human-labels-v1.json`, `control_human_label_identity.json`
- **Mechanical Verdict**: `POSITIVE_CONTROL_HUMAN_LABELS_FROZEN`

### 8.4 Combined Benchmark Specification (38 Claims)

With both suspicious failure cases (B-FORENSIC-1A) and positive-control candidates (B-FORENSIC-1C) frozen into content-bound overlays, the composite forensic evaluation benchmark is established:

| Benchmark Slice | Source Artifact SHA-256 | Labeled Claims | `SUPPORTED` | `CONTRADICTED` | `INSUFFICIENT` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Suspicious Forensic** | `bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733` | 11 | 2 | 5 | 4 |
| **Positive Controls** | `60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373` | 27 | 16 | 2 | 9 |
| **Total Composite Benchmark** | — | **38** | **18** | **7** | **13** |

> [!WARNING]
> **Prevalence Disclaimer**: This combined benchmark is an intentionally stratified evaluation set containing deliberate failure cases and stratified controls. The distribution ($18/7/13$) does **not** reflect production system error prevalence.

---

## 9. Controlled V0 Rule-Based vs V1 Semantic-Verifier Benchmark Protocol & Kaggle Runbook

### 9.1 Evaluation Objective & Anti-Overfitting Invariant

The objective of this offline benchmark is to evaluate the pre-existing `ModelBackedCitationVerifier` (V1) against the deterministic `RuleBasedCitationVerifier` (V0) on the frozen composite 38-claim human-annotated dataset.

> [!IMPORTANT]
> **Strict Anti-Overfitting Rule**:
> - `src/legal_agentic_rag/generation/semantic_verifier.py` is **NOT modified**.
> - Prompt template, schema, system instruction, and label definitions are strictly frozen as originally implemented prior to human labeling.
> - Zero few-shot examples or prompt tuning are permitted.
> - This first benchmark evaluates the verifier implementation as it existed. If future prompt/model tuning is motivated by these results, the 38 claims become **development data**, requiring a fresh independently human-reviewed holdout before any production promotion.
> - **Production Semantic Verification Remains Disabled**: `semantic_verifier_promotion_authorized = false`.

### 9.2 Benchmark Arms & Configuration

| Arm | Implementation | Configuration / Model Identity |
| :--- | :--- | :--- |
| **V0 (Baseline)** | `RuleBasedCitationVerifier` | `ClaimVerificationConfig(enabled=True, require_inline_citations=True, minimum_lexical_support=0.25, minimum_claim_tokens=2, require_numeric_match=True, require_negation_match=True, max_claims=20)` |
| **V1 (Candidate)** | `ModelBackedCitationVerifier` | Base: V0 `RuleBasedCitationVerifier`<br>Provider: `TransformersChatProvider`<br>Model Name: `Qwen/Qwen2.5-3B-Instruct`<br>Revision: `a1d308dfcc03e09da285d49d912439a655a571e8`<br>Device: `cuda`, Dtype: `float16`<br>Temperature: `0.0`, Retries: `1`<br>Timeout: `180s`, Tokens: `max_in=8192, max_out=512` |

### 9.3 Benchmark Dataset Composition (38 Claims)

- **Slice A (Suspicious Forensic)**:
  - Packets: `verification-forensic-review-packets.zip` (SHA-256 `996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a`)
  - Labels: `verification-human-forensic-labels-v1.json` (SHA-256 `bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733`)
  - Labeled Claims: 11 (2 `SUPPORTED`, 5 `CONTRADICTED`, 4 `INSUFFICIENT`)
- **Slice B (Positive Controls)**:
  - Packets: `verification-positive-control-review-packets-v1.zip` (SHA-256 `cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4`)
  - Labels: `verification-positive-control-human-labels-v1.json` (SHA-256 `60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373`)
  - Labeled Claims: 27 (16 `SUPPORTED`, 2 `CONTRADICTED`, 9 `INSUFFICIENT`)
- **Composite Benchmark**: 38 Claims (18 `SUPPORTED`, 7 `CONTRADICTED`, 13 `INSUFFICIENT`)

### 9.4 Two-Pass Stability & Provenance Gates

1. **Source Provenance Gate**: All 4 external benchmark files verified by exact SHA-256 before model initialization. Claim texts bound 1-to-1 via `claim_text_sha256`.
2. **V0 Replay Gate**: 100% (22/22) historical arms replayed against V0 rule verifier. Must match historical verification flags exactly before V1 model initialization.
3. **Deterministic Stability Gate (`repeat_count=2`)**: Pass 1 computes primary accuracy metrics. Pass 2 tests deterministic stability at `temperature=0.0`. If any claim prediction diverges between Pass 1 and Pass 2, verdict is `SEMANTIC_VERIFIER_LABEL_INSTABILITY`.

### 9.5 Metrics Specification

- **Claim-Level Binary Metrics**: TP, FP, TN, FN, Accuracy, Precision, Recall (Supported-Retention), Specificity (Negative-Catch Rate), F1, Balanced Accuracy.
- **V1 Three-Way Metrics**: 3x3 Confusion Matrix (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`), Per-Class Precision/Recall/F1, Macro-Averaged F1.
- **Paired V0 vs V1 Deltas**: `BOTH_CORRECT`, `V0_ONLY_CORRECT`, `V1_ONLY_CORRECT`, `BOTH_WRONG`, `v1_fixed_v0_error_count`, `v1_regressed_from_v0_correct_count`, `net_correctness_delta`.
- **False Accept / Reject Profile**: Explicit counts and rates across human positive and negative claims.
- **Error-Tag Catch Diagnostics**: V1 negative catch rate across granular forensic tags (`SCOPE_OVERGENERALIZED`, `CONDITION_INVERTED`, `ACTOR_ROLE_INVERTED`, `WRONG_DOCUMENT`, `WRONG_ARTICLE`, `QUANTITY_ERROR`, `OTHER`).
- **Slice & Strata Breakdown**: Separate evaluations for Slice A, Slice B, and positive-control strata (`A_SINGLE_CLAIM_CLEAN`, `B_MULTI_CLAIM_CLEAN`, `C_NUMERIC`, `D_NEGATION_MODALITY`).
- **Answer-Level Metrics**: Answer validity determined by all-claims-supported rule; V0 vs V1 accuracy against human ground truth.

---

### 9.6 V0 Deterministic Baseline Replay & Metrics

During local preflight execution (`--skip-model-run`), the harness replayed `RuleBasedCitationVerifier` over all 22 historical benchmark arms with **100% exact fidelity** against frozen historical records (`v0_replay_arm_passes = 22/22`).

Evaluating V0 against the 38 frozen human claim labels yields the exact pre-semantic baseline:

| Metric Category | Metric | V0 Rule-Based Value |
| :--- | :--- | :--- |
| **Claim-Level Binary (38 claims)** | Total Evaluated Claims | `38` (`18` human supported, `20` human negative) |
| | True Positives (TP) | `18` |
| | False Positives (FP) | `20` (all negative claims accepted lexically) |
| | True Negatives (TN) | `0` |
| | False Negatives (FN) | `0` |
| | **Supported Retention (Recall)** | **`100.0%`** ($18/18$) |
| | **Negative Catch Rate (Specificity)** | **`0.0%`** ($0/20$) |
| | **Binary Accuracy** | **`47.37%`** ($18/38$) |
| | Precision | `47.37%` ($18/38$) |
| | F1 Score | `0.6429` |
| | Balanced Accuracy | `50.0%` |
| **Answer-Level Binary (22 arms)** | Total Evaluated Arms | `22` (`7` human valid, `15` human invalid) |
| | True Positives (TP) | `7` |
| | False Positives (FP) | `15` |
| | True Negatives (TN) | `0` |
| | False Negatives (FN) | `0` |
| | **Supported Answer Retention** | **`100.0%`** ($7/7$) |
| | **Invalid Answer Catch Rate** | **`0.0%`** ($0/15$) |
| | **Answer Binary Accuracy** | **`31.82%`** ($7/22$) |

---

### 9.7 Kaggle Execution Runbook (Copy-Paste Cells)

> [!CAUTION]
> **Execution Gate**: DO NOT RUN on Kaggle until the committed benchmark harness and test suite have been externally reviewed. The placeholder commit SHA below MUST be replaced with the exact reviewed commit SHA prior to execution.

#### Cell 1: Environment Setup, Dependency Pinning & CUDA Gate
```bash
%%bash
set -euo pipefail

# 1. Target Commit Verification
REVIEWED_COMMIT_SHA="PLACEHOLDER_REVIEWED_COMMIT_SHA"

if [ "$REVIEWED_COMMIT_SHA" = "PLACEHOLDER_REVIEWED_COMMIT_SHA" ]; then
    echo "ERROR: DO NOT RUN UNTIL EXTERNAL REVIEW REPLACES THIS VALUE WITH THE REVIEWED COMMIT SHA."
    exit 1
fi

cd /kaggle/working
if [ ! -d "legal-agentic-rag" ]; then
    git clone https://github.com/Chef221/legal-agentic-rag.git
fi
cd legal-agentic-rag
git fetch origin --prune
git checkout "$REVIEWED_COMMIT_SHA"

echo "=== Repository Authority Verified ==="
git rev-parse HEAD

# 2. Install editable package and pin dependencies
pip install -q -e .
python -m pip install -q transformers==4.47.1 accelerate==1.2.1

# 3. Environment & Hardware Verification
python -c "
import torch, transformers, legal_agentic_rag
print('=== Dependency & Hardware Gate ===')
print('legal_agentic_rag version:', legal_agentic_rag.__version__)
assert legal_agentic_rag.__version__ == '0.50.7', 'Package version mismatch'
print('transformers version:', transformers.__version__)
assert transformers.__version__ == '4.47.1', 'Transformers version mismatch'
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA is required for benchmark execution'
print('CUDA device name:', torch.cuda.get_device_name(0))
print('CUDA device count:', torch.cuda.device_count())
"
```

#### Cell 2: Verify Benchmark Sources by Exact SHA-256 & Persist Map
```python
import hashlib
import json
from pathlib import Path

SOURCE_CHECKSUMS = {
    "verification-forensic-review-packets.zip": "996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a",
    "verification-human-forensic-labels-v1.json": "bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733",
    "verification-positive-control-review-packets-v1.zip": "cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4",
    "verification-positive-control-human-labels-v1.json": "60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373",
}

found_sources = {}
input_root = Path("/kaggle/input")

for filename, expected_sha in SOURCE_CHECKSUMS.items():
    matches = []
    for p in sorted(input_root.rglob(filename)):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest == expected_sha:
            matches.append(p)
    if not matches:
        raise FileNotFoundError(f"Source file {filename} matching SHA {expected_sha} not found under /kaggle/input")
    if len(matches) > 1:
        print(f"Notice: multiple SHA matches for {filename}; selecting deterministic first: {matches[0]}")
    found_sources[filename] = str(matches[0])
    print(f"Verified: {filename} -> {matches[0]} (SHA: {expected_sha[:16]}...)")

map_path = Path("/kaggle/working/verifier_benchmark_sources.json")
map_path.write_text(json.dumps(found_sources, indent=2), encoding="utf-8")
print(f"\n=== All 4 Sources Verified & Saved to {map_path} ===")
```

#### Cell 3: Execute Benchmark Harness via Verified Sources
```bash
%%bash
set -euo pipefail

cd /kaggle/working/legal-agentic-rag

# Read verified source paths from JSON
python -c "
import json, subprocess
from pathlib import Path

sources = json.loads(Path('/kaggle/working/verifier_benchmark_sources.json').read_text(encoding='utf-8'))

cmd = [
    'python', 'scripts/evaluate_verification_semantic_benchmark.py',
    '--forensic-review-packets', sources['verification-forensic-review-packets.zip'],
    '--forensic-labels', sources['verification-human-forensic-labels-v1.json'],
    '--control-review-packets', sources['verification-positive-control-review-packets-v1.zip'],
    '--control-labels', sources['verification-positive-control-human-labels-v1.json'],
    '--output-dir', '/kaggle/working/semantic_benchmark_output',
    '--package-zip', '/kaggle/working/verification-semantic-benchmark-evidence.zip',
    '--repeat-count', '2',
]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
"
```

#### Cell 4: Verify Output Package & Checksum
```python
import hashlib
from pathlib import Path

zip_path = Path("/kaggle/working/verification-semantic-benchmark-evidence.zip")
if not zip_path.is_file():
    raise FileNotFoundError(f"Output evidence package not found at {zip_path}")

size_bytes = zip_path.stat().st_size
sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

print(f"Benchmark Evidence Package: {zip_path.name}")
print(f"Size (bytes): {size_bytes}")
print(f"SHA-256: {sha}")
```

---

### 9.8 Implementation Status

- **Status**: `VERIFIER BENCHMARK HARNESS IMPLEMENTED & HARDENED — REAL MODEL EXECUTION PENDING REVIEW`
- **Preflight Mechanical Verdict**: `VERIFIER_BENCHMARK_READY`
- **V0 Exact Replay Status**: `22/22 arms passed with 100% fidelity`
- **Production Status**: Semantic verification remains disabled; `semantic_verifier_promotion_authorized = false`.
