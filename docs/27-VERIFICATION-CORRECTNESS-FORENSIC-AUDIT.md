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

