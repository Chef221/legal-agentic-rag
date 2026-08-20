# 27 — Priority B: Verification-Correctness Forensic Audit (B-FORENSIC-0)

## 1. Executive Summary & Audit State

- **Frontier**: Priority B — Verification-Correctness Audit
- **Milestone Task**: B-FORENSIC-0 (Four-Question Paired Forensic Source Materialization)
- **Status**: `FORENSIC_SOURCE_READY` — Materializer Implemented & Real Source Execution Verified (Human Review Pending)
- **Production Code Changes**: NONE (`src/` untouched)
- **Model / Verifier Status**:
  - `ModelBackedCitationVerifier` remains DISABLED (`semantic_verification.enabled = false`).
  - Production `RuleBasedCitationVerifier` behavior is UNCHANGED.
  - S20 remains the active production configuration; H40 remains UNPROMOTED.
- **Auto-Labeling Policy**: Strictly ZERO auto-generated semantic/legal correctness labels. Human review status is `unreviewed`.

---

## 2. Frozen Historical Evidence Provenance

The forensic materializer operates strictly on frozen historical data without executing any live retrieval, reranking, or generative models.

### 2.1 External B1A Evidence Source
- **Archive Path**: `C:\Users\Nguyen\Downloads\phase-b1a-graph-routing-ablation-evidence.zip`
- **Observed Archive Size**: `42,993 bytes`
- **Observed Archive SHA-256**:
  `b88ccce928b4cecfc9239d490c59f91405834c5a3199f917d757c5735b1d6631`
- **Archive Inventory Validation**:
  - `configs/phase-b1a-graph-routing-cases.json`
  - `evidence/materialized_questions_identity.json`
  - `configs/base_runtime_config.json`
  - `configs/candidate_runtime_config.json`
  - `results/phase_b1a_paired_report.json`
  - `results/phase_b1a_decision_report.json`
  - `base_batch/manifest.json`
  - `candidate_batch/manifest.json`
  - `base_batch/results.jsonl`
  - `candidate_batch/results.jsonl`
  - `base_batch/batch_state.json`
  - `candidate_batch/batch_state.json`

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
  - Path: `artifacts/uit-dsc-2026-task2-m44-dev-split/development.json`
  - Record Count: 991
  - SHA-256: `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8` (Exact match)
- **Canonical Serving Artifact Root**:
  - Root: `artifacts/uit-dsc-2026-task2-v0400/legal_chunks`
  - `artifact_type`: `legal_chunks`
  - `dataset_name`: `uit-dsc-2026-task2-selected-contexts`
  - `dataset_revision`: `sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`
  - `record_count`: 330,768

---

## 3. Four Target Cases & Reconstructed Outcomes

The initial audit investigates 4 target questions across both historical arms (total 8 historical execution arms):

| Question ID | BASE Stop Reason | BASE Verified | CANDIDATE Stop Reason | CANDIDATE Verified | Selected Chunks | Lookup Pass | Replay Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **102047** | `answer_verified` | Yes (`is_valid=True`) | `answer_verified` | Yes (`is_valid=True`) | 5 / 5 | 100% Pass | Matched 100% |
| **147239** | `generation_failed` | No (Replay N/A) | `answer_verified` | Yes (`is_valid=True`) | 5 / 5 | 100% Pass | Matched 100% |
| **26541** | `answer_verified` | Yes (`is_valid=True`) | `generation_failed` | No (Replay N/A) | 5 / 5 | 100% Pass | Matched 100% |
| **95861** | `answer_verified` | Yes (`is_valid=True`) | `answer_verified` | Yes (`is_valid=True`) | 5 / 5 | 100% Pass | Matched 100% |

### 3.1 Verifier Replay Results
- **Applicable Arms**: 6 (where historical `citation_verification` was reached and `stop_reason == answer_verified`)
- **Non-Applicable Arms**: 2 (`147239` BASE and `26541` CANDIDATE, preserved with `reason: "historical_verifier_not_reached"`)
- **Replay Pass Rate**: 6 / 6 (100% bit-for-bit fidelity with frozen citation verification records including claim IDs, claim texts, token overlap, numeric matches, and negation checks).

---

## 4. Paired Forensic Packet Schema

Each target question is emitted as a unified paired JSON packet structured for human review:

```json
{
  "schema_version": "1.0",
  "question_id": "102047",
  "source_identity": {
    "source_kind": "b1a_frozen_historical_pair",
    "archive_path": "...",
    "archive_sha256_observed": "b88ccce928b4cecfc9239d490c59f91405834c5a3199f917d757c5735b1d6631",
    "base_results_sha256": "c72dc38f37945831095c71ba4b0f24f328de2e01dc4f7ecac6e527b997dc3cac",
    "candidate_results_sha256": "420d2297d529c3d8c246de499111840d4a4a98b010bd95e42639b7ba9f3fb6ad",
    "code_version": "0.50.6",
    "materialized_question_source_sha256": "f5d681c447a2bb964de90298207af0363c76b3546bfa027603d7fa98322a3ce3",
    "canonical_development_sha256": "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8",
    "serving_artifact_identity": {
      "artifact_type": "legal_chunks",
      "dataset_name": "uit-dsc-2026-task2-selected-contexts",
      "dataset_revision": "sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e",
      "code_version": "0.40.0",
      "record_count": 330768
    }
  },
  "question": "...",
  "reference_answer_context": {
    "text": "...",
    "ground_truth_status": "human_review_context_only_not_claim_entailment_ground_truth"
  },
  "arms": {
    "BASE": {
      "historical_response": { ... },
      "agent_outcome": { ... },
      "selected_evidence": [ ... ],
      "context_selection_trace": [ ... ],
      "historical_verification": { ... },
      "rule_verifier_replay": {
        "replay_applicable": true,
        "replay_matches_historical": true,
        "replay_result": { ... }
      }
    },
    "CANDIDATE": {
      "historical_response": { ... },
      "agent_outcome": { ... },
      "selected_evidence": [ ... ],
      "context_selection_trace": [ ... ],
      "historical_verification": { ... },
      "rule_verifier_replay": {
        "replay_applicable": true,
        "replay_matches_historical": true,
        "replay_result": { ... }
      }
    }
  },
  "human_forensic_review": {
    "review_status": "unreviewed",
    "base_claim_labels": null,
    "candidate_claim_labels": null,
    "cross_arm_notes": null,
    "root_cause_classification": null
  }
}
```

---

## 5. Human Review Contract (For Subsequent Phases)

When human forensic annotation begins, claims will be labeled using:

1. **Claim Entailment Labels**:
   - `SUPPORTED`: The claim is directly and accurately entailed by the cited evidence.
   - `CONTRADICTED`: The claim asserts a legal statement that contradicts the cited evidence.
   - `INSUFFICIENT`: The cited evidence does not contain sufficient facts to establish the claim.
   - `REVIEW_REQUIRES_EXTERNAL_LEGAL_KNOWLEDGE`: Cannot be resolved from provided evidence chunks alone.

2. **Granular Error Tags**:
   - `CONDITION_OMITTED`
   - `CONDITION_INVERTED`
   - `EXCEPTION_IGNORED`
   - `ACTOR_ROLE_INVERTED`
   - `NEGATION_INVERTED`
   - `QUANTITY_ERROR`
   - `SCOPE_OVERGENERALIZED`
   - `WRONG_DOCUMENT`
   - `WRONG_ARTICLE`
   - `TEMPORAL_APPLICABILITY_ERROR`
   - `OTHER` / `NONE`

---

## 6. Execution Invariants & Verification Checklist

- [x] Zero retrieval reruns
- [x] Zero generation reruns
- [x] Zero model-backed semantic verifier calls
- [x] Zero auto-generated correctness labels
- [x] Paired BASE and CANDIDATE arms preserved
- [x] 100% deterministic chunk reconstruction from canonical `uit-dsc-2026-task2-v0400/legal_chunks`
- [x] 100% verifier replay fidelity across all 6 applicable historical arms
- [x] 24/24 unit tests passing
- [x] Full pytest test suite passing
- [x] Materialized packet files written outside tracked repository content
