# 27 — Priority B: Verification-Correctness Forensic Audit (B-FORENSIC-0)

## 1. Executive Summary & Audit State

- **Frontier**: Priority B — Verification-Correctness Audit
- **Milestone Task**: B-FORENSIC-0 (Four-Question Paired Forensic Source Materialization)
- **Status**: `FORENSIC_SOURCE_READY` — Hardened Materializer Implemented & Real Source Execution Verified (Human Review Pending)
- **Production Code Changes**: NONE (`src/` untouched)
- **Model / Verifier Status**:
  - `ModelBackedCitationVerifier` remains DISABLED (`semantic_verification.enabled = false`).
  - Production `RuleBasedCitationVerifier` behavior is UNCHANGED.
  - S20 remains the active production configuration; H40 remains UNPROMOTED.
- **Auto-Labeling Policy**: Strictly ZERO auto-generated semantic/legal correctness labels. Human review status is `unreviewed`.
- **Review Package**:
  - Archive: `verification-forensic-review-packets.zip`
  - SHA-256: `996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a`
  - Size: `42,826 bytes`

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

## 4. Paired Forensic Packet Schema

Each target question is emitted as a unified paired JSON packet structured for human review:

```json
{
  "schema_version": "1.0",
  "question_id": "102047",
  "source_identity": {
    "source_kind": "canonical_zip",
    "archive_filename": "phase-b1a-graph-routing-ablation-evidence.zip",
    "archive_sha256_observed": "b88ccce928b4cecfc9239d490c59f91405834c5a3199f917d757c5735b1d6631",
    "canonical_zip_sha256_expected": "b88ccce928b4cecfc9239d490c59f91405834c5a3199f917d757c5735b1d6631",
    "base_results_sha256": "c72dc38f37945831095c71ba4b0f24f328de2e01dc4f7ecac6e527b997dc3cac",
    "candidate_results_sha256": "420d2297d529c3d8c246de499111840d4a4a98b010bd95e42639b7ba9f3fb6ad",
    "code_version": "0.50.6",
    "materialized_question_source_sha256": "f5d681c447a2bb964de90298207af0363c76b3546bfa027603d7fa98322a3ce3",
    "canonical_development_sha256": "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8",
    "development_filename": "development.json",
    "serving_artifact_identity": {
      "artifact_type": "legal_chunks",
      "dataset_name": "uit-dsc-2026-task2-selected-contexts",
      "dataset_revision": "sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e",
      "code_version": "0.40.0",
      "record_count": 330768,
      "payload_integrity_verified": true,
      "payload_sha256": "3a769121f07aa1c65b69569ce296b416f40048ba47b9761a393c245ece609872"
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
- [x] In-protocol payload integrity verification of serving artifacts (`records.jsonl` SHA verified)
- [x] 100% source mapping cross-check between `selected_evidence` and `selection_trace` (across both verified and generation_failed arms)
- [x] 100% verifier replay fidelity across all 6 applicable historical arms
- [x] 35/35 unit tests passing
- [x] Full pytest test suite passing
- [x] Materialized packet files and review ZIP written outside tracked repository content
