# Generation Grounding G1 — Material-Fidelity Development Specification

## 1. Context & Motivation

Following the closure of the V2 semantic verifier track (Decision **D127**), the repository shifts primary engineering resources to **Generation Grounding & Prompt Optimization** (highest ROI for Task 2 METEOR and ROUGE-L metrics).

The postmortem forensic analysis in [`docs/32-V2-D3-HOLDOUT-CLOSURE-AND-POSTMORTEM.md`](file:///c:/legal-agentic-rag/docs/32-V2-D3-HOLDOUT-CLOSURE-AND-POSTMORTEM.md) established that the primary driver of hallucinated legal propositions is not syntax errors or arithmetic failure, but **material proposition drift**:
1. **Actor / Role Mismatch**: Substituting legal subjects (e.g. advocate's rights attributed to litigant in `125893:C1/C3`; reserve NCO discharge powers conflated with commissioned officers in `90897:C1`).
2. **Condition / Exception Omission**: Generalizing a strictly conditional statutory rule into an unconditional statement (e.g. dropping non-state capital prerequisite in `45427:C1`).
3. **Action / Object Mismatch**: Substituting regulated activities and fine brackets (e.g. feed testing vs production in `95695:C1`).

Candidate **G1** addresses this root cause at generation time by instructing the model to preserve all material dimensions of the legal proposition before emitting any claim.

---

## 2. G1 Hypothesis & Architecture Contract

### 2.1 The Material-Fidelity Hypothesis
A generated legal claim should only be emitted when every material dimension is supported by its cited evidence:
- **Actor / Role**: Exact legal subject and legal capacity.
- **Action / Object**: Exact regulated activity and legal object.
- **Conditions / Exceptions**: Preservation of all prerequisites, conditions, and exceptions (no unconditioned broadening).
- **Legal Scope**: Exact entity type, jurisdictional level, and scope boundaries.
- **Numeric / Temporal Exactness**: Strict verbatim copying of figures, percentages, dates, and brackets.
- **Full Coverage**: Every material component of `claims[].text` must be established by cited evidence.
- **List / Noun-Phrase Preservation**: Direct faithful noun phrases from evidence are valid for list/category questions without requiring artificial predicate clauses.

### 2.2 Strict Non-Proliferation Invariants
- **No Extra Model Calls**: Normal generation requires exactly **1 provider call** (same as baseline). No second-pass reflection, critic, or verifier calls.
- **No Schema Change**: Output contract remains strictly `ModelAnswerDraft` (`claims: list[ModelClaimDraft]`, `insufficient_evidence: bool`, `warnings: list[str]`). No extra fields.
- **Single Experimental Variable**: The ONLY changed variable is `grounding_profile` (`"baseline"` vs `"material_fidelity_v1"`).
- **Default Behavior Unchanged**: Production configuration default remains `grounding_profile="baseline"`. G1 is an evaluation candidate and is not promoted in this task.

---

## 3. Burned Diagnostic Dataset Governance

The 16 primary review packets (31 claims) from `verification-v2-holdout-review-packets-v1.zip` are classified as **burned diagnostic development data**:
- **Diagnostic Role**: Used solely for offline development A/B analysis and prompt calibration.
- **Strict Prohibition**: This data is **BURNED** and can **NEVER** be used as authoritative promotion evidence.
- **Unbiased Prompting**: Model inputs contain ONLY the `original_question` and `retrieved_evidence` text. No human labels, historical D3 predictions, error classifications, or diagnostic tags are passed to the model.

---

## 4. Pre-Registered G1 Development Success Criteria

Before executing inference on Kaggle GPU, the following development criteria are frozen:

| Criterion | Target Metric / Gate | Authoritative Evaluation Source |
| :--- | :--- | :--- |
| **Criterion A: Operational Integrity** | **0 generation execution errors** across all 16 diagnostic questions for G1 | Automated Telemetry |
| **Criterion B: Material Error Elimination** | Eliminates known material error on $\ge 4 / 5$ ($80.0\%$) historical error mechanisms (`125893:C1`, `125893:C3`, `45427:C1`, `90897:C1`, `95695:C1`) without introducing new errors | Blinded Pairwise Human Review |
| **Criterion C: Valid Answer Preservation** | Preserves a materially-supported, useful answer on $\ge 9 / 10$ ($90.0\%$) gold-valid diagnostic cases | Blinded Pairwise Human Review |
| **Criterion D: Bounded Abstention** | G1 abstention rate must not exceed baseline abstention rate by more than $15$ percentage points ($\Delta_{\text{abstain}} \le +0.15$) | Automated Evaluation Report |
| **Criterion E: Output Schema Invariance** | $100\%$ valid `ModelAnswerDraft` structures with $0$ schema regressions | Automated Evaluation Report |
| **Criterion F: Call Count Parity** | Normal successful generation uses exactly $1$ provider call per question for both baseline and G1 | Automated Telemetry |

*Note: G1 must not achieve Criterion B merely by globally abstaining (enforced by Criterion C and Criterion D).*

---

## 5. Evaluation Harness & Blinded Pairwise Protocol

The evaluation is managed by [`scripts/evaluate_generation_grounding_g1.py`](file:///c:/legal-agentic-rag/scripts/evaluate_generation_grounding_g1.py):
1. **Side-by-Side Execution**: Runs Baseline (`grounding_profile="baseline"`) and G1 (`grounding_profile="material_fidelity_v1"`) over the 16 diagnostic questions using identical model weights, context, and generation parameters.
2. **Deterministic Blinding**: Randomly swaps Option 1 vs Option 2 per question using a deterministic hash seed (`sha256(f"g1_blind_salt_v1:{qid}")`).
3. **Outputs Generated**:
   - `results/generation_g1_ab_report.json`: Aggregate metrics, abstention rates, criteria statuses.
   - `results/generation_g1_ab_predictions.jsonl`: Complete structured results for both arms.
   - `results/generation_g1_human_review_worksheet.md`: Blinded pairwise review worksheet for human evaluation.
   - `results/generation_g1_blinding_key.json`: Secret blinding key mapping.
   - `telemetry/provider_calls.jsonl`: Content-safe operational telemetry.
   - `execution/generation_g1_source_identity.json`: Exact provenance metadata.

---

## 6. One-Run Kaggle Execution Runbook

Following code review approval, exactly ONE canonical Kaggle GPU development run is authorized:

```bash
# 1. Verify preflight integrity
python scripts/evaluate_generation_grounding_g1.py \
  --diagnostic-packets /path/to/verification-v2-holdout-review-packets-v1.zip \
  --output-dir generation_g1_ab_output \
  --preflight-only

# 2. Execute canonical development A/B run
python scripts/evaluate_generation_grounding_g1.py \
  --diagnostic-packets /path/to/verification-v2-holdout-review-packets-v1.zip \
  --output-dir generation_g1_ab_output \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --model-revision a1d308dfcc03e09da285d49d912439a655a571e8 \
  --device cuda \
  --torch-dtype float16
```
