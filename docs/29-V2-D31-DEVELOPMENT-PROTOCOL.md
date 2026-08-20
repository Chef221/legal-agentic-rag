# V2-D3.1 Development Protocol & Kaggle Runbook

---

## 1. Executive Summary & Governance Authority

This document defines the formal development, execution, and selection protocol for **Candidate V2-D3.1** (`StructuredSemanticCitationVerifierD31`).

- **Forensic Authority:** [`docs/28-V2-D3-DEVELOPMENT-FORENSIC-AUDIT.md`](file:///c:/legal-agentic-rag/docs/28-V2-D3-DEVELOPMENT-FORENSIC-AUDIT.md)
- **Current Canonical Best Development Candidate:** **V2-D3**
  - Canonical D3 Evidence: `verification-v2-d3-development-evidence.zip` (SHA-256 `0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff`)
  - Canonical D3 Performance: Binary $28/38$ ($73.68\%$), Three-Way $24/38$ ($63.16\%$), Supported Retained $17/18$ ($94.44\%$), Negative Caught $11/20$ ($55.00\%$), Answer Accuracy $14/22$ ($63.64\%$), Contradictions Caught $0/7$.
- **Candidate V2-D3.1 Architecture:** **Option A (Hierarchical Single-Call Two-Gate Semantic Verifier)**
- **Candidate ID:** `V2-D3.1`
- **Current Status:** **`V2-D3.1 IMPLEMENTED — REAL DEVELOPMENT EXECUTION PENDING EXTERNAL REVIEW`**
- **Fresh Holdout Status:** **`SEALED_UNREVIEWED`** (Strictly unopened, uninspected, uncalculated).

---

## 2. Architecture: Hierarchical Single-Call Two-Gate Verifier (Option A)

V2-D3.1 preserves the per-claim invocation architecture (1 provider call per claim; 38 calls per pass) while restructuring the model's single semantic call into **two explicit, ordered boolean evaluation gates**:

```
                               Claim + Cited Statutory Evidence
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Qwen2.5-3B Model Invocation  │
                             │ (Single-Call Structured Output) │
                             └────────────────┬────────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
           [Gate 1: is_contradicted]                 [Gate 2: is_fully_established]
           Positively incompatible rule,             100% of material propositions,
           inverted condition, authority             actors, ranks, and conditions
           conflict, or conflicting values?          explicitly established?
                         │                                         │
                         └────────────────────┬────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Deterministic State Machine  │
                             │         (Trusted Code)          │
                             └────────────────┬────────────────┘
                                              │
             ┌────────────────────────────────┼────────────────────────────────┐
             ▼                                ▼                                ▼
       State A: (T, F)                  State B: (F, T)                  State C: (F, F)
        CONTRADICTED                       SUPPORTED                       INSUFFICIENT

                      [Invalid State: (T, T) -> Logically Inconsistent -> Retry]
```

### 2.1 Output Schema & State Machine

```python
class StructuredSemanticVerificationDraftD31(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str
    is_contradicted: bool
    is_fully_established: bool
```

**Deterministic Label Mapping (`derive_claim_semantic_label_d31`):**
1. **State A (`is_contradicted = True, is_fully_established = False`):** $\to$ `SemanticSupportLabel.CONTRADICTED`
2. **State B (`is_contradicted = False, is_fully_established = True`):** $\to$ `SemanticSupportLabel.SUPPORTED`
3. **State C (`is_contradicted = False, is_fully_established = False`):** $\to$ `SemanticSupportLabel.INSUFFICIENT`
4. **Invalid State (`is_contradicted = True, is_fully_established = True`):** Logically inconsistent. The verifier rejects the draft under category `LOGICALLY_INCONSISTENT_STATE` and issues a content-safe retry. If repeated, it produces an isolated semantic execution error for that claim.

---

## 3. Semantic Definitions & Guardrails

### 3.1 Gate 1: Material Contradiction Semantics
- `is_contradicted = true` **ONLY** when cited evidence positively establishes a proposition materially incompatible with or opposite to the claim.
- **Material Incompatibility Includes:**
  - Incompatible exclusive deciding authority or legally responsible actor (e.g. claim asserts Unit Head replaces Deputy Head, but statute vests exclusive replacement authority in State Auditor General).
  - Inverted statutory condition, prerequisite, or applicability trigger (e.g. claim asserts electricity rate applies to $\ge 12$ months, but statute restricts it to $< 12$ months).
  - Incompatible exception, prohibition, or conflicting legal modality (permission vs mandatory prohibition).
  - Incompatible numerical quantity, duration, or threshold.
- **Non-Contradiction Invariant:** Absence of evidence, uninformative text, silent statutes, or documents from unrelated topics are **NOT** contradictions (`is_contradicted = false`).

### 3.2 Gate 2: Full Statutory Establishment Semantics
- `is_fully_established = true` **ONLY** when cited evidence establishes 100% of all material propositions required by the claim within the question's legal scope.
- Every material component must be satisfied: actor, duty/right, statutory conditions/prerequisites, exceptions, quantities/durations, and specific legal category or rank.
- Shared vocabulary, topical overlap, or matching one sub-proposition is **NOT** sufficient (`is_fully_established = false`).
- A rule governing one professional rank (e.g. Hạng III $\to$ II) does not establish a claim about a different rank (Hạng II $\to$ I) merely because the surrounding wording is similar.

### 3.3 Internal Cross-Reference Guardrail
- A statutory provision may fully establish a claim by explicitly incorporating or requiring compliance with another provision through an internal cross-reference (e.g. "phải đáp ứng điều kiện tại Điều 11 Thông tư này").
- When a claim merely asserts that compliance with that referenced provision is required, the cited text satisfies entailment (`is_fully_established = true`).

---

## 4. Pinned Model & Environment Identity

Candidate V2-D3.1 uses the exact same pinned model identity as V1 and V2-D3:

| Parameter | Value | Constraint |
| :--- | :--- | :--- |
| **Model Name** | `Qwen/Qwen2.5-3B-Instruct` | Exact Hugging Face ID |
| **Model Revision** | `a1d308dfcc03e09da285d49d912439a655a571e8` | Immutable 40-char commit hash |
| **Provider Backend** | `transformers` | Pinned backend |
| **Provider Version** | `4.47.1` | Pinned package |
| **Device** | `cuda` | Mandatory GPU execution |
| **Torch Dtype** | `float16` | Float16 precision |
| **Temperature** | `0.0` | Greedy deterministic decoding |
| **Max Input Tokens** | `8192` | Context ceiling |
| **Max Output Tokens** | `512` | JSON response ceiling |
| **Max Retries** | `1` | Max structured correction attempt |
| **Evaluation Passes** | `2` | Two-pass stability evaluation |

---

## 5. Primary Comparator & Pre-Registered Selection Gate

### 5.1 Primary Comparator: V2-D3
The baseline comparison authority for V2-D3.1 is **V2-D3** (loaded from `verification-v2-d3-development-evidence.zip`):
- Binary Correct: $28 / 38$ ($73.68\%$)
- Supported Retained: $17 / 18$ ($94.44\%$)
- Negative Caught: $11 / 20$ ($55.00\%$)
- Three-Way Correct: $24 / 38$ ($63.16\%$)
- Contradictions Caught: $0 / 7$ ($0.00\%$)
- Answer Correct: $14 / 22$ ($63.64\%$)

### 5.2 Pre-Registered D3.1 Selection Gate
D3.1 will supersede D3 (`D31_SUPERSEDES_D3`) **ONLY** if all mechanical and quality gates pass:

1. **Mechanical Gates (Must All Pass):**
   - `verdict == "V2_D31_DEVELOPMENT_BENCHMARK_PASS"`
   - `model_errors == 0`
   - `provider_invocation_errors == 0`
   - `execution_error_in_any_pass == 0`
   - `unstable_semantic_claims == 0` (100% two-pass stability across all 38 claims)
   - `binary execution_errors == 0`
   - `three-way execution_errors == 0`
   - `38 / 38` successfully evaluated claims
2. **Quality Gates vs D3 (Must All Pass via Exact Integer Counts):**
   - `d31_binary_correct > 28 / 38` (at least $29/38$)
   - `d31_supported_retained >= 17 / 18`
   - `d31_negative_caught > 11 / 20` (at least $12/20$)
   - `paired_net_delta_vs_d3 > 0`
   - `d31_three_way_correct > 24 / 38` (at least $25/38$)
   - `d31_correctly_predicted_contradicted > 0` (at least 1 contradiction caught)
   - `d31_answer_correct >= 14 / 22`

If any gate fails $\to$ Decision is **`KEEP_D3`**.
In all cases, `promotion_authorized = false` (fail-closed development policy).

---

## 6. Scientific Development Diagnostics

The D3.1 benchmark harness automatically tracks and reports:
1. **Contradiction Capability Diagnostic:** Gold CONTRADICTED count (7), correctly predicted CONTRADICTED count, and recall.
2. **7 D3 Gains Preservation Diagnostic:** Outcome tracking on the 7 D3 fixes (`26541:BASE:C1`, `95861:BASE:C3`, `95861:CANDIDATE:C2`, `95861:CANDIDATE:C3`, `103983:PRIMARY:C2`, `108497:PRIMARY:C1`, `155139:PRIMARY:C1`).
3. **Four Forensic Error Groups Diagnostic:** Resolution tracking across Group A (3 false entailments of contradiction), Group B (4 contradiction undercalls), Group C (6 false entailments of insufficient), and Group D (1 supported overrejection).
4. **Dimension & State Diagnostics:** Distribution of `is_contradicted`, `is_fully_established`, and the 3 semantic states in Pass 1.

---

## 7. Kaggle Offline Development Runbook (For Later Execution)

> [!IMPORTANT]
> **DO NOT RUN THIS RUNBOOK YET.**
> Real execution is blocked pending external review of this document and approved commit authority.

### Cell 1 — Verify Environment & GPU
```bash
!nvidia-smi
!python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

### Cell 2 — Install Pinned Package
```bash
!pip install -q --no-deps \
  transformers==4.47.1 \
  accelerate==1.2.1 \
  pydantic==2.13.4 \
  pydantic-core==2.13.4 \
  typing-extensions==4.12.2
```

### Cell 3 — Clone Repository & Verify Reviewed Commit
```bash
!git clone https://github.com/Chef221/legal-agentic-rag.git /kaggle/working/legal-agentic-rag
%cd /kaggle/working/legal-agentic-rag
!git checkout PLACEHOLDER_REVIEWED_COMMIT_SHA
!git status --short
!pip install -q -e . --no-deps
```

### Cell 4 — Discover Canonical Benchmark Sources
```python
import os, glob
from pathlib import Path
from hashlib import sha256

EXPECTED_HASHES = {
    "forensic_review_packets.zip": "996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a",
    "forensic_labels.json": "bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733",
    "control_review_packets.zip": "cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4",
    "control_labels.json": "60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373",
    "v1_evidence.zip": "bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6",
    "d3_evidence.zip": "0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff",
}

def find_by_hash(expected_sha):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            p = Path(root) / f
            try:
                h = sha256(p.read_bytes()).hexdigest()
                if h == expected_sha:
                    return p
            except Exception:
                pass
    raise FileNotFoundError(f"Missing source with SHA-256: {expected_sha}")

sources = {k: find_by_hash(sha) for k, sha in EXPECTED_HASHES.items()}
for k, p in sources.items():
    print(f"Found {k}: {p}")
```

### Cell 5 — Execute Canonical V2-D3.1 Development Benchmark
```bash
!python scripts/evaluate_verification_v2_d31_development.py \
  --forensic-packets "{sources['forensic_review_packets.zip']}" \
  --forensic-labels "{sources['forensic_labels.json']}" \
  --control-packets "{sources['control_review_packets.zip']}" \
  --control-labels "{sources['control_labels.json']}" \
  --v1-evidence "{sources['v1_evidence.zip']}" \
  --d3-evidence "{sources['d3_evidence.zip']}" \
  --output-dir "/kaggle/working/v2_d31_development_output" \
  --package-zip "/kaggle/working/verification-v2-d31-development-evidence.zip" \
  --device cuda
```

### Cell 6 — Display Decision Report
```python
import json
from pathlib import Path

dec_path = Path("/kaggle/working/v2_d31_development_output/results/v2_d31_development_decision_report.json")
if dec_path.is_file():
    print(json.dumps(json.loads(dec_path.read_text()), indent=2))
```

---

## 8. Explicit Governance Invariants

- **V2-D3 REMAINS CURRENT BEST DEVELOPMENT CANDIDATE**
- **V2-D3.1 IMPLEMENTED — NOT YET EXECUTED**
- **D3.1 NOT YET SELECTED OVER D3**
- **FRESH HOLDOUT REMAINS SEALED & UNREVIEWED**
- **NO HOLDOUT DATA ACCESSED OR PROCESSED**
- **NO PRODUCTION WIRING MODIFIED (SEMANTIC VERIFIER DISABLED IN PROD)**
- **ZERO CHANGES TO RETRIEVAL / RERANKING / GENERATION CORE**
