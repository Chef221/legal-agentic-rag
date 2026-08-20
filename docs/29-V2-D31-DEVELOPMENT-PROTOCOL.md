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

### Cell 1 — Exact Clone & Reviewed Commit Checkout
```bash
!rm -rf /kaggle/working/legal-agentic-rag
!git clone https://github.com/Chef221/legal-agentic-rag.git /kaggle/working/legal-agentic-rag
%cd /kaggle/working/legal-agentic-rag
!git checkout PLACEHOLDER_REVIEWED_COMMIT_SHA

import subprocess

head_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], text=True).strip()

REVIEWED_COMMIT_SHA = "PLACEHOLDER_REVIEWED_COMMIT_SHA"
assert head_commit == REVIEWED_COMMIT_SHA, f"Commit mismatch: {head_commit} != {REVIEWED_COMMIT_SHA}"
assert len(status) == 0, f"Git worktree not clean:\n{status}"
print(f"Verified clean checkout of reviewed commit: {REVIEWED_COMMIT_SHA}")
```

### Cell 2 — Pinned Model Dependencies & Environment Gate
```bash
!pip install -q --no-deps transformers==4.47.1 accelerate==1.2.1
!pip install -q -e . --no-deps

import importlib.metadata, sys, torch, transformers, accelerate, pydantic
import legal_agentic_rag

source_ver = getattr(legal_agentic_rag, "__version__", "unknown")
installed_ver = importlib.metadata.version("legal-agentic-rag")
tf_ver = getattr(transformers, "__version__", "unknown")
cuda_avail = torch.cuda.is_available()

print("--- Runtime Environment ---")
print(f"Package Source Version: {source_ver}")
print(f"Installed Distribution Version: {installed_ver}")
print(f"Transformers Version: {tf_ver}")
print(f"CUDA Available: {cuda_avail}")
print(f"CUDA Device: {torch.cuda.get_device_name(0) if cuda_avail else 'NONE'}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Accelerate Version: {accelerate.__version__}")
print(f"Pydantic Version: {pydantic.__version__}")

assert source_ver == "0.50.7", f"Source package version must be 0.50.7, got {source_ver}"
assert installed_ver == "0.50.7", f"Installed distribution version must be 0.50.7, got {installed_ver}"
assert tf_ver == "4.47.1", f"Transformers version must be 4.47.1, got {tf_ver}"
assert cuda_avail is True, "CUDA device required for canonical execution"
print("Environment gate passed.")
```

### Cell 3 — Holdout / Phase-A Blindness Guard & Exact Six Development Sources Discovery
```python
import os, base64
from pathlib import Path
from hashlib import sha256

# 1. HOLDOUT / PHASE-A BLINDNESS GUARD (Fail-Closed)
FORBIDDEN_PATTERNS = [
    "verification-v2-holdout",
    "verification_v2_holdout",
    "verification-v2-holdout-selection-v1.json",
    "verification-v2-holdout-review-packets-v1.zip",
    "verification_v2_holdout_output",
    "phase-a-current-system-census",
    "phase-a-current-system-census-final-evidence.zip",
]

for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        full_path = os.path.join(root, f)
        for pat in FORBIDDEN_PATTERNS:
            if pat in full_path or pat in f:
                raise RuntimeError(
                    f"BLINDNESS VIOLATION: Forbidden fresh holdout or Phase-A source detected: {full_path}. "
                    "Halting execution immediately without opening or inspecting file."
                )

# 2. DISCOVER EXACT SIX DEVELOPMENT SOURCES
CANONICAL_SOURCES = {
    "verification-forensic-review-packets.zip": "996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a",
    "verification-human-forensic-labels-v1.json": "bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733",
    "verification-positive-control-review-packets-v1.zip": "cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4",
    "verification-positive-control-human-labels-v1.json": "60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373",
    "verification-semantic-benchmark-evidence.zip": "bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6",
    "verification-v2-d3-development-evidence.zip": "0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff",
}

def sha256_bytes(b: bytes) -> str:
    return sha256(b).hexdigest()

discovered_sources = {}
recovery_dir = Path("/kaggle/working/recovered_sources")
recovery_dir.mkdir(parents=True, exist_ok=True)

# Scan /kaggle/input for direct matches or base64 encoded versions
for name, expected_sha in CANONICAL_SOURCES.items():
    matches = []
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            p = Path(root) / f
            # Check direct raw bytes
            try:
                raw = p.read_bytes()
                if sha256_bytes(raw) == expected_sha:
                    matches.append(p)
                    continue
            except Exception:
                pass
            # Check base64 encoded text
            if f.endswith(".b64") or f.endswith(".txt"):
                try:
                    decoded = base64.b64decode(p.read_text().strip())
                    if sha256_bytes(decoded) == expected_sha:
                        recovered_path = recovery_dir / name
                        recovered_path.write_bytes(decoded)
                        matches.append(recovered_path)
                except Exception:
                    pass

    if len(matches) == 0:
        raise FileNotFoundError(f"Missing required development source '{name}' (expected SHA: {expected_sha})")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous matches for development source '{name}': {matches}")
    discovered_sources[name] = str(matches[0])
    print(f"Discovered {name}: {matches[0]}")

import json
Path("/kaggle/working/v2_d31_development_sources.json").write_text(
    json.dumps(discovered_sources, indent=2), encoding="utf-8"
)
print("All 6 canonical development sources discovered and verified.")
```

### Cell 4 — Model-Free Kaggle Preflight Gate
```bash
!python scripts/evaluate_verification_v2_d31_development.py \
  --forensic-packets "{discovered_sources['verification-forensic-review-packets.zip']}" \
  --forensic-labels "{discovered_sources['verification-human-forensic-labels-v1.json']}" \
  --control-packets "{discovered_sources['verification-positive-control-review-packets-v1.zip']}" \
  --control-labels "{discovered_sources['verification-positive-control-human-labels-v1.json']}" \
  --v1-evidence "{discovered_sources['verification-semantic-benchmark-evidence.zip']}" \
  --d3-evidence "{discovered_sources['verification-v2-d3-development-evidence.zip']}" \
  --output-dir "/kaggle/working/v2_d31_preflight_output" \
  --preflight-only

import json
from pathlib import Path

pf_rep = json.loads(Path("/kaggle/working/v2_d31_preflight_output/results/v2_d31_development_report.json").read_text())
assert pf_rep["verdict"] == "V2_D31_DEVELOPMENT_BENCHMARK_READY", f"Unexpected verdict: {pf_rep['verdict']}"
assert pf_rep["total_claims"] == 38, f"Expected 38 claims, got {pf_rep['total_claims']}"
assert pf_rep["v0_replay_stats"]["v0_replay_arm_passes"] == 22, "V0 replay failed"
assert pf_rep["v0_replay_stats"]["v0_replay_100_percent_fidelity"] is True, "V0 replay fidelity not 100%"

d3_comp = pf_rep["d3_comparison_metrics"]
assert d3_comp["d3_claim_binary"]["tp"] == 17, f"D3 TP mismatch: {d3_comp['d3_claim_binary']['tp']}"
assert d3_comp["d3_claim_binary"]["tn"] == 11, f"D3 TN mismatch: {d3_comp['d3_claim_binary']['tn']}"
assert d3_comp["d3_claim_binary"]["tp"] + d3_comp["d3_claim_binary"]["tn"] == 28, "D3 binary correct mismatch"
assert d3_comp["d3_answer_level"]["valid_answers_retained"] + d3_comp["d3_answer_level"]["invalid_answers_caught"] == 14, "D3 answer correct mismatch"

print("Model-free preflight passed with 100% fidelity. Ready for real execution.")
```

### Cell 5 — Real V2-D3.1 Canonical Model Execution
```bash
!python scripts/evaluate_verification_v2_d31_development.py \
  --forensic-packets "{discovered_sources['verification-forensic-review-packets.zip']}" \
  --forensic-labels "{discovered_sources['verification-human-forensic-labels-v1.json']}" \
  --control-packets "{discovered_sources['verification-positive-control-review-packets-v1.zip']}" \
  --control-labels "{discovered_sources['verification-positive-control-human-labels-v1.json']}" \
  --v1-evidence "{discovered_sources['verification-semantic-benchmark-evidence.zip']}" \
  --d3-evidence "{discovered_sources['verification-v2-d3-development-evidence.zip']}" \
  --output-dir "/kaggle/working/v2_d31_development_output" \
  --package-zip "/kaggle/working/verification-v2-d31-development-evidence.zip" \
  --device cuda \
  --repeat-count 2
```

### Cell 6 — Canonical Evidence Verification, Telemetry Gates, & Scientific Decision
```python
import json, zipfile
from pathlib import Path
from hashlib import sha256

# 1. VERIFY EVIDENCE ARCHIVE INVENTORY
zip_path = Path("/kaggle/working/verification-v2-d31-development-evidence.zip")
assert zip_path.is_file(), "Evidence archive ZIP not found"

REQUIRED_MEMBERS = [
    "execution/v2_d31_development_source_identity.json",
    "results/v2_d31_development_report.json",
    "results/v2_d31_development_decision_report.json",
    "results/v2_d31_dimension_diagnostics.json",
    "results/v0_claim_predictions.jsonl",
    "results/v1_claim_predictions.jsonl",
    "results/v2_d3_claim_predictions.jsonl",
    "results/v2_d31_claim_predictions_pass1.jsonl",
    "results/v2_d31_claim_predictions_pass2.jsonl",
    "results/v2_d31_claim_comparisons.jsonl",
    "telemetry/provider_calls.jsonl",
]

with zipfile.ZipFile(zip_path, "r") as zf:
    archive_members = set(zf.namelist())
    for req in REQUIRED_MEMBERS:
        assert req in archive_members, f"Missing required archive member: {req}"

print(f"Verified {len(REQUIRED_MEMBERS)} canonical evidence archive members.")

# 2. LOAD REPORTS & RECONCILE TELEMETRY
out_dir = Path("/kaggle/working/v2_d31_development_output")
report = json.loads((out_dir / "results/v2_d31_development_report.json").read_text())
decision = json.loads((out_dir / "results/v2_d31_development_decision_report.json").read_text())

# Verify provider call telemetry
provider_calls_lines = (out_dir / "telemetry/provider_calls.jsonl").read_text().strip().splitlines()
provider_calls = [json.loads(line) for line in provider_calls_lines]
total_calls = report["telemetry"]["total_provider_calls"]
assert len(provider_calls) == total_calls, f"Provider call count mismatch: {len(provider_calls)} != {total_calls}"

expected_sys_sha = report["execution_identity"]["prompt_identity"]["system_instruction_sha256"]
for i, call in enumerate(provider_calls):
    assert call["system_instruction_sha256"] == expected_sys_sha, f"Call {i} system instruction SHA mismatch"

# 3. ASSERT EXECUTION IDENTITY GATES
exec_id = report["execution_identity"]
assert exec_id["candidate_id"] == "V2-D3.1"
assert exec_id["package_version"] == "0.50.7"
assert exec_id["installed_distribution_version"] == "0.50.7"
assert exec_id["provider"]["backend"] == "transformers"
assert exec_id["provider"]["provider_version"] == "4.47.1"
assert exec_id["provider"]["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
assert exec_id["provider"]["model_revision"] == "a1d308dfcc03e09da285d49d912439a655a571e8"
assert decision["promotion_authorized"] is False, "promotion_authorized must remain false"

# 4. HARD BENCHMARK PASS ASSERTIONS (IF PASS)
if report["verdict"] == "V2_D31_DEVELOPMENT_BENCHMARK_PASS":
    assert report["telemetry"]["model_errors"] == 0
    assert report["telemetry"]["provider_invocation_errors"] == 0
    assert report["stability"]["execution_error_in_any_pass_count"] == 0
    assert report["stability"]["unstable_semantic_claim_count"] == 0
    assert report["stability"]["claims_with_two_valid_semantic_labels"] == 38
    assert report["metrics"]["v2_d31_claim_binary"]["execution_errors"] == 0
    assert report["metrics"]["v2_d31_three_way"]["execution_errors"] == 0
    assert report["dimension_diagnostics"]["successfully_structured_claim_count"] == 38
    assert report["dimension_diagnostics"]["execution_error_claim_count"] == 0

# 5. SCIENTIFIC SUMMARY DISPLAY
d3_binary = report["metrics"]["v2_d3_claim_binary"]
d31_binary = report["metrics"]["v2_d31_claim_binary"]
d31_three_way = report["metrics"]["v2_d31_three_way"]
paired = report["metrics"]["paired_v2_d31_vs_v2_d3"]
contra_diag = report["metrics"]["contradiction_capability"]
gain_diag = report["gain_preservation_diagnostic"]
d3_ans = report["metrics"]["v2_d3_answer_metrics"]
d31_ans = report["metrics"]["v2_d31_answer_metrics"]

print("\n" + "="*70)
print("             V2-D3.1 SCIENTIFIC DEVELOPMENT OUTCOME")
print("="*70)
print(f"Verdict: {report['verdict']}")
print(f"Development Decision: {decision['development_evaluation_decision']}")
print(f"D3.1 Supersedes D3: {decision['d31_supersedes_d3']}")
print(f"Promotion Authorized: {decision['promotion_authorized']}")
print("-"*70)
print(f"Binary Claim Accuracy:    D3: {d3_binary['tp']+d3_binary['tn']}/38 ({d3_binary['accuracy']:.2%})  -->  D3.1: {d31_binary['tp']+d31_binary['tn']}/38 ({d31_binary['accuracy']:.2%})")
print(f"Supported Retained:       D3: {d3_binary['tp']}/18 ({d3_binary['supported_retention']:.2%})      -->  D3.1: {d31_binary['tp']}/18 ({d31_binary['supported_retention']:.2%})")
print(f"Negative Caught:          D3: {d3_binary['tn']}/20 ({d3_binary['negative_catch']:.2%})          -->  D3.1: {d31_binary['tn']}/20 ({d31_binary['negative_catch']:.2%})")
print(f"Three-Way Claim Accuracy: D3: {report['metrics']['v2_d3_three_way']['accuracy']:.2%}                  -->  D3.1: {d31_three_way['accuracy']:.2%}")
print(f"Contradictions Caught:    D3: 0/7 (0.00%)               -->  D3.1: {contra_diag['d31_correctly_predicted_contradicted_count']}/7 ({contra_diag['d31_contradicted_recall']:.2%})")
print(f"Answer-Level Accuracy:    D3: {d3_ans['valid_answers_retained']+d3_ans['invalid_answers_caught']}/22 ({d3_ans['answer_level_accuracy']:.2%})  -->  D3.1: {d31_ans['valid_answers_retained']+d31_ans['invalid_answers_caught']}/22 ({d31_ans['answer_level_accuracy']:.2%})")
print("-"*70)
print(f"Paired Metrics vs D3: Both Correct={paired['both_correct']}, D3 Only={paired['base_only_correct']}, D3.1 Only={paired['candidate_only_correct']}, Net Delta={paired['net_correctness_delta']}")
print(f"D3 Gains Preserved:   {gain_diag['preserved_gain_count']}/7 (Regressed: {gain_diag['regressed_gain_claim_ids']})")
print("-"*70)

# 6. ZIP SELF-IDENTITY
zip_sha = sha256(zip_path.read_bytes()).hexdigest()
zip_size = zip_path.stat().st_size
print("Canonical Evidence Archive:")
print(f"  Path:         {zip_path}")
print(f"  SHA-256:      {zip_sha}")
print(f"  Size (Bytes): {zip_size:,}")
print(f"  Member Count: {len(archive_members)}")
print("\n*** DOWNLOAD BEFORE ENDING SESSION ***\n" + "="*70)
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
