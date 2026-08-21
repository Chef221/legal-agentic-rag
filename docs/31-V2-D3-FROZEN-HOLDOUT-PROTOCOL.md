# docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md

## 1. Executive Summary & V2-D3 Frozen Selection

### 1.1 Formal Closure of Development Iterations
The development phase of the V2 Semantic Citation Verifier is officially and permanently **CLOSED**.
Across the 38-claim development benchmark:
- **V2-D3** established high calibration, retaining **17/18 (94.44%)** supported claims and fixing 7 historical V0/V1 errors with 0 regressions.
- **V2-D3.1** (monolithic 2-gate) suffered severe overcalling, regressing supported retention to **12/18 (66.67%)** and losing 6/7 D3 fixes $\to$ closed as `KEEP_D3`.
- **V2-D3.2** (asymmetric strict-conflict overlay) produced 0 false overrides, preserving 100% of D3 gains but failing to catch contradictions $\to$ closed as `KEEP_D3`.

There is **NO D3.3**, no further prompt tuning, no threshold tuning, no new overlays, and no case-specific development rules.

### 1.2 Formal Selection & Freeze of V2-D3
Candidate **V2-D3** is officially selected and frozen as the sole V2 candidate to proceed to the Fresh Holdout evaluation.

```
══════════════════════════════════════════════════════════════════════════
Selected Candidate:         V2-D3
Model Identity:             Qwen/Qwen2.5-3B-Instruct
Model Immutable Revision:   a1d308dfcc03e09da285d49d912439a655a571e8
Runtime Backend:            transformers (4.47.1)
Hardware Configuration:     CUDA, float16, temperature=0.0
Max Input / Output Tokens:  8192 / 512
Timeout / Retries:          180.0s / 1 max structured retry
Development Evidence:       verification-v2-d3-development-evidence.zip
SHA-256:                    0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff
Canonical Execution Commit: 9eecac4c2f75c1caf3c2f39b7fb0e87aed315503
══════════════════════════════════════════════════════════════════════════
```

---

## 2. Frozen V2-D3 Architecture & Code Signatures

### 2.1 Code & Prompt Cryptographic Signatures
To guarantee that the exact evaluated model behavior is verified against the development champion, the holdout harness strictly validates the following source identities:

| Component | Target Identity | Required SHA-256 Checksum |
| :--- | :--- | :--- |
| **System Instruction** | `STRUCTURED_SEMANTIC_D3_SYSTEM_INSTRUCTION` | `546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b` |
| **Implementation File** | `structured_semantic_verifier_d3.py` | `a6e8bca15ad14d869e103e1f94fe94bb9a81f9ddc8bc650b280b69b7d57e9826` |
| **JSON Schema** | `D3StructuredClaimAssessmentDraft` | `37cf5da1fa15c3298ec2fb11d4d03e911295ea912eb89b4f9fbcf5ba878a87ea` |

### 2.2 Inference & Combination Architecture
For each cited claim:
1. **Single-Claim Prompting:** Formats isolated statutory evidence and claim text with strictly zero human labels, zero reference answers, and zero prior run predictions.
2. **Structured Output Draft:** Returns `claim_id`, `relation` (`ENTAILS`, `CONTRADICTS`, `DOES_NOT_ESTABLISH`), and 5 diagnostic flags (`actor_mismatch`, `condition_exception_mismatch`, `quantity_temporal_mismatch`, `negation_modality_mismatch`, `source_scope_mismatch`).
3. **Trusted Python Mapping:**
   - `relation == ENTAILS` AND all 5 flags `False` $\to$ `SUPPORTED` (Binary: `ACCEPT`).
   - `relation == CONTRADICTS` $\to$ `CONTRADICTED` (Binary: `REJECT`).
   - Any flag `True` OR `relation == DOES_NOT_ESTABLISH` $\to$ `INSUFFICIENT` (Binary: `REJECT`).

---

## 3. Fresh Holdout Benchmark Specification

### 3.1 Pre-Registered Holdout Signatures (Outer Only)
The fresh holdout evaluation dataset was generated, reviewed, and sealed under Milestone 27.

| Artifact File | Canonical SHA-256 Checksum | Size (Bytes) | Status |
| :--- | :--- | :--- | :--- |
| `verification-v2-holdout-selection-v1.json` | `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b` | 16,788 | `sealed_unreviewed` |
| `verification-v2-holdout-review-packets-v1.zip` | `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4` | 108,532 | `sealed_unreviewed` |

### 3.2 Strict Holdout Blindness Invariants
To preserve the scientific validity of the fresh holdout:
1. **Zero Inspection:** No coding agent or developer may open, uncompress, list internal file members, inspect question IDs, inspect question texts, inspect claims, inspect evidence texts, or inspect gold labels before canonical execution.
2. **One-Shot Decision:** The holdout evaluation is a one-time, final decision. No post-hoc tuning or rerun iterations are permitted.
3. **Model-Free Preflight Gate:** A strict preflight mode (`--preflight-only`) must verify file checksums, runtime dependencies, and schema compatibility before loading weights or running inference.

---

## 4. Pre-Registered Promotion Gates & Decision Protocol

### 4.1 Evaluation Architecture (Two-Pass Protocol)
The holdout evaluation executes in two consecutive passes:
- **Pass 1 (Authoritative):** Evaluates all holdout claims and computes all primary metrics.
- **Pass 2 (Stability Confirmation):** Evaluates all holdout claims a second time to verify deterministic zero-temperature stability.

### 4.2 Pre-Registered Rate Gates
To recommend promoting V2-D3 into production, the authoritative Pass 1 metrics must satisfy all of the following rate thresholds:

| Metric Gate | Pre-Registered Threshold | Primary Rationale |
| :--- | :---: | :--- |
| **Supported Claim Retention Rate** | $\ge \mathbf{88.00\%}$ | Prevents false rejections of valid grounded claims; preserves generation utility. |
| **Negative Claim Catch Rate** | $\ge \mathbf{50.00\%}$ | Ensures meaningful filtering of hallucinated, contradicted, or unsupported claims. |
| **Valid Answer Retention Rate** | $\ge \mathbf{80.00\%}$ | Protects end-to-end answer preservation for fully grounded answers. |
| **Full Answer-Level Accuracy** | $\ge \mathbf{60.00\%}$ | Guarantees net positive answer verification correctness across all answer types. |
| **Binary Claim Accuracy** | $\ge \mathbf{70.00\%}$ | Ensures solid overall binary discrimination on unseen holdout claims. |

### 4.3 Pre-Registered Mechanical Gates
In addition to rate thresholds, execution must satisfy:
1. **Zero Provider Invocation Errors:** `provider_invocation_errors == 0`.
2. **Zero Execution Errors:** `execution_error_in_any_pass_count == 0`.
3. **Zero Instability:** `unstable_semantic_claim_count == 0` (100% agreement between Pass 1 and Pass 2).
4. **Complete Output Validation:** Exactly 2 valid semantic labels per claim across both passes.
5. **Frozen D3 Source Verification:** Implementation and prompt SHA-256 must match frozen D3 hashes.

### 4.4 Decision Logic & Verdicts
```
                             Holdout Execution Complete
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Check Mechanical & Stability Gates  │
                      └───────────────────┬───────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │ Pass                              │ Fail
                        ▼                                   ▼
        ┌───────────────────────────────┐        ┌───────────────────────────────┐
        │   Check Rate Quality Gates    │        │  V2_D3_HOLDOUT_EXECUTION_FAIL │
        └───────────────┬───────────────┘        │   (REJECT_V2_D3_PROMOTION)    │
                        │                        └───────────────────────────────┘
            ┌───────────┴───────────┐
            │ Pass                  │ Fail
            ▼                       ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ V2_D3_HOLDOUT_PROMOTION_RECOMM│ │ V2_D3_HOLDOUT_PROMOTION_REJ   │
│ (PROMOTE_V2_D3_TO_PRODUCTION) │ │   (REJECT_V2_D3_PROMOTION)    │
└───────────────────────────────┘ └───────────────────────────────┘
```

### 4.5 Promotion vs Authorization Boundary (Fail-Closed Security)
- Harness outputs `promotion_recommended: true` if and only if all gates pass.
- Harness output `promotion_authorized: false` is an **unconditional invariant**.
- Production semantic verifier enablement (`enabled = true` in config/factory) requires human governance review and authorization after inspecting the canonical holdout evidence package.

---

## 5. Kaggle Holdout Execution Runbook (Cells H1–H6)

### Cell H1: Environment Setup & Git Clone
```python
# ==============================================================================
# CELL H1: ENVIRONMENT SETUP & CODEBASE CLONE
# ==============================================================================
import os
import subprocess
from pathlib import Path

# Execution Authority Commit
REVIEWED_COMMIT_SHA = "PLACEHOLDER_REVIEWED_COMMIT_SHA"

REPO_URL = "https://github.com/Chef221/legal-agentic-rag.git"
REPO_DIR = Path("/kaggle/working/legal-agentic-rag")

if not REPO_DIR.exists():
    print(f"Cloning repository from {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
else:
    print(f"Repository directory already exists at {REPO_DIR}. Fetching latest...")
    subprocess.run(["git", "fetch", "origin"], cwd=str(REPO_DIR), check=True)

# Checkout specific reviewed commit
subprocess.run(["git", "checkout", REVIEWED_COMMIT_SHA], cwd=str(REPO_DIR), check=True)

current_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), text=True
).strip()
print(f"Repository verified at reviewed commit: {current_commit}")
assert current_commit == REVIEWED_COMMIT_SHA, f"Commit mismatch: {current_commit} != {REVIEWED_COMMIT_SHA}"

# Install project in editable mode
subprocess.run(["pip", "install", "-e", "."], cwd=str(REPO_DIR), check=True)
print("Environment and editable package installation complete.")
```

### Cell H2: Dataset Discovery & Outer Checksum Verification
```python
# ==============================================================================
# CELL H2: HOLDOUT DATASET DISCOVERY & CHECKSUM VERIFICATION
# ==============================================================================
from hashlib import sha256
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# Known input dataset locations on Kaggle
INPUT_DIR = Path("/kaggle/input")

HOLDOUT_PACKETS_CANDIDATES = list(INPUT_DIR.glob("**/verification-v2-holdout-review-packets-v1.zip"))
HOLDOUT_SELECTION_CANDIDATES = list(INPUT_DIR.glob("**/verification-v2-holdout-selection-v1.json"))
HOLDOUT_LABELS_CANDIDATES = list(INPUT_DIR.glob("**/verification-v2-holdout-reviewed-labels-v1.json"))

assert HOLDOUT_PACKETS_CANDIDATES, "verification-v2-holdout-review-packets-v1.zip not found in /kaggle/input"
assert HOLDOUT_SELECTION_CANDIDATES, "verification-v2-holdout-selection-v1.json not found in /kaggle/input"
assert HOLDOUT_LABELS_CANDIDATES, "verification-v2-holdout-reviewed-labels-v1.json not found in /kaggle/input"

HOLDOUT_PACKETS_PATH = HOLDOUT_PACKETS_CANDIDATES[0]
HOLDOUT_SELECTION_PATH = HOLDOUT_SELECTION_CANDIDATES[0]
HOLDOUT_LABELS_PATH = HOLDOUT_LABELS_CANDIDATES[0]

EXPECTED_PACKETS_SHA = "a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4"
EXPECTED_SELECTION_SHA = "08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b"

actual_packets_sha = sha256_file(HOLDOUT_PACKETS_PATH)
actual_selection_sha = sha256_file(HOLDOUT_SELECTION_PATH)
actual_labels_sha = sha256_file(HOLDOUT_LABELS_PATH)

print(f"Holdout Packets:   {HOLDOUT_PACKETS_PATH} (SHA: {actual_packets_sha})")
print(f"Holdout Selection: {HOLDOUT_SELECTION_PATH} (SHA: {actual_selection_sha})")
print(f"Holdout Labels:    {HOLDOUT_LABELS_PATH} (SHA: {actual_labels_sha})")

assert actual_packets_sha == EXPECTED_PACKETS_SHA, f"Packets SHA mismatch: {actual_packets_sha}"
assert actual_selection_sha == EXPECTED_SELECTION_SHA, f"Selection SHA mismatch: {actual_selection_sha}"

print("All pre-registered holdout input checksums successfully verified.")
```

### Cell H3: Model-Free Preflight Gate Verification
```python
# ==============================================================================
# CELL H3: MODEL-FREE PREFLIGHT GATE VERIFICATION
# ==============================================================================
import subprocess
from pathlib import Path

REPO_DIR = Path("/kaggle/working/legal-agentic-rag")
OUTPUT_DIR = Path("/kaggle/working/holdout_eval_output")

cmd_preflight = [
    "python", "scripts/evaluate_verification_v2_d3_holdout.py",
    "--holdout-packets", str(HOLDOUT_PACKETS_PATH),
    "--holdout-labels", str(HOLDOUT_LABELS_PATH),
    "--holdout-selection", str(HOLDOUT_SELECTION_PATH),
    "--output-dir", str(OUTPUT_DIR / "preflight"),
    "--preflight-only",
]

print("Executing model-free preflight check...")
subprocess.run(cmd_preflight, cwd=str(REPO_DIR), check=True)
print("Preflight verification PASSED. Ready for model inference.")
```

### Cell H4: Model Weight Download & Device Check
```python
# ==============================================================================
# CELL H4: MODEL WEIGHT DOWNLOAD & DEVICE VERIFICATION
# ==============================================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "a1d308dfcc03e09da285d49d912439a655a571e8"

assert torch.cuda.is_available(), "CUDA device is required for canonical V2-D3 holdout execution"
device_name = torch.cuda.get_device_name(0)
print(f"CUDA Available: {device_name} (Device Count: {torch.cuda.device_count()})")

print(f"Pre-caching model weights for {MODEL_NAME} (revision: {MODEL_REVISION})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("Model weights successfully loaded and verified.")
```

### Cell H5: Live Execution & Canonical Evidence Packaging
```python
# ==============================================================================
# CELL H5: LIVE HOLDOUT EXECUTION & EVIDENCE PACKAGING
# ==============================================================================
import subprocess
from pathlib import Path

REPO_DIR = Path("/kaggle/working/legal-agentic-rag")
OUTPUT_DIR = Path("/kaggle/working/holdout_eval_output/final")
PACKAGE_ZIP = Path("/kaggle/working/verification-v2-d3-holdout-evidence.zip")

cmd_eval = [
    "python", "scripts/evaluate_verification_v2_d3_holdout.py",
    "--holdout-packets", str(HOLDOUT_PACKETS_PATH),
    "--holdout-labels", str(HOLDOUT_LABELS_PATH),
    "--holdout-selection", str(HOLDOUT_SELECTION_PATH),
    "--output-dir", str(OUTPUT_DIR),
    "--package-zip", str(PACKAGE_ZIP),
    "--device", "cuda",
    "--torch-dtype", "float16",
    "--temperature", "0.0",
]

print("Launching canonical V2-D3 Holdout Evaluation...")
subprocess.run(cmd_eval, cwd=str(REPO_DIR), check=True)
print(f"Holdout evaluation finished. Evidence package generated at {PACKAGE_ZIP}")
```

### Cell H6: Evidence Archive Integrity & Forensic Reconciliation
```python
# ==============================================================================
# CELL H6: EVIDENCE ARCHIVE INTEGRITY & FORENSIC RECONCILIATION
# ==============================================================================
from hashlib import sha256
import json
from pathlib import Path
import zipfile

PACKAGE_ZIP = Path("/kaggle/working/verification-v2-d3-holdout-evidence.zip")
assert PACKAGE_ZIP.is_file(), f"Evidence ZIP missing: {PACKAGE_ZIP}"

zip_bytes = PACKAGE_ZIP.read_bytes()
zip_sha = sha256(zip_bytes).hexdigest()
zip_size = len(zip_bytes)

print("="*75)
print("             V2-D3 FRESH HOLDOUT CANONICAL EVIDENCE SUMMARY")
print("="*75)
print(f"Archive Path:      {PACKAGE_ZIP}")
print(f"Archive SHA-256:   {zip_sha}")
print(f"Archive Size:      {zip_size:,} bytes")

REQUIRED_MEMBERS = {
    "execution/v2_d3_holdout_source_identity.json",
    "results/v2_d3_holdout_report.json",
    "results/v2_d3_holdout_decision_report.json",
    "results/v2_d3_holdout_stability_report.json",
    "results/v0_claim_predictions.jsonl",
    "results/v2_d3_holdout_claim_predictions_pass1.jsonl",
    "results/v2_d3_holdout_claim_predictions_pass2.jsonl",
    "results/v2_d3_holdout_claim_comparisons.jsonl",
    "telemetry/provider_calls.jsonl",
}

with zipfile.ZipFile(PACKAGE_ZIP, "r") as zf:
    members = set(zf.namelist())
    for req in REQUIRED_MEMBERS:
        assert req in members, f"Required member '{req}' missing from evidence ZIP"
    
    report = json.loads(zf.read("results/v2_d3_holdout_report.json").decode("utf-8"))
    decision = json.loads(zf.read("results/v2_d3_holdout_decision_report.json").decode("utf-8"))
    stability = json.loads(zf.read("results/v2_d3_holdout_stability_report.json").decode("utf-8"))
    exec_id = json.loads(zf.read("execution/v2_d3_holdout_source_identity.json").decode("utf-8"))
    calls_raw = zf.read("telemetry/provider_calls.jsonl").decode("utf-8").strip().splitlines()
    provider_calls = [json.loads(line) for line in calls_raw]

print(f"Member Count:      {len(members)} (All required members verified)")
print(f"Provider Calls:    {len(provider_calls)} reconciled across 2 passes")
print(f"Stability:         {stability['stability_summary']['stable_semantic_claim_count']}/{stability['stability_summary']['total_claims']} stable claims")
print(f"Verdict:           {report['verdict']}")
print(f"Decision:          {decision['holdout_evaluation_decision']}")
print(f"Promotion Recom:   {decision['promotion_recommended']}")
print(f"Promotion Auth:    {decision['promotion_authorized']} (Fail-Closed)")
print("="*75)
print("Metrics Summary:")
for k, v in decision.get("metrics_summary", {}).items():
    print(f"  - {k}: {v}")
print("="*75)
print("\n*** DOWNLOAD EVIDENCE ARCHIVE BEFORE TERMINATING KAGGLE SESSION ***\n")
```

---

## 6. Explicit Governance Invariants

- **V2-D3.1 FORMALLY CLOSED AS `KEEP_D3`**
- **V2-D3.2 FORMALLY CLOSED AS `KEEP_D3`**
- **V2-D3 IS FORMALLY FROZEN AS THE EXCLUSIVE V2 CANDIDATE**
- **DEVELOPMENT BENCHMARK IS PERMANENTLY CLOSED FOR CANDIDATE TUNING**
- **ZERO HUMAN INSPECTION OF HOLDOUT TEXTS OR LABELS BEFORE EXECUTION**
- **PRE-REGISTERED RATE GATES AND MECHANICAL GATES ARE IMMUTABLE**
- **`promotion_authorized` IS UNCONDITIONALLY FALSE IN HARNESS OUTPUTS**
- **NO PRODUCTION WIRING MODIFIED (SEMANTIC VERIFIER REMAINS DISABLED IN PROD)**
- **ZERO CHANGES TO RETRIEVAL / RERANKING / GENERATION PIPELINE CORE**
