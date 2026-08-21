# docs/31-V2-D3-FROZEN-HOLDOUT-PROTOCOL.md

## 1. Executive Summary & V2-D3 Frozen Selection

### 1.1 Formal Closure of Development Iterations
The development phase of the V2 Semantic Citation Verifier is officially and permanently **CLOSED**.
Across the 38-claim development benchmark:
- **V2-D3** established high calibration, retaining **17/18 (94.44%)** supported claims, achieving **6/7 (85.71%)** valid answer retention, and fixing 7 historical V0/V1 errors with 0 regressions.
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
| **JSON Schema** | `D3StructuredClaimAssessmentDraft` | `3591144a40b0519d5da9dd262e8edf8814531d798b69deea94fd81fae39f5f61` |

### 2.2 Inference & Combination Architecture
For each cited claim:
1. **Single-Claim Prompting:** Formats isolated statutory evidence and claim text with strictly zero human labels, zero reference answers, and zero prior run predictions.
2. **Structured Output Draft:** Returns `claim_id`, `relation` (`ENTAILS`, `CONTRADICTS`, `DOES_NOT_ESTABLISH`), and 5 diagnostic flags (`actor_mismatch`, `condition_exception_mismatch`, `quantity_temporal_mismatch`, `negation_modality_mismatch`, `source_scope_mismatch`).
3. **Trusted Python Mapping:**
   - `relation == ENTAILS` AND all 5 flags `False` $\to$ `SUPPORTED` (Binary: `ACCEPT`).
   - `relation == CONTRADICTS` $\to$ `CONTRADICTED` (Binary: `REJECT`).
   - Any flag `True` OR `relation == DOES_NOT_ESTABLISH` $\to$ `INSUFFICIENT` (Binary: `REJECT`).

---

## 3. Fresh Holdout Benchmark Lifecycle & Governance

The holdout evaluation follows a strict, two-phase irreversible lifecycle to guarantee scientific integrity.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE H-LABEL: HUMAN GOLD FREEZE                │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Candidate V2-D3 is FROZEN (model, prompt SHA, code SHA).            │
│ 2. Evaluation protocol and promotion rate gates are FROZEN.            │
│ 3. Human reviewers unseal ONLY the 16 primary review packets.          │
│ 4. Zero D3 model predictions exist or are visible during review.       │
│ 5. Reviewers assign gold labels (SUPPORTED, CONTRADICTED, INSUFFICIENT)│
│ 6. Labels are frozen with freeze_verification_v2_holdout_labels.py.    │
│ 7. Produces verification-v2-holdout-reviewed-labels-v1.json and        │
│    content-free label commitment verification-v2-d3-holdout-label-     │
│    commitment.json (SHA, size, counts, review status).                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE H-EXEC: CANONICAL EXECUTION               │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Initiated ONLY after label commitment SHA is externally reviewed.   │
│ 2. Single canonical execution of V2-D3 on Kaggle GPU.                  │
│ 3. Harness verifies exact SHA binding across packets, selection, and   │
│    labels against the frozen commitment.                               │
│ 4. Two-pass deterministic stability evaluation.                        │
│ 5. Evaluates non-vacuous coverage and rate gates fail-closed.          │
│ 6. Zero post-hoc model tuning, prompt edits, threshold edits, or       │
│    label modifications permitted.                                      │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Pre-Registered Holdout Signatures (Outer Only)
The fresh holdout evaluation dataset was generated, reviewed, and sealed under Milestone 27.

| Artifact File | Canonical SHA-256 Checksum | Size (Bytes) | Status |
| :--- | :--- | :--- | :--- |
| `verification-v2-holdout-selection-v1.json` | `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b` | 16,788 | Pre-Registered Selection |
| `verification-v2-holdout-review-packets-v1.zip` | `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4` | 108,532 | Sealed Review Packets |

### 3.2 Scientifically Correct Blindness Invariants
1. **Before Phase H-LABEL Authorization:** Zero holdout inspection. No files opened or listed.
2. **During Phase H-LABEL:** Human inspection was permitted solely for human reviewers to assign independent gold labels. Zero D3 model predictions existed or were visible.
3. **After Gold Labels are Frozen:** The label artifact SHA is committed immutably. Zero label edits are permitted.
4. **During & After Phase H-EXEC:** Zero candidate tuning, zero prompt edits, zero threshold edits, zero label edits, and zero reruns to improve metrics.

### 3.3 Completed Phase H-LABEL Frozen Identities & Approved Commitment
Phase H-LABEL human review and label freezing completed successfully.

| Artifact File | Canonical SHA-256 Checksum | Size (Bytes) | Status / Governance |
| :--- | :--- | :--- | :--- |
| `verification-v2-holdout-selection-v1.json` | `08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b` | 16,788 | Pre-Registered Selection Binding |
| `verification-v2-holdout-review-packets-v1.zip` | `a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4` | 108,532 | Sealed Review Packets |
| `verification-v2-holdout-reviewed-labels-v1.json` | `85d348dbb7da1567398836b96156a9d08fcfe181b676c5ecd593535ec8904215` | 9,383 | Frozen Human Gold Labels (31 claims: 24 S, 1 C, 6 I) |
| `verification-v2-d3-holdout-label-commitment.json` (Pending) | `c7755e37e394e80484f73c52ee6965c34c65917c38fa83b1dc453bbb466bcf86` | 823 | Historical Pending-Review Commitment |
| `configs/verification-v2-d3-holdout-label-commitment.json` (Approved) | `5cc7f58ed52c43d091bce98d5296ab68cded981495736a479644aefc1428b6dc` | 1,060 | Tracked Approved Commitment (`EXTERNALLY_REVIEWED_FOR_H_EXEC`) |

---

## 4. Pre-Registered Promotion Gates & Decision Protocol

### 4.1 Non-Vacuous Coverage Denominator Gates
To prevent vacuous passes (e.g. evaluating on a dataset with 0 negative claims), promotion eligibility requires that all four evaluation class denominators are strictly greater than zero:
1. `gold_supported_claims > 0`
2. `gold_negative_claims > 0`
3. `gold_valid_answers > 0`
4. `gold_invalid_answers > 0`

If any denominator is zero:
- `verdict = "V2_D3_HOLDOUT_COVERAGE_INSUFFICIENT"`
- `promotion_recommended = False`
- `promotion_authorized = False`
- Affected rate metrics are reported as `null` / `None` (never `1.0`).

### 4.2 Pre-Registered Rate Gates (Pass 1 Authoritative)
To recommend promoting V2-D3 into production, the authoritative Pass 1 metrics must satisfy all of the following rate thresholds:

| Metric Gate | Pre-Registered Threshold | Primary Rationale |
| :--- | :---: | :--- |
| **Supported Claim Retention Rate** | $\ge \mathbf{88.00\%}$ | Prevents false rejections of valid grounded claims (Dev: 17/18 = 94.44%). |
| **Negative Claim Catch Rate** | $\ge \mathbf{50.00\%}$ | Ensures meaningful filtering of hallucinated/contradicted claims (Dev: 11/20 = 55.00%). |
| **Valid Answer Retention Rate** | $\ge \mathbf{80.00\%}$ | Protects end-to-end preservation of valid answers (Dev: 6/7 = 85.71%). |
| **Full Answer-Level Accuracy** | $\ge \mathbf{60.00\%}$ | Guarantees net positive answer verification correctness (Dev: 14/22 = 63.64%). |
| **Binary Claim Accuracy** | $\ge \mathbf{70.00\%}$ | Ensures solid overall binary discrimination on unseen claims (Dev: 28/38 = 73.68%). |

### 4.3 Pre-Registered Mechanical Gates
1. **Zero Provider Invocation Errors:** `provider_invocation_errors == 0`.
2. **Zero Execution Errors:** `execution_error_in_any_pass_count == 0`.
3. **Zero Instability:** `unstable_semantic_claim_count == 0` (100% agreement between Pass 1 and Pass 2).
4. **Complete Output Validation:** Exactly 2 valid semantic labels per claim across both passes.
5. **Frozen D3 Source Verification:** Implementation and prompt SHA-256 must match frozen D3 hashes.
6. **Strict Source Binding:** Packets SHA, selection SHA, and labels SHA must match the pre-registered commitment.
7. **Exact Packet-Label Set Equality:** Exact claim key match `(question_id, arm_id, claim_id)` with zero missing, extra, or duplicate labels.

### 4.4 Decision Logic & Verdicts
```
                             Holdout Execution Complete
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Check Non-Vacuous Coverage Gates   │
                      └───────────────────┬───────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │ Sufficient                        │ Insufficient
                        ▼                                   ▼
        ┌───────────────────────────────┐        ┌───────────────────────────────┐
        │   Check Mechanical Gates      │        │ V2_D3_HOLDOUT_COVERAGE_INSUFF │
        └───────────────┬───────────────┘        │   (REJECT_V2_D3_PROMOTION)    │
                        │                        └───────────────────────────────┘
            ┌───────────┴───────────┐
            │ Pass                  │ Fail
            ▼                       ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│   Check Quality Rate Gates    │ │ V2_D3_HOLDOUT_EXECUTION_FAIL  │
└───────────────┬───────────────┘ │   (REJECT_V2_D3_PROMOTION)    │
                │                 └───────────────────────────────┘
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
- Production semantic verifier enablement (`enabled = true` in config) requires human governance review and authorization after inspecting the canonical holdout evidence package.

---

## 5. Kaggle Holdout Execution Runbook (Cells H1–H6)

### Cell H1: Environment Setup, Pinned HF Runtime & Git Checkout
```python
# ==============================================================================
# CELL H1: ENVIRONMENT SETUP, PINNED HF RUNTIME & CODEBASE CHECKOUT
# ==============================================================================
import os
import subprocess
import sys
from pathlib import Path

# Pinned Execution Authority Commit
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

# Verify clean worktree
git_status = subprocess.check_output(["git", "status", "--short"], cwd=str(REPO_DIR), text=True).strip()
assert len(git_status) == 0, f"Git worktree is not clean:\n{git_status}"

# Install exact pinned Hugging Face runtime without reinstalling Torch
subprocess.run([
    "pip", "install", "-q",
    "transformers==4.47.1",
    "tokenizers==0.21.4",
    "huggingface-hub==0.27.1",
    "accelerate==1.2.1",
], check=True)

# Install project in editable mode without dependencies
subprocess.run(["python", "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=str(REPO_DIR), check=True)

# Run subprocess import smoke test and version assertions
smoke_cmd = """\
import importlib.metadata
import torch, transformers, tokenizers, huggingface_hub, accelerate, pydantic, legal_agentic_rag

print(f"Torch Version:        {torch.__version__}")
print(f"CUDA Available:       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:           {torch.cuda.get_device_name(0)}")
print(f"Transformers Version: {transformers.__version__}")
print(f"Tokenizers Version:   {tokenizers.__version__}")
print(f"HF Hub Version:       {huggingface_hub.__version__}")
print(f"Accelerate Version:   {accelerate.__version__}")
print(f"Pydantic Version:     {pydantic.__version__}")
print(f"Package Version:      {legal_agentic_rag.__version__}")

assert torch.cuda.is_available(), "CUDA device required!"
assert importlib.metadata.version("transformers") == "4.47.1", f"Transformers mismatch: {importlib.metadata.version('transformers')}"
assert importlib.metadata.version("tokenizers") == "0.21.4", f"Tokenizers mismatch: {importlib.metadata.version('tokenizers')}"
assert importlib.metadata.version("huggingface_hub") == "0.27.1", f"HF Hub mismatch: {importlib.metadata.version('huggingface_hub')}"
assert importlib.metadata.version("accelerate") == "1.2.1", f"Accelerate mismatch: {importlib.metadata.version('accelerate')}"
assert importlib.metadata.version("legal_agentic_rag") == "0.50.7", f"Package version mismatch: {importlib.metadata.version('legal_agentic_rag')}"
"""
subprocess.run([sys.executable, "-c", smoke_cmd], cwd=str(REPO_DIR), check=True)
print("Environment setup and runtime verification complete.")
```

### Cell H2: Dataset Discovery & Exact Byte Checksum Verification
```python
# ==============================================================================
# CELL H2: HOLDOUT DATASET DISCOVERY & BYTE CHECKSUM VERIFICATION
# ==============================================================================
from hashlib import sha256
import json
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

INPUT_DIR = Path("/kaggle/input")
REPO_DIR = Path("/kaggle/working/legal-agentic-rag")

HOLDOUT_PACKETS_CANDIDATES = list(INPUT_DIR.glob("**/verification-v2-holdout-review-packets-v1.zip"))
HOLDOUT_SELECTION_CANDIDATES = list(INPUT_DIR.glob("**/verification-v2-holdout-selection-v1.json"))
HOLDOUT_LABELS_CANDIDATES = list(INPUT_DIR.glob("**/verification-v2-holdout-reviewed-labels-v1.json"))

assert HOLDOUT_PACKETS_CANDIDATES, "verification-v2-holdout-review-packets-v1.zip not found in /kaggle/input"
assert HOLDOUT_SELECTION_CANDIDATES, "verification-v2-holdout-selection-v1.json not found in /kaggle/input"
assert HOLDOUT_LABELS_CANDIDATES, "verification-v2-holdout-reviewed-labels-v1.json not found in /kaggle/input"

HOLDOUT_PACKETS_PATH = HOLDOUT_PACKETS_CANDIDATES[0]
HOLDOUT_SELECTION_PATH = HOLDOUT_SELECTION_CANDIDATES[0]
HOLDOUT_LABELS_PATH = HOLDOUT_LABELS_CANDIDATES[0]
LABEL_COMMITMENT_PATH = REPO_DIR / "configs" / "verification-v2-d3-holdout-label-commitment.json"

assert LABEL_COMMITMENT_PATH.is_file(), f"Label commitment missing: {LABEL_COMMITMENT_PATH}"
commitment = json.loads(LABEL_COMMITMENT_PATH.read_text(encoding="utf-8"))

EXPECTED_PACKETS_SHA = "a7a591752f0e9aa376f424217d5d06f7fa90e66fce0d67ed4af78ae048b53be4"
EXPECTED_SELECTION_SHA = "08c480f6ffad2e950319f487111ecd0ac549d2f8b10149820ecc84d34ea00a4b"
EXPECTED_LABELS_SHA = commitment["labels_sha256"]

actual_packets_sha = sha256_file(HOLDOUT_PACKETS_PATH)
actual_selection_sha = sha256_file(HOLDOUT_SELECTION_PATH)
actual_labels_sha = sha256_file(HOLDOUT_LABELS_PATH)

print(f"Holdout Packets:   {HOLDOUT_PACKETS_PATH} (SHA: {actual_packets_sha})")
print(f"Holdout Selection: {HOLDOUT_SELECTION_PATH} (SHA: {actual_selection_sha})")
print(f"Holdout Labels:    {HOLDOUT_LABELS_PATH} (SHA: {actual_labels_sha})")

assert actual_packets_sha == EXPECTED_PACKETS_SHA, f"Packets SHA mismatch: {actual_packets_sha}"
assert actual_selection_sha == EXPECTED_SELECTION_SHA, f"Selection SHA mismatch: {actual_selection_sha}"
assert actual_labels_sha == EXPECTED_LABELS_SHA, f"Labels SHA mismatch: {actual_labels_sha}"

print("All holdout input checksums successfully verified against frozen commitment.")
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
    "--label-commitment", str(LABEL_COMMITMENT_PATH),
    "--output-dir", str(OUTPUT_DIR / "preflight"),
    "--preflight-only",
]

print("Executing model-free preflight check...")
subprocess.run(cmd_preflight, cwd=str(REPO_DIR), check=True)
print("Preflight verification PASSED. Ready for model inference.")
```

### Cell H4: Model Access Verification (No Double Loading in Notebook Memory)
```python
# ==============================================================================
# CELL H4: MODEL ACCESS VERIFICATION (NO NOTEBOOK MEMORY RESIDENCE)
# ==============================================================================
import torch
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "a1d308dfcc03e09da285d49d912439a655a571e8"

assert torch.cuda.is_available(), "CUDA device is required for canonical V2-D3 holdout execution"
device_name = torch.cuda.get_device_name(0)
print(f"CUDA Available: {device_name} (Device Count: {torch.cuda.device_count()})")

print(f"Verifying tokenizer and model accessibility for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
print("Model accessibility verified. Model will be loaded exclusively by the evaluation subprocess.")
```

### Cell H5: Live Canonical Execution & Evidence Packaging
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
    "--label-commitment", str(LABEL_COMMITMENT_PATH),
    "--output-dir", str(OUTPUT_DIR),
    "--package-zip", str(PACKAGE_ZIP),
    "--device", "cuda",
    "--torch-dtype", "float16",
    "--temperature", "0.0",
]

print("Launching canonical V2-D3 Holdout Evaluation subprocess...")
subprocess.run(cmd_eval, cwd=str(REPO_DIR), check=True)
print(f"Holdout evaluation finished. Evidence package generated at {PACKAGE_ZIP}")
```

### Cell H6: Independent Evidence Verification & Forensic Reconciliation
```python
# ==============================================================================
# CELL H6: INDEPENDENT EVIDENCE VERIFICATION & RECOMPUTATION
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

# 1. Independent Recomputations from Evidence Metrics
d3_binary = report["metrics"]["v2_d3_claim_binary"]
d3_answer = report["metrics"]["v2_d3_answer_metrics"]
telemetry = report["telemetry"]
stability_data = stability["stability"]

# Coverage Denominators
supp_denom_valid = d3_binary["gold_supported_claims"] > 0
neg_denom_valid = d3_binary["gold_negative_claims"] > 0
val_ans_denom_valid = d3_answer["gold_valid_answers_count"] > 0
inv_ans_denom_valid = d3_answer["gold_invalid_answers_count"] > 0
recomputed_coverage_sufficient = (
    supp_denom_valid and neg_denom_valid and val_ans_denom_valid and inv_ans_denom_valid
)

# Mechanical Gates & Provider Call Reconciliation
total_claims = report["total_claims"]
total_retries = telemetry["total_structured_retries"]
expected_calls = 2 * total_claims + total_retries
recomputed_calls_reconciled = (
    len(provider_calls) == telemetry["total_provider_calls"] == expected_calls
)

recomputed_mechanical_pass = (
    telemetry["model_errors"] == 0
    and stability_data["execution_error_in_any_pass_count"] == 0
    and stability_data["unstable_semantic_claim_count"] == 0
    and stability_data["claims_with_two_valid_semantic_labels"] == total_claims
    and recomputed_calls_reconciled
    and exec_id.get("frozen_d3_source_identity_verified", False) is True
)

# Quality Gates (Pass 1 Authoritative)
GATE_MIN_SUPPORTED_RETENTION = 0.88
GATE_MIN_NEGATIVE_CATCH = 0.50
GATE_MIN_VALID_ANSWER_RETENTION = 0.80
GATE_MIN_FULL_ANSWER_ACCURACY = 0.60
GATE_MIN_CLAIM_BINARY_ACCURACY = 0.70

supp_ret = d3_binary["supported_retention"]
neg_catch = d3_binary["negative_catch"]
val_ans_ret = d3_answer["valid_answer_retention_rate"]
full_ans_acc = d3_answer["full_denominator_answer_accuracy"]
claim_bin_acc = d3_binary["accuracy"]

recomputed_quality_gates_pass = (
    recomputed_coverage_sufficient
    and (supp_ret is not None and supp_ret >= GATE_MIN_SUPPORTED_RETENTION)
    and (neg_catch is not None and neg_catch >= GATE_MIN_NEGATIVE_CATCH)
    and (val_ans_ret is not None and val_ans_ret >= GATE_MIN_VALID_ANSWER_RETENTION)
    and (full_ans_acc is not None and full_ans_acc >= GATE_MIN_FULL_ANSWER_ACCURACY)
    and (claim_bin_acc is not None and claim_bin_acc >= GATE_MIN_CLAIM_BINARY_ACCURACY)
)

if not recomputed_coverage_sufficient:
    recomputed_verdict = "V2_D3_HOLDOUT_COVERAGE_INSUFFICIENT"
    recomputed_decision = "REJECT_V2_D3_PROMOTION"
    recomputed_promotion_recommended = False
elif not recomputed_mechanical_pass:
    recomputed_verdict = "V2_D3_HOLDOUT_EXECUTION_FAILURE"
    recomputed_decision = "REJECT_V2_D3_PROMOTION"
    recomputed_promotion_recommended = False
elif recomputed_quality_gates_pass:
    recomputed_verdict = "V2_D3_HOLDOUT_PROMOTION_RECOMMENDED"
    recomputed_decision = "PROMOTE_V2_D3_TO_PRODUCTION"
    recomputed_promotion_recommended = True
else:
    recomputed_verdict = "V2_D3_HOLDOUT_PROMOTION_REJECTED"
    recomputed_decision = "REJECT_V2_D3_PROMOTION"
    recomputed_promotion_recommended = False

# 2. Strict Invariant Assertions
assert decision["coverage_sufficient"] == recomputed_coverage_sufficient, "Coverage boolean mismatch!"
assert decision["mechanical_pass"] == recomputed_mechanical_pass, "Mechanical pass boolean mismatch!"
assert decision["quality_gates_pass"] == recomputed_quality_gates_pass, "Quality gates pass boolean mismatch!"
assert decision["promotion_recommended"] == recomputed_promotion_recommended, "Promotion recommendation mismatch!"
assert decision["verdict"] == recomputed_verdict == report["verdict"], "Verdict mismatch!"
assert decision["holdout_evaluation_decision"] == recomputed_decision, "Holdout decision mismatch!"
assert decision["promotion_authorized"] is False, "CRITICAL: promotion_authorized MUST BE False (Fail-Closed Invariant)!"

# 3. Canonical Hashes & Provenance Assertions
CANONICAL_D3_IMPL_SHA = "a6e8bca15ad14d869e103e1f94fe94bb9a81f9ddc8bc650b280b69b7d57e9826"
CANONICAL_D3_SYS_SHA = "546cd8bd33b3c640c66023f653c87955418569b56ab9d68c5d2c325fb9bd283b"
CANONICAL_D3_SCHEMA_SHA = "3591144a40b0519d5da9dd262e8edf8814531d798b69deea94fd81fae39f5f61"

assert exec_id["candidate_id"] == "V2-D3"
assert exec_id["package_version"] == "0.50.7"
assert exec_id["repeat_count"] == 2
assert exec_id["prompt_identities"]["d3_base_system_instruction_sha256"] == CANONICAL_D3_SYS_SHA
assert exec_id["prompt_identities"]["d3_schema_sha256"] == CANONICAL_D3_SCHEMA_SHA
assert exec_id["implementation_identities"]["structured_semantic_verifier_d3_sha256"] == CANONICAL_D3_IMPL_SHA

for call in provider_calls:
    assert call["system_instruction_sha256"] == CANONICAL_D3_SYS_SHA
    assert "error_message_summary" not in call, "Raw error message must not be in call telemetry!"

print("="*75)
print("             V2-D3 FRESH HOLDOUT INDEPENDENT VERIFICATION")
print("="*75)
print(f"Archive SHA-256:         {zip_sha}")
print(f"Archive Size:            {zip_size:,} bytes")
print(f"Total Claims:            {total_claims}")
print(f"Total Provider Calls:    {len(provider_calls)} (Reconciled: {recomputed_calls_reconciled})")
print(f"Coverage Sufficient:     {recomputed_coverage_sufficient}")
print(f"Mechanical Pass:         {recomputed_mechanical_pass}")
print(f"Quality Gates Pass:      {recomputed_quality_gates_pass}")
print(f"Verdict:                 {recomputed_verdict}")
print(f"Promotion Recommended:   {recomputed_promotion_recommended}")
print(f"Promotion Authorized:    {decision['promotion_authorized']} (Strict Invariant)")
print("="*75)
print("ALL INDEPENDENT EVIDENCE VERIFICATION ASSERTIONS PASSED.")
print("\n*** DOWNLOAD EVIDENCE ARCHIVE BEFORE TERMINATING KAGGLE SESSION ***\n")
```

---

## 6. Explicit Governance Invariants

- **V2-D3.1 FORMALLY CLOSED AS `KEEP_D3`**
- **V2-D3.2 FORMALLY CLOSED AS `KEEP_D3`**
- **V2-D3 IS FORMALLY FROZEN AS THE EXCLUSIVE V2 CANDIDATE**
- **DEVELOPMENT BENCHMARK IS PERMANENTLY CLOSED FOR CANDIDATE TUNING**
- **ZERO D3 PREDICTIONS VISIBLE OR EXISTING DURING PHASE H-LABEL REVIEW**
- **ZERO MODEL TUNING, PROMPT EDITS, OR THRESHOLD EDITS DURING OR AFTER H-EXEC**
- **PRE-REGISTERED RATE GATES AND NON-VACUOUS COVERAGE GATES ARE IMMUTABLE**
- **`promotion_authorized` IS UNCONDITIONALLY FALSE IN HARNESS OUTPUTS**
- **NO PRODUCTION WIRING MODIFIED (SEMANTIC VERIFIER REMAINS DISABLED IN PROD)**
- **ZERO CHANGES TO RETRIEVAL / RERANKING / GENERATION PIPELINE CORE**
