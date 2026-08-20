# docs/30-V2-D32-STRICT-CONFLICT-OVERLAY-PROTOCOL.md

## 1. Executive Summary & V2-D3.1 Formal Closure

### 1.1 V2-D3.1 Canonical Benchmark Closure: `KEEP_D3`
On August 21, 2026, Candidate **V2-D3.1** completed its canonical development benchmark execution on Kaggle.

```
══════════════════════════════════════════════════════════════════════════
Candidate:                  V2-D3.1
Execution Git Commit:       1383bf379a01c3f7456e3c41ba3be42846ceee2c
Canonical Evidence Archive: verification-v2-d31-development-evidence.zip
SHA-256:                    e14f9656a13a04b8e545d88a5dca13653fa317166ff530f45e4b13124f864041
Archive Size:               18,379 bytes (11 members)
Verdict:                    V2_D31_DEVELOPMENT_BENCHMARK_PASS
Development Decision:       KEEP_D3
d31_supersedes_d3:          false
promotion_authorized:       false (Fail-Closed)
══════════════════════════════════════════════════════════════════════════
```

### 1.2 Comparison: Canonical D3 vs D3.1
| Metric Category | Canonical D3 (Baseline) | Candidate V2-D3.1 | Delta |
| :--- | :---: | :---: | :---: |
| **Binary Claim Accuracy** | 28 / 38 (73.68%) | 30 / 38 (78.95%) | +2 |
| **Supported Retained (Gold=18)** | **17 / 18 (94.44%)** | 12 / 18 (66.67%) | **-5 (Severe Regression)** |
| **Negative Caught (Gold=20)** | 11 / 20 (55.00%) | 18 / 20 (90.00%) | +7 |
| **Three-Way Claim Accuracy** | **24 / 38 (63.16%)** | 20 / 38 (52.63%) | **-4 (Severe Regression)** |
| **Contradictions Caught (Gold=7)** | 0 / 7 (0.00%) | 6 / 7 (85.71%) | +6 |
| **Answer-Level Accuracy** | 14 / 22 (63.64%) | 17 / 22 (77.27%) | +3 |
| **D3 Gains Preserved (Out of 7)** | 7 / 7 (100.0%) | **1 / 7 (14.29%)** | **-6 Regressions** |
| **Mechanical Stability** | 38 / 38 (100%) | 38 / 38 (100%) | 0 errors, 0 retries, 76 calls |

### 1.3 Forensic Finding: High-Recall / Low-Precision Contradiction Detector
D3.1 Pass 1 emitted **20 CONTRADICTED predictions** across the 38 claims:
- **6** were true human `CONTRADICTED` (out of 7 total $\to$ 85.71% recall).
- **10** were human `INSUFFICIENT` (false contradiction positives).
- **4** were human `SUPPORTED` (false contradiction positives $\to$ severe damage to supported retention).
- **Contradiction Precision:** $6 / 20 = 30.00\%$.

**Conclusion:** D3.1 demonstrated that Qwen2.5-3B can detect contradiction signals with high sensitivity, but a single two-gate monolithic classifier severely overcalls contradiction, destroying D3's high supported retention (dropping from 17/18 down to 12/18) and regressing 6 of 7 D3 fixes.

**Canonical Champion Remains:** **V2-D3**.

---

## 2. Candidate V2-D3.2 Design Principle

> [!IMPORTANT]
> **V2-D3.2 is the FINAL PLANNED DEVELOPMENT ITERATION.**
> There is NO automatic D3.3 planned. V2-D3.2 will not attempt another wholesale replacement classifier.

### Architecture: Frozen D3 Base + Strict Contradiction Confirmation Overlay
V2-D3.2 is designed as an **asymmetric multi-stage verifier**:
1. **CALL A (Frozen D3 Base Verifier):** Evaluates the claim using the exact frozen D3 prompt and schema to obtain `base_d3_label` (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`).
2. **CALL B (Strict Contradiction Confirmation Verifier):** Independently evaluates whether the cited evidence affirmatively establishes a proposition that is strictly mutually exclusive with the claim.
3. **Trusted Python Combination (Asymmetric Override):**
   - If strict contradiction is confirmed by Call B: `final_label = CONTRADICTED` (Override applied).
   - Otherwise: `final_label = base_d3_label` (D3 base label strictly preserved).

```
                      Single Claim + Cited Statutory Evidence
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            [CALL A: Frozen D3]                   [CALL B: Strict Conflict]
          3-way relation + 5 flags               same_material_proposition
                    │                              cannot_both_be_true
                    ▼                                         │
             base_d3_label                                    ▼
                    │                            strict_conflict_confirmed?
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                             ┌───────────────────────┐
                             │  Trusted Combination  │
                             │    (Asymmetric Rule)  │
                             └───────────┬───────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
     [If Conflict Confirmed (T, T)]                   [All Other States]
              CONTRADICTED                              base_d3_label
           (Override Applied)                        (Strictly Preserved)
```

### Invariant: Protection of D3 Calibration
D3.2 can **only change a D3 prediction TO `CONTRADICTED`**. It cannot change `SUPPORTED -> INSUFFICIENT`, `INSUFFICIENT -> SUPPORTED`, or `CONTRADICTED -> anything else`. This explicitly protects D3's $17/18$ supported retention and support/insufficient calibration unless the strict conflict test confirms mutual exclusivity.

---

## 3. Strict Contradiction Confirmation Specification

### 3.1 Model Output Schema (Call B)
```python
class StructuredSemanticConflictDraftD32(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str
    same_material_proposition: bool
    cannot_both_be_true: bool
```
- **Exactly three fields:** No explanation, no reasoning, no final label, no benchmark information.

### 3.2 Conflict State Machine & Logical Consistency
- **State A (`same_material_proposition = True, cannot_both_be_true = True`):** $\to$ `STRICT_CONTRADICTION_CONFIRMED` (triggers override).
- **State B (`same_material_proposition = True, cannot_both_be_true = False`):** $\to$ `NO_STRICT_CONTRADICTION` (preserves D3).
- **State C (`same_material_proposition = False, cannot_both_be_true = False`):** $\to$ `NO_STRICT_CONTRADICTION` (preserves D3).
- **Invalid State (`same_material_proposition = False, cannot_both_be_true = True`):** Rejected as `DraftRejectionCategoryD32Conflict.LOGICALLY_INCONSISTENT_STATE`. Triggers structured retry. If repeated, logged as an isolated execution error.

### 3.3 Semantic Definitions & Co-Truth Test
1. **`same_material_proposition = true`:** Cited evidence makes an affirmative statutory statement about the **exact same material semantic slot** asserted by the claim (e.g. deciding authority role, prerequisite trigger, exception rule, quantitative/temporal threshold, or legal modality).
   - Unrelated documents, different articles on separate subjects, different actors for other duties, or numbers serving other roles are NOT on the same proposition (`same_material_proposition = false`).
2. **`cannot_both_be_true = true` (Strict Co-Truth Test):** "Under the question's legal scope, if the statutory proposition asserted in the evidence is true, is it legally impossible for the claim to also be true?"
   - If YES (they can legally coexist or evidence is silent) $\to$ `cannot_both_be_true = false`.
   - If NO (they are legally mutually exclusive / structurally incompatible) $\to$ `cannot_both_be_true = true`.
3. **Critical Non-Contradiction Invariants:**
   - Wrong document, wrong article, partial evidence, silence, uninformative text, failure to prove the complete claim are NOT contradictions (`cannot_both_be_true = false`).
   - Absence of evidence is NEVER a contradiction.

---

## 4. Call Accounting & Execution Identity

- **Model Identity:** `Qwen/Qwen2.5-3B-Instruct` (`a1d308dfcc03e09da285d49d912439a655a571e8`), `transformers==4.47.1`, `cuda`, `float16`, `temperature=0.0`.
- **Call Budget:**
  - 38 claims $\times$ 2 calls per claim (Call A + Call B) = **76 base calls per pass**.
  - Across 2 passes (Pass 1 authoritative + Pass 2 stability) = **152 base provider calls**.
- **Telemetry Separation:** `D32ClaimVerificationTelemetry` tracks `d3_base_calls`, `d3_base_retries`, `conflict_calls`, `conflict_retries`, `total_provider_calls`, and `override_applied`.

---

## 5. Pre-Registered Selection Gates (D3.2 vs D3)

Candidate **V2-D3.2** supersedes D3 (`D32_SUPERSEDES_D3`) if and only if ALL mechanical and quality gates pass:

1. **Mechanical Gates (Must All Pass):**
   - `verdict == "V2_D32_DEVELOPMENT_BENCHMARK_PASS"`
   - `model_errors == 0`
   - `provider_invocation_errors == 0`
   - `execution_error_in_any_pass_count == 0`
   - `unstable_semantic_claim_count == 0` (100% two-pass stability)
   - `38 / 38` evaluated claims with valid semantic labels
2. **Quality Gates vs D3 Comparator (Exact Integer Counts):**
   - `d32_binary_correct > 28 / 38` ($\ge 29$)
   - `d32_supported_retained >= 17 / 18`
   - `d32_negative_caught > 11 / 20` ($\ge 12$)
   - `paired_net_delta_vs_d3 > 0`
   - `d32_three_way_correct > 24 / 38` ($\ge 25$)
   - `d32_correctly_predicted_contradicted > 0 / 7`
   - `d32_answer_correct >= 14 / 22`

If any gate fails $\to$ Decision is **`KEEP_D3`**.
In all cases, `promotion_authorized = false` (fail-closed development policy).

---

## 6. Scientific Development Diagnostics

The D3.2 benchmark harness automatically computes and reports:
1. **Strict-Conflict Precision & Recall Diagnostic:**
   - Total strict conflict positives.
   - True human `CONTRADICTED` count among positives.
   - False positive overrides breakdown (`SUPPORTED -> CONTRADICTED` vs `INSUFFICIENT -> CONTRADICTED`).
   - Conflict precision and recall against 7 gold contradictions.
2. **D3.1 Learning Diagnostic:**
   - Evaluates D3.2 on the 20 contradiction positives emitted by D3.1.
   - Quantifies how many of D3.1's 20 positives are retained vs filtered out by D3.2, with human label distributions.
3. **7 D3 Gains Preservation Diagnostic:**
   - Tracks outcomes on `26541:BASE:C1`, `95861:BASE:C3`, `95861:CANDIDATE:C2`, `95861:CANDIDATE:C3`, `103983:PRIMARY:C2`, `108497:PRIMARY:C1`, `155139:PRIMARY:C1`.
4. **Four Forensic Error Groups Diagnostic:**
   - Group A (3 false entailments of contradiction), Group B (4 contradiction undercalls), Group C (6 false entailments of insufficient), Group D (1 supported overrejection).

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

### Cell 3 — Holdout / Phase-A Blindness Guard & Exact Seven Development Sources Discovery
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

# 2. DISCOVER EXACT SEVEN DEVELOPMENT SOURCES
CANONICAL_SOURCES = {
    "verification-forensic-review-packets.zip": "996909f83c5e3e7d092323153fe780e713509022e64b7eab6135e64ebc2c379a",
    "verification-human-forensic-labels-v1.json": "bad739b6d4faff74d028c9f18594564c5d0bb58babde9a6498b298ec4fee7733",
    "verification-positive-control-review-packets-v1.zip": "cbb120bffe4d4592e8f5efafbeae42993dc7b7e49a722f451a3fc4eec9236cc4",
    "verification-positive-control-human-labels-v1.json": "60037c4353063357d993e727586581660244b8fdca77483f6fe3c42397053373",
    "verification-semantic-benchmark-evidence.zip": "bcded65f2bd72423ac7d6c46ff3f8c05d52bd96ab6095321a8c4b9694ed802c6",
    "verification-v2-d3-development-evidence.zip": "0d95aeee73a18dd75d617c5e493891a8407646826df0fe4b6b74d362d23184ff",
    "verification-v2-d31-development-evidence.zip": "e14f9656a13a04b8e545d88a5dca13653fa317166ff530f45e4b13124f864041",
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
Path("/kaggle/working/v2_d32_development_sources.json").write_text(
    json.dumps(discovered_sources, indent=2), encoding="utf-8"
)
print("All 7 canonical development sources discovered and verified.")
```

### Cell 4 — Model-Free Kaggle Preflight Gate
```bash
!python scripts/evaluate_verification_v2_d32_development.py \
  --forensic-packets "{discovered_sources['verification-forensic-review-packets.zip']}" \
  --forensic-labels "{discovered_sources['verification-human-forensic-labels-v1.json']}" \
  --control-packets "{discovered_sources['verification-positive-control-review-packets-v1.zip']}" \
  --control-labels "{discovered_sources['verification-positive-control-human-labels-v1.json']}" \
  --v1-evidence "{discovered_sources['verification-semantic-benchmark-evidence.zip']}" \
  --d3-evidence "{discovered_sources['verification-v2-d3-development-evidence.zip']}" \
  --d31-evidence "{discovered_sources['verification-v2-d31-development-evidence.zip']}" \
  --output-dir "/kaggle/working/v2_d32_preflight_output" \
  --preflight-only

import json
from pathlib import Path

pf_rep = json.loads(Path("/kaggle/working/v2_d32_preflight_output/results/v2_d32_development_report.json").read_text())
assert pf_rep["verdict"] == "V2_D32_DEVELOPMENT_BENCHMARK_READY", f"Unexpected verdict: {pf_rep['verdict']}"
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

### Cell 5 — Real V2-D3.2 Canonical Model Execution
```bash
!python scripts/evaluate_verification_v2_d32_development.py \
  --forensic-packets "{discovered_sources['verification-forensic-review-packets.zip']}" \
  --forensic-labels "{discovered_sources['verification-human-forensic-labels-v1.json']}" \
  --control-packets "{discovered_sources['verification-positive-control-review-packets-v1.zip']}" \
  --control-labels "{discovered_sources['verification-positive-control-human-labels-v1.json']}" \
  --v1-evidence "{discovered_sources['verification-semantic-benchmark-evidence.zip']}" \
  --d3-evidence "{discovered_sources['verification-v2-d3-development-evidence.zip']}" \
  --d31-evidence "{discovered_sources['verification-v2-d31-development-evidence.zip']}" \
  --output-dir "/kaggle/working/v2_d32_development_output" \
  --package-zip "/kaggle/working/verification-v2-d32-development-evidence.zip" \
  --device cuda \
  --repeat-count 2
```

### Cell 6 — Canonical Evidence Verification, Telemetry Gates, & Scientific Decision
```python
import json, zipfile
from pathlib import Path
from hashlib import sha256

# 1. VERIFY EVIDENCE ARCHIVE INVENTORY
zip_path = Path("/kaggle/working/verification-v2-d32-development-evidence.zip")
assert zip_path.is_file(), "Evidence archive ZIP not found"

REQUIRED_MEMBERS = [
    "execution/v2_d32_development_source_identity.json",
    "results/v2_d32_development_report.json",
    "results/v2_d32_development_decision_report.json",
    "results/v2_d32_strict_conflict_diagnostics.json",
    "results/v2_d32_d31_learning_diagnostic.json",
    "results/v0_claim_predictions.jsonl",
    "results/v1_claim_predictions.jsonl",
    "results/v2_d3_claim_predictions.jsonl",
    "results/v2_d31_claim_predictions.jsonl",
    "results/v2_d32_claim_predictions_pass1.jsonl",
    "results/v2_d32_claim_predictions_pass2.jsonl",
    "results/v2_d32_claim_comparisons.jsonl",
    "telemetry/provider_calls.jsonl",
]

with zipfile.ZipFile(zip_path, "r") as zf:
    archive_members = set(zf.namelist())
    for req in REQUIRED_MEMBERS:
        assert req in archive_members, f"Missing required archive member: {req}"

print(f"Verified {len(REQUIRED_MEMBERS)} canonical evidence archive members.")

# 2. LOAD REPORTS & RECONCILE TELEMETRY
out_dir = Path("/kaggle/working/v2_d32_development_output")
report = json.loads((out_dir / "results/v2_d32_development_report.json").read_text())
decision = json.loads((out_dir / "results/v2_d32_development_decision_report.json").read_text())

# Verify provider call telemetry
provider_calls_lines = (out_dir / "telemetry/provider_calls.jsonl").read_text().strip().splitlines()
provider_calls = [json.loads(line) for line in provider_calls_lines]
total_calls = report["telemetry"]["total_provider_calls"]
assert len(provider_calls) == total_calls, f"Provider call count mismatch: {len(provider_calls)} != {total_calls}"

d3_sys_sha = report["execution_identity"]["prompt_identities"]["d3_base_system_instruction_sha256"]
conflict_sys_sha = report["execution_identity"]["prompt_identities"]["d32_conflict_system_instruction_sha256"]
valid_sys_shas = {d3_sys_sha, conflict_sys_sha}

for i, call in enumerate(provider_calls):
    assert call["system_instruction_sha256"] in valid_sys_shas, f"Call {i} system instruction SHA mismatch"

# 3. ASSERT EXECUTION IDENTITY GATES
exec_id = report["execution_identity"]
assert exec_id["candidate_id"] == "V2-D3.2"
assert exec_id["package_version"] == "0.50.7"
assert exec_id["installed_distribution_version"] == "0.50.7"
assert exec_id["provider"]["backend"] == "transformers"
assert exec_id["provider"]["provider_version"] == "4.47.1"
assert exec_id["provider"]["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
assert exec_id["provider"]["model_revision"] == "a1d308dfcc03e09da285d49d912439a655a571e8"
assert decision["promotion_authorized"] is False, "promotion_authorized must remain false"

# 4. HARD BENCHMARK PASS ASSERTIONS (IF PASS)
if report["verdict"] == "V2_D32_DEVELOPMENT_BENCHMARK_PASS":
    assert report["telemetry"]["model_errors"] == 0
    assert report["telemetry"]["provider_invocation_errors"] == 0
    assert report["stability"]["execution_error_in_any_pass_count"] == 0
    assert report["stability"]["unstable_semantic_claim_count"] == 0
    assert report["stability"]["claims_with_two_valid_semantic_labels"] == 38
    assert report["metrics"]["v2_d32_claim_binary"]["execution_errors"] == 0
    assert report["metrics"]["v2_d32_three_way"]["execution_errors"] == 0

# 5. SCIENTIFIC SUMMARY DISPLAY
d3_binary = report["metrics"]["v2_d3_claim_binary"]
d31_binary = report["metrics"]["v2_d31_claim_binary"]
d32_binary = report["metrics"]["v2_d32_claim_binary"]
d32_three_way = report["metrics"]["v2_d32_three_way"]
paired_d3 = report["metrics"]["paired_v2_d32_vs_v2_d3"]
contra_diag = report["metrics"]["contradiction_capability"]
conflict_diag = report["strict_conflict_diagnostics"]
gain_diag = report["gain_preservation_diagnostic"]
d3_ans = report["metrics"]["v2_d3_answer_metrics"]
d32_ans = report["metrics"]["v2_d32_answer_metrics"]

print("\n" + "="*75)
print("               V2-D3.2 SCIENTIFIC DEVELOPMENT OUTCOME")
print("="*75)
print(f"Verdict: {report['verdict']}")
print(f"Development Decision: {decision['development_evaluation_decision']}")
print(f"D3.2 Supersedes D3: {decision['d32_supersedes_d3']}")
print(f"Promotion Authorized: {decision['promotion_authorized']}")
print("-"*75)
print(f"Binary Claim Accuracy:    D3: {d3_binary['tp']+d3_binary['tn']}/38 ({d3_binary['accuracy']:.2%}) | D3.1: {d31_binary['tp']+d31_binary['tn']}/38 ({d31_binary['accuracy']:.2%}) | D3.2: {d32_binary['tp']+d32_binary['tn']}/38 ({d32_binary['accuracy']:.2%})")
print(f"Supported Retained:       D3: {d3_binary['tp']}/18 ({d3_binary['supported_retention']:.2%}) | D3.1: {d31_binary['tp']}/18 ({d31_binary['supported_retention']:.2%}) | D3.2: {d32_binary['tp']}/18 ({d32_binary['supported_retention']:.2%})")
print(f"Negative Caught:          D3: {d3_binary['tn']}/20 ({d3_binary['negative_catch']:.2%}) | D3.1: {d31_binary['tn']}/20 ({d31_binary['negative_catch']:.2%}) | D3.2: {d32_binary['tn']}/20 ({d32_binary['negative_catch']:.2%})")
print(f"Three-Way Claim Accuracy: D3: {report['metrics']['v2_d3_three_way']['accuracy']:.2%} | D3.1: {report['metrics']['v2_d31_three_way']['accuracy']:.2%} | D3.2: {d32_three_way['accuracy']:.2%}")
print(f"Contradictions Caught:    D3: 0/7 (0.00%) | D3.1: 6/7 (85.71%) | D3.2: {contra_diag['d32_correctly_predicted_contradicted_count']}/7 ({contra_diag['d32_contradicted_recall']:.2%})")
print(f"Answer-Level Accuracy:    D3: {d3_ans['valid_answers_retained']+d3_ans['invalid_answers_caught']}/22 ({d3_ans['answer_level_accuracy']:.2%}) | D3.2: {d32_ans['valid_answers_retained']+d32_ans['invalid_answers_caught']}/22 ({d32_ans['answer_level_accuracy']:.2%})")
print("-"*75)
print(f"Strict Conflict Precision: {conflict_diag['strict_conflict_precision']:.2%} ({conflict_diag['true_human_contradicted_among_positives']}/{conflict_diag['strict_conflict_positives_count']})")
print(f"False Overrides by Class:  {conflict_diag['false_overrides_by_class']}")
print(f"Paired Metrics vs D3:      Both Correct={paired_d3['both_correct']}, D3 Only={paired_d3['base_only_correct']}, D3.2 Only={paired_d3['candidate_only_correct']}, Net Delta={paired_d3['net_correctness_delta']}")
print(f"D3 Gains Preserved:        {gain_diag['preserved_gain_count']}/7 (Regressed: {gain_diag['regressed_gain_claim_ids']})")
print("-"*75)

# 6. ZIP SELF-IDENTITY
zip_sha = sha256(zip_path.read_bytes()).hexdigest()
zip_size = zip_path.stat().st_size
print("Canonical Evidence Archive:")
print(f"  Path:         {zip_path}")
print(f"  SHA-256:      {zip_sha}")
print(f"  Size (Bytes): {zip_size:,}")
print(f"  Member Count: {len(archive_members)}")
print("\n*** DOWNLOAD BEFORE ENDING SESSION ***\n" + "="*75)
```

---

## 8. Explicit Governance Invariants

- **V2-D3.1 CLOSED AS `KEEP_D3`**
- **V2-D3 REMAINS CURRENT BEST EXECUTED CANDIDATE**
- **V2-D3.2 IMPLEMENTED — NOT YET EXECUTED**
- **V2-D3.2 IS THE FINAL PLANNED DEVELOPMENT ITERATION**
- **NO D3.3 PLANNED**
- **D3.2 NOT YET SELECTED OVER D3**
- **FRESH HOLDOUT REMAINS STRICTLY SEALED & UNREVIEWED**
- **NO HOLDOUT QIDS COMPUTED OR INSPECTED**
- **NO PRODUCTION WIRING MODIFIED (SEMANTIC VERIFIER DISABLED IN PROD)**
- **ZERO CHANGES TO RETRIEVAL / RERANKING / GENERATION CORE**
