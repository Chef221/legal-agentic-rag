# Phase B1A — Paired Graph-Routing Behavioral Ablation Runbook

## 1. Executive Summary & Pre-Registered Scientific Question

- **Status**: **EXPERIMENT PROTOCOL READY — KAGGLE EXECUTION PENDING**
- **Phase B1B (Structural Graph Deletion)**: **STRICTLY NOT AUTHORIZED**
- **Production Graph Behavior**: **UNCHANGED**
- **Package Version**: **`0.50.6`**
- **Historical Reference Baseline**: M49.6 (`9b0cd0b`) frozen reliability baseline.

### 1.1 The Controlled Scientific Question
Phase A established that the UIT DSC Task 2 competition graph path is architecturally suspicious:
1. The competition graph artifact contains 8,532 nodes and 0 edges by design (`relationships.jsonl` has 0 records).
2. The keyword heuristic matches 8 substring cues on 22 / 991 development questions.
3. All 22 attempted `GRAPH_SEARCH` and terminated with `retrieval_strategy = "graph"`.
4. Because the graph has 0 edges, `graph_search` evaluates only 20 seed chunks instead of the configured 40 candidates in `hybrid_rerank`.

However, Phase A was an observational census, not a causal experiment. Phase B1A answers ONE controlled question:

> **For the exact same 22 Phase-A graph-routed questions, what happens if we replace the adaptive relationship graph route with the fixed `hybrid_rerank@40` candidate, holding all other components, models, prompts, context limits, and verification rules strictly identical?**

---

## 2. Canonical 22-Case Benchmark Set

B1A operates exclusively on the exact 22 graph-affected questions identified during Phase A, in their exact historical `development.json` order:

| Index | Question ID | Historical Ordinal (991) | Substring Cue | Phase-A Outcome |
|---|---|---|---|---|
| 01 | `102047` | 11 | "bổ sung" | answer_verified |
| 02 | `107487` | 53 | "bổ sung" | answer_verified |
| 03 | `110287` | 61 | "bổ sung" | generation_failed |
| 04 | `111905` | 75 | "bổ sung" | answer_verified |
| 05 | `113537` | 91 | "bổ sung" | answer_verified |
| 06 | `122659` | 148 | "hướng dẫn" | answer_verified |
| 07 | `125393` | 168 | "sửa đổi" | answer_verified |
| 08 | `133075` | 218 | "sửa đổi" | answer_verified |
| 09 | `134605` | 229 | "thay thế" | answer_verified |
| 10 | `147239` | 316 | "bổ sung" | answer_verified |
| 11 | `147869` | 320 | "sửa đổi" | answer_verified |
| 12 | `150051` | 339 | "bổ sung" | answer_verified |
| 13 | `26541` | 528 | "hướng dẫn" | answer_verified |
| 14 | `29491` | 542 | "sửa đổi" | answer_verified |
| 15 | `29877` | 545 | "bổ sung" | answer_verified |
| 16 | `39671` | 608 | "bổ sung" | answer_verified |
| 17 | `45219` | 647 | "hướng dẫn" | answer_verified |
| 18 | `47537` | 664 | "sửa đổi" | answer_verified |
| 19 | `48905` | 670 | "hướng dẫn" | generation_failed |
| 20 | `64035` | 760 | "thay thế" | generation_failed |
| 21 | `95861` | 954 | "hướng dẫn" | generation_failed |
| 22 | `99639` | 985 | "hướng dẫn" | answer_verified |

- **Source Question Count**: 991
- **Source Question SHA-256**: `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`
- **Case Manifest**: `configs/phase-b1a-graph-routing-cases.json`

> [!IMPORTANT]
> The 991-question development set is a historical engineering benchmark, not an untouched holdout. B1A results must not be cited as generalizable unseen-data claims.

---

## 3. Pre-Registered Decision Gate

The evaluation decision must be applied mechanically without post-hoc threshold adjustment:

### 3.1 Hard Protocol Gate
All conditions must hold:
- 22/22 exact question identity matching the canonical case manifest.
- BASE: `graph_search_attempt_count == 22` and `graph_terminal_count == 22`.
- CANDIDATE: `graph_search_attempt_count == 0` and `rerank_search_primary_count == 22`.
- No retrieval model errors, no CUDA illegal memory access, and no unhandled process crashes.
- If ANY hard condition fails $\to$ **`INVALID_EXPERIMENT`**.

### 3.2 Reliability Non-Regression Gate
Candidate must satisfy (evaluated against observed BASE rerun counts):
- `candidate.generation_failed <= base.generation_failed`
- `candidate.citation_verification_failed <= base.citation_verification_failed`
- `candidate.answer_verified >= base.answer_verified`
- If reliability regresses $\to$ **`FAIL_RETAIN_CURRENT_GRAPH_PATH`**.

### 3.3 Semantic Quality Gates
- **Strong Pass (`PASS_TO_B1B`)**:
  - $\Delta\text{METEOR}_{\text{mean}} \ge 0.0$ AND $\Delta\text{ROUGE-L}_{\text{mean}} \ge 0.0$, alongside reliability non-regression.
- **Clear Failure (`FAIL_RETAIN_CURRENT_GRAPH_PATH`)**:
  - $\Delta\text{METEOR}_{\text{mean}} \le -0.005$ OR $\Delta\text{ROUGE-L}_{\text{mean}} \le -0.005$.
- **Inconclusive Band (`INCONCLUSIVE`)**:
  - Reliability passes, neither metric suffers a drop $\le -0.005$, but one or both mean deltas are negative. (Requires user review; no automatic B1B deletion).

---

## 4. Kaggle Dual T4 Execution Runbook

### Prerequisites
- **Kaggle Session**: Accelerator `GPU T4 x2`, Internet `ON`.
- **Attached Datasets**:
  1. UIT DSC Task 2 serving artifact package (`uit-dsc-2026-task2-serving-v0430`).
  2. Canonical `development.json` (SHA-256 `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`).

---

### Kaggle Cell K1 — Environment & Dependency Pinning

```bash
%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1

nvidia-smi

# Clone exact B1A infrastructure commit
rm -rf legal-agentic-rag
git clone https://github.com/Chef221/legal-agentic-rag.git
cd legal-agentic-rag

echo "Current Git Commit:"
git rev-parse HEAD

# Uninstall conflicting torchao if present
pip uninstall -y torchao || true

# Install exact pinned dependencies matching Phase A
pip install --no-cache-dir \
  "transformers==4.51.3" \
  "sentence-transformers==5.4.1" \
  "accelerate==1.6.0" \
  "nltk==3.7"

pip install --no-deps -e .

python -c "
import legal_agentic_rag
import torch
print('Package version:', legal_agentic_rag.__version__)
assert legal_agentic_rag.__version__ == '0.50.6', f'Expected 0.50.6, got {legal_agentic_rag.__version__}'
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
assert torch.cuda.device_count() >= 2, f'Expected at least 2 GPUs, got {torch.cuda.device_count()}'
"
```

---

### Kaggle Cell K2 — Input & Artifact Discovery

```python
import hashlib
import json
from pathlib import Path

CANONICAL_DEV_SHA = "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"

# 1. Discover development.json
dev_candidates = list(Path("/kaggle/input").rglob("development.json"))
dev_path = None
for cand in dev_candidates:
    digest = hashlib.sha256(cand.read_bytes()).hexdigest()
    if digest == CANONICAL_DEV_SHA:
        dev_path = cand
        break

assert dev_path is not None, "Could not find canonical development.json with matching SHA"
print(f"Found canonical development.json at: {dev_path}")

# 2. Discover serving artifact root
serving_dirs = [p for p in Path("/kaggle/input").rglob("build_validation_full_corpus.json")]
assert len(serving_dirs) > 0, "Could not find build_validation_full_corpus.json in serving dataset"
serving_root = serving_dirs[0].parent
print(f"Found serving artifact root at: {serving_root}")

# Verify startup report exists
assert (serving_root / "build_validation_full_corpus.json").exists()
print("Validated persisted startup report discovered successfully.")
```

---

### Kaggle Cell K3 — Materialize Exact 22-Question Benchmark

```python
import subprocess
import json

work_dir = Path("/kaggle/working")
mat_output = work_dir / "phase_b1a_cases_22.json"
ident_output = work_dir / "phase_b1a_cases_22_identity.json"

cmd = [
    "python", "legal-agentic-rag/scripts/phase_b1a_graph_routing_ablation.py", "prepare",
    "--development", str(dev_path),
    "--manifest", "legal-agentic-rag/configs/phase-b1a-graph-routing-cases.json",
    "--output", str(mat_output),
    "--identity-output", str(ident_output),
]

subprocess.run(cmd, check=True)

ident = json.loads(ident_output.read_text())
print("Materialized B1A Case Identity:")
print(json.dumps(ident, indent=2))
assert ident["materialized_case_count"] == 22
```

---

### Kaggle Cell K4 — Create BASE and CANDIDATE Runtime Configurations

```python
import json

base_example = json.loads(Path("legal-agentic-rag/configs/phase-a-current-system-census-kaggle.example.json").read_text())
cand_example = json.loads(Path("legal-agentic-rag/configs/phase-b1a-graph-routing-ablation-kaggle.example.json").read_text())

# Apply identical runtime overrides
for cfg in [base_example, cand_example]:
    cfg["artifacts"]["root_path"] = str(serving_root)
    cfg["online"]["generation"]["device"] = "cuda:1"
    cfg["online"]["vector_runtime"]["search_device"] = "cuda:0"
    cfg["offline"]["embedding"]["device"] = "cuda:0"
    cfg["online"]["reranker"]["device"] = "cuda:0"

base_runtime_path = work_dir / "base_runtime_config.json"
cand_runtime_path = work_dir / "candidate_runtime_config.json"

base_runtime_path.write_text(json.dumps(base_example, indent=2))
cand_runtime_path.write_text(json.dumps(cand_example, indent=2))

# Verify config diff using B1A tooling
subprocess.run([
    "python", "legal-agentic-rag/scripts/phase_b1a_graph_routing_ablation.py", "verify-configs",
    "--base-config", str(base_runtime_path),
    "--candidate-config", str(cand_runtime_path),
], check=True)

print("Runtime configs created and verified successfully.")
```

---

### Kaggle Cell K5 — Routing-Only Preflight Verification

```python
from legal_agentic_rag.agent import DeterministicStrategyRouter
from legal_agentic_rag.configuration import AgentConfig, QueryUnderstandingConfig
from legal_agentic_rag.schemas import QueryAnalysis, QueryIntent, RetrievalQuery, RetrievalStrategy, ToolName

query = RetrievalQuery(
    query_id="preflight-rel",
    original_question="Văn bản nào sửa đổi quy định này?",
    normalized_question="văn bản sửa đổi quy định",
    query_analysis=QueryAnalysis(
        intent=QueryIntent.RELATIONSHIP,
        relationship_cues=["sửa đổi"],
    ),
)
tools = {ToolName.GRAPH_SEARCH, ToolName.RERANK_SEARCH, ToolName.HYBRID_SEARCH}

# BASE router: adaptive_routing_enabled = True
base_router = DeterministicStrategyRouter(
    AgentConfig(strategy_order=[RetrievalStrategy.HYBRID_RERANK]),
    QueryUnderstandingConfig(adaptive_routing_enabled=True),
)
base_routes = base_router.plan(query, tools)
print("BASE Routes:", [r.strategy.value for r in base_routes])
assert base_routes[0].strategy == RetrievalStrategy.GRAPH, "BASE must route to GRAPH first"

# CANDIDATE router: adaptive_routing_enabled = False
cand_router = DeterministicStrategyRouter(
    AgentConfig(strategy_order=[RetrievalStrategy.HYBRID_RERANK]),
    QueryUnderstandingConfig(adaptive_routing_enabled=False),
)
cand_routes = cand_router.plan(query, tools)
print("CANDIDATE Routes:", [r.strategy.value for r in cand_routes])
assert cand_routes[0].strategy == RetrievalStrategy.HYBRID_RERANK, "CANDIDATE must route to HYBRID_RERANK first"
print("Routing preflight check passed.")
```

---

### Kaggle Cell K6 — BASE 22 Batch Execution

```bash
%%bash
set -euo pipefail

BASE_OUT="/kaggle/working/base_batch"
mkdir -p "$BASE_OUT"

python -m legal_agentic_rag.serving.cli batch \
  --config /kaggle/working/base_runtime_config.json \
  --questions /kaggle/working/phase_b1a_cases_22.json \
  --output-dir "$BASE_OUT" \
  --device-override "cuda"

echo "BASE batch completed."
```

---

### Kaggle Cell K7 — CANDIDATE 22 Batch Execution

```bash
%%bash
set -euo pipefail

CAND_OUT="/kaggle/working/candidate_batch"
mkdir -p "$CAND_OUT"

python -m legal_agentic_rag.serving.cli batch \
  --config /kaggle/working/candidate_runtime_config.json \
  --questions /kaggle/working/phase_b1a_cases_22.json \
  --output-dir "$CAND_OUT" \
  --device-override "cuda"

echo "CANDIDATE batch completed."
```

---

### Kaggle Cell K8 — Paired Evaluation & Bootstrap Analysis

```python
import subprocess
import nltk

# Ensure NLTK resources
try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

paired_report_path = work_dir / "phase_b1a_paired_report.json"
decision_report_path = work_dir / "phase_b1a_decision_report.json"

subprocess.run([
    "python", "legal-agentic-rag/scripts/phase_b1a_graph_routing_ablation.py", "analyze",
    "--questions", str(mat_output),
    "--base-batch", str(work_dir / "base_batch"),
    "--candidate-batch", str(work_dir / "candidate_batch"),
    "--output-report", str(paired_report_path),
    "--output-decision", str(decision_report_path),
    "--seed", "20260819",
    "--resamples", "10000",
], check=True)

paired_rep = json.loads(paired_report_path.read_text())
print("==================================================")
print("PHASE B1A PAIRED COMPARISON SUMMARY")
print("==================================================")
print(json.dumps(paired_rep["summary"], indent=2))
```

---

### Kaggle Cell K9 — Mechanical Decision Gate Output

```python
decision_rep = json.loads(decision_report_path.read_text())
print("==================================================")
print("PRE-REGISTERED DECISION VERDICT:")
print(decision_rep["verdict"])
print("==================================================")
for reason in decision_rep["reasons"]:
    print(f"- {reason}")
```

---

### Kaggle Cell K10 — Final Evidence Package

```python
evidence_zip = work_dir / "phase-b1a-graph-routing-ablation-evidence.zip"

subprocess.run([
    "python", "legal-agentic-rag/scripts/phase_b1a_graph_routing_ablation.py", "package",
    "--output-zip", str(evidence_zip),
    "--manifest", "legal-agentic-rag/configs/phase-b1a-graph-routing-cases.json",
    "--questions-identity", str(ident_output),
    "--base-config", str(base_runtime_path),
    "--candidate-config", str(cand_runtime_path),
    "--base-batch", str(work_dir / "base_batch"),
    "--candidate-batch", str(work_dir / "candidate_batch"),
    "--paired-report", str(paired_report_path),
    "--decision-report", str(decision_report_path),
], check=True)

print(f"Evidence ZIP created: {evidence_zip}")
print(f"ZIP Size: {evidence_zip.stat().st_size} bytes")
print(f"ZIP SHA-256: {hashlib.sha256(evidence_zip.read_bytes()).hexdigest()}")
print(">>> DOWNLOAD phase-b1a-graph-routing-ablation-evidence.zip BEFORE ENDING KAGGLE SESSION <<<")
```
