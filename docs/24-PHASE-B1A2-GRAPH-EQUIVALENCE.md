# Phase B1A.2: Graph Equivalence and Candidate-Pool Isolation Protocol

---

## 1. Scientific Objective

Phase A established that the UIT DSC Task 2 competition graph artifact contains **zero graph edges**, resulting in **zero graph expansion** across the entire 991-question development census.

Phase B1A compared:
- **BASE**: Adaptive `RELATIONSHIP -> GRAPH` (effective candidate pool = 20)
- **CANDIDATE**: Fixed `HYBRID_RERANK@40` (effective candidate pool = 40)

on the exact 22 graph-routed questions. The result was **`INCONCLUSIVE`** (METEOR delta: `-0.00020`, ROUGE-L delta: `+0.00270`), with forensic analysis revealing that Phase B1A conflated two independent mechanisms:
1. The behavioral presence/redundancy of the zero-edge graph traversal shell.
2. The candidate pool expansion from 20 to 40 pre-rerank candidates.

**Phase B1A.2 is a pure retrieval-only experiment designed to isolate graph equivalence from candidate-pool depth.**

---

## 2. Experimental Arms

For each of the 22 canonical Phase-A graph-routed questions:

### 2.1 ARM G — Current Graph Path
- **Original Query**: `top_k = 8`, `candidate_k = 40`
- **Seed Retrieval**: `top_k = 20`, `candidate_k = 40`, `requested_strategy = HYBRID`
  - Sparse (BM25) branch depth = 40
  - Dense (E5) branch depth = 40
  - Multi-query RRF fusion -> top 20 seeds
- **Graph Traversal**: 0 edges, 0 steps -> emits `no_graph_expansion` warning
- **Reranker**: Cross-encoder reranks the 20 seeds -> final top 8 hits.

### 2.2 ARM S20 — Seed-Equivalent Direct Path
- **Original Query**: `top_k = 8`, `candidate_k = 40`
- **Direct Seed Retrieval**: `top_k = 20`, `candidate_k = 40`, `requested_strategy = HYBRID`
  - Sparse (BM25) branch depth = 40
  - Dense (E5) branch depth = 40
  - Multi-query RRF fusion -> top 20 hits
- **NO Graph Traversal**: Direct handoff to `RerankingRetriever.rerank_candidates()`.
- **Reranker**: Cross-encoder reranks exactly those 20 candidates -> final top 8 hits.

### 2.3 ARM H40 — Diagnostic Normal Hybrid-Rerank Path
- **Original Query**: `top_k = 8`, `candidate_k = 40`
- **Standard Candidate Retrieval**: `top_k = 40`, `candidate_k = 40`, `requested_strategy = HYBRID`
  - Sparse (BM25) branch depth = 40
  - Dense (E5) branch depth = 40
  - Multi-query RRF fusion -> top 40 candidates
- **Reranker**: Cross-encoder reranks all 40 candidates -> final top 8 hits.
- **Role**: Diagnostic comparison against S20 only. **H40 does NOT affect the graph-redundancy verdict.**

> [!IMPORTANT]
> **CRITICAL INVARIANT: S20 IS NOT `candidate_k = 20`.**
> A standard `candidate_k = 20` query reduces both the branch retrieval depth (to 20) and the RRF output (to 20).
> ARM S20 strictly preserves branch candidate depth = 40 while limiting the hybrid output / reranker input to 20, matching the exact graph seed pipeline.

---

## 3. Pre-Registered Decision Gate

The evaluation script mechanically evaluates one of three mutually exclusive verdicts:

| Verdict | Condition | Downstream Authority |
| :--- | :--- | :--- |
| **`GRAPH_REDUNDANCY_PROVEN`** | ALL 22 cases match exactly: seed chunk IDs, final top-8 chunk IDs, final document IDs, and reranker scores within tolerance $\le 10^{-6}$, with 0 graph steps and 1 seed call. | **Authorizes Phase B1B structural graph removal design.** Does NOT authorize switching traffic to H40. |
| **`GRAPH_REDUNDANCY_NOT_PROVEN`** | Experiment is valid, but $\ge 1$ case differs in seed sequence, final top-8 chunk sequence, document sequence, or score delta $> 10^{-6}$. | **Phase B1B remains blocked.** Current graph path must be retained. |
| **`INVALID_EXPERIMENT`** | Any hard failure: case count $\ne 22$, SHA mismatch, graph edges $> 0$, graph steps $> 0$, hybrid calls $\ne 1$, missing results, retrieval model error, or LLM generation loaded/invoked. | **Invalid.** Must repair infrastructure failure without altering experiment semantics and rerun cleanly. |

---

## 4. Kaggle Execution Runbook (A2-K1 to A2-K7)

This experiment is **retrieval-only**. No Qwen weights are downloaded, loaded, or executed. A single GPU (e.g. Kaggle T4 x1) is sufficient.

### Cell A2-K1 — Environment & Commit Verification

```bash
%%bash
set -euo pipefail

export PYTHONUNBUFFERED=1
nvidia-smi

REVIEWED_COMMIT_SHA="REPLACE_WITH_REVIEWED_B1A2_COMMIT_SHA"

rm -rf legal-agentic-rag
git clone https://github.com/Chef221/legal-agentic-rag.git
cd legal-agentic-rag

git checkout "$REVIEWED_COMMIT_SHA"
ACTUAL_COMMIT_SHA="$(git rev-parse HEAD)"
echo "Verified Execution Commit: $ACTUAL_COMMIT_SHA"

test "$ACTUAL_COMMIT_SHA" = "$REVIEWED_COMMIT_SHA"

pip uninstall -y torchao || true

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
assert torch.cuda.is_available(), 'CUDA is required for embedding and reranking'
"
```

---

### Cell A2-K2 — Identity-Based Input & Serving Discovery

```python
import hashlib
import json
from pathlib import Path

CANONICAL_DEV_SHA = "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"

# 1. Discover canonical development.json uniquely
dev_candidates = list(Path("/kaggle/input").rglob("development.json"))
matching_devs = []
for cand in dev_candidates:
    if cand.is_file() and hashlib.sha256(cand.read_bytes()).hexdigest() == CANONICAL_DEV_SHA:
        matching_devs.append(cand)

assert len(matching_devs) == 1, (
    f"Expected exactly 1 canonical development.json matching SHA {CANONICAL_DEV_SHA}, "
    f"found {len(matching_devs)}: {matching_devs}"
)
dev_path = matching_devs[0]
print(f"Found unique canonical development.json at: {dev_path}")

# 2. Discover serving artifact root uniquely satisfying Phase-A contract
val_candidates = list(Path("/kaggle/input").rglob("build_validation_full_corpus.json"))
valid_serving_roots = []
discovered_graph_metadata = {}

for val_file in val_candidates:
    root = val_file.parent
    if not all((root / sub).is_dir() for sub in ["legal_chunks", "bm25", "vector", "graph"]):
        continue
    try:
        report = json.loads(val_file.read_text(encoding="utf-8"))
    except Exception:
        continue
    dataset_manifest = report.get("dataset_manifest")
    if (
        report.get("is_valid") is True
        and report.get("is_full_corpus") is True
        and isinstance(dataset_manifest, dict)
        and dataset_manifest.get("dataset_name") == "uit-dsc-2026-task2-selected-contexts"
    ):
        graph_man_path = root / "graph" / "manifest.json"
        if graph_man_path.exists():
            g_man = json.loads(graph_man_path.read_text(encoding="utf-8"))
            if g_man.get("record_count") == 0:
                valid_serving_roots.append(root)
                discovered_graph_metadata = g_man.get("metadata", {})

assert len(valid_serving_roots) == 1, (
    f"Expected exactly 1 valid serving artifact root with 0 graph records, "
    f"found {len(valid_serving_roots)}: {valid_serving_roots}"
)
serving_root = valid_serving_roots[0]
print(f"Found unique validated serving artifact root at: {serving_root}")
print(f"Graph artifact record_count: 0, edge_count: {discovered_graph_metadata.get('edge_count')}")
```

---

### Cell A2-K3 — Materialize Exact 22-Question Benchmark

```python
import subprocess
import json
from pathlib import Path

work_dir = Path("/kaggle/working")
mat_output = work_dir / "phase_b1a2_cases_22.json"
ident_output = work_dir / "phase_b1a2_cases_22_identity.json"

cmd = [
    "python", "legal-agentic-rag/scripts/phase_b1a2_graph_equivalence.py", "prepare",
    "--development", str(dev_path),
    "--manifest", "legal-agentic-rag/configs/phase-b1a-graph-routing-cases.json",
    "--output", str(mat_output),
    "--identity-output", str(ident_output),
]

res = subprocess.run(cmd, capture_output=True, text=True, check=True)
print(res.stdout)

identity = json.loads(ident_output.read_text(encoding="utf-8"))
print("Materialized cases SHA-256:", identity["materialized_case_sha256"])
print("Question count:", identity["materialized_case_count"])
assert identity["materialized_case_count"] == 22
```

---

### Cell A2-K4 — Runtime Configuration Freeze

```python
import json
from pathlib import Path
from legal_agentic_rag.configuration import ApplicationConfig

base_example = Path("legal-agentic-rag/configs/phase-a-current-system-census-kaggle.example.json")
cfg = json.loads(base_example.read_text(encoding="utf-8"))

# Point artifacts to discovered Kaggle serving root
cfg["artifacts"]["root_path"] = str(serving_root)

# Set runtime devices to single GPU ("cuda") for retrieval components
cfg["online"]["vector_runtime"]["search_device"] = "cuda"
cfg["offline"]["embedding"]["device"] = "cuda"
cfg["online"]["reranker"]["device"] = "cuda"

# Strictly validate final raw configuration through ApplicationConfig schema
app_cfg = ApplicationConfig.model_validate(cfg)
assert app_cfg.online.vector_runtime.search_device == "cuda"
assert app_cfg.online.retrieval.top_k == 8
assert app_cfg.online.retrieval.candidate_k == 40
assert app_cfg.online.retrieval.graph_seed_chunk_k == 20
assert app_cfg.online.query_understanding.multi_query_enabled is True

runtime_cfg_path = work_dir / "runtime_config_b1a2.json"
runtime_cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"Runtime configuration validated and written to: {runtime_cfg_path}")
```

---

### Cell A2-K5 — Retrieval-Only Execution & Protocol Verification

```python
import subprocess
import json
from pathlib import Path

raw_results_path = work_dir / "phase_b1a2_retrieval_results.jsonl"
summary_path = work_dir / "phase_b1a2_run_summary.json"

cmd = [
    "python", "legal-agentic-rag/scripts/phase_b1a2_graph_equivalence.py", "run",
    "--config", str(runtime_cfg_path),
    "--questions", str(mat_output),
    "--output", str(raw_results_path),
    "--summary-output", str(summary_path),
]

res = subprocess.run(cmd, capture_output=False, text=True, check=True)

# Load run summary and verify protocol fidelity before running analysis
summary = json.loads(summary_path.read_text(encoding="utf-8"))
counts = summary["aggregate_protocol_counts"]

print("\n" + "=" * 60)
print("PHASE B1A.2 EXECUTION SUMMARY")
print("=" * 60)
print(f"Total Cases Completed:        {summary['case_count']} / 22")
print(f"Graph Zero-Record Cases:      {counts['graph_zero_record_cases']} / 22")
print(f"Graph Zero-Step Cases:        {counts['graph_zero_step_cases']} / 22")
print(f"Graph Single-Seed Calls:      {counts['graph_single_seed_call_cases']} / 22")
print(f"G Seed Invariant Passes:      {counts['g_seed_invariant_pass_count']} / 22")
print(f"S20 Seed Invariant Passes:    {counts['s20_invariant_pass_count']} / 22")
print(f"Artifact Lineage Status:      PASS (startup validated)")
print(f"LLM Generation Path:          NOT LOADED / NOT CALLED (retrieval-only)")
print("=" * 60)

assert summary["case_count"] == 22
assert counts["graph_zero_step_cases"] == 22
assert counts["graph_single_seed_call_cases"] == 22
assert counts["g_seed_invariant_pass_count"] == 22
assert counts["s20_invariant_pass_count"] == 22
```

---

### Cell A2-K6 — Analysis & Verdict Gate

```python
import subprocess
import json
from pathlib import Path

report_path = work_dir / "phase_b1a2_graph_equivalence_report.json"
decision_path = work_dir / "phase_b1a2_decision_report.json"
case_metrics_path = work_dir / "phase_b1a2_case_metrics.jsonl"

cmd = [
    "python", "legal-agentic-rag/scripts/phase_b1a2_graph_equivalence.py", "analyze",
    "--results", str(raw_results_path),
    "--manifest", "legal-agentic-rag/configs/phase-b1a-graph-routing-cases.json",
    "--output-report", str(report_path),
    "--output-decision", str(decision_path),
    "--output-case-metrics", str(case_metrics_path),
]

res = subprocess.run(cmd, capture_output=True, text=True, check=True)
print(res.stdout)

decision = json.loads(decision_path.read_text(encoding="utf-8"))
report = json.loads(report_path.read_text(encoding="utf-8"))

print("\n" + "=" * 60)
print(f"VERDICT: {decision['verdict']}")
print("=" * 60)
print(f"B1B Design Authorized: {decision['b1b_design_authorized']}")
print("\nGRAPH vs S20 Equivalence:")
for k, v in report["g_vs_s20_equivalence"].items():
    print(f"  {k}: {v}")

print("\nS20 vs H40 Candidate-Pool Diagnostics:")
for k, v in report["s20_vs_h40_candidate_pool_diagnostics"].items():
    print(f"  {k}: {v}")

print("\nReasons:")
for r in decision["reasons"]:
    print(f"  - {r}")
print("=" * 60)
```

---

### Cell A2-K7 — Evidence Packaging

```python
import subprocess
import json
from pathlib import Path

zip_path = work_dir / "phase-b1a2-graph-equivalence-evidence.zip"

cmd = [
    "python", "legal-agentic-rag/scripts/phase_b1a2_graph_equivalence.py", "package",
    "--output-zip", str(zip_path),
    "--manifest", "legal-agentic-rag/configs/phase-b1a-graph-routing-cases.json",
    "--questions-identity", str(ident_output),
    "--runtime-config", str(runtime_cfg_path),
    "--run-summary", str(summary_path),
    "--results", str(raw_results_path),
    "--report", str(report_path),
    "--decision", str(decision_path),
    "--case-metrics", str(case_metrics_path),
]

res = subprocess.run(cmd, capture_output=True, text=True, check=True)
pkg_res = json.loads(res.stdout)

print("\n" + "=" * 60)
print("EVIDENCE PACKAGE READY")
print("=" * 60)
print("ZIP Path:      ", pkg_res["zip_path"])
print("ZIP SHA-256:   ", pkg_res["zip_sha256"])
print("ZIP Size Bytes:", pkg_res["zip_size_bytes"])
print("=" * 60)
print(">>> DOWNLOAD BEFORE ENDING KAGGLE SESSION <<<")
```
