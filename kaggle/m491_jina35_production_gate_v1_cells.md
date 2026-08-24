# M49.1-JINA35 — Production Gate V1 Kaggle Runner Cells
**Authority Bundle:** `m491_jina35_production_gate_v1.zip`  
**Source Commit:** `90de9a9d813df87432bc9183f8edebd4ed1f0b24`  
**Control Baseline:** `10681c8c05008432cd1c7170cd3917f4317c1c69`  
**Target Architecture:** Kaggle Notebook (Single T4 GPU, 16GB VRAM, offline)

---

### CELL 1 — Environment + Input Discovery
```python
import os
import sys
import subprocess
import hashlib
from pathlib import Path
import torch

print("=== CELL 1: ENVIRONMENT & INPUT DISCOVERY ===")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024*1024):.1f} MB")
else:
    raise SystemError("CUDA GPU is required for T4 mechanical validation gates.")

# Locate execution bundle and datasets
input_dir = Path("/kaggle/input")
work_dir = Path("/kaggle/working")

def find_file(name: str) -> Path:
    matches = list(input_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Required input {name} not found under /kaggle/input")
    return matches[0]

bundle_zip = find_file("m491_jina35_production_gate_v1.zip")
print(f"Found execution bundle: {bundle_zip}")
```

---

### CELL 2 — Authority Bundle Verification
```python
import hashlib
import json
import zipfile

print("=== CELL 2: AUTHORITY BUNDLE VERIFICATION ===")

FROZEN_SHAS = {
    "clean100_shared_candidate_pools.jsonl": "45a9bd9716f14c7a5a72c54bd82f5ee17a822caa56a26a6a3998f8234e899bb0",
    "clean100_jina_reranked.jsonl": "eaafc39d9e3a5e5b11949d5546fea1b7b4da058cf56d99d463a1b2e642e337c9",
    "clean100_questions_only.json": "84e83c26357fc2a08fe183ec16b7df6ee02cc01555be9e1abd101ba3fca2d073",
    "clean100_phase1_manifest.json": "2f733ac8a2d1d5ca94c8f18844226865f598b21f4a109959daf9bef4ea3992c3",
}

with zipfile.ZipFile(bundle_zip, "r") as zf:
    manifest_bytes = zf.read("bundle_manifest.json")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    print(f"Bundle Manifest Version: {manifest.get('schema_version')}")
    print(f"Source Commit: {manifest.get('created_from_commit')}")
    print(f"Root Control: {manifest.get('root_control_commit')}")

    for fname, exp_sha in FROZEN_SHAS.items():
        data = zf.read(f"authority/{fname}")
        actual_sha = hashlib.sha256(data).hexdigest()
        assert actual_sha == exp_sha, f"SHA mismatch for {fname}: {actual_sha} != {exp_sha}"
        print(f"  ✓ {fname}: SHA verified")

    for info in zf.infolist():
        name_l = info.filename.lower()
        for forbidden in ["train.json", "official_scorer", "reference_answer", "gold_answer", "phase2_exact", "recovery_per_qid"]:
            assert forbidden not in name_l, f"Forbidden material in bundle: {info.filename}"

print("REFERENCE MATERIAL IN BUNDLE: NO")
print("AUTHORITY BUNDLE VERIFICATION: PASSED")
```

---

### CELL 3 — Extract Source
```python
import zipfile

print("=== CELL 3: EXTRACT SOURCE ===")
target_dir = Path("/kaggle/working/m491_jina35_runtime")
target_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(bundle_zip, "r") as zf:
    zf.extractall(target_dir)

src_path = str(target_dir / "source" / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"Extracted execution bundle to {target_dir}")
print(f"Added {src_path} to sys.path")
```

---

### CELL 4 — Offline Model-Cache Verification
```python
print("=== CELL 4: OFFLINE MODEL CACHE VERIFICATION ===")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Identify model cache paths
# Expecting Kaggle datasets: uit-dsc-2026-task2-m45-artifacts, m491_clean100_model_cache / huggingface cache
m45_dir = list(input_dir.rglob("uit-dsc-2026-task2-m45-artifacts"))
if m45_dir:
    print(f"M45 Artifacts: {m45_dir[0]}")

hf_caches = list(input_dir.rglob("hub"))
if hf_caches:
    os.environ["HF_HOME"] = str(hf_caches[0].parent)
    print(f"Set HF_HOME to: {os.environ['HF_HOME']}")

print("Offline environment configured (zero external HTTP permitted).")
```

---

### CELL 5 — Parameter-Budget Preflight
```python
print("=== CELL 5: PARAMETER-BUDGET PREFLIGHT ===")
budget_file = target_dir / "docs" / "m491-jina35-parameter-budget-authority.json"
budget = json.loads(budget_file.read_text(encoding="utf-8"))

print(f"Competition Limit: {budget['competition_limit']:,} parameters")
for comp in budget["components"]:
    print(f"  - {comp['role']}: {comp['model']} -> {comp['exact_parameters']:,} params")

print(f"Exact Candidate Total: {budget['exact_total_parameters']:,} parameters")
print(f"Headroom: {budget['headroom_parameters']:,} parameters ({budget['headroom_percentage']:.2f}%)")

assert budget["exact_total_parameters"] == 3311750784, "Candidate total mismatch"
assert budget["exact_total_parameters"] < 4000000000, "Competition parameter cap exceeded"
print("PARAMETER BUDGET PREFLIGHT: PASSED")
```

---

### CELL 6 — Gate A Launch (Mechanical Parity on 100 QIDs)
```python
print("=== CELL 6: GATE A LAUNCH (MECHANICAL PARITY) ===")
output_dir = Path("/kaggle/working/reports/mechanical_validation")
output_dir.mkdir(parents=True, exist_ok=True)
log_path = output_dir / "gate_a_execution.log"

cmd = [
    sys.executable,
    str(target_dir / "scripts" / "m491_jina35_mechanical_validation.py"),
    "--gate", "A",
    "--authority-dir", str(target_dir / "authority"),
    "--output-dir", str(output_dir),
    "--log-path", str(log_path),
    "--device", "cuda:0",
]

print(f"Executing: {' '.join(cmd)}")
res = subprocess.run(cmd, check=False)
if res.returncode != 0:
    print(f"Gate A process failed with exit code {res.returncode}")
```

---

### CELL 7 — Gate A Result Audit
```python
print("=== CELL 7: GATE A RESULT AUDIT ===")
report_a_file = output_dir / "gate_a_parity_report.json"
if not report_a_file.exists():
    raise FileNotFoundError(f"Gate A report missing at {report_a_file}")

report_a = json.loads(report_a_file.read_text(encoding="utf-8"))
print(f"Gate A Status: {report_a['status']}")
print(f"Total QIDs: {report_a['total_qids']}/100")
print(f"Top-1 Exact Matches: {report_a['top1_exact']}/100")
print(f"Top-10 Ordered Matches: {report_a['top10_ordered_exact']}/100")
print(f"Full-K (40) Ordered Matches: {report_a['full_k_ordered_exact']}/100")
print(f"Max Absolute Score Diff: {report_a['max_abs_score_diff']:.6f} (Tolerance: {report_a['numerical_tolerance']})")

assert report_a["status"] == "GATE_A_PASSED", "GATE A FAILED. STOPPING EXECUTION."
assert report_a["top1_exact"] == 100
assert report_a["top10_ordered_exact"] == 100
assert report_a["full_k_ordered_exact"] == 100
assert report_a["max_abs_score_diff"] <= 1e-3

print(">>> GATE A RESULT: PASSED <<<")
```

---

### CELL 8 — GPU Cleanup / Memory Sanity
```python
import gc
import torch

print("=== CELL 8: GPU CLEANUP & MEMORY SANITY ===")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated() / (1024*1024)
    reserved = torch.cuda.memory_reserved() / (1024*1024)
    print(f"Post-Gate A GPU Memory: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB")
```

---

### CELL 9 — Gate B Launch (Full M49.1 T4 Runtime Smoke)
```python
print("=== CELL 9: GATE B LAUNCH (T4 RUNTIME SMOKE) ===")
log_b_path = output_dir / "gate_b_execution.log"

cmd_b = [
    sys.executable,
    str(target_dir / "scripts" / "m491_jina35_mechanical_validation.py"),
    "--gate", "B",
    "--config", str(target_dir / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"),
    "--questions", str(target_dir / "authority" / "clean100_questions_only.json"),
    "--output-dir", str(output_dir),
    "--log-path", str(log_b_path),
    "--device", "cuda:0",
    "--max-gate-b-questions", "5",
]

print(f"Executing: {' '.join(cmd_b)}")
res_b = subprocess.run(cmd_b, check=False)
if res_b.returncode != 0:
    print(f"Gate B process failed with exit code {res_b.returncode}")
```

---

### CELL 10 — Gate B Result Audit
```python
print("=== CELL 10: GATE B RESULT AUDIT ===")
report_b_file = output_dir / "gate_b_t4_smoke_report.json"
if not report_b_file.exists():
    raise FileNotFoundError(f"Gate B report missing at {report_b_file}")

report_b = json.loads(report_b_file.read_text(encoding="utf-8"))
print(f"Gate B Status: {report_b['status']}")
print(f"Successful Questions: {report_b['successful_questions']}/{report_b['total_questions']}")
print(f"Startup VRAM: {report_b['vram_startup_mb']:.1f} MB")
print(f"Peak VRAM: {report_b['vram_peak_mb']:.1f} MB")
print(f"Runtime Total Parameters: {report_b['runtime_identity']['actual_total_model_parameters']:,}")
print(f"Compliance Status: {report_b['runtime_identity']['compliance_status']}")

assert report_b["status"] == "GATE_B_PASSED", "GATE B FAILED."
assert report_b["successful_questions"] == report_b["total_questions"]
assert report_b["runtime_identity"]["actual_total_model_parameters"] < 4000000000

print(">>> GATE B RESULT: PASSED <<<")
```

---

### CELL 11 — Final Evidence Freeze
```python
import zipfile

print("=== CELL 11: FINAL EVIDENCE FREEZE ===")
results_zip = work_dir / "m491_jina35_production_gate_v1_results.zip"

with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in output_dir.rglob("*.*"):
        rel = p.relative_to(output_dir)
        zf.write(p, arcname=f"reports/{rel.as_posix()}")

zip_size = results_zip.stat().st_size
zip_sha = hashlib.sha256(results_zip.read_bytes()).hexdigest()
print(f"Created Final Results Package: {results_zip}")
print(f"Size: {zip_size:,} bytes")
print(f"SHA-256: {zip_sha}")
```
