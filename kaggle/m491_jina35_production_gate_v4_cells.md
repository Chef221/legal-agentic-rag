# M49.1-JINA35 — Production Gate V4 Kaggle Runner Cells
**Authority Bundle:** `m491_jina35_production_gate_v4.zip`  
**Target Environment:** Kaggle Notebook (Single T4 GPU, 16GB VRAM, Internet OFF)  
**Required Pre-requisites:**
- Offline Wheelhouse dataset containing `transformers==5.15.0`, `safetensors==0.8.0`, `accelerate==1.14.0`, `tokenizers==0.22.2`, `huggingface-hub==1.11.0`
- M45 Database dataset
- M49 Merged Generator dataset (`/kaggle/input/m49-generator-merged`)
- Offline HuggingFace Cache dataset (Qwen embedding + Jina v3.5)

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

bundle_zip = find_file("m491_jina35_production_gate_v4.zip")
print(f"Found execution bundle: {bundle_zip}")
```

---

### CELL 2 — Offline Wheelhouse Installation (Strict Internet OFF)
```python
import subprocess
import sys

print("=== CELL 2: OFFLINE RUNTIME WHEELHOUSE INSTALLATION ===")

# Find offline wheelhouse directory
wheelhouse_candidates = list(Path("/kaggle/input").rglob("*.whl"))
if not wheelhouse_candidates:
    print("WARNING: No .whl files found in /kaggle/input. Assuming pre-installed 5.15.0 environment.")
else:
    whl_dir = wheelhouse_candidates[0].parent
    print(f"Installing wheels from offline wheelhouse: {whl_dir}")
    cmd = [
        sys.executable, "-m", "pip", "install", "--no-index",
        f"--find-links={whl_dir}",
        "transformers==5.15.0",
        "safetensors==0.8.0",
        "accelerate==1.14.0",
        "tokenizers==0.22.2",
        "huggingface-hub==1.11.0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR:", res.stderr)
        raise RuntimeError("Offline wheelhouse installation failed")

import transformers
print(f"Active Transformers Version: {transformers.__version__}")
assert transformers.__version__ == "5.15.0", f"Expected transformers==5.15.0, got {transformers.__version__}"
print("Dependency authority verified: transformers==5.15.0")
```

---

### CELL 3 — Preflight Generator & AutoModel Architecture Verification
```python
from transformers import AutoConfig, AutoModelForImageTextToText
from pathlib import Path

print("=== CELL 3: GENERATOR ARCHITECTURE PREFLIGHT ===")

gen_candidates = list(Path("/kaggle/input").rglob("m49-generator-merged")) + list(Path("/kaggle/working").rglob("m49-generator-merged"))
assert gen_candidates, "m49-generator-merged not found in /kaggle/input or /kaggle/working"
gen_path = gen_candidates[0]
print(f"Inspecting generator at: {gen_path}")

cfg = AutoConfig.from_pretrained(str(gen_path))
print(f"AutoConfig model_type: {cfg.model_type}")
print(f"AutoConfig architectures: {cfg.architectures}")
assert cfg.model_type == "qwen3_5", f"Expected model_type qwen3_5, got {cfg.model_type}"
assert "Qwen3_5ForConditionalGeneration" in cfg.architectures

print("Generator architecture validated: Qwen3_5ForConditionalGeneration")
```

---

### CELL 4 — Authority Bundle Verification & Extraction
```python
import hashlib
import json
import zipfile

print("=== CELL 4: AUTHORITY BUNDLE VERIFICATION & EXTRACTION ===")

FROZEN_SHAS = {
    "clean100_shared_candidate_pools.jsonl": "45a9bd9716f14c7a5a72c54bd82f5ee17a822caa56a26a6a3998f8234e899bb0",
    "clean100_jina_reranked.jsonl": "eaafc39d9e3a5e5b11949d5546fea1b7b4da058cf56d99d463a1b2e642e337c9",
    "clean100_questions_only.json": "84e83c26357fc2a08fe183ec16b7df6ee02cc01555be9e1abd101ba3fca2d073",
    "clean100_phase1_manifest.json": "2f733ac8a2d1d5ca94c8f18844226865f598b21f4a109959daf9bef4ea3992c3",
}

extract_dir = Path("/kaggle/working/m491_runtime")
extract_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"Verified authority/{fname}: {actual_sha[:16]}... OK")

    zf.extractall(extract_dir)

print(f"Extracted execution bundle to {extract_dir}")
```

---

### CELL 5 — Environment Variables & HuggingFace Offline Cache
```python
import os
import sys

print("=== CELL 5: ENVIRONMENT & PATH CONFIGURATION ===")

src_dir = extract_dir / "source" / "src"
scripts_dir = extract_dir / "scripts"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(extract_dir))

# Configure HuggingFace Cache
hf_cache_candidates = list(Path("/kaggle/input").rglob("hub"))
if hf_cache_candidates:
    hf_root = hf_cache_candidates[0].parent
    os.environ["HF_HOME"] = str(hf_root)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_root)
    print(f"Configured HF_HOME to: {hf_root}")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
print("Configured HuggingFace strict offline mode.")
```

---

### CELL 6 — Generator Tree SHA Verification
```python
import hashlib
from pathlib import Path

print("=== CELL 6: GENERATOR TREE SHA VERIFICATION ===")

gen_dir = Path("/kaggle/working/m49-generator-merged")
if not gen_dir.exists():
    import shutil
    shutil.copytree(str(gen_path), str(gen_dir))
    print(f"Staged generator to: {gen_dir}")

def compute_dir_tree_sha256(directory: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(directory).as_posix()
            file_bytes = file_path.read_bytes()
            file_sha = hashlib.sha256(file_bytes).hexdigest()
            hasher.update(f"{rel_path}:{file_sha}
".encode("utf-8"))
    return hasher.hexdigest()

gen_tree_sha = compute_dir_tree_sha256(gen_dir)
print(f"Generator Tree SHA256: {gen_tree_sha}")
EXPECTED_GEN_TREE_SHA = "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b"
assert gen_tree_sha == EXPECTED_GEN_TREE_SHA, f"Generator tree SHA mismatch: {gen_tree_sha} != {EXPECTED_GEN_TREE_SHA}"
print("Generator immutable tree SHA verified OK.")
```

---

### CELL 7 — Exact Parameter Budget Compliance Preflight
```python
import json

print("=== CELL 7: EXACT PARAMETER BUDGET AUDIT ===")

param_auth_file = extract_dir / "docs" / "artifacts" / "m491-jina35-parameter-budget-authority.json"
budget = json.loads(param_auth_file.read_text(encoding="utf-8"))

exact_accounting = budget["independently_proven_exact_accounting"]
exact_embed = exact_accounting["embedding"]["proven_exact_numel"]
exact_jina = exact_accounting["candidate_reranker"]["proven_exact_numel"]
exact_gen = exact_accounting["generator_merged"]["proven_exact_numel"]
exact_total = exact_accounting["exact_active_learned_total"]
headroom = exact_accounting["exact_headroom"]

print(f"Exact Embedding Parameters: {exact_embed:,}")
print(f"Exact Jina Parameters:      {exact_jina:,}")
print(f"Exact Generator Parameters: {exact_gen:,}")
print(f"Exact Active Learned Total: {exact_total:,}")
print(f"Headroom under 4B Cap:      {headroom:,} ({exact_accounting['exact_headroom_percentage']:.4f}%)")

assert exact_embed == 595776512
assert exact_jina == 596836352
assert exact_gen == 2213241664
assert exact_total == 3405854528
assert exact_total < 4000000000, "EXCEEDS 4B PARAMETER CAP"
assert budget["historical_registered_accounting"]["historical_candidate_total"] == 3311750784

print("Parameter budget compliance: STRICT PASS (< 4,000,000,000)")
```

---

### CELL 8 — GATE A: Strict Mechanical Parity Verification (Clean100)
```python
from m491_jina35_mechanical_validation import run_gate_a_parity
from pathlib import Path

print("=== CELL 8: GATE A MECHANICAL PARITY RUN ===")

out_dir_a = Path("/kaggle/working/gate_a_results")
authority_dir = extract_dir / "authority"

report_a = run_gate_a_parity(
    authority_dir=authority_dir,
    output_dir=out_dir_a,
    device="cuda:0",
)

print(f"Gate A Status: {report_a['status']}")
print(f"Top-1 Matches: {report_a['top1_exact_matches']}/100")
print(f"Top-10 Matches: {report_a['top10_ordered_matches']}/100")
print(f"Full-K Matches: {report_a['full_k_ordered_matches']}/100")
print(f"Max Absolute Score Diff: {report_a['max_abs_score_diff']:.6f}")

assert report_a["status"] == "GATE_A_PASSED", "GATE A FAILED."
assert report_a["top1_exact_matches"] == 100
assert report_a["top10_ordered_matches"] == 100
assert report_a["full_k_ordered_matches"] == 100
assert report_a["max_abs_score_diff"] <= 0.001
print("GATE A MECHANICAL PARITY: PASSED (100/100 EXACT MATCHES)")
```

---

### CELL 9 — GATE B: Strict Full-Runtime T4 Coexistence Smoke (5 Questions)
```python
from m491_jina35_mechanical_validation import run_gate_b_smoke
from pathlib import Path

print("=== CELL 9: GATE B FULL RUNTIME T4 SMOKE ===")

config_path = extract_dir / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"
questions_path = extract_dir / "authority" / "clean100_questions_only.json"
out_dir_b = Path("/kaggle/working/gate_b_results")

report_b = run_gate_b_smoke(
    config_path=config_path,
    questions_path=questions_path,
    output_dir=out_dir_b,
    device="cuda:0",
    max_questions=5,
)

print(f"Gate B Status: {report_b['status']}")
print(f"Total Questions: {report_b['total_questions']}")
print(f"Strict Successful Questions: {report_b['strict_successful_questions']}")
print(f"Generation Successful Questions: {report_b['generation_successful_questions']}")
print(f"Verified Answer Questions: {report_b['verified_answer_successful_questions']}")
print(f"Peak VRAM: {report_b['vram_peak_mb']:.1f} MB")
print(f"Total Execution Latency: {report_b['latency_total_seconds']:.2f} s")

for e in report_b["executions"]:
    print(f"QID {e['qid']}: success={e['success']} | stop={e['stop_reason']} | "
          f"ans_len={e['answer_length']} | ev_count={e['selected_evidence_count']} | "
          f"insufficient_ev={e['insufficient_evidence']} | latency={e['latency_seconds']:.2f}s")

assert report_b["status"] == "GATE_B_PASSED", "GATE B FAILED."
assert report_b["strict_successful_questions"] == 5
assert report_b["generation_successful_questions"] == 5
assert all(e["insufficient_evidence"] is False for e in report_b["executions"])
print("GATE B FULL-RUNTIME T4 SMOKE: STRICT PASS (5/5 VERIFIED ANSWERS)")
```

---

### CELL 10 — Package Evidence & Output Results
```python
import zipfile
from pathlib import Path

print("=== CELL 10: PACKAGING VERIFICATION RESULTS ===")

results_zip = Path("/kaggle/working/m491_jina35_production_gate_v4_results.zip")
with zipfile.ZipFile(results_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for out_file in list(Path("/kaggle/working/gate_a_results").rglob("*")) + list(Path("/kaggle/working/gate_b_results").rglob("*")):
        if out_file.is_file():
            zf.write(out_file, arcname=out_file.relative_to("/kaggle/working").as_posix())

print(f"Saved results package: {results_zip} ({results_zip.stat().st_size:,} bytes)")
```
