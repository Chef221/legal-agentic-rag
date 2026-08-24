# M49.1-JINA35 — Production Gate V4 Kaggle Runner Cells
**Authority Bundle:** `m491_jina35_production_gate_v4.zip`
**Target Environment:** Kaggle Notebook (Single Tesla T4 GPU, 16GB VRAM, Internet OFF)
**Required Attached Datasets / Inputs:**
- `offline-wheelhouse` (containing `transformers-5.15.0-*.whl`, `safetensors-0.8.0-*.whl`, `accelerate-1.14.0-*.whl`, `tokenizers-0.22.2-*.whl`, `huggingface_hub-1.11.0-*.whl`)
- `m49-generator-merged` (Merged Qwen3.5 generator weights, tree SHA `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b`)
- `huggingface-cache` (Offline Hub cache containing `Qwen/Qwen3-Embedding-0.6B` and `jinaai/jina-reranker-v3.5`)
- `m45-database` / unified corpus database
- `m491_jina35_production_gate_v4.zip` (Execution code bundle)
- `public-official.json` (Ban To Chuc official Public-1000 questions)
- *(Optional for Session 2+)* `public1000_checkpoint_latest.zip` or `.zip.bin` from previous session

---

### CELL 1 — Environment & Hardware Discovery
```python
import os
import sys
import subprocess
import hashlib
from pathlib import Path
import torch

print("=== CELL 1: ENVIRONMENT & HARDWARE DISCOVERY ===")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024*1024):.1f} MB")
else:
    raise SystemError("CUDA GPU is required for T4 mechanical validation and Public-1000 execution.")

input_dir = Path("/kaggle/input")
work_dir = Path("/kaggle/working")
print("Input datasets found under /kaggle/input:")
for d in input_dir.iterdir():
    if d.is_dir():
        print(f" - {d.name}")
```

---

### CELL 2 — Offline Wheelhouse Installation (Strict Internet OFF)
```python
import subprocess
import sys
from pathlib import Path

print("=== CELL 2: OFFLINE RUNTIME WHEELHOUSE INSTALLATION ===")

wheelhouse_candidates = list(Path("/kaggle/input").rglob("*.whl"))
if not wheelhouse_candidates:
    print("WARNING: No .whl files found in /kaggle/input. Checking active transformers version...")
else:
    whl_dir = wheelhouse_candidates[0].parent
    print(f"Installing pinned wheels from offline wheelhouse: {whl_dir}")
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
        raise RuntimeError(f"Offline wheelhouse installation failed: {res.stderr}")

import transformers
print(f"Active Transformers Version: {transformers.__version__}")
assert transformers.__version__ == "5.15.0", f"Expected transformers==5.15.0, got {transformers.__version__}"
print("Dependency authority verified: transformers==5.15.0")
```

---

### CELL 3 — Post-Install Kernel Re-entry & Environment Paths Configuration
```python
import os
import sys
from pathlib import Path
import transformers

print("=== CELL 3: KERNEL RE-ENTRY & PATHS CONFIGURATION ===")
print(f"Transformers version: {transformers.__version__}")
assert transformers.__version__ == "5.15.0", "Kernel must be running transformers==5.15.0"

extract_dir = Path("/kaggle/working/m491_runtime")
src_dir = extract_dir / "source" / "src"
scripts_dir = extract_dir / "scripts"

for p in [str(src_dir), str(scripts_dir), str(extract_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Configure HuggingFace Cache offline
hf_cache_candidates = list(Path("/kaggle/input").rglob("hub"))
if hf_cache_candidates:
    hf_root = hf_cache_candidates[0].parent
    os.environ["HF_HOME"] = str(hf_root)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_root)
    print(f"Configured HF_HOME to: {hf_root}")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
print("Environment paths and strict offline configuration ready.")
```

---

### CELL 4 — Authority Bundle Verification & Extraction
```python
import hashlib
import json
import zipfile
from pathlib import Path

print("=== CELL 4: AUTHORITY BUNDLE VERIFICATION & EXTRACTION ===")

input_dir = Path("/kaggle/input")
bundle_candidates = list(input_dir.rglob("m491_jina35_production_gate_v4.zip*"))
assert bundle_candidates, "m491_jina35_production_gate_v4.zip not found under /kaggle/input"
bundle_zip = bundle_candidates[0]
print(f"Found execution bundle: {bundle_zip}")

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

### CELL 5 — Generator Staging & Tree SHA Verification
```python
import hashlib
import shutil
from pathlib import Path
from transformers import AutoConfig, AutoModelForImageTextToText

print("=== CELL 5: GENERATOR STAGING & TREE SHA VERIFICATION ===")

gen_candidates = list(Path("/kaggle/input").rglob("m49-generator-merged"))
assert gen_candidates, "m49-generator-merged not found in /kaggle/input"
gen_src = gen_candidates[0]

gen_target = Path("/kaggle/working/m49-generator-merged")
if not gen_target.exists():
    shutil.copytree(str(gen_src), str(gen_target))
    print(f"Staged generator to: {gen_target}")

def compute_dir_tree_sha256(directory: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(directory).as_posix()
            file_bytes = file_path.read_bytes()
            file_sha = hashlib.sha256(file_bytes).hexdigest()
            hasher.update(f"{rel_path}:{file_sha}\n".encode("utf-8"))
    return hasher.hexdigest()

gen_tree_sha = compute_dir_tree_sha256(gen_target)
print(f"Generator Tree SHA256: {gen_tree_sha}")
EXPECTED_GEN_TREE_SHA = "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b"
assert gen_tree_sha == EXPECTED_GEN_TREE_SHA, f"Generator tree SHA mismatch: {gen_tree_sha} != {EXPECTED_GEN_TREE_SHA}"

cfg = AutoConfig.from_pretrained(str(gen_target))
assert cfg.model_type == "qwen3_5", f"Expected model_type qwen3_5, got {cfg.model_type}"
assert "Qwen3_5ForConditionalGeneration" in cfg.architectures
print("Generator immutable tree SHA and architecture verified OK.")
```

---

### CELL 6 — Parameter Budget Authority Preflight
```python
import json
from pathlib import Path

print("=== CELL 6: EXACT PARAMETER BUDGET AUDIT ===")

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
print("Parameter budget compliance: STRICT PASS (< 4,000,000,000)")
```

---

### CELL 7 — GATE A: Strict Mechanical Parity Verification (Clean100)
```python
from m491_jina35_mechanical_validation import run_gate_a_parity
from pathlib import Path

print("=== CELL 7: GATE A MECHANICAL PARITY RUN ===")

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

### CELL 8 — GATE B: Strict Full-Runtime T4 Coexistence Smoke (5 Questions)
```python
from m491_jina35_mechanical_validation import run_gate_b_smoke
from pathlib import Path

print("=== CELL 8: GATE B FULL RUNTIME T4 SMOKE ===")

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
print(f"Strict Successful Questions: {report_b['strict_successful_questions']}/5")
print(f"Generation Successful Questions: {report_b['generation_successful_questions']}/5")
print(f"Verified Answer Questions: {report_b['verified_answer_successful_questions']}/5")
print(f"Peak VRAM: {report_b['vram_peak_mb']:.1f} MB")
print(f"Total Latency: {report_b['latency_total_seconds']:.2f} s")

assert report_b["status"] == "GATE_B_PASSED", "GATE B FAILED."
assert report_b["strict_successful_questions"] == 5
print("GATE B FULL-RUNTIME T4 SMOKE: STRICT PASS (5/5 VERIFIED ANSWERS)")
```

---

### CELL 9 — MULTI-SESSION PUBLIC-1000 EXECUTION CONTROLLER
```python
from pathlib import Path
from legal_agentic_rag.competition.uit_dsc_2026.public1000_session_runner import Public1000SessionRunner
from legal_agentic_rag.runtime.online import OnlineRuntimeFactory
from legal_agentic_rag.serving.config_loader import load_application_config

print("=== CELL 9: MULTI-SESSION PUBLIC-1000 EXECUTION CONTROLLER ===")

# Discover official Public-1000 questions
public_q_candidates = (
    list(Path("/kaggle/input").rglob("public-official.json"))
    + list(Path("/kaggle/input").rglob("public_official.json"))
)
assert public_q_candidates, "public-official.json not found under /kaggle/input"
public_q_path = public_q_candidates[0]
print(f"Found official public questions: {public_q_path}")

# Check for optional prior session checkpoint archive (.zip or .zip.bin)
prior_checkpoint_path = None
checkpoint_candidates = (
    list(Path("/kaggle/input").rglob("public1000_checkpoint_latest.zip*"))
    + list(Path("/kaggle/input").rglob("m491_jina35_public1000_checkpoint*"))
)
if checkpoint_candidates:
    prior_checkpoint_path = checkpoint_candidates[0]
    print(f"Found previous session checkpoint: {prior_checkpoint_path}")
else:
    print("No previous checkpoint detected. Initializing Session 1.")

# Load resolved production config
config_path = extract_dir / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"
app_cfg = load_application_config(config_path)

def runtime_builder():
    factory = OnlineRuntimeFactory.from_config(app_cfg, device="cuda:0")
    return factory.build()

public_working_dir = Path("/kaggle/working/public1000_execution")

# Initialize Public1000SessionRunner with 9.5-hour safe budget
runner = Public1000SessionRunner(
    app_config=app_cfg,
    working_dir=public_working_dir,
    questions_path=public_q_path,
    session_budget_hours=9.5,  # Stops cleanly well before 12h limit
    runtime_builder=runtime_builder,
)

# Execute session
session_audit = runner.run_session(checkpoint_archive_path=prior_checkpoint_path)
print(f"Session Status: {session_audit['status']}")
```

---

### CELL 10 — OFFICIAL CODABENCH SUBMISSION PACKAGING (Gated on 1000/1000)
```python
from pathlib import Path

print("=== CELL 10: CODABENCH SUBMISSION PACKAGING ===")

submission_out_dir = Path("/kaggle/working/submission_package")

try:
    sub_zip_path = runner.package_final_submission(submission_out_dir)
    print(f"SUCCESS: Submission zip created at: {sub_zip_path}")
    print(f"Submission zip size: {sub_zip_path.stat().st_size:,} bytes")
    print("READY TO DOWNLOAD SUBMISSION.ZIP FOR CODABENCH SUBMISSION!")
except Exception as err:
    print(f"SUBMISSION PACKAGING HELD: {err}")
    print("If more sessions are required, download 'public1000_checkpoint_latest.zip' and resume in the next session.")
```


---

### CELL 11 — GATE C: DUAL-T4 INFRASTRUCTURE & CONCURRENCY SMOKE (10 QUESTIONS)
```python
import os
import sys
import json
import time
from pathlib import Path
import torch
from legal_agentic_rag.competition.uit_dsc_2026.dual_session_runner import DualPublic1000SessionRunner
from legal_agentic_rag.runtime.online import OnlineRuntimeFactory
from legal_agentic_rag.serving.config_loader import load_application_config

print("=== CELL 11: GATE C — DUAL-T4 CONCURRENCY SMOKE VALIDATION ===")

# Verify Dual-GPU hardware is present
assert torch.cuda.is_available(), "CUDA is not available"
gpu_count = torch.cuda.device_count()
print(f"Detected {gpu_count} CUDA GPUs:")
for idx in range(gpu_count):
    print(f" - GPU {idx}: {torch.cuda.get_device_name(idx)} ({torch.cuda.get_device_properties(idx).total_memory / (1024*1024):.0f} MB VRAM)")
assert gpu_count >= 2, f"Gate C requires >= 2 CUDA GPUs for dual-T4 validation, found {gpu_count}"

# Load Clean100 questions-only authority and extract exactly first 10 questions
clean100_qfile = extract_dir / "authority" / "clean100_questions_only.json"
assert clean100_qfile.exists(), f"Authority clean100_questions_only.json missing: {clean100_qfile}"
all_clean_q = json.loads(clean100_qfile.read_text(encoding="utf-8"))

gate_c_qids = list(all_clean_q.keys())[:10]
print(f"Gate C 10 Deterministic QIDs: {gate_c_qids}")
gate_c_qdict = {qid: all_clean_q[qid] for qid in gate_c_qids}

gate_c_qpath = Path("/kaggle/working/gate_c_10_questions.json")
gate_c_qpath.write_text(json.dumps(gate_c_qdict, indent=2, ensure_ascii=False), encoding="utf-8")

config_path = extract_dir / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"
app_cfg = load_application_config(config_path)

# Build runtime builders for Worker 0 and Worker 1
def _build_worker_runtime_0():
    factory = OnlineRuntimeFactory.from_config(app_cfg, device="cuda:0")
    return factory.build()

def _build_worker_runtime_1():
    factory = OnlineRuntimeFactory.from_config(app_cfg, device="cuda:0")
    return factory.build()

gate_c_work_dir = Path("/kaggle/working/gate_c_dual_t4_smoke")

gate_c_runner = DualPublic1000SessionRunner(
    app_config=app_cfg,
    working_dir=gate_c_work_dir,
    questions_path=gate_c_qpath,
    session_budget_hours=1.0,
    session_id="gate_c_smoke_run",
    runtime_builders={0: _build_worker_runtime_0, 1: _build_worker_runtime_1},
    worker_devices=("0", "1"),
)

start_t = time.perf_counter()
gate_c_audit = gate_c_runner.run_session()
elapsed_t = time.perf_counter() - start_t

print("=" * 80)
print(" GATE C VALIDATION SUMMARY")
print("=" * 80)
print(f" Status:               {gate_c_audit['status']}")
print(f" Total Completed:      {gate_c_audit['global_completed_count']} / 10")
print(f" Worker 0 Completed:   {gate_c_audit['worker_0_completed']} / 5")
print(f" Worker 1 Completed:   {gate_c_audit['worker_1_completed']} / 5")
print(f" Total Elapsed Time:   {elapsed_t:.2f}s (Avg {elapsed_t/10.0:.2f}s/q)")
print(f" Combined Checkpoint:  {gate_c_audit['checkpoint_zip_path']}")
print(f" Checkpoint SHA256:    {gate_c_audit['checkpoint_zip_sha256']}")
print("=" * 80)

assert gate_c_audit["status"] == "ALL_QUESTIONS_COMPLETED", f"Gate C failed with status: {gate_c_audit['status']}"
assert gate_c_audit["global_completed_count"] == 10, "Gate C did not complete all 10 questions"
assert gate_c_audit["worker_0_completed"] == 5, "Worker 0 did not complete 5 questions"
assert gate_c_audit["worker_1_completed"] == 5, "Worker 1 did not complete 5 questions"
print("GATE C CONCURRENCY SMOKE: PASS")
```

---

### CELL 12 — GATE C RESUME MICRO-TEST (Zero-Rerun Checkpoint Validation)
```python
print("=== CELL 12: GATE C RESUME MICRO-TEST ===")

# Instantiate a fresh runner and restore the Gate C combined checkpoint
gate_c_resumer = DualPublic1000SessionRunner(
    app_config=app_cfg,
    working_dir=gate_c_work_dir,
    questions_path=gate_c_qpath,
    session_id="gate_c_resume_test",
    worker_devices=("0", "1"),
)

records_0, records_1 = gate_c_resumer.restore_and_validate_checkpoint()
total_resumed = len(records_0) + len(records_1)
print(f"Resumed Records: {total_resumed} (Worker 0: {len(records_0)}, Worker 1: {len(records_1)})")
assert total_resumed == 10, f"Expected 10 resumed records, found {total_resumed}"
assert len(records_0) == 5, f"Expected 5 records for Worker 0, found {len(records_0)}"
assert len(records_1) == 5, f"Expected 5 records for Worker 1, found {len(records_1)}"

print("GATE C RESUME MICRO-TEST: PASS (All 10/10 questions restored without reruns)")
```

---

### CELL 13 — DUAL-T4 PUBLIC-1000 MULTI-SESSION EXECUTION CONTROLLER
```python
from pathlib import Path
from legal_agentic_rag.competition.uit_dsc_2026.dual_session_runner import DualPublic1000SessionRunner
from legal_agentic_rag.runtime.online import OnlineRuntimeFactory
from legal_agentic_rag.serving.config_loader import load_application_config

print("=== CELL 13: DUAL-T4 PUBLIC-1000 MULTI-SESSION CONTROLLER ===")

# Discover official Public-1000 questions
public_q_candidates = (
    list(Path("/kaggle/input").rglob("public-official.json"))
    + list(Path("/kaggle/input").rglob("public_official.json"))
)
assert public_q_candidates, "public-official.json not found under /kaggle/input"
public_q_path = public_q_candidates[0]
print(f"Official Public Questions: {public_q_path}")

# Check for prior combined dual-GPU checkpoint (.zip or .zip.bin)
prior_combined_checkpoint = None
combined_chk_candidates = (
    list(Path("/kaggle/input").rglob("public1000_dual_gpu_checkpoint_latest.zip*"))
    + list(Path("/kaggle/input").rglob("m491_jina35_dual_gpu_checkpoint*"))
)
if combined_chk_candidates:
    prior_combined_checkpoint = combined_chk_candidates[0]
    print(f"Found previous combined dual-GPU checkpoint: {prior_combined_checkpoint}")
else:
    print("No previous combined checkpoint found. Starting Dual-GPU Session 1.")

config_path = extract_dir / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"
app_cfg = load_application_config(config_path)

def _build_worker_runtime_0():
    factory = OnlineRuntimeFactory.from_config(app_cfg, device="cuda:0")
    return factory.build()

def _build_worker_runtime_1():
    factory = OnlineRuntimeFactory.from_config(app_cfg, device="cuda:0")
    return factory.build()

dual_public_work_dir = Path("/kaggle/working/public1000_dual_gpu_execution")

dual_runner = DualPublic1000SessionRunner(
    app_config=app_cfg,
    working_dir=dual_public_work_dir,
    questions_path=public_q_path,
    session_budget_hours=9.5,  # Stops cleanly before 12h limit
    runtime_builders={0: _build_worker_runtime_0, 1: _build_worker_runtime_1},
    worker_devices=("0", "1"),
)

session_audit = dual_runner.run_session(combined_checkpoint_archive_path=prior_combined_checkpoint)
print(f"Dual Session Final Status: {session_audit['status']}")
```

---

### CELL 14 — DUAL-T4 OFFICIAL CODABENCH SUBMISSION PACKAGING (Gated on 1000/1000)
```python
from pathlib import Path

print("=== CELL 14: DUAL-T4 CODABENCH SUBMISSION PACKAGING ===")

submission_out_dir = Path("/kaggle/working/submission_package_dual")

try:
    sub_zip_path = dual_runner.package_final_submission(submission_out_dir)
    print(f"SUCCESS: Dual-T4 Submission zip created at: {sub_zip_path}")
    print(f"Submission zip size: {sub_zip_path.stat().st_size:,} bytes")
    print("READY TO DOWNLOAD SUBMISSION.ZIP FOR CODABENCH SUBMISSION!")
except Exception as err:
    print(f"SUBMISSION PACKAGING HELD: {err}")
    print("If additional sessions are required, download 'public1000_dual_gpu_checkpoint_latest.zip' and resume in the next session.")
```
