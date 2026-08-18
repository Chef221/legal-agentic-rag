# 20. M50-C1 Post-Mortem and M50-C2 Conservative QLoRA Pilot Runbook

## 1. M50-C1 Post-Mortem & Diagnostic Findings

### 1.1 Training Execution & Teacher-Forced Convergence
In Milestone 50.1, Candidate 1 (`M50-C1`) was executed on Kaggle GPU using `Qwen/Qwen2.5-3B-Instruct` (revision `a1d308dfcc03e09da285d49d912439a655a571e8`) and the following configuration:
- LoRA rank $r=8, \alpha=16, \text{dropout}=0.05$, target modules $\{q\_proj, k\_proj, v\_proj, o\_proj\}$ (3,686,400 trainable parameters);
- Learning rate $\text{LR} = 5\times 10^{-5}$, cosine scheduler, 1 epoch, microbatch 2, gradient accumulation 8, total steps 282;
- Teacher-forced validation loss on `sft_val.json` converged to **1.09828**.

### 1.2 The Free-Generation Degeneration Phenomenon
Despite smooth loss convergence and a positive reference-anchored ROUGE-L lexical signal ($\Delta\text{ROUGE-L} = +0.04153$ over BASE across 20 paired cases, 13 wins / 7 losses), free autoregressive generation exhibited severe degradation.

The authoritative diagnostic measurements on the 20 direct-QA screening cases are:

| Metric | BASE (Pretrained) | M50-C1 (Fine-Tuned) | Delta / Empirical Impact |
| :--- | :--- | :--- | :--- |
| **Question Count** | 20 | 20 | Identical paired subset |
| **EOS Emitted (`<|im_end|>`)** | 15 / 20 (75%) | 6 / 20 (30%) | **-45%** (Failure to terminate) |
| **Reached Max Tokens (512)** | 5 / 20 (25%) | 14 / 20 (70%) | **+45%** (Runaway generation) |
| **Cap Without EOS Rate** | 5 / 20 (25%) | 14 / 20 (70%) | **+45%** (Unfinished responses) |
| **High Repetition (repeat8 $\ge 0.25$)** | 4 / 20 (20%) | 18 / 20 (90%) | **+70%** (Autoregressive loops) |
| **High Line Duplication ($\ge 0.25$)** | 1 / 20 (5%) | 1 / 20 (5%) | Identical line-level metric |
| **Generated $\ge 2\times$ Reference** | 0 / 20 (0%) | 3 / 20 (15%) | Verbosity blowup |
| **Mean Generated Tokens** | 293.30 | 439.15 | **+145.85 tokens** (+49.7%) |
| **Median Generated Tokens** | 262.50 | 512.00 | Reached 512 token ceiling |

### 1.3 Observed Failure Modes and Candidate Contributing Factors

#### A. Observed Empirical Phenomena (Facts)
1. **Teacher-Forced vs Free-Running Dissociation**: Optimization loss decreased smoothly, yet free autoregressive generation developed pathological looping.
2. **Termination Failure**: 13 out of 15 previously stable BASE cases failed to emit `<|im_end|>` under C1, running until hitting the 512-token ceiling.
3. **Autoregressive Phrase Repetition**: 90% of C1 generations contained repeated 8-grams exceeding the 0.25 threshold (e.g. infinite repetition of statutory introductory clauses).

#### B. Candidate Contributing Factors (Hypotheses)
1. **Excessive Adaptation Capacity & Target Scope**: LoRA across all 4 attention projections ($q, k, v, o$) with $r=8$ and $\text{LR}=5\times 10^{-5}$ for 282 steps may have over-adapted the model and is a candidate contributor to the observed stopping-distribution regression.
2. **Missing Terminal EOS in Truncated Examples**: In sequence truncation without explicit EOS preservation, truncated assistant targets lacked `<|im_end|>`. While only ~39 of 5,617 training examples ($0.694\%$) were truncated at $L=1536$, this absence may have contributed minor negative gradient bias against stopping.
3. **Undetected Trajectory Drift**: Because no intermediate free-generation probes existed, the onset and trajectory of degeneration are unknown.

### 1.4 Production Decision on Candidate 1
Candidate 1 is **conclusively rejected for production promotion**. Execution on SCREEN 617, smoke 50, and dev 991 remains halted.

---

## 2. M50-C2 Conservative Pilot Strategy

M50-C2 is designed as a conservative, generation-safe pilot to test whether reduced capacity, lower learning rate, explicit terminal EOS preservation, and early checkpoint probing preserve semantic gains while preventing autoregressive degeneration.

### 2.1 Architecture & Parameter Reduction
- **Rank & Alpha**: $r=4, \alpha=8, \text{dropout}=0.05$;
- **Target Modules**: restricted strictly to $\{q\_proj, v\_proj\}$;
- **Trainable Parameter Derivation**:
  - $q\_proj$: $(2048 \times 4 + 4 \times 2048) \times 36 = 16,384 \times 36 = 589,824$
  - $v\_proj$: $(2048 \times 4 + 4 \times 256) \times 36 = 9,216 \times 36 = 331,776$
  - **Total Trainable Parameters**: $\mathbf{921,600}$ ($4\times$ smaller than C1).
- **Exact Parameter Verification**: The training runner strictly asserts `trainable_params == 921_600` before the first optimizer step.

### 2.2 Conservative Optimization & Step Bound
- **Learning Rate**: $\text{LR} = 10^{-5}$ ($5\times$ lower than C1);
- **Scheduler**: Cosine decay with warmup ratio 0.05;
- **Step Bound**: Capped at `max_optimizer_steps = 150`;
- **Multi-Checkpoint Gates**: Intermediate checkpoints and free-generation diagnostic probes executed at steps **50, 100, and 150**.

### 2.3 Terminal EOS Preservation
`encode_sft_example` strictly guarantees that when sequence length exceeds `max_seq_length = 1536`, the final sequence position is reserved for `<|im_end|>` with unmasked label `eos_token_id`.

### 2.4 Strict Holdout Isolation & Deterministic VAL Probe
- `screen_holdout.json` (617 records) is strictly frozen;
- Probing uses 20 deterministic questions extracted exclusively from canonical `sft_val.json` via salted SHA-256 (`m50-c2-val-probe-v1:{question_id}`).
- Creation uses `create_m50_c2_canonical_val_probe` with mandatory content and split-manifest SHA-256 assertions.

---

## 3. Checkpoint Health & Selection Gates

### 3.1 Checkpoint Safety Gate (Health Gate)
A checkpoint is eligible on safety only if all 7 conditions hold on the 20 VAL probe cases:
1. `candidate_generation_error_count == 0`
2. `candidate_cap_without_eos_count <= base_cap_without_eos_count + 1`
3. `candidate_repeat8_high_count <= base_repeat8_high_count + 1`
4. `candidate_duplicate_line_high_count <= base_duplicate_line_high_count + 1`
5. `candidate_eos_emitted_count >= base_eos_emitted_count - 1`
6. `candidate_mean_generated_tokens <= base_mean_generated_tokens * 1.35`
7. `candidate_median_generated_tokens <= max(base_median_generated_tokens * 1.35, base_median_generated_tokens + 64.0)`

### 3.2 Checkpoint Semantic Gate
A checkpoint is eligible on semantics only if:
1. $\Delta\text{ROUGE-L} \ge -0.01$
2. If METEOR is available: $\Delta\text{METEOR} \ge -0.01$
3. At least one metric satisfies $\Delta > 0.0$

### 3.3 Checkpoint Selection Algorithm
If multiple checkpoints pass both gates, rank by:
1. Highest combined semantic delta ($\frac{\Delta\text{ROUGE-L} + \Delta\text{METEOR}}{2}$ or $\Delta\text{ROUGE-L}$ if METEOR unavailable);
2. Lower `candidate_cap_without_eos_count`;
3. Lower `candidate_repeat8_high_count`;
4. Lower teacher-forced validation loss.

If 0 checkpoints pass both gates, the report emits `status = "no_promotable_checkpoint"`.

---

## 4. Deterministic 5-Cell Kaggle Runbook

### Kaggle Input Datasets Contract
Before running, the user must Add Input to the notebook:
- **Canonical M44 Dataset**: `uit-dsc-2026-task2-m44-dev-split` (or `uit-dsc-2026-task2`)
- **Canonical Source Files & Verified Hashes**:
  - `training.json`: `0834091ea06dce76d45b693b679b92002c6cf17f82fc8e23f6d413d5155a38c3`
  - `development.json` (optional verification): `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`
  - `quarantined.json` (optional verification): `4202c3853c2333755b8a0c5a58429e7db0db50a19d16048a3e089131301c39a8`
  - `split_manifest.json` (optional verification): `891e482d09892992818e0f1c183f454d85e4c3d3a73114247b9d1dfee326a0c5`
- **M50 Deterministic Split Target Hashes** (derived to `/kaggle/working/artifacts/m50-split`):
  - `sft_train.json`: `39ae95060c76dce63083d747ce8a12d82d0587907ffe8a093a21ab69c8b19be9`
  - `sft_val.json`: `545dcbf6119db077373ce3cb8dee0c0da74cb7465dde4a03ea488788e52a715f`
  - `screen_holdout.json`: `a165d4a6fba2e2ec460f856a2a67580607d72648f1012fb6dbd5b779c1eb7367`
  - `m50_split_manifest.json`: `5dc52813ddcda3124a51aea848058f6008481a65f5b8a167686603416ec82fb7`

---

### Cell 1: Environment Setup & Package Installation
```python
# Cell 1: Install exact pinned C1 runtime dependencies and local repository (--no-deps)
import os
import subprocess
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
REPO_DIR = WORKING_DIR / "legal-agentic-rag"

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/Chef221/legal-agentic-rag.git", str(REPO_DIR)], check=True)

# Install proven C1 runtime dependency pins
subprocess.run([
    "pip", "install", "-q",
    "transformers==4.51.3",
    "peft==0.15.2",
    "bitsandbytes==0.45.5",
    "accelerate==1.6.0",
    "nltk==3.7",
], check=True)

# Install repository in editable mode without modifying pinned dependencies
subprocess.run(["pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)], check=True)

import transformers
import peft
import bitsandbytes
import accelerate
import nltk
import torch
import legal_agentic_rag

print("=== RUNTIME ENVIRONMENT ===")
print(f"legal_agentic_rag: {legal_agentic_rag.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"nltk: {nltk.__version__}")
print(f"torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Strict version assertions
assert legal_agentic_rag.__version__ == "0.50.4", f"Expected 0.50.4, got {legal_agentic_rag.__version__}"
assert transformers.__version__ == "4.51.3", f"Expected 4.51.3, got {transformers.__version__}"
assert peft.__version__ == "0.15.2", f"Expected 0.15.2, got {peft.__version__}"
assert bitsandbytes.__version__ == "0.45.5", f"Expected 0.45.5, got {bitsandbytes.__version__}"
assert accelerate.__version__ == "1.6.0", f"Expected 1.6.0, got {accelerate.__version__}"
assert nltk.__version__ == "3.7", f"Expected 3.7, got {nltk.__version__}"
```

### Cell 2: Canonical M44 Dataset Discovery, Deterministic M50 Splitting & VAL Probe Extraction
```python
# Cell 2: Discover canonical M44 training.json, generate M50 split, and extract 20-question VAL probe
from pathlib import Path
from legal_agentic_rag.fine_tuning.splitting import (
    M50FineTuningSplitter,
    M50_SPLIT_MANIFEST_FILENAME,
    SFT_TRAIN_FILENAME,
    SFT_VAL_FILENAME,
    SCREEN_HOLDOUT_FILENAME,
)
from legal_agentic_rag.fine_tuning.val_probe import (
    CANONICAL_M50_SFT_TRAIN_SHA256,
    CANONICAL_M50_SFT_VAL_SHA256,
    CANONICAL_M50_SCREEN_HOLDOUT_SHA256,
    CANONICAL_M50_SPLIT_MANIFEST_SHA256,
    create_m50_c2_canonical_val_probe,
)
from legal_agentic_rag.competition.uit_dsc_2026.development_split import _file_sha256

WORKING_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
CANONICAL_TRAINING_SHA256 = "0834091ea06dce76d45b693b679b92002c6cf17f82fc8e23f6d413d5155a38c3"

# A. Recursively discover canonical M44 training.json under /kaggle/input
candidate_training_paths = list(INPUT_DIR.rglob("training.json"))
matching_training_paths = [p for p in candidate_training_paths if _file_sha256(p) == CANONICAL_TRAINING_SHA256]
assert len(matching_training_paths) == 1, (
    f"Expected exactly 1 canonical training.json under {INPUT_DIR}, found {len(matching_training_paths)} "
    f"matching out of {len(candidate_training_paths)} candidates."
)
TRAINING_PATH = matching_training_paths[0]
DATASET_ROOT = TRAINING_PATH.parent
print(f"Canonical training.json verified at: {TRAINING_PATH} (Dataset root: {DATASET_ROOT})")

# B. Optionally verify additional M44 dataset files if present
dev_path = DATASET_ROOT / "development.json"
if dev_path.exists():
    assert _file_sha256(dev_path) == "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8", "development.json SHA mismatch"
quar_path = DATASET_ROOT / "quarantined.json"
if quar_path.exists():
    assert _file_sha256(quar_path) == "4202c3853c2333755b8a0c5a58429e7db0db50a19d16048a3e089131301c39a8", "quarantined.json SHA mismatch"

# C & D. Run existing deterministic M50 splitting implementation to /kaggle/working
SPLIT_DIR = WORKING_DIR / "artifacts" / "m50-split"
splitter = M50FineTuningSplitter()
split_manifest = splitter.split(
    clean_training_path=TRAINING_PATH,
    output_directory=SPLIT_DIR,
)

TRAIN_PATH = SPLIT_DIR / SFT_TRAIN_FILENAME
VAL_PATH = SPLIT_DIR / SFT_VAL_FILENAME
SCREEN_PATH = SPLIT_DIR / SCREEN_HOLDOUT_FILENAME
MANIFEST_PATH = SPLIT_DIR / M50_SPLIT_MANIFEST_FILENAME

# E. Assert exact canonical partition SHAs (fail-closed)
assert _file_sha256(TRAIN_PATH) == CANONICAL_M50_SFT_TRAIN_SHA256, "sft_train.json SHA mismatch"
assert _file_sha256(VAL_PATH) == CANONICAL_M50_SFT_VAL_SHA256, "sft_val.json SHA mismatch"
assert _file_sha256(SCREEN_PATH) == CANONICAL_M50_SCREEN_HOLDOUT_SHA256, "screen_holdout.json SHA mismatch"
assert _file_sha256(MANIFEST_PATH) == CANONICAL_M50_SPLIT_MANIFEST_SHA256, "m50_split_manifest.json SHA mismatch"
print(f"M50 split successfully generated at {SPLIT_DIR} with all 4 SHA256 hashes verified.")

# F. Extract 20 deterministic probe questions from sft_val.json with mandatory lineage check
PROBE_DIR = WORKING_DIR / "artifacts" / "m50-c2-val-probe"
val_questions, val_manifest = create_m50_c2_canonical_val_probe(
    sft_val_path=VAL_PATH,
    output_directory=PROBE_DIR,
    split_manifest_path=MANIFEST_PATH,
    probe_count=20,
)

print(f"C2 VAL probe created: {len(val_questions)} questions, probe SHA: {val_manifest.probe_sha256}")
```

### Cell 3: BASE Model Evaluation on the 20-Case VAL Probe
```python
# Cell 3: Run BASE model on the 20 VAL probe cases to build cache
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from legal_agentic_rag.fine_tuning.val_probe import (
    generate_and_save_val_probe_base_cache,
    load_and_validate_val_probe_base_cache,
)
from legal_agentic_rag.competition.uit_dsc_2026.data_loader import UitDsc2026DataLoader
from legal_agentic_rag.schemas import ValProbeManifest

WORKING_DIR = Path("/kaggle/working")
PROBE_DIR = WORKING_DIR / "artifacts" / "m50-c2-val-probe"
PROBE_PATH = PROBE_DIR / "m50-c2-val-probe.json"
PROBE_MANIFEST_PATH = PROBE_DIR / "m50-c2-val-probe-manifest.json"

# Reload probe from disk (ensuring kernel-restart independence)
data_loader = UitDsc2026DataLoader()
val_questions = data_loader.load_questions(PROBE_PATH, require_reference_answers=True)
val_manifest = ValProbeManifest.model_validate_json(PROBE_MANIFEST_PATH.read_text(encoding="utf-8"))

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BASE_REVISION = "a1d308dfcc03e09da285d49d912439a655a571e8"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    revision=BASE_REVISION,
    quantization_config=bnb_config,
    device_map="auto",
)

base_cases, base_manifest = generate_and_save_val_probe_base_cache(
    val_probe_questions=val_questions,
    model=base_model,
    tokenizer=tokenizer,
    output_directory=PROBE_DIR,
    val_probe_manifest=val_manifest,
)

print(f"BASE VAL probe cache saved: {len(base_cases)} cases. Results SHA: {base_manifest.results_sha256}")

# Free BASE model GPU memory before training
del base_model
torch.cuda.empty_cache()
```

### Cell 4: M50-C2 Bounded Training Runner with Multi-Checkpoint Gates
```python
# Cell 4: Train Candidate 2 with gates at steps [50, 100, 150]
from pathlib import Path
from legal_agentic_rag.fine_tuning.training_runner import (
    M50QLoRATrainer,
    load_qlora_candidate_config,
)

WORKING_DIR = Path("/kaggle/working")
REPO_DIR = WORKING_DIR / "legal-agentic-rag"
CONFIG_PATH = REPO_DIR / "configs" / "m50-c2-qwen3b-conservative-qlora-kaggle.example.json"
PILOT_OUTPUT_DIR = WORKING_DIR / "artifacts" / "m50-c2-pilot"
PROBE_DIR = WORKING_DIR / "artifacts" / "m50-c2-val-probe"

# Check for existing checkpoint to resume if interrupted
latest_checkpoint = None
if PILOT_OUTPUT_DIR.exists():
    checkpoints = sorted(PILOT_OUTPUT_DIR.glob("checkpoint-step-*"))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming training from latest checkpoint: {latest_checkpoint}")

config, config_sha = load_qlora_candidate_config(CONFIG_PATH)
print(f"Loaded Candidate 2 config (SHA: {config_sha})")

trainer = M50QLoRATrainer(config=config)

training_manifest = trainer.train(
    train_partition_path=TRAIN_PATH,
    val_partition_path=VAL_PATH,
    output_directory=PILOT_OUTPUT_DIR,
    val_probe_path=PROBE_DIR / "m50-c2-val-probe.json",
    val_probe_manifest_path=PROBE_DIR / "m50-c2-val-probe-manifest.json",
    val_probe_base_results_path=PROBE_DIR / "m50-c2-val-probe-base-results.jsonl",
    val_probe_base_manifest_path=PROBE_DIR / "m50-c2-val-probe-base-manifest.json",
    split_manifest_path=MANIFEST_PATH,
    config_source_path=CONFIG_PATH,
    resume_from_checkpoint_dir=latest_checkpoint,
)

print("Training finished. Total steps:", training_manifest.total_steps)
```

### Cell 5: Checkpoint Selection Report & Pilot Artifact Packaging
```python
# Cell 5: Inspect selection report and package all pilot artifacts
import json
from pathlib import Path
from legal_agentic_rag.fine_tuning.packaging import package_c2_pilot_artifacts

WORKING_DIR = Path("/kaggle/working")
PILOT_OUTPUT_DIR = WORKING_DIR / "artifacts" / "m50-c2-pilot"
SELECTION_REPORT_PATH = PILOT_OUTPUT_DIR / "checkpoint-selection-report.json"

if SELECTION_REPORT_PATH.exists():
    selection_report = json.loads(SELECTION_REPORT_PATH.read_text(encoding="utf-8"))
    print("=== CHECKPOINT SELECTION REPORT ===")
    print(json.dumps(selection_report, indent=2))
else:
    print("No checkpoint selection report found.")

PACKAGE_ZIP = WORKING_DIR / "m50-c2-pilot-complete.zip"
checksums = package_c2_pilot_artifacts(
    pilot_directory=PILOT_OUTPUT_DIR,
    output_zip_path=PACKAGE_ZIP,
    probe_steps=[50, 100, 150],
)

print(f"\nPilot packaged to {PACKAGE_ZIP} ({len(checksums)} files).")
```
