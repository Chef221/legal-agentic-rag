# 26. S20 vs H40 Candidate-Pool / Reranker Mechanics Audit — Stage R1

## 1. Executive Summary & Scientific Objective

Phase B1B concluded with mechanical verdict **`B1B_EQUIVALENCE_PASS`**, officially proving that graph traversal is redundant on the UIT DSC competition corpus (0 edges, 0 traversal steps) and removing all graph traversal components from the active online and offline competition pipelines while preserving exact S20 retrieval behavior.

In the frozen Phase B1A.2 experiment, the two hybrid-rerank candidate-pool depth configurations exhibited significant behavioral divergence on relationship queries:

```text
ARM S20 (Production Attempt 1):
    Sparse branch depth = 40, Dense branch depth = 40
    → Reciprocal Rank Fusion (RRF) candidate pool limit ≤ 20
    → CrossEncoder reranking pool limit ≤ 20
    → Final top-k ≤ 8

ARM H40 (Attempt 2 Diagnostic / Non-promoted):
    Sparse branch depth = 40, Dense branch depth = 40
    → Reciprocal Rank Fusion (RRF) candidate pool limit ≤ 40
    → CrossEncoder reranking pool limit ≤ 40
    → Final top-k ≤ 8
```

Historical B1A.2 divergence results on the 22 canonical relationship queries:
- Identical top-8 chunk sequence: **5 / 22 cases**
- Changed top-8 chunk sequence: **17 / 22 cases**
- Mean top-8 overlap: **6.4091 / 8** (min: 3/8, max: 8/8)
- Mean top-8 Jaccard similarity: **0.7036**

### Scientific Question for Stage R1:
When the fused candidate pool is expanded from 20 to 40 while keeping the exact same branch retrieval (BM25 + dense), query understanding, fusion, cross-encoder reranker, and final top-k behavior:
1. Exactly which candidates from fused ranks 21–40 enter the final top-8 (*tail entrants*)?
2. What S20 candidates do they displace (*displaced seed-20 candidates*)?
3. By what cross-encoder logit score margins are they promoted?

> [!IMPORTANT]
> **Stage R1 is a Mechanics Audit, not an Evaluation of Semantic Ground Truth.**
> The official UIT DSC 2026 dataset provides zero retrieval relevance labels.
> This audit does **not** evaluate chunk legal relevance against reference answers and does **not** generate synthetic relevance labels.
> **H40 remains unpromoted in Attempt 2.** Stage R1 has zero authority to alter production routing or claim H40 superiority.

---

## 2. Experimental Design & Variable Isolation

To guarantee that candidate-pool depth is the **sole experimental variable**, the audit enforces strict single-pass retrieval execution:

```mermaid
flowchart TD
    Q[Input Question] --> QU[Query Understanding: Enrich Query]
    QU --> HYB[Single Top-Level Hybrid Retrieval\nBM25 Depth 40 + Dense Depth 40]
    HYB --> F40[Fused Candidates List\nExact RRF Ranking 1..40]
    
    F40 --> S20_POOL[S20 Candidate Pool\nExact Prefix Fused 1..20]
    F40 --> H40_POOL[H40 Candidate Pool\nFull Fused 1..40]
    
    F40 --> CE[Single CrossEncoder Scoring Pass\nScore all 40 candidates once]
    
    CE --> S20_SORT[Apply Production Tie-break on S20 Pool\n(-score, fused_rank, chunk_id)]
    CE --> H40_SORT[Apply Production Tie-break on H40 Pool\n(-score, fused_rank, chunk_id)]
    
    S20_SORT --> S20_TOP8[Derived S20 Final Top-8]
    H40_SORT --> H40_TOP8[Derived H40 Final Top-8]
    
    S20_TOP8 --> DIAG[Mechanics & Churn Diagnostics\nTail Entrants, Displacements, Margins]
    H40_TOP8 --> DIAG
```

### Protocol Invariants:
1. **Single Branch Search & Fusion**: Hybrid retrieval is called exactly once with `top_k=40, candidate_k=40`. Both arms share the exact same fused ranking.
2. **Real Branch Depth Observations**: `RecordingBranchRetriever` wrappers record every branch query. Every sparse query and dense query must have `top_k=40, candidate_k=40`.
3. **Shared Scoring**: Cross-encoder scoring is executed once on the 40 fused candidates. A candidate's score is identical regardless of whether it is evaluated under S20 or H40.
4. **Exact Production Tie-Breaking**: Sorting for both candidate pools uses `(-score, fused_rank, chunk_id)`.
5. **Graphless & Generator-Free**: Online runtime loads exactly 3 serving artifacts (`legal_chunks`, `bm25_index`, `vector_index`). `graph/` and `relationships/` are absent. Qwen and generation are not invoked.

---

## 3. Canonical Frozen Reproduction Authorities

Execution must pass strict reproduction gates against historical frozen baselines:

| Authority Artifact | Canonical SHA-256 / Identity | Purpose |
|---|---|---|
| `development.json` (991 questions) | `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8` | Source questions |
| `phase-b1a-graph-routing-cases.json` | 22 frozen canonical question IDs | Test case manifest |
| Canonical B1A.2 Baseline Evidence | Canonical ZIP (`1fcc9150840573023d8ae443324d431635f59b54cd8325aa3324611bc1cb7117`) or Extracted Bundle (8 verified member hashes) | Mandatory baseline evidence |
| B1A.2 Results JSONL | `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a` | Frozen baseline results |
| B1A.2 Execution Commit | `9265f3dadcf1ef0170f0abe618519da1657fc55e` | Execution provenance |

### Fail-Closed Reproduction Gates:
1. **Mandatory B1A.2 Baseline Verification**: Exact ZIP SHA-256 (for canonical ZIP) or exact 8-member package hashes (for extracted bundle), internal results SHA-256, execution commit, run-summary results SHA-256, and verdict `GRAPH_REDUNDANCY_PROVEN` are verified.
2. **Seed Prefix Invariance**: Current `fused40[:20]` chunk sequence must **exactly match** frozen B1A.2 `s20_arm.seed_hits` chunk sequence for all 22 cases.
3. **S20 Top-8 Reproduction**: Derived S20 final top-8 chunk IDs, document IDs, and scores must match frozen B1A.2 `s20_arm.final_hits` within $|score\_diff| \le 10^{-6}$ for all 22 cases.
4. **H40 Top-8 Reproduction**: Derived H40 final top-8 chunk IDs, document IDs, and scores must match frozen B1A.2 `h40_arm.final_hits` within $|score\_diff| \le 10^{-6}$ for all 22 cases.
5. **Real Branch-Depth Fidelity**: 22/22 cases must execute branch queries with candidate depth 40 and top-k 40.
6. **Historical Divergence Reproduction**: Must reproduce exactly 5 identical top-8 cases and 17 changed top-8 cases.

---

## 4. Verdict Contract

| Verdict | Meaning | Authority / Next Action |
|---|---|---|
| **`CANDIDATE_POOL_AUDIT_PASS`** | Protocol executed cleanly, frozen B1A.2 mechanics reproduced (22/22 seed match, 22/22 S20 top-8 match, 22/22 H40 top-8 match, 22/22 branch depth fidelity, 5 identical / 17 changed), candidate-pool churn characterized. | `"h40_promotion_authorized": false`. H40 remains in Attempt 2. Proceed to Priority B verification audit. |
| **`CANDIDATE_POOL_DRIFT_DETECTED`** | Execution completed with 0 model errors, but derived S20/H40 hits diverged from frozen baseline expectations. | Protocol halted. Investigate ranking or retrieval drift. |
| **`INVALID_EXPERIMENT`** | Artifact corruption, SHA mismatch, missing baseline summary, branch depth violation, or $\ge 1$ `retrieval:model_error`. | Protocol invalidated. Fix runtime environment. |

---

## 5. Per-Case and Aggregate Diagnostic Schema

### Per-Case Diagnostics:
- `branch_depth_observations`: real count and candidate depths of sparse and dense queries for this case.
- `fused_candidates_40`: 40 fused items with fused rank, chunk ID, doc ID, RRF score, BM25 rank/contribution, dense rank/contribution.
- `cross_encoder_scored_candidates_40`: 40 items sorted by cross-encoder score with reranker rank, fused rank, chunk ID, doc ID, score.
- `derived_s20_final_hits`: Final top-8 hits derived from fused 1..20.
- `derived_h40_final_hits`: Final top-8 hits derived from fused 1..40.
- `s20_vs_h40_comparison`: `top8_identical`, `overlap_count`, `jaccard`, `s20_only_chunks`, `h40_only_chunks`.
- `tail_entrants`: For each H40-only chunk entering top-8: chunk ID, doc ID, fused rank (21..40), reranker rank, reranker score, fused rank bucket (`21-25`, `26-30`, `31-35`, `36-40`).
- `displaced_s20_candidates`: For each displaced S20 candidate: chunk ID, doc ID, fused rank (1..20), S20 reranker rank, reranker score.
- `score_cutoff_margin_diagnostics`: Set-level cutoffs (`s20_top8_cutoff_score`, `h40_top8_cutoff_score`, `min_h40_entrant_score`, `max_displaced_s20_score`, `entrant_vs_displaced_margin`).

### Aggregate Diagnostics:
- `identical_top8_cases` (5), `changed_top8_cases` (17)
- `total_tail_entrants`, `cases_with_tail_entrants`
- `tail_entrants_per_changed_case`: mean, median, min, max
- `entrant_fused_rank_bucket_counts`: distribution across `21-25`, `26-30`, `31-35`, `36-40`
- `top8_overlap`: mean, median, min, max
- `top8_jaccard`: mean, median, min, max
- `document_level_churn_count`: count of novel documents introduced into top-8 by H40
- `score_cutoff_margin_distributions`: summary of cutoff and promotion score margins
- `cases_ordered_by_churn`: cases ranked by entrant count descending and Jaccard ascending.

---

## 6. Evidence Package Inventory

Target file: **`candidate-pool-reranker-audit-evidence.zip`**

Deterministic archive contents:
1. `execution/audit_execution_identity.json`
2. `baseline/b1a2_baseline_identity.json`
3. `execution/graphless_root_inventory.json`
4. `configs/runtime_config.json`
5. `configs/phase-b1a-graph-routing-cases.json`
6. `results/candidate_pool_case_results.jsonl`
7. `results/candidate_pool_case_metrics.jsonl`
8. `results/candidate_pool_audit_report.json`
9. `results/candidate_pool_decision_report.json`

---

## 7. Kaggle Execution Runbook (Copy-Paste Cells)

### Cell R1-K1 — Environment & Commit Verification

```bash
%%bash
set -euo pipefail

REVIEWED_COMMIT_SHA="PLACEHOLDER_REVIEWED_COMMIT_SHA"

echo "=== R1-K1: Environment & Commit Verification ==="
echo "Target Execution Commit: ${REVIEWED_COMMIT_SHA}"

# 1. Clone repository or checkout exact commit
if [ -d "legal-agentic-rag" ]; then
  cd legal-agentic-rag
  git fetch origin main --prune
else
  git clone https://github.com/Chef221/legal-agentic-rag.git
  cd legal-agentic-rag
fi

git checkout "${REVIEWED_COMMIT_SHA}"
ACTUAL_COMMIT=$(git rev-parse HEAD)
echo "Checked out commit: ${ACTUAL_COMMIT}"
test "${ACTUAL_COMMIT}" = "${REVIEWED_COMMIT_SHA}" || { echo "FATAL: Execution SHA mismatch!"; exit 1; }

# 2. Pin exact reviewed dependency versions
pip uninstall -y torchao || true
pip install -q \
  transformers==4.51.3 \
  sentence-transformers==5.4.1 \
  accelerate==1.6.0 \
  nltk==3.7

# 3. Install repository in editable mode
pip install -q --no-deps -e .

# 4. Verify environment identity
python -c '
import torch, transformers, sentence_transformers, nltk
import legal_agentic_rag

print("Package version:      ", legal_agentic_rag.__version__)
assert legal_agentic_rag.__version__ == "0.50.7"
print("PyTorch version:      ", torch.__version__)
print("CUDA available:       ", torch.cuda.is_available())
assert torch.cuda.is_available(), "FATAL: CUDA GPU is required for retrieval audit"
print("Device name:          ", torch.cuda.get_device_name(0))
print("Transformers version: ", transformers.__version__)
print("SentenceTransformers: ", sentence_transformers.__version__)
print("NLTK version:         ", nltk.__version__)
'
echo "=== R1-K1: PASS ==="
```

---

### Cell R1-K2 — Identity-Based Input Discovery

```python
import hashlib
import json
from pathlib import Path

print("=== R1-K2: Identity-Based Input Discovery ===")

CANONICAL_DEV_SHA = "8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8"
CANONICAL_B1A2_ZIP_SHA = "1fcc9150840573023d8ae443324d431635f59b54cd8325aa3324611bc1cb7117"
CANONICAL_B1A2_RESULTS_SHA = "51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a"
EXPECTED_CORPUS_DATASET = "uit-dsc-2026-task2-selected-contexts"

CANONICAL_B1A2_MEMBERS = {
    "configs/phase-b1a-graph-routing-cases.json": "b1efe824f320d9323af462869fd8842ef8544fa14d5f81ae35decca99e1ee99f",
    "evidence/materialized_questions_identity.json": "055f45c702dde9f8147dbd57e6a198d2938a007dba02efd68857b8b9574b7dc1",
    "configs/runtime_config.json": "23d154feafa46300215e8498e9738d345c48122739e377dcab43e9e5475b1a31",
    "evidence/phase_b1a2_run_summary.json": "7f000dc5841b1569a9d2e2a045ba9466ffbb56f31078d4e27d0054a381a904d0",
    "results/phase_b1a2_retrieval_results.jsonl": "51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a",
    "results/phase_b1a2_graph_equivalence_report.json": "7f9b477441754328eb4e116fb28f56f6c567e846c878177e1c3762ca8af15058",
    "results/phase_b1a2_decision_report.json": "bc66414a1f0a30669f46f8edfe3df2d1d4b51ba000ce6a476ca6cd65afa64ed0",
    "results/phase_b1a2_case_metrics.jsonl": "583d74ef1f81c63d255fefe79eba563a464d78ec38aecad30d2c554a5df50030",
}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# 1. Discover canonical development.json uniquely
dev_candidates = list(Path("/kaggle/input").rglob("development.json"))
matching_devs = [
    p for p in dev_candidates
    if p.is_file() and sha256_file(p) == CANONICAL_DEV_SHA
]

assert len(matching_devs) == 1, (
    f"Expected exactly 1 canonical development.json matching SHA {CANONICAL_DEV_SHA}, "
    f"found {len(matching_devs)}: {matching_devs}"
)
found_dev = matching_devs[0]
print(f"Found unique canonical development.json at: {found_dev}")

# 2. Discover canonical B1A.2 baseline evidence (ZIP mode or Extracted Bundle mode)
zip_candidates = list(Path("/kaggle/input").rglob("*.zip"))
matching_b1a2_zips = [
    p for p in zip_candidates
    if p.is_file() and sha256_file(p) == CANONICAL_B1A2_ZIP_SHA
]

found_b1a2 = None
b1a2_source_kind = None

if len(matching_b1a2_zips) == 1:
    found_b1a2 = matching_b1a2_zips[0]
    b1a2_source_kind = "canonical_zip"
    print(f"Found unique canonical B1A.2 baseline ZIP at: {found_b1a2}")
elif len(matching_b1a2_zips) > 1:
    raise AssertionError(f"Multiple canonical B1A.2 ZIP archives found: {matching_b1a2_zips}")
else:
    print("Zero canonical B1A.2 ZIP archives found. Searching for extracted canonical bundle...")
    valid_b1a2_dirs = []
    for res_p in Path("/kaggle/input").rglob("phase_b1a2_retrieval_results.jsonl"):
        if not res_p.is_file():
            continue
        if sha256_file(res_p) != CANONICAL_B1A2_RESULTS_SHA:
            continue
        cand_root = res_p.parents[1] if res_p.parent.name == "results" else res_p.parent
        # Verify all 8 canonical member hashes
        all_members_match = True
        for rel_path, exp_sha in CANONICAL_B1A2_MEMBERS.items():
            member_file = cand_root / rel_path
            if not member_file.is_file() or sha256_file(member_file) != exp_sha:
                all_members_match = False
                break
        if all_members_match:
            valid_b1a2_dirs.append(cand_root)

    valid_b1a2_dirs = list(dict.fromkeys(valid_b1a2_dirs))
    assert len(valid_b1a2_dirs) == 1, (
        f"Expected exactly 1 valid extracted B1A.2 canonical bundle with all 8 verified member hashes, "
        f"found {len(valid_b1a2_dirs)}: {valid_b1a2_dirs}"
    )
    found_b1a2 = valid_b1a2_dirs[0]
    b1a2_source_kind = "canonical_extracted_bundle"
    print(f"Found unique verified extracted B1A.2 canonical bundle at: {found_b1a2}")

# 3. Discover serving artifact root uniquely satisfying full corpus validation contract
val_candidates = list(Path("/kaggle/input").rglob("build_validation_full_corpus.json"))
if not val_candidates:
    val_candidates = list(Path("/kaggle/input").rglob("build_validation.json"))

valid_serving_roots = []
for val_file in val_candidates:
    root = val_file.parent
    if not all((root / sub).is_dir() for sub in ["legal_chunks", "bm25", "vector"]):
        continue
    try:
        report = json.loads(val_file.read_text(encoding="utf-8"))
    except Exception:
        continue
    dataset_manifest = report.get("dataset_manifest", {})
    if (
        report.get("is_valid") is True
        and report.get("is_full_corpus") is True
        and isinstance(dataset_manifest, dict)
        and dataset_manifest.get("dataset_name") == EXPECTED_CORPUS_DATASET
    ):
        valid_serving_roots.append(root)

valid_serving_roots = list(dict.fromkeys(valid_serving_roots))
assert len(valid_serving_roots) == 1, (
    f"Expected exactly 1 valid serving artifact root matching full corpus validation contract, "
    f"found {len(valid_serving_roots)}: {valid_serving_roots}"
)
found_serving_root = valid_serving_roots[0]
print(f"Found unique validated serving artifact root at: {found_serving_root}")

# Persist discovered paths for subsequent cells
discovery_info = {
    "dev_json": str(found_dev),
    "b1a2_evidence": str(found_b1a2),
    "b1a2_source_kind": b1a2_source_kind,
    "serving_root": str(found_serving_root),
}
Path("/kaggle/working/discovery_info.json").write_text(json.dumps(discovery_info, indent=2))
print("=== R1-K2: PASS ===")
```

---

### Cell R1-K3 — Graphless Staging Preparation

```python
import json
import os
from pathlib import Path
import shutil

print("=== R1-K3: Graphless Staging Preparation ===")

discovery = json.loads(Path("/kaggle/working/discovery_info.json").read_text())
source_root = Path(discovery["serving_root"])
staging_root = Path("/kaggle/working/staging_graphless")

if staging_root.exists():
    shutil.rmtree(staging_root)
staging_root.mkdir(parents=True, exist_ok=True)

# Copy/link required serving components excluding graph and relationships
for item in source_root.iterdir():
    if item.name in ("graph", "relationships"):
        continue
    dest = staging_root / item.name
    try:
        os.symlink(item.resolve(), dest, target_is_directory=item.is_dir())
    except (OSError, NotImplementedError):
        if item.is_dir():
            shutil.copytree(item, dest, symlinks=True)
        else:
            shutil.copy2(item, dest)

# Hard assertions
assert not (staging_root / "graph").exists(), "FATAL: Staging root contains graph"
assert not (staging_root / "relationships").exists(), "FATAL: Staging root contains relationships"
assert (staging_root / "legal_chunks").is_dir(), "FATAL: Missing legal_chunks"
assert (staging_root / "bm25").is_dir(), "FATAL: Missing bm25"
assert (staging_root / "vector").is_dir(), "FATAL: Missing vector"

print(f"Graphless staging root prepared at: {staging_root}")
print("Active components:", [p.name for p in staging_root.iterdir()])
print("=== R1-K3: PASS ===")
```

---

### Cell R1-K4 — Runtime Configuration Freeze

```python
import json
from pathlib import Path

from legal_agentic_rag.configuration import ApplicationConfig

print("=== R1-K4: Runtime Configuration Freeze ===")

example_config_path = Path("/kaggle/working/legal-agentic-rag/configs/phase-a-current-system-census-kaggle.example.json")
raw_cfg = json.loads(example_config_path.read_text(encoding="utf-8"))

# Set staging root
staging_root = "/kaggle/working/staging_graphless"
raw_cfg["artifacts"]["root_path"] = staging_root

# Set retrieval devices to CUDA
raw_cfg["online"]["vector_runtime"]["search_device"] = "cuda"
raw_cfg["offline"]["embedding"]["device"] = "cuda"
raw_cfg["online"]["reranker"]["device"] = "cuda"

# Enforce retrieval invariants
raw_cfg["online"]["retrieval"]["top_k"] = 8
raw_cfg["online"]["retrieval"]["candidate_k"] = 40
raw_cfg["online"]["query_understanding"]["multi_query_enabled"] = True

# Validate full config schema
app_config = ApplicationConfig.model_validate(raw_cfg)

runtime_config_path = Path("/kaggle/working/runtime_config_r1.json")
runtime_config_path.write_text(json.dumps(app_config.model_dump(mode="json"), indent=2), encoding="utf-8")

print(f"Validated and froze runtime config at: {runtime_config_path}")
print("=== R1-K4: PASS ===")
```

---

### Cell R1-K5 — Execute Stage R1 Audit

```bash
%%bash
set -euo pipefail

echo "=== R1-K5: Execute Stage R1 Audit ==="
cd /kaggle/working/legal-agentic-rag

DEV_JSON=$(python -c 'import json; print(json.load(open("/kaggle/working/discovery_info.json"))["dev_json"])')
B1A2_EVIDENCE=$(python -c 'import json; print(json.load(open("/kaggle/working/discovery_info.json"))["b1a2_evidence"])')

python scripts/candidate_pool_reranker_audit.py \
  --config /kaggle/working/runtime_config_r1.json \
  --manifest configs/phase-b1a-graph-routing-cases.json \
  --questions "${DEV_JSON}" \
  --baseline-evidence "${B1A2_EVIDENCE}" \
  --output-dir /kaggle/working/artifacts/candidate_pool_audit \
  --staging-root /kaggle/working/staging_graphless

echo "=== R1-K5: Execution Complete ==="
```

---

### Cell R1-K6 — Mechanical Verdict Gate

```python
import json
from pathlib import Path

print("=== R1-K6: Mechanical Verdict Gate ===")

out_dir = Path("/kaggle/working/artifacts/candidate_pool_audit")
decision_path = out_dir / "results" / "candidate_pool_decision_report.json"
report_path = out_dir / "results" / "candidate_pool_audit_report.json"

assert decision_path.is_file(), f"FATAL: Decision report missing at {decision_path}"
assert report_path.is_file(), f"FATAL: Audit report missing at {report_path}"

decision = json.loads(decision_path.read_text(encoding="utf-8"))
report = json.loads(report_path.read_text(encoding="utf-8"))
summary = decision.get("summary", {})

print(f"Verdict:                    {decision.get('verdict')}")
print(f"Audit Verified:             {decision.get('audit_verified')}")
print(f"H40 Promotion Authorized:   {decision.get('h40_promotion_authorized')} (MUST BE FALSE)")
print(f"Total Cases Evaluated:      {summary.get('total_cases')} / 22")
print(f"Seed Prefix Passes:         {summary.get('seed_prefix_passes')} / 22")
print(f"S20 Top-8 Passes:           {summary.get('s20_top8_passes')} / 22")
print(f"H40 Top-8 Passes:           {summary.get('h40_top8_passes')} / 22")
print(f"Branch Depth Passes:        {summary.get('branch_depth_passes')} / 22")
print(f"Identical Top-8 Cases:      {summary.get('identical_top8_cases')} (expected 5)")
print(f"Changed Top-8 Cases:        {summary.get('changed_top8_cases')} (expected 17)")
print(f"Total Tail Entrants:        {summary.get('total_tail_entrants')}")
print(f"Document Churn Count:       {summary.get('document_level_churn_count')}")
print(f"Retrieval Model Errors:     {summary.get('retrieval_model_error_count')}")

# Hard assertions
assert decision["verdict"] == "CANDIDATE_POOL_AUDIT_PASS", f"FATAL: Verdict was {decision['verdict']}, reasons: {decision.get('reasons')}"
assert decision["audit_verified"] is True, "FATAL: audit_verified must be True"
assert decision["h40_promotion_authorized"] is False, "FATAL: h40_promotion_authorized must be False"
assert summary["total_cases"] == 22, "FATAL: Must evaluate exactly 22 cases"
assert summary["seed_prefix_passes"] == 22, "FATAL: Seed prefix match failed"
assert summary["s20_top8_passes"] == 22, "FATAL: S20 top-8 reproduction failed"
assert summary["h40_top8_passes"] == 22, "FATAL: H40 top-8 reproduction failed"
assert summary["branch_depth_passes"] == 22, "FATAL: Branch depth fidelity failed"
assert summary["identical_top8_cases"] == 5, "FATAL: Identical top-8 case count mismatch"
assert summary["changed_top8_cases"] == 17, "FATAL: Changed top-8 case count mismatch"
assert summary["retrieval_model_error_count"] == 0, "FATAL: Retrieval model errors occurred"

print("\n>>> MECHANICAL VERDICT: CANDIDATE_POOL_AUDIT_PASS <<<")
print("=== R1-K6: PASS ===")
```

---

### Cell R1-K7 — Evidence Package Verification

```python
import hashlib
from pathlib import Path
import zipfile

print("=== R1-K7: Evidence Package Verification ===")

evidence_zip = Path("/kaggle/working/artifacts/candidate_pool_audit/candidate-pool-reranker-audit-evidence.zip")
assert evidence_zip.is_file(), f"FATAL: Evidence ZIP not found at {evidence_zip}"

# Verify ZIP integrity
with zipfile.ZipFile(evidence_zip, "r") as z:
    names = sorted(z.namelist())
    print(f"Archive contains {len(names)} files:")
    for n in names:
        info = z.getinfo(n)
        print(f"  - {n:50s} ({info.file_size} bytes)")

# Compute SHA-256
h = hashlib.sha256()
with evidence_zip.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
zip_sha = h.hexdigest()
zip_size = evidence_zip.stat().st_size

print("\n" + "=" * 60)
print(f"CANONICAL STAGE R1 EVIDENCE ARCHIVE:")
print(f"Path:    {evidence_zip}")
print(f"SHA-256: {zip_sha}")
print(f"Size:    {zip_size} bytes")
print("=" * 60)
print("OPERATOR ACTION: Download candidate-pool-reranker-audit-evidence.zip before terminating Kaggle session.")
print("=== R1-K7: PASS ===")
```
