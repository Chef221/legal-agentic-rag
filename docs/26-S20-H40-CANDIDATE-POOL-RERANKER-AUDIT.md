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
2. **Shared Scoring**: Cross-encoder scoring is executed once on the 40 fused candidates. A candidate's score is identical regardless of whether it is evaluated under S20 or H40.
3. **Exact Production Tie-Breaking**: Sorting for both candidate pools uses `(-score, fused_rank, chunk_id)`.
4. **Graphless & Generator-Free**: Online runtime loads exactly 3 serving artifacts (`legal_chunks`, `bm25_index`, `vector_index`). `graph/` and `relationships/` are absent. Qwen and generation are not invoked.

---

## 3. Canonical Frozen Reproduction Authorities

Execution must pass strict reproduction gates against historical frozen baselines:

| Authority Artifact | Canonical SHA-256 / Identity | Purpose |
|---|---|---|
| `development.json` (991 questions) | `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8` | Source questions |
| `phase-b1a-graph-routing-cases.json` | 22 frozen canonical question IDs | Test case manifest |
| B1A.2 Evidence ZIP | `1fcc9150840573023d8ae443324d431635f59b54cd8325aa3324611bc1cb7117` | Baseline evidence |
| B1A.2 Results JSONL | `51ed1d8ba99690973f16ff023300b060d6b03e60d905efe6498325626484e39a` | Frozen baseline results |
| B1A.2 Execution Commit | `9265f3dadcf1ef0170f0abe618519da1657fc55e` | Execution provenance |

### Fail-Closed Reproduction Gates:
1. **Seed Prefix Invariance**: Current `fused40[:20]` chunk sequence must **exactly match** frozen B1A.2 `s20_arm.seed_hits` chunk sequence for all 22 cases.
2. **S20 Top-8 Reproduction**: Derived S20 final top-8 chunk IDs, document IDs, and scores must match frozen B1A.2 `s20_arm.final_hits` within $|score\_diff| \le 10^{-6}$ for all 22 cases.
3. **H40 Top-8 Reproduction**: Derived H40 final top-8 chunk IDs, document IDs, and scores must match frozen B1A.2 `h40_arm.final_hits` within $|score\_diff| \le 10^{-6}$ for all 22 cases.
4. **Historical Divergence Reproduction**: Must reproduce exactly 5 identical top-8 cases and 17 changed top-8 cases.

---

## 4. Verdict Contract

| Verdict | Meaning | Authority / Next Action |
|---|---|---|
| **`CANDIDATE_POOL_AUDIT_PASS`** | Protocol executed cleanly, frozen B1A.2 mechanics reproduced (22/22 seed match, 22/22 S20 top-8 match, 22/22 H40 top-8 match, 5 identical / 17 changed), candidate-pool churn characterized. | `"h40_promotion_authorized": false`. H40 remains in Attempt 2. Proceed to Priority B verification audit. |
| **`CANDIDATE_POOL_DRIFT_DETECTED`** | Execution completed with 0 model errors, but derived S20/H40 hits diverged from frozen baseline expectations. | Protocol halted. Investigate ranking or retrieval drift. |
| **`INVALID_EXPERIMENT`** | Artifact corruption, SHA mismatch, missing baseline summary, or $\ge 1$ `retrieval:model_error`. | Protocol invalidated. Fix runtime environment. |

---

## 5. Per-Case and Aggregate Diagnostic Schema

### Per-Case Diagnostics:
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

## 7. Kaggle Execution Runbook (Copy-Paste)

```bash
#!/usr/bin/env bash
# ==============================================================================
# S20 vs H40 Candidate-Pool / Reranker Mechanics Audit — Stage R1
# Runbook for Kaggle Environment (GPU x1, Internet ON)
# ==============================================================================
set -euo pipefail

# 1. PIN EXECUTION COMMIT (Reviewed commit SHA)
REVIEWED_COMMIT_SHA="PLACEHOLDER_REVIEWED_COMMIT_SHA"

echo "=== Stage R1 Candidate-Pool Audit Execution ==="
echo "Target Commit: ${REVIEWED_COMMIT_SHA}"

# 2. CLONE AND CHECKOUT REPOSITORY
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
test "${ACTUAL_COMMIT}" = "${REVIEWED_COMMIT_SHA}" || { echo "FATAL: Commit SHA mismatch!"; exit 1; }

# 3. ENVIRONMENT & DEPENDENCIES
pip install -q -e .
pip install -q pytest sentence-transformers

# 4. VERIFY TEST SUITE
python -m pytest tests/unit/evaluation/test_candidate_pool_reranker_audit.py -q
python -m pytest -q

# 5. EXECUTE AUDIT PROTOCOL
python scripts/candidate_pool_reranker_audit.py \
  --config configs/serving_config.json \
  --manifest configs/phase-b1a-graph-routing-cases.json \
  --questions data/raw/development.json \
  --baseline-evidence-dir data/evidence/phase-b1a2-graph-equivalence-evidence.zip \
  --output-dir artifacts/candidate_pool_audit \
  --staging-root artifacts/serving

# 6. VERIFY EVIDENCE PACKAGE
ls -lh artifacts/candidate_pool_audit/candidate-pool-reranker-audit-evidence.zip
sha256sum artifacts/candidate_pool_audit/candidate-pool-reranker-audit-evidence.zip

echo "=== Stage R1 Candidate-Pool Audit Complete ==="
```
